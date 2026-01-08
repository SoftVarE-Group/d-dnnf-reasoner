pub mod anomalies;
pub mod counting;
pub mod extended_ddnnf;
pub mod multiple_queries;
pub mod node;
pub mod statistics;

use self::node::Node;
use crate::NodeType;
use crate::parser::graph::{DdnnfGraph, rebuild_graph};
use num::BigInt;
use petgraph::stable_graph::NodeIndex;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::path::Path;

/// Value for indicating whether the d-DNNF is a special case: tautology or contradiction.
/// If it is neither a tautology nor a contradiction, it is non-trivial.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub enum DdnnfKind {
    #[default]
    NonTrivial,
    Tautology,
    Contradiction,
}

/// A Ddnnf holds all the nodes as a vector, also includes meta data and further information that is used for optimations
#[derive(Clone, Debug)]
pub struct Ddnnf {
    /// Flag indicating whether the d-DNNF is a special case: tautology or contradiction.
    ///
    /// **Note:** A d-DNNF representing a tautology or contradiction contains no nodes.
    pub kind: DdnnfKind,
    /// The actual nodes of the d-DNNF in postorder
    pub nodes: Vec<Node>,
    /// Literals for upwards propagation
    pub literals: HashMap<i32, usize>, // <var_number of the Literal, and the corresponding indize>
    /// The core/dead features of the model corresponding with this ddnnf
    pub core: HashSet<i32>,
    /// An interim save for the marking algorithm
    pub md: Vec<usize>,
    pub number_of_variables: u32,
    /// The number of threads
    pub max_worker: u16,
}

impl Default for Ddnnf {
    fn default() -> Self {
        Ddnnf {
            kind: DdnnfKind::default(),
            nodes: Vec::new(),
            literals: HashMap::new(),
            core: HashSet::new(),
            md: Vec::new(),
            number_of_variables: 0,
            max_worker: 4,
        }
    }
}

impl Ddnnf {
    /// Creates a new ddnnf including dead and core features
    pub fn new(graph: DdnnfGraph, root: NodeIndex, number_of_variables: u32) -> Ddnnf {
        let (kind, nodes, literals) = rebuild_graph(graph, root);

        let mut ddnnf = Ddnnf {
            kind,
            nodes,
            literals,
            core: HashSet::new(),
            md: Vec::new(),
            number_of_variables,
            max_worker: 4,
        };

        ddnnf.calculate_core();

        ddnnf
    }

    /// Checks whether this d-DNNF is trivial. A trivial d-DNNF is either a tautology or a
    /// contradiction.
    pub fn is_trivial(&self) -> bool {
        match self.kind {
            DdnnfKind::Tautology | DdnnfKind::Contradiction => true,
            DdnnfKind::NonTrivial => false,
        }
    }

    /// Loads a d-DNNF from file.
    pub fn from_file(path: &Path, features: Option<u32>) -> Self {
        crate::parser::build_ddnnf(path, features)
    }

    /// Returns the current count of the root node in the d-DNNF.
    ///
    /// This value is the same during all computations.
    pub fn rc(&self) -> BigInt {
        match self.kind {
            DdnnfKind::NonTrivial => self.nodes[self.nodes.len() - 1].count.clone(),
            DdnnfKind::Tautology => BigInt::from(2).pow(self.number_of_variables),
            DdnnfKind::Contradiction => BigInt::ZERO,
        }
    }

    /// Returns the core features of this d-DNNF.
    ///
    /// This is only calculated once at creation of the d-DNNF.
    pub fn get_core(&self) -> HashSet<i32> {
        self.core.clone()
    }

    // Returns the current temp count of the root node in the ddnnf.
    // That value is changed during computations
    fn rt(&self) -> BigInt {
        self.nodes[self.nodes.len() - 1].temp.clone()
    }

    /// Computes the total number of nodes in the dDNNF
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Computes the total number of edges in the dDNNF
    pub fn edge_count(&self) -> usize {
        let mut total_edges = 0;

        for node in self.nodes.iter() {
            match &node.ntype {
                NodeType::And { children } | NodeType::Or { children } => {
                    total_edges += children.len();
                }
                _ => (),
            }
        }
        total_edges
    }

    /// Computes the sharing of nodes in the dDNNF.
    /// We define sharing as #nodes / #nodes as tree
    pub fn sharing(&self) -> f64 {
        use crate::NodeType::*;
        let mut sub_nodes = vec![0_u64; self.node_count()];

        for (index, node) in self.nodes.iter().enumerate() {
            match &node.ntype {
                And { children } | Or { children } => {
                    sub_nodes[index] = children.iter().fold(0, |acc, &i| acc + sub_nodes[i]) + 1
                }
                _ => sub_nodes[index] = 1,
            }
        }
        self.node_count() as f64 / sub_nodes.last().unwrap().to_owned() as f64
    }

    /// Determines the positions of the inverted featueres
    pub fn map_features_opposing_indexes(&self, features: &[i32]) -> Vec<usize> {
        let mut indexes = Vec::with_capacity(features.len());
        for number in features {
            if let Some(i) = self.literals.get(&-number).cloned() {
                indexes.push(i);
            }
        }
        indexes
    }

    /// Executes a query.
    /// We use the in our opinion best type of query depending on the amount of features.
    ///
    /// # Example
    /// ```
    /// use std::path::Path;
    /// use ddnnife::Ddnnf;
    /// use ddnnife::parser::*;
    /// use num::BigInt;
    ///
    /// // create a ddnnf
    /// let file_path = Path::new("./tests/data/small_ex_c2d.nnf");
    /// let mut ddnnf: Ddnnf = build_ddnnf(file_path, None);
    ///
    /// assert_eq!(BigInt::from(1), ddnnf.execute_query(&vec![3,4]));
    /// assert_eq!(BigInt::from(2), ddnnf.execute_query(&vec![3]));
    pub fn execute_query(&mut self, features: &[i32]) -> BigInt {
        match features.len() {
            0 => self.rc(),
            1 => self.card_of_feature_with_marker(features[0]),
            2..=20 => {
                self.operate_on_partial_config_marker(features, Ddnnf::calc_count_marked_node)
            }
            _ => self.operate_on_partial_config_default(features, Ddnnf::calc_count),
        }
    }
}

impl Display for Ddnnf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "nnf {} {} {}",
            self.nodes.len(),
            0,
            self.number_of_variables
        )?;

        self.nodes.iter().try_for_each(|node| writeln!(f, "{node}"))
    }
}

#[cfg(test)]
mod test {
    use crate::parser::build_ddnnf;
    use std::path::Path;

    #[test]
    fn features_opposing_indexes() {
        let ddnnf = build_ddnnf(Path::new("tests/data/small_ex_c2d.nnf"), None);

        assert_eq!(
            vec![4, 2, 9],
            ddnnf.map_features_opposing_indexes(&[1, 2, 3, 4])
        );
        assert_eq!(
            vec![0, 1, 5, 8],
            ddnnf.map_features_opposing_indexes(&[-1, -2, -3, -4])
        );
    }
}
