mod stream;
pub mod node;
mod heuristics;
mod counting;
mod multiple_queries;
pub mod anomalies;

use std::{collections::{HashMap, HashSet}, cmp::max};

use rug::{Integer};

use crate::parser::intermediate_representation::IntermediateGraph;

use self::node::{Node};

#[derive(Clone, Debug)]
/// A Ddnnf holds all the nodes as a vector, also includes meta data and further information that is used for optimations
pub struct Ddnnf {
    /// An intermediate representation that can be changed without destroying the structure
    pub inter_graph: IntermediateGraph,
    /// The nodes of the dDNNF
    pub nodes: Vec<Node>,
    /// Literals for upwards propagation
    pub literals: HashMap<i32, usize>, // <var_number of the Literal, and the corresponding indize>
    true_nodes: Vec<usize>, // Indices of true nodes. In some cases those nodes needed to have special treatment
    /// The core/dead features of the model corresponding with this ddnnf
    pub core: HashSet<i32>,
    /// An interim save for the marking algorithm
    pub md: Vec<usize>,
    /// The total number of variables
    pub number_of_variables: u32,
    /// The number of threads
    pub max_worker: u16,
}

impl Default for Ddnnf {
    fn default() -> Self {
        Ddnnf {
            inter_graph: IntermediateGraph::default(),
            nodes: Vec::new(),
            literals: HashMap::new(),
            true_nodes: Vec::new(),
            core: HashSet::new(),
            md: Vec::new(),
            number_of_variables: 0,
            max_worker: 4,
        }
    }
}

impl Ddnnf {
    /// Creates a new ddnnf including dead and core features
    pub fn new(
        mut inter_graph: IntermediateGraph,
        number_of_variables: u32,
    ) -> Ddnnf {
        let dfs_ig = inter_graph.rebuild(None);
        let mut ddnnf = Ddnnf {
            inter_graph,
            nodes: dfs_ig.0,
            literals: dfs_ig.1,
            true_nodes: dfs_ig.2,
            core: HashSet::new(),
            md: Vec::new(),
            number_of_variables,
            max_worker: 4,
        };
        ddnnf.get_core();
        ddnnf
    }

    /// We invalidate all collected data that belongs to the dDNNF and build it again
    /// by doing a DFS. That is necessary if we altered the intermedidate graph in any way.
    pub fn rebuild(&mut self) {
        let dfs_ig = self.inter_graph.rebuild(None);
        self.nodes = dfs_ig.0;
        self.literals = dfs_ig.1;
        self.true_nodes = dfs_ig.2;
        
        self.get_core();
        self.md.clear();
        // The highest absolute value of literals must be is also the number of variables because
        // there are no gaps in the feature to number mapping
        self.number_of_variables = self.literals.keys().fold(0, |acc, x| max(acc, x.abs() as u32));
    }

    /// Takes a list of clauses. Each clause consists out of one or multiple variables that are conjuncted.
    /// The clauses are disjuncted.
    /// Example: [[1, -2, 3], [4]] would represent (1 ∨ ¬2 ∨ 3) ∧ (4) 
    pub fn apply_changes(&mut self, clauses: &Vec<&[i32]>) {
        for clause in clauses {
            self.inter_graph.add_clause(clause);
        }
        
        self.rebuild();
    }

    /// Returns the current count of the root node in the ddnnf
    /// that value is the same during all computations
    pub fn rc(&self) -> Integer {
        self.nodes[self.nodes.len() - 1].count.clone()
    }

    // returns the current temp count of the root node in the ddnnf
    // that value is changed during computations
    fn rt(&self) -> Integer {
        self.nodes[self.nodes.len() - 1].temp.clone()
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

    /// executes a query
    /// we use the in our opinion best type of query depending on the amount of features
    /// 
    /// # Example
    /// ```
    /// extern crate ddnnf_lib;
    /// use ddnnf_lib::Ddnnf;
    /// use ddnnf_lib::parser::*;
    /// use rug::Integer;
    ///
    /// // create a ddnnf
    /// let file_path = "./tests/data/small_ex_c2d.nnf";
    /// let mut ddnnf: Ddnnf = build_ddnnf(file_path, None);
    ///
    /// assert_eq!(1, ddnnf.execute_query(&vec![3,4]));
    /// assert_eq!(2, ddnnf.execute_query(&vec![3]));
    pub fn execute_query(&mut self, features: &[i32]) -> Integer {
        match features.len() {
            0 => self.rc(),
            1 => self.card_of_feature_with_marker(features[0]),
            2..=20 => self.operate_on_partial_config_marker(features, Ddnnf::calc_count_marked_node),
            _ => self.operate_on_partial_config_default(features, Ddnnf::calc_count)
        }
    }
}


#[cfg(test)]
mod test {
    use serial_test::serial;

    use crate::parser::build_ddnnf;

    #[test]
    fn features_opposing_indexes() {
        let ddnnf = build_ddnnf("tests/data/small_ex_c2d.nnf", None);
        
        assert_eq!(vec![4, 2, 9], ddnnf.map_features_opposing_indexes(&[1, 2, 3, 4]));
        assert_eq!(vec![0, 1, 5, 8], ddnnf.map_features_opposing_indexes(&[-1, -2, -3, -4]));
    }

    #[test]
    fn rebuild_ddnnf() {
        let mut ddnnfs = Vec::new();
        ddnnfs.push(build_ddnnf("tests/data/auto1_c2d.nnf", None));
        ddnnfs.push(build_ddnnf("tests/data/auto1_d4.nnf", Some(2513)));
        ddnnfs.push(build_ddnnf("tests/data/VP9_d4.nnf", Some(42)));
        ddnnfs.push(build_ddnnf("tests/data/small_ex_c2d.nnf", None));

        for ddnnf in ddnnfs {
            let mut rebuild_clone = ddnnf.clone();
            // rebuild 10 times to ensure that negative effects do not occur no matter
            // how many times we rebuild
            for _ in 0..10 { rebuild_clone.rebuild(); }

            // check each field seperatly due to intermediate_graph.graph not being able to derive PartialEq
            assert_eq!(ddnnf.nodes, rebuild_clone.nodes);
            assert_eq!(ddnnf.literals, rebuild_clone.literals);
            assert_eq!(ddnnf.core, rebuild_clone.core);
            assert_eq!(ddnnf.true_nodes, rebuild_clone.true_nodes);
            assert_eq!(ddnnf.number_of_variables, rebuild_clone.number_of_variables);
        }
    }

    #[test]
    #[serial]
    fn incremental_applying_clause() {
        let ddnnf_file_paths = vec![
            ("tests/data/small_ex_c2d.nnf", 4, vec![4]),
            //("tests/data/VP9_d4.nnf", 42, vec![vec![4, 5]])
        ];

        for (path, features, clause) in ddnnf_file_paths {
            let mut ddnnf = build_ddnnf(path, Some(features));
            println!("Card of Features before change:");
            for i in 0..ddnnf.number_of_variables {
                println!("{i}: {:?}", ddnnf.execute_query(&[i as i32]));
            }

            ddnnf.apply_changes(&vec![&clause]);
            println!("Card of Features after change:");
            for i in 0..ddnnf.number_of_variables {
                println!("{i}: {:?}", ddnnf.execute_query(&[i as i32]));
            }
        }
    }
}