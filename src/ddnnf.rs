pub mod stream;
pub mod node;
pub mod heuristics;
pub mod counting;
pub mod anomalies;
pub mod config_creation;

use rug::{Integer};
use rustc_hash::{FxHashMap, FxHashSet};

use self::node::{Node};

#[derive(Clone, Debug)]
/// A Ddnnf holds all the nodes as a vector, also includes meta data and further information that is used for optimations
pub struct Ddnnf {
    pub nodes: Vec<Node>,
    /// Literals for upwards propagation
    pub literals: FxHashMap<i32, usize>, // <var_number of the Literal, and the corresponding indize>
    true_nodes: Vec<usize>, // Indices of true nodes. In some cases those nodes needed to have special treatment
    /// The core features of the modell corresponding with this ddnnf
    pub core: FxHashSet<i32>,
    /// The dead features of the modell
    pub dead: FxHashSet<i32>,
    /// An interim save for the marking algorithm
    pub md: Vec<usize>,
    pub number_of_variables: u32,
    /// The number of threads
    pub max_worker: u16,
}

impl Default for Ddnnf {
    fn default() -> Self {
        Ddnnf {
            nodes: Vec::new(),
            literals: FxHashMap::default(),
            true_nodes: Vec::new(),
            core: FxHashSet::default(),
            dead: FxHashSet::default(),
            md: Vec::new(),
            number_of_variables: 0,
            max_worker: 4,
        }
    }
}

impl Ddnnf {
    /// Creates a new ddnnf including dead and core features
    pub fn new(
        nodes: Vec<Node>,
        literals: FxHashMap<i32, usize>,
        true_nodes: Vec<usize>,
        number_of_variables: u32,
    ) -> Ddnnf {
        let mut ddnnf = Ddnnf {
            nodes,
            literals,
            true_nodes,
            core: FxHashSet::default(),
            dead: FxHashSet::default(),
            md: Vec::new(),
            number_of_variables,
            max_worker: 4,
        };
        ddnnf.get_core();
        ddnnf.get_dead();
        ddnnf
    }

    // returns the current count of the root node in the ddnnf
    // that value is the same during all computations
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
    /// let file_path = "./tests/data/small_test.dimacs.nnf";
    /// let mut ddnnf: Ddnnf = build_ddnnf(file_path, None);
    ///
    /// assert_eq!(3, ddnnf.execute_query(&vec![3,1]));
    /// assert_eq!(3, ddnnf.execute_query(&vec![3]));
    pub fn execute_query(&mut self, features: &[i32]) -> Integer {
        match features.len() {
            0 => self.rc(),
            1 => self.card_of_feature_with_marker(features[0]),
            2..=20 => self.operate_on_partial_config_marker(features, Ddnnf::calc_count_marked_node),
            _ => self.operate_on_partial_config_default(features, Ddnnf::calc_count)
        }
    }
}