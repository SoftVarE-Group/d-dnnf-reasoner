pub mod stream;
pub mod node;
pub mod heuristics;
pub mod counting;
pub mod anomalies;

use rug::{Integer};

use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use self::node::{Node};

#[derive(Debug, Clone)]
/// A Ddnnf holds all the nodes as a vector, also includes meta data and further information that is used for optimations
pub struct Ddnnf {
    pub nodes: Vec<Node>,
    /// Literals for upwards propagation
    pub literals: HashMap<i32, usize>, // <var_number of the Literal, and the corresponding indize>
    pub number_of_variables: u32,
    pub number_of_nodes: usize,
    /// The number of threads
    pub max_worker: u16,
    /// An interim save for the marking algorithm
    pub md: Vec<usize>,
    /// The core features of the modell corresponding with this ddnnf
    pub core: HashSet<i32>,
    /// The dead features of the modell
    pub dead: HashSet<i32>,
}

impl Ddnnf {
    /// Creates a new ddnnf including dead and core features
    pub fn new(
        nodes: Vec<Node>,
        literals: HashMap<i32, usize>,
        number_of_variables: u32,
        number_of_nodes: usize,
    ) -> Ddnnf {
        let mut ddnnf = Ddnnf {
            nodes,
            literals,
            number_of_variables,
            number_of_nodes,
            max_worker: 4,
            md: Vec::new(),
            core: HashSet::new(),
            dead: HashSet::new(),
        };
        ddnnf.get_core();
        ddnnf.get_dead();
        ddnnf
    }

    // returns the current count of the root node in the ddnnf
    // that value is the same during all computations
    pub fn rc(&self) -> Integer {
        self.nodes[self.number_of_nodes - 1].count.clone()
    }

    // returns the current temp count of the root node in the ddnnf
    // that value is changed during computations
    fn rt(&self) -> Integer {
        self.nodes[self.number_of_nodes - 1].temp.clone()
    }

    /// Executes a single query. This is used for the interactive mode of ddnnife
    /// We change between the marking and non-marking approach depending on the number of features
    /// that are either included or excluded. All configurations <= 10 use the marking approach all
    /// other the "standard" approach. Results are printed directly on the console together with the needed time.
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
    /// let mut ddnnf: Ddnnf = build_ddnnf_tree_with_extras(file_path);
    ///
    /// ddnnf.execute_query_interactive(vec![3,1]); // 3
    /// ddnnf.execute_query_interactive(vec![3]); // also 3
    ///
    /// ```
    pub fn execute_query_interactive(&mut self, mut features: Vec<i32>) {
        if features.len() == 1 {
            let f = features.pop().unwrap();
            let time = Instant::now();
            let res: (usize, Integer) = self.card_of_feature_with_marker(f);
            let elapsed_time = time.elapsed().as_secs_f32();

            println!("Feature count for feature number {:?}: {:#?}", f, res);
            println!("{:?} nodes were marked (note that nodes which are on multiple paths are only marked once). That are ≈{:.2}% of the ({}) total nodes",
                res.0, f64::from(res.0 as u32)/self.number_of_nodes as f64 * 100.0, self.number_of_nodes);
            println!(
                "Elapsed time for one feature: {:.6} seconds.\n",
                elapsed_time
            );
        } else if features.len() <= 10 {
            let time = Instant::now();
            let res: (usize, Integer) =
                self.operate_on_partial_config_marker(&features, Ddnnf::calc_count_marked_node);
            let elapsed_time = time.elapsed().as_secs_f32();

            println!(
                "The cardinality for the partial configuration {:?} is: {:#?}",
                features, res.1
            );
            println!("{:?} nodes were marked (note that nodes which are on multiple paths are only marked once). That are ≈{:.2}% of the ({}) total nodes",
                res.0, f64::from(res.0 as u32)/self.number_of_nodes as f64 * 100.0, self.number_of_nodes);
            println!(
                "Elapsed time for a partial configuration in seconds: {:.6}s.",
                elapsed_time
            );
        } else {
            let time = Instant::now();
            let res: Integer = self.operate_on_partial_config_default(&features, Ddnnf::calc_count);
            let elapsed_time = time.elapsed().as_secs_f32();

            println!(
                "The cardinality for the partial configuration {:?} is: {:#?}",
                features, res
            );
            println!(
                "Elapsed time for a partial configuration in seconds: {:.6}s.",
                elapsed_time
            );
        }
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
    /// let mut ddnnf: Ddnnf = build_ddnnf_tree_with_extras(file_path);
    ///
    /// assert_eq!(3, ddnnf.execute_query(&vec![3,1]));
    /// assert_eq!(3, ddnnf.execute_query(&vec![3]));
    pub fn execute_query(&mut self, features: &[i32]) -> Integer {
        match features.len() {
            0 => self.rc(),
            1 => self.card_of_feature_with_marker(features[0]).1,
            2..=50 => self.operate_on_partial_config_marker(&features, Ddnnf::calc_count_marked_node).1,
            _ => self.operate_on_partial_config_default(&features, Ddnnf::calc_count)
        }
    }

    pub fn is_sat_for(&mut self, features: &[i32]) -> bool {
        match features.len() {
            0 => self.rc() > 0,
            1..=20 => self.operate_on_partial_config_marker(features, Ddnnf::sat_marked_node).1 > 0,
            _ => self.operate_on_partial_config_default(features, Ddnnf::sat_node_default) > 0
        }
    }
}