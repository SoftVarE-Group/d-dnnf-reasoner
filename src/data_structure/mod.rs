use rug::{Assign, Complete, Float, Integer};

use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    thread,
    time::Instant,
};

use std::sync::mpsc;
use workctl::WorkQueue;

use crate::parser;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
/// The Type of the Node declares how we handle the computation for the different types of cardinalities
pub enum NodeType {
    /// The cardinality of an And node is always the product of its childs
    And,
    /// The cardinality of an Or node is the sum of its two children
    Or,
    /// The cardinality is one if not declared otherwise due to some query
    Literal,
    /// The cardinality is one
    True,
    /// The cardinality is zero
    False,
}

use NodeType::{And, False, Literal, Or, True};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Represents all types of Nodes with its different parts
pub struct Node {
    marker: bool,
    /// The cardinality of the node for the cardinality of a feature model
    pub count: Integer,
    /// The cardinality during the different queries
    pub temp: Integer,
    pub node_type: NodeType,
    /// And and Or nodes hold children
    pub children: Option<Vec<usize>>,
    /// Every node excpet the root has (multiple) parent nodes
    pub parents: Vec<usize>,
    /// Literals hold a variable number
    pub var_number: Option<i32>,
}

static mut COUNTER: u64 = 0;

impl Node {
    #[inline]
    /// Creates a new And node
    pub fn new_and(overall_count: Integer, children: Vec<usize>) -> Node {
        Node {
            marker: false,
            count: overall_count,
            temp: Integer::from(0),
            node_type: And,
            children: Some(children),
            parents: Vec::new(),
            var_number: None,
        }
    }

    #[inline]
    /// Creates a new Or node
    pub fn new_or(
        var_number: i32,
        overall_count: Integer,
        children: Vec<usize>,
    ) -> Node {
        Node {
            marker: false,
            count: overall_count,
            temp: Integer::from(0),
            node_type: Or,
            children: Some(children),
            parents: Vec::new(),
            var_number: Some(var_number),
        }
    }

    #[inline]
    /// Creates a new Literal node
    pub fn new_literal(var_number: i32) -> Node {
        Node {
            marker: false,
            count: Integer::from(1),
            temp: Integer::from(0),
            node_type: Literal,
            children: None,
            parents: Vec::new(),
            var_number: Some(var_number),
        }
    }

    #[inline]
    /// Creates either a new True or False node
    pub fn new_bool(node_type: NodeType) -> Node {
        let count: Integer = match node_type {
            True => Integer::from(1),
            False => Integer::from(0),
            _ => panic!("NodeType {:?} is not a Bool", node_type),
        };

        Node {
            marker: false,
            count,
            temp: Integer::from(0),
            node_type,
            children: None,
            parents: Vec::new(),
            var_number: None,
        }
    }
}

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

    /// Computes all core features
    /// A feature is a core feature iff there exists only the positiv occurence of that feature
    fn get_core(&mut self) {
        self.core = (1..=self.number_of_variables as i32)
            .filter(|f| {
                self.literals.get(f).is_some()
                    && self.literals.get(&-f).is_none()
            })
            .collect::<HashSet<i32>>()
    }

    /// Computes all dead features
    /// A feature is a dead feature iff there exists only the negativ occurence of that feature
    fn get_dead(&mut self) {
        self.dead = (1..=self.number_of_variables as i32)
            .filter(|f| {
                self.literals.get(f).is_none()
                    && self.literals.get(&-f).is_some()
            })
            .collect::<HashSet<i32>>()
    }

    #[inline]
    /// Reduces a query by removing included core features and excluded dead features
    fn reduce_query(&mut self, features: &[i32]) -> Vec<i32> {
        features
            .iter()
            .filter({
                // filter keeps the elements which do fulfill the defined boolean formula. Thats why we need to use the ! operator
                |&f| {
                    if f > &0 {
                        // remove included core and excluded dead features
                        !self.core.contains(f)
                    } else {
                        !self.dead.contains(&-f)
                    }
                }
            })
            .copied()
            .collect::<Vec<i32>>()
    }

    #[inline]
    /// Checks if a query is satisfiable. That is not the case if either a core feature is excluded or a dead feature is included
    fn query_is_not_sat(&mut self, features: &[i32]) -> bool {
        // if there is an included dead or an excluded core feature
        features.iter().any({
            |&f| {
                if f > 0 {
                    self.dead.contains(&f)
                } else {
                    self.core.contains(&-f)
                }
            }
        })
    }

    #[inline]
    // Computes the cardinality of a node according to the rules mentioned at the Nodetypes and saves it in tmp
    fn calc_count(&mut self, i: usize) {
        match self.nodes[i].node_type {
            And => {
                self.nodes[i].temp = Integer::product(
                    self.nodes[i]
                        .children
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|&indize| &self.nodes[indize].temp),
                )
                .complete()
            }
            Or => {
                self.nodes[i].temp = Integer::sum(
                    self.nodes[i]
                        .children
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|&indize| &self.nodes[indize].temp),
                )
                .complete()
            }
            False => self.nodes[i].temp.assign(0),
            _ => self.nodes[i].temp.assign(1),
        }
    }

    // returns the current count of the root node in the ddnnf
    // that value is the same during all computations
    fn rc(&self) -> Integer {
        self.nodes[self.number_of_nodes - 1].count.clone()
    }

    // returns the current temp count of the root node in the ddnnf
    // that value is changed during computations
    fn rt(&self) -> Integer {
        self.nodes[self.number_of_nodes - 1].temp.clone()
    }

    #[inline]
    /// Computes the cardinality of a feature according
    /// to the rules mentioned at the Nodetypes and returns the result.
    /// The function does not use the marking approach.
    /// This function trys to apply optimisations based on core and dead features.
    ///
    /// # Example
    /// ```
    /// extern crate ddnnf_lib;
    /// use ddnnf_lib::data_structure::*;
    /// use ddnnf_lib::parser::*;
    /// use rug::Integer;
    ///
    /// // create a ddnnf
    /// let file_path = "./tests/data/small_test.dimacs.nnf";
    /// let mut ddnnf: Ddnnf = build_ddnnf_tree_with_extras(file_path);
    ///
    /// assert_eq!(Integer::from(3), ddnnf.card_of_feature(3));
    ///
    /// ```
    pub fn card_of_feature(&mut self, feature: i32) -> Integer {
        if self.core.contains(&feature) {
            self.rc()
        } else if self.dead.contains(&feature) {
            Integer::from(0)
        } else {
            for i in 0..self.number_of_nodes {
                if self.nodes[i].node_type == Literal // search for the node we want to adjust
                    && feature == -self.nodes[i].var_number.unwrap()
                {
                    self.nodes[i].temp.assign(0);
                } else {
                    // all other values stay the same, respectevly get adjusted because the count of the one node got changed
                    self.calc_count(i)
                }
            }
            self.rt()
        }
    }

    #[inline]
    /// Computes the cardinality of a partial configuration
    /// according to the rules mentioned at the Nodetypes and returns the result.
    /// The function does not use the marking approach.
    /// This function trys to apply optimisations based on core and dead features.
    ///
    /// # Example
    /// ```
    /// extern crate ddnnf_lib;
    /// use ddnnf_lib::data_structure::*;
    /// use ddnnf_lib::parser::*;
    /// use rug::Integer;
    ///
    /// // create a ddnnf
    /// let file_path = "./tests/data/small_test.dimacs.nnf";
    /// let mut ddnnf: Ddnnf = build_ddnnf_tree_with_extras(file_path);
    ///
    /// assert_eq!(Integer::from(3), ddnnf.card_of_partial_config(&vec![1,3]));
    ///
    /// ```
    pub fn card_of_partial_config(&mut self, features: &[i32]) -> Integer {
        if self.query_is_not_sat(features) {
            Integer::from(0)
        } else {
            let features: Vec<i32> = self.reduce_query(features);
            for i in 0..self.number_of_nodes {
                if self.nodes[i].node_type == Literal
                    && features.contains(&-self.nodes[i].var_number.unwrap())
                {
                    self.nodes[i].temp.assign(0);
                } else {
                    self.calc_count(i)
                }
            }

            self.rt()
        }
    }

    #[inline]
    // Computes the cardinality of a node using the marking algorithm:
    // And and Or nodes that base the computation on their child nodes use the
    // .temp value if the child node is marked and the .count value if not
    fn calc_count_marked_node(&mut self, i: usize) {
        match self.nodes[i].node_type {
            And => {
                self.nodes[i].temp = Integer::product(
                    self.nodes[i].children.as_ref().unwrap().iter().map(
                        |&indize| {
                            if self.nodes[indize].marker {
                                &self.nodes[indize].temp
                            } else {
                                &self.nodes[indize].count
                            }
                        },
                    ),
                )
                .complete()
            }
            Or => {
                self.nodes[i].temp = Integer::sum(
                    self.nodes[i].children.as_ref().unwrap().iter().map(
                        |&indize| {
                            if self.nodes[indize].marker {
                                &self.nodes[indize].temp
                            } else {
                                &self.nodes[indize].count
                            }
                        },
                    ),
                )
                .complete()
            }
            False => self.nodes[i].temp.assign(0),
            _ => self.nodes[i].temp.assign(1),
        }
    }

    #[inline]
    // Computes the cardinality of a feature and partial configurations using the marking algorithm
    fn calc_count_marker(&mut self, indize: &[usize]) -> Integer {
        for i in indize.iter().copied() {
            self.nodes[i].temp.assign(0); // change the value of the node
            self.mark_nodes(i); // go through the path til the root node is marked
        }

        // sort the marked nodes so that we make sure to first calculate the childnodes and then their parents
        self.md.sort_unstable();

        // calc the count for all marked nodes, respectevly all nodes that matter
        for j in 0..self.md.len() {
            if !indize.contains(&self.md[j]) {
                self.calc_count_marked_node(self.md[j]);
            }
        }

        // reset everything
        for j in 0..self.md.len() {
            self.nodes[self.md[j]].marker = false
        }
        unsafe { COUNTER = self.md.len() as u64 }
        self.md.clear();

        // the result is propagated through the whole graph up to the root
        self.rt()
    }

    #[inline]
    /// Computes the cardinality of a feature using the marking algorithm.
    /// The marking algorithm differs to the standard variation by only reomputing the
    /// marked nodes. Further, the marked nodes use the .temp value of the childs nodes if they
    /// are also marked and the .count value if they are not.
    ///
    /// # Example
    /// ```
    /// extern crate ddnnf_lib;
    /// use ddnnf_lib::data_structure::*;
    /// use ddnnf_lib::parser::*;
    /// use rug::Integer;
    ///
    /// // create a ddnnf
    /// let file_path = "./tests/data/small_test.dimacs.nnf";
    /// let mut ddnnf: Ddnnf = build_ddnnf_tree_with_extras(file_path);
    ///
    /// assert_eq!(Integer::from(3), ddnnf.card_of_feature_with_marker(3));
    ///
    /// ```
    pub fn card_of_feature_with_marker(&mut self, feature: i32) -> Integer {
        if self.core.contains(&feature) || self.dead.contains(&-feature) {
            self.rc()
        } else if self.dead.contains(&feature) || self.core.contains(&-feature)
        {
            Integer::from(0)
        } else {
            match self.literals.get(&-feature).cloned() {
                Some(i) => self.calc_count_marker(&[i]),
                // there is no literal corresponding to the feature number and because of that we don't have to do anything besides returning the count of the model
                None => self.rc(),
            }
        }
    }

    #[inline]
    /// Computes the cardinality of a partial configuration using the marking algorithm.
    /// Works analog to the card_of_feature_with_marker()
    ///
    /// # Example
    /// ```
    /// extern crate ddnnf_lib;
    /// use ddnnf_lib::data_structure::*;
    /// use ddnnf_lib::parser::*;
    /// use rug::Integer;
    ///
    /// // create a ddnnf
    /// let file_path = "./tests/data/small_test.dimacs.nnf";
    /// let mut ddnnf: Ddnnf = build_ddnnf_tree_with_extras(file_path);
    ///
    /// assert_eq!(Integer::from(3), ddnnf.card_of_partial_config_with_marker(&vec![3,1]));
    ///
    /// ```
    pub fn card_of_partial_config_with_marker(
        &mut self,
        features: &[i32],
    ) -> Integer {
        if self.query_is_not_sat(features) {
            Integer::from(0)
        } else {
            let features: Vec<i32> = self.reduce_query(features);

            let mut indize: Vec<usize> = Vec::new();

            for number in features {
                // we set the negative occurences to 0 for the features we want to include
                if let Some(i) = self.literals.get(&-number).cloned() {
                    indize.push(i)
                }
            }

            if indize.is_empty() {
                return self.rc();
            }

            self.calc_count_marker(&indize)
        }
    }

    #[inline]
    // marks the nodes starting from an initial Literal. All parents and parents of parents til
    // the root nodes get marked
    fn mark_nodes(&mut self, i: usize) {
        self.nodes[i].marker = true;
        self.md.push(i);

        for j in self.nodes[i].parents.clone() {
            // check for parent nodes and adjust their count resulting of the changes to their children
            if !self.nodes[j].marker {
                // only mark those nodes which aren't already marked to specificly avoid marking nodes near the root multple times
                self.mark_nodes(j);
            }
        }
    }

    /// Executes a single query. This is used for the interactive mode of ddnnife
    /// We change between the marking and non-marking approach depending on the number of features
    /// that are either included or excluded. All configurations <= 10 use the marking approach all
    /// other the "standard" approach. Results are printed directly on the console together with the needed time.
    ///
    /// # Example
    /// ```
    /// extern crate ddnnf_lib;
    /// use ddnnf_lib::data_structure::*;
    /// use ddnnf_lib::parser::*;
    /// use rug::Integer;
    ///
    /// // create a ddnnf
    /// let file_path = "./tests/data/small_test.dimacs.nnf";
    /// let mut ddnnf: Ddnnf = build_ddnnf_tree_with_extras(file_path);
    ///
    /// ddnnf.execute_query(vec![3,1]); // 3
    /// ddnnf.execute_query(vec![3]); // also 3
    ///
    /// ```
    pub fn execute_query(&mut self, mut features: Vec<i32>) {
        if features.len() == 1 {
            let f = features.pop().unwrap();
            let time = Instant::now();
            let res: Integer = self.card_of_feature_with_marker(f);
            let elapsed_time = time.elapsed().as_secs_f32();
            let count = unsafe { COUNTER };

            println!("Feature count for feature number {:?}: {:#?}", f, res);
            println!("{:?} nodes were marked (note that nodes which are on multiple paths are only marked once). That are ≈{:.2}% of the ({}) total nodes",
                count, f64::from(count as u32)/self.number_of_nodes as f64 * 100.0, self.number_of_nodes);
            println!(
                "Elapsed time for one feature: {:.6} seconds.\n",
                elapsed_time
            );
        } else if features.len() <= 10 {
            let time = Instant::now();
            let res: Integer =
                self.card_of_partial_config_with_marker(&features);
            let elapsed_time = time.elapsed().as_secs_f32();
            let count = unsafe { COUNTER };

            println!(
                "The cardinality for the partial configuration {:?} is: {:#?}",
                features, res
            );
            println!("{:?} nodes were marked (note that nodes which are on multiple paths are only marked once). That are ≈{:.2}% of the ({}) total nodes",
                count, f64::from(count as u32)/self.number_of_nodes as f64 * 100.0, self.number_of_nodes);
            println!(
                "Elapsed time for a partial configuration in seconds: {:.6}s.",
                elapsed_time
            );
        } else {
            let time = Instant::now();
            let res: Integer = self.card_of_partial_config(&features);
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
}

// The Problem:
//  We have lots of computations that each require a huge data structure (our Ddnnf).
//  Each thread needs a Ddnnf and they are not allowed to share one.
//  Furthermore, we don't want to clone the initial ddnnf for each thread because
//  the cost for that operation is even bigger than one computation.
//
// The Solution:
//  Create a queue which can be safely shared between threads from which those
//  threads can pull work (the feature that should be computed now) each time they finish their current work.
//
// We assume that we have MAX_WORKER processor cores which will do work for us.
// You could use the num_cpus crate to find this for a particular machine.

impl Ddnnf {
    #[inline]
    /// Computes the cardinality of features for all features in a modell.
    /// The results are saved in the file_path. The .csv ending always gets added to the user input.
    /// The function exclusively uses the marking based function.
    /// Here the number of threads influence the speed by using a shared work queue.
    /// # Example
    /// ```
    /// extern crate ddnnf_lib;
    /// use ddnnf_lib::data_structure::*;
    /// use ddnnf_lib::parser::*;
    /// use rug::Integer;
    /// use std::fs;
    /// 
    /// // create a ddnnf
    /// // and run the queries
    /// let mut ddnnf: Ddnnf = build_ddnnf_tree_with_extras("./tests/data/small_test.dimacs.nnf");
    /// ddnnf.card_of_each_feature_to_csv("./tests/data/smt_out.csv")
    ///      .unwrap_or_default();
    /// let _rm = fs::remove_file("./tests/data/smt_out.csv");
    ///
    /// ```
    pub fn card_of_each_feature_to_csv(
        &mut self,
        file_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        let mut queue: WorkQueue<i32> =
            WorkQueue::with_capacity(self.number_of_variables as usize);

        // Create a MPSC (Multiple Producer, Single Consumer) channel. Every worker
        // is a producer, the main thread is a consumer; the producers put their
        // work into the channel when it's done.
        let (results_tx, results_rx) = mpsc::channel();

        let mut threads = Vec::new();

        for e in 1..&self.number_of_variables + 1 {
            queue.push_work(e as i32); // x..y are all numbers between x and y including x, excluding y
        }

        for _ in 0..self.max_worker {
            let mut t_queue = queue.clone();
            let t_results_tx = results_tx.clone();
            let mut ddnnf: Ddnnf = self.clone();

            // thread::spawn takes a closure (an anonymous function that "closes"
            // over its environment). The move keyword means it takes ownership of
            // those variables, meaning they can't be used again in the main thread.
            let handle = thread::spawn(move || {
                // Loop while there's expected to be work, looking for work.
                // If work is available, do that work.
                while let Some(work) = t_queue.pull_work() {
                    let result = ddnnf.card_of_feature_with_marker(work);

                    // Send the work and the result of that work.
                    //
                    // Sending could fail. If so, there's no use in
                    // doing any more work, so abort.
                    match t_results_tx.send((work, result)) {
                        Ok(_) => (),
                        Err(_) => {
                            break;
                        }
                    }
                }

                // Signal to the operating system that now is a good time
                // to give another thread a chance to run.
                std::thread::yield_now();
            });

            // Add the handle for the newly spawned thread to the list of handles
            threads.push(handle);
        }

        let mut results: Vec<(i32, String, String)> =
            Vec::with_capacity(self.number_of_variables as usize);

        // Get completed work from the channel while there's work to be done.
        for _ in 0..self.number_of_variables {
            match results_rx.recv() {
                // If the control thread successfully receives, a job was completed.
                Ok((feature, cardinality)) => {
                    // Caching the results
                    results.push((
                        feature,
                        cardinality.to_string(),
                        format!(
                            "{:.20}",
                            Float::with_val(200, cardinality) / self.rc()
                        ),
                    ));
                }
                Err(_) => {
                    panic!("All workers died unexpectedly.");
                }
            }
        }

        // Just make sure that all the other threads are done.
        for handle in threads {
            handle.join().unwrap();
        }

        results.sort_unstable_by_key(|k| k.0);
        // start the csv writer with the file_path
        let mut wtr = csv::Writer::from_path(file_path)?;

        for element in results {
            wtr.write_record(vec![
                element.0.to_string(),
                element.1,
                element.2,
            ])?;
        }

        // Flush everything into the file that is still in a buffer
        // Now we finished writing the csv file
        wtr.flush()?;

        // If everything worked as expected, then we can return Ok(()) and we are happy :D
        Ok(())
    }

    #[inline]
    /// Computes the cardinality of partial configurations for all queries in path_in.
    /// The results are saved in the path_out. The .txt ending always gets added to the user input.
    /// The function uses the marking approach for configurations with <= 10 features. For larger
    /// queries we use the "standard" approach without marking.
    /// Here the number of threads influence the speed by using a shared work queue.
    /// The result can be in a different order for iff multiple threads are used.
    /// # Example
    /// ```
    /// extern crate ddnnf_lib;
    /// use ddnnf_lib::data_structure::*;
    /// use ddnnf_lib::parser::*;
    /// use rug::Integer;
    /// use std::fs;
    /// 
    /// // create a ddnnf
    /// // and run the queries
    /// let mut ddnnf: Ddnnf = build_ddnnf_tree_with_extras("./tests/data/small_test.dimacs.nnf");
    /// ddnnf.card_multi_queries(
    ///     "./tests/data/small_test.config",
    ///     "./tests/data/smt_out.txt",)
    ///     .unwrap_or_default();
    /// let _rm = fs::remove_file("./tests/data/smt_out.txt");
    /// 
    /// ```
    pub fn card_multi_queries(
        &mut self,
        path_in: &str,
        path_out: &str,
    ) -> Result<(), Box<dyn Error>> {
        let work: Vec<Vec<i32>> = parser::parse_queries_file(path_in);
        let mut queue = WorkQueue::with_capacity(work.len());

        for e in work.clone() {
            queue.push_work(e);
        }

        let (results_tx, results_rx) = mpsc::channel();

        let mut threads = Vec::new();

        for _ in 0..self.max_worker {
            let mut t_queue = queue.clone();
            let t_results_tx = results_tx.clone();
            let mut ddnnf: Ddnnf = self.clone();

            // thread::spawn takes a closure (an anonymous function that "closes"
            // over its environment). The move keyword means it takes ownership of
            // those variables, meaning they can't be used again in the main thread.
            let handle = thread::spawn(move || {
                // Loop while there's expected to be work, looking for work.
                // If work is available, do that work.
                while let Some(work) = t_queue.pull_work() {
                    // Do some work.
                    let result;
                    if work.len() <= 100 {
                        result = ddnnf.card_of_partial_config_with_marker(&work);
                    } else {
                        result = ddnnf.card_of_partial_config(&work);
                    };

                    // Send the work and the result of that work.
                    //
                    // Sending could fail. If so, there's no use in
                    // doing any more work, so abort.
                    match t_results_tx.send((work, result)) {
                        Ok(_) => (),
                        Err(_) => {
                            break;
                        }
                    }
                }

                // Signal to the operating system that now is a good time
                // to give another thread a chance to run.
                std::thread::yield_now();
            });

            // Add the handle for the newly spawned thread to the list of handles
            threads.push(handle);
        }

        // start the file writer with the file_path
        let f = File::create(path_out).expect("Unable to create file");
        let mut wtr = BufWriter::new(f);

        // Get completed work from the channel while there's work to be done.
        for _ in 0..work.len() {
            match results_rx.recv() {
                // If the control thread successfully receives, a job was completed.
                Ok((features, cardinality)) => {
                    let mut features_str =
                        features.iter().fold(String::new(), |acc, &num| {
                            acc + &num.to_string() + " "
                        });
                    features_str.pop();
                    let data = &format!("{},{}\n", features_str, cardinality);
                    wtr.write_all(data.as_bytes())?;
                }
                // If the control thread is the one left standing, that's pretty
                // problematic.
                Err(_) => {
                    panic!("All workers died unexpectedly.");
                }
            }
        }

        // Just make sure that all the other threads are done.
        for handle in threads {
            handle.join().unwrap();
        }

        // Flush everything into the file that is still in a buffer
        // Now we finished writing the csv file
        wtr.flush()?;

        // If everything worked as expected, then we can return Ok(()) and we are happy :D
        Ok(())
    }
}

/*---------------------------------HEURISTICS-----------------------------------*/

impl Ddnnf {
    /// Computes and prints some heuristics including:
    /// 1) The distribution of the different types of nodes
    /// 2) The number of child nodes (averages, ...)
    /// 3) The length of paths starting from the root to the leafs (averages, ...)
    pub fn print_all_heuristics(&mut self) {
        self.get_nodetype_numbers();
        self.get_child_number();
        self.get_depths()
    }

    // computes the occurences of different node types (number of and nodes, or, positive literal, negative literal, true, false)
    fn get_nodetype_numbers(&mut self) {
        let mut and_counter = 0;
        let mut or_counter = 0;
        let mut literal_counter = 0;
        let mut true_counter = 0;
        let mut false_counter = 0;

        for i in 0..self.nodes.len() {
            match &self.nodes[i].node_type {
                And => and_counter += 1,
                Or => or_counter += 1,
                Literal => literal_counter += 1,
                True => true_counter += 1,
                False => false_counter += 1,
            }
        }

        let node_count: u64 = self.nodes.len() as u64;
        println!(
            "\nThe d-DNNF consists out of the following node types:\n\
            \t |-> {:?} out of {:?} are And nodes (≈{:.2}% of total)\n\
            \t |-> {:?} out of {:?} are Or nodes (≈{:.2}% of total)\n\
            \t |-> {:?} out of {:?} are Literal nodes (≈{:.2}% of total)\n\
            \t |-> {:?} out of {:?} are True nodes (≈{:.2}% of total)\n\
            \t |-> {:?} out of {:?} are False nodes (≈{:.2}% of total)\n",
            and_counter,
            node_count,
            (f64::from(and_counter) / node_count as f64) * 100_f64,
            or_counter,
            node_count,
            (f64::from(or_counter) / node_count as f64) * 100_f64,
            literal_counter,
            node_count,
            (f64::from(literal_counter) / node_count as f64) * 100_f64,
            true_counter,
            node_count,
            (f64::from(true_counter) / node_count as f64) * 100_f64,
            false_counter,
            node_count,
            (f64::from(false_counter) / node_count as f64) * 100_f64
        );
    }

    // computes the number of childs for the differnt nodes (count of total nodes, childs relativ to number of nodes)
    fn get_child_number(&mut self) {
        let mut total_child_counter: u64 = 0;
        let mut and_child_counter: u64 = 0;
        let mut and_counter = 0;

        for i in 0..self.nodes.len() {
            match &self.nodes[i].children {
                Some(x) => {
                    total_child_counter += x.len() as u64;
                    if self.nodes[i].node_type == And {
                        and_child_counter += x.len() as u64;
                    }
                }
                None => continue,
            }
            if self.nodes[i].node_type == And {
                and_counter += 1
            }
        }

        let node_count: u64 = self.nodes.len() as u64;
        println!(
            "\nThe d-DNNF has the following information regarding node count:\n\
                \t |-> The overall count of child connections is {:?}\n\
                \t |-> The overall node count is {:?}.\n\
                \t |-> There are {:.2} times as much connections as nodes\n\
                \t |-> Each of the {:?} And nodes has an average of ≈{:.2} child nodes\n",
            total_child_counter,
            node_count,
            total_child_counter as f64 / node_count as f64,
            and_counter,
            and_child_counter as f64 / and_counter as f64
        );
    }

    // the standard deviation (s_x) is defined as sqrt((1/n) * sum over (length of a path - length of the mean path)² for each path)
    // (lowest, highest, mean, s_x, #paths)
    #[inline]
    fn get_depths(&mut self) {
        let mut lowest: u64 = u64::MAX;
        let mut highest: u64 = 0;
        let mut mean: f64 = 0.0;

        let depths: Vec<u64> = get_depth(&self.nodes, self.nodes.len() - 1, 0);
        let length: u64 = depths.len() as u64;

        for depth in depths.clone() {
            if depth > highest {
                highest = depth;
            }
            if depth < lowest {
                lowest = depth;
            }
            mean += depth as f64;
        }
        mean /= length as f64;

        let mut derivation: f64 = 0.0;

        for depth in depths {
            derivation += (depth as f64 - mean).powi(2);
        }

        let s_x: f64 = (derivation / length as f64).sqrt();

        println!("\nThe d-DNNF has the following length attributes:\n\
                \t |-> The shortest path is {:?} units long\n\
                \t |-> The longest path is {:?} units long\n\
                \t |-> The mean path is ≈{:.2} units long\n\
                \t |-> The standard derivation is ≈{:.2} units\n\
                \t |-> There are {:?} different paths. (different paths can sometimes just differ by one node)\n",
                lowest, highest, mean, s_x, length);
    }
}

#[inline]
// computes the depth/length of a path starting from indize to the leaf
fn get_depth(nodes: &[Node], indize: usize, count: u64) -> Vec<u64> {
    let current: &Node = &nodes[indize];

    if current.node_type != And && current.node_type != Or {
        vec![count]
    } else {
        match &current.children {
            Some(children) => {
                let mut child_depths: Vec<Vec<u64>> = Vec::new();
                for i in children {
                    child_depths.push(get_depth(nodes, *i, count + 1));
                }
                child_depths.into_iter().flatten().collect()
            }
            None => panic!("An inner node does not have child nodes. That should not happen!"),
        }
    }
}

// Functions that are currently not used but necessary to collect data regarding marking percentages,...
// With them we can compute the average, median and stdev.
// They were used to collect the data about the nodes visited when using the marking algorithm
#[allow(dead_code)]
fn average(data: &[u64]) -> f64 {
    let sum = data.iter().sum::<u64>() as f64;
    println!("{}", sum);
    let count = data.len();

    match count {
        positive if positive > 0 => sum / count as f64,
        _ => -1.0,
    }
}

#[allow(dead_code)]
fn median(data: &mut [u64]) -> f64 {
    data.sort_unstable();
    let size = data.len();
    if size < 1 {
        return -1.0;
    }

    match size {
        even if even % 2 == 0 => {
            let fst_med = data[(even / 2) - 1];
            let snd_med = data[even / 2];

            (fst_med + snd_med) as f64 / 2.0
        }
        odd => data[odd / 2] as f64,
    }
}

#[allow(dead_code)]
fn std_deviation(data: &[u64]) -> f64 {
    match (average(data), data.len()) {
        (data_mean, count) if count > 0 && data_mean >= 0.0 => {
            let variance = data
                .iter()
                .map(|value| {
                    let diff = data_mean - (*value as f64);

                    diff * diff
                })
                .sum::<f64>()
                / count as f64;

            variance.sqrt()
        }
        _ => -1.0,
    }
}
