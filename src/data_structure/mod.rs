use rug::{Assign, Complete, Integer};

use std::{
    collections::{HashMap, HashSet, VecDeque},
    thread,
    sync::{
        Mutex,
        Arc},
    error::Error,
    time::Instant,
    fs::File,
    io::{BufWriter, Write}
};

use workctl::WorkQueue;

use crate::parser::parse_queries_file;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum NodeType{
    And,
    Or,
    Literal,
    True,
    False
}

use NodeType::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Node{
    pub marker: bool,
    pub count: Integer,
    pub temp: Integer,
    pub node_type: NodeType,
    pub children: Option<Vec<usize>>,
    pub parents: Vec<usize>,
    pub var_number: Option<i32>
}

static mut COUNTER: u64 = 0;

impl Node{
    #[inline]
    pub fn new_and(node_type: NodeType, children: Vec<usize>, overall_count: Integer) -> Node{
        Node {
            marker: false,
            count: overall_count,
            temp: Integer::from(0),
            node_type,
            children: Some(children),
            parents: Vec::new(),
            var_number: None,
        }
    }

    #[inline]
    pub fn new_or(node_type: NodeType, var_number: i32, children: Vec<usize>, overall_count: Integer) -> Node{
        Node {
            marker: false,
            count: overall_count,
            temp: Integer::from(0),
            node_type,
            children: Some(children),
            parents: Vec::new(),
            var_number: Some(var_number)
        }
    }

    #[inline]
    pub fn new_literal(var_number: i32) -> Node{
        Node {
            marker: false,
            count: Integer::from(1),
            temp: Integer::from(0),
            node_type: Literal,
            children: None,
            parents: Vec::new(),
            var_number: Some(var_number)
        }
    }

    #[inline]
    pub fn new_bool(node_type: NodeType) -> Node{
        let count: Integer = match node_type{
            True => Integer::from(1),
            False => Integer::new(),
            _ => panic!("NodeType {:?} is not a Bool", node_type)
        };

        Node {
            marker: false,
            count,
            temp: Integer::from(0),
            node_type,
            children: None,
            parents: Vec::new(),
            var_number: None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Ddnnf {
    pub nodes: Vec<Node>,
    pub literals: HashMap<i32, usize>, // <var_number of the Literal, and the corresponding indize>
    pub number_of_variables: u32,
    pub number_of_nodes: usize,
    pub md: Vec<usize>,
    pub core: HashSet<i32>,
    pub dead: HashSet<i32>
}

impl Ddnnf {
    pub fn new(nodes: Vec<Node>, literals: HashMap<i32, usize>, number_of_variables: u32, number_of_nodes: usize) -> Ddnnf{
        let mut ddnnf = Ddnnf{
            nodes,
            literals,
            number_of_variables,
            number_of_nodes,
            md: Vec::new(),
            core: HashSet::new(),
            dead: HashSet::new()
        };
        ddnnf.get_core();
        ddnnf.get_dead();
        ddnnf
    }

    fn get_core(&mut self){
        self.core = (1..self.number_of_variables as i32+1).filter( 
            |f| 
            match self.literals.get(&-f){
                Some(x) => {
                    let par = self.nodes[*x].parents.clone();
                    let mut res = false;
                    for p in par{
                        if self.nodes[p].node_type == And{
                            let chi = self.nodes[p].children.clone();
                            for c in chi.unwrap(){
                                if self.nodes[c].node_type == False{
                                    res = true;
                                }
                            }
                        }
                    }
                    res
                },
                None => true
            }
        ).collect::<HashSet<i32>>()
    }

    fn get_dead(&mut self){
        self.dead = (1..self.number_of_variables as i32+1).filter(
            |f|
            match self.literals.get(f){
                Some(x) => {
                    let par = self.nodes[*x].parents.clone();
                    let mut res = false;
                    for p in par{
                        if self.nodes[p].node_type == And{
                            let chi = self.nodes[p].children.clone();
                            for c in chi.unwrap(){
                                if self.nodes[c].node_type == False{
                                    res = true;
                                }
                            }
                        }
                    }
                    res
                },
                None => true
            }
        ).collect::<HashSet<i32>>()
    }

    fn reduce_query(&mut self, features: &Vec<i32>) -> Vec<i32>{
        features.iter().filter({ // filter keeps the elements which do fulfill the defined boolean formula. Thats why we need to use the ! operator
            |&f| 
            if f > &0{
                // remove included core and excluded dead features
                !self.core.contains(f)
            } else {
                !self.dead.contains(&-f)
            }
        }).map(|f| *f).collect::<Vec<i32>>()
    }

    fn query_is_not_sat(&mut self, features: &Vec<i32>) -> bool{
        // if there is an included dead or an excluded core feature 
        features.iter().any({
            |&f| 
            if f > 0{
                self.dead.contains(&f)
            } else {
                self.core.contains(&-f)
            }
        })
    }

    #[inline]
    fn calc_count(&mut self, i: usize) {
        match self.nodes[i].node_type{
            And => self.nodes[i].temp = Integer::product(
                self.nodes[i].children.as_ref().unwrap().iter().map(|&indize| &self.nodes[indize].temp)
            ).complete(),
            Or => {
                let indizes: &Vec<usize> = self.nodes[i].children.as_ref().unwrap();
                self.nodes[i].temp = Integer::from(&self.nodes[indizes[0]].temp + &self.nodes[indizes[1]].temp)
            },
            False => self.nodes[i].temp.assign(0),
            _ => self.nodes[i].temp.assign(1),
        }
    }

    #[inline]
    pub fn card_of_feature(&mut self, feature: &i32) -> Integer{
        if self.core.contains(feature){
            self.nodes[self.number_of_nodes-1].count.clone()
        } else if self.dead.contains(feature){
            Integer::from(0)
        }else{
            for i in 0..self.number_of_nodes{
                if self.nodes[i].node_type == Literal // search for the node we want to adjust
                    && feature == &-self.nodes[i].var_number.unwrap(){
                    self.nodes[i].temp.assign(0);
                }else{ // all other values stay the same, respectevly get adjusted because the count of the one node got changed
                    self.calc_count(i)
                }
            }
            self.nodes[self.number_of_nodes-1].temp.clone()
        }
    }

    #[inline]
    pub fn card_of_partial_config(&mut self, features: &Vec<i32>) -> Integer{
        if self.query_is_not_sat(features) {
            Integer::from(0)  
        }else{
            let features: Vec<i32> = self.reduce_query(features);
            for i in 0..self.number_of_nodes{
                if self.nodes[i].node_type == Literal && features.contains(&-self.nodes[i].var_number.unwrap()){
                    self.nodes[i].temp.assign(0);
                }else{
                    self.calc_count(i)
                }
            }

            self.nodes[self.number_of_nodes - 1].temp.clone()
        }
    }

    // an recursive version that uses decision nodes to reduce the amount of nodes we have to compute
    #[inline]
    pub fn card_of_partial_config_rec(&mut self, features: &Vec<i32>) -> Integer{
        if self.query_is_not_sat(features) {
            Integer::from(0)  
        }else{
            let features: Vec<i32> = self.reduce_query(features);
            for f in &features{
                match self.literals.get(&-f) {
                    Some(&i) => {
                        self.nodes[i].marker = true;
                        self.nodes[i].temp.assign(0);
                    },
                    None => ()
                }
            }

            self.card_of_partial_config_rec_h(self.number_of_nodes-1, &features);
            /*
            for i in 0..self.number_of_nodes{
                if self.nodes[i].marker{
                    for j in self.nodes[i].parents.clone(){
                        self.nodes[j].marker = true;
                    }
                    self.calc_count_marked_node(i);
                }
            }
            */

            for i in 0..self.number_of_nodes{
                self.nodes[i].marker = false;
            }

            self.nodes[self.number_of_nodes - 1].temp.clone()
        }
    }

    #[inline]
    fn card_of_partial_config_rec_h(&mut self, current: usize, features: &Vec<i32>){
        if !self.nodes[current].marker{
            if self.nodes[current].node_type == And {
                let children = self.nodes[current].children.take().unwrap();
                for &child in children.iter() {
                    self.card_of_partial_config_rec_h(child, features);
                    self.nodes[child].marker = true
                }
                self.nodes[current].children = Some(children);
            } else if self.nodes[current].node_type == Or {
                // Decision Node for a selected feature
                if features.contains(&self.nodes[current].var_number.unwrap()) || features.contains(&-self.nodes[current].var_number.unwrap()){
                    let children = self.nodes[current].children.take().unwrap();
                        
                    self.card_of_partial_config_rec_h(children[0], features);   
                    self.nodes[children[0]].marker = true;

                    self.nodes[current].children = Some(children);
                }
                let children = self.nodes[current].children.take().unwrap();
                for &child in children.iter() {
                    self.card_of_partial_config_rec_h(child, features);
                    self.nodes[child].marker = true
                }
                self.nodes[current].children = Some(children);
            }
            self.calc_count(current);
            self.nodes[current].marker = true
        }
    }

    #[inline]
    fn calc_count_marked_node(&mut self, i: usize) {
        match self.nodes[i].node_type{
            And => self.nodes[i].temp = Integer::product(self.nodes[i].children.as_ref().unwrap().iter()
                .map(|&indize| {
                        if self.nodes[indize].marker {
                            &self.nodes[indize].temp
                        }else{
                            &self.nodes[indize].count
                        }
                    }
                )).complete(),
            Or => self.nodes[i].temp = Integer::sum(self.nodes[i].children.as_ref().unwrap().iter()
                .map(|&indize| 
                    if self.nodes[indize].marker {
                        &self.nodes[indize].temp
                    }else{
                        &self.nodes[indize].count
                    }
                )).complete(),
            False => self.nodes[i].temp.assign(0),
            _ => self.nodes[i].temp.assign(1),
        }
    }

    #[inline]
    fn calc_count_marker(&mut self, indize: Vec<usize>) -> Integer{
        for i in indize.clone(){
            self.nodes[i].temp.assign(0); // change the value of the node
            self.mark_nodes(i); // go through the path til the root node is marked
        }

        // sort the marked nodes so that we make sure to first calculate the childnodes and then their parents
        self.md.sort_unstable();

        // calc the count for all marked nodes, respectevly all nodes that matter
        for j in 0..self.md.len(){
            if !indize.contains(&self.md[j]){
                self.calc_count_marked_node(self.md[j]);
            }
        }

        // reset everything
        for j in 0..self.md.len(){
            self.nodes[self.md[j]].marker = false
        }
        unsafe{COUNTER = self.md.len() as u64}
        self.md.clear();

        // the result is propagated through the whole graph up to the root
        self.nodes[self.number_of_nodes - 1].temp.clone()
    }

    #[inline]
    pub fn card_of_feature_with_marker(&mut self, feature: &i32) -> Integer{
        if self.core.contains(feature) || self.dead.contains(&-feature){
            self.nodes[self.number_of_nodes-1].count.clone()
        } else if self.dead.contains(feature) || self.core.contains(&-feature){
            Integer::from(0)
        }else{
            match self.literals.get(&-feature).cloned(){
                Some(i) => self.calc_count_marker(vec![i]),
                // there is no literal corresponding to the feature number and because of that we don't have to do anything besides returning the count of the model
                None => self.nodes[self.number_of_nodes - 1].count.clone()
            }
        }   
    }

    #[inline]
    pub fn card_of_partial_config_with_marker(&mut self, features: &Vec<i32>) -> Integer{
        if self.query_is_not_sat(features) {
            Integer::from(0)  
        }else{
            let features: Vec<i32> = self.reduce_query(features);

            let mut indize: Vec<usize> = Vec::new();

            for number in features{
                // we set the negative occurences to 0 for the features we want to include
                if let Some(i) = self.literals.get(&-number).cloned() { indize.push(i) }
            }

            self.calc_count_marker(indize)
        }
    }

    #[inline]
    fn mark_nodes(&mut self, i: usize) {
        self.nodes[i].marker = true;
        self.md.push(i);

        for j in self.nodes[i].parents.clone() { // check for parent nodes and adjust their count resulting of the changes to their children
            if !self.nodes[j].marker{ // only mark those nodes which aren't already marked to specificly avoid marking nodes near the root multple times
                self.mark_nodes(j);
            }
        }
    }

    pub fn execute_query(&mut self, features: Vec<i32>){
        if features.len() == 1{
            let f = &features.clone().pop().unwrap();
            let time = Instant::now();
            let res: Integer = self.card_of_feature_with_marker(f);
            let elapsed_time = time.elapsed().as_secs_f32();
            let count = get_counter();
        
            println!("Feature count for feature number {:?}: {:#?}", f, res);
            println!("{:?} nodes were marked (note that nodes which are on multiple paths are only marked once). That are ≈{:.2}% of the ({}) total nodes",
                count, count as f64/self.number_of_nodes as f64 * 100.0, self.number_of_nodes);
            println!("Elapsed time for one feature: {:.6} seconds.\n", elapsed_time);
        }else if features.len() <= 10{
            let time = Instant::now();
            let res: Integer = self.card_of_partial_config_with_marker(&features);
            let elapsed_time = time.elapsed().as_secs_f32();

            println!("The cardinality for the partial configuration {:?} is: {:#?}", features, res);
            println!("{:?} nodes were marked (note that nodes which are on multiple paths are only marked once). That are ≈{:.2}% of the ({}) total nodes",
                get_counter(), get_counter() as f64/self.number_of_nodes as f64 * 100.0, self.number_of_nodes);
            println!("Elapsed time for a partial configuration in seconds: {:.6}s.", elapsed_time);
        } else{
            let time = Instant::now();
            let res: Integer = self.card_of_partial_config(&features);
            let elapsed_time = time.elapsed().as_secs_f32();

            println!("The cardinality for the partial configuration {:?} is: {:#?}", features, res);
            println!("Elapsed time for a partial configuration in seconds: {:.6}s.", elapsed_time);
        }
    }
}

pub fn get_counter() -> u64{
    unsafe{COUNTER}
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

const MAX_WORKER: u16 = 4;

impl Ddnnf{
    #[inline]
    pub fn card_of_each_feature_to_csv(&mut self, file_path: &str) -> Result<(), Box<dyn Error>>{
        // initialize WorkQueue
        let mut queue: WorkQueue<i32> = WorkQueue::with_capacity(self.number_of_variables as usize);
        
        // Create a MPSC (Multiple Producer, Single Consumer) channel. Every worker
        // is a producer, the main thread is a consumer; the producers put their
        // work into the channel when it's done.
        use std::sync::mpsc::channel;
        let (results_tx, results_rx) = channel();

        // This Vec will hold thread join handles to allow us to not exit while work
        // is still being done. These handles provide a .join() method which blocks
        // the current thread until the thread referred to by the handle exits.
        let mut threads = Vec::new();

        for e in 1..&self.number_of_variables+1 {
            queue.push_work(e as i32); // x..y are all numbers between x and y including x, excluding y
        }

        for _ in 0..MAX_WORKER {
            // Get a reference to the queue for the thread to use
            // .clone() here doesn't clone the actual queue data, but rather the
            // internal Arc produces a new reference for use in the new queue 
            // instance.
            let mut t_queue = queue.clone();

            // Similarly, create a new transmitter for the thread to use
            let t_results_tx = results_tx.clone();

            // get a clone of the ddnnf
            let mut ddnnf: Ddnnf = self.clone();

            // thread::spawn takes a closure (an anonymous function that "closes"
            // over its environment). The move keyword means it takes ownership of
            // those variables, meaning they can't be used again in the main thread.
            let handle = thread::spawn(move || {
                // Loop while there's expected to be work, looking for work.
                // If work is available, do that work.
                while let Some(work) = t_queue.pull_work() {
                    // Do some work.
                    let result = ddnnf.card_of_feature_with_marker(&work);

                    // Send the work and the result of that work.
                    //
                    // Sending could fail. If so, there's no use in
                    // doing any more work, so abort.
                    match t_results_tx.send((work, result)) {
                        Ok(_) => (),
                        Err(_) => { break; },
                    }

                }

                // Signal to the operating system that now is a good time
                // to give another thread a chance to run.
                //
                // This isn't strictly necessary - the OS can preemptively 
                // switch between threads, without asking - but it helps make
                // sure that other threads do get a chance to get some work.
                std::thread::yield_now();
            });

            // Add the handle for the newly spawned thread to the list of handles
            threads.push(handle);
        }

        let mut results: Vec<(i32, String)> = Vec::with_capacity(self.number_of_variables as usize);

        // Get completed work from the channel while there's work to be done.
        for _ in 0..self.number_of_variables {
            match results_rx.recv() {
                // If the control thread successfully receives, a job was completed.
                Ok((feature, cardinality)) => {
                    // Caching the results
                    results.push((feature, cardinality.to_string()));
                },
                // If the control thread is the one left standing, that's pretty
                // problematic.
                Err(_) => {panic!("All workers died unexpectedly.");}
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
            wtr.write_record(vec![element.0.to_string(), element.1])?;
        }

        // Flush everything into the file that is still in a buffer
        // Now we finished writing the csv file
        wtr.flush()?;

        // If everything worked as expected, then we can return Ok(()) and we are happy :D
        Ok(())
    }

    pub fn card_multi_queries(&mut self, path_in: &str, path_out: &str) -> Result<(), Box<dyn Error>>{
        let work: Vec<Vec<i32>> = parse_queries_file(path_in);
        
        // Create a new work queue to keep track of what work needs to be done.
        // Note that the queue is internally mutable (or, rather, the Mutex is),
        // but this binding doesn't need to be mutable. This isn't unsound because
        // the Mutex ensures at runtime that no two references can be used;
        // therefore no mutation can occur at the same time as aliasing.
        let mut queue = WorkQueue::with_capacity(work.len());

        for e in work.clone() {
            queue.push_work(e);
        }

        // Create a MPSC (Multiple Producer, Single Consumer) channel. Every worker
        // is a producer, the main thread is a consumer; the producers put their
        // work into the channel when it's done.
        use std::sync::mpsc::channel;
        let (results_tx, results_rx) = channel();

        // This Vec will hold thread join handles to allow us to not exit while work
        // is still being done. These handles provide a .join() method which blocks
        // the current thread until the thread referred to by the handle exits.
        let mut threads = Vec::new();

        for _ in 0..MAX_WORKER {
            // Get a reference to the queue for the thread to use
            // .clone() here doesn't clone the actual queue data, but rather the
            // internal Arc produces a new reference for use in the new queue 
            // instance.
            let mut t_queue = queue.clone();

            // Similarly, create a new transmitter for the thread to use
            let t_results_tx = results_tx.clone();

            // get a clone of the ddnnf
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
                    if work.len() <= 10 {
                        result = ddnnf.card_of_partial_config_with_marker(&work);
                    }else{
                        result = ddnnf.card_of_partial_config(&work);
                    }

                    // Send the work and the result of that work.
                    //
                    // Sending could fail. If so, there's no use in
                    // doing any more work, so abort.
                    match t_results_tx.send((work, result)) {
                        Ok(_) => (),
                        Err(_) => { break; },
                    }

                }

                // Signal to the operating system that now is a good time
                // to give another thread a chance to run.
                //
                // This isn't strictly necessary - the OS can preemptively 
                // switch between threads, without asking - but it helps make
                // sure that other threads do get a chance to get some work.
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
                    let mut features_str = features.iter().fold(String::new(), |acc, &num| acc + &num.to_string() + " ");
                    features_str.pop();
                    let data = &format!("{};{}\n", features_str, cardinality);
                    wtr.write(data.as_bytes())?;
                },
                // If the control thread is the one left standing, that's pretty
                // problematic.
                Err(_) => {panic!("All workers died unexpectedly.");}
            }
        }

        // If we didn't do that, the workers would just look for work forever.
        // This is useful because many applications of this technique don't 
        // have a defined stopping point that is known in advance - that is,
        // they will have to perform a lot of work that isn't known at the time
        // the work queue is created.
        //
        // A SyncFlag can be used so that when the main thread encounters a
        // kill condition (e.g. Ctrl+C, or perhaps a fatal error of some kind),
        // it can gracefully shut down all of those workers at once.


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

impl Ddnnf{
    // (count of total nodes)
    pub fn get_child_number(&mut self) -> (u64, u64){
        let mut total_child_counter: u64 = 0;
        let mut and_child_counter: u64 = 0;

        for i in 0..self.nodes.len(){
            match &self.nodes[i].children{
                Some(x) => {
                    total_child_counter += x.len() as u64;
                    if self.nodes[i].node_type == And{
                        and_child_counter += x.len() as u64;
                    }
                },
                None => continue,
            }
        }
        (total_child_counter, and_child_counter)
    }

    // (number of and nodes, or, positive literal, negative literal, true, false)
    pub fn get_nodetype_numbers(&mut self) -> (u64, u64, u64, u64, u64){
        let mut and_counter = 0;
        let mut or_counter = 0;
        let mut literal_counter = 0;
        let mut true_counter = 0;
        let mut false_counter = 0;

        for i in 0..self.nodes.len(){
            match &self.nodes[i].node_type{
                And => and_counter += 1,
                Or => or_counter += 1,
                Literal => literal_counter += 1,
                True => true_counter += 1,
                False => false_counter += 1
            }
        }

        (and_counter, or_counter, literal_counter, true_counter, false_counter)
    }

    // the standard deviation (s_x) is defined as sqrt((1/n) * sum over (length of a path - length of the mean path)² for each path)
    // (lowest, highest, mean, s_x, #paths)
    #[inline]
    pub fn get_depths(&mut self) -> (u64, u64, f64, f64, u64){
        let mut lowest: u64 = u64::MAX;
        let mut highest: u64 = 0;
        let mut mean: f64 = 0.0;

        let depths: Vec<u64> = get_depth(&self.nodes, self.nodes.len()-1, 0);
        let length: u64 = depths.len() as u64;

        for depth in depths.clone(){
            if depth > highest{
                highest =  depth;
            }
            if depth < lowest{
                lowest = depth;
            }
            mean += depth as f64;
        }
        mean /= length as f64;

        let mut derivation: f64 = 0.0;

        for depth in depths{
            derivation += (depth as f64 - mean).powi(2);
        }

        let s_x: f64 = (derivation/length as f64).sqrt();

        (lowest, highest, mean, s_x, length)
    }
}

#[inline]
fn get_depth(nodes: &[Node], indize: usize, count: u64) -> Vec<u64>{
    let current: &Node = &nodes[indize];

    if current.node_type != And && current.node_type != Or{
        vec![count]
    }else{
        match &current.children{
            Some(children) => {
                let mut child_depths: Vec<Vec<u64>> = Vec::new();
                for i in children{
                    child_depths.push(get_depth(nodes, *i, count+1));
                }
                child_depths.into_iter().flatten().collect()
            },
            None => panic!("An inner node does not have child nodes. That should not happen!")
        }
    }
}