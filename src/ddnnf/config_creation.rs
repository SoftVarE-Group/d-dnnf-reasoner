use itertools::Itertools;
use rand::seq::SliceRandom;
use rand_distr::{WeightedAliasIndex, Distribution, Binomial};
use rand_pcg::{Pcg32, Lcg64Xsh32};
use rand::{SeedableRng};

use rug::{Assign, Rational, Integer};

use crate::Ddnnf;

use super::node::NodeType::*;

impl Ddnnf {
    /// Creates satisfiable complete configurations for a ddnnf and given assumptions
    /// If the ddnnf on itself or in combination with the assumption is unsatisfiable,
    /// then we can not create any satisfiable configuration and simply return None.
    pub(crate) fn enumerate(&mut self, assumptions: &mut Vec<i32>, _amount: usize) -> Option<Vec<Vec<i32>>> {        
        self.preprocess_config_creation(assumptions);
        
        if self.execute_query(&assumptions) > 0 {
            let mut sample_list = 
                self.enumerate_node(&self.rt(), self.number_of_nodes-1);
            for sample in sample_list.iter_mut() {
                sample.sort_unstable_by_key(|f| f.abs());
            }
            return Some(sample_list);
        }
        None
    }

        /// Generates amount many uniform random samples under a given set of assumptions and a seed.
    /// Each sample is sorted by the number of the features. Each sample is a complete configuration with #SAT of 1.
    /// If the ddnnf itself or in combination with the assumptions is unsatisfiable, None is returned. 
    pub(crate) fn uniform_random_sampling(&mut self, assumptions: &mut Vec<i32>, amount: usize, seed: u64) -> Option<Vec<Vec<i32>>> {
        self.preprocess_config_creation(assumptions);
        
        if self.execute_query(&assumptions) > 0 {
            let mut sample_list = self.sample_node(amount, self.number_of_nodes-1, &mut Pcg32::seed_from_u64(seed));
            for sample in sample_list.iter_mut() {
                sample.sort_unstable_by_key(|f| f.abs());
            }
            return Some(sample_list);
        }
        None
    }

    // resets the temp count of each node to the cached count,
    // computes the count under the assumptions to set some of the temp values,
    // and handle the literals properly.
    fn preprocess_config_creation(&mut self, assumptions: &mut Vec<i32>) {
        for node in self.nodes.iter_mut() {
            node.temp.assign(&node.count);
        }

        for literal in assumptions.iter() {
            match self.literals.get(&-literal) {
                Some(&x) => self.nodes[x].temp.assign(0),
                None => (),
            }
        }
    }

    // Handles a node appropiate depending on its kind to produce complete
    // satisfiable configurations
    fn enumerate_node(&self, amount: &Integer, index: usize) -> Vec<Vec<i32>> {
        let mut enumeration_list = Vec::new();
        if *amount == 0 { return enumeration_list; }
        match &self.nodes[index].ntype {
            And { children } => {
                let mut enumeration_child_lists = Vec::new();
                for &child in children {
                    let child_enum = self.enumerate_node(&self.nodes[child].temp, child);
                    if !child_enum.is_empty() {
                        enumeration_child_lists.push(child_enum);
                    }
                }
                // combine all combinations of children
                // example:
                //      enumeration_child_lists: Vec<Vec<Vec<i32>>>
                //          [[[1,2],[3]],[[4,5,-5]]]
                //      enumeration_list: Vec<Vec<i32>>
                //          [[1,2,4],[1,2,5],[1,2,-5],[3,4],[3,5],[3,-5]]
                enumeration_list = enumeration_child_lists
                    .into_iter()
                    .multi_cartesian_product()
                    .map(|elem| elem.into_iter().flatten().collect())
                    .collect();
            },
            Or { children } => {
                for &child in children {
                    let mut child_enum = self.enumerate_node(&self.nodes[child].temp, child);
                    if !child_enum.is_empty() {
                        enumeration_list.append(&mut child_enum);
                    }
                }
            },
            Literal { literal } => {
                enumeration_list.push(vec![*literal]);
            },
            _ => (),
        }
        enumeration_list
    }

    // Performs the operations needed to generate random samples.
    // The algorithm is based upon KUS's uniform random sampling algorithm.
    fn sample_node(&self, amount: usize, index: usize, rng: &mut Lcg64Xsh32) -> Vec<Vec<i32>> {
        let mut sample_list = Vec::new();
        if amount == 0 { return sample_list; }
        match &self.nodes[index].ntype {
            And { children } => {
                for _ in 0..amount {
                    sample_list.push(Vec::new());
                }
                for &child in children {
                    let mut child_sample_list = self.sample_node(amount, child, rng);
                    // shuffle operation from KUS algorithm
                    child_sample_list.shuffle(rng);
                    
                    // stitch operation
                    for (index, sample) in child_sample_list.iter_mut().enumerate() {
                        sample_list[index].append(sample);
                    }
                }
            },
            Or { children } => {
                let mut pick_amount = vec![0; children.len()];
                let mut choices = Vec::new();
                let mut weights = Vec::new();
                
                // compute the probability of getting a sample of a child node
                let parent_count_as_float = Rational::from((&self.nodes[index].temp, 1));
                for child_index in 0..children.len() {
                    let child_count_as_float = Rational::from((&self.nodes[children[child_index]].temp, 1));
                    
                    // can't get a sample of a children with no more valid configuration
                    if child_count_as_float != 0 {
                        let child_amount = (&parent_count_as_float / child_count_as_float).to_f64() * amount as f64;
                        choices.push(child_index);
                        weights.push(child_amount);
                    }
                }

                // choice some sort of weighted distribution depending on the number of children with count > 0
                match weights.len() {
                    1 => pick_amount[choices[0]] += amount,
                    2 => {
                        let binomial_dist = Binomial::new(amount as u64, weights[0]/(weights[0] + weights[1])).unwrap();
                        pick_amount[choices[0]] += binomial_dist.sample(rng) as usize;
                        pick_amount[choices[1]] = amount - pick_amount[choices[0]];
                    },
                    _ => {
                        let weighted_dist = WeightedAliasIndex::new(weights).unwrap();
                        for _ in 0..amount {
                            pick_amount[choices[weighted_dist.sample(rng)]] += 1;
                        }
                    }
                }

                for &choice in choices.iter() {
                    sample_list.append(&mut self.sample_node(pick_amount[choice], children[choice], rng));
                }

                // add empty lists for child nodes that have a count of zero
                while sample_list.len() != amount {
                    sample_list.push(Vec::new());
                }

                sample_list.shuffle(rng);
            },
            Literal { literal } => {
                for _ in 0..amount {
                    sample_list.push(vec![*literal]);
                } 
            },
            _ => (),
        }
        sample_list
    }
}