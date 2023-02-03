use std::{cmp::min, sync::Mutex, collections::HashMap};

use itertools::Itertools;
use once_cell::sync::Lazy;
use rand::seq::SliceRandom;
use rand_distr::{WeightedAliasIndex, Distribution, Binomial};
use rand_pcg::{Pcg32, Lcg64Xsh32};
use rand::{SeedableRng};

use rug::{Assign, Rational, Integer};

use crate::Ddnnf;

use super::node::NodeType::*;

static ENUMERATION_CACHE: Lazy<Mutex<HashMap<Vec<i32>,usize>>> = Lazy::new(|| Mutex::new(HashMap::new()));

impl Ddnnf {
    /// Creates satisfiable complete configurations for a ddnnf and given assumptions
    /// If the ddnnf on itself or in combination with the assumption is unsatisfiable,
    /// then we can not create any satisfiable configuration and simply return None.
    pub(crate) fn enumerate(&mut self, assumptions: &mut Vec<i32>, amount: usize) -> Option<Vec<Vec<i32>>> { 
        if amount == 0 { return Some(Vec::new()); }
        
        if !self.preprocess_config_creation(assumptions) {
            return None;
        }
        assumptions.sort_unstable_by_key(|f| f.abs());
        
        if self.execute_query(&assumptions) > 0 {
            let last_stop = match ENUMERATION_CACHE.lock().unwrap().get(assumptions) {
                Some(&x) => x,
                None => 0,
            };

            let mut sample_list = 
                self.enumerate_node((&Integer::from(last_stop), &min(self.rt(), Integer::from(last_stop + amount))), self.number_of_nodes-1);
            for sample in sample_list.iter_mut() {
                sample.sort_unstable_by_key(|f| f.abs());
            }

            ENUMERATION_CACHE.lock().unwrap()
                .insert(
                    assumptions.to_vec(),
                    (min(self.rt(), Integer::from(last_stop + amount)) % self.rt()).to_usize_wrapping()
                );
            return Some(sample_list);
        }
        None
    }

    /// Generates amount many uniform random samples under a given set of assumptions and a seed.
    /// Each sample is sorted by the number of the features. Each sample is a complete configuration with #SAT of 1.
    /// If the ddnnf itself or in combination with the assumptions is unsatisfiable, None is returned. 
    pub(crate) fn uniform_random_sampling(&mut self, assumptions: &mut Vec<i32>, amount: usize, seed: u64) -> Option<Vec<Vec<i32>>> {
        if !self.preprocess_config_creation(assumptions) {
            return None;
        }
        
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
    fn preprocess_config_creation(&mut self, assumptions: &Vec<i32>) -> bool {
        // if any of the assumptions isn't valid by being in the range of +-#variables, then we return false
        if assumptions.iter().any(|f| f.abs() > self.number_of_variables as i32) {
            return false;
        }
        
        for node in self.nodes.iter_mut() {
            node.temp.assign(&node.count);
        }

        for literal in assumptions.iter() {
            match self.literals.get(&-literal) {
                Some(&x) => self.nodes[x].temp.assign(0),
                None => (),
            }
        }

        // We can't create a config that contains a true node.
        // Hence, we have to hide the true node by changing its count to 0
        for &index in self.true_nodes.iter() {
            self.nodes[index].temp.assign(0);
        }
        return true;
    }

    // Handles a node appropiate depending on its kind to produce complete
    // satisfiable configurations
    fn enumerate_node(&self, range: (&Integer, &Integer), index: usize) -> Vec<Vec<i32>> {
        let _range2 = (range.0.to_usize_wrapping(), range.1.to_usize_wrapping());
        let mut enumeration_list = Vec::new();
        if *range.1 == 0 || self.nodes[index].temp == 0 { return enumeration_list; }

        match &self.nodes[index].ntype {
            And { children } => {
                let mut acc_amount = Integer::from(1);
                let mut enumeration_child_lists = Vec::new();
                
                for &child in children {
                    // skip the true nodes
                    if self.true_nodes.contains(&child) {
                        continue;
                    }

                    if &acc_amount < range.1 {
                        let change = (&Integer::ZERO, min(range.1, &self.nodes[child].temp));
                        enumeration_child_lists.push(
                            self.enumerate_node(change,child)
                        );
                        acc_amount *= change.1;
                    } else {
                        // restrict the creation of any more configs
                        enumeration_child_lists.push(
                            vec![self.enumerate_node(
                                    (&Integer::ZERO, &Integer::from(1)),
                                    child
                                )[0].clone()
                            ]
                        );
                    }
                }

                // cartesian product of  all combinations of children
                // example:
                //      enumeration_child_lists: Vec<Vec<Vec<i32>>>
                //          [[[1,2,-3],[3]],[[4,5,-5]]]
                //      enumeration_list: Vec<Vec<i32>>
                //          [[1,2,-3,4],[1,2,-3,5],[1,2,-3,-5],[3,4],[3,5],[3,-5]]
                //
                // reverse is important to ensure a total order with additions of configs at the end
                enumeration_child_lists.reverse();
                enumeration_list = enumeration_child_lists
                    .into_iter()
                    .multi_cartesian_product()
                    .map(|elem| elem.into_iter().flatten().collect())
                    .skip(range.0.to_usize_wrapping())
                    .take(range.1.to_usize_wrapping() - range.0.to_usize_wrapping()) // stop after we got our required amount of configs
                    .collect();
            },
            Or { children } => {
                let mut acc_amount = Integer::ZERO;

                for &child in children {
                    if self.nodes[child].temp == Integer::ZERO {
                        continue;
                    }

                    if &acc_amount < range.1 {
                        let change = (&Integer::ZERO, min(range.1, &self.nodes[child].temp));
                        enumeration_list.append(
                            &mut self.enumerate_node(change,child)
                        );
                        acc_amount += change.1;
                    } else {
                        break;
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