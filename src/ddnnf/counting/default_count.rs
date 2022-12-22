use rug::{Integer, Assign, Complete};

use super::super::node::{NodeType::*};
use crate::Ddnnf;

impl Ddnnf {
    #[inline]
    // Computes the cardinality of a node according to the rules mentioned at the Nodetypes and saves it in tmp
    pub(crate) fn calc_count(&mut self, i: usize) {
        match &self.nodes[i].ntype {
            And { children } => {
                self.nodes[i].temp = Integer::product(
                        children
                        .iter()
                        .map(|&indice| &self.nodes[indice].temp),
                )
                .complete()
            }
            Or { children } => {
                self.nodes[i].temp =
                    Integer::sum(children
                        .iter()
                        .map(|&indice| &self.nodes[indice].temp))
                        .complete()
            }
            False => self.nodes[i].temp.assign(0),
            _ => self.nodes[i].temp.assign(1), // True and Literal
        }
    }

    #[inline]
    /// Computes the cardinality of a feature according
    /// to the rules mentioned at the Nodetypes and returns the result.
    /// The function does not use the marking approach.
    /// This function trys to apply optimisations based on core and dead features.
    fn _operate_on_single_feature(&mut self, feature: i32, operation: fn(&mut Ddnnf, usize)) -> Integer {
        if self.core.contains(&feature) {
            self.rc()
        } else if self.dead.contains(&feature) {
            Integer::ZERO
        } else {
            for i in 0..self.number_of_nodes {
                match &mut self.nodes[i].ntype {
                    // search for the node we want to adjust
                    Literal { literal } => {
                        if feature == -*literal {
                            self.nodes[i].temp.assign(0);
                        } else {
                            operation(self, i)   
                        }
                    }
                    // all other values stay the same, respectevly get adjusted because the count of the one node got changed
                    _ => operation(self, i),
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
    pub(crate) fn operate_on_partial_config_default(&mut self, features: &[i32], operation: fn(&mut Ddnnf, usize)) -> Integer {
        if self.query_is_not_sat(features) {
            Integer::ZERO
        } else {
            let features: Vec<i32> = self.reduce_query(features);
            for i in 0..self.number_of_nodes {
                match &self.nodes[i].ntype {
                    // search for the nodes we want to adjust
                    Literal { literal } => {
                        if features.contains(&-literal) {
                            self.nodes[i].temp.assign(0);
                        } else {
                            operation(self, i)   
                        }
                    }
                    _ => operation(self, i),
                }
            }
            self.rt()
        }
    }
}