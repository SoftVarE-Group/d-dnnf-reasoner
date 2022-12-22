use rug::{Integer, Assign};

use super::super::node::{NodeType::*};
use crate::Ddnnf;

impl Ddnnf {
    #[inline]
    // Computes if a node is satisfiable with the marking algorithm:
    // Here, we use the number representation of 1 for true and 0 for false.
    pub(crate) fn sat_marked_node(&mut self, i: usize) {
        match &self.nodes[i].ntype {
            And { children } => {
                // if any child count is 0 => And has to be 0 too, because we multiply all children
                self.nodes[i].temp = 
                    if children.iter().any(|&index| {
                            if self.nodes[index].marker {
                                self.nodes[index].temp == Integer::ZERO
                            } else {
                                self.nodes[index].count == Integer::ZERO
                            }}) {
                        Integer::ZERO 
                    } else {
                        Integer::from(1)
                    };
            },
            Or { children } => {
                // if any child count is > 0 => Or has to be > 0 too, because we add all children
                self.nodes[i].temp = 
                    if children.iter().any(|&index| {
                        if self.nodes[index].marker {
                            self.nodes[index].temp > Integer::ZERO
                        } else {
                            self.nodes[index].count > Integer::ZERO
                        }}) {
                    Integer::from(1)
                } else {
                    Integer::ZERO
                };
            }
            False => self.nodes[i].temp.assign(0),
            _ => self.nodes[i].temp.assign(1) // True and Literal
        }
    }

    #[inline]
    // CComputes if a node is sat
    pub(crate) fn sat_node_default(&mut self, i: usize) {
        match &self.nodes[i].ntype {
            And { children } => {
                // if any child count is 0 => And has to be 0 too, because we multiply all children
                self.nodes[i].temp = if children
                    .iter()
                    .any(|&indice| self.nodes[indice].temp == Integer::ZERO) {
                    Integer::ZERO
                } else {
                    Integer::from(1)
                }
            },
            Or { children } => {
                // if any child count is > 0 => Or has to be > 0 too, because we add all children
                self.nodes[i].temp = if children
                    .iter()
                    .any(|&indice| self.nodes[indice].temp > Integer::ZERO) {
                    Integer::from(1)
                } else {
                    Integer::ZERO
                };
            }
            False => self.nodes[i].temp.assign(0),
            _ => self.nodes[i].temp.assign(1), // True and Literal
        }
    }
}