use rug::{Integer, Assign, Complete};

use super::super::node::{NodeType::*};
use crate::Ddnnf;

impl Ddnnf {
    #[inline]
    // Computes the cardinality of a node using the marking algorithm:
    // And and Or nodes that base the computation on their child nodes use the
    // .temp value if the child node is marked and the .count value if not
    pub(crate) fn calc_count_marked_node(&mut self, i: usize) {
        match &self.nodes[i].ntype {
            And { children } => {
                self.nodes[i].temp = Integer::product(
                    children.iter().map(
                        |&index| {
                            if self.nodes[index].marker {
                                &self.nodes[index].temp
                            } else {
                                &self.nodes[index].count
                    }
                }))
                .complete()
            }
            Or { children } => {
                self.nodes[i].temp = Integer::sum(children.iter().map(|&index| {
                    if self.nodes[index].marker {
                        &self.nodes[index].temp
                    } else {
                        &self.nodes[index].count
                    }
                }))
                .complete()
            }
            False => self.nodes[i].temp.assign(0),
            _ => self.nodes[i].temp.assign(1) // True and Literal
        }
    }

    #[inline]
    // Computes the cardinality of a feature and partial configurations using the marking algorithm
    fn operate_on_marker(&mut self, indize: &[usize], operation: fn(&mut Ddnnf, usize)) -> (usize, Integer) {
        for i in indize.iter().copied() {
            self.nodes[i].temp.assign(0); // change the value of the node
            self.mark_nodes(i); // go through the path til the root node is marked
        }

        // sort the marked nodes so that we make sure to first calculate the childnodes and then their parents
        self.md.sort_unstable();

        // calc the count for all marked nodes, respectevly all nodes that matter
        for j in 0..self.md.len() {
            if !indize.contains(&self.md[j]) {
                operation(self, self.md[j]);
            }
        }

        // reset everything
        for index in &self.md {
            self.nodes[*index].marker = false
        }
        let marked_nodes = self.md.len();
        self.md.clear();

        // the result is propagated through the whole graph up to the root
        (marked_nodes, self.rt())
    }

    #[inline]
    /// Computes the cardinality of a feature using the marking algorithm.
    /// The marking algorithm differs to the standard variation by only reomputing the
    /// marked nodes. Further, the marked nodes use the .temp value of the childs nodes if they
    /// are also marked and the .count value if they are not.
    pub(crate) fn card_of_feature_with_marker(&mut self, feature: i32) -> (usize, Integer) {
        if self.core.contains(&feature) || self.dead.contains(&-feature) {
            (0, self.rc())
        } else if self.dead.contains(&feature) || self.core.contains(&-feature) {
            (0, Integer::ZERO)
        } else {
            match self.literals.get(&-feature).cloned() {
                Some(i) => self.operate_on_marker(&[i], Ddnnf::calc_count_marked_node),
                // there is no literal corresponding to the feature number and because of that we don't have to do anything besides returning the count of the model
                None => (0, self.rc()),
            }
        }
    }

    #[inline]
    /// Computes the cardinality of a partial configuration using the marking algorithm.
    /// Works analog to the card_of_feature_with_marker()
    pub(crate) fn operate_on_partial_config_marker(
        &mut self,
        features: &[i32],
        operation: fn(&mut Ddnnf, usize)
    ) -> (usize, Integer) {
        if self.query_is_not_sat(features) {
            (0, Integer::ZERO)
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
                return (0, self.rc());
            }

            self.operate_on_marker(&indize, operation)
        }
    }

    #[inline]
    // marks the nodes starting from an initial Literal. All parents and parents of parents til
    // the root nodes get marked
    fn mark_nodes(&mut self, i: usize) {
        self.nodes[i].marker = true;
        self.md.push(i);

        for j in 0..self.nodes[i].parents.len() {
            // check for parent nodes and adjust their count resulting of the changes to their children
            let index = self.nodes[i].parents[j];
            if !self.nodes[index].marker {
                // only mark those nodes which aren't already marked to specificly avoid marking nodes near the root multple times
                self.mark_nodes(index);
            }
        }
    }
}