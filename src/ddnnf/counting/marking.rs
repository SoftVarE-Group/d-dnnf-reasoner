use rug::{Integer, Assign, Complete};

use super::super::node::{NodeType::*};
use crate::Ddnnf;

impl Ddnnf {

    // TODO tests/*.rs Dateien in src umlagern
        // CI anpassen => nur noch ein job für alle tests
    // TODO test schreiben, doku schreiben


    /* alte version
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
    */

    /* // TODO in calc_count_node einbauen, wenn für alle queries performant
        self.nodes[i].temp = children.iter()
        .fold(self.nodes[i].count.clone(), |mut acc, &index| {
            if self.nodes[index].marker {
                if &self.nodes[index].count != &0 {
                    acc /= &self.nodes[index].count;
                }
                acc *= &self.nodes[index].temp;
            }
            acc
        });
    */

    #[inline]
    // Computes the cardinality of a node using the marking algorithm:
    // And and Or nodes that base the computation on their child nodes use the
    // .temp value if the child node is marked and the .count value if not
    pub(crate) fn calc_count_marked_node(&mut self, i: usize) {
        match &self.nodes[i].ntype {
            And { children } => {
                self.nodes[i].temp = children.iter()
                .fold(self.nodes[i].count.clone(), |mut acc, &index| {
                    let node = &self.nodes[index];
                    if node.marker {
                        if node.count != 0 {
                            acc /= &node.count;
                        }
                        acc *= &node.temp;
                    }
                    acc
                });
            }
            Or { children } => {
                self.nodes[i].temp = Integer::sum(children.iter().map(|&index| {
                    let node = &self.nodes[index];
                    if node.marker {
                        &node.temp
                    } else {
                        &node.count
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
    fn operate_on_marker(&mut self, indexes: &[usize], operation: fn(&mut Ddnnf, usize)) -> (usize, Integer) {
        for index in indexes.iter().copied() {
            self.nodes[index].temp.assign(0); // change the value of the node
            self.mark_nodes(index); // go through the path til the root node is marked
        }

        // sort the marked nodes so that we make sure to first calculate the childnodes and then their parents
        self.md.sort_unstable();

        // calc the count for all marked nodes, respectevly all nodes that matter
        for j in 0..self.md.len() {
            if !indexes.contains(&self.md[j]) {
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

            let mut indexes: Vec<usize> = Vec::new();

            for number in features {
                // we set the negative occurences to 0 for the features we want to include
                if let Some(i) = self.literals.get(&-number).cloned() {
                    indexes.push(i)
                }
            }

            if indexes.is_empty() {
                return (0, self.rc());
            }

            self.operate_on_marker(&indexes, operation)
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