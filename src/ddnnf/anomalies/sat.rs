use crate::{Ddnnf, NodeType::*};


impl Ddnnf {
    /// Computes if a node is satisfiable with the marking algorithm:
    /// For each feature of the query, we search for its complementary literal.
    /// Each complementary literal gets annoted similar to the marking algorithm for counting.
    /// For each node, we check whether the node can be satisfiable.
    ///      1) If that node has to be satisfiable, we stop
    ///      2) If that node is unsatisfiable we continue with its parents til we either
    ///          reach the root (declaring the whole query as unsatisfiable) or reach nodes that
    ///         are satisfiable.
    /// If any of those literal propagations reaches the root, the query is unsatisfiable.
    /// Vice versa the query is satisfiable.
    #[inline]
    pub fn sat(&mut self, features: &[i32]) -> bool {
        let mut mark = vec![false; self.nodes.len()];
        self.sat_propagate(features, &mut mark)
    }

    /// Does the exact same as 'sat' with the difference of choosing the marking Vec by ourself.
    /// That allows reusing that vector and therefore enabeling an efficient method to do decision propogation.
    #[inline]
    pub fn sat_propagate(&mut self, features: &[i32], mark: &mut Vec<bool>) -> bool {
        if features.iter().any(|f| self.makes_query_unsat(f)) {
            return false;
        }

        for feature in features {
            match self.literals.get(&-feature) {
                Some(&index) => {
                    self.propagate_mark(index, mark);
                    // if the root is unsatisfiable after any of the literals in the query,
                    // then the whole query must be unsatisfiable too. 
                    if mark[self.nodes.len() - 1] { return false; }
                },
                None => (),
            }
        }
        
        // the result is the marking of the root node
        !mark[self.nodes.len() - 1]
    }

    // marks a node and decides whether we have to continue the marking with its parent nodes
    #[inline]
    fn propagate_mark(&self, index: usize, mark: &mut Vec<bool>) {
        // if the node is already marked, we looked at its path and can stop
        if mark[index] {
            return;
        }

        match &self.nodes[index].ntype {
            // And nodes are always unsatisfiable if any of its children is unsatisfiable
            // (multiplying anything with zero equals always zero)
            Or { children } => {
                // An Or node is only unsatisfiable if all of its children are either marked
                // or have an count of zero (that handle False nodes).
                if !children.iter().all(|&c| mark[c] || self.nodes[c].count == 0) {
                    return;
                }
            }
            // Literals, True, and False nodes are trivial and handled at a higher level
            _ => (),
        }

        mark[index] = true;
        // check the marking for all parents
        self.nodes[index]
            .parents
            .iter()
            .for_each(|&p| self.propagate_mark(p, mark))
    }
}

#[cfg(test)]
mod test {
    use crate::parser::build_ddnnf;

    use super::*;

    #[test]
    fn sat_urs() {
        let mut vp9: Ddnnf =
            build_ddnnf("tests/data/VP9_d4.nnf", Some(42));
        let mut auto1: Ddnnf =
            build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));

        // Uniform random samples produce an SATISFIABLE complete configuration.
        for sample in vp9.uniform_random_sampling(&vec![], 1000, 42).unwrap().iter() {
            assert!(vp9.sat(sample));
        }
        for sample in auto1.uniform_random_sampling(&vec![], 1000, 42).unwrap().iter() {
            assert!(auto1.sat(sample));
        }
    }

    #[test]
    fn sat_card_of_features() {
        let mut vp9: Ddnnf =
            build_ddnnf("tests/data/VP9_d4.nnf", Some(42));
        let mut auto1: Ddnnf =
            build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));

        // If the count is greater than zero, there has to be at least on satisfiable configuration.
        // Vice versa, if the count is equal to zero, the query should be identified as unsatisfiable.
        for i in 1..=vp9.number_of_variables as i32 {
            assert_eq!(vp9.execute_query(&[i]) > 0, vp9.sat(&[i]));
        }
        for i in 1..=auto1.number_of_variables as i32 {
            assert_eq!(auto1.execute_query(&[i]) > 0, auto1.sat(&[i]));
        }
    }
}