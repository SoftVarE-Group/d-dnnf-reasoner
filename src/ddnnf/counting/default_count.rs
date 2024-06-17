use super::super::node::NodeType::*;
use crate::Ddnnf;
use num::{BigInt, One, Zero};

impl Ddnnf {
    #[inline]
    // Computes the cardinality of a node according to the rules mentioned at the Nodetypes and saves it in tmp
    pub(crate) fn calc_count(&mut self, i: usize) {
        match &self.nodes[i].ntype {
            And { children } => {
                self.nodes[i].temp = children
                    .iter()
                    .map(|&indice| &self.nodes[indice].temp)
                    .product()
            }
            Or { children } => {
                self.nodes[i].temp = children
                    .iter()
                    .map(|&indice| &self.nodes[indice].temp)
                    .sum()
            }
            False => self.nodes[i].temp.set_zero(),
            _ => self.nodes[i].temp.set_one(), // True and Literal
        }
    }

    #[inline]
    /// Computes the cardinality of a feature according
    /// to the rules mentioned at the Nodetypes and returns the result.
    /// The function does not use the marking approach.
    /// This function trys to apply optimisations based on core and dead features.
    fn _operate_on_single_feature(
        &mut self,
        feature: i32,
        operation: fn(&mut Ddnnf, usize),
    ) -> BigInt {
        if self.has_no_effect_on_query(&feature) {
            self.rc()
        } else if self.makes_query_unsat(&feature) {
            BigInt::ZERO
        } else {
            for i in 0..self.nodes.len() {
                match &mut self.nodes[i].ntype {
                    // search for the node we want to adjust
                    Literal { literal } => {
                        if feature == -*literal {
                            self.nodes[i].temp.set_zero();
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
    pub(crate) fn operate_on_partial_config_default(
        &mut self,
        features: &[i32],
        operation: fn(&mut Ddnnf, usize),
    ) -> BigInt {
        if self.query_is_not_sat(features) {
            BigInt::ZERO
        } else {
            let features: Vec<i32> = self.reduce_query(features);
            for i in 0..self.nodes.len() {
                match &self.nodes[i].ntype {
                    // search for the nodes we want to adjust
                    Literal { literal } => {
                        if features.contains(&-literal) {
                            self.nodes[i].temp.set_zero();
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

#[cfg(test)]
mod test {
    use crate::parser::build_ddnnf;

    use super::*;

    #[test]
    fn operate_on_single_feature() {
        let mut vp9: Ddnnf = build_ddnnf("tests/data/VP9_d4.nnf", Some(42));
        let mut auto1: Ddnnf = build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));

        for i in 1..=vp9.number_of_variables as i32 {
            assert_eq!(
                vp9.card_of_feature_with_marker(i),
                vp9._operate_on_single_feature(i, Ddnnf::calc_count)
            );
            assert_eq!(
                vp9.card_of_feature_with_marker(-i),
                vp9._operate_on_single_feature(-i, Ddnnf::calc_count)
            );
        }

        for i in (1..=auto1.number_of_variables as i32).step_by(100) {
            assert_eq!(
                auto1.card_of_feature_with_marker(i),
                auto1._operate_on_single_feature(i, Ddnnf::calc_count)
            );
            assert_eq!(
                auto1.card_of_feature_with_marker(-i),
                auto1._operate_on_single_feature(-i, Ddnnf::calc_count)
            );
        }
    }
}
