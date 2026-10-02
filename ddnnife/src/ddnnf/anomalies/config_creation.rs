use crate::BigUrational;
use crate::Ddnnf;
use crate::NodeType;
use crate::ddnnf::counting::default_count::Counts;
use itertools::Itertools;
use log::warn;
use num::{BigUint, ToPrimitive, Zero};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_distr::{Binomial, Distribution, weighted::WeightedAliasIndex};
use rand_pcg::{Lcg64Xsh32, Pcg32};
use std::range::Range;

impl Ddnnf {
    /// Creates satisfiable complete configurations for a d-DNNF and given assumptions.
    ///
    /// Returns `None` if the d-DNNF itself or with the assumptions represents a tautology or
    /// contradiction.
    ///
    /// # Note
    ///
    /// `offset` can cause the enumeration to wrap around, i.e. when it becomes larger than
    /// the possible number of enumerations, it will become `offset % n` where `n` is the number of possible
    /// enumerations.
    ///
    /// Additionally `offset + amount` can be larger than the possible number of enumeration.
    /// In this case the enumeration will stop after the last configuration.
    ///
    /// # Panics
    ///
    /// Panics when `amount + offset` is outside the range of [usize].
    pub fn enumerate(
        &self,
        assumptions: &[i32],
        amount: usize,
        offset: usize,
    ) -> Option<Vec<Vec<i32>>> {
        if self.is_trivial() || !self.check_assumptions(assumptions) {
            return None;
        }

        if amount == 0 {
            return Some(Vec::new());
        }

        let (root_count, counts) =
            self.operate_on_partial_config_default_external(assumptions, Self::calc_count_external);

        if root_count.is_zero() {
            return None;
        }

        let mut offset = offset;

        if BigUint::from(offset) >= root_count {
            warn!(
                "`offset` is larger than the possible number of enumerations: {offset} >= {root_count}, wrapping around."
            );

            offset %= &root_count;
        }

        let root_index = self.nodes.len() - 1;

        let mut samples = self.enumerate_node(
            Range::from(offset..min_mixed(offset + amount, &root_count)),
            root_index,
            &counts,
        );

        samples
            .iter_mut()
            .for_each(|sample| sample.sort_unstable_by_key(|literal| literal.abs()));

        Some(samples)
    }

    /// Generates amount many uniform random samples under a given set of assumptions and a seed.
    /// Each sample is sorted by the number of the features. Each sample is a complete configuration
    /// with #SAT of 1.
    ///
    /// Returns `None` if the d-DNNF itself or with the assumptions represents a tautology or
    /// contradiction.
    pub fn uniform_random_sampling(
        &self,
        assumptions: &[i32],
        amount: usize,
        seed: u64,
    ) -> Option<Vec<Vec<i32>>> {
        if self.is_trivial() || !self.check_assumptions(assumptions) {
            return None;
        }

        let (root_count, counts) =
            self.operate_on_partial_config_default_external(assumptions, Self::calc_count_external);

        if root_count.is_zero() {
            return None;
        }

        let mut sample_list = self.sample_node(
            amount,
            self.nodes.len() - 1,
            &mut Pcg32::seed_from_u64(seed),
            &counts,
        );

        for sample in sample_list.iter_mut() {
            sample.sort_unstable_by_key(|f| f.abs());
        }

        Some(sample_list)
    }

    /// Checks whether the given assumptions are valid.
    ///
    /// Assumptions are invalid if they contain a literal out of the variable bounds.
    fn check_assumptions(&self, assumptions: &[i32]) -> bool {
        !assumptions
            .iter()
            .map(|literal| literal.unsigned_abs())
            .any(|variable| variable > self.number_of_variables)
    }

    fn enumerate_node(
        &self,
        range: Range<usize>,
        node_index: usize,
        counts: &Counts,
    ) -> Vec<Vec<i32>> {
        if range.is_empty() || counts[node_index].is_zero() {
            return Vec::new();
        }

        match &self.nodes[node_index].ntype {
            NodeType::And { children } => {
                let mut enumerations_count = 1;

                let enumeration_child_lists: Vec<Vec<Vec<i32>>> = children
                    .iter()
                    .map(|&child_index| {
                        if enumerations_count >= range.end {
                            // restrict the creation of any more configs
                            return vec![
                                self.enumerate_node(Range::from(0..1), child_index, counts)[0]
                                    .clone(),
                            ];
                        }

                        // Range for this child enumeration.
                        // Enumerate not more than either we want globally or the child can provide.
                        let child_range =
                            Range::from(0..min_mixed(range.end, &counts[child_index]));

                        enumerations_count *= child_range.end;

                        self.enumerate_node(child_range, child_index, counts)
                    })
                    .collect();

                // cartesian product of  all combinations of children
                // example:
                //      enumeration_child_lists: Vec<Vec<Vec<i32>>>
                //          [[[1,2,-3],[3]],[[4,5,-5]]]
                //      enumeration_list: Vec<Vec<i32>>
                //          [[1,2,-3,4],[1,2,-3,5],[1,2,-3,-5],[3,4],[3,5],[3,-5]]
                //
                // reverse is important to ensure a total order with additions of configs at the end
                enumeration_child_lists
                    .into_iter()
                    .rev()
                    .multi_cartesian_product()
                    .map(|product| product.concat())
                    .skip(range.start)
                    .take(range.end - range.start)
                    .collect()
            }
            NodeType::Or { children } => {
                // Generated enumerations of the child nodes.
                let mut enumerations = Vec::new();

                // The total count of all generated configurations.
                let mut enumerations_count = 0;

                for &child_index in children
                    .iter()
                    .filter(|&&child_index| !counts[child_index].is_zero())
                {
                    if enumerations_count >= range.end {
                        break;
                    }

                    // Range for this child enumeration.
                    // Enumerate not more than either we want globally or the child can provide.
                    let child_range = Range::from(0..min_mixed(range.end, &counts[child_index]));
                    enumerations_count += child_range.end;

                    enumerations.append(&mut self.enumerate_node(child_range, child_index, counts));
                }

                enumerations
            }
            NodeType::Literal { literal } => vec![vec![*literal]],
        }
    }

    // Performs the operations needed to generate random samples.
    // The algorithm is based upon KUS's uniform random sampling algorithm.
    fn sample_node(
        &self,
        amount: usize,
        index: usize,
        rng: &mut Lcg64Xsh32,
        counts: &Counts,
    ) -> Vec<Vec<i32>> {
        let mut sample_list = Vec::new();
        if amount == 0 {
            return sample_list;
        }
        match &self.nodes[index].ntype {
            NodeType::And { children } => {
                for _ in 0..amount {
                    sample_list.push(Vec::new());
                }
                for &child in children {
                    let mut child_sample_list = self.sample_node(amount, child, rng, counts);
                    // shuffle operation from KUS algorithm
                    child_sample_list.shuffle(rng);

                    // stitch operation
                    for (index, sample) in child_sample_list.iter_mut().enumerate() {
                        sample_list[index].append(sample);
                    }
                }
            }
            NodeType::Or { children } => {
                let mut pick_amount = vec![0; children.len()];
                let mut choices = Vec::new();
                let mut weights = Vec::new();

                // compute the probability of getting a sample of a child node
                let parent_count_as_float = BigUrational::from(counts[index].clone());
                #[allow(clippy::needless_range_loop)]
                for child_index in 0..children.len() {
                    let child_count_as_float =
                        BigUrational::from(counts[children[child_index]].clone());

                    // can't get a sample of a children with no more valid configuration
                    if !child_count_as_float.is_zero() {
                        let child_amount = (child_count_as_float / &parent_count_as_float)
                            .to_f64()
                            .expect("Failed to convert BigUrational to f64!")
                            * amount as f64;
                        choices.push(child_index);
                        weights.push(child_amount);
                    }
                }

                // choice some sort of weighted distribution depending on the number of children with count > 0
                match weights.len() {
                    1 => pick_amount[choices[0]] += amount,
                    2 => {
                        let binomial_dist =
                            Binomial::new(amount as u64, weights[0] / (weights[0] + weights[1]))
                                .unwrap();
                        pick_amount[choices[0]] += binomial_dist.sample(rng) as usize;
                        pick_amount[choices[1]] = amount - pick_amount[choices[0]];
                    }
                    _ => {
                        let weighted_dist = WeightedAliasIndex::new(weights).unwrap();
                        for _ in 0..amount {
                            pick_amount[choices[weighted_dist.sample(rng)]] += 1;
                        }
                    }
                }

                for &choice in choices.iter() {
                    sample_list.append(&mut self.sample_node(
                        pick_amount[choice],
                        children[choice],
                        rng,
                        counts,
                    ));
                }

                // add empty lists for child nodes that have a count of zero
                while sample_list.len() != amount {
                    sample_list.push(Vec::new());
                }

                sample_list.shuffle(rng);
            }
            NodeType::Literal { literal } => {
                for _ in 0..amount {
                    sample_list.push(vec![*literal]);
                }
            }
        }
        sample_list
    }
}

/// Returns the smaller value of a [usize] and a [BigUint].
///
/// This should never fail as the [BigUint] must be smaller than the largest possible [usize]
/// in case it is smaller than the provided [usize].
fn min_mixed(a: usize, b: &BigUint) -> usize {
    if &BigUint::from(a) < b {
        return a;
    }

    b.to_usize()
        .expect("Failed to convert arbitrary length integer")
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::rng;
    use std::cmp::min;
    use std::collections::HashSet;
    use std::path::Path;

    #[test]
    fn enumeration_small_ddnnf() {
        let mut vp9: Ddnnf = Ddnnf::from_file(Path::new("tests/data/VP9_d4.nnf"), Some(42));

        let mut res_all = HashSet::new();
        let mut res_assumptions = HashSet::new();

        let assumptions = [
            1, 2, 3, -4, -5, 6, 7, -8, -9, 10, 11, -12, -13, -14, 15, 16, -17, -18, 19, 20, 27,
        ];
        let inter_res_assumptions_1 = vp9.enumerate(&assumptions, 40, 0).unwrap();
        for inter in inter_res_assumptions_1 {
            assert!(vp9.sat(&inter));
            assert_eq!(
                vp9.number_of_variables as usize,
                inter.len(),
                "we got only a partial config"
            );
            res_assumptions.insert(inter);
        }
        assert_eq!(
            40,
            res_assumptions.len(),
            "we did not get as many configs as we requested"
        );

        let amount = 50000;
        let mut offset = 0;
        for i in 1..=4 {
            let inter_res_all = vp9.enumerate(&[], amount, offset).unwrap();
            offset += amount;

            assert_eq!(50000, inter_res_all.len());
            for inter in inter_res_all {
                res_all.insert(inter);
            }
            assert_eq!(i * 50000, res_all.len(), "there are duplicates");
        }
        let inter_res = vp9.enumerate(&[], amount, offset).unwrap();
        assert_eq!(16000, inter_res.len(), "there are only 16000 configs left");
        for inter in inter_res {
            res_all.insert(inter);
        }
        assert_eq!(
            vp9.rc(),
            BigUint::from(res_all.len()),
            "there are duplicates"
        );

        assert_eq!(BigUint::from(80u32), vp9.execute_query(&assumptions));

        let amount = 40;
        let mut offset = 40;
        let inter_res_assumptions_2 = vp9.enumerate(&assumptions, amount, offset).unwrap();
        offset += amount;

        for inter in inter_res_assumptions_2.clone() {
            res_assumptions.insert(inter);
        }
        assert_eq!(40, inter_res_assumptions_2.len());

        // the cycle for that set of assumptions starts again
        let inter_res_assumptions_3 = vp9.enumerate(&assumptions, amount, offset).unwrap();
        for inter in inter_res_assumptions_3.clone() {
            res_assumptions.insert(inter);
        }
        assert_eq!(40, inter_res_assumptions_3.len());
        // if there is no cycle, we request 40 configs for the 3rd time resulting in a total of 120
        assert_eq!(
            80,
            res_assumptions.len(),
            "because of the cycle we should have gotten duplicates"
        );
    }

    #[test]
    fn enumeration_big_ddnnf() {
        let auto1: Ddnnf = Ddnnf::from_file(Path::new("tests/data/auto1_d4.nnf"), Some(2513));

        let mut res_all = HashSet::new();
        let mut assumptions = vec![
            1, -2, -3, 4, -5, 6, 7, 8, -9, -10, 11, -12, -13, 100, -101, 102,
        ];

        let amount = 1_000;
        let mut offset = 0;
        for i in (1_000..=10_000).step_by(1_000) {
            let configs = auto1.enumerate(&assumptions, amount, offset).unwrap();
            offset += amount;

            for inter in configs {
                res_all.insert(inter);
            }
            assert_eq!(i, res_all.len(), "there are duplicates");

            // shuffeling the assumptions should have no effect on the caching of the number of configs that we already looked at
            assumptions.shuffle(&mut rng())
        }
    }

    #[test]
    fn enumeration_step_by_step() {
        let mut vp9: Ddnnf = Ddnnf::from_file(Path::new("tests/data/VP9_d4.nnf"), Some(42));

        let mut res_all = HashSet::new();
        let mut assumptions = [-35, 42];

        let amount = 1;
        let mut offset = 0;
        for i in 1..=1_000 {
            let configs = vp9.enumerate(&assumptions, amount, offset).unwrap();
            offset += amount;
            for inter in configs {
                assert!(vp9.sat(&inter));
                assert_eq!(
                    vp9.number_of_variables as usize,
                    inter.len(),
                    "we got only a partial config"
                );

                // ensure that the assumptions are fulfilled
                assert!(inter.contains(&-35) && inter.contains(&42));
                assert!(!inter.contains(&35) && !inter.contains(&-42));

                res_all.insert(inter);
            }
            assert_eq!(i, res_all.len(), "there are duplicates");
        }

        // changing the order of the assumptions. This should have no effect on the position
        assumptions = [42, -35];

        // vp9.rt() under the assumptions is 86400. Hence, we should never get more than 86400 different configs
        let amount = 2_000;
        for i in (1_000..=100_000).step_by(2_000) {
            let configs = vp9.enumerate(&assumptions, 2_000, offset).unwrap();
            offset += amount;
            for inter in configs {
                assert!(vp9.sat(&inter));
                assert_eq!(
                    vp9.number_of_variables as usize,
                    inter.len(),
                    "we got only a partial config"
                );

                // ensure that the assumptions are fulfilled
                assert!(inter.contains(&-35) && inter.contains(&42));
                assert!(!inter.contains(&35) && !inter.contains(&-42));

                res_all.insert(inter);
            }
            assert_eq!(
                min(86400, 2000 + i),
                res_all.len(),
                "there are duplicates or more configs then wanted"
            );
        }
    }

    #[test]
    fn enumeration_is_not_possible() {
        let vp9: Ddnnf = Ddnnf::from_file(Path::new("tests/data/VP9_d4.nnf"), Some(42));
        let auto1: Ddnnf = Ddnnf::from_file(Path::new("tests/data/auto1_d4.nnf"), Some(2513));

        assert!(vp9.enumerate(&[1, -1], 1, 0).is_none());
        assert!(
            vp9.enumerate(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, 0)
                .is_none()
        );
        assert!(vp9.enumerate(&[100], 1, 0).is_none());

        assert!(auto1.enumerate(&[1, -1], 1, 0).is_none());
        assert!(
            auto1
                .enumerate(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, 0)
                .is_none()
        );
        assert!(auto1.enumerate(&[-10_000], 1, 0).is_none());
    }

    #[test]
    fn sampling_validity() {
        let mut vp9: Ddnnf = Ddnnf::from_file(Path::new("tests/data/VP9_d4.nnf"), Some(42));
        let mut auto1: Ddnnf = Ddnnf::from_file(Path::new("tests/data/auto1_d4.nnf"), Some(2513));

        let vp9_assumptions = vec![38, 2, -14];
        let vp9_samples = vp9
            .uniform_random_sampling(&vp9_assumptions, 1_000, 42)
            .unwrap();
        for sample in vp9_samples {
            assert!(vp9.sat(&sample));
            assert_eq!(vp9.number_of_variables as usize, sample.len());
        }

        let auto1_samples = auto1
            .uniform_random_sampling(&[-546, 55, 646, -872, -873, 102, 23, 764, -1111], 1_000, 42)
            .unwrap();
        for sample in auto1_samples {
            assert!(auto1.sat(&sample));
            assert_eq!(auto1.number_of_variables as usize, sample.len());
        }
    }

    #[test]
    fn sampling_seeding() {
        let vp9: Ddnnf = Ddnnf::from_file(Path::new("tests/data/VP9_d4.nnf"), Some(42));
        let auto1: Ddnnf = Ddnnf::from_file(Path::new("tests/data/auto1_d4.nnf"), Some(2513));

        // same seeding should yield same results, different seeding should (normally) yield different results
        assert_eq!(
            vp9.uniform_random_sampling(&[], 100, 42),
            vp9.uniform_random_sampling(&[], 100, 42)
        );
        assert_eq!(
            vp9.uniform_random_sampling(&[23, 4, -17], 100, 99),
            vp9.uniform_random_sampling(&[23, 4, -17], 100, 99),
        );
        assert_ne!(
            vp9.uniform_random_sampling(&[38, 2, -14], 100, 99),
            vp9.uniform_random_sampling(&[38, 2, -14], 100, 50),
        );

        assert_eq!(
            auto1.uniform_random_sampling(&[], 100, 42),
            auto1.uniform_random_sampling(&[], 100, 42)
        );
        assert_eq!(
            auto1.uniform_random_sampling(
                &[-546, 55, 646, -872, -873, 102, 23, 764, -1111],
                100,
                1970
            ),
            auto1.uniform_random_sampling(
                &[-546, 55, 646, -872, -873, 102, 23, 764, -1111],
                100,
                1970
            ),
        );
        assert_ne!(
            auto1.uniform_random_sampling(&[11, 12, 13, -14, -15, -16], 100, 1),
            auto1.uniform_random_sampling(&[11, 12, 13, -14, -15, -16], 100, 2)
        );
    }

    #[test]
    fn sampling_is_not_possible() {
        let vp9: Ddnnf = Ddnnf::from_file(Path::new("tests/data/VP9_d4.nnf"), Some(42));
        let auto1: Ddnnf = Ddnnf::from_file(Path::new("tests/data/auto1_d4.nnf"), Some(2513));

        assert!(vp9.uniform_random_sampling(&[1, -1], 1, 42).is_none());
        assert!(
            vp9.uniform_random_sampling(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, 42)
                .is_none()
        );
        assert!(vp9.uniform_random_sampling(&[100], 1, 42).is_none());

        assert!(auto1.uniform_random_sampling(&[1, -1], 1, 42).is_none());
        assert!(
            auto1
                .uniform_random_sampling(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, 42)
                .is_none()
        );
        assert!(auto1.uniform_random_sampling(&[-10_000], 1, 42).is_none());
    }
}
