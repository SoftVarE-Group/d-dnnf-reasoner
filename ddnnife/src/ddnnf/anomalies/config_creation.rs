use std::{
    cmp::min,
    collections::HashMap,
    sync::{Arc, Mutex},
};

use itertools::Itertools;
use num::{BigInt, BigRational, ToPrimitive, Zero};
use once_cell::sync::Lazy;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution, WeightedAliasIndex};
use rand_pcg::{Lcg64Xsh32, Pcg32};

use crate::Ddnnf;
use crate::NodeType::*;

#[allow(clippy::type_complexity)]
static ENUMERATION_CACHE: Lazy<Arc<Mutex<HashMap<Vec<i32>, usize>>>> =
    Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));

impl Ddnnf {
    /// Creates satisfiable complete configurations for a ddnnf and given assumptions
    /// If the ddnnf on itself or in combination with the assumption is unsatisfiable,
    /// then we can not create any satisfiable configuration and simply return None.
    pub fn enumerate(
        &mut self,
        assumptions: &mut Vec<i32>,
        amount: usize,
    ) -> Option<Vec<Vec<i32>>> {
        if amount == 0 {
            return Some(Vec::new());
        }

        if !self.preprocess_config_creation(assumptions) {
            return None;
        }
        assumptions.sort_unstable_by_key(|f| f.abs());

        if self.execute_query(assumptions) > BigInt::ZERO {
            let last_stop = match ENUMERATION_CACHE.lock().unwrap().get(assumptions) {
                Some(&x) => x,
                None => 0,
            };

            let mut sample_list = self.enumerate_node(
                (
                    &BigInt::from(last_stop),
                    &min(self.rt(), BigInt::from(last_stop + amount)),
                ),
                self.nodes.len() - 1,
            );
            for sample in sample_list.iter_mut() {
                sample.sort_unstable_by_key(|f| f.abs());
            }

            ENUMERATION_CACHE.lock().unwrap().insert(
                assumptions.to_vec(),
                (min(self.rt(), BigInt::from(last_stop + amount)) % self.rt())
                    .to_usize()
                    .expect("Attempt to convert to large integer!"),
            );
            return Some(sample_list);
        }
        None
    }

    /// Generates amount many uniform random samples under a given set of assumptions and a seed.
    /// Each sample is sorted by the number of the features. Each sample is a complete configuration with #SAT of 1.
    /// If the ddnnf itself or in combination with the assumptions is unsatisfiable, None is returned.
    pub fn uniform_random_sampling(
        &mut self,
        assumptions: &[i32],
        amount: usize,
        seed: u64,
    ) -> Option<Vec<Vec<i32>>> {
        if !self.preprocess_config_creation(assumptions) {
            return None;
        }

        if self.execute_query(assumptions) > BigInt::ZERO {
            let mut sample_list = self.sample_node(
                amount,
                self.nodes.len() - 1,
                &mut Pcg32::seed_from_u64(seed),
            );
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
    fn preprocess_config_creation(&mut self, assumptions: &[i32]) -> bool {
        // if any of the assumptions isn't valid by being in the range of +-#variables, then we return false
        if assumptions
            .iter()
            .any(|f| f.abs() > self.number_of_variables as i32)
        {
            return false;
        }

        for node in self.nodes.iter_mut() {
            node.temp.clone_from(&node.count);
        }

        for literal in assumptions.iter() {
            if let Some(&x) = self.literals.get(&-literal) {
                self.nodes[x].temp.set_zero();
            }
        }

        // We can't create a config that contains a true node.
        // Hence, we have to hide the true node by changing its count to 0
        for &index in self.true_nodes.iter() {
            self.nodes[index].temp.set_zero();
        }
        true
    }

    // Handles a node appropiate depending on its kind to produce complete
    // satisfiable configurations
    fn enumerate_node(&self, range: (&BigInt, &BigInt), index: usize) -> Vec<Vec<i32>> {
        let _range2 = (
            range
                .0
                .to_usize()
                .expect("Attempt to convert to large integer!"),
            range
                .1
                .to_usize()
                .expect("Attempt to convert to large integer!"),
        );
        let mut enumeration_list = Vec::new();
        if range.1.is_zero() || self.nodes[index].temp.is_zero() {
            return enumeration_list;
        }

        match &self.nodes[index].ntype {
            And { children } => {
                let mut acc_amount = BigInt::from(1);
                let mut enumeration_child_lists = Vec::new();

                for &child in children {
                    // skip the true nodes
                    if self.true_nodes.contains(&child) {
                        continue;
                    }

                    if &acc_amount < range.1 {
                        let change = (&BigInt::ZERO, min(range.1, &self.nodes[child].temp));
                        enumeration_child_lists.push(self.enumerate_node(change, child));
                        acc_amount *= change.1;
                    } else {
                        // restrict the creation of any more configs
                        enumeration_child_lists.push(vec![self
                            .enumerate_node((&BigInt::ZERO, &BigInt::from(1)), child)[0]
                            .clone()]);
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
                    .skip(
                        range
                            .0
                            .to_usize()
                            .expect("Attempt to convert to large integer!"),
                    )
                    .take(
                        range
                            .1
                            .to_usize()
                            .expect("Attempt to convert to large integer!")
                            - range
                                .0
                                .to_usize()
                                .expect("Attempt to convert to large integer!"),
                    ) // stop after we got our required amount of configs
                    .collect();
            }
            Or { children } => {
                let mut acc_amount = BigInt::ZERO;

                for &child in children {
                    if self.nodes[child].temp == BigInt::ZERO {
                        continue;
                    }

                    if &acc_amount < range.1 {
                        let change = (&BigInt::ZERO, min(range.1, &self.nodes[child].temp));
                        enumeration_list.append(&mut self.enumerate_node(change, child));
                        acc_amount += change.1;
                    } else {
                        break;
                    }
                }
            }
            Literal { literal } => {
                enumeration_list.push(vec![*literal]);
            }
            _ => (),
        }
        enumeration_list
    }

    // Performs the operations needed to generate random samples.
    // The algorithm is based upon KUS's uniform random sampling algorithm.
    fn sample_node(&self, amount: usize, index: usize, rng: &mut Lcg64Xsh32) -> Vec<Vec<i32>> {
        let mut sample_list = Vec::new();
        if amount == 0 {
            return sample_list;
        }
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
            }
            Or { children } => {
                let mut pick_amount = vec![0; children.len()];
                let mut choices = Vec::new();
                let mut weights = Vec::new();

                // compute the probability of getting a sample of a child node
                let parent_count_as_float = BigRational::from(self.nodes[index].temp.clone());
                #[allow(clippy::needless_range_loop)]
                for child_index in 0..children.len() {
                    let child_count_as_float =
                        BigRational::from(self.nodes[children[child_index]].temp.clone());

                    // can't get a sample of a children with no more valid configuration
                    if !child_count_as_float.is_zero() {
                        let child_amount = (child_count_as_float / &parent_count_as_float)
                            .to_f64()
                            .expect("Failed to convert BigRational to f64!")
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
                    ));
                }

                // add empty lists for child nodes that have a count of zero
                while sample_list.len() != amount {
                    sample_list.push(Vec::new());
                }

                sample_list.shuffle(rng);
            }
            Literal { literal } => {
                for _ in 0..amount {
                    sample_list.push(vec![*literal]);
                }
            }
            _ => (),
        }
        sample_list
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use rand::thread_rng;

    use super::*;
    use crate::parser::build_ddnnf;

    #[test]
    fn enumeration_small_ddnnf() {
        let mut vp9: Ddnnf = build_ddnnf("tests/data/VP9_d4.nnf", Some(42));

        let mut res_all = HashSet::new();
        let mut res_assumptions = HashSet::new();

        let mut assumptions = vec![
            1, 2, 3, -4, -5, 6, 7, -8, -9, 10, 11, -12, -13, -14, 15, 16, -17, -18, 19, 20, 27,
        ];
        let inter_res_assumptions_1 = vp9.enumerate(&mut assumptions, 40).unwrap();
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

        for i in 1..=4 {
            let inter_res_all = vp9.enumerate(&mut vec![], 50000).unwrap();
            assert_eq!(50000, inter_res_all.len());
            for inter in inter_res_all {
                res_all.insert(inter);
            }
            assert_eq!(i * 50000, res_all.len(), "there are duplicates");
        }
        let inter_res = vp9.enumerate(&mut vec![], 50000).unwrap();
        assert_eq!(16000, inter_res.len(), "there are only 16000 configs left");
        for inter in inter_res {
            res_all.insert(inter);
        }
        assert_eq!(
            vp9.rt(),
            BigInt::from(res_all.len()),
            "there are duplicates"
        );

        assert_eq!(BigInt::from(80), vp9.execute_query(&assumptions));
        let inter_res_assumptions_2 = vp9.enumerate(&mut assumptions, 40).unwrap();
        for inter in inter_res_assumptions_2.clone() {
            res_assumptions.insert(inter);
        }
        assert_eq!(40, inter_res_assumptions_2.len());

        // the cycle for that set of assumptions starts again
        let inter_res_assumptions_3 = vp9.enumerate(&mut assumptions, 40).unwrap();
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
        let mut auto1: Ddnnf = build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));

        let mut res_all = HashSet::new();
        let mut assumptions = vec![
            1, -2, -3, 4, -5, 6, 7, 8, -9, -10, 11, -12, -13, 100, -101, 102,
        ];

        for i in (1_000..=10_000).step_by(1_000) {
            let configs = auto1.enumerate(&mut assumptions, 1_000).unwrap();
            for inter in configs {
                res_all.insert(inter);
            }
            assert_eq!(i, res_all.len(), "there are duplicates");

            // shuffeling the assumptions should have no effect on the caching of the number of configs that we already looked at
            assumptions.shuffle(&mut thread_rng())
        }
    }

    #[test]
    fn enumeration_step_by_step() {
        let mut vp9: Ddnnf = build_ddnnf("tests/data/VP9_d4.nnf", Some(42));

        let mut res_all = HashSet::new();
        let mut assumptions = vec![-35, 42];

        for i in 1..=1_000 {
            let configs = vp9.enumerate(&mut assumptions, 1).unwrap();
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
        assumptions = vec![42, -35];

        // vp9.rt() under the assumptions is 86400. Hence, we should never get more than 86400 different configs
        for i in (1_000..=100_000).step_by(2_000) {
            let configs = vp9.enumerate(&mut assumptions, 2_000).unwrap();
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
        let mut vp9: Ddnnf = build_ddnnf("tests/data/VP9_d4.nnf", Some(42));
        let mut auto1: Ddnnf = build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));

        assert!(vp9.enumerate(&mut vec![1, -1], 1).is_none());
        assert!(vp9
            .enumerate(&mut vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)
            .is_none());
        assert!(vp9.enumerate(&mut vec![100], 1).is_none());

        assert!(auto1.enumerate(&mut vec![1, -1], 1).is_none());
        assert!(auto1
            .enumerate(&mut vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)
            .is_none());
        assert!(auto1.enumerate(&mut vec![-10_000], 1).is_none());
    }

    #[test]
    fn sampling_validity() {
        let mut vp9: Ddnnf = build_ddnnf("tests/data/VP9_d4.nnf", Some(42));
        let mut auto1: Ddnnf = build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));

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
        let mut vp9: Ddnnf = build_ddnnf("tests/data/VP9_d4.nnf", Some(42));
        let mut auto1: Ddnnf = build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));

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
        let mut vp9: Ddnnf = build_ddnnf("tests/data/VP9_d4.nnf", Some(42));
        let mut auto1: Ddnnf = build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));

        assert!(vp9.uniform_random_sampling(&[1, -1], 1, 42).is_none());
        assert!(vp9
            .uniform_random_sampling(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, 42)
            .is_none());
        assert!(vp9.uniform_random_sampling(&[100], 1, 42).is_none());

        assert!(auto1.uniform_random_sampling(&[1, -1], 1, 42).is_none());
        assert!(auto1
            .uniform_random_sampling(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, 42)
            .is_none());
        assert!(auto1.uniform_random_sampling(&[-10_000], 1, 42).is_none());
    }
}
