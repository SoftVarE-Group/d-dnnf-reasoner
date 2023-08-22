use crate::ddnnf::anomalies::t_wise_sampling::covering_strategies::cover_with_caching;
use crate::ddnnf::anomalies::t_wise_sampling::data_structure::{Config, Sample};
use crate::ddnnf::anomalies::t_wise_sampling::sample_merger::{AndMerger, SampleMerger};
use crate::ddnnf::anomalies::t_wise_sampling::sat_wrapper::SatWrapper;
use crate::ddnnf::anomalies::t_wise_sampling::t_iterator::TInteractionIter;
use crate::Ddnnf;
use std::cmp::min;

use rand::prelude::{SliceRandom, StdRng};

use streaming_iterator::StreamingIterator;

#[derive(Debug, Clone)]
pub struct ZippingMerger<'a> {
    pub t: usize,
    pub sat_solver: &'a SatWrapper<'a>,
    pub ddnnf: &'a Ddnnf,
}

// Mark ZippingMerger as an AndMerger
impl AndMerger for ZippingMerger<'_> {}

impl SampleMerger for ZippingMerger<'_> {
    fn merge(&self, node_id: usize, left: &Sample, right: &Sample, rng: &mut StdRng) -> Sample {
        if left.is_empty() {
            return right.clone();
        } else if right.is_empty() {
            return left.clone();
        }

        let mut sample =
            ZippingMerger::zip_samples(left, right, self.ddnnf.number_of_variables as usize);
        /*
        Iterate over the remaining interactions. Those are all interactions
        that contain at least one literal of the left and one of the right subgraph.
         */
        let mut left_literals: Vec<i32> = left.get_literals().to_vec();
        let mut right_literals: Vec<i32> = right.get_literals().to_vec();
        left_literals.shuffle(rng);
        right_literals.shuffle(rng);

        debug_assert!(!left_literals.iter().any(|x| *x == 0));
        debug_assert!(!right_literals.iter().any(|x| *x == 0));

        for k in 1..self.t {
            // take k literals of the left subgraph and t-k literals of the right subgraph
            let left_len = min(left_literals.len(), k);
            let right_len = min(right_literals.len(), self.t - k);
            //let left_iter = t_wise_over(left_literals, left_len);
            let mut left_iter = TInteractionIter::new(&left_literals, left_len);

            while let Some(left_part) = left_iter.next() {
                //let right_iter = t_wise_over(right_literals, right_len);
                let mut right_iter = TInteractionIter::new(&right_literals, right_len);
                while let Some(right_part) = right_iter.next() {
                    let mut interaction = right_part.to_vec();
                    interaction.extend_from_slice(left_part);
                    cover_with_caching(
                        &mut sample,
                        &interaction,
                        self.sat_solver,
                        node_id,
                        self.ddnnf.number_of_variables as usize,
                    );
                }
            }
        }
        sample
    }

    fn merge_all(&self, node_id: usize, samples: &[&Sample], rng: &mut StdRng) -> Sample {
        let (singles, mut samples): (Vec<&Sample>, Vec<&Sample>) =
            samples.iter().partition(|sample| sample.len() <= 1);

        let single = singles.iter().fold(Sample::default(), |acc, s| {
            self.merge_in_place(node_id, acc, s, rng)
        });

        samples.push(&single);
        samples.sort_unstable();

        samples.iter().fold(Sample::default(), |acc, s| {
            self.merge_in_place(node_id, acc, s, rng)
        })
    }
}

impl ZippingMerger<'_> {
    /// Creates a new sample by zipping the given samples together. It is assumed that the vars
    /// covered by the samples are disjoint. For decomposable AND nodes this is always the case.
    fn zip_samples(left: &Sample, right: &Sample, number_of_variables: usize) -> Sample {
        let mut new_sample = Sample::new_from_samples(&[left, right]);

        left.iter_with_completeness()
            .zip(right.iter_with_completeness())
            .for_each(
                |((left_config, left_complete), (right_config, right_complete))| {
                    let new_config =
                        Config::from_disjoint(left_config, right_config, number_of_variables);
                    if left_complete && right_complete {
                        new_sample.add_complete(new_config);
                    } else {
                        new_sample.add_partial(new_config);
                    }
                },
            );

        let remaining = if left.len() >= right.len() {
            left.iter().skip(right.len())
        } else {
            right.iter().skip(left.len())
        };
        remaining
            .cloned()
            .for_each(|config| new_sample.add_partial(config));

        new_sample
    }
}

#[cfg(test)]
mod test {
    use crate::parser::build_ddnnf;

    use super::*;
    use rand::SeedableRng;
    use std::collections::HashSet;

    #[test]
    fn test_zip_samples() {
        let left =
            Sample::new_from_configs(vec![Config::from(&[1, 2], 4), Config::from(&[-1, -2], 4)]);
        let right =
            Sample::new_from_configs(vec![Config::from(&[3, 4], 4), Config::from(&[-3, -4], 4)]);

        let zipped = ZippingMerger::zip_samples(&left, &right, 4);
        let mut iter = zipped.iter();

        assert_eq!(Some(&Config::from(&[1, 2, 3, 4], 4)), iter.next());
        assert_eq!(Some(&Config::from(&[-1, -2, -3, -4], 4)), iter.next());
        assert_eq!(None, iter.next());
    }

    /// Create an empty sample that may contain the given variables and will certainly contain
    /// the given literals. Only use this if you know that the configs you are going to add to
    /// this sample contain the given literals.
    fn new_with_literals(vars: HashSet<u32>, mut literals: Vec<i32>) -> Sample {
        literals.sort_unstable();
        literals.dedup();
        Sample {
            complete_configs: vec![],
            partial_configs: vec![],
            vars,
            literals,
        }
    }

    #[test]
    fn test_zipping_merger() {
        let ddnnf = build_ddnnf("./tests/data/small_ex_c2d.nnf", None);
        let node = ddnnf.nodes.len() - 1;
        let sat_solver = SatWrapper::new(&ddnnf);

        let zipping_merger = ZippingMerger {
            t: 2,
            sat_solver: &sat_solver,
            ddnnf: &ddnnf,
        };

        let mut rng = StdRng::seed_from_u64(42);
        let mut left_sample = new_with_literals(HashSet::from([2, 3]), vec![-2, 3]);
        left_sample.add_partial(Config::from(&[3], 4));
        let right_sample = Sample::new_from_configs(vec![Config::from(&[1, 4], 4)]);

        let result = zipping_merger.merge(node, &left_sample, &right_sample, &mut rng);
        let expected = Sample::new_from_configs(vec![Config::from(&[-2, 1, 3, 4], 4)]);
        assert_eq!(result, expected);
    }
}
