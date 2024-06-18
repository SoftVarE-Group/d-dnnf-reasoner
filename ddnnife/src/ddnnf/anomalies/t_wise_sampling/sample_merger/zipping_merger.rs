use super::super::{
    covering_strategies::cover_with_caching_twise, t_iterator::TInteractionIter, SamplingResult,
    SatWrapper,
};
use super::{AndMerger, SampleMerger};
use super::{Config, Sample};
use crate::Ddnnf;
use std::cmp::min;
use std::collections::HashSet;
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
    fn merge(&self, node_id: usize, left: &Sample, right: &Sample) -> Sample {
        if left.is_empty() {
            return right.clone();
        }

        if right.is_empty() {
            return left.clone();
        }

        let mut sample = Self::zip_samples(left, right, self.ddnnf.number_of_variables as usize);

        for interaction in Self::interactions(left, right, self.t).iter() {
            cover_with_caching_twise(
                &mut sample,
                interaction,
                self.sat_solver,
                node_id,
                self.ddnnf.number_of_variables as usize,
            );
        }

        sample
    }

    fn merge_all(&self, node_id: usize, samples: &[&Sample]) -> Sample {
        let (singles, mut samples): (Vec<&Sample>, Vec<&Sample>) =
            samples.iter().partition(|sample| sample.len() <= 1);

        let single = singles.iter().fold(Sample::default(), |acc, s| {
            self.merge_in_place(node_id, acc, s)
        });

        samples.push(&single);
        samples.sort_unstable();

        samples.iter().fold(Sample::default(), |acc, s| {
            self.merge_in_place(node_id, acc, s)
        })
    }

    // For an and node, a single void sample will render the resulting sample void.
    fn is_void(&self, samples: &[&SamplingResult]) -> bool {
        samples
            .iter()
            .any(|result| matches!(result, SamplingResult::Void))
    }
}

impl ZippingMerger<'_> {
    /// Generates all t-wise interactions between two samples.
    fn interactions(left: &Sample, right: &Sample, t: usize) -> HashSet<Vec<i32>> {
        Self::generate_interactions(
            Self::generate_self_interactions(left, t),
            Self::generate_self_interactions(right, t),
        )
    }

    /// Generates all interactions between to sets of interactions.
    ///
    /// Both sets are expected to have at least an empty set for each length.
    fn generate_interactions(
        left: impl Iterator<Item = HashSet<Vec<i32>>>,
        right: impl DoubleEndedIterator<Item = HashSet<Vec<i32>>>,
    ) -> HashSet<Vec<i32>> {
        let mut interactions = HashSet::new();

        // Both sets are sorted from 1 to t-1.
        // To combine 1 with t-1, 2 with t-2, etc. we zip left with the reverse of right.
        // This will give us this exact ordering as both are complete with at least an empty config for each length.
        left.zip(right.rev()).for_each(|(left, right)| {
            for left in left {
                for right in &right {
                    let mut interaction = left.clone();
                    interaction.extend_from_slice(right);
                    interactions.insert(interaction);
                }
            }
        });

        interactions
    }

    /// Generates a set of interactions inside a sample ordered by interactions size.
    ///
    /// The first element will be all interactions of size 1, the second of size 2, ...
    ///
    /// In case no interaction of any given size is found, an empty set is still present.
    fn generate_self_interactions(
        sample: &Sample,
        t: usize,
    ) -> impl DoubleEndedIterator<Item = HashSet<Vec<i32>>> {
        // Extract all decided literals from the existing configs.
        let configs: Vec<Vec<i32>> = sample
            .iter()
            .map(|config| config.get_decided_literals().collect())
            .collect();

        // From 1 to t ...
        (1..t)
            // consider each length once.
            .map(move |k| {
                let mut interactions = HashSet::new();
                // For each config ...
                configs.iter().for_each(|config| {
                    // generate all interactions of the current length ...
                    TInteractionIter::new(config, min(config.len(), k)).for_each(|interaction| {
                        // and insert them into the set.
                        interactions.insert(interaction.to_vec());
                    });
                });

                // There needs to be at least one (empty) element for the merge to happen.
                if interactions.is_empty() {
                    interactions.insert(Vec::new());
                }

                interactions
            })
    }

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
                        new_sample.add(new_config);
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
            t: 3,
            sat_solver: &sat_solver,
            ddnnf: &ddnnf,
        };

        let mut left_sample = new_with_literals(HashSet::from([2, 3]), vec![-2, 3]);
        left_sample.add_partial(Config::from(&[3, -2], 4));
        let right_sample = Sample::new_from_configs(vec![Config::from(&[1, 4], 4)]);

        let sample = zipping_merger.merge(node, &left_sample, &right_sample);
        let result = sample.iter().next().unwrap();
        let expected = Config::from(&[-2, 1, 3, 4], 4);

        assert_eq!(result, &expected);
    }
}
