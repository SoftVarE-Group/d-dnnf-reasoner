use crate::sampler::covering_strategies::cover_with_caching;
use crate::sampler::data_structure::{Config, Sample};
use crate::sampler::iterator::t_wise_over;
use crate::sampler::sample_merger::{AndMerger, SampleMerger};
use crate::sampler::sat_solver::SatSolver;
use std::cmp::min;

#[derive(Debug, Clone)]
pub struct ZippingMerger<'a> {
    pub t: usize,
    pub sat_solver: &'a SatSolver<'a>,
}

// Mark ZippingMerger as an AndMerger
impl AndMerger for ZippingMerger<'_> {}

impl SampleMerger for ZippingMerger<'_> {
    fn merge(&self, node_id: usize, left: &Sample, right: &Sample) -> Sample {
        if left.is_empty() {
            return right.clone();
        } else if right.is_empty() {
            return left.clone();
        }

        let mut sample = ZippingMerger::zip_samples(left, right);

        /*
        Iterate over the remaining interactions. Those are all interactions
        that contain at least one literal of the left and one of the right subgraph.
         */
        let left_literals = left.get_literals();
        let right_literals = right.get_literals();

        for k in 1..self.t {
            // take k literals of the left subgraph and t-k literals of the right subgraph
            let left_len = min(left_literals.len(), k);
            let right_len = min(right_literals.len(), self.t - k);
            let left_iter = t_wise_over(left_literals, left_len);

            for left_part in left_iter {
                let right_iter = t_wise_over(right_literals, right_len);
                for mut right_part in right_iter {
                    right_part.extend(&left_part);
                    cover_with_caching(
                        &mut sample,
                        &right_part,
                        self.sat_solver,
                        node_id,
                    );
                }
            }
        }
        sample
    }
}

impl ZippingMerger<'_> {
    /// Creates a new sample by zipping the given samples together. It is assumed that the vars
    /// covered by the samples are disjoint. For decomposable AND nodes this is always the case.
    fn zip_samples(left: &Sample, right: &Sample) -> Sample {
        let mut new_sample = Sample::new_from_samples(&[left, right]);

        left.iter_with_completeness()
            .zip(right.iter_with_completeness())
            .for_each(
                |(
                     (left_config, left_complete),
                     (right_config, right_complete),
                 )| {
                    let new_config =
                        Config::from_disjoint(left_config, right_config);
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
    use super::*;
    use crate::parser::build_ddnnf_tree_with_extras;
    use std::collections::HashSet;

    #[test]
    fn test_zip_samples() {
        let left = Sample::new_from_configs(vec![
            Config::from(&[1, 2]),
            Config::from(&[-1, -2]),
        ]);
        let right = Sample::new_from_configs(vec![
            Config::from(&[3, 4]),
            Config::from(&[-3, -4]),
        ]);

        let zipped = ZippingMerger::zip_samples(&left, &right);
        let mut iter = zipped.iter();

        assert_eq!(Some(&Config::from(&[1, 2, 3, 4])), iter.next());
        assert_eq!(Some(&Config::from(&[-1, -2, -3, -4])), iter.next());
        assert_eq!(None, iter.next());
    }

    #[test]
    fn test_zipping_merger() {
        let ddnnf =
            build_ddnnf_tree_with_extras("./tests/data/small_test.dimacs.nnf");
        let node = ddnnf.number_of_nodes - 1;
        let sat_solver = SatSolver::new(&ddnnf);

        let zipping_merger = ZippingMerger {
            t: 2,
            sat_solver: &sat_solver,
        };

        let mut left_sample = Sample::new_with_literals(
            HashSet::from([2, 3]),
            vec![-2, 3],
        );
        left_sample.add_partial(Config::from(&[3]));
        let right_sample =
            Sample::new_from_configs(vec![Config::from(&[1, 4])]);

        let result = zipping_merger.merge(node, &left_sample, &right_sample);
        let expected =
            Sample::new_from_configs(vec![Config::from(&[-2, 1, 3, 4])]);
        assert_eq!(result, expected);
    }
}
