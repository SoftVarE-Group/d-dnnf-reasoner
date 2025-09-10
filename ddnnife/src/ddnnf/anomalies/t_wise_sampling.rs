mod config;
mod covering_strategies;
mod sample;
mod sample_merger;
mod sampling_result;
mod sat_wrapper;
mod t_iterator;
mod t_wise_sampler;

use crate::Ddnnf;
use crate::ddnnf::extended_ddnnf::ExtendedDdnnf;
use crate::ddnnf::extended_ddnnf::objective_function::FloatOrd;
use SamplingResult::ResultWithSample;
pub use config::Config;
use covering_strategies::cover_with_caching_sorted;
use itertools::Itertools;
pub use sample::Sample;
use sample_merger::attribute_similarity_merger::AttributeSimilarityMerger;
use sample_merger::attribute_zipping_merger::AttributeZippingMerger;
use sample_merger::similarity_merger::SimilarityMerger;
use sample_merger::zipping_merger::ZippingMerger;
pub use sampling_result::SamplingResult;
use sat_wrapper::SatWrapper;
use std::cmp::min;
use streaming_iterator::StreamingIterator;
use t_iterator::TInteractionIter;
use t_wise_sampler::TWiseSampler;
use t_wise_sampler::{complete_partial_configs_optimal, trim_and_resample};

impl Ddnnf {
    /// Generates samples so that all t-wise interactions between literals are covered.
    pub fn sample_t_wise(&self, t: usize) -> SamplingResult {
        // Setup everything needed for the sampling process.
        let sat_solver = SatWrapper::new(self);
        let and_merger = ZippingMerger {
            t,
            sat_solver: &sat_solver,
            ddnnf: self,
        };
        let or_merger = SimilarityMerger { t };

        TWiseSampler::new(self, and_merger, or_merger).sample(t)
    }
}

impl ExtendedDdnnf {
    pub fn sample_t_wise(&self, t: usize) -> SamplingResult {
        let sat_solver = SatWrapper::new(&self.ddnnf);
        let and_merger = AttributeZippingMerger {
            t,
            sat_solver: &sat_solver,
            ext_ddnnf: self,
        };
        let or_merger = AttributeSimilarityMerger { t, ext_ddnnf: self };

        let mut sampler = TWiseSampler::new(&self.ddnnf, and_merger, or_merger);

        for node_id in 0..sampler.ddnnf.nodes.len() {
            let partial_sample = sampler.partial_sample(node_id);
            sampler.partial_samples.insert(node_id, partial_sample);
        }

        let root_id = sampler.ddnnf.nodes.len() - 1;

        let sampling_result = sampler
            .partial_samples
            .remove(&root_id)
            .expect("Root sample does not exist!");

        if let ResultWithSample(mut sample) = sampling_result {
            debug_assert!(
                sample
                    .complete_configs
                    .iter()
                    .all(|config| !config.get_literals().contains(&0)),
                "Complete Configs must not contain undecided variables."
            );

            sample = trim_and_resample(
                root_id,
                sample,
                t,
                self.ddnnf.number_of_variables as usize,
                &sat_solver,
            );

            complete_partial_configs_optimal(&mut sample, self);

            ResultWithSample(sample)
        } else {
            sampling_result
        }
    }

    pub fn sample_t_wise_yasa(&self, t: usize) -> Sample {
        let sat_solver = SatWrapper::new(&self.ddnnf);
        let vars = (1..=self.ddnnf.number_of_variables).collect_vec();
        let literals = (-(self.ddnnf.number_of_variables as i32)
            ..=self.ddnnf.number_of_variables as i32)
            .filter(|&literal| literal != 0)
            .collect_vec();
        let root_id = self.ddnnf.nodes.len() - 1;
        let mut sample = Sample::new(vars.into_iter().collect());

        let mut interactions = Vec::new();
        TInteractionIter::new(&literals[..], min(literals.len(), t))
            .for_each(|interaction| interactions.push(interaction.to_vec()));

        interactions
            .iter()
            .sorted_by_cached_key(|interaction| {
                FloatOrd::from(self.get_objective_fn_val_of_literals(&interaction[..]))
            })
            .rev()
            .for_each(|interaction| {
                cover_with_caching_sorted(
                    &mut sample,
                    interaction,
                    &sat_solver,
                    root_id,
                    self.ddnnf.number_of_variables as usize,
                    self,
                );
            });

        sample = trim_and_resample(
            root_id,
            sample,
            t,
            self.ddnnf.number_of_variables as usize,
            &sat_solver,
        );
        complete_partial_configs_optimal(&mut sample, self);

        sample.literals = literals;

        sample
    }
}

#[cfg(test)]
mod test {
    use crate::ddnnf::anomalies::t_wise_sampling::Sample;
    use crate::ddnnf::anomalies::t_wise_sampling::t_iterator::TInteractionIter;
    use crate::ddnnf::extended_ddnnf::optimal_configs::test::build_sandwich_ext_ddnnf_with_objective_function_values;
    use crate::{Ddnnf, parser::build_ddnnf};
    use itertools::Itertools;
    use std::collections::HashSet;
    use std::path::Path;
    use streaming_iterator::StreamingIterator;

    fn check_validity_of_sample(sample: &Sample, ddnnf: &Ddnnf, t: usize) {
        let sample_literals: HashSet<i32> = sample.get_literals().iter().copied().collect();
        sample.iter().for_each(|config| {
            assert!(
                config
                    .get_decided_literals()
                    .all(|literal| sample_literals.contains(&literal))
            );
        });

        sample
            .iter()
            .map(|config| config.get_decided_literals().collect_vec())
            .for_each(|literals| {
                // every config must be complete and satisfiable
                assert_eq!(
                    ddnnf.number_of_variables as usize,
                    literals.len(),
                    "config is not complete"
                );
                assert!(ddnnf.sat_immutable(&literals[..]));
            });

        let all_literals = (-(ddnnf.number_of_variables as i32)..=ddnnf.number_of_variables as i32)
            .filter(|&literal| literal != 0)
            .collect_vec();

        TInteractionIter::new(&all_literals[..], t)
            .filter(|interaction| ddnnf.sat_immutable(interaction))
            .for_each(|interaction| {
                assert!(
                    sample.covers(interaction),
                    "Valid interaction {:?} is not covered.",
                    interaction
                )
            });
    }

    #[test]
    fn ddnnf_t_wise_sampling_validity_small_model() {
        let vp9: Ddnnf = build_ddnnf(Path::new("tests/data/VP9_d4.nnf"), Some(42));

        for t in 1..=4 {
            check_validity_of_sample(vp9.sample_t_wise(t).get_sample().unwrap(), &vp9, t);
        }
    }

    #[test]
    fn ddnnf_t_wise_sampling_validity_big_model() {
        let mut auto1: Ddnnf = build_ddnnf(Path::new("tests/data/auto1_d4.nnf"), Some(2513));
        let t = 1;

        check_validity_of_sample(auto1.sample_t_wise(t).get_sample().unwrap(), &mut auto1, t);
    }

    #[test]
    fn ext_ddnnf_t_wise_sampling_validity() {
        let ext_ddnnf = build_sandwich_ext_ddnnf_with_objective_function_values();

        for t in 1..=4 {
            check_validity_of_sample(
                ext_ddnnf.sample_t_wise(t).get_sample().unwrap(),
                &ext_ddnnf.ddnnf,
                t,
            );
        }
    }

    #[test]
    fn ext_ddnnf_t_wise_sampling_yasa_validity() {
        let ext_ddnnf = build_sandwich_ext_ddnnf_with_objective_function_values();

        for t in 1..=4 {
            check_validity_of_sample(&ext_ddnnf.sample_t_wise_yasa(t), &ext_ddnnf.ddnnf, t);
        }
    }
}
