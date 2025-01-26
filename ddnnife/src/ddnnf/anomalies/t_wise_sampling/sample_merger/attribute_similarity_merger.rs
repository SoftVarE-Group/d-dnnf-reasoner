use crate::ddnnf::anomalies::t_wise_sampling::sample_merger::{OrMerger, SampleMerger};
use crate::ddnnf::anomalies::t_wise_sampling::{Sample, SamplingResult};
use crate::ddnnf::extended_ddnnf::ExtendedDdnnf;

#[derive(Debug, Copy, Clone)]
pub struct AttributeSimilarityMerger<'a> {
    pub t: usize,
    pub ext_ddnnf: &'a ExtendedDdnnf,
}

// Mark AttributeSimilarityMerger as an OrMerger
impl OrMerger for AttributeSimilarityMerger<'_> {}

impl SampleMerger for AttributeSimilarityMerger<'_> {
    fn merge<'a>(&self, _node_id: usize, left: &Sample, right: &Sample) -> Sample {
        // debug_assert!(self.ext_ddnnf.are_configs_sorted(left.partial_configs.iter().collect()));
        // debug_assert!(self.ext_ddnnf.are_configs_sorted(left.complete_configs.iter().collect()));
        // debug_assert!(self.ext_ddnnf.are_configs_sorted(right.partial_configs.iter().collect()));
        // debug_assert!(self.ext_ddnnf.are_configs_sorted(right.complete_configs.iter().collect()));

        if left.is_empty() {
            return right.clone();
        } else if right.is_empty() {
            return left.clone();
        }

        let mut new_sample = Sample::new_from_samples(&[left, right]);
        let left_merged_sorted_configs = self.ext_ddnnf.merge_sorted_configs(
            left.partial_configs.iter().collect(),
            left.complete_configs.iter().collect(),
        );
        let right_merged_sorted_configs = self.ext_ddnnf.merge_sorted_configs(
            right.partial_configs.iter().collect(),
            right.complete_configs.iter().collect(),
        );
        let candidates = self
            .ext_ddnnf
            .merge_sorted_configs(left_merged_sorted_configs, right_merged_sorted_configs);
        // let candidates = left
        //     .iter()
        //     .chain(right.iter())
        //     .sorted_by_cached_key(|config|
        //         FloatOrd::from(self.ext_ddnnf.get_average_objective_fn_val_of_config(config))
        //     )
        //     .rev();

        for candidate in candidates {
            if new_sample.is_t_wise_covered(candidate, self.t) {
                continue;
            }

            new_sample.add(candidate.clone())
        }

        // debug_assert!(self.ext_ddnnf.are_configs_sorted(new_sample.partial_configs.iter().collect()));
        // debug_assert!(self.ext_ddnnf.are_configs_sorted(new_sample.complete_configs.iter().collect()));

        new_sample
    }

    // For an or node, all samples have to be void for the resulting sample to also be void.
    fn is_void(&self, samples: &[&SamplingResult]) -> bool {
        samples
            .iter()
            .all(|result| matches!(result, SamplingResult::Void))
    }
}

#[cfg(test)]
mod test {}
