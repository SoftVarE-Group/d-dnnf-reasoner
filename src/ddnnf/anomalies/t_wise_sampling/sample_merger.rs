use crate::ddnnf::anomalies::t_wise_sampling::data_structure::{Config, Sample};
use crate::Ddnnf;
use rand::prelude::StdRng;

pub mod similarity_merger;
pub mod zipping_merger;

pub(super) trait SampleMerger {
    /// Creates a new sample by merging two samples.
    /// The merging follows the behaviour defined by the merger.
    fn merge(&self, node_id: usize, left: &Sample, right: &Sample, rng: &mut StdRng) -> Sample;

    /// Creates a new sample by merging two samples.
    /// The merging follows the behaviour defined by the merger.
    ///
    /// This method only works in place if the used merger actually overrides this method.
    /// The default implementation calls [Self::merge()] and is therefore not in place.
    fn merge_in_place(
        &self,
        node_id: usize,
        left: Sample,
        right: &Sample,
        rng: &mut StdRng,
    ) -> Sample {
        self.merge(node_id, &left, right, rng)
    }

    /// Creates a new sample by merging all given samples.
    /// The merging follows the behaviour defined by the merger.
    /// Returns [Sample::empty] if the given slice is empty.
    fn merge_all(&self, node_id: usize, samples: &[&Sample], rng: &mut StdRng) -> Sample {
        samples.iter().fold(Sample::default(), |acc, &sample| {
            self.merge_in_place(node_id, acc, sample, rng)
        })
    }
}

/// This is a marker trait that indicates that a [SampleMerger] is for merging the samples
/// in an AND node.
pub(super) trait AndMerger: SampleMerger {}

/// This is a marker trait that indicates that a [SampleMerger] is for merging the samples
/// in an OR node.
pub(super) trait OrMerger: SampleMerger {}

/// A simple [AndMerger] that just builds all valid configs
#[derive(Debug, Clone, Copy)]
pub(super) struct DummyAndMerger<'a> {
    ddnnf: &'a Ddnnf,
}

impl SampleMerger for DummyAndMerger<'_> {
    fn merge(&self, _node_id: usize, left: &Sample, right: &Sample, _rng: &mut StdRng) -> Sample {
        if left.is_empty() {
            return right.clone();
        } else if right.is_empty() {
            return left.clone();
        }

        let mut sample = Sample::new_from_samples(&[left, right]);

        for left_part in left.iter() {
            for right_part in right.iter() {
                let new_config = Config::from_disjoint(
                    left_part,
                    right_part,
                    self.ddnnf.number_of_variables as usize,
                );
                sample.add_complete(new_config);
            }
        }

        sample
    }
}

impl AndMerger for DummyAndMerger<'_> {}

/// A simple [OrMerger] that just builds all valid configs
#[derive(Debug, Clone, Copy)]
pub(super) struct DummyOrMerger {}

impl SampleMerger for DummyOrMerger {
    fn merge(&self, _node_id: usize, left: &Sample, right: &Sample, _rng: &mut StdRng) -> Sample {
        if left.is_empty() {
            return right.clone();
        } else if right.is_empty() {
            return left.clone();
        }

        let mut sample = Sample::new_from_samples(&[left, right]);

        left.iter()
            .cloned()
            .for_each(|config| sample.add_complete(config));

        right
            .iter()
            .cloned()
            .for_each(|config| sample.add_complete(config));

        sample
    }
}

impl OrMerger for DummyOrMerger {}
