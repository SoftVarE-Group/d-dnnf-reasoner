use crate::sampler::data_structure::{Config, Sample};

pub mod similarity_merger;
pub mod zipping_merger;

pub trait SampleMerger {
    /// Creates a new sample by merging two samples.
    /// The merging follows the behaviour defined by the merger.
    fn merge(&self, node_id: usize, left: &Sample, right: &Sample) -> Sample;

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
    ) -> Sample {
        self.merge(node_id, &left, right)
    }

    /// Creates a new sample by merging all given samples.
    /// The merging follows the behaviour defined by the merger.
    /// Returns [Sample::empty] if the given slice is empty.
    fn merge_all(&self, node_id: usize, samples: &[&Sample]) -> Sample {
        samples.iter().fold(Sample::empty(), |acc, &sample| {
            self.merge_in_place(node_id, acc, sample)
        })
    }
}

/// This is a marker trait that indicates that a [SampleMerger] is for merging the samples
/// in an AND node.
pub trait AndMerger: SampleMerger {}

/// This is a marker trait that indicates that a [SampleMerger] is for merging the samples
/// in an OR node.
pub trait OrMerger: SampleMerger {}

/// A simple [AndMerger] that just builds all valid configs
#[derive(Debug, Clone, Copy)]
pub struct DummyAndMerger {}

impl SampleMerger for DummyAndMerger {
    fn merge(&self, _node_id: usize, left: &Sample, right: &Sample) -> Sample {
        if left.is_empty() {
            return right.clone();
        } else if right.is_empty() {
            return left.clone();
        }

        let mut sample = Sample::new_from_samples(&[left, right]);

        for left_part in left.iter() {
            for right_part in right.iter() {
                let new_config = Config::from_disjoint(left_part, right_part);
                sample.add_complete(new_config);
            }
        }

        sample
    }
}

impl AndMerger for DummyAndMerger {}

/// A simple [OrMerger] that just builds all valid configs
#[derive(Debug, Clone, Copy)]
pub struct DummyOrMerger {}

impl SampleMerger for DummyOrMerger {
    fn merge(&self, _node_id: usize, left: &Sample, right: &Sample) -> Sample {
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
