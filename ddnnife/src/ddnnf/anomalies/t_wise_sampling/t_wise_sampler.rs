use super::covering_strategies::cover_with_caching;
use super::sample_merger::{AndMerger, OrMerger, SampleMerger};
use super::t_iterator::TInteractionIter;
use super::{Sample, SamplingResult, SatWrapper};
use crate::util::rng;
use crate::Ddnnf;
use crate::NodeType;
use rand::prelude::SliceRandom;
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use streaming_iterator::StreamingIterator;

pub struct TWiseSampler<'a, A: AndMerger, O: OrMerger> {
    /// The d-DNNF to sample.
    ddnnf: &'a Ddnnf,
    /// Map that holds the [SamplingResult]s for the nodes.
    partial_samples: HashMap<usize, SamplingResult>,
    /// The merger for and nodes.
    and_merger: A,
    /// The merger for or nodes.
    or_merger: O,
}

impl<'a, A: AndMerger, O: OrMerger> TWiseSampler<'a, A, O> {
    /// Constructs a new sampler.
    pub fn new(ddnnf: &'a Ddnnf, and_merger: A, or_merger: O) -> Self {
        Self {
            ddnnf,
            partial_samples: HashMap::with_capacity(ddnnf.nodes.len()),
            and_merger,
            or_merger,
        }
    }

    pub fn sample(&mut self, t: usize) -> SamplingResult {
        let sat_solver = SatWrapper::new(self.ddnnf);

        // Sample each node and keep the result as a partial sample.
        for node_id in 0..self.ddnnf.nodes.len() {
            let partial_sample = self.partial_sample(node_id);
            self.partial_samples.insert(node_id, partial_sample);
        }

        let root_id = self.ddnnf.nodes.len() - 1;

        // Extract the resulting (root node) sample for further processing.
        let result = self
            .partial_samples
            .remove(&root_id)
            .expect("Root sample does not exist!");

        // Trim and resample as the finishing step (if there is anything to do).
        if let SamplingResult::ResultWithSample(mut sample) = result {
            sample = trim_and_resample(
                root_id,
                sample,
                t,
                self.ddnnf.number_of_variables as usize,
                &sat_solver,
            );

            self.complete_partial_configs(&mut sample, root_id, &sat_solver);
            return sample.into();
        }

        result
    }

    /// Generates a sample for the sub-graph rooted at the given node.
    ///
    /// If the node is an and or an or node, then it is assumed that all direct children of the node already have a sample.
    /// The caller has to make sure that this is the case (usually by calling this method for the children first).
    ///
    /// # Panics
    /// Panics if one child does not have a [SamplingResult] in [TWiseSampler::partial_samples].
    fn partial_sample(&mut self, node_id: usize) -> SamplingResult {
        let node = self.ddnnf.nodes.get(node_id).expect("Node does not exist!");

        match &node.ntype {
            NodeType::Literal { literal } => SamplingResult::ResultWithSample(
                Sample::from_literal(*literal, self.ddnnf.number_of_variables as usize),
            ),
            NodeType::And { children } => {
                let sample = self.sample_node(&self.and_merger, node_id, children);
                self.remove_unneeded(node_id, children);
                sample
            }
            NodeType::Or { children } => {
                let sample = self.sample_node(&self.or_merger, node_id, children);
                self.remove_unneeded(node_id, children);
                sample
            }
            NodeType::True => SamplingResult::Empty,
            NodeType::False => SamplingResult::Void,
        }
    }

    /// Merges the samples of the given children by using the specified sampler.
    fn sample_node<M: SampleMerger>(
        &self,
        sampler: &M,
        id: usize,
        children: &[usize],
    ) -> SamplingResult {
        // Get the samples of all child nodes.
        let children: Vec<&SamplingResult> = children
            .iter()
            .map(|child| {
                self.partial_samples
                    .get(child)
                    .expect("Samples of child node not present!")
            })
            .collect();

        // Check whether the set of child nodes short-circuits to a void sample.
        if sampler.is_void(&children) {
            return SamplingResult::Void;
        }

        // Only keep samples with a result.
        let samples: Vec<&Sample> = children
            .iter()
            .filter_map(|sample: &&SamplingResult| sample.optional())
            .collect();

        // Merge the samples using the specified sampler.
        sampler.merge_all(id, &samples).into()
    }

    /// Removes samples that are no longer needed to reduce memory usage.
    ///
    /// A sample is no longer needed if all parent nodes have a sample.
    fn remove_unneeded(&mut self, node_id: usize, children: &[usize]) {
        // Of all children ...
        children
            .iter()
            // ... find the ones which have all parents processed ...
            .filter(|&&id| {
                let node = self.ddnnf.nodes.get(id).expect("Node does not exist!");
                node.parents.iter().all(|&parent| parent <= node_id)
            })
            // ... and remove those.
            .for_each(|id| {
                self.partial_samples
                    .remove(id)
                    .expect("Sample does not exist!");
            });
    }

    fn complete_partial_configs(&self, sample: &mut Sample, root: usize, sat_solver: &SatWrapper) {
        let vars: Vec<i32> = (1..=self.ddnnf.number_of_variables as i32).collect();
        for config in sample.partial_configs.iter_mut() {
            for &var in vars.iter() {
                if config.contains(var) || config.contains(-var) {
                    continue;
                }

                config.update_sat_state(sat_solver, root);

                // clone sat state so that we don't change the state that is cached in the config
                let mut sat_state = config
                    .get_sat_state()
                    .cloned()
                    .expect("sat state should exist after calling update_sat_state()");

                if sat_solver.is_sat_cached(&[var], &mut sat_state) {
                    config.add(var);
                } else {
                    config.add(-var);
                }
            }
        }

        debug_assert!(sample
            .iter()
            .all(|config| !config.get_literals().contains(&0)));
    }
}

#[inline]
fn trim_and_resample(
    node_id: usize,
    sample: Sample,
    t: usize,
    number_of_variables: usize,
    sat_solver: &SatWrapper,
) -> Sample {
    if sample.is_empty() {
        return sample;
    }

    let t = min(sample.get_vars().len(), t);
    let (ranks, avg_rank) = calc_stats(&sample, t);

    let (mut new_sample, literals_to_resample) = trim_sample(&sample, &ranks, avg_rank);

    let mut literals_to_resample: Vec<i32> = literals_to_resample.into_iter().collect();
    literals_to_resample.sort_unstable();
    literals_to_resample.shuffle(&mut rng());

    let mut iter = TInteractionIter::new(&literals_to_resample, t);
    while let Some(interaction) = iter.next() {
        cover_with_caching(
            &mut new_sample,
            interaction,
            sat_solver,
            node_id,
            number_of_variables,
        );
    }

    if new_sample.len() < sample.len() {
        new_sample
    } else {
        sample
    }
}

#[inline]
fn trim_sample(sample: &Sample, ranks: &[f64], avg_rank: f64) -> (Sample, HashSet<i32>) {
    let mut literals_to_resample: HashSet<i32> = HashSet::new();
    let mut new_sample = Sample::new_from_samples(&[sample]);
    let complete_len = sample.complete_configs.len();

    for (index, config) in sample.iter().enumerate() {
        if ranks[index] < avg_rank {
            literals_to_resample.extend(config.get_decided_literals());
        } else if index < complete_len {
            new_sample.add_complete(config.clone());
        } else {
            new_sample.add_partial(config.clone());
        }
    }
    (new_sample, literals_to_resample)
}

#[inline]
fn calc_stats(sample: &Sample, t: usize) -> (Vec<f64>, f64) {
    let mut unique_coverage = vec![0; sample.len()];
    let mut iter = TInteractionIter::new(sample.get_literals(), t);
    while let Some(interaction) = iter.next() {
        if let Some(conf_index) = find_unique_covering_conf(sample, interaction) {
            unique_coverage[conf_index] += 1;
        }
    }

    let mut ranks = vec![0.0; sample.len()];
    let mut sum: f64 = 0.0;

    for (index, config) in sample.iter().enumerate() {
        let config_size = config.get_decided_literals().count();
        ranks[index] = unique_coverage[index] as f64 / config_size.pow(t as u32) as f64;
        sum += ranks[index];
    }

    let avg_rank = sum / sample.len() as f64;
    (ranks, avg_rank)
}

#[inline]
fn find_unique_covering_conf(sample: &Sample, interaction: &[i32]) -> Option<usize> {
    let mut result = None;

    for (index, config) in sample.iter().enumerate() {
        if config.covers(interaction) {
            if result.is_none() {
                result = Some(index);
            } else {
                return None;
            }
        }
    }

    result
}
