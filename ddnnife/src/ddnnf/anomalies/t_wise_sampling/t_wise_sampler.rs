use super::covering_strategies::cover_with_caching;
use super::sample_merger::{AndMerger, OrMerger, SampleMerger};
use super::t_iterator::TInteractionIter;
use super::{Sample, SamplingResult, SatWrapper};
use crate::NodeType;
use crate::ddnnf::extended_ddnnf::ExtendedDdnnf;
use crate::int_hash::{self, IntMap, IntSet};
use crate::rand::rng;
use crate::{Ddnnf, DdnnfKind};
use itertools::Itertools;
use rand::prelude::SliceRandom;
use std::cmp::min;
use std::mem;
use streaming_iterator::StreamingIterator;

pub struct TWiseSampler<'a, 'l, 'p, A: AndMerger, O: OrMerger> {
    /// The d-DNNF to sample.
    pub(crate) ddnnf: &'a Ddnnf,
    /// Map that holds the [SamplingResult]s for the nodes.
    pub(crate) partial_samples: IntMap<usize, SamplingResult>,
    /// The set of literals to cover `t`-wise.
    ///
    /// Can be used to restrict the covering to a given set of literals or variables.
    /// If unset, all literals are covered.
    literals: Option<&'l IntSet<i32>>,
    preset: &'p Sample,
    /// The merger for and nodes.
    and_merger: A,
    /// The merger for or nodes.
    or_merger: O,
}

impl<'a, 'l, 'p, A: AndMerger, O: OrMerger> TWiseSampler<'a, 'l, 'p, A, O> {
    /// Constructs a new sampler.
    pub fn new(
        ddnnf: &'a Ddnnf,
        and_merger: A,
        or_merger: O,
        literals: Option<&'l IntSet<i32>>,
        preset: &'p Sample,
    ) -> Self {
        Self {
            ddnnf,
            partial_samples: int_hash::map_with_capacity(ddnnf.nodes.len()),
            literals,
            and_merger,
            or_merger,
            preset,
        }
    }

    pub fn sample(&mut self, t: usize) -> SamplingResult {
        match self.ddnnf.kind {
            DdnnfKind::Tautology => return SamplingResult::Empty,
            DdnnfKind::Contradiction => return SamplingResult::Void,
            _ => {}
        }

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
            sample.trim_and_resample(
                root_id,
                t,
                self.ddnnf.number_of_variables as usize,
                &sat_solver,
                self.literals,
                self.preset,
            );

            sample.complete_partial_configs(
                root_id,
                &sat_solver,
                self.ddnnf.number_of_variables as i32,
            );

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
    pub(crate) fn partial_sample(&mut self, node_id: usize) -> SamplingResult {
        match self.ddnnf.kind {
            DdnnfKind::Tautology => return SamplingResult::Empty,
            DdnnfKind::Contradiction => return SamplingResult::Void,
            _ => {}
        }

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
}

impl Sample {
    fn complete_partial_configs(
        &mut self,
        root: usize,
        sat_solver: &SatWrapper,
        number_of_variables: i32,
    ) {
        let vars: Vec<i32> = (1..=number_of_variables).collect();
        for config in self.partial_configs.iter_mut() {
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

        debug_assert!(
            self.iter()
                .all(|config| !config.get_literals().contains(&0))
        );
    }

    pub fn complete_partial_configs_optimal(&mut self, ext_ddnnf: &ExtendedDdnnf) {
        while let Some(config) = self.partial_configs.pop() {
            let literals = config.get_decided_literals().collect_vec();
            let completed_config = ext_ddnnf
                .calc_best_config(&literals[..])
                .expect("Config should be exist");

            debug_assert!(
                completed_config.config.get_n_decided_literals() == self.vars.len(),
                "{:?} != {:?}",
                completed_config.config.get_n_decided_literals(),
                self.vars.len()
            );

            self.add(completed_config.config);
        }
    }

    pub fn trim_and_resample(
        &mut self,
        node_id: usize,
        t: usize,
        number_of_variables: usize,
        sat_solver: &SatWrapper,
        literals: Option<&IntSet<i32>>,
        preset: &Sample,
    ) {
        if self.is_empty() {
            let _ = mem::replace(self, preset.clone());
            return;
        }

        let t = min(self.get_vars().len(), t);

        // Trim the sample before adding the preset configurations to ensure they are not removed.
        let (mut new_sample, literals_to_resample) = trim_sample(self, t);
        new_sample.extend(preset.clone());

        // Convert the set of literals to resample into a vector.
        // In case a restriction on the literals to cover is given, apply it during this conversion.
        let mut literals_to_resample: Vec<i32> = if let Some(literals) = literals {
            literals_to_resample
                .into_iter()
                .filter(|literal| literals.contains(literal))
                .collect()
        } else {
            literals_to_resample.into_iter().collect()
        };

        // Sort and then shuffle to allow for deterministic processing if enabled.
        literals_to_resample.sort_unstable();
        literals_to_resample.shuffle(&mut rng());

        let mut iter =
            TInteractionIter::new(&literals_to_resample, min(t, literals_to_resample.len()));
        while let Some(interaction) = iter.next() {
            cover_with_caching(
                &mut new_sample,
                interaction,
                sat_solver,
                node_id,
                number_of_variables,
            );
        }

        // Choose the smaller sample of the resampled or the original one.
        // Account for the preset configurations when considering the original as they were not added previously.
        if new_sample.len() <= self.len() + preset.len() {
            let _ = mem::replace(self, new_sample);
        } else {
            self.extend(preset.clone());
        }
    }
}

/// Removes those configs from the given sample that rank below the average.
///
/// Returns the remaining sample as well as the literals to resample.
fn trim_sample(sample: &Sample, t: usize) -> (Sample, IntSet<i32>) {
    let mut literals_to_resample: IntSet<i32> = IntSet::default();
    let mut new_sample = Sample::new_from_samples(&[sample]);
    let complete_len = sample.complete_configs.len();

    let (ranks, avg_rank) = sample.calc_ranks(t);

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

impl Sample {
    /// Calculates the ranks of each configuration and the average.
    fn calc_ranks(&self, t: usize) -> (Vec<f64>, f64) {
        let mut unique_coverage = vec![0; self.len()];
        let mut iter =
            TInteractionIter::new(self.get_literals(), min(self.get_literals().len(), t));
        while let Some(interaction) = iter.next() {
            if let Some(conf_index) = self.find_unique_covering_conf(interaction) {
                unique_coverage[conf_index] += 1;
            }
        }

        let mut ranks = vec![0.0; self.len()];
        let mut sum: f64 = 0.0;

        for (index, config) in self.iter().enumerate() {
            let config_size = config.n_decided_literals;
            ranks[index] = unique_coverage[index] as f64 / config_size.pow(t as u32) as f64;
            sum += ranks[index];
        }

        let avg_rank = sum / self.len() as f64;
        (ranks, avg_rank)
    }

    fn find_unique_covering_conf(&self, interaction: &[i32]) -> Option<usize> {
        let mut result = None;

        for (index, config) in self.iter().enumerate() {
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
}
