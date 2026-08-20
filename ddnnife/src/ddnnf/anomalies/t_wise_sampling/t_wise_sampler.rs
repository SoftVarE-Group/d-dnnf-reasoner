use super::covering_strategies::cover_with_caching;
use super::sample_merger::{AndMerger, OrMerger, SampleMerger};
use super::t_iterator::TInteractionIter;
use super::{Sample, SamplingResult, SatWrapper};
use crate::NodeType;
use crate::ddnnf::anomalies::t_wise_sampling::Config;
use crate::ddnnf::extended_ddnnf::ExtendedDdnnf;
use crate::int_hash::{self, IntMap, IntSet};
use crate::rand::rng;
use crate::{Ddnnf, DdnnfKind};
use itertools::Itertools;
use rand::prelude::SliceRandom;
use std::cmp::min;
use std::collections::HashSet;
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
        self.complete_configs.reserve(self.partial_configs.len());

        let vars: Vec<i32> = (1..=number_of_variables).collect();
        for mut config in self.partial_configs.drain(..self.partial_configs.len()) {
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

            self.complete_configs.push(config);
        }

        self.partial_configs.shrink_to_fit();

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
        // Ensure that the preset is always part of the final sample,
        // even if it is empty.
        if self.is_empty() {
            *self = preset.clone();
            return;
        }

        self.extend(preset.clone());

        let t = min(self.get_vars().len(), t);

        // Trim the sample and collect the literals to resample.
        let (mut new_sample, literals_to_resample) = trim_sample(self, t, preset);

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
        if new_sample.len() <= self.len() {
            *self = new_sample;
        }
    }
}

/// Removes those configs from the given sample that rank score the average.
/// Does not trim configs that are part of the preset.
///
/// Returns the remaining sample as well as the literals to resample.
fn trim_sample(sample: &Sample, t: usize, preset: &Sample) -> (Sample, IntSet<i32>) {
    let mut literals_to_resample: IntSet<i32> = IntSet::default();
    let mut new_sample = Sample::new_from_samples(&[sample]);
    let complete_len = sample.complete_configs.len();

    let (scores, average) = sample.scores(t, preset);

    let preset: HashSet<Config> = preset.iter().cloned().collect();
    for (index, config) in sample.iter().enumerate() {
        // Trim those configs that score below the average and are not part of the preset.
        if scores[index] < average && !preset.contains(config) {
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
    /// Calculates the scores of all configurations to be used for trimming and resampling.
    ///
    /// Currently calculates two scores: Unique interaction coverage and preset interaction coverage.
    fn scores(&self, t: usize, _preset: &Sample) -> (Vec<f64>, f64) {
        // With an empty preset, do not calculate the preset score.
        let scores: Vec<f64> = self.unique_coverage_scores(t).collect();

        // Calculate the average score over all configurations.
        let average = scores.iter().sum::<f64>() / scores.len() as f64;
        (scores, average)
    }

    /// Calculates the preset score of all configs.
    ///
    /// The preset score is higher for those configs that cover many interactions that are not part of the preset,
    /// relative to their size.
    fn _preset_scores(&self, t: usize, preset: &Sample) -> impl Iterator<Item = f64> {
        // Calculate how many non-preset interactions each config covers.
        let mut preset_coverage = vec![0; self.len()];

        // For each interaction ...
        TInteractionIter::new(self.get_literals(), min(self.get_literals().len(), t))
            // ... that is not covered by the preset ...
            .filter(|interaction| !preset.covers(interaction))
            // ... find and mark those configs that cover this interaction.
            .for_each(|interaction| {
                self.iter()
                    .enumerate()
                    .filter(|(_, config)| config.covers(interaction))
                    .for_each(|(index, _)| preset_coverage[index] += 1)
            });

        self.iter().enumerate().map(move |(index, config)| {
            preset_coverage[index] as f64 / config.n_decided_literals.pow(t as u32) as f64
        })
    }

    /// Calculates the unique coverage score of each configuration.
    ///
    /// The unique coverage score is higher for those configs that cover many unique interactions,
    /// relative to their size.
    fn unique_coverage_scores(&self, t: usize) -> impl Iterator<Item = f64> {
        // Calculate how many unique interactions each configuration covers.
        let mut unique_coverage = vec![0; self.len()];

        // For each interaction ...
        TInteractionIter::new(self.get_literals(), min(self.get_literals().len(), t))
            // ... check whether there is a config uniquely covering this interaction ...
            .filter_map(|interaction| self.find_unique_covering_conf(interaction))
            // ... and in case there is, mark the corresponding config as such.
            .for_each(|config| unique_coverage[*config] += 1);

        // Calculate the rank of each configuration based on its unique coverage.
        self.iter().enumerate().map(move |(index, config)| {
            unique_coverage[index] as f64 / config.n_decided_literals.pow(t as u32) as f64
        })
    }

    /// Finds the index of the configuration that uniquely covers the given interaction, if such a configuration exists.
    ///
    /// Returns `None` if no or more than one configurations cover the given interaction.
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
