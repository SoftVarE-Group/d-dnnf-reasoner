use rand::prelude::{SliceRandom, StdRng};
use rand::SeedableRng;
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::{io, iter};
use streaming_iterator::StreamingIterator;

use crate::data_structure::NodeType::{And, False, Literal, Or, True};
use crate::sampler::covering_strategies::cover_with_caching;
use crate::sampler::data_structure::Sample;
use crate::sampler::iterator::TInteractionIter;
use crate::sampler::sample_merger::similarity_merger::SimilarityMerger;
use crate::sampler::sample_merger::zipping_merger::ZippingMerger;
use crate::sampler::sample_merger::{AndMerger, OrMerger};
use crate::sampler::sat_solver::SatSolver;
use crate::sampler::SamplingResult::ResultWithSample;
use crate::Ddnnf;

pub mod covering_strategies;
pub mod data_structure;
pub mod iterator;
pub mod sample_merger;
pub mod sat_solver;

struct TWiseSampler<'a, A: AndMerger, O: OrMerger> {
    ddnnf: &'a Ddnnf,
    /// Map that holds the [SamplingResult]s for the nodes.
    partial_samples: HashMap<usize, SamplingResult>,
    and_merger: A,
    or_merger: O,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplingResult {
    True,
    False,
    ResultWithSample(Sample),
}

impl SamplingResult {
    pub fn get_sample(&self) -> Option<&Sample> {
        if let ResultWithSample(sample) = self {
            Some(sample)
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        match self {
            SamplingResult::True | SamplingResult::False => 0,
            ResultWithSample(sample) => sample.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            SamplingResult::True | SamplingResult::False => true,
            ResultWithSample(sample) => sample.is_empty(),
        }
    }
}

const EXPECT_SAMPLE: &str =
    "children should have a sampling result when sampling their parent";

impl<'a, A: AndMerger, O: OrMerger> TWiseSampler<'a, A, O> {
    /// Constructs a new sampler
    fn new(ddnnf: &'a Ddnnf, and_merger: A, or_merger: O) -> Self {
        Self {
            ddnnf,
            partial_samples: HashMap::with_capacity(ddnnf.number_of_nodes),
            and_merger,
            or_merger,
        }
    }

    /// Generates a sample for the sub-graph rooted at the given node. If the node is an AND or an
    /// OR node, then it is assumed that all direct children of the node already have a sample.
    /// The caller has to make sure that this is the case (usually by calling this method for the
    /// children first).
    ///
    /// # Panics
    /// Panics if one child does not have a [SamplingResult] in [TWiseSampler::partial_samples].
    fn make_partial_sample(
        &self,
        node_id: usize,
        rng: &mut StdRng,
    ) -> SamplingResult {
        let node = self.ddnnf.nodes.get(node_id).expect("Node does not exist!");

        match &node.ntype {
            Literal { literal } => ResultWithSample(Sample::from_literal(
                *literal,
                self.ddnnf.number_of_variables as usize,
            )),
            And { children } => {
                self.sample_and(node_id, self.get_child_results(children), rng)
            }
            Or { children } => {
                self.sample_or(node_id, self.get_child_results(children), rng)
            }
            True => SamplingResult::True,
            False => SamplingResult::False,
        }
    }

    /// Returns a vec with references to [SamplingResult]s for the given children
    fn get_child_results(&self, children: &[usize]) -> Vec<&SamplingResult> {
        children
            .iter()
            .map(|child| self.partial_samples.get(child).expect(EXPECT_SAMPLE))
            .collect()
    }

    /// Merges the samples of the given children by assuming they are the children of a
    /// AND node.
    fn sample_and(
        &self,
        node_id: usize,
        child_results: Vec<&SamplingResult>,
        rng: &mut StdRng,
    ) -> SamplingResult {
        if child_results
            .iter()
            .any(|result| matches!(result, SamplingResult::False))
        {
            return SamplingResult::False;
        }

        let child_samples: Vec<&Sample> = child_results
            .into_iter()
            .filter_map(SamplingResult::get_sample)
            .collect();

        let sample = self.and_merger.merge_all(node_id, &child_samples, rng);
        if sample.is_empty() {
            SamplingResult::True
        } else {
            ResultWithSample(sample)
        }
    }

    /// Merges the samples of the given children by assuming they are the children of a
    /// OR node.
    fn sample_or(
        &self,
        node_id: usize,
        child_results: Vec<&SamplingResult>,
        rng: &mut StdRng,
    ) -> SamplingResult {
        if child_results
            .iter()
            .all(|result| matches!(result, SamplingResult::False))
        {
            return SamplingResult::False;
        }

        let child_samples: Vec<&Sample> = child_results
            .into_iter()
            .filter_map(SamplingResult::get_sample)
            .collect();

        let sample = self.or_merger.merge_all(node_id, &child_samples, rng);
        if sample.is_empty() {
            SamplingResult::True
        } else {
            ResultWithSample(sample)
        }
    }

    fn complete_partial_configs(
        &self,
        sample: &mut Sample,
        t: usize,
        sat_solver: &SatSolver,
    ) {
        let vars: Vec<i32> =
            (1..=self.ddnnf.number_of_variables as i32).collect();
        for config in sample.partial_configs.iter_mut() {
            for &var in vars.iter() {
                if config.contains(var) || config.contains(-var) {
                    continue;
                }

                config.update_sat_state(sat_solver, t);
                let sat_state = config.get_sat_state().expect(
                    "sat state should exist after calling update_sat_state()",
                );

                if sat_solver.is_sat_cached(&[var], sat_state) {
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
    sat_solver: &SatSolver,
    rng: &mut StdRng,
) -> Sample {
    if sample.is_empty() {
        return sample;
    }

    let t = min(sample.get_vars().len(), t);
    let (ranks, avg_rank) = calc_stats(&sample, t);

    let (mut new_sample, literals_to_resample) =
        trim_sample(&sample, &ranks, avg_rank);

    let mut literals_to_resample: Vec<i32> =
        literals_to_resample.into_iter().collect();
    literals_to_resample.sort_unstable();
    literals_to_resample.shuffle(rng);

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
fn trim_sample(
    sample: &Sample,
    ranks: &[f64],
    avg_rank: f64,
) -> (Sample, HashSet<i32>) {
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
        if let Some(conf_index) = find_unique_covering_conf(sample, interaction)
        {
            unique_coverage[conf_index] += 1;
        }
    }

    let mut ranks = vec![0.0; sample.len()];
    let mut sum: f64 = 0.0;

    for (index, config) in sample.iter().enumerate() {
        let config_size = config.get_decided_literals().count();
        ranks[index] =
            unique_coverage[index] as f64 / config_size.pow(t as u32) as f64;
        sum += ranks[index];
    }

    let avg_rank = sum / sample.len() as f64;
    (ranks, avg_rank)
}

#[inline]
fn find_unique_covering_conf(
    sample: &Sample,
    interaction: &[i32],
) -> Option<usize> {
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

pub fn sample_t_wise(ddnnf: &Ddnnf, t: usize) -> SamplingResult {
    let sat_solver = SatSolver::new(ddnnf);
    let and_merger = ZippingMerger {
        t,
        sat_solver: &sat_solver,
        ddnnf,
    };
    let or_merger = SimilarityMerger { t };
    let mut rng = StdRng::seed_from_u64(42);
    let mut sampler = TWiseSampler::new(ddnnf, and_merger, or_merger);

    for node_id in 0..sampler.ddnnf.number_of_nodes {
        let partial_sample = sampler.make_partial_sample(node_id, &mut rng);
        sampler.partial_samples.insert(node_id, partial_sample);
    }

    let root_id = sampler.ddnnf.number_of_nodes - 1;

    let sampling_result = sampler
        .partial_samples
        .remove(&root_id)
        .expect("Root sample does not exist!");

    if let ResultWithSample(mut sample) = sampling_result {
        sample = trim_and_resample(
            root_id,
            sample,
            t,
            ddnnf.number_of_variables as usize,
            &sat_solver,
            &mut rng,
        );
        sampler.complete_partial_configs(&mut sample, t, &sat_solver);
        ResultWithSample(sample)
    } else {
        sampling_result
    }
}

pub fn save_sample_to_file(
    sampling_result: &SamplingResult,
    number_of_variables: u32,
    file_path: &str,
) -> io::Result<()> {
    let mut wtr = csv::Writer::from_path(file_path)?;

    // write the header - it looks like
    // Configuration;1;2;3
    let names = (1..=number_of_variables).map(|name| name.to_string());
    let header = iter::once(String::from("Configuration")).chain(names);
    wtr.write_record(header)?;

    match sampling_result {
        /*
        Writing "true" and "false" to the file does not really fit the format of the file but we
        want to somehow distinguish between true and false sampling results.
        True means that the feature model contains no variables and therefore an empty sample
        covers all t-wise interactions.
        False means that the feature model is void.
        */
        SamplingResult::True => wtr.write_record(iter::once("true"))?,
        SamplingResult::False => wtr.write_record(iter::once("false"))?,
        ResultWithSample(sample) => {
            for (index, config) in sample.iter().enumerate() {
                /*
                write the index of the config then a 0 (deselected) or 1 (selected) for each feature
                 */
                let literals = config.get_literals().iter().map(|literal| {
                    if *literal < 0 {
                        "0"
                    } else {
                        "1"
                    }
                });
                let index = index.to_string();
                let record = iter::once(index.as_str()).chain(literals);
                wtr.write_record(record)?;
            }
        }
    }

    wtr.flush()
}
