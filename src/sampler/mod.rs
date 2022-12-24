use std::collections::HashMap;

use crate::data_structure::NodeType::{And, False, Literal, Or, True};
use crate::sampler::data_structure::Sample;
use crate::sampler::sample_merger::zipping_merger::ZippingMerger;
use crate::sampler::sample_merger::{AndMerger, DummyOrMerger, OrMerger};
use crate::sampler::sat_solver::SatSolver;
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
        if let SamplingResult::ResultWithSample(sample) = self {
            Some(sample)
        } else {
            None
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
    fn make_partial_sample(&self, node_id: usize) -> SamplingResult {
        let node = self.ddnnf.nodes.get(node_id).expect("Node does not exist!");

        match &node.ntype {
            Literal { literal } => SamplingResult::ResultWithSample(
                Sample::from_literal(*literal),
            ),
            And { children } => {
                self.sample_and(node_id, self.get_child_results(children))
            }
            Or { children } => {
                self.sample_or(node_id, self.get_child_results(children))
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

        let sample = self.and_merger.merge_all(node_id, &child_samples);
        if sample.is_empty() {
            SamplingResult::True
        } else {
            SamplingResult::ResultWithSample(sample)
        }
    }

    /// Merges the samples of the given children by assuming they are the children of a
    /// OR node.
    fn sample_or(
        &self,
        node_id: usize,
        child_results: Vec<&SamplingResult>,
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

        let sample = self.or_merger.merge_all(node_id, &child_samples);
        if sample.is_empty() {
            SamplingResult::True
        } else {
            SamplingResult::ResultWithSample(sample)
        }
    }
}

pub fn sample_t_wise(ddnnf: &Ddnnf, t: usize) -> SamplingResult {
    let sat_solver = SatSolver::new(ddnnf);
    let and_merger = ZippingMerger {
        t,
        sat_solver: &sat_solver,
    };
    let or_merger = DummyOrMerger {};
    let mut sampler = TWiseSampler::new(ddnnf, and_merger, or_merger);

    for node_id in 0..sampler.ddnnf.number_of_nodes {
        let partial_sample = sampler.make_partial_sample(node_id);
        sampler.partial_samples.insert(node_id, partial_sample);
    }

    let root_id = sampler.ddnnf.number_of_nodes - 1;

    sampler
        .partial_samples
        .remove(&root_id)
        .expect("Root sample does not exist!")
}
