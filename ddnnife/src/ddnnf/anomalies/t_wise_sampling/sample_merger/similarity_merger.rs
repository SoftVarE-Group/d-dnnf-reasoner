use super::super::t_iterator::TInteractionIter;
use super::super::SamplingResult;
use super::{Config, Sample};
use super::{OrMerger, SampleMerger};
use crate::util::rng;
use rand::prelude::SliceRandom;
use std::cmp::{min, Ordering};
use std::collections::HashSet;
use streaming_iterator::StreamingIterator;

#[derive(Debug, Copy, Clone)]
pub struct SimilarityMerger {
    pub t: usize,
}

// Mark SimilarityMerger as an OrMerger
impl OrMerger for SimilarityMerger {}

impl SampleMerger for SimilarityMerger {
    fn merge<'a>(&self, _node_id: usize, left: &Sample, right: &Sample) -> Sample {
        if left.is_empty() {
            return right.clone();
        } else if right.is_empty() {
            return left.clone();
        }

        // init sample
        let mut new_sample = Sample::new_from_samples(&[left, right]);

        // init candidates
        let mut candidates: Vec<Candidate> = left
            .iter()
            .chain(right.iter())
            .map(Candidate::new)
            .collect();

        // (randomly) pick first candidate
        let next = candidates.pop()
            .expect("There should be at least one candidate because we checked that both samples are not empty");

        candidates.iter_mut().for_each(|c| c.update(&next.literals));
        new_sample.add(next.config.clone());

        while let Some(next) = candidates
            .iter()
            .enumerate()
            .max_by_key(snd)
            .map(|(index, _)| index)
        {
            let next = candidates.swap_remove(next);
            if next.is_t_wise_covered_by(&new_sample, self.t) {
                continue;
            }

            new_sample.add(next.config.clone());

            candidates.iter_mut().for_each(|c| c.update(&next.literals));
        }
        new_sample
    }

    // For an or node, all samples have to be void for the resulting sample to also be void.
    fn is_void(&self, samples: &[&SamplingResult]) -> bool {
        samples
            .iter()
            .all(|result| matches!(result, SamplingResult::Void))
    }
}

fn snd<'a>((_, candidate): &(usize, &'a Candidate<'_>)) -> &'a Candidate<'a> {
    candidate
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Candidate<'a> {
    config: &'a Config,
    literals: HashSet<i32>,
    max_intersect: usize,
    total_intersect: usize,
}

impl PartialOrd<Self> for Candidate<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        let compare = (self.total_intersect * self.literals.len())
            .cmp(&(other.total_intersect * other.literals.len()));

        if compare.is_eq() {
            (self.max_intersect * self.literals.len())
                .cmp(&(other.max_intersect * other.literals.len()))
        } else {
            compare
        }
    }
}

impl<'a> Candidate<'a> {
    fn new(config: &'a Config) -> Self {
        let literals: HashSet<i32> = config.get_decided_literals().collect();
        debug_assert!(!literals.contains(&0));
        debug_assert!(!literals.is_empty());
        Self {
            literals,
            max_intersect: 0,
            config,
            total_intersect: 0,
        }
    }

    fn update(&mut self, other_literals: &HashSet<i32>) {
        let intersect = self.literals.intersection(other_literals).count();

        self.total_intersect += intersect;

        if intersect > self.max_intersect {
            self.max_intersect = intersect;
        }
    }

    fn is_t_wise_covered_by(&self, sample: &Sample, t: usize) -> bool {
        if self.max_intersect == self.literals.len() {
            return true;
        }

        /*
        If the largest intersect between the candidate and the samples configs is smaller than
        t, then no interactions of the candidate are covered by the sample.
        Example:
        Sample = [[1,2],[1,-2]], Candidate = [-1,2], t = 2
        Max intersect is [2] and has len < t.
        => No interaction of the candidate is covered.
         */
        if self.literals.len() >= t && self.max_intersect < t {
            return false;
        }

        let mut literals: Vec<i32> = self.config.get_decided_literals().collect();
        literals.shuffle(&mut rng());
        debug_assert!(!literals.contains(&0));

        TInteractionIter::new(&literals, min(t, literals.len()))
            .all(|interaction| sample.covers(interaction))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_similarity_merger() {
        let merger = SimilarityMerger { t: 2 };

        let left = Sample::new_from_configs(vec![Config::from(&[1], 1)]);
        let right = Sample::new_from_configs(vec![Config::from(&[1], 1)]);
        let merged = merger.merge(0, &left, &right);
        assert_eq!(
            merged,
            Sample::new_from_configs(vec![Config::from(&[1], 1)])
        );
    }

    #[test]
    fn test_is_t_wise_covered() {
        let number_of_variables = 4;
        let candidate_config = Config::from(&[1, 2, 3, 4], number_of_variables);
        let mut candidate = Candidate::new(&candidate_config);

        let sample = Sample::new_from_configs(vec![
            Config::from(&[1, 2, 3], number_of_variables),
            Config::from(&[1, 4], number_of_variables),
            Config::from(&[2, 4], number_of_variables),
            Config::from(&[3, 4], number_of_variables),
        ]);

        sample
            .iter()
            .for_each(|c| candidate.update(&c.get_decided_literals().collect()));

        assert!(candidate.is_t_wise_covered_by(&sample, 2));

        let mut candidate = Candidate::new(&candidate_config);
        let sample = Sample::new_from_configs(vec![
            Config::from(&[1, 2, 3], number_of_variables),
            Config::from(&[1, 4], number_of_variables),
            Config::from(&[2, 4], number_of_variables),
        ]);

        sample
            .iter()
            .for_each(|c| candidate.update(&c.get_decided_literals().collect()));

        assert!(!candidate.is_t_wise_covered_by(&sample, 2));
    }
}
