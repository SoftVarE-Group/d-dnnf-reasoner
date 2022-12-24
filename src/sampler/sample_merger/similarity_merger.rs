use crate::sampler::data_structure::{Config, Sample};
use crate::sampler::iterator::t_wise_over;
use crate::sampler::sample_merger::{OrMerger, SampleMerger};

use std::collections::HashSet;
use std::iter::zip;

#[derive(Debug, Copy, Clone)]
pub struct SimilarityMerger {
    pub t: usize,
}

// Mark SimilarityMerger as an OrMerger
impl OrMerger for SimilarityMerger {}

impl SampleMerger for SimilarityMerger {
    fn merge(&self, _node_id: usize, left: &Sample, right: &Sample) -> Sample {
        if left.is_empty() {
            return right.clone();
        } else if right.is_empty() {
            return left.clone();
        }

        // init sample
        let mut new_sample = Sample::new_from_samples(&[left, right]);
        let number_of_vars = new_sample.get_vars().len();
        let mut sample_vector_sum = vec![0.0; number_of_vars];

        // init candidates
        let mut candidates: Vec<Candidate> = left
            .iter()
            .chain(right.iter())
            .map(|config| Candidate::new(config, number_of_vars))
            .collect();

        // (randomly) pick first candidate
        let next = candidates.pop()
            .expect("There should be at least one candidate because we checked that both samples are not empty");

        candidates.iter_mut().for_each(|c| c.update(&next.literals));
        update_sum(&mut sample_vector_sum, &next.vector);
        new_sample.add(next.config.clone());

        while let Some(next) =
            min_by_similarity(&candidates, &sample_vector_sum)
        {
            let next = candidates.swap_remove(next);

            if next.is_t_wise_covered_by(&new_sample, self.t) {
                continue;
            }

            new_sample.add(next.config.clone());

            candidates.iter_mut().for_each(|c| c.update(&next.literals));
            update_sum(&mut sample_vector_sum, &next.vector);
        }

        new_sample
    }
}

/// Update sum in place by adding v to it.
fn update_sum(sum: &mut [f64], v: &[f64]) {
    debug_assert_eq!(sum.len(), v.len());
    sum.iter_mut().zip(v.iter()).for_each(|(a, b)| *a += *b);
}

/// Find the candidate with the least similarity to the sample vector and return its index.
/// Returns None if candidates is empty.
fn min_by_similarity(
    candidates: &[Candidate],
    sample_vector_sum: &[f64],
) -> Option<usize> {
    candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            (index, similarity(sample_vector_sum, &candidate.vector))
        })
        .min_by(|left, right| {
            let (_, left_sim) = left;
            let (_, right_sim) = right;
            debug_assert!(left_sim.is_finite());
            debug_assert!(right_sim.is_finite());
            left_sim.total_cmp(right_sim)
        })
        .map(|(index, _)| index)
}

/// Calculate the cos of the angle between the two vectors.
fn similarity(u: &[f64], v: &[f64]) -> f64 {
    assert_eq!(u.len(), v.len());
    let angle = dot_product(u, v) / (magnitude(u) * magnitude(v));
    angle.cos()
}

/// Calculate the magnitude of a vector.
fn magnitude(v: &[f64]) -> f64 {
    v.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
}

/// Calculate the dot product of two vectors.
fn dot_product(u: &[f64], v: &[f64]) -> f64 {
    zip(u, v).map(|(x, y)| x * y).sum()
}

struct Candidate<'a> {
    config: &'a Config,
    literals: HashSet<i32>,
    min_diff: HashSet<i32>,
    vector: Vec<f64>,
}

impl<'a> Candidate<'a> {
    fn new(config: &'a Config, number_of_vars: usize) -> Self {
        let literals = config.get_literals();
        Self {
            vector: Candidate::calc_vector(literals, number_of_vars),
            literals: literals.iter().copied().collect(),
            min_diff: literals.iter().copied().collect(),
            config,
        }
    }

    fn calc_vector(literals: &[i32], number_of_vars: usize) -> Vec<f64> {
        let mut similarity_vector = vec![0.0; number_of_vars];
        for literal in literals {
            similarity_vector[literal.unsigned_abs() as usize - 1] =
                literal.signum() as f64;
        }
        similarity_vector
    }

    fn update(&mut self, other_literals: &HashSet<i32>) {
        let diff: HashSet<i32> =
            self.literals.difference(other_literals).copied().collect();

        if diff.len() < self.min_diff.len() {
            self.min_diff = diff;
        }
    }

    fn is_t_wise_covered_by(&self, sample: &Sample, t: usize) -> bool {
        if self.min_diff.is_empty() {
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
        let max_intersect = self.literals.len() - self.min_diff.len();
        if max_intersect < t {
            return false;
        }

        t_wise_over(&self.literals.iter().copied().collect::<Vec<i32>>(), t)
            // only check interactions that intersect (are not disjoint) with min_diff
            // because interactions that are disjoint with min_diff are already covered
            .filter(|interaction| {
                let interaction_set: HashSet<i32> =
                    interaction.iter().copied().collect();
                !self.min_diff.is_disjoint(&interaction_set)
            })
            .all(|interaction| sample.covers(&interaction))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_update_sum() {
        let mut sum = [1.0; 3];
        let v = [0.5; 3];

        update_sum(&mut sum, &v);

        for x in sum {
            let abs_difference = (x - 1.5).abs();
            assert!(abs_difference < 1e-10);
        }
    }

    #[test]
    fn test_similarity_merger() {
        let merger = SimilarityMerger { t: 2 };

        let left = Sample::new_from_configs(vec![Config::from(&[1])]);
        let right = Sample::new_from_configs(vec![Config::from(&[1])]);
        let merged = merger.merge(0, &left, &right);
        assert_eq!(merged, Sample::new_from_configs(vec![Config::from(&[1])]));
    }
}
