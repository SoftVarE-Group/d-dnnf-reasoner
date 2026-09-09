use std::ops::Index;

use crate::ddnnf::anomalies::t_wise_sampling::Sample;
use crate::int_hash::IntMap;
use bitvec::bitvec;
use bitvec::vec::BitVec;

/// Helper structure to find covering configuration for literals.
///
/// Constructs a map of literals to all configurations, indicating whether any given
/// configuration covers the respective literal.
pub struct CoverageMap {
    coverages: IntMap<i32, BitVec>,
    empty: BitVec,
}

impl Index<&i32> for CoverageMap {
    type Output = BitVec;

    fn index(&self, literal: &i32) -> &Self::Output {
        self.coverages.get(literal).unwrap_or(&self.empty)
    }
}

impl CoverageMap {
    /// Create a new coverage map for all literals in the given sample.
    pub fn new(sample: &Sample) -> Self {
        let mut coverages = IntMap::default();

        // Consider each configuration.
        sample.iter().enumerate().for_each(|(i, config)| {
            // For each literal in the configuration ...
            config.get_decided_literals().for_each(|literal| {
                // ... mark the configuration as covering this literal.
                let bitset = coverages.entry(literal).or_insert(bitvec![0; sample.len()]);
                bitset.set(i, true);
            })
        });

        Self {
            coverages,
            empty: bitvec![0; sample.len()],
        }
    }

    /// Computes the bitset representing the configurations covering the given interaction.
    pub fn coverage(&self, interaction: &[i32]) -> BitVec {
        if interaction.is_empty() {
            return BitVec::new();
        }

        // Bitwise AND between the coverage bitsets of all literals.
        interaction
            .iter()
            // Skip the first bitset as it is the fold base case.
            .skip(1)
            .fold(self[&interaction[0]].clone(), |covering, literal| {
                covering & &self[literal]
            })
    }

    /// Finds the index of the configuration that uniquely covers the given interaction, if such a configuration exists.
    ///
    /// Returns `None` if no or more than one configurations cover the given interaction.
    pub fn find_unqiuely_covering(&self, interaction: &[i32]) -> Option<usize> {
        let coverage = self.coverage(interaction);

        if coverage.count_ones() == 1 {
            return coverage.first_one();
        }

        None
    }
}
