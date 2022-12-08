use core::slice::Iter;
use std::collections::HashSet;
use std::iter::Chain;

/// Represents a (partial) configuration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Config {
    /// A vector of selected features (positive values) and deselected features (negative values)
    literals: Vec<i32>,
}

/// Checks if this config covers the given interaction
impl Config {
    pub fn covers(&self, interaction: &[i32]) -> bool {
        interaction
            .iter()
            .all(|literal| self.literals.contains(literal))
    }
}

/// Represents a (partial) sample of configs.
/// The sample differentiates between complete and partial configs.
/// A config is complete (in the context of this sample) if it contains all variables this sample
/// defines. Otherwise the config is partial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sample {
    /// Configs that contain all variables of this sample
    complete_configs: Vec<Config>,
    /// Configs that do not contain all variables of this sample
    partial_configs: Vec<Config>,
    /// The variables that Configs of this sample may contain
    vars: HashSet<u32>,
    /// The literals that actually occur in this sample
    literals: HashSet<i32>,
}

impl Sample {
    /// Create an empty sample that may contain the given variables
    pub fn new(vars: HashSet<u32>) -> Self {
        Self {
            complete_configs: vec![],
            partial_configs: vec![],
            vars,
            literals: HashSet::new(),
        }
    }

    /// Create a sample that only contains a single configuration with a single literal
    pub fn from_literal(literal: i32) -> Self {
        let mut sample = Self::new(HashSet::from([literal.unsigned_abs()]));
        sample.add_complete(Config {
            literals: vec![literal],
        });
        sample
    }

    /// Adds a config to this sample. Only use this method if you know that the config is
    /// complete. The added config is treated as a complete config without checking
    /// if it actually is complete.
    pub fn add_complete(&mut self, config: Config) {
        self.complete_configs.push(config)
    }

    /// Adds a config to this sample. Only use this method if you know that the config is
    /// partial. The added config is treated as a partial config without checking
    /// if it actually is partial.
    pub fn add_partial(&mut self, config: Config) {
        self.partial_configs.push(config)
    }

    /// Adds a config to this sample and automatically determines whether the config is complete
    /// or partial.
    pub fn add(&mut self, config: Config) {
        debug_assert!(config.literals.len() <= self.vars.len(), "Can not insert config with more \
        vars than the sample defines");
        if config.literals.len() < self.vars.len() {
            self.add_partial(config)
        } else {
            self.add_complete(config)
        }
    }

    /// Creates an iterator that first iterates over complete_configs and then over partial_configs
    pub fn iter(&self) -> Chain<Iter<Config>, Iter<Config>> {
        self.complete_configs
            .iter()
            .chain(self.partial_configs.iter())
    }

    /// Returns the number of configs in this sample
    pub fn len(&self) -> usize {
        self.complete_configs.len() + self.partial_configs.len()
    }

    /// Checks if this sample covers the given interaction
    pub fn covers(&self, interaction: &[i32]) -> bool {
        self.iter().any(|conf| conf.covers(interaction))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sample_covering() {
        let sample = Sample {
            complete_configs: vec![Config {
                literals: vec![1, 2, 3, -4, -5],
            }],
            partial_configs: vec![],
            vars: HashSet::from([1, 2, 3, 4, 5]),
            literals: HashSet::from([1, 2, 3, -4, -5]),
        };

        let covered_interaction = vec![1, 2, -4];
        assert!(sample.covers(&covered_interaction));

        let uncovered_interaction = vec![1, 2, 4];
        assert!(!sample.covers(&uncovered_interaction));
    }
}
