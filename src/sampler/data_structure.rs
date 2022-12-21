use std::collections::HashSet;
use std::iter;

/// Represents a (partial) configuration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Config {
    /// A vector of selected features (positive values) and deselected features (negative values)
    literals: Vec<i32>,
}

impl Config {
    /// Creates a new config with the given literals
    pub fn from(literals: &[i32]) -> Self {
        Self {
            literals: Vec::from(literals),
        }
    }

    /// Creates a new config from two disjoint configs.
    pub fn from_disjoint(left: &Self, right: &Self) -> Self {
        let mut literals = left.literals.clone();
        literals.extend(right.literals.iter());
        Self { literals }
    }

    /// Checks if this config covers the given interaction
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

impl Extend<Config> for Sample {
    fn extend<T: IntoIterator<Item = Config>>(&mut self, iter: T) {
        for config in iter {
            self.add(config);
        }
    }
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

    /// Create a new sample that will contain the given configs
    ///
    /// # Examples
    /// ```
    /// use ddnnf_lib::sampler::data_structure::{Config, Sample};
    ///
    /// let conf_a = Config::from(&[1,2]);
    /// let conf_b = Config::from(&[1,2,3]);
    /// let sample = Sample::new_from_configs(vec![conf_a, conf_b]);
    ///
    /// let mut iter = sample.iter();
    /// assert_eq!(Some(&Config::from(&[1,2,3])), iter.next());
    /// assert_eq!(Some(&Config::from(&[1,2])), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn new_from_configs(configs: Vec<Config>) -> Self {
        let literals: HashSet<i32> = configs
            .iter()
            .flat_map(|c| c.literals.iter())
            .cloned()
            .collect();

        let vars: HashSet<u32> =
            literals.iter().map(|x| x.unsigned_abs()).collect();

        let mut sample = Self {
            complete_configs: vec![],
            partial_configs: vec![],
            vars,
            literals,
        };

        sample.extend(configs);
        sample
    }

    pub fn new_from_samples(samples: &[&Self]) -> Self {
        let vars: HashSet<u32> = samples
            .iter()
            .flat_map(|sample| sample.vars.iter())
            .cloned()
            .collect();

        Self::new(vars)
    }

    /// Create an empty sample that may contain the given variables and will certainly contain
    /// the given literals. Only use this if you know that the configs you are going to add to
    /// this sample contain the given literals.
    pub fn new_with_literals(
        vars: HashSet<u32>,
        literals: HashSet<i32>,
    ) -> Self {
        Self {
            complete_configs: vec![],
            partial_configs: vec![],
            vars,
            literals,
        }
    }

    /// Create an empty sample with no variables defined
    pub fn empty() -> Self {
        Self {
            complete_configs: vec![],
            partial_configs: vec![],
            vars: HashSet::new(),
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

    pub fn get_literals(&self) -> &HashSet<i32> {
        &self.literals
    }

    /// Adds a config to this sample. Only use this method if you know that the config is
    /// complete. The added config is treated as a complete config without checking
    /// if it actually is complete.
    pub fn add_complete(&mut self, config: Config) {
        self.literals.extend(&config.literals);
        self.complete_configs.push(config)
    }

    /// Adds a config to this sample. Only use this method if you know that the config is
    /// partial. The added config is treated as a partial config without checking
    /// if it actually is partial.
    pub fn add_partial(&mut self, config: Config) {
        self.literals.extend(&config.literals);
        self.partial_configs.push(config)
    }

    /// Adds a config to this sample and automatically determines whether the config is complete
    /// or partial.
    pub fn add(&mut self, config: Config) {
        debug_assert!(
            config.literals.len() <= self.vars.len(),
            "Can not insert config with more vars than the sample defines"
        );
        if config.literals.len() < self.vars.len() {
            self.add_partial(config)
        } else {
            self.add_complete(config)
        }
    }

    /// Creates an iterator that first iterates over complete_configs and then over partial_configs
    pub fn iter(&self) -> impl Iterator<Item = &Config> {
        self.complete_configs
            .iter()
            .chain(self.partial_configs.iter())
    }

    pub fn iter_with_completeness(
        &self,
    ) -> impl Iterator<Item = (&Config, bool)> {
        let partial_iter = self.partial_configs.iter().zip(iter::repeat(false));

        self.complete_configs
            .iter()
            .zip(iter::repeat(true))
            .chain(partial_iter)
    }

    /// Returns the number of configs in this sample
    pub fn len(&self) -> usize {
        self.complete_configs.len() + self.partial_configs.len()
    }

    /// Returns true if the sample contains no configs
    ///
    /// # Examples
    /// ```
    /// use std::collections::HashSet;
    /// use ddnnf_lib::sampler::data_structure::{Config, Sample};
    /// let mut s = Sample::new(HashSet::from([1,2,3]));
    ///
    /// assert!(s.is_empty());
    /// s.add_partial(Config::from(&[1,3]));
    /// assert!(!s.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.complete_configs.is_empty() && self.partial_configs.is_empty()
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
