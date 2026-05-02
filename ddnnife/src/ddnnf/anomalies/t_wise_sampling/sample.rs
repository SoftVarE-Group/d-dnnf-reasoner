use super::Config;
use super::t_iterator::TInteractionIter;
use crate::Ddnnf;
use crate::ddnnf::anomalies::t_wise_sampling::sat_wrapper::SatWrapper;
use crate::int_hash::IntSet;
use log::debug;
use std::cmp::{Ordering, min};
use std::fmt::{Display, Formatter};
use std::iter;
use std::num::ParseIntError;
use streaming_iterator::StreamingIterator;

/// Represents a (partial) sample of configs.
/// The sample differentiates between complete and partial configs.
/// A config is complete (in the context of this sample) if it contains all variables this sample
/// defines. Otherwise the config is partial.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Sample {
    /// Configs that contain all variables of this sample
    pub complete_configs: Vec<Config>,
    /// Configs that do not contain all variables of this sample
    pub partial_configs: Vec<Config>,
    /// The variables that Configs of this sample may contain
    pub vars: IntSet<u32>,
    /// The literals that actually occur in this sample, this is not a HashSet because we want
    /// a stable iteration order.
    pub literals: Vec<i32>,
}

impl PartialOrd<Self> for Sample {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Sample {
    fn cmp(&self, other: &Self) -> Ordering {
        self.len().cmp(&other.len())
    }
}

impl Extend<Config> for Sample {
    fn extend<T: IntoIterator<Item = Config>>(&mut self, iter: T) {
        for config in iter {
            self.add(config);
        }
    }
}

impl Display for Sample {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let last_index = self.len() - 1;

        self.iter().enumerate().try_for_each(|(index, config)| {
            config.fmt(f)?;

            if index == last_index {
                return Ok(());
            }

            writeln!(f)
        })
    }
}

impl Sample {
    /// Create an empty sample that may contain the given variables
    pub fn new(vars: IntSet<u32>) -> Self {
        Self {
            complete_configs: vec![],
            partial_configs: vec![],
            vars,
            literals: vec![],
        }
    }

    /// Create a new sample that will contain the given configs
    ///
    /// # Examples
    /// ```
    /// use ddnnife::ddnnf::anomalies::t_wise_sampling::{Config, Sample};
    ///
    /// let conf_a = Config::from(&[1,2], 3);
    /// let conf_b = Config::from(&[1,2,3], 3);
    /// let sample = Sample::new_from_configs(vec![conf_a, conf_b]);
    ///
    /// assert_eq!(2, sample.len());
    /// assert_eq!(1, sample.complete_configs.len());
    /// assert_eq!(1, sample.partial_configs.len());
    /// assert_eq!(Some(&Config::from(&[1,2,3], 3)), sample.complete_configs.get(0));
    /// assert_eq!(Some(&Config::from(&[1,2], 3)), sample.partial_configs.get(0));
    /// ```
    pub fn new_from_configs(configs: Vec<Config>) -> Self {
        let mut literals: Vec<i32> = configs
            .iter()
            .flat_map(|c| c.get_decided_literals())
            .collect();
        literals.sort_unstable();
        literals.dedup();

        let vars: IntSet<u32> = literals.iter().map(|x| x.unsigned_abs()).collect();

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
        let vars: IntSet<u32> = samples
            .iter()
            .flat_map(|sample| sample.vars.iter())
            .cloned()
            .collect();

        let literals: IntSet<i32> = samples
            .iter()
            .flat_map(|sample| sample.get_literals().iter().copied())
            .collect();

        let mut sample = Self::new(vars);
        sample.literals = literals.into_iter().collect();
        sample.literals.sort_unstable();
        sample
    }

    /// Create a sample that only contains a single configuration with a single literal
    pub fn from_literal(literal: i32, number_of_variables: usize) -> Self {
        let mut sample = Self::new([literal.unsigned_abs()].into_iter().collect());
        sample.literals = vec![literal];
        sample.add_complete(Config::from(&[literal], number_of_variables));
        sample
    }

    /// Creates a new sample from a set of lines representing a configuration each.
    pub fn from_str(input: &str, number_of_variables: usize) -> Result<Self, ParseIntError> {
        // Transform each line into a configuration.
        let configs: Vec<Config> = input
            .lines()
            .map(|line| Config::from_str(line, number_of_variables))
            .collect::<Result<Vec<Config>, ParseIntError>>()?;

        // Collect the literals in all configurations.
        let literals: IntSet<i32> = configs
            .iter()
            .flat_map(|config| config.literals.iter())
            .copied()
            .collect();

        // Collect the corresponding variables.
        let vars: IntSet<u32> = literals
            .iter()
            .map(|literal| literal.unsigned_abs())
            .collect();

        // Split the complete and partial configurations.
        let mut complete_configs = Vec::new();
        let mut partial_configs = Vec::new();

        configs.into_iter().for_each(|config| {
            if config.is_complete() {
                complete_configs.push(config);
                return;
            }

            partial_configs.push(config);
        });

        Ok(Self {
            complete_configs,
            partial_configs,
            vars,
            literals: literals.into_iter().collect(),
        })
    }

    pub fn get_literals(&self) -> &[i32] {
        &self.literals
    }

    pub fn get_vars(&self) -> &IntSet<u32> {
        &self.vars
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
        if self.is_config_complete(&config) {
            self.add_complete(config)
        } else {
            self.add_partial(config)
        }
    }

    /// Determines whether the config is complete (true) or partial (false).
    ///
    /// # Examples
    /// ```
    /// use ddnnife::ddnnf::anomalies::t_wise_sampling::{Config, Sample};
    ///
    /// let sample = Sample::new([1,2,3].into_iter().collect());
    ///
    /// assert!(sample.is_config_complete(&Config::from(&[1,2,3], 3)));
    /// assert!(!sample.is_config_complete(&Config::from(&[1,2], 3)));
    /// ```
    pub fn is_config_complete(&self, config: &Config) -> bool {
        let decided_literals = config.n_decided_literals;
        debug_assert!(
            decided_literals <= self.vars.len(),
            "Can not insert config with more vars than the sample defines"
        );
        decided_literals == self.vars.len()
    }

    /// Creates an iterator that first iterates over complete_configs and then over partial_configs
    pub fn iter(&self) -> impl Iterator<Item = &Config> {
        self.complete_configs
            .iter()
            .chain(self.partial_configs.iter())
    }

    pub fn iter_with_completeness(&self) -> impl Iterator<Item = (&Config, bool)> {
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
    /// use ddnnife::ddnnf::anomalies::t_wise_sampling::{Config, Sample};
    /// let mut s = Sample::new([1,2,3].into_iter().collect());
    ///
    /// assert!(s.is_empty());
    /// s.add_partial(Config::from(&[1,3], 3));
    /// assert!(!s.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.complete_configs.is_empty() && self.partial_configs.is_empty()
    }

    /// Checks if this sample covers the given interaction
    pub fn covers(&self, interaction: &[i32]) -> bool {
        debug_assert!(!interaction.contains(&0));
        self.iter().any(|conf| conf.covers(interaction))
    }

    pub fn is_t_wise_covered(&self, config: &Config, t: usize) -> bool {
        let literals: Vec<i32> = config.get_decided_literals().collect();
        debug_assert!(!literals.contains(&0));

        TInteractionIter::new(&literals, min(t, literals.len()))
            .all(|interaction| self.covers(interaction))
    }

    /// Checks whether all configurations of this sample are complete, i.e. the number
    /// of literals they contain is the same as the number of variables in this sample
    /// overall.
    pub fn all_complete(&self) -> bool {
        self.iter().all(|config| config.is_complete())
    }

    /// Checks whether all configurations in this sample are SAT.
    pub fn all_sat(&self, ddnnf: &Ddnnf) -> bool {
        let sat = SatWrapper::new(ddnnf);

        // Consider each config.
        self.iter().all(|config| {
            // Create a mutable clone for checking SAT.
            let mut config = config.clone();
            let literals: Vec<i32> = config.get_decided_literals().collect();

            if config.sat_state.is_none() {
                config.set_sat_state(sat.new_state());
            }

            // Check that the config is SAT.
            sat.is_sat_cached(
                &literals,
                config
                    .get_sat_state()
                    .expect("sat_state should exist because we initialized it a few lines before"),
            )
        })
    }

    /// Checks whether this sample covers all SAT t-wise interactions of the given literals.
    pub fn covers_literals(&self, ddnnf: &Ddnnf, literals: &[i32], t: usize) -> bool {
        let sat = SatWrapper::new(ddnnf);

        // Keep track of how many actually vaild interactions are checked.
        let mut count = 0;

        // Generate all possible t-wise interactions between the given literals.
        let result = TInteractionIter::new(literals, t)
            // Only consier those that are actually valid (SAT).
            .filter(|interaction| sat.is_sat_cached(interaction, &mut sat.new_state()))
            .inspect(|_| count += 1)
            // Check that each is covered by this sample.
            .all(|interaction| self.covers(interaction));

        debug!("Checked {count} interactions.");

        result
    }

    /// Checks whether this sample covers all SAT t-wise interactions of the given variables.
    pub fn covers_variables(&self, ddnnf: &Ddnnf, variables: &[u32], t: usize) -> bool {
        // Generate both polarities of each variable.
        // Interactions will be later filtered on whether they are SAT.
        let literals: Vec<i32> = variables
            .iter()
            .map(|&variable| variable as i32)
            .flat_map(|variable| [variable, -variable].into_iter())
            .collect();

        // Check the generated literals.
        self.covers_literals(ddnnf, &literals, t)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sample_covering() {
        let sample = Sample {
            complete_configs: vec![Config::from(&[1, 2, 3, -4, -5], 5)],
            partial_configs: vec![],
            vars: [1, 2, 3, 4, 5].into_iter().collect(),
            literals: vec![1, 2, 3, -4, -5],
        };

        let covered_interaction = vec![1, 2, -4];
        assert!(sample.covers(&covered_interaction));

        let uncovered_interaction = vec![1, 2, 4];
        assert!(!sample.covers(&uncovered_interaction));
    }
}
