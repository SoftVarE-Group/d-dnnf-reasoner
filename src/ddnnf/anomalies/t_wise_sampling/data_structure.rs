use crate::ddnnf::anomalies::t_wise_sampling::sat_wrapper::SatWrapper;
use crate::parser::util::format_vec;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt::Display;
use std::iter;

/// Represents a (partial) configuration
#[derive(Debug, Clone, Eq)]
pub struct Config {
    /// A vector of selected features (positive values) and deselected features (negative values)
    literals: Vec<i32>,
    pub sat_state: Option<Vec<bool>>,
    sat_state_complete: bool,
}

impl PartialEq for Config {
    fn eq(&self, other: &Self) -> bool {
        debug_assert_eq!(self.literals.len(), other.literals.len());
        self.literals.eq(&other.literals)
    }
}

impl Extend<i32> for Config {
    fn extend<T: IntoIterator<Item = i32>>(&mut self, iter: T) {
        self.sat_state_complete = false;
        for literal in iter {
            self.add(literal);
        }
    }
}

impl Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format_vec(self.literals.iter()))
    }
}

impl Config {
    /// Creates a new config with the given literals
    pub fn from(literals: &[i32], number_of_variables: usize) -> Self {
        let mut config = Self {
            literals: vec![0; number_of_variables],
            sat_state: None,
            sat_state_complete: false,
        };
        config.extend(literals.iter().copied());
        config
    }

    /// Creates a new config from two disjoint configs.
    pub fn from_disjoint(left: &Self, right: &Self, number_of_variables: usize) -> Self {
        let sat_state = match (left.sat_state.clone(), right.sat_state.clone()) {
            (Some(left_state), Some(right_state)) => {
                /*
                We pick the cached state of the larger config because we can not combine the
                cached states. This would break the upward propagation of the marks.
                Example: There is an AND with two children A and B.
                A is marked in the left state
                B is marked in the right state
                If we simply combine the two states then A is marked and B is marked but the
                marker does not propagate upward to the AND. So the AND remains unmarked which
                is wrong and may cause wrong results when SAT solving.
                 */
                if left.get_decided_literals().count() >= right.get_decided_literals().count() {
                    Some(left_state)
                } else {
                    Some(right_state)
                }
            }
            (Some(state), None) | (None, Some(state)) => Some(state),
            (None, None) => None,
        };

        let mut config = Self {
            literals: vec![0; number_of_variables],
            sat_state,
            sat_state_complete: false, // always false because we can not combine the states
        };
        config.extend(left.get_decided_literals());
        config.extend(right.get_decided_literals());
        config
    }

    /// Returns a slice of this configs literals (may contain zeros)
    pub fn get_literals(&self) -> &[i32] {
        &self.literals
    }

    /// Returns an iterator over the selected and unselected features
    pub fn get_decided_literals(&self) -> impl Iterator<Item = i32> + '_ {
        self.literals
            .iter()
            .copied()
            .filter(|&literal| literal != 0)
    }

    /// Returns the cached sat state if there is one
    pub fn get_sat_state(&mut self) -> Option<&mut Vec<bool>> {
        self.sat_state.as_mut()
    }

    /// Sets the cached sat state
    pub fn set_sat_state(&mut self, sat_state: Vec<bool>) {
        self.sat_state_complete = true;
        self.sat_state = Some(sat_state);
    }

    /// Returns whether the cached sat state is complete (true) or incomplete (false)
    fn is_sat_state_complete(&self) -> bool {
        self.sat_state_complete
    }

    /// Uses the given [SatWrapper] to update the cached sat solver state in this config.
    /// This does nothing if the cache is up to date.
    pub fn update_sat_state(&mut self, sat_solver: &SatWrapper, root: usize) {
        if self.is_sat_state_complete() {
            debug_assert!(
                self.sat_state.is_some(),
                "sat_state should be Some(_) if sat_state_complete is true"
            );
            return;
        }

        // clone literals to avoid borrow problems in the sat solver call below
        let literals: Vec<i32> = self.get_decided_literals().collect();

        if self.sat_state.is_none() {
            self.set_sat_state(sat_solver.new_state());
        }

        sat_solver.is_sat_in_subgraph_cached(
            &literals,
            root,
            self.get_sat_state()
                .expect("sat_state should exist because we initialized it a few lines before"),
        );
    }

    /// Checks if this config obviously conflicts with the interaction.
    /// This is the case when the config contains a literal *l* and the interaction contains *-l*
    pub fn conflicts_with(&self, interaction: &[i32]) -> bool {
        interaction
            .iter()
            .filter(|&&literal| literal != 0)
            .any(|&literal| self.contains(-literal))
    }

    /// Checks if this config covers the given interaction
    pub fn covers(&self, interaction: &[i32]) -> bool {
        interaction
            .iter()
            .filter(|&&literal| literal != 0)
            .all(|&literal| self.contains(literal))
    }

    pub fn contains(&self, literal: i32) -> bool {
        debug_assert!(literal != 0);
        let index = literal.unsigned_abs() as usize - 1;
        self.literals[index] == literal
    }

    pub fn add(&mut self, literal: i32) {
        if literal == 0 {
            return;
        }
        debug_assert!(literal != 0);
        self.sat_state_complete = false;
        let index = literal.unsigned_abs() as usize - 1;
        self.literals[index] = literal;
    }
}

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
    pub(super) vars: HashSet<u32>,
    /// The literals that actually occur in this sample, this is not a HashSet because we want
    /// a stable iteration order.
    pub(super) literals: Vec<i32>,
}

impl PartialOrd<Self> for Sample {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.len().cmp(&other.len()))
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

impl Sample {
    /// Create an empty sample that may contain the given variables
    pub fn new(vars: HashSet<u32>) -> Self {
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
    /// use ddnnf_lib::ddnnf::anomalies::t_wise_sampling::data_structure::{Config, Sample};
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

        let vars: HashSet<u32> = literals.iter().map(|x| x.unsigned_abs()).collect();

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

        let literals: HashSet<i32> = samples
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
        let mut sample = Self::new(HashSet::from([literal.unsigned_abs()]));
        sample.literals = vec![literal];
        sample.add_complete(Config::from(&[literal], number_of_variables));
        sample
    }

    pub fn get_literals(&self) -> &[i32] {
        &self.literals
    }

    pub fn get_vars(&self) -> &HashSet<u32> {
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
    /// use std::collections::HashSet;
    /// use ddnnf_lib::ddnnf::anomalies::t_wise_sampling::data_structure::{Config, Sample};
    ///
    /// let sample = Sample::new(HashSet::from([1,2,3]));
    ///
    /// assert!(sample.is_config_complete(&Config::from(&[1,2,3], 3)));
    /// assert!(!sample.is_config_complete(&Config::from(&[1,2], 3)));
    /// ```
    pub fn is_config_complete(&self, config: &Config) -> bool {
        let decided_literals = config.get_decided_literals().count();
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
    /// use std::collections::HashSet;
    /// use ddnnf_lib::ddnnf::anomalies::t_wise_sampling::data_structure::{Config, Sample};
    /// let mut s = Sample::new(HashSet::from([1,2,3]));
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
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sample_covering() {
        let sample = Sample {
            complete_configs: vec![Config::from(&[1, 2, 3, -4, -5], 5)],
            partial_configs: vec![],
            vars: HashSet::from([1, 2, 3, 4, 5]),
            literals: vec![1, 2, 3, -4, -5],
        };

        let covered_interaction = vec![1, 2, -4];
        assert!(sample.covers(&covered_interaction));

        let uncovered_interaction = vec![1, 2, 4];
        assert!(!sample.covers(&uncovered_interaction));
    }
}
