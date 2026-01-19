use super::SatWrapper;
use crate::ddnnf::anomalies::t_wise_sampling::t_iterator::TInteractionIter;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use streaming_iterator::StreamingIterator;

/// Represents a (partial) configuration
#[derive(Debug, Clone, Eq)]
pub struct Config {
    /// A vector of selected features (positive values) and deselected features (negative values)
    pub literals: Vec<i32>,
    pub sat_state: Option<Vec<bool>>,
    sat_state_complete: bool,
    /// The number of decided literals
    pub n_decided_literals: usize,
}

impl Hash for Config {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.literals.hash(state)
    }
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
        let last_index = self.literals.len() - 1;

        self.literals
            .iter()
            .enumerate()
            .try_for_each(|(index, literal)| {
                literal.fmt(f)?;

                if index == last_index {
                    return Ok(());
                }

                write!(f, " ")
            })
    }
}

impl Config {
    /// Creates a new config with the given literals
    pub fn from(literals: &[i32], number_of_variables: usize) -> Self {
        let mut config = Self {
            literals: vec![0; number_of_variables],
            sat_state: None,
            sat_state_complete: false,
            n_decided_literals: 0,
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
                if left.n_decided_literals >= right.n_decided_literals {
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
            n_decided_literals: 0,
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

    /// Returns the number of decided literals
    pub fn get_n_decided_literals(&self) -> usize {
        debug_assert!(self.n_decided_literals == self.get_decided_literals().count());
        self.n_decided_literals
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
        if self.literals[index] == 0 {
            self.n_decided_literals += 1;
        }
        self.literals[index] = literal;
    }

    /// Checks if the config is complete (all literals are decided)
    pub fn is_complete(&self) -> bool {
        self.n_decided_literals == self.literals.len()
    }

    /// Generates all `t`-wise interactions covered by this configuration.
    pub fn interactions(&self, t: usize) -> Vec<Vec<i32>> {
        let mut out = Vec::new();
        TInteractionIter::new(&self.literals, t)
            .for_each(|interaction| out.push(interaction.to_vec()));

        out
    }
}
