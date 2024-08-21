use crate::Ddnnf;
use num::Zero;
use std::collections::HashSet;

impl Ddnnf {
    /// Computes all dead and core features.
    /// A feature is a core feature iff there exists only the positiv occurence of that feature.
    /// A feature is a dead feature iff there exists only the negativ occurence of that feature.
    pub(crate) fn calculate_core(&mut self) {
        self.core = (-(self.number_of_variables as i32)..=self.number_of_variables as i32)
            .filter(|f| self.literals.contains_key(f) && !self.literals.contains_key(&-f))
            .collect::<HashSet<i32>>()
    }

    /// Checks if removing the feature assigment from the query does not change the query
    /// i.e. that feature is an included core feature or an excluded dead feature
    pub(crate) fn has_no_effect_on_query(&self, feature: &i32) -> bool {
        feature.is_positive() && self.core.contains(feature)
            || feature.is_negative() && self.core.contains(feature)
    }

    /// Checks if that feature assignment alone must result in an unsat query
    /// i.e. that feature is an excluded core feature or an included dead feature
    pub(crate) fn makes_query_unsat(&self, feature: &i32) -> bool {
        feature.is_negative() && self.core.contains(&-feature)
            || feature.is_positive() && self.core.contains(&-feature)
    }

    #[inline]
    /// Reduces a query by removing included core features and excluded dead features
    pub(crate) fn reduce_query(&mut self, features: &[i32]) -> Vec<i32> {
        features
            .iter()
            .filter({
                // filter keeps the elements which do fulfill the defined boolean formula. Thats why we need to use the ! operator
                |&f| !self.has_no_effect_on_query(f)
            })
            .copied()
            .collect::<Vec<i32>>()
    }

    #[inline]
    /// Checks if a query is satisfiable. That is not the case if either a core feature is excluded or a dead feature is included
    pub(crate) fn query_is_not_sat(&mut self, features: &[i32]) -> bool {
        // if there is an included dead or an excluded core feature
        features.iter().any(|f| self.makes_query_unsat(f))
    }

    /// Calculates the core and dead features for a given assumption.
    ///
    /// The return values are not deduplicated, they might be put into a `HashSet`.
    pub fn core_dead_with_assumptions(self: &mut Ddnnf, assumptions: &[i32]) -> Vec<i32> {
        if assumptions.is_empty() {
            return self.core.iter().copied().collect();
        }

        let mut assumptions = assumptions.to_vec();
        let mut core = Vec::new();
        let reference = self.execute_query(&assumptions);

        for i in 1_i32..=self.number_of_variables as i32 {
            assumptions.push(i);
            let inter = self.execute_query(&assumptions);
            if reference == inter {
                core.push(i);
            }
            if inter.is_zero() {
                core.push(-i);
            }
            assumptions.pop();
        }

        core
    }

    pub fn core_with_assumptions(self: &mut Ddnnf, assumptions: &[i32]) -> Vec<i32> {
        let mut features = self.core_dead_with_assumptions(assumptions);
        features.retain(|feature| feature.is_positive());
        features
    }

    pub fn dead_with_assumptions(self: &mut Ddnnf, assumptions: &[i32]) -> Vec<i32> {
        let mut features = self.core_dead_with_assumptions(assumptions);
        features.retain(|feature| feature.is_negative());
        features
    }
}
