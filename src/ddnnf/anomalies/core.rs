use rustc_hash::FxHashSet;

use crate::{Ddnnf};

impl Ddnnf {
        /// Computes all core features
    /// A feature is a core feature iff there exists only the positiv occurence of that feature
    pub(crate) fn get_core(&mut self) {
        self.core = (1..=self.number_of_variables as i32)
            .filter(|f| {
                self.literals.get(f).is_some()
                    && self.literals.get(&-f).is_none()
            })
            .collect::<FxHashSet<i32>>()
    }

    /// Computes all dead features
    /// A feature is a dead feature iff there exists only the negativ occurence of that feature
    pub(crate) fn get_dead(&mut self) {
        self.dead = (1..=self.number_of_variables as i32)
            .filter(|f| {
                self.literals.get(f).is_none()
                    && self.literals.get(&-f).is_some()
            })
            .collect::<FxHashSet<i32>>()
    }

    #[inline]
    /// Reduces a query by removing included core features and excluded dead features
    pub(crate) fn reduce_query(&mut self, features: &[i32]) -> Vec<i32> {
        features
            .iter()
            .filter({
                // filter keeps the elements which do fulfill the defined boolean formula. Thats why we need to use the ! operator
                |&f| {
                    if f > &0 {
                        // remove included core and excluded dead features
                        !self.core.contains(f)
                    } else {
                        !self.dead.contains(&-f)
                    }
                }
            })
            .copied()
            .collect::<Vec<i32>>()
    }

    #[inline]
    /// Checks if a query is satisfiable. That is not the case if either a core feature is excluded or a dead feature is included
    pub(crate) fn query_is_not_sat(&mut self, features: &[i32]) -> bool {
        // if there is an included dead or an excluded core feature
        features.iter().any({
            |&f| {
                if f > 0 {
                    self.dead.contains(&f)
                } else {
                    self.core.contains(&-f)
                }
            }
        })
    }
}