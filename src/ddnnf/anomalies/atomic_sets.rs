use itertools::Itertools;
use rug::Integer;

use crate::Ddnnf;

impl Ddnnf {
    /// Compute all false-optional features
    /// A feature is false optional iff there is no valid configuration that excludes
    /// the feature while simultanously excluding its parent
    #[allow(dead_code)]
    fn get_false_optional(&mut self) {
        /***
         * Probleme: 
         *      - Wie findet man den parent eines features?
         *      - Welches feature ist das root feature?
         * 
         * Beziehung eines features zu parent:
         * p => c (Damit das child ausgewählt werden kann muss der parent gewählt sein)
         * 
         * F_FM and p and not c is unsatisfiable
         * <=> #SAT(F_FM and p) == #SAT(F_FM and p and c) und f ist nicht mandatory
         * Wie ermittelt man ob ein feature mandatory ist?
         *      -> SAT(F_FM and p and other child von p)
         * 
        */
    }

    /// Compute all atomic sets
    /// A group forms an atomic set iff every valid configuration either includes
    /// or excludes all mebers of that atomic set
    pub(crate) fn get_atomic_sets(&mut self) -> Vec<Vec<i32>> {
        let mut combinations: Vec<(Integer, i32)> = Vec::new();

        for i in 1..=self.number_of_variables as i32 {
            combinations.push((self.execute_query(&[i]), i));
        }
        combinations.sort_unstable();
        
        let mut data_grouped = Vec::new();
        
        let mut current_key = combinations[0].0.clone();
        let mut values_current_key = Vec::new();
        for (key, value) in combinations.into_iter() {
            if current_key == key {
                values_current_key.push(value);
            } else {
                data_grouped.push((current_key, std::mem::take(&mut values_current_key)));
                current_key = key;
                values_current_key.push(value);
            }
        }
        data_grouped.push((current_key, values_current_key));

        let mut result = Vec::new();
        for (key, group) in data_grouped {
            let atomic_set = self._incremental_check(key, &group);
            if !atomic_set.is_empty() {
                result.push(atomic_set);
            }
        }

        return result;
    }

    /// first naive approach to compute atomic sets by incrementally add a feature one by one
    /// while checking if the atomic set property (i.e. the count stays the same) still holds
    fn _incremental_check(&mut self, control: Integer, pot_atomic_set: &Vec<i32>) -> Vec<i32> {
        if pot_atomic_set.len() == 1 { return pot_atomic_set.to_vec();}

        let mut atomic_set: Vec<i32> = self.find_atomic_pair(&control, pot_atomic_set);
        if atomic_set.is_empty() { return vec![]; }

        for candidate in pot_atomic_set {
            if atomic_set.contains(candidate) { continue; }
            if self.execute_query(&atomic_set) == control {
                atomic_set.push(*candidate);
            }
        }
        atomic_set
    }

    /// searches for a starting pair of two features that fulfill the following property:
    /// #(mc,1) == #(mc,1) == #(mc,[1,2])
    fn find_atomic_pair(&mut self, control: &Integer, pot_atomic_set: &Vec<i32>) -> Vec<i32> {
        for pair in pot_atomic_set.clone().into_iter().combinations(2) {
            if &self.execute_query(&pair) == control {
                return pair;
            }
        }

        return vec![];
    }
}