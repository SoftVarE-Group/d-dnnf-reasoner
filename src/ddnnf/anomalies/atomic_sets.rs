use std::collections::HashMap;
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
        let combinations: HashMap<Integer, Vec<i32>> = HashMap::new();

        for (key, value) in &combinations {
            if &self.execute_query(value) == key {
                // is atomic set
            } else {
                // try to remove
            }
        }

        return vec![];
    }

    /**
     * Hat der inkrementelle Ansatz Probleme???
     * Bsp. atomic set should be 1,2
     * Candidates sind 1,2,3,4
     * 
     * Erster Kandidat wird random 3
     * => gleiche Problematik der kombinatorischen Explusion wie bei Rauswerfen der Werte
     * => Valider Ausgangspunkt ist nötig aka ein feature das sicher Teil des atomic sets ist
     *      => random suchen bis eine Kombination aus zwei features stimmt
     *      => wenn erstes feature mit keinem passt => nächster Kandidat
    */
    #[allow(dead_code)]
    fn _incremental_check(&mut self, control: Integer, pot_atomic_set: &mut Vec<i32>) -> Vec<i32> {
        let mut atomic_set = vec![pot_atomic_set.pop().unwrap()];

        for candidate in pot_atomic_set {
            atomic_set.push(*candidate);
            if self.execute_query(&atomic_set) == control {
            }
        }

        atomic_set
    }
}