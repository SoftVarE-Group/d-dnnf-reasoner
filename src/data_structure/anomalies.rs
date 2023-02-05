use std::{fs::File, io::{LineWriter, Write}, collections::{HashSet, HashMap}};
use rug::Integer;

use crate::data_structure::Ddnnf;

impl Ddnnf {
    /// Computes all core features
    /// A feature is a core feature iff it is included in every valid configuration,
    /// or with our data structure: there exists only the positiv literal of that feature
    pub(super) fn get_core(&mut self) {
        self.core = (1..=self.number_of_variables as i32)
            .filter(|f| {
                self.literals.get(f).is_some()
                    && self.literals.get(&-f).is_none()
            })
            .collect::<HashSet<i32>>()
    }

    /// Computes all dead features
    /// A feature is a dead feature iff it is excluded in every valid configuration
    /// or with our data structure: there exists only the negativ literal of that feature
    pub(super) fn get_dead(&mut self) {
        self.dead = (1..=self.number_of_variables as i32)
            .filter(|f| {
                self.literals.get(f).is_none()
                    && self.literals.get(&-f).is_some()
            })
            .collect::<HashSet<i32>>()
    }

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
    fn get_atomic_sets(&mut self) {
        let mut combinations: HashMap<Integer, Vec<i32>> = HashMap::new();

        for (key, value) in &combinations {
            if &self.card_of_partial_config_with_marker(value) == key {
                // is atomic set
            } else {
                // try to remove
            }
        }
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
    fn incrementalCheck(&mut self, control: Integer, pot_atomic_set: &mut Vec<i32>) -> Vec<i32> {
        let mut atomic_set = vec![pot_atomic_set.pop().unwrap()];

        for candidate in pot_atomic_set {
            atomic_set.push(*candidate);
            if self.card_of_partial_config_with_marker(&atomic_set) == control {
            }
        }

        atomic_set
    }
    
    /// Takes a d-DNNF and writes the string representation into a file with the provided name
    pub fn write_anomalies(&mut self, path_out: &str) -> std::io::Result<()> {    
        let file = File::create(path_out)?;
        let mut file = LineWriter::with_capacity(1000, file);
        
        // core features
        let mut core = self.core.clone().into_iter().collect::<Vec<i32>>();
        core.sort();
        file.write_all(format!("core: {:?}\n", core).as_bytes())?;

        // dead features
        let mut dead = self.dead.clone().into_iter().collect::<Vec<i32>>();
        dead.sort();
        file.write_all(format!("dead: {:?}\n", dead).as_bytes())?;

        // false-optionals

        // atomic sets
        self.get_atomic_sets();
        println!("{:?}", self.cos);

        Ok(())
    }
}