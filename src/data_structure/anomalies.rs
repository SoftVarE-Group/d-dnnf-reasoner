use std::{fs::File, io::{LineWriter, Write}, collections::HashSet};

use crate::data_structure::Ddnnf;

impl Ddnnf {
    /// Computes all core features
    /// A feature is a core feature iff there exists only the positiv occurence of that feature
    pub(super) fn get_core(&mut self) {
        self.core = (1..=self.number_of_variables as i32)
            .filter(|f| {
                self.literals.get(f).is_some()
                    && self.literals.get(&-f).is_none()
            })
            .collect::<HashSet<i32>>()
    }

    /// Computes all dead features
    /// A feature is a dead feature iff there exists only the negativ occurence of that feature
    pub(super) fn get_dead(&mut self) {
        self.dead = (1..=self.number_of_variables as i32)
            .filter(|f| {
                self.literals.get(f).is_none()
                    && self.literals.get(&-f).is_some()
            })
            .collect::<HashSet<i32>>()
    }
    
    /// Takes a d-DNNF and writes the string representation into a file with the provided name
    pub fn write_anomalies(&mut self, path_out: &str) -> std::io::Result<()> {    
        let file = File::create(path_out)?;
        let mut file = LineWriter::with_capacity(1000, file);
        
        file.write_all(b"TODO")?;

        Ok(())
    }
}