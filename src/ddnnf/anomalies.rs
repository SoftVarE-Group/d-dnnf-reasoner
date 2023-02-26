pub mod atomic_sets;
pub mod core;
pub mod false_optional;
pub mod sat;

use std::{fs::File, io::{LineWriter, Write}};

use crate::Ddnnf;

impl Ddnnf {
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
        let atomic_sets = self.get_atomic_sets();
        file.write_all(format!("atomic sets: {:?}\n", atomic_sets).as_bytes())?;

        Ok(())
    }
}