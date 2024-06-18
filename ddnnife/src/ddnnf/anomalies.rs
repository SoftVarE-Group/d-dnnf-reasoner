pub mod atomic_sets;
pub mod config_creation;
pub mod core;
pub mod false_optional;
pub mod sat;
pub mod t_wise_sampling;

use std::{
    fs::File,
    io::{LineWriter, Write},
};

use crate::Ddnnf;

impl Ddnnf {
    /// Takes a d-DNNF and writes the string representation into a file with the provided name
    pub fn write_anomalies(&mut self, path_out: &str) -> std::io::Result<()> {
        let file = File::create(path_out)?;
        let mut file = LineWriter::with_capacity(1000, file);

        // core/dead features
        let mut core = self.core.clone().into_iter().collect::<Vec<i32>>();
        core.sort();
        file.write_all(format!("core: {core:?}\n").as_bytes())?;

        // false-optionals

        // atomic sets
        let mut atomic_sets = self.get_atomic_sets(None, &[], false);
        atomic_sets.sort_unstable();
        file.write_all(format!("atomic sets: {atomic_sets:?}\n").as_bytes())?;

        Ok(())
    }
}
