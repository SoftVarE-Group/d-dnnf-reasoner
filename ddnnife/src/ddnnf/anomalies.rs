pub mod atomic_sets;
pub mod config_creation;
pub mod core;
pub mod false_optional;
pub mod sat;
pub mod t_wise_sampling;

use crate::Ddnnf;
use std::io::Write;

impl Ddnnf {
    /// Takes a d-DNNF and writes the string representation into a file with the provided name
    pub fn write_anomalies(&mut self, mut output: impl Write) -> std::io::Result<()> {
        // core/dead features
        let mut core = self.core.clone().into_iter().collect::<Vec<i32>>();
        core.sort();
        output.write_all(format!("core: {core:?}\n").as_bytes())?;

        // false-optionals

        // atomic sets
        let mut atomic_sets = self.get_atomic_sets(None, &[], false);
        atomic_sets.sort_unstable();
        output.write_all(format!("atomic sets: {atomic_sets:?}\n").as_bytes())?;

        Ok(())
    }
}
