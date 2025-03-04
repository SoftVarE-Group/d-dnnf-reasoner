use crate::Ddnnf;
use ddnnife::util;
use num::BigInt;
use std::sync::Mutex;

/// A mutable version of a d-DNNF, required for some computations.
///
/// This version has thread-safe access to computations requiring mutability.
/// A lock will be managed directly by the library.
/// Converting into and out from the mutable version will create new instances.
#[derive(uniffi::Object)]
pub struct DdnnfMut(pub Mutex<Ddnnf>);

#[uniffi::export]
impl DdnnfMut {
    /// Creates a non-mutable copy of this d-DNNF.
    #[uniffi::method]
    fn as_ddnnf(&self) -> Ddnnf {
        self.0.lock().expect("Failed to lock d-DNNF.").clone()
    }

    /// Computes the cardinality of this d-DNNF.
    #[uniffi::method]
    fn count(&self, assumptions: &[i32]) -> BigInt {
        self.0.lock().unwrap().0.execute_query(assumptions)
    }

    /// Computes the cardinality of this d-DNNF for multiple variables.
    #[uniffi::method]
    fn count_multiple(&self, assumptions: &[i32], variables: &[i32]) -> Vec<BigInt> {
        let mut ddnnf = self.0.lock().unwrap();
        util::zip_assumptions_variables(assumptions, variables)
            .map(|assumptions| ddnnf.0.execute_query(&assumptions))
            .collect()
    }

    /// Computes whether this d-DNNF is satisfiable.
    #[uniffi::method]
    fn is_sat(&self, assumptions: &[i32]) -> bool {
        self.0.lock().unwrap().0.sat(assumptions)
    }

    /// Computes the core features of this d-DNNF.
    #[uniffi::method]
    fn core(&self, assumptions: &[i32]) -> Vec<i32> {
        self.0.lock().unwrap().0.core_with_assumptions(assumptions)
    }

    /// Computes the core features of this d-DNNF for multiple variables.
    #[uniffi::method]
    fn core_multiple(&self, assumptions: &[i32], variables: &[i32]) -> Vec<i32> {
        let mut ddnnf = self.0.lock().unwrap();

        let mut core: Vec<i32> = util::zip_assumptions_variables(assumptions, variables)
            .flat_map(|assumptions| ddnnf.0.core_with_assumptions(&assumptions).into_iter())
            .collect();

        core.sort();
        core.dedup();
        core
    }

    /// Computes the dead features of this d-DNNF.
    #[uniffi::method]
    fn dead(&self, assumptions: &[i32]) -> Vec<i32> {
        self.0.lock().unwrap().0.dead_with_assumptions(assumptions)
    }

    /// Computes the dead features of this d-DNNF for multiple variables.
    #[uniffi::method]
    fn dead_multiple(&self, assumptions: &[i32], variables: &[i32]) -> Vec<i32> {
        let mut ddnnf = self.0.lock().unwrap();

        let mut dead: Vec<i32> = util::zip_assumptions_variables(assumptions, variables)
            .flat_map(|assumptions| ddnnf.0.dead_with_assumptions(&assumptions).into_iter())
            .collect();

        dead.sort();
        dead.dedup();
        dead
    }

    /// Generates satisfiable configurations for this d-DNNF.
    #[uniffi::method]
    fn enumerate(&self, assumptions: &[i32], amount: usize) -> Vec<Vec<i32>> {
        let mut assumptions = assumptions.to_vec();
        self.0
            .lock()
            .unwrap()
            .0
            .enumerate(&mut assumptions, amount)
            .unwrap_or_default()
    }

    /// Generates random satisfiable configurations for this d-DNNF.
    #[uniffi::method]
    fn random(&self, assumptions: &[i32], amount: usize, seed: u64) -> Vec<Vec<i32>> {
        self.0
            .lock()
            .unwrap()
            .0
            .uniform_random_sampling(assumptions, amount, seed)
            .unwrap_or_default()
    }

    /// Compute all atomic sets.
    ///
    /// A group forms an atomic set iff every valid configuration either includes
    /// or excludes all members of that atomic set.
    #[uniffi::method]
    fn atomic_sets(
        &self,
        candidates: Option<Vec<u32>>,
        assumptions: &[i32],
        cross: bool,
    ) -> Vec<Vec<i16>> {
        self.0
            .lock()
            .unwrap()
            .0
            .get_atomic_sets(candidates, assumptions, cross)
    }
}
