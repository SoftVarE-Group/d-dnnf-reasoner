use crate::Ddnnf;
use num::BigUint;
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
    fn count(&self, assumptions: &[i32]) -> BigUint {
        self.0.lock().unwrap().0.execute_query(assumptions)
    }

    /// Computes whether this d-DNNF is satisfiable.
    #[uniffi::method]
    fn is_sat(&self, assumptions: &[i32]) -> bool {
        self.0.lock().unwrap().0.sat(assumptions)
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
}
