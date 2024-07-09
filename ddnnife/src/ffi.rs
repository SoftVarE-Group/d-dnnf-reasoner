use crate::parser::persisting::write_ddnnf_to_file;
use crate::util;
use crate::{Ddnnf, UniffiCustomTypeConverter};
use itertools::Itertools;
use num::BigInt;
use std::collections::HashSet;
use std::sync::Mutex;

uniffi::custom_type!(BigInt, Vec<u8>);

impl UniffiCustomTypeConverter for BigInt {
    type Builtin = Vec<u8>;

    fn into_custom(value: Self::Builtin) -> uniffi::Result<Self> {
        Ok(BigInt::from_signed_bytes_be(&value))
    }

    fn from_custom(custom: Self) -> Self::Builtin {
        custom.to_signed_bytes_be()
    }
}

uniffi::custom_type!(usize, u64);

impl UniffiCustomTypeConverter for usize {
    type Builtin = u64;

    fn into_custom(value: Self::Builtin) -> uniffi::Result<Self> {
        Ok(value as usize)
    }

    fn from_custom(custom: Self) -> Self::Builtin {
        custom as u64
    }
}

type HashSetu32 = HashSet<u32>;
type Vecu32 = Vec<u32>;

uniffi::custom_type!(HashSetu32, Vecu32);

impl UniffiCustomTypeConverter for HashSet<u32> {
    type Builtin = Vec<u32>;

    fn into_custom(value: Self::Builtin) -> uniffi::Result<Self> {
        Ok(value.into_iter().collect())
    }

    fn from_custom(custom: Self) -> Self::Builtin {
        custom.into_iter().collect()
    }
}

type HashSeti32 = HashSet<i32>;
type Veci32 = Vec<i32>;

uniffi::custom_type!(HashSeti32, Veci32);

impl UniffiCustomTypeConverter for HashSet<i32> {
    type Builtin = Vec<i32>;

    fn into_custom(value: Self::Builtin) -> uniffi::Result<Self> {
        Ok(value.into_iter().collect())
    }

    fn from_custom(custom: Self) -> Self::Builtin {
        custom.into_iter().collect()
    }
}

/// A mutable version of a d-DNNF, required for some computations.
///
/// This version has thread-safe access to computations requiring mutability.
/// A lock will be managed directly by the library.
/// Converting into and out from the mutable version will create new instances.
#[derive(uniffi::Object)]
pub struct DdnnfMut(pub Mutex<Ddnnf>);

#[uniffi::export]
impl Ddnnf {
    /// Creates a mutable copy of this d-DNNF.
    #[uniffi::method]
    fn as_mut(&self) -> DdnnfMut {
        DdnnfMut(Mutex::new(self.clone()))
    }

    /// Saves this d-DNNF to the given file.
    #[uniffi::method]
    fn save(&self, path: &str) {
        write_ddnnf_to_file(self, path).unwrap();
    }
}

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
        self.0.lock().unwrap().execute_query(assumptions)
    }

    /// Computes the cardinality of this d-DNNF for multiple variables.
    #[uniffi::method]
    fn count_multiple(&self, assumptions: &[i32], variables: &[i32]) -> Vec<BigInt> {
        let mut ddnnf = self.0.lock().unwrap();
        util::zip_assumptions_variables(assumptions, variables)
            .map(|assumptions| ddnnf.execute_query(&assumptions))
            .collect()
    }

    /// Computes whether this d-DNNF is satisfiable.
    #[uniffi::method]
    fn is_sat(&self, assumptions: &[i32]) -> bool {
        self.0.lock().unwrap().sat(assumptions)
    }

    /// Computes the core features of this d-DNNF.
    #[uniffi::method]
    fn core(&self, assumptions: &[i32]) -> Vec<i32> {
        self.0.lock().unwrap().core_with_assumptions(assumptions)
    }

    /// Computes the core features of this d-DNNF for multiple variables.
    #[uniffi::method]
    fn core_multiple(&self, assumptions: &[i32], variables: &[i32]) -> Vec<i32> {
        let mut ddnnf = self.0.lock().unwrap();
        util::zip_assumptions_variables(assumptions, variables)
            .flat_map(|assumptions| ddnnf.core_with_assumptions(&assumptions).into_iter())
            .sorted()
            .dedup()
            .collect()
    }

    /// Computes the dead features of this d-DNNF.
    #[uniffi::method]
    fn dead(&self, assumptions: &[i32]) -> Vec<i32> {
        self.0.lock().unwrap().dead_with_assumptions(assumptions)
    }

    /// Computes the dead features of this d-DNNF for multiple variables.
    #[uniffi::method]
    fn dead_multiple(&self, assumptions: &[i32], variables: &[i32]) -> Vec<i32> {
        let mut ddnnf = self.0.lock().unwrap();
        util::zip_assumptions_variables(assumptions, variables)
            .flat_map(|assumptions| ddnnf.dead_with_assumptions(&assumptions).into_iter())
            .sorted()
            .dedup()
            .collect()
    }

    /// Generates satisfiable configurations for this d-DNNF.
    #[uniffi::method]
    fn enumerate(&self, assumptions: &[i32], amount: usize) -> Vec<Vec<i32>> {
        let mut assumptions = assumptions.to_vec();
        self.0
            .lock()
            .unwrap()
            .enumerate(&mut assumptions, amount)
            .unwrap_or_default()
    }

    /// Generates random satisfiable configurations for this d-DNNF.
    #[uniffi::method]
    fn random(&self, assumptions: &[i32], amount: usize, seed: u64) -> Vec<Vec<i32>> {
        self.0
            .lock()
            .unwrap()
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
            .get_atomic_sets(candidates, assumptions, cross)
    }
}
