use crate::DdnnfMut;
use ddnnife::ddnnf;
use num::BigInt;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

type DdnnfKind = ddnnife::DdnnfKind;

/// Value for indicating whether the d-DNNF is a special case: tautology or contradiction.
/// If it is neither a tautology nor a contradiction, it is non-trivial.
#[uniffi::remote(Enum)]
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub enum DdnnfKind {
    #[default]
    NonTrivial,
    Tautology,
    Contradiction,
}

/// A wrapped version of a d-DNNF.
#[derive(uniffi::Object, Clone)]
pub struct Ddnnf(pub(crate) ddnnife::Ddnnf);

#[uniffi::export]
impl Ddnnf {
    /// Loads a d-DNNF from file.
    #[uniffi::constructor]
    fn from_file(path: String, features: Option<u32>) -> Self {
        Self(ddnnf::Ddnnf::from_file(Path::new(&path), features))
    }

    /// Returns the d-DNNF kind.
    #[uniffi::method]
    pub fn kind(&self) -> DdnnfKind {
        self.0.kind
    }

    /// Checks whether this d-DNNF is trivial. A trivial d-DNNF is either a tautology or a
    /// contradiction.
    #[uniffi::method]
    pub fn is_trivial(&self) -> bool {
        self.0.is_trivial()
    }

    /// Returns the current count of the root node in the d-DNNF.
    ///
    /// This value is the same during all computations.
    #[uniffi::method]
    pub fn rc(&self) -> BigInt {
        self.0.rc()
    }

    /// Returns the core features of this d-DNNF.
    ///
    /// This is only calculated once at creation of the d-DNNF.
    #[uniffi::method]
    pub fn get_core(&self) -> HashSet<i32> {
        self.0.get_core()
    }

    /// Generates the c2d format representation of this d-DNNF.
    #[uniffi::method]
    pub fn serialize(&self) -> String {
        self.0.to_string()
    }

    /// Saves this d-DNNF to the given file.
    #[uniffi::method]
    fn save(&self, path: &str) {
        let mut file = File::create(path).expect("Failed to open file");
        file.write_all(self.0.to_string().as_bytes())
            .expect("Failed to serialize d-DNNF");
    }

    /// Creates a mutable copy of this d-DNNF.
    #[uniffi::method]
    fn as_mut(&self) -> DdnnfMut {
        DdnnfMut(Mutex::new(self.clone()))
    }
}
