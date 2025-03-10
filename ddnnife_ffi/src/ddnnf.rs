use crate::DdnnfMut;
use ddnnife::ddnnf;
use ddnnife::parser::persisting::write_ddnnf_to_file;
use num::BigInt;
use std::collections::HashSet;
use std::sync::Mutex;

/// A wrapped version of a d-DNNF.
#[derive(uniffi::Object, Clone)]
pub struct Ddnnf(pub(crate) ddnnife::Ddnnf);

#[uniffi::export]
impl Ddnnf {
    /// Loads a d-DNNF from file.
    #[uniffi::constructor]
    fn from_file(path: String, features: Option<u32>) -> Self {
        Self(ddnnf::Ddnnf::from_file(path, features))
    }

    /// Loads a d-DNNF from file, using the projected d-DNNF compilation.
    ///
    /// Panics when not including d4 as it is required for projected compilation.
    #[uniffi::constructor]
    fn from_file_projected(path: String, features: Option<u32>) -> Self {
        Self(ddnnf::Ddnnf::from_file_projected(path, features))
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

    /// Saves this d-DNNF to the given file.
    #[uniffi::method]
    fn save(&self, path: &str) {
        write_ddnnf_to_file(&self.0, path).unwrap();
    }

    /// Creates a mutable copy of this d-DNNF.
    #[uniffi::method]
    fn as_mut(&self) -> DdnnfMut {
        DdnnfMut(Mutex::new(self.clone()))
    }
}
