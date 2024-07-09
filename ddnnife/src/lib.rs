#[cfg(feature = "uniffi")]
uniffi::setup_scaffolding!();

pub mod parser;
pub mod util;
pub use crate::parser::c2d_lexer;
pub use crate::parser::d4_lexer;

pub mod ddnnf;
pub use crate::ddnnf::{node::*, Ddnnf};

#[cfg(feature = "uniffi")]
mod ffi;
