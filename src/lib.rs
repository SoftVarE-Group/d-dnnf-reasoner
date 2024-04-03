//#![warn(missing_docs)]
#![warn(unused_qualifications)]
//#![deny(unreachable_pub)]
#![deny(deprecated)]
#![deny(missing_copy_implementations)]
#![warn(clippy::disallowed_types)]
#[cfg(all(test, feature = "benchmarks"))]
extern crate test;

pub mod ddnnf;
pub mod maybe_parallel;
pub mod parser;

pub use crate::ddnnf::{node::*, Ddnnf};
pub use crate::parser::c2d_lexer;
pub use crate::parser::d4_lexer;
