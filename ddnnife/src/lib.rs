//#![warn(missing_docs)]
#![warn(unused_qualifications)]
//#![deny(unreachable_pub)]
#![deny(deprecated)]
#![deny(missing_copy_implementations)]
#![warn(clippy::disallowed_types)]
#[cfg(all(test, feature = "benchmarks"))]
extern crate test;

pub mod parser;
pub mod util;
pub use crate::parser::c2d_lexer;
pub use crate::parser::d4_lexer;

pub mod ddnnf;
pub use crate::ddnnf::{node::*, Ddnnf};
