//#![warn(missing_docs)]
#![warn(unused_qualifications)]
//#![deny(unreachable_pub)]
#![deny(deprecated)]
#![deny(missing_copy_implementations)]

#[cfg(all(test, feature = "benchmarks"))]
extern crate test;

pub mod parser;
pub use crate::parser::d4_lexer;
pub use crate::parser::c2d_lexer;

pub mod data_structure;
pub use crate::data_structure::{Ddnnf, Node};

pub mod stream;
pub use crate::stream::*;