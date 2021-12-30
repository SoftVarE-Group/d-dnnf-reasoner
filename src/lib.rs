//#![warn(missing_docs)]
#![warn(unused_qualifications)]
#![deny(unreachable_pub)]
#![deny(deprecated)]
#![deny(missing_copy_implementations)]

#[cfg(all(test, feature = "benchmarks"))]
extern crate test;

pub mod parser;
pub use crate::parser::{build_ddnnf_tree_with_extras, parse_features, parse_queries_file};

pub mod data_structure;
pub use crate::data_structure::{Ddnnf};