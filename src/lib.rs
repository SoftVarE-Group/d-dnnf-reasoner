//! The ddnnife library provides different kinds of queries on dDNNFs.
//! This includes atomic sets, enumeration of valid configurations, uniform random sampling,
//! core and dead features, false optional features, SAT solving, and t-wise sampling.
//! The main focus of this library is computation speed.

#![warn(missing_docs)]
#![warn(unused_qualifications)]
#![deny(unreachable_pub)]
// #![deny(deprecated)]
#![deny(missing_copy_implementations)]
#![warn(clippy::disallowed_types)]

#[cfg(all(test, feature = "benchmarks"))]
extern crate test;

/// Parsing related functions. From the file input to the internal data structure representation.
pub mod parser;
pub use crate::parser::c2d_lexer;
pub use crate::parser::d4_lexer;
pub use crate::parser::intermediate_representation::IncrementalStrategy;

/// The actual dDNNF with its methods to compute different kinds of queries and operations.
pub mod ddnnf;
pub use crate::ddnnf::{node::*, Ddnnf};
