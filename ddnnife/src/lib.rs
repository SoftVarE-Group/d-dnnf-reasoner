pub mod config;
pub mod ddnnf;
pub mod int_hash;
pub mod parser;
pub mod rand;
pub mod util;

use num::BigUint;
use num::rational::Ratio;

pub use crate::ddnnf::{Ddnnf, DdnnfKind, node::*};
pub use crate::parser::c2d_lexer;
pub use crate::parser::d4_lexer;

pub mod cnf;

/// Unsigned arbitrary precision rational.
///
/// As [num::BigRational], but based on [num::BigUint].
type BigUrational = Ratio<BigUint>;
