use std::fmt::{Display, Formatter};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Header {
    pub num_variables: usize,
    pub num_clauses: usize,
}

impl Display for Header {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "p cnf {} {}", self.num_variables, self.num_clauses)
    }
}
