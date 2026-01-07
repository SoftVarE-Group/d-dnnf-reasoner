mod header;

use header::Header;
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};

pub type Variable = usize;
pub type Literal = isize;
pub type Clause = Vec<Literal>;

#[derive(Debug, Default, Clone, Eq, PartialEq, Hash)]
pub struct Cnf {
    pub num_variables: usize,
    pub clauses: Vec<Clause>,
}

impl Cnf {
    /// Creates a new CNF with model count 0.
    pub fn with_count_0() -> Self {
        Self {
            num_variables: 1,
            clauses: vec![vec![1], vec![-1]],
        }
    }

    /// Creates a new CNF with model count 1.
    pub fn with_count_1() -> Self {
        Self {
            num_variables: 1,
            clauses: vec![vec![1]],
        }
    }
}

impl Display for Cnf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let num_clauses = self.clauses.len();

        let header = Header {
            num_clauses,
            num_variables: self.num_variables,
        };

        writeln!(f, "{header}")?;

        self.clauses
            .iter()
            .enumerate()
            .try_for_each(|(index, clause)| {
                clause
                    .iter()
                    .try_for_each(|literal| write!(f, "{literal} "))?;

                write!(f, "0")?;

                if index < num_clauses - 1 {
                    writeln!(f)?;
                }

                Ok(())
            })
    }
}

impl FromIterator<Clause> for Cnf {
    fn from_iter<T: IntoIterator<Item = Clause>>(iter: T) -> Self {
        let clauses: Vec<Clause> = iter.into_iter().collect();
        let variables: BTreeSet<Variable> = clauses
            .iter()
            .flat_map(|clause| clause.iter())
            .map(|literal| literal.unsigned_abs())
            .collect();

        Cnf {
            num_variables: variables.len(),
            clauses,
        }
    }
}

impl From<Vec<Vec<Literal>>> for Cnf {
    fn from(value: Vec<Vec<Literal>>) -> Self {
        value.into_iter().collect()
    }
}

#[cfg(test)]
mod test {
    use super::Cnf;

    #[test]
    fn display_cnf() {
        let cnf = Cnf {
            num_variables: 3,
            clauses: vec![vec![1, -2, 3]],
        };

        let expected = r#"p cnf 3 1
1 -2 3 0"#;

        assert_eq!(cnf.to_string(), expected);
    }
}
