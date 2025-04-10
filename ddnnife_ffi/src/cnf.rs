use crate::ddnnf::Ddnnf;
use ddnnife_cnf::Clause;

#[uniffi::export]
impl Ddnnf {
    /// Generates an equi-countable CNF representation via Tseitin transformation.
    #[uniffi::method]
    pub fn to_cnf(&self) -> Cnf {
        Cnf(ddnnife_cnf::Cnf::from(&self.0))
    }
}

#[derive(uniffi::Object)]
pub struct Cnf(pub ddnnife_cnf::Cnf);

#[uniffi::export]
impl Cnf {
    /// Generates the DIMACS representation.
    #[uniffi::method]
    pub fn serialize(&self) -> String {
        self.0.to_string()
    }

    /// Returns the number of variables in the CNF.
    #[uniffi::method]
    pub fn num_variables(&self) -> usize {
        self.0.num_variables
    }

    /// Returns a copy of the clause set.
    #[uniffi::method]
    pub fn clauses(&self) -> Vec<Clause> {
        self.0.clauses.clone()
    }
}
