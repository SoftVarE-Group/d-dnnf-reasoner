uniffi::setup_scaffolding!();

mod cnf;
mod ddnnf;
mod ddnnf_mut;
mod statistics;
mod t_wise_sampling;
mod types;

use ddnnf::Ddnnf;
use ddnnf_mut::DdnnfMut;
use std::iter;

fn zip_assumptions_variables<'a>(
    assumptions: &'a [i32],
    variables: &'a [i32],
) -> Box<dyn Iterator<Item = Vec<i32>> + 'a> {
    if assumptions.is_empty() && variables.is_empty() {
        return Box::new(iter::once(Vec::new()));
    }

    if assumptions.is_empty() {
        return Box::new(variables.iter().copied().map(|variable| vec![variable]));
    }

    if variables.is_empty() {
        return Box::new(iter::once(assumptions.to_vec()));
    }

    Box::new(
        iter::repeat(assumptions)
            .zip(variables.iter())
            .map(|(assumptions, &variable)| {
                let mut assumptions = assumptions.to_vec();
                assumptions.push(variable);
                assumptions
            }),
    )
}
