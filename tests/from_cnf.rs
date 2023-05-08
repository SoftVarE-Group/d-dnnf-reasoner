extern crate ddnnf_lib;

use assert_cmd::prelude::*; // Add methods on commands
use predicates::prelude::*; // Used for writing assertions
use std::process::Command; // Run programs

#[test]
fn compiles_cnf_to_d_dnnf() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("ddnnife")?;

    cmd.arg("tests/data/auto1.cnf");

    // compiling from CNF to dDNNF
    cmd.assert().stdout(predicate::str::contains(
    "Compiling dDNNF from CNF file..."
    ));
    // count is correct
    cmd.assert().stdout(predicate::str::contains(
        "Ddnnf overall count: 54337953889526644797436357304783500234473556203012469981705794070419609376066883019863858681556047971579366711252721976681982553481954710208375451836305175948768348959659511355551303323044387225600000000000000000000000"
    ));

    Ok(())
}