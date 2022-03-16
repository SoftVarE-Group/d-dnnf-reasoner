extern crate ddnnf_lib;

use ddnnf_lib::data_structure::Ddnnf;
use ddnnf_lib::parser;

use file_diff::diff_files;
use std::fs::File;

#[test]
fn card_of_features_test() {
    let mut ddnnf: Ddnnf =
        parser::build_ddnnf_tree_with_extras("./tests/data/axTLS.dimacs.nnf");
    ddnnf
        .card_of_each_feature_to_csv("./tests/data/axTLS_features_out.csv")
        .unwrap_or_default();

    let mut should = File::open("./tests/data/axTLS_should_be_fs.csv").unwrap();
    let mut is = File::open("./tests/data/axTLS_features_out.csv").unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
}

#[test]
fn card_of_pc_test() {
    let mut ddnnf: Ddnnf =
        parser::build_ddnnf_tree_with_extras("tests/data/axTLS.dimacs.nnf");
    ddnnf.max_worker = 1;
    ddnnf
        .card_multi_queries(
            "./tests/data/axTLS.config",
            "./tests/data/axTLS_pc_out.txt",
        )
        .unwrap_or_default();

    let mut should = File::open("./tests/data/axTLS_should_be_pc.txt").unwrap();
    let mut is = File::open("./tests/data/axTLS_pc_out.txt").unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
}

#[test]
fn heuristics_test() {
    let mut ddnnf: Ddnnf =
        parser::build_ddnnf_tree("./tests/data/axTLS.dimacs.nnf");
    ddnnf.print_all_heuristics();
}
