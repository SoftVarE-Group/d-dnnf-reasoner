extern crate ddnnf_lib;

use ddnnf_lib::data_structure::Ddnnf;
use ddnnf_lib::parser;

use file_diff::diff_files;
use std::fs::File;
use std::fs;

#[test]
fn card_of_features_test() {
    let c2d_out = "./tests/data/auto1_c2d_fs.csv";
    let mut ddnnf: Ddnnf =
        parser::build_ddnnf_tree_with_extras("./tests/data/auto1_c2d.nnf");
    ddnnf
        .card_of_each_feature_to_csv(c2d_out)
        .unwrap_or_default();

    let mut should = File::open("./tests/data/auto1_sb_fs.csv").unwrap();
    let mut is = File::open(c2d_out).unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
    let _res = fs::remove_file(c2d_out);

    let d4_out = "./tests/data/auto1_d4_fs.csv";
    let mut ddnnf: Ddnnf =
        parser::build_d4_ddnnf_tree("./tests/data/auto1_d4.nnf", 2513);
    ddnnf
        .card_of_each_feature_to_csv(d4_out)
        .unwrap_or_default();

    let mut should = File::open("./tests/data/auto1_sb_fs.csv").unwrap();
    let mut is = File::open(d4_out).unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
    let _res = fs::remove_file(d4_out);
}

#[test]
fn card_of_pc_test() {
    let c2d_out = "./tests/data/auto1_c2d_pc.txt";
    let d4_out = "./tests/data/auto1_d4_pc.txt";
    let sb_file_path = "./tests/data/auto1_sb_pc.txt";
    let config_file = "./tests/data/auto1.config";

    let mut ddnnf: Ddnnf =
        parser::build_ddnnf_tree_with_extras("tests/data/auto1_c2d.nnf");
    ddnnf.max_worker = 1;
    ddnnf
        .card_multi_queries(
            config_file,
            c2d_out,
        )
        .unwrap_or_default();

    let mut should = File::open(sb_file_path).unwrap();
    let mut is = File::open(c2d_out).unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
    let _res = fs::remove_file(c2d_out);

    let mut ddnnf: Ddnnf =
        parser::build_d4_ddnnf_tree("tests/data/auto1_d4.nnf", 2513);
    ddnnf.max_worker = 1;
    ddnnf
        .card_multi_queries(
            config_file,
            d4_out,
        )
        .unwrap_or_default();

    let mut should = File::open(sb_file_path).unwrap();
    let mut is = File::open(d4_out).unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
    let _res = fs::remove_file(d4_out);
}

#[test]
fn heuristics_test() {
    let mut ddnnf: Ddnnf =
        parser::build_ddnnf_tree("./tests/data/auto1_c2d.nnf");
    ddnnf.print_all_heuristics();
}
