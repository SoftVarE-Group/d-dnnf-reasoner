use ddnnife::ddnnf::Ddnnf;
use ddnnife::parser;
use file_diff::diff_files;
use serial_test::serial;
use std::fs::{self, File};
use std::io::BufWriter;

#[test]
fn card_of_features_c2d() {
    let c2d_out = "./tests/data/auto1_c2d_fs.csv";
    let mut ddnnf: Ddnnf = parser::build_ddnnf("./tests/data/auto1_c2d.nnf", None);
    ddnnf.card_of_each_feature_csv(c2d_out).unwrap_or_default();

    let mut should = File::open("./tests/data/auto1_sb_fs.csv").unwrap();
    let mut is = File::open(c2d_out).unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
    let _res = fs::remove_file(c2d_out);
}

#[test]
fn card_of_features_d4() {
    let d4_out = "./tests/data/auto1_d4_fs.csv";
    let mut ddnnf: Ddnnf = parser::build_ddnnf("./tests/data/auto1_d4.nnf", Some(2513));
    ddnnf.card_of_each_feature_csv(d4_out).unwrap_or_default();

    let mut should = File::open("./tests/data/auto1_sb_fs.csv").unwrap();
    let mut is = File::open(d4_out).unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
    let _res = fs::remove_file(d4_out);
}

#[cfg(feature = "d4")]
#[test]
#[serial]
fn card_of_features_cnf() {
    let cnf_out = "./tests/data/auto1_cnf_fs.csv";
    let mut ddnnf: Ddnnf = parser::build_ddnnf("./tests/data/auto1.cnf", None);
    ddnnf.card_of_each_feature_csv(cnf_out).unwrap_or_default();

    let mut should = File::open("./tests/data/auto1_sb_fs.csv").unwrap();
    let mut is = File::open(cnf_out).unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
    let _res = fs::remove_file(cnf_out);
}

#[test]
fn card_of_pc_c2d() {
    let c2d_out = "./tests/data/auto1_c2d_pc.csv";
    let sb_file_path = "./tests/data/auto1_sb_pc.csv";
    let config_file = "./tests/data/auto1.config";

    let output = BufWriter::new(File::create(c2d_out).expect("Unable to create file"));

    let mut ddnnf: Ddnnf = parser::build_ddnnf("tests/data/auto1_c2d.nnf", None);
    ddnnf.max_worker = 1;
    ddnnf
        .operate_on_queries(Ddnnf::execute_query, config_file, output)
        .unwrap_or_default();

    let mut should = File::open(sb_file_path).unwrap();
    let mut is = File::open(c2d_out).unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
    fs::remove_file(c2d_out).unwrap();
}

#[test]
fn card_of_pc_d4() {
    let d4_out = "./tests/data/auto1_d4_pc.csv";
    let sb_file_path = "./tests/data/auto1_sb_pc.csv";
    let config_file = "./tests/data/auto1.config";

    let output = BufWriter::new(File::create(d4_out).expect("Unable to create file"));

    let mut ddnnf: Ddnnf = parser::build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));
    ddnnf.max_worker = 1;
    ddnnf
        .operate_on_queries(Ddnnf::execute_query, config_file, output)
        .unwrap_or_default();

    let mut should = File::open(sb_file_path).unwrap();
    let mut is = File::open(d4_out).unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
    fs::remove_file(d4_out).unwrap();
}

#[cfg(feature = "d4")]
#[test]
#[serial]
fn card_of_pc_cnf() {
    let cnf_out = "./tests/data/auto1_cnf_pc.csv";
    let sb_file_path = "./tests/data/auto1_sb_pc.csv";
    let config_file = "./tests/data/auto1.config";

    let output = BufWriter::new(File::create(cnf_out).expect("Unable to create file"));

    let mut ddnnf: Ddnnf = parser::build_ddnnf("tests/data/auto1.cnf", None);
    ddnnf.max_worker = 1;
    ddnnf
        .operate_on_queries(Ddnnf::execute_query, config_file, output)
        .unwrap_or_default();

    let mut should = File::open(sb_file_path).unwrap();
    let mut is = File::open(cnf_out).unwrap();

    // diff_files is true if the files are identical
    assert!(diff_files(&mut should, &mut is));
    fs::remove_file(cnf_out).unwrap();
}

#[test]
fn heuristics_test() {
    let mut ddnnf: Ddnnf = parser::build_ddnnf("./tests/data/auto1_c2d.nnf", None);
    ddnnf.print_all_heuristics();
}