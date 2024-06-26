use ddnnife::ddnnf::Ddnnf;
use ddnnife::parser::{self, persisting::write_ddnnf_to_file};
use file_diff::diff_files;
use std::fs;
use std::fs::File;

#[test]
fn card_of_features_normal_and_reloaded_test() {
    // default way to compute card of features with a d-DNNF in d4 standard
    let d4_out = "./tests/data/auto1_d4_fs.csv";
    let mut ddnnf: Ddnnf = parser::build_ddnnf("./tests/data/auto1_d4.nnf", Some(2513));
    ddnnf.card_of_each_feature(d4_out).unwrap_or_default();

    // save nnf in c2d format
    let saved_nnf = "./tests/data/auto1_d4_to_c2d.nnf";
    write_ddnnf_to_file(&ddnnf, saved_nnf).unwrap();

    // compute the cardinality of features for the saved file
    let saved_out = "./tests/data/auto1_d4_to_c2d_fs.csv";
    let mut ddnnf: Ddnnf = parser::build_ddnnf(saved_nnf, None);
    ddnnf.card_of_each_feature(saved_out).unwrap_or_default();

    // compare the results
    let mut is_d4 = File::open(d4_out).unwrap();
    let mut is_saved = File::open(saved_out).unwrap();

    assert!(diff_files(&mut is_d4, &mut is_saved));

    fs::remove_file(d4_out).unwrap();
    fs::remove_file(saved_nnf).unwrap();
    fs::remove_file(saved_out).unwrap();
}
