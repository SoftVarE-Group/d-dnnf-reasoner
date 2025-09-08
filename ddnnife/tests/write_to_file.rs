use ddnnife::ddnnf::Ddnnf;
use ddnnife::parser;
use file_diff::diff_files;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[test]
fn card_of_features_normal_and_reloaded_test() {
    // default way to compute card of features with a d-DNNF in d4 standard
    let d4_out = Path::new("./tests/data/auto1_d4_fs.csv");
    let mut ddnnf: Ddnnf = parser::build_ddnnf(Path::new("./tests/data/auto1_d4.nnf"), Some(2513));
    ddnnf.card_of_each_feature_csv(d4_out).unwrap_or_default();

    // save nnf in c2d format
    let saved_nnf = Path::new("./tests/data/auto1_d4_to_c2d.nnf");
    let mut file = File::create(&saved_nnf).unwrap();
    file.write_all(ddnnf.to_string().as_bytes()).unwrap();

    // compute the cardinality of features for the saved file
    let saved_out = Path::new("./tests/data/auto1_d4_to_c2d_fs.csv");
    let mut ddnnf: Ddnnf = parser::build_ddnnf(saved_nnf, None);
    ddnnf
        .card_of_each_feature_csv(saved_out)
        .unwrap_or_default();

    // compare the results
    let mut is_d4 = File::open(d4_out).unwrap();
    let mut is_saved = File::open(saved_out).unwrap();

    assert!(diff_files(&mut is_d4, &mut is_saved));

    fs::remove_file(d4_out).unwrap();
    fs::remove_file(saved_nnf).unwrap();
    fs::remove_file(saved_out).unwrap();
}
