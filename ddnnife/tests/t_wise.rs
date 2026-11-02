use ddnnife::ddnnf::Ddnnf;
use std::path::Path;

#[test]
fn only_core() {
    let ddnnf: Ddnnf = Ddnnf::from_file(Path::new("./tests/data/VP9_d4.nnf"), Some(42));

    let result = ddnnf.sample_t_wise(2, Some(&[2].into_iter().collect()));
    assert_eq!(
        result.get_sample().unwrap().len(),
        1,
        "With only a core literal, there should be one config."
    );

    let result = ddnnf.sample_t_wise(2, Some(&[2, -2].into_iter().collect()));
    assert_eq!(
        result.get_sample().unwrap().len(),
        1,
        "With only a core literal, there should be one config."
    );
}
