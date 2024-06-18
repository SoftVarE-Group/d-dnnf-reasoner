use ddnnife::ddnnf::{node::NodeType::*, Ddnnf};
use ddnnife::parser;
use num::BigInt;

#[test]
fn ddnnf_parsing_test() {
    let ddnnf_d4: Ddnnf = parser::build_ddnnf("./tests/data/small_ex_d4.nnf", Some(4));

    let mut ddnnf_c2d: Ddnnf = parser::build_ddnnf("./tests/data/small_ex_c2d.nnf", None);

    assert_eq!(ddnnf_c2d.number_of_variables, 4);
    assert_eq!(ddnnf_c2d.rc(), BigInt::from(4));
    assert_eq!(ddnnf_c2d.nodes.len(), 12);

    assert_eq!(ddnnf_d4.number_of_variables, 4);
    assert_eq!(ddnnf_d4.rc(), BigInt::from(4));
    assert_eq!(ddnnf_d4.nodes.len(), 18);

    let and_node = ddnnf_c2d.nodes.pop().unwrap();
    match and_node.ntype {
        And { children } => {
            assert_eq!(children.len(), 3_usize);
            assert_eq!(and_node.count, BigInt::from(4))
        }
        _ => panic!("Node isn't an and node"),
    }

    let or_node = ddnnf_c2d.nodes.pop().unwrap();
    match or_node.ntype {
        Or { children } => {
            assert_eq!(children.len(), 2_usize);
            assert_eq!(or_node.count, BigInt::from(2))
        }
        _ => panic!("Node isn't an or node"),
    }
}
