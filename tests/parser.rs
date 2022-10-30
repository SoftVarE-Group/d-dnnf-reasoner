extern crate ddnnf_lib;

use ddnnf_lib::data_structure::{Ddnnf, NodeType::*};
use ddnnf_lib::parser;

use rug::Integer;

#[test]
fn token_parsing_test() {
    let mut ddnnf: Ddnnf =
        parser::build_ddnnf_tree_with_extras("./tests/data/small_test.dimacs.nnf");

    assert_eq!(ddnnf.number_of_variables, 4);

    let or_node = ddnnf.nodes.pop().unwrap();
    match or_node.ntype {
        Or { children } => {
            assert_eq!(children.len(), 2_usize);
            assert_eq!(or_node.count, Integer::from(4_u32))
        },
        _ => panic!("Node isn't an or node")
    }

    let and_node = ddnnf.nodes.pop().unwrap();
    match and_node.ntype {
        And { children } => {
            assert_eq!(children.len(), 2_usize);
            assert_eq!(and_node.count, Integer::from(2_u32))
        },
        _ => panic!("Node isn't an and node")
    }
}
