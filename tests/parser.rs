extern crate ddnnf_lib;

use ddnnf_lib::data_structure::{Ddnnf, NodeType::*};
use ddnnf_lib::parser;

use rug::Integer;

#[test]
fn token_parsing_test() {
    let mut ddnnf: Ddnnf =
        parser::build_ddnnf_tree("./tests/data/small_test.dimacs.nnf");

    assert_eq!(ddnnf.number_of_variables, 4);

    let or_node = ddnnf.nodes.pop().unwrap();
    let or_node_childs = or_node.children.unwrap();

    assert_eq!(or_node_childs.len().clone(), 2_usize);
    assert_eq!(or_node.node_type, Or);
    assert_eq!(or_node.count, Integer::from(4));

    assert_eq!(ddnnf.nodes[or_node_childs[0]].node_type, And);
    assert_eq!(ddnnf.nodes[or_node_childs[1]].node_type, And);

    let and_node_childs = ddnnf.nodes.pop().unwrap().children.unwrap();
    assert_eq!(and_node_childs.len(), 2_usize);
    assert_eq!(ddnnf.nodes[and_node_childs[0]].node_type, Or);
    assert_eq!(ddnnf.nodes[and_node_childs[1]].node_type, And);
}
