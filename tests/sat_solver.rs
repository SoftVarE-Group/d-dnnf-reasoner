extern crate ddnnf_lib;

use std::collections::HashMap;
use rug::Integer;
use ddnnf_lib::{Ddnnf, Node};
use ddnnf_lib::sampler::sat_solver::SatSolver;

#[test]
fn sat_solving() {
    let ddnnf = create_ddnnf();
    let solver = SatSolver::new(&ddnnf);

    // basic SAT solving
    assert_eq!(solver.is_sat(&[1]), true);
    assert_eq!(solver.is_sat(&[-1]), false);

    // state caching
    let mut state = solver.new_state();
    assert_eq!(solver.is_sat_cached(&[1], &mut state), true); // init state by calculating SAT for [1]
    // use cached state to efficiently calculate SAT for [1, 2] and [1, -2]
    assert_eq!(solver.is_sat_cached(&[2], &mut state.clone()), true);
    assert_eq!(solver.is_sat_cached(&[-2], &mut state.clone()), true);

    // subgraph SAT
    assert_eq!(solver.is_sat_in_subgraph(&[1,2], 3), true); // SAT in subgraph rooted at 3
    assert_eq!(solver.is_sat_in_subgraph(&[1,2], 4), false); // not SAT in subgraph rooted at 4
}

fn create_ddnnf() -> Ddnnf {
    let mut nodes: Vec<Node> = vec![
        Node::new_literal(1), // 0
        Node::new_literal(2), // 1
        Node::new_literal(-2), // 2
        Node::new_and(Integer::from(1), vec![0, 1]), // 3
        Node::new_and(Integer::from(1), vec![0, 2]), // 4
        Node::new_or(2, Integer::from(2), vec![3, 4]), // 5
    ];
    nodes[0].parents.push(3);
    nodes[0].parents.push(4);
    nodes[1].parents.push(3);
    nodes[2].parents.push(4);
    nodes[3].parents.push(5);
    nodes[4].parents.push(5);
    let mut literals: HashMap<i32, usize> = HashMap::new();
    literals.insert(1, 0);
    literals.insert(2, 1);
    literals.insert(-2, 2);

    let number_of_variables = 2;
    let number_of_nodes = 6;
    Ddnnf::new(nodes, literals, number_of_variables, number_of_nodes)
}