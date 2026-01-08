use super::{calc_and_count, calc_or_count};
use crate::c2d_lexer::TId;
use crate::{DdnnfKind, Node, NodeType};
use petgraph::algo::is_cyclic_directed;
use petgraph::prelude::DfsPostOrder;
use petgraph::stable_graph::{NodeIndex, StableGraph};
use std::collections::HashMap;

pub type DdnnfGraph = StableGraph<TId, ()>;

pub fn rebuild_graph(
    graph: DdnnfGraph,
    root: NodeIndex,
) -> (DdnnfKind, Vec<Node>, HashMap<i32, usize>) {
    // Check for special cases where there is a boolean root node.
    match graph[root] {
        TId::True => return (DdnnfKind::Tautology, Vec::new(), HashMap::new()),
        TId::False => return (DdnnfKind::Contradiction, Vec::new(), HashMap::new()),
        _ => {}
    }

    // always make sure that there are no cycles
    debug_assert!(!is_cyclic_directed(&graph));

    // perform a depth first search to get the nodes ordered such
    // that child nodes are listed before their parents
    // transform that interim representation into a node vector
    let mut dfs = DfsPostOrder::new(&graph, root);
    let mut nd_to_usize: HashMap<NodeIndex, usize> = HashMap::new();

    let mut parsed_nodes: Vec<Node> = Vec::with_capacity(graph.node_count());
    let mut literals: HashMap<i32, usize> = HashMap::new();

    while let Some(nx) = dfs.next(&graph) {
        nd_to_usize.insert(nx, parsed_nodes.len());
        let neighs = graph
            .neighbors(nx)
            .map(|n| *nd_to_usize.get(&n).unwrap())
            .collect::<Vec<usize>>();
        let next: Node = match graph[nx] {
            // extract the parsed Token
            TId::Literal { feature } => Node::new_literal(feature),
            TId::And => Node::new_and(calc_and_count(&mut parsed_nodes, &neighs), neighs),
            TId::Or => Node::new_or(calc_or_count(&mut parsed_nodes, &neighs), neighs),
            TId::True | TId::False => panic!("Boolean nodes are only allowed as root nodes."),
            TId::Header => {
                panic!("The d4 standard does not include a header!")
            }
        };

        match &next.ntype {
            // build additional references from the child to its parent
            NodeType::And { children } | NodeType::Or { children } => {
                let next_indize: usize = parsed_nodes.len();
                for &i in children {
                    parsed_nodes[i].parents.push(next_indize);
                }
            }
            // create mapping from literal to its node index
            NodeType::Literal { literal } => {
                literals.insert(*literal, parsed_nodes.len());
            }
        }

        parsed_nodes.push(next);
    }

    (DdnnfKind::NonTrivial, parsed_nodes, literals)
}
