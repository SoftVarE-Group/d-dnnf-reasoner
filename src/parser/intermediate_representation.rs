use petgraph::{stable_graph::{StableGraph, NodeIndex}, visit::DfsPostOrder};
use rustc_hash::FxHashMap;

use crate::{c2d_lexer::TId, Node, NodeType};

use super::{calc_and_count, calc_or_count};

#[derive(Clone, Debug)]
pub struct IntermediateGraph {
    graph: StableGraph::<TId, ()>,
    root: NodeIndex,
    nx_literals: FxHashMap<NodeIndex, i32>
}

impl Default for IntermediateGraph {
    fn default() -> Self {
        IntermediateGraph {
            graph: StableGraph::<TId, ()>::default(),
            root: NodeIndex::default(),
            nx_literals: FxHashMap::default()
        }
    }
}


impl IntermediateGraph {
    pub fn new(graph: StableGraph::<TId, ()>, root: NodeIndex, nx_literals: FxHashMap<NodeIndex, i32>) -> IntermediateGraph {
        IntermediateGraph { graph, root, nx_literals }       
    }

    pub fn redo_nodes(&self) {
        // perform a depth first search to get the nodes ordered such
        // that child nodes are listed before their parents
        // transform that interim representation into a node vector
        let mut dfs = DfsPostOrder::new(&self.graph, self.root);
        let mut nd_to_usize: FxHashMap<NodeIndex, usize> = FxHashMap::default();

        let mut parsed_nodes: Vec<Node> = Vec::with_capacity(self.graph.node_count());
        let mut literals: FxHashMap<i32, usize> = FxHashMap::default();
        let mut true_nodes = Vec::new();

        while let Some(nx) = dfs.next(&self.graph) {
            nd_to_usize.insert(nx, parsed_nodes.len());
            let neighs = self.graph
                .neighbors(nx)
                .map(|n| *nd_to_usize.get(&n).unwrap())
                .collect::<Vec<usize>>();
            let next: Node = match self.graph[nx] {
                // extract the parsed Token
                TId::PositiveLiteral |
                TId::NegativeLiteral => {
                    Node::new_literal(self.nx_literals.get(&nx).unwrap().to_owned())
                }
                TId::And => Node::new_and(
                    calc_and_count(&mut parsed_nodes, &neighs),
                    neighs,
                ),

                TId::Or => Node::new_or(
                    0,
                    calc_or_count(&mut parsed_nodes, &neighs),
                    neighs,
                ),
                TId::True => Node::new_bool(true),
                TId::False => Node::new_bool(false),
                TId::Header => panic!("The d4 standard does not include a header!"),
            };

            match &next.ntype {
                NodeType::And { children } |
                NodeType::Or { children } => {
                    let next_indize: usize = parsed_nodes.len();
                    for &i in children {
                        parsed_nodes[i].parents.push(next_indize);
                    }
                }
                // fill the FxHashMap with the literals
                NodeType::Literal { literal } => {
                    literals.insert(*literal, parsed_nodes.len());
                }
                NodeType::True => {
                    true_nodes.push(parsed_nodes.len());
                }
                _ => (),
            }

            parsed_nodes.push(next);
        }
    }
}