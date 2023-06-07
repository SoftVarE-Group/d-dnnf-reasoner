use std::collections::{HashMap, HashSet};

use petgraph::{stable_graph::{StableGraph, NodeIndex}, visit::{DfsPostOrder, Bfs}, algo::is_cyclic_directed};

use crate::{c2d_lexer::TId, Node, NodeType, parser::get_literal_diffs};

use super::{calc_and_count, calc_or_count};

/// The IntermediateGraph enables us to modify the dDNNF. The structure of a vector of nodes does not allow
/// for that because deleting or removing nodes would mess up the indices. 
#[derive(Clone, Debug, Default)]
pub struct IntermediateGraph {
    graph: StableGraph::<TId, ()>,
    root: NodeIndex,
    nx_literals: HashMap<NodeIndex, i32>,
    literal_children: HashMap<NodeIndex, HashSet<i32>>
}

impl IntermediateGraph {
    /// Creates a new IntermediateGraph 
    pub fn new(graph: StableGraph::<TId, ()>, root: NodeIndex, nx_literals: HashMap<NodeIndex, i32>) -> IntermediateGraph {
        debug_assert!(!is_cyclic_directed(&graph));
        let mut inter_graph = IntermediateGraph { graph, root, nx_literals,
            literal_children: HashMap::new() };
        inter_graph.literal_children = get_literal_diffs(&inter_graph.graph, &inter_graph.nx_literals, inter_graph.root);
        inter_graph
    }

    /// Starting for the IntermediateGraph, we do a PostOrder walk through the graph the create the
    /// list of nodes which we use for counting operations and other types of queries.
    pub fn rebuild(&self) -> (Vec<Node>, HashMap<i32, usize>, Vec<usize>)  {
        // always make sure that there are no cycles
        debug_assert!(!is_cyclic_directed(&self.graph));

        // perform a depth first search to get the nodes ordered such
        // that child nodes are listed before their parents
        // transform that interim representation into a node vector
        let mut dfs = DfsPostOrder::new(&self.graph, self.root);
        let mut nd_to_usize: HashMap<NodeIndex, usize> = HashMap::new();

        let mut parsed_nodes: Vec<Node> = Vec::with_capacity(self.graph.node_count());
        let mut literals: HashMap<i32, usize> = HashMap::new();
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
                TId::NegativeLiteral => Node::new_literal(
                    self.nx_literals.get(&nx).unwrap().to_owned()
                ),
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
                // build additional references from the child to its parent
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

        (parsed_nodes, literals, true_nodes)
    }

    pub fn closest_unsplittable_and(&mut self, clause: &[i32]) -> (usize, HashSet<i32>) {
        use crate::c2d_lexer::TokenIdentifier::*;

        if clause.is_empty() { return (0, HashSet::default()) }

        let mut last_cached_and: NodeIndex<u32> = NodeIndex::new(usize::MAX);
        let mut last_cached_literals: &HashSet<i32> = &HashSet::default();
        let mut bfs = Bfs::new(&self.graph, self.root);
        while let Some(nx) = bfs.next(&self.graph) {
            match self.graph[nx] {
                And => {
                    let diffs = self.literal_children.get(&nx).unwrap();
                    if clause.iter().all(|e| diffs.contains(e)) 
                        && (diffs.len() < last_cached_literals.len() && diffs.is_subset(last_cached_literals)
                            || last_cached_literals.is_empty()) {
                        last_cached_and = nx;
                        last_cached_literals = diffs;
                    }
                },
                _ => (), // we are only interested in AND nodes
            }
        }
        (last_cached_and.index(), last_cached_literals.clone())
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use crate::parser::build_ddnnf;

    #[test]
    fn closest_unsplittable_and() {
        let mut ddnnf = build_ddnnf("tests/data/VP9_d4.nnf", Some(42));
        let _nodes = ddnnf.inter_graph.rebuild().0;

        let input = vec![vec![], vec![4], vec![5], vec![4, 5], vec![42]];
        let output = vec![vec![], vec![-5, 4], vec![-4, 5], vec![-5, -4, -3, 4, 5], vec![-41, 42]];

        for (index, inp) in input.iter().enumerate() {
            let mut literals_as_vec = HashSet::<_>::from_iter(
                (ddnnf.inter_graph.closest_unsplittable_and(inp)).1.iter().copied())
                .into_iter()
                .collect::<Vec<i32>>();
            literals_as_vec.sort();
            assert_eq!(output[index], literals_as_vec);
        }
    }
}