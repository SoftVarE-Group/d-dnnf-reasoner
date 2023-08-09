use std::{collections::{HashMap, HashSet}, fs::{File, self}, io::Write, time::Instant, path::Path, cmp::{max, Reverse}, panic};

use itertools::Itertools;
use petgraph::{
    stable_graph::{StableGraph, NodeIndex},
    visit::{DfsPostOrder, Dfs, Bfs, NodeIndexable},
    algo::is_cyclic_directed,
    Direction::{Incoming, Outgoing}
};

use crate::{c2d_lexer::TId, Node, NodeType, parser::{util::format_vec, build_ddnnf, extend_literal_diffs, from_cnf::{get_all_clauses_cnf, simplify_clauses}, persisting::write_as_mermaid_md}, Ddnnf};

use super::{calc_and_count, calc_or_count, d4v2_wrapper::compile_cnf, from_cnf::{reduce_clause, apply_decisions}};

/// The IntermediateGraph enables us to modify the dDNNF. The structure of a vector of nodes does not allow
/// for that because deleting or removing nodes would mess up the indices. 
#[derive(Clone, Debug, Default)]
pub struct IntermediateGraph {
    graph: StableGraph::<TId, Option<Vec<i32>>>,
    pub root: NodeIndex,
    number_of_variables: u32,
    tseitin_offset: i32,
    literals_nx: HashMap<i32, NodeIndex>,
    literal_children: HashMap<NodeIndex, HashSet<i32>>,
    cnf_clauses: Vec<Vec<i32>>
}

const CHUNKS: f64 = 1.0;
const DEBUG: bool = false;

impl IntermediateGraph {
    /// Creates a new IntermediateGraph 
    pub fn new(graph: StableGraph::<TId, Option<Vec<i32>>>, root: NodeIndex, number_of_variables: u32,
               literals_nx: HashMap<i32, NodeIndex>, literal_children: HashMap<NodeIndex, HashSet<i32>>,
               cnf_path: Option<&str>) -> IntermediateGraph {
        debug_assert!(!is_cyclic_directed(&graph));
        let mut inter_graph = IntermediateGraph {
            graph, root, tseitin_offset: 1_000_000, number_of_variables,
            literals_nx, literal_children,
            cnf_clauses: match cnf_path {
                Some(path) => get_all_clauses_cnf(path),
                None => Vec::new()
            }
        };
        inter_graph.add_decision_nodes_fu();
        inter_graph.cnf_clauses = simplify_clauses(inter_graph.cnf_clauses);
        inter_graph
    }

    // move all values from other to self
    fn move_inter_graph(&mut self, other: IntermediateGraph) {
        self.graph = other.graph;
        self.root = other.root;
        self.number_of_variables = other.number_of_variables;
        self.tseitin_offset = other.tseitin_offset;
        self.literals_nx = other.literals_nx;
        self.literal_children = other.literal_children;
        self.cnf_clauses = other.cnf_clauses;
    }

    /// Starting for the IntermediateGraph, we do a PostOrder walk through the graph the create the
    /// list of nodes which we use for counting operations and other types of queries.
    pub fn rebuild(&mut self, alt_root: Option<NodeIndex>) -> (Vec<Node>, HashMap<i32, usize>, Vec<usize>)  {
        // always make sure that there are no cycles
        debug_assert!(!is_cyclic_directed(&self.graph));

        // perform a depth first search to get the nodes ordered such
        // that child nodes are listed before their parents
        // transform that interim representation into a node vector
        let mut dfs = DfsPostOrder::new(&self.graph, alt_root.unwrap_or(self.root));
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
                TId::Literal { feature } => Node::new_literal(
                    feature
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
                // create mapping from literal to its node index
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

    pub fn closest_unsplitable_bridge(&mut self, clause: &[i32]) -> Option<(NodeIndex, HashSet<i32>)> {
        if clause.is_empty() { return None }
        let mut closest_node = (self.root, self.literal_children.get(&self.root).unwrap());
        
        let mut bridge_end_point = Vec::new();        
        for bridge_endpoint in self.find_bridges() {
            match self.graph[bridge_endpoint] {
                TId::And | TId::Or => {
                    let diffs = self.literal_children.get(&bridge_endpoint).unwrap();
                    // we only consider and nodes that include all literals of the clause
                    if clause.iter().all(|e| diffs.contains(e) && diffs.contains(&-e)) {
                        if closest_node.1.len() > diffs.len() {
                            closest_node = (bridge_endpoint, diffs);
                        }
                    }
                    bridge_end_point.push(bridge_endpoint);
                }
                _ => ()
            }
        }

        let devided_and = self.divide_bridge(closest_node.0, &bridge_end_point, clause);
        Some((devided_and, self.literal_children.get(&devided_and).unwrap().clone()))
    }

    pub fn _find_bridges_rec(&self) -> HashSet<NodeIndex> {
        let mut bridges = HashSet::new();
        let mut visited = vec![false; self.graph.node_count()];
        let mut tin = vec![0usize; self.graph.node_count()];
        let mut low = vec![0usize; self.graph.node_count()];
        let mut timer = 0usize;
    
        self.dfs_helper(
            self.graph.to_index(self.root),
            self.graph.node_count() + 1,
            &mut bridges,
            &mut visited,
            &mut timer,
            &mut tin,
            &mut low,
        );
        bridges
    }
    
    #[allow(clippy::too_many_arguments)]
    fn dfs_helper(
        &self,
        v: usize,
        p: usize,
        bridges: &mut HashSet<NodeIndex>,
        visited: &mut Vec<bool>,
        timer: &mut usize,
        tin: &mut Vec<usize>,
        low: &mut Vec<usize>,
    ) {
        visited[v] = true;
        *timer += 1;
        tin[v] = *timer;
        low[v] = *timer;
    
        let neighbours_inc = self.graph.neighbors_directed(self.graph.from_index(v), Incoming);
        let neighbours_out = self.graph.neighbors_directed(self.graph.from_index(v), Outgoing);

        for n in neighbours_inc.chain(neighbours_out) {
            let to = self.graph.to_index(n);
            if to == p {
                continue;
            }
            if visited[to] {
                low[v] = low[v].min(tin[to]);
            } else {
                self.dfs_helper(to, v, bridges, visited, timer, tin, low);
                low[v] = low[v].min(low[to]);
                if low[to] > tin[v] {
                    bridges.insert(self.graph.from_index(to));
                }
            }
        }
    }

    // Evgeny: https://stackoverflow.com/questions/23179579/finding-bridges-in-graph-without-recursion
    #[inline]
    fn find_bridges(&self) -> HashSet<NodeIndex> {
        let mut bridges = HashSet::new();

        // calculate neighbours beforehand, because we have to index them multiple times
        let mut neighbours = vec![Vec::new(); self.graph.node_bound()];
        for i in 0..self.graph.node_count() {
            let neighbours_inc = self.graph.neighbors_directed(self.graph.from_index(i), Incoming);
            let neighbours_out = self.graph.neighbors_directed(self.graph.from_index(i), Outgoing);
            neighbours[i] = neighbours_inc.chain(neighbours_out).map(|nx| self.graph.to_index(nx)).collect::<Vec<usize>>();
        }

        let mut visited = vec![false; self.graph.node_bound()];
        let mut tin = vec![0usize; self.graph.node_bound()]; // time-in
        let mut low = vec![0usize; self.graph.node_bound()]; // f-up (minimum discovery time)
        let mut timer = 0usize;
        let mut stack: Vec<(usize, usize, usize)> = Vec::new(); // (node, parent) pairs
    
        stack.push((self.root.index(), self.graph.node_bound() + 1, 0));

        while let Some((v, parent, i)) = stack.pop() {
            if i == 0 {
                visited[v] = true;
                timer += 1;
                tin[v] = timer;
                low[v] = timer;
            }
        
            if i < neighbours[v].len() {
                let to = neighbours[v][i];
                stack.push((v, parent, i + 1));
                if to != parent {
                    if visited[to] {
                        low[v] = low[v].min(tin[to]);
                    } else {
                        stack.push((to, v, 0));
                    }
                }
            }
            if i > 0 && i <= neighbours[v].len() {
                let to = neighbours[v][i - 1];
                if to != parent {
                    low[v] = low[v].min(low[to]);
                    if low[to] > tin[v] {
                        bridges.insert(self.graph.from_index(to));
                    }
                }
            }
        }

        bridges
    }


    pub fn closest_unsplitable_and0(&mut self, clause: &[i32]) -> Option<(NodeIndex, HashSet<i32>)> {
        use crate::c2d_lexer::TokenIdentifier::*;

        if clause.is_empty() { return None }

        let mut cached_ands: Vec<(NodeIndex<u32>, &HashSet<i32>)> = Vec::new();
        let mut bfs = Bfs::new(&self.graph, self.root);
        while let Some(nx) = bfs.next(&self.graph) {
            match self.graph[nx] {
                And => {
                    let diffs = self.literal_children.get(&nx).unwrap();
                    // we only consider and nodes that include all literals of the clause
                    if clause.iter().all(|e| diffs.contains(e) && diffs.contains(&-e)) {
                        cached_ands.push((nx, diffs));
                    }
                },
                _ => (), // we are only interested in AND nodes
            }
        }
        
        // sort by descending length, aka from closest to farthest from root
        cached_ands.sort_unstable_by_key(|and| Reverse(and.1.len()));
        let mut try_and = 
            if !cached_ands.is_empty() {
                cached_ands[0]
            } else {
                return Some((self.root, self.literal_children.get(&self.root).unwrap().clone()));
            };

        if DEBUG {
            println!("cached ands: {:?}", cached_ands);
        }

        // TODO FIX ME
        for i in 0..cached_ands.len() {
            if cached_ands[i+1..].iter()
                .all(|(_nx, and)| and.is_subset(cached_ands[i].1)) {
                try_and = cached_ands[i];
            } else {
                break;
            }
        }
        if DEBUG { 
            println!("Bridge AND: {:?}", Some((try_and.0, try_and.1.clone())));
        }
        Some((try_and.0, try_and.1.clone()))
    }

    #[inline]
    fn first_mark(&mut self, set: &mut HashSet<NodeIndex>, current: NodeIndex) {
        if set.contains(&current) { return; }
        set.insert(current);

        let mut parents = self.graph.neighbors_directed(current, Incoming).detach();
        while let Some(parent) = parents.next_node(&self.graph) {
            self.first_mark(set, parent);
        }
    }

    /// For a given clause we search for the AND node that contains all literals of that clause
    /// and therefore all other clauses that contain those literals and that has as little children
    /// as possible.
    /// If we can't find any suitable literal (for instance the clause contains a new literal)
    /// or the best AND node is the root node, we return new solution because our best option is to recompile
    /// the whole CNF.
    pub fn closest_unsplitable_and(&mut self, clause: &[i32]) -> Option<(NodeIndex, &HashSet<i32>)> {
        use crate::c2d_lexer::TokenIdentifier::*;

        if clause.is_empty() { return None }

        let mut sets = Vec::new();

        for literal in clause {
            for literal_pair in [literal, &-literal] {
                match self.literals_nx.get(literal_pair) {
                    // A literal node can contain multiple parents. We need to make sure that each direct
                    // parent gets its own set to ensure including all connections towards the literal within
                    // out target AND node.
                    Some(lit) => {
                        let mut literal_parents = 
                        self.graph.neighbors_directed(*lit, Incoming).detach();
                    
                        while let Some(parent) = literal_parents.next_node(&self.graph) {
                            let mut set = HashSet::new();
                            self.first_mark(&mut set, parent);
                            //println!("{:?}", set.len());
                            sets.push(set);
                        }
                    },
                    // Feature might be core or dead to this point.
                    // Hence, we can't find a matching literal and we don't have to replace anything.
                    None => (),
                }
                //println!("----------sign swap----------");
            }
            //println!("----------next literal----------");
        }

        // TODO Handle complete new clauses
        if sets.is_empty() { return None; }

        // Get the intersection of all sets
        let mut common_binaries: Box<dyn Iterator<Item = NodeIndex>> =
        Box::new(sets.pop().into_iter().flatten());
        
        for current_set in sets {
            common_binaries = Box::new(common_binaries.filter(move |it| current_set.contains(it)));
        }

        // Find the AND node in that intersection set that contains the least amount of literals in its children.
        // This directly gives us the AND node with the minimum amount of nodes we have to convert to CNF and back.
        let mut cached_and: Option<(_, &HashSet<i32>)> = None;
        for node in common_binaries.collect::<Vec<_>>() {
            match self.graph[node] {
                And => {
                    let diffs = self.literal_children.get(&node).unwrap();
                    cached_and = match cached_and {
                        Some((old_node, old_diffs)) => {
                            //println!("cached_and old: {:?} {:?}, new: {:?} {:?}", old_node, old_diffs.len(), node, diffs.len());
                            if old_diffs.len() < diffs.len() {
                                Some((old_node, old_diffs))
                            } else {
                                Some((node, diffs))
                            }
                        },
                        None => Some((node, diffs)),
                    }
                },
                _ => ()
            }
        }
        //println!("cached_and result: {:?}", cached_and.unwrap().1.len());
        cached_and
    }

    /// For a given clause we search for the AND node that contains all literals of that clause
    /// and therefore all other clauses that contain those literals and that has as little children
    /// as possible.
    /// If we can't find any suitable literal (for instance the clause contains a new literal)
    /// or the best AND node is the root node, we return new solution because our best option is to recompile
    /// the whole CNF.
    pub fn closest_unsplitable_ands(&mut self, clause: &[i32]) -> Option<Vec<(NodeIndex, HashSet<i32>)>> {
        if clause.is_empty() { return None }

        let mut and_nodes: Vec<(NodeIndex, HashSet<i32>)> = Vec::new();
        for literal in clause {
            for literal_pair in [literal, &-literal] {
                match self.literals_nx.get(literal_pair) {
                    // A literal node can contain multiple parents. We need to make sure that each direct
                    // parent gets its own set to ensure including all connections towards the literal within
                    // out target AND node.
                    Some(lit) => {
                        let mut literal_parents = 
                        self.graph.neighbors_directed(*lit, Incoming).detach();
                    
                        while let Some(parent) = literal_parents.next_node(&self.graph) {
                            let first_encounter = self.get_first_and_encounter(parent);
                            let lits_encounter = self.literal_children.get(&first_encounter).unwrap();

                            and_nodes.push((first_encounter, lits_encounter.clone()));
                        }
                    },
                    // Feature might be core or dead to this point.
                    // Hence, we can't find a matching literal and we don't have to replace anything.
                    None => (),
                }
            }
        }

        if and_nodes.is_empty() { return None; }

        // Remove duplicates based on NodeIndex
        and_nodes.sort_by_key(|(nx, _)| nx.index());
        and_nodes.dedup_by_key(|(node_index, _)| *node_index);

        let mut and_nodes_c = and_nodes.clone();
        and_nodes_c.retain(|(_, set)| {
            !and_nodes.iter().any(|(_, other_set)| set.is_subset(other_set) && set != other_set)
        });

        Some(and_nodes_c)
    }

    #[inline]
    fn get_first_and_encounter(&mut self, current: NodeIndex) -> NodeIndex {
        use crate::c2d_lexer::TokenIdentifier::*;

        return match self.graph[current] {
            And => current,
            _ => {
                let mut parents = self.graph.neighbors_directed(current, Incoming).detach();
                while let Some(parent) = parents.next_node(&self.graph) {
                    return self.get_first_and_encounter(parent);
                }
                panic!("dDNNF does not contain a root AND node!");
            },
        }
    }

    /// For a given clause we search for the AND node that contains all literals of that clause
    /// and therefore all other clauses that contain those literals and that has as little children
    /// as possible.
    /// If we can't find any suitable literal (for instance the clause contains a new literal)
    /// or the best AND node is the root node, we return new solution because our best option is to recompile
    /// the whole CNF.
    pub fn closest_unsplitable_ands2(&mut self, clause: &[i32]) -> Option<Vec<(NodeIndex, HashSet<i32>)>> {
        use crate::c2d_lexer::TokenIdentifier::*;

        if clause.is_empty() { return None }

        let mut sets = Vec::new();
        let mut starters = HashSet::new();

        for literal in clause {
            for literal_pair in [literal, &-literal] {
                match self.literals_nx.get(literal_pair) {
                    // A literal node can contain multiple parents. We need to make sure that each direct
                    // parent gets its own set to ensure including all connections towards the literal within
                    // out target AND node.
                    Some(lit) => {
                        starters.insert(lit.clone());
                        let mut literal_parents = 
                        self.graph.neighbors_directed(*lit, Incoming).detach();
                    
                        while let Some(parent) = literal_parents.next_node(&self.graph) {
                            let mut set = HashSet::new();
                            self.first_mark(&mut set, parent);
                            sets.push(set);
                        }
                    },
                    // Feature might be core or dead to this point.
                    // Hence, we can't find a matching literal and we don't have to replace anything.
                    None => (),
                }
            }
        }

        // TODO Handle complete new clauses
        if sets.is_empty() { return None; }
        println!("AND_2 -> STARTER SIZE: {}, SETS COUNT: {}", starters.len(), sets.len());

        // Split the vector into three equal-sized chunks
        let chunks = sets.chunks((sets.len() as f64 / CHUNKS).ceil() as usize);
        // Collect the chunks into a vector of vectors
        let result: Vec<Vec<_>> = chunks.map(|chunk| chunk.to_vec()).collect();
    
        let mut cached_ands = Vec::new();

        for mut chunk in result {
            // Get the intersection of all sets
            let mut common_binaries: Box<dyn Iterator<Item = NodeIndex>> =
            Box::new(chunk.pop().into_iter().flatten());
            
            for current_set in chunk {
                common_binaries = Box::new(common_binaries.filter(move |it| current_set.contains(it)));
            }

            // Find the AND node in that intersection set that contains the least amount of literals in its children.
            // This directly gives us the AND node with the minimum amount of nodes we have to convert to CNF and back.
            let mut cached_and: Option<(_, &HashSet<i32>)> = None;
            for node in common_binaries.collect::<Vec<_>>() {
                match self.graph[node] {
                    And => {
                        let diffs = self.literal_children.get(&node).unwrap();
                        cached_and = match cached_and {
                            Some((old_node, old_diffs)) => {
                                if old_diffs.len() < diffs.len() {
                                    Some((old_node, old_diffs))
                                } else {
                                    Some((node, diffs))
                                }
                            },
                            None => Some((node, diffs)),
                        }
                    },
                    _ => ()
                }
            }
            match cached_and {
                Some((nx, lits)) => cached_ands.push((nx, lits.clone())),
                None => (),
            }
        }

        if cached_ands.is_empty() { return None; }

        // Remove duplicates based on NodeIndex
        cached_ands.sort_by_key(|(nx, _)| nx.index());
        cached_ands.dedup_by_key(|(node_index, _)| *node_index);

        let mut and_nodes_c = cached_ands.clone();
        and_nodes_c.retain(|(_, set)| {
            !cached_ands.iter().any(|(_, other_set)| set.is_subset(other_set) && set != other_set)
        });

        Some(and_nodes_c)
    }

    pub fn closest_unsplitable_ands3(&mut self, clause: &[i32]) -> Option<Vec<(NodeIndex, HashSet<i32>)>> {
        use crate::c2d_lexer::TokenIdentifier::*;

        if clause.is_empty() { return None }

        let mut sets = Vec::new();

        let starting_points_reduced = self.starting_points_from_reduce_clause(clause);
        for starting_point in starting_points_reduced {
            let mut set = HashSet::new();
            self.first_mark(&mut set, starting_point);
            sets.push(set);
        }

        // TODO Handle complete new clauses
        if sets.is_empty() { return None; }
        println!("AND_3 -> STARTER SIZE: {}, SETS COUNT: {}", self.starting_points_from_reduce_clause(clause).len(), sets.len());

        // Split the vector into three equal-sized chunks
        let chunks = sets.chunks((sets.len() as f64 / CHUNKS).ceil() as usize);
        // Collect the chunks into a vector of vectors
        let result: Vec<Vec<_>> = chunks.map(|chunk| chunk.to_vec()).collect();
    
        let mut cached_ands = Vec::new();

        for mut chunk in result {
            // Get the intersection of all sets
            let mut common_binaries: Box<dyn Iterator<Item = NodeIndex>> =
            Box::new(chunk.pop().into_iter().flatten());
            
            for current_set in chunk {
                common_binaries = Box::new(common_binaries.filter(move |it| current_set.contains(it)));
            }

            // Find the AND node in that intersection set that contains the least amount of literals in its children.
            // This directly gives us the AND node with the minimum amount of nodes we have to convert to CNF and back.
            let mut cached_and: Option<(_, &HashSet<i32>)> = None;
            for node in common_binaries.collect::<Vec<_>>() {
                match self.graph[node] {
                    And => {
                        let diffs = self.literal_children.get(&node).unwrap();
                        cached_and = match cached_and {
                            Some((old_node, old_diffs)) => {
                                if old_diffs.len() < diffs.len() {
                                    Some((old_node, old_diffs))
                                } else {
                                    Some((node, diffs))
                                }
                            },
                            None => Some((node, diffs)),
                        }
                    },
                    _ => ()
                }
            }
            match cached_and {
                Some((nx, lits)) => cached_ands.push((nx, lits.clone())),
                None => (),
            }
        }

        if cached_ands.is_empty() { return None; }

        // Remove duplicates based on NodeIndex
        cached_ands.sort_by_key(|(nx, _)| nx.index());
        cached_ands.dedup_by_key(|(node_index, _)| *node_index);

        let mut and_nodes_c = cached_ands.clone();
        and_nodes_c.retain(|(_, set)| {
            !cached_ands.iter().any(|(_, other_set)| set.is_subset(other_set) && set != other_set)
        });

        Some(and_nodes_c)
    }

    // Change the structure by splitting one AND/OR node into two.
    // In the following, we focus on the new AND'/OR' instead of AND/OR, resulting
    // in a smaller portion of the dDNNF, we have to travers.
    //
    // We can exclude all children fulfill:
    //      1) don't contain any of the variables of the clause
    //      2) the edge beteen the AND an its child is a bridge
    //
    //
    // Example (with c1 and c2 as the relevant_children):
    //
    //           AND                        AND
    //      ______|______      =>    _______|_______
    //     /   /  |  \   \          |      |   \    \
    //    c1  c2 c3  ... cn        AND'   c3   ...  cn
    //                            /   \
    //                           c1   c2
    //
    fn divide_bridge(&mut self, initial_start: NodeIndex, bridges: &[NodeIndex], clause: &[i32]) -> NodeIndex {
        if clause.is_empty() { return initial_start; }
        
        // Find those children that are relevant to the provided clause.
        // Those all contain at least one of the literals of the clause.
        let mut relevant_children = Vec::new();
        let mut contains_at_least_one_irrelevant = false;
        let mut children = self.graph.neighbors_directed(initial_start, Outgoing).detach();
        while let Some(child) = children.next_node(&self.graph) {
            let lits = self.literal_children.get(&child).unwrap();
            if clause.iter().any(|f| lits.contains(f) || lits.contains(&-f))
                || bridges.contains(&child) {
                relevant_children.push(child);
            } else {
                contains_at_least_one_irrelevant = true;
            }
        }

        // If there is no irrelevant child, it makes no sense to divide the AND node.
        if !contains_at_least_one_irrelevant {
            return initial_start;
        }

        // Create the new node and adjust the edges.
        let new_node = self.graph.add_node(self.graph[initial_start]);
        let mut new_node_lits = HashSet::new();
        for child in relevant_children {
            if let Some(edge) = self.graph.find_edge(initial_start, child) {
                self.graph.remove_edge(edge);
            }
            self.graph.add_edge(new_node, child, None);

            new_node_lits.extend(self.literal_children.get(&child).unwrap());
        }
        self.graph.add_edge(initial_start, new_node, None);

        // Add child literals of new_and to mapping
        self.literal_children.insert(new_node, new_node_lits);

        new_node
    }

    // Change the structure by splitting one AND node into two.
    // In the following, we focus on the new AND' instead of AND, resulting
    // in a smaller portion of the dDNNF, we have to travers
    // Example (with c1 and c2 as the relevant_children):
    //
    //           AND                        AND
    //      ______|______      =>    _______|_______
    //     /   /  |  \   \          |      |   \    \
    //    c1  c2 c3  ... cn        AND'   c3   ...  cn
    //                            /   \
    //                           c1   c2
    //
    fn divide_and(&mut self, initial_start: NodeIndex, clause: &[i32]) -> NodeIndex {
        if clause.is_empty() { return initial_start; }
        
        // Find those children that are relevant to the provided clause.
        // Those all contain at least one of the literals of the clause.
        let mut relevant_children = Vec::new();
        let mut contains_at_least_one_irrelevant = false;
        let mut children = self.graph.neighbors_directed(initial_start, Outgoing).detach();
        while let Some(child) = children.next_node(&self.graph) {
            let lits = self.literal_children.get(&child).unwrap();
            if clause.iter().any(|f| lits.contains(f) || lits.contains(&-f)) {
                relevant_children.push(child);
            } else {
                contains_at_least_one_irrelevant = true;
            }
        }

        // If there is no irrelevant child, it makes no sense to divide the AND node.
        if !contains_at_least_one_irrelevant {
            return initial_start;
        }

        // Create the new AND node and adjust the edges.
        let new_and = self.graph.add_node(TId::And);
        let mut new_and_lits = HashSet::new();
        for child in relevant_children {
            if let Some(edge) = self.graph.find_edge(initial_start, child) {
                self.graph.remove_edge(edge);
            }
            self.graph.add_edge(new_and, child, None);

            new_and_lits.extend(self.literal_children.get(&child).unwrap());
        }
        self.graph.add_edge(initial_start, new_and, None);

        // Add child literals of new_and to mapping
        self.literal_children.insert(new_and, new_and_lits);

        new_and
    }

    /// From a starting point in the dDNNF, we transform that subgraph into the CNF format,
    /// using Tseitin's transformation.
    /// Besides the CNF itself, the return type also gives a map to map the literals to their
    /// new corresponding number. That is necessary because the CNF format does not allow gaps in their
    /// variables. All the new literal indices have a lower index than the following Tseitin variables.
    pub fn transform_to_cnf_tseitin(&mut self, starting_point: NodeIndex, clause: &[i32]) -> (Vec<String>, NodeIndex, HashMap<i32, i32>) {
        let adjusted_starting_point = self.divide_and(starting_point, clause);
        let (nodes, _, _) = self.rebuild(Some(adjusted_starting_point));

        let initial_node_count = self.rebuild(None).0.len();
        print!(
            "Replacing {}/{} nodes. That are {:.3}%; ",
            nodes.len(), initial_node_count, (nodes.len() as f64 / initial_node_count as f64) * 100.0
        );

        let mut re_index_mapping: HashMap<i32, i32> = HashMap::new();
        let mut cnf = vec![String::from("p cnf ")];
        // compute the offset for the Tseitin variables. We need want to reserve
        let mut counter_set = self.literal_children
            .get(&adjusted_starting_point)
            .unwrap()
            .into_iter()
            .map(|v| v.unsigned_abs())
            .collect::<HashSet<u32>>();
        counter_set.extend(clause.iter().map(|v| v.unsigned_abs()));
        let mut counter = counter_set.len() as i32 + 1;

        let mut lit_counter = 1;
        let mut clause_var: Vec<i32> = std::iter::repeat(0).take(nodes.len() + clause.len()).collect::<Vec<_>>();

        let mut index = 0;
        for (_index, node) in nodes.iter().enumerate() {
            match &node.ntype {
                // Handle And and Or nodes like described in Tseitins Transformation for transforming
                // any arbitrary boolean formula into CNF. https://en.wikipedia.org/wiki/Tseytin_transformation
                NodeType::And { children } => {
                    for &child in children {
                        cnf.push(format!("{} {} 0\n", -counter, clause_var[child]));
                    }
                    cnf.push(format!("{} {} 0\n", counter, format_vec(children.iter().map(|&c| -clause_var[c]))));
                    
                    clause_var[index] = counter;
                    counter += 1;
                },
                NodeType::Or { children } => {
                    for &child in children {
                        cnf.push(format!("{} {} 0\n", counter, -clause_var[child]));
                    }
                    cnf.push(format!("{} {} 0\n", -counter, format_vec(children.iter().map(|&c| clause_var[c]))));

                    clause_var[index] = counter;
                    counter += 1;
                }
                NodeType::Literal { literal } => {
                    // Literals have to be mapped to a new index because we may have to
                    // transform parts of the dDNNF that do not contain all variables. The resulting
                    // gaps must be filled. Example: 1 5 42 -> 1 2 3.
                    let cached_re_index = re_index_mapping.get(&(literal.unsigned_abs() as i32));
                    let re_index;
                    if cached_re_index.is_some() {
                        re_index = *cached_re_index.unwrap();
                    } else {
                        re_index_mapping.insert(literal.unsigned_abs() as i32, lit_counter);
                        re_index = lit_counter;
                        lit_counter += 1;
                    }

                    clause_var[index] = if literal.is_positive() { re_index as i32 } else { -(re_index as  i32) };
                },
                _ => panic!("Node is of type: {:?} which is not allowed here!", node.ntype)
            }
            index += 1;
        }
        // add root as unit clause
        cnf.push(format!("{} 0\n", clause_var[nodes.len() - 1]));

        // add the new clause to the CNF
        if !clause.is_empty() {

            for c in clause {
                let cached_re_index = re_index_mapping.get(&(c.unsigned_abs() as i32));
                let re_index;
                if cached_re_index.is_some() {
                    re_index = *cached_re_index.unwrap();
                } else {
                    re_index_mapping.insert(c.unsigned_abs() as i32, lit_counter);
                    re_index = lit_counter;
                    lit_counter += 1;
                }

                clause_var[index] = if c.is_positive() { re_index as i32 } else { -(re_index as  i32) };
                
                index += 1;
            }

            cnf.push(format!("{} 0\n", format_vec(
                clause.iter()
                    .map(|f| {
                        // mapping must contain feature because we searched for the fitting AND node earlier
                        match re_index_mapping.get(&(f.unsigned_abs() as i32)) {
                            Some(&re_index) => if f.is_positive() { re_index as i32 } else { -(re_index as  i32) },
                            None => {
                                re_index_mapping.insert(f.unsigned_abs() as i32, lit_counter);
                                lit_counter += 1;
                                if f.is_positive() { (lit_counter - 1) as i32 } else { -((lit_counter - 1) as  i32) }
                            },
                        }
                    })
            )));
        }

        // add the header information about the number of variables and clauses
        let clause_count = cnf.len() - 1;
        cnf[0] += &format!("{} {}\n", counter - 1, clause_count);

        // Swaps key with value in the key value pairs
        let pairs: Vec<(i32, i32)> = re_index_mapping.drain().collect();
        for (key, value) in pairs {
            re_index_mapping.insert(value, key);
        }

        (cnf, adjusted_starting_point, re_index_mapping)
    }

    /// Experimental distributive translation method to get from dDNNF to CNF and back
    pub fn transform_to_cnf_distributive(&mut self, starting_point: NodeIndex, clause: &[i32]) -> (Vec<String>, NodeIndex) {
        let adjusted_starting_point = self.divide_and(starting_point, clause);
        let (nodes, _, _) = self.rebuild(Some(adjusted_starting_point));

        let mut clauses: Vec<Vec<Vec<i32>>> = Vec::new();
        let mut literals = HashSet::new();
        for node in nodes.iter() {
            use NodeType::*;
            match &node.ntype {
                And { children } => {
                    let mut and_clauses = Vec::new();
                    for &child in children {
                        and_clauses.extend_from_slice(&clauses[child]);
                    }
                    clauses.push(and_clauses);
                },
                Or { children } => {
                    let mut or_clauses = HashSet::new();
                    for &child in children {
                        if !clauses[child].is_empty() {
                            or_clauses.insert(clauses[child].clone());
                        }
                    }

                    let cartesian_product: HashSet<Vec<Vec<i32>>> = or_clauses.iter().cloned().multi_cartesian_product().collect();
                    let mut flattened_cp: HashSet<Vec<i32>> = cartesian_product.iter().map(|clause| clause.concat() ).collect();
                    
                    flattened_cp = flattened_cp.into_iter().map(|clause| clause.into_iter().unique().collect::<Vec<i32>>()).collect();

                    // remove the clauses that contain at least one literal as selected and deselected
                    flattened_cp.retain(|inner_vec| {
                        !inner_vec.iter().any(|&num| inner_vec.contains(&-num))
                    });

                    // remove clauses that are subsets of other clauses
                    let mut v:Vec<HashSet<i32>> = flattened_cp.iter()
                        .map(|x| x.iter().cloned().collect())
                        .collect();
            
                    let mut pos = 0;
                    while pos < v.len() {
                        let is_sub = v[pos+1..].iter().any(|x| v[pos].is_subset(x)) 
                            || v[..pos].iter().any(|x| v[pos].is_subset(x));
                
                        if is_sub {
                            v.swap_remove(pos);
                        } else {
                            pos+=1;
                        }
                    }

                    let w = v.into_iter()
                        .map(|clause| HashSet::<_>::from_iter(clause.iter().copied())
                        .into_iter()
                        .collect::<Vec<i32>>()).collect();

                    clauses.push(w);
                },
                Literal { literal } => {
                    literals.insert(literal.unsigned_abs());
                    clauses.push(vec![vec![*literal]])
                },
                _ => panic!("Node is of type: {:?} which is not allowed here!", node.ntype)
            }
        }
        let mut cnf = vec![format!("p cnf {} {}\n", literals.len(), clauses.last().unwrap().len())];
        for clause in clauses.last().unwrap() {
            cnf.push(format!("{} 0\n", format_vec(clause.iter())));
        }
        if !clause.is_empty() { cnf.push(format!("{} 0\n", format_vec(clause.iter()))); }
        (cnf, adjusted_starting_point)
    }

    // Experimental method to use the initial CNF for recompiling
    pub fn transform_to_cnf_from_starting_cnf(&mut self, clause: Vec<i32>) -> (Vec<String>, NodeIndex, HashMap<u32, u32>) {
        if self.cnf_clauses.is_empty() { return (Vec::new(), NodeIndex::new(0), HashMap::new()); }

        // 1) Find the closest node and the decisions to that point
        let (closest_node, relevant_literals) = match self.closest_unsplitable_bridge(&clause) {
            Some((nx, lits)) => (nx, lits.clone()),
            None => (self.root, HashSet::new()), // Just do the whole CNF without any adjustments
        };

        // Recompile the whole dDNNF if at least 80% are children of the closest_node we could replace
        if relevant_literals.len() as f32 / 2.0 > self.number_of_variables as f32 * 0.8 {
            return (Vec::new(), self.root, HashMap::new());
        }

        let mut accumlated_decisions = self.get_decisions_target_nx(closest_node);
        // Add unit clauses. Those decisions are implicit in the dDNNF.
        // Hence, we have to search for them
        self.cnf_clauses.iter().for_each(
            |clause| if clause.len() == 1 { accumlated_decisions.insert(clause[0]); });

        if DEBUG {
            println!("lits count: {} lits: {:?}", relevant_literals.len(), relevant_literals);
        }

        // 2) Filter for clauses that contain at least one of the relevant literals
        //    aka the literals that are children of the closest AND node
        let mut relevant_clauses: Vec<Vec<i32>> = if clause.is_empty() {
            self.cnf_clauses.clone()
        } else {
            self.cnf_clauses
                .clone()
                .into_iter()
                .filter(|initial_clause| 
                    initial_clause.iter()
                        .any(|elem| relevant_literals.contains(elem)
                            || relevant_literals.contains(&-elem)))
                .collect_vec()
        };
        relevant_clauses.push(clause.clone());

        if DEBUG {
            let mut removed_clauses = self.cnf_clauses.clone();
            removed_clauses.retain(|clause| !relevant_clauses.contains(clause));
            println!("removed clauses: {:?}", removed_clauses);
        }

        // 2.5 add each feature as optional to account for features that become optional
        //     due to the newly added clause.
        let mut variables = HashSet::new();
        for var in relevant_literals.iter() {
            variables.insert(var.unsigned_abs());
        }

        // 3) Repeatedly apply the summed up decions to the remaining clauses
        (relevant_clauses, accumlated_decisions) = apply_decisions(relevant_clauses, accumlated_decisions);

        let initial_clause = &clause;
        match reduce_clause(&clause, &accumlated_decisions) {
            Some(clause) => {
                match clause.len() {
                    0 => return (Vec::new(), self.root, HashMap::new()),
                    1 => {
                        self.add_unit_clause(clause[0]);
                        return (Vec::new(), self.root, HashMap::new())
                    },
                    _ => {
                        if initial_clause.len() > clause.len() {
                            println!("AND ANOTHER ROUND FROM CLAUSE {:?} TO CLAUSE {:?}", initial_clause, clause);
                            return self.transform_to_cnf_from_starting_cnf(clause);
                        }
                    }
                }
            },
            None => panic!("dDNNF becomes UNSAT for clause: {:?} under the decisions: {:?}",
                clause, accumlated_decisions),
        };

        // Continue 2.5
        let mut red_variables = HashSet::new();
        for clause in relevant_clauses.iter() {
            for variable in clause {
                red_variables.insert(variable.unsigned_abs());
            }
        }

        if DEBUG {
            println!("Potential optional features: {:?}", variables.symmetric_difference(&red_variables).cloned());
        }

        if DEBUG {
            println!("decisions: {:?}", accumlated_decisions.clone());
        }

        for variable in variables.difference(&red_variables).cloned() {
            let v_i32 = &(variable as i32);
            if !accumlated_decisions.contains(v_i32)
            && !accumlated_decisions.contains(&-v_i32)
            && (relevant_literals.contains(v_i32) || relevant_literals.contains(&-v_i32)) {
                relevant_clauses.push(vec![variable as i32, -(variable as i32)]);
            }

            if accumlated_decisions.contains(v_i32) {
                relevant_clauses.push(vec![*v_i32]);
            } else if accumlated_decisions.contains(&-v_i32) {
                relevant_clauses.push(vec![-v_i32]);
            }
        }

        if DEBUG {
            println!("relevant clause: {:?}", relevant_clauses);
        }

        // 4) Adjust indices of the literals
        //    (to avoid unwanted gaps in the CNF which would be interpreted as optional features)
        let mut index = 1_u32;
        let mut re_index: HashMap<u32,u32> = HashMap::new();
        for clause in relevant_clauses.iter_mut() {
            for elem in clause {
                let elem_signum = elem.signum();
                match re_index.get(&(elem.unsigned_abs())) {
                    Some(val) => { *elem = *val as i32 * elem_signum; },
                    None => {
                        re_index.insert(elem.unsigned_abs(), index);
                        *elem = index as i32 * elem_signum;
                        index += 1;
                    },
                }
            }
        }

        if DEBUG { println!("reindex: {:?}", re_index); }

        // write the meta information of the header
        let mut cnf = vec![format!("p cnf {} {}\n", index - 1, relevant_clauses.len() + 1)];
        
        for clause in relevant_clauses {
            cnf.push(format!("{} 0\n", format_vec(clause.iter())));
        }

        // Swaps key with value in the key value pairs
        let pairs: Vec<(u32, u32)> = re_index.drain().collect();
        for (key, value) in pairs {
            re_index.insert(value, key);
        }

        (cnf, closest_node, re_index)
    }

    pub fn add_clause(&mut self, clause: &[i32]) -> bool {
        if clause.len() == 1 { self.add_unit_clause(clause[0]); return true; }
        let mut _start = Instant::now();
        const INTER_CNF: &str = ".intermediate.cnf"; const INTER_NNF: &str = ".intermediate.nnf";
        let replace_multiple = match self.closest_unsplitable_ands2(&clause) {
            Some(and_node) => and_node,
            None => { return false; }
        };
        // if replace == self.root { return false; }
        println!("Replacing {} nodes", replace_multiple.len());

        for (replace, _) in replace_multiple {
            //println!("elapsed time: finding AND: {}", _start.elapsed().as_secs_f64());
            _start = Instant::now();
            let (cnf, adjusted_replace, re_indices) = self.transform_to_cnf_tseitin(replace, clause);
            println!("Tseitin CNF clauses: {}", cnf.len());
            if cnf.len() > 10_000 { return false; }
            //println!("elapsed time: tseitin: {}", _start.elapsed().as_secs_f64());

            _start = Instant::now();
            // persist CNF
            let cnf_flat = cnf.join("");
            let mut cnf_file = File::create(INTER_CNF).unwrap();
            cnf_file.write_all(cnf_flat.as_bytes()).unwrap();

            // transform the CNF to dDNNF and load it
            compile_cnf(INTER_CNF, INTER_NNF);
            let last_lit_number = re_indices.keys().map(|&k| k.unsigned_abs()).max().unwrap();
            if !Path::new(INTER_NNF).exists() {
                println!("There is no INTER.nnf file...");
                let contents = fs::read_to_string(INTER_CNF)
                .expect("Should have been able to read the file");
        
                println!("With text:\n{contents}");
                continue;
            }

            // Remove all edges of the old graph. We keep the nodes.
            // Most are useless but some are used in other parts of the graph. Hence, we keep them
            let mut dfs = DfsPostOrder::new(&self.graph, adjusted_replace);
            while let Some(nx) = dfs.next(&self.graph) {
                let mut children = self.graph.neighbors_directed(nx, Outgoing).detach();
                while let Some(edge_to_child) = children.next_edge(&self.graph) {
                    self.graph.remove_edge(edge_to_child);
                }
            }

            // If the new subgraph is unsatisfiable, we don't have to add anything
            if fs::read_to_string(INTER_NNF).unwrap() == "f 1 0\n" {
                println!("Unsatisfiable!");
                continue;
            }

            let sup_ddnnf = build_ddnnf(INTER_NNF, Some(last_lit_number));

            // add the new subgraph as unconnected additional graph
            let sub = sup_ddnnf.inter_graph;
            let mut dfs = DfsPostOrder::new(&sub.graph, sub.root);
            let mut cache = HashMap::new();
            let mut max_tseitin_value = 0;

            while let Some(nx) = dfs.next(&sub.graph) {
                let new_nx = match sub.graph[nx] {
                    TId::Literal { feature } => {
                        let re_lit = re_indices.get(&(feature.unsigned_abs() as i32));
                        if re_lit.is_some() {
                            let signed_lit = re_lit.unwrap() * feature.signum();
                            match self.literals_nx.get(&signed_lit) {
                                Some(&lit) => {
                                    lit
                                },
                                // Feature was core/dead in the starting dDNNF -> no occurence but also not tseitin
                                None => {
                                    let new_lit_nx = self.graph.add_node(TId::Literal { feature: signed_lit });
                                    self.literals_nx.insert(signed_lit, new_lit_nx);
                                    new_lit_nx
                                },
                            }
                        } else { // tseitin
                            max_tseitin_value = max(max_tseitin_value, feature.abs());
                            let offset_lit = if feature.is_positive() { feature + self.tseitin_offset } else { feature - self.tseitin_offset };
                            let new_lit = TId::Literal { feature: offset_lit };
                            let new_lit_nx = self.graph.add_node(new_lit);
                            self.literals_nx.insert(offset_lit, new_lit_nx);
                            new_lit_nx
                        }
                    },
                    _ => self.graph.add_node(sub.graph[nx]),
                };
                cache.insert(nx, new_nx);

                let mut children = sub.graph.neighbors_directed(nx, Outgoing).detach();
                while let Some(child) = children.next_node(&sub.graph) {
                    self.graph.add_edge(new_nx, *cache.get(&child).unwrap(), None);
                }
            }
            self.tseitin_offset += max_tseitin_value + 1;

            // remove the reference to the starting node with the new subgraph
            let new_sub_root = *cache.get(&sub.root).unwrap();
            let mut parents = self.graph.neighbors_directed(adjusted_replace, Incoming).detach();
            while let Some((parent_edge, parent_node)) = parents.next(&self.graph) {
                self.graph.remove_edge(parent_edge);
                self.graph.add_edge(parent_node, new_sub_root, None);
            }

            // add the literal_children of the newly created ddnnf
            extend_literal_diffs(&self.graph, &mut self.literal_children, new_sub_root);

            // clean up temp files
            if Path::new(INTER_CNF).exists() { fs::remove_file(INTER_CNF).unwrap(); }
            if Path::new(INTER_NNF).exists() { fs::remove_file(INTER_NNF).unwrap(); }
        }
        true
    }

    pub fn add_clause_alt(&mut self, clause: Vec<i32>) -> bool {
        if clause.len() == 1 { self.add_unit_clause(clause[0]); return true; }
        let mut _start = Instant::now();
        const INTER_CNF: &str = ".sub.cnf"; const INTER_NNF: &str = ".sub.nnf";

        //println!("elapsed time: finding AND: {}", _start.elapsed().as_secs_f64());
        _start = Instant::now();
        let (mut cnf, adjusted_replace, re_indices) = self.transform_to_cnf_from_starting_cnf(clause.clone());
        
        if cnf.is_empty() {
            if adjusted_replace == self.root {
                cnf = vec![format!("p cnf {} {}\n", self.number_of_variables, self.cnf_clauses.len() + 1)];
                for clause in self.cnf_clauses.iter() {
                    cnf.push(format!("{} 0\n", format_vec(clause.iter())));
                }
                cnf.push(format!("{} 0\n", format_vec(clause.iter())));

                let cnf_flat = cnf.join("");
                let mut cnf_file = File::create(INTER_CNF).unwrap();
                cnf_file.write_all(cnf_flat.as_bytes()).unwrap();

                compile_cnf(INTER_CNF, INTER_NNF);
                let sup = build_ddnnf(INTER_NNF, Some(self.number_of_variables));
                self.move_inter_graph(sup.inter_graph);
                self.rebuild(None);
                println!("recompiled everything");
            } else {
                println!("clause was redundant :D");
            }
            return true;
        }
        
        println!("Replace CNF clauses: {}", cnf.len());
        //if cnf.len() > 20_000 { return false; }
        //println!("elapsed time: tseitin: {}", _start.elapsed().as_secs_f64());

        _start = Instant::now();
        // persist CNF
        let cnf_flat = cnf.join("");
        let mut cnf_file = File::create(INTER_CNF).unwrap();
        cnf_file.write_all(cnf_flat.as_bytes()).unwrap();

        // transform the CNF to dDNNF and load it
        compile_cnf(INTER_CNF, INTER_NNF);
        let last_lit_number = *re_indices.keys().max().unwrap();
        if !Path::new(INTER_NNF).exists() {
            println!("There is no INTER.nnf file...");
            let contents = fs::read_to_string(INTER_CNF)
            .expect("Should have been able to read the file");
    
            println!("With text:\n{contents}");
            return false;
        }

        if DEBUG {
            let mut ddnnf_remove = Ddnnf::default();
            ddnnf_remove.inter_graph = self.clone(); ddnnf_remove.rebuild();
            write_as_mermaid_md(&ddnnf_remove, &[], "removed_sub.md", Some((adjusted_replace, 1_000))).unwrap();
        }

        // Remove all edges of the old graph. We keep the nodes.
        // Most are useless but some are used in other parts of the graph. Hence, we keep them
        // TODO Knoten die wiederverwendet werden benötigen noch die Kanten zu ihren Kindern!!!
        /*let mut dfs = DfsPostOrder::new(&self.graph, adjusted_replace);
        while let Some(nx) = dfs.next(&self.graph) {
            let mut children = self.graph.neighbors_directed(nx, Outgoing).detach();
            while let Some(edge_to_child) = children.next_edge(&self.graph) {
                self.graph.remove_edge(edge_to_child);
            }
        }*/

        let mut parents = self.graph.neighbors_directed(adjusted_replace, Outgoing).detach();
        while let Some(parent_edge) = parents.next_edge(&self.graph) {
            self.graph.remove_edge(parent_edge);
        }

        if DEBUG {
            let mut ddnnf_after_remove = Ddnnf::default();
            ddnnf_after_remove.inter_graph = self.clone(); ddnnf_after_remove.rebuild();
            write_as_mermaid_md(&ddnnf_after_remove, &[], "after_rm.md", None).unwrap();
        }

        // If the new subgraph is unsatisfiable, we don't have to add anything
        if fs::read_to_string(INTER_NNF).unwrap() == "f 1 0\n" {
            println!("Unsatisfiable!");
            return false;
        }

        let mut sup = build_ddnnf(INTER_NNF, Some(last_lit_number));

        if DEBUG {
            sup.rebuild();
            write_as_mermaid_md(&sup, &[], "sub_before_re.md", None).unwrap();
        }

        // add the new subgraph as additional graph (unconnected to self)
        let mut dfs = DfsPostOrder::new(&sup.inter_graph.graph, sup.inter_graph.root);
        while let Some(nx) = dfs.next(&sup.inter_graph.graph) {
            match sup.inter_graph.graph[nx] {
                TId::Literal { feature } => {
                    let re_lit = *re_indices.get(&(feature.unsigned_abs())).unwrap() as i32;
                    sup.inter_graph.graph[nx] = TId::Literal { feature: re_lit * feature.signum() };
                },
                _ => (),
            }
        }

        if DEBUG {
            sup.rebuild();
            write_as_mermaid_md(&sup, &[], "sub.md", None).unwrap();
        }

        let mut dfs = DfsPostOrder::new(&sup.inter_graph.graph, sup.inter_graph.root);
        let mut cache = HashMap::new();
        while let Some(nx) = dfs.next(&sup.inter_graph.graph) {
            let new_nx = match sup.inter_graph.graph[nx] {
                // reindex the literal nodes and attach them to already existing nodes
                TId::Literal { feature } => {
                    match self.literals_nx.get(&feature) {
                        Some(nx) => *nx,
                        None => self.graph.add_node(TId::Literal { feature }),
                    }
                }
                // everything else can just be added
                _ => self.graph.add_node(sup.inter_graph.graph[nx]),
            };
            cache.insert(nx, new_nx);

            let mut children = sup.inter_graph.graph.neighbors_directed(nx, Outgoing).detach();
            while let Some((child_ex, child_nx)) = children.next(&sup.inter_graph.graph) {
                self.graph.add_edge(new_nx, *cache.get(&child_nx).unwrap(), sup.inter_graph.graph[child_ex].clone());
            }
        }

        if DEBUG {
            let mut ddnnf_after_add = Ddnnf::default();
            ddnnf_after_add.inter_graph = self.clone(); ddnnf_after_add.rebuild();
            write_as_mermaid_md(&ddnnf_after_add, &[], "after_adding.md", None).unwrap();
        }

        // replace the reference to the starting node with the new subgraph
        let new_sub_root = *cache.get(&sup.inter_graph.root).unwrap();
        self.graph.add_edge(adjusted_replace, new_sub_root, None);

        // add the literal_children of the newly created ddnnf
        extend_literal_diffs(&self.graph, &mut self.literal_children, new_sub_root);

        if DEBUG {
            let mut ddnnf_at_end = Ddnnf::default();
            ddnnf_at_end.inter_graph = self.clone(); ddnnf_at_end.rebuild();   
            write_as_mermaid_md(&ddnnf_at_end, &[], "at_end.md", None).unwrap();
        }

        // clean up temp files
        //if Path::new(INTER_CNF).exists() { fs::remove_file(INTER_CNF).unwrap(); }
        //if Path::new(INTER_NNF).exists() { fs::remove_file(INTER_NNF).unwrap(); }

        true
    }

    fn add_decision_nodes_fu(&mut self) {
        let mut dfs = Dfs::new(&self.graph, self.root);
        while let Some(nx) = dfs.next(&self.graph) {
            let mut children = self.graph.neighbors_directed(nx, Outgoing).detach();
            
            match self.graph[nx] {
                TId::Or => {
                    let literals_parent = self.literal_children.get(&nx).unwrap();
                    let mut literals_children = Vec::new();
                    let mut deciding_intersection: HashSet<u32> = literals_parent.iter().map(|l| l.unsigned_abs()).collect();
                    while let Some((edge, child)) = children.next(&self.graph) {
                        let literals_child = self.literal_children.get(&child).unwrap();
                        
                        let diff = literals_parent
                            .difference(literals_child)
                            .map(|literal| -literal)
                            .collect_vec();
                        
                        let unsigned_diff: HashSet<u32> = diff.iter().map(|l| l.unsigned_abs()).collect();
                        deciding_intersection.retain(|&x| unsigned_diff.contains(&x));
                        literals_children.push((diff, edge));
                    }
        
                    for (mut diff, edge) in literals_children {
                        diff.retain(|literal| deciding_intersection.contains(&literal.unsigned_abs()));
                        self.graph[edge] = match &self.graph[edge] {
                            Some(existing_diff) => {
                                diff.extend(existing_diff.iter());
                                Some(diff)
                            },
                            None => Some(diff),
                        }
                    }
                },
                TId::And => {
                    let mut literal_children = Vec::new();
                    let mut non_literal_ex = Vec::new();
                    while let Some((edge, child)) = children.next(&self.graph) {
                        match self.graph[child] {
                            TId::Literal { feature } => literal_children.push(feature),
                            _ => non_literal_ex.push(edge)
                        }
                    }

                    for edge in non_literal_ex {
                        let new_decisions = match &self.graph[edge] {
                            Some(dec) => {
                                let mut ndec = dec.clone();
                                ndec.extend(literal_children.iter());
                                Some(ndec)
                            },
                            None => Some(literal_children.clone()),
                        };

                        self.graph[edge] = new_decisions;
                    }
                },
                _ => (),
            }
            /*
             * Debug printing decision nodes
             */
            /*if false {
                println!("Literals parent: {:?}", literals_parent);
                let mut children = self.graph.neighbors_directed(nx, Outgoing).detach();
                while let Some((edge, child)) = children.next(&self.graph) {
                    println!("Literals child: {:?}", self.literal_children.get(&child).unwrap());
                    println!("Edges: {:?}", self.graph[edge]);
                }
                println!("-----------------------------");
            }*/
        }
    }

    /// Adds the necessary reference / removes it to extend a dDNNF by a unit clause
    fn add_unit_clause(&mut self, feature: i32) {
        if feature.unsigned_abs() <= self.number_of_variables {
            // it's an old feature
            match self.literals_nx.get(&-feature) {
                // Add a unit clause by removing the existence of its contradiction
                // Removing it is the same as replacing it by a FALSE node.
                // We resolve the FALSE node immidiatly.
                Some(&node) => {
                    let mut need_to_be_removed = vec![node];
                    
                    while !need_to_be_removed.is_empty() {
                        let mut remove_in_next_step = Vec::new();
                        for node in need_to_be_removed {
                            let mut children = self.graph.neighbors_directed(node, Incoming).detach();
                            while let Some(parent) = children.next_node(&self.graph) {
                                if self.graph[parent] == TId::And {
                                    remove_in_next_step.push(parent);
                                }
                            }
                            self.graph.remove_node(node);
                        }
                        need_to_be_removed = remove_in_next_step;
                    }
                },
                // If its contradiction does not exist, we don't have to do anything
                None => (),
            }
        } else {
            // one / multiple new features depending on feature number
            // Example: If the current #variables = 42 and the new feature number is 50,
            // we have to add all the features from 43 to 49 (including) as optional features (or triangle at root)
            while self.number_of_variables != feature.unsigned_abs() {
                self.number_of_variables += 1;
                
                let new_or = self.graph.add_node(TId::Or);
                self.graph.add_edge(self.root, new_or, None);
                
                let new_positive_lit = self.graph.add_node(TId::Literal { feature: self.number_of_variables as i32 });
                let new_negative_lit = self.graph.add_node(TId::Literal { feature: -(self.number_of_variables as i32) });
                self.graph.add_edge(new_or, new_positive_lit, None);
                self.graph.add_edge(new_or, new_negative_lit, None);
            }
            
            // Add the actual new feature
            let new_feature = self.graph.add_node(TId::Literal { feature });
            self.graph.add_edge(self.root, new_feature, None);
            self.number_of_variables = feature.unsigned_abs();
        }
    }

    fn annotate_decisions(&self, clause: &[i32]) -> HashMap<NodeIndex, Vec<i32>> {
        let mut nx_remaining_clause = HashMap::new();
        // the root node always still contains the whole clause
        nx_remaining_clause.insert(self.root, Vec::from(clause));

        let mut dfs = Dfs::new(&self.graph, self.root);
        while let Some(nx) = dfs.next(&self.graph) {
            let mut children = self.graph.neighbors_directed(nx, Outgoing).detach();
            while let Some((edge, child)) = children.next(&self.graph) {
                let remaining_clause = nx_remaining_clause.get(&nx).unwrap().clone();
                if remaining_clause.is_empty() { nx_remaining_clause.insert(child, vec![]); continue; }
                if nx_remaining_clause.contains_key(&child) { continue; }
                
                match &self.graph[edge] {
                    Some(decisions) => {
                        if decisions.iter().any(|literal| remaining_clause.contains(&literal)) {
                            nx_remaining_clause.insert(child, vec![])
                        } else {
                            let removed_impossible_assignments = remaining_clause
                                .into_iter()
                                .filter(|&literal| !decisions.contains(&-literal))
                                .collect_vec();
                            nx_remaining_clause.insert(
                                child,
                                removed_impossible_assignments
                            )
                        }
                    },
                    None => nx_remaining_clause.insert(child, remaining_clause),
                };
            }
        }
        nx_remaining_clause
    }

    fn get_decisions_target_nx(&self, target: NodeIndex) -> HashSet<i32> {
        let mut nx_decisions = HashMap::new();
        // the root node can't have any decision yet
        nx_decisions.insert(self.root, Vec::new());

        let mut dfs = Dfs::new(&self.graph, self.root);
        while let Some(nx) = dfs.next(&self.graph) {
            let mut children = self.graph.neighbors_directed(nx, Outgoing).detach();
            while let Some((edge, child)) = children.next(&self.graph) {
                let mut current_decisions: Vec<i32> = nx_decisions.get(&nx).unwrap().clone();
                //println!("current before: {:?}", current_decisions);
                let edge_decisions = self.graph[edge].clone().unwrap_or(Vec::new());
                current_decisions.extend(edge_decisions.iter());
                //println!("current after: {:?}", current_decisions);
                nx_decisions.insert(child, current_decisions);
            }
        }

        match nx_decisions.get(&target) {
            Some(decision) => decision.clone().into_iter().collect(),
            None => HashSet::new()
        }
    }

    /// Adds the necessary reference / removes them to extend a dDNNF by a unit clause
    fn starting_points_from_reduce_clause(&mut self, clause: &[i32]) -> HashSet<NodeIndex> {
        let nx_remaining_clause = self.annotate_decisions(clause);

        //print!("clause: {:?} with ", clause);
        //let mut replace_counter = 0; let mut ignore_counter = 0;
        let mut starting_points = HashSet::new();
        for literal_repr in clause {
            for literal in [literal_repr, &-literal_repr] {
                match self.literals_nx.get(literal) {
                    Some(nx) => {
                        let mut parents = self.graph.neighbors_directed(*nx, Incoming).detach();
                        while let Some((edge, parent)) = parents.next(&self.graph) {
                            let mut parent_rc = nx_remaining_clause.get(&parent).unwrap().clone();
                            parent_rc.retain(|&l| {
                                match &self.graph[edge] {
                                    Some(d) => !d.contains(&-l),
                                    None => true
                                }
                            });
                            if parent_rc.is_empty() {
                                //ignore_counter += 1
                            } else {
                                //replace_counter += 1;
                                starting_points.insert(parent);
                            }
                            //print!(" {:?}", parent_rc);   
                        }
                    },
                    None => ()//ignore_counter += 1 //print!(" []"),
                }
            }
        }
        //println!("Can be ignored: {ignore_counter}. Have to be replaced: {replace_counter}");
        //println!("{:?}", nx_remaining_clause);
        starting_points
    }

    pub fn get_partial_graph_til_depth(&self, start: NodeIndex, depth: i32) -> IntermediateGraph {
        let mut sub_ig = self.clone();
        let mut distance_mapping = HashMap::new();

        let mut dfs = Dfs::new(&self.graph, start);
        while let Some(nx) = dfs.next(&self.graph) {
            let mut highest_parent_depth = 0;

            let mut parents = self.graph.neighbors_directed(nx, Incoming).detach();
            while let Some(node) = parents.next_node(&self.graph) {
                highest_parent_depth = max(highest_parent_depth, *distance_mapping.get(&node).unwrap_or(&0));
            }

            distance_mapping.insert(nx, highest_parent_depth + 1);
        }

        // remove all nodes from the clone that are not within the depth limit
        let mut dfs = Dfs::new(&self.graph, self.root);
        while let Some(nx) = dfs.next(&self.graph) {
            let remove_node = match distance_mapping.get(&nx) {
                Some(&nx_depth) => nx_depth > depth, // depth does not fit
                None => true, // no entry -> node is neither parent nor any child of start
            };

            if remove_node {
                let mut parents = sub_ig.graph.neighbors_directed(nx, Incoming).detach();
                while let Some(edge_inc) = parents.next_edge(&sub_ig.graph) {
                    sub_ig.graph.remove_edge(edge_inc);
                }

                let mut children = sub_ig.graph.neighbors_directed(nx, Outgoing).detach();
                while let Some(edge_out) = children.next_edge(&sub_ig.graph) {
                    sub_ig.graph.remove_edge(edge_out);
                }

                sub_ig.graph.remove_node(nx);
            }
        }
        
        sub_ig.root = start;
        sub_ig
    }
}

#[cfg(test)]
mod test {
    use std::{collections::HashSet, fs::{File, self}, io::Write};

    use petgraph::Direction::Incoming;
    use rand::rngs::StdRng;
    use serial_test::serial;

    use crate::{parser::{build_ddnnf, d4v2_wrapper::compile_cnf, persisting::write_as_mermaid_md, from_cnf::{remove_clause_cnf, get_all_clauses_cnf, add_clause_cnf}}, Ddnnf};

    #[test]
    #[serial]
    fn closest_unsplittable_and() {
        let bridge_comparison = |mut ddnnf: Ddnnf, input: Vec<Vec<i32>>, output: Vec<Option<Vec<i32>>>| {
            for (index, inp) in input.iter().enumerate() {
                match ddnnf.inter_graph.closest_unsplitable_bridge(inp) {
                    Some((_replace_and_node, literals)) => {
                        let mut literals_as_vec = HashSet::<_>::from_iter(
                            literals.iter().copied())
                            .into_iter()
                            .collect::<Vec<i32>>();
    
                        literals_as_vec.sort();
                        assert_eq!(output[index].clone().unwrap(), literals_as_vec);
                    },
                    None => {
                        assert!(output[index].is_none());
                    }
                }
            }
        };
        
        let ddnnf_vp9 = build_ddnnf("tests/data/VP9.cnf", Some(42));
        let input_vp9 = vec![
            vec![], vec![4], vec![5], vec![4, 5],
            vec![42], vec![-5], vec![-8]
        ];
        let output_vp9 = vec![
            None, Some(vec![-5, -4, -3, 3, 4, 5]), Some(vec![-5, -4, -3, 3, 4, 5]), Some(vec![-5, -4, -3, 3, 4, 5]),
            Some(vec![-42, -41, 41, 42]), Some(vec![-5, -4, -3, 3, 4, 5]), Some(vec![-9, -8, -7, 7, 8, 9])
        ];
        bridge_comparison(ddnnf_vp9, input_vp9, output_vp9);
    }

    #[test]
    #[serial]
    fn from_ddnnf_to_cnf() {
        const CNF_REDONE: &str = "tests/data/.redone.cnf"; const DDNNF_REDONE: &str = "tests/data/.redone.nnf";

        let ddnnf_file_paths = vec![
            ("tests/data/small_ex_c2d.nnf", 4),
            ("tests/data/small_ex_d4.nnf", 4),
            ("tests/data/VP9_d4.nnf", 42)
        ];

        for (path, features) in ddnnf_file_paths {
            let mut ddnnf = build_ddnnf(path, Some(features));
            let mut complete_configs_direct = ddnnf.enumerate(&mut vec![], 1_000_000).unwrap();
            
            let (cnf, _, reverse_indexing) = ddnnf.inter_graph.transform_to_cnf_tseitin(ddnnf.inter_graph.root, &[]);
            let cnf_flat = cnf.join("");
            let mut cnf_file = File::create(CNF_REDONE).unwrap();
            cnf_file.write_all(cnf_flat.as_bytes()).unwrap();

            compile_cnf(CNF_REDONE, DDNNF_REDONE);
            let mut ddnnf_redone = build_ddnnf(DDNNF_REDONE, Some(features));
            let mut complete_configs_recompilation = ddnnf_redone.enumerate(&mut vec![], 1_000_000).unwrap();

            assert_eq!(complete_configs_direct.len(), complete_configs_recompilation.len());

            // adjust the indices of the recompiled configurations
            for config in complete_configs_recompilation.iter_mut() {
                for i in 0..config.len() {
                    match reverse_indexing.get(&(config[i].unsigned_abs() as i32)) {
                        Some(&val) => { 
                            config[i] = if config[i].is_positive() { val } else { -val }
                        },
                        None => (), // We don't have to remap tseitin variables that correspond to And an Or nodes
                    }
                }
                config.sort_by_key(|v| v.abs());
                config.drain((ddnnf.number_of_variables as usize)..); // remove tseitin variables
            }

            complete_configs_direct.sort();
            complete_configs_recompilation.sort();
            assert_eq!(complete_configs_direct, complete_configs_recompilation);

            fs::remove_file(CNF_REDONE).unwrap();
            fs::remove_file(DDNNF_REDONE).unwrap();
        }
    }

    #[test]
    #[serial]
    fn from_ddnnf_to_cnf_distributive() {
        const CNF_REDONE: &str = "tests/data/.redone_d.cnf"; const DDNNF_REDONE: &str = "tests/data/.redone_d.nnf";

        let ddnnf_file_paths = vec![
            ("tests/data/small_ex_c2d.nnf", 4),
            ("tests/data/small_ex_d4.nnf", 4),
            ("tests/data/VP9_d4.nnf", 42),
        ];

        for (path, features) in ddnnf_file_paths {
            let mut ddnnf = build_ddnnf(path, Some(features));
            let (cnf, _) = ddnnf.inter_graph.transform_to_cnf_distributive(ddnnf.inter_graph.root, &[]);
            let cnf_flat = cnf.join("");
            let mut cnf_file = File::create(CNF_REDONE).unwrap();
            cnf_file.write_all(cnf_flat.as_bytes()).unwrap();

            compile_cnf(CNF_REDONE, DDNNF_REDONE);
            let ddnnf_redone = build_ddnnf(DDNNF_REDONE, Some(features));

            assert_eq!(ddnnf.rc(), ddnnf_redone.rc());

            fs::remove_file(CNF_REDONE).unwrap();
            fs::remove_file(DDNNF_REDONE).unwrap();
        }
    }

    fn check_for_cardinality_correctness(path: &str, break_point: usize) {
        let copy_path_string = format!(".{}_copy.cnf", path.split("/").collect::<Vec<&str>>().last().unwrap());
        let copy_path = &copy_path_string;
        fs::copy(path, copy_path).unwrap();

        let mut clauses = get_all_clauses_cnf(copy_path);
        use rand::SeedableRng; let mut rng: StdRng = SeedableRng::seed_from_u64(42);
        use rand::prelude::SliceRandom; clauses.shuffle(&mut rng);
        for (index, clause) in clauses.into_iter().enumerate() {
            if index >= break_point { break; }
            println!("-------------------------------------------------------------");
            println!("Current clause: {:?}", clause);
            
            remove_clause_cnf(copy_path, &clause, None);
            let mut ddnnf_wo = build_ddnnf(copy_path, None);
            //write_as_mermaid_md(&ddnnf_wo, &[], "before.md", None).unwrap();
            ddnnf_wo.inter_graph.add_clause_alt(clause.clone());
            ddnnf_wo.rebuild();
            //write_as_mermaid_md(&ddnnf_wo, &[], "after.md", None).unwrap();

            add_clause_cnf(copy_path, &clause);
            let mut ddnnf_w = build_ddnnf(copy_path, None);
            //write_as_mermaid_md(&ddnnf_w, &[], "with.md", None).unwrap();

            assert_eq!(ddnnf_wo.rc(), ddnnf_w.rc());
            for feature in 0_i32..ddnnf_w.number_of_variables as i32 {
                assert_eq!(
                    ddnnf_wo.execute_query(&[feature]),
                    ddnnf_w.execute_query(&[feature])
                );
            }
        }
        fs::remove_file(copy_path).unwrap();

    }

    #[test]
    #[serial]
    fn transform_to_cnf_from_starting_cnf_clauses_small_models() {
        let ddnnf_file_paths = vec![
            "tests/data/VP9.cnf",
            "tests/data/X264.cnf",
            "tests/data/HiPAcc.cnf"
        ];

        for path in ddnnf_file_paths {
            check_for_cardinality_correctness(path, usize::MAX);    
        }
    }

    #[test]
    #[serial]
    fn transform_to_cnf_from_starting_cnf_clauses_medium_models() {
        let ddnnf_file_paths = vec![
            "tests/data/kc_axTLS.cnf",
            "tests/data/toybox.cnf"
        ];

        for path in ddnnf_file_paths {
            check_for_cardinality_correctness(path, 100);
        }
    }

    #[test]
    #[serial]
    fn transform_to_cnf_from_starting_cnf_clauses_big_models() {
        let ddnnf_file_paths = vec![
            "tests/data/auto1.cnf",
            "tests/data/auto2.cnf"
        ];

        for path in ddnnf_file_paths {
            check_for_cardinality_correctness(path, 10);
        }
    }

    #[test]
    #[serial]
    fn incremental_adding_clause() {
        let ddnnf_file_paths = vec![
            ("tests/data/VP9.cnf", "tests/data/VP9_wo_-4-5.cnf", 42, vec![-4, -5])
        ];

        for (path_w_clause, path_wo_clause, features, clause) in ddnnf_file_paths {
            let mut ddnnf_w = build_ddnnf(path_w_clause, Some(features));

            let mut expected_results = Vec::new();
            for f in 1..=features {
                expected_results.push(ddnnf_w.execute_query(&[f as i32]));
            }
            
            let mut ddnnf_wo = build_ddnnf(path_wo_clause, Some(features));
            write_as_mermaid_md(&mut ddnnf_wo, &[], "before.md", None).unwrap();
            ddnnf_wo.inter_graph.add_clause(&clause);
            ddnnf_wo.rebuild();
            write_as_mermaid_md(&mut ddnnf_wo, &[], "after.md", None).unwrap();

            let mut results_after_addition = Vec::new();
            for f in 1..=features {
                results_after_addition.push(ddnnf_wo.execute_query(&[f as i32]));
            }

            assert_eq!(expected_results.len(), results_after_addition.len());
            assert_eq!(expected_results, results_after_addition);
        }
    }

    #[test]
    #[serial]
    fn adding_unit_clause() {
        // small example, missing the "1 0" unit clause
        const CNF_PATH: &str = "tests/data/.small_missing_unit.cnf";
        let cnf_flat = "p cnf 4 2\n2 3 0\n-2 -3 0\n";
        let mut cnf_file = File::create(CNF_PATH).unwrap();
        cnf_file.write_all(cnf_flat.as_bytes()).unwrap();

        let mut ddnnf_sb = build_ddnnf("tests/data/small_ex_c2d.nnf", Some(4));
        let mut ddnnf_missing_clause1 = build_ddnnf(CNF_PATH, None);
        let mut ddnnf_missing_clause2 = ddnnf_missing_clause1.clone();

        ddnnf_missing_clause1.inter_graph.add_unit_clause(1); // add the unit clause directly
        ddnnf_missing_clause2.inter_graph.add_clause(&vec![1]); // indirectly via adding a clause
        ddnnf_missing_clause1.rebuild();
        ddnnf_missing_clause2.rebuild();

        // We have to sort the results because the inner structure does not have to be identical
        let mut sb_enumeration = ddnnf_sb.enumerate(&mut vec![], 1_000).unwrap();
        sb_enumeration.sort();

        let mut ms1_enumeration = ddnnf_missing_clause1.enumerate(&mut vec![], 1_000).unwrap();
        ms1_enumeration.sort();

        let mut ms2_enumeration = ddnnf_missing_clause2.enumerate(&mut vec![], 1_000).unwrap();
        ms2_enumeration.sort();

        // check whether the dDNNFs contain the same configurations
        assert_eq!(sb_enumeration, ms1_enumeration);
        assert_eq!(sb_enumeration, ms2_enumeration);

        fs::remove_file(CNF_PATH).unwrap();
    }

    #[test]
    #[serial]
    fn analyze_closest_and_node() {
        let mut ddnnf = build_ddnnf("copy.cnf", Some(1360));
        let clause = vec![-904, 1111];

        let replace_multiple = match ddnnf.inter_graph.closest_unsplitable_ands2(&clause) {
            Some(and_node) => and_node,
            None => panic!()
        };

        let (replace_naive, _) = match ddnnf.inter_graph.closest_unsplitable_and(&clause) {
            Some(and_node) => and_node,
            None => panic!()
        };

        let adjusted_starting_point = ddnnf.inter_graph.divide_and(replace_multiple[0].0, &clause);
        let unadjusted_starting_point = replace_multiple[0].0;
        let phycore_root = ddnnf.inter_graph.root;
        
        println!("creating mermaid graphs...");
        
        write_as_mermaid_md(&mut ddnnf, &[], "replace_naive.md", Some((replace_naive, 5))).unwrap();
        write_as_mermaid_md(&mut ddnnf, &[], "closest_phycore.md", Some((unadjusted_starting_point, 7))).unwrap(); 
        write_as_mermaid_md(&mut ddnnf, &[], "adjusted_closest_phycore.md", Some((adjusted_starting_point, 7))).unwrap();   
        write_as_mermaid_md(&mut ddnnf, &[], "phycore_root.md", Some((phycore_root, 7))).unwrap(); 

        let nx_43 = *ddnnf.inter_graph.literals_nx.get(&-43).unwrap();
        let mut parent_43 = None;

        let mut parents = ddnnf.inter_graph.graph.neighbors_directed(nx_43, Incoming).detach();
        while let Some(parent) = parents.next_node(&ddnnf.inter_graph.graph) {
            parent_43 = Some(parent);
        }

        write_as_mermaid_md(&mut ddnnf, &[], "phycore_l43p.md", Some((parent_43.unwrap(), 3))).unwrap();
    }
}