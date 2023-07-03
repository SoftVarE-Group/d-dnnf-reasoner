use std::{collections::{HashMap, HashSet}, fs::{File, self}, io::Write, time::Instant};

use itertools::{Itertools};
use petgraph::{
    stable_graph::{StableGraph, NodeIndex},
    visit::{DfsPostOrder},
    algo::is_cyclic_directed,
    Direction::{Incoming, Outgoing}
};

use crate::{c2d_lexer::TId, Node, NodeType, parser::{get_literal_diffs, util::format_vec, build_ddnnf}};

use super::{calc_and_count, calc_or_count, d4v2_wrapper::compile_cnf};

/// The IntermediateGraph enables us to modify the dDNNF. The structure of a vector of nodes does not allow
/// for that because deleting or removing nodes would mess up the indices. 
#[derive(Clone, Debug, Default)]
pub struct IntermediateGraph {
    graph: StableGraph::<TId, ()>,
    pub root: NodeIndex,
    number_of_variables: u32,
    literals_nx: HashMap<i32, NodeIndex>,
    literal_children: HashMap<NodeIndex, HashSet<i32>>
}

impl IntermediateGraph {
    /// Creates a new IntermediateGraph 
    pub fn new(graph: StableGraph::<TId, ()>, root: NodeIndex, number_of_variables: u32, literals_nx: HashMap<i32, NodeIndex>) -> IntermediateGraph {
        debug_assert!(!is_cyclic_directed(&graph));
        let mut inter_graph = IntermediateGraph {
            graph, root, number_of_variables, literals_nx,
            literal_children: HashMap::new()
        };
        inter_graph.literal_children = get_literal_diffs(&inter_graph.graph, inter_graph.root);
        inter_graph
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
            match self.literals_nx.get(literal) {
                // A literal node can contain multiple parents. We need to make sure that each direct
                // parent gets its own set to ensure including all connections towards the literal within
                // out target AND node.
                Some(lit) => {
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
                        Some((old_node, old_diffs)) =>
                            if old_diffs.len() < diffs.len() {
                                Some((old_node, old_diffs))
                            } else {
                                Some((node, diffs))
                            }
                        ,
                        None => Some((node, diffs)),
                    }
                },
                _ => ()
            }
        }
        cached_and
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
            if clause.iter().any(|f| lits.contains(f)) {
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
        for child in relevant_children {
            if let Some(edge) = self.graph.find_edge(initial_start, child) {
                self.graph.remove_edge(edge);
            }
            self.graph.add_edge(new_and, child, ());
        }
        self.graph.add_edge(initial_start, new_and, ());

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
        let mut counter = self.literal_children
            .get(&starting_point)
            .unwrap()
            .into_iter()
            .map(|v| v.unsigned_abs())
            .collect::<HashSet<u32>>().len() as i32 + 1;
        let mut lit_counter = 1;
        let mut clause_var: Vec<i32> = std::iter::repeat(0).take(nodes.len()).collect::<Vec<_>>();

        for (index, node) in nodes.iter().enumerate() {
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
        }
        // add root as unit clause
        cnf.push(format!("{} 0\n", clause_var[nodes.len() - 1]));

        // add the new clause to the CNF
        if !clause.is_empty() {
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
    pub fn transform_to_cnf_distributive(&mut self, starting_point: NodeIndex, clause: &[i32]) -> Vec<String> {
        let (nodes, _, _) = self.rebuild(Some(starting_point));
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
        cnf
    }

    pub fn add_clause(&mut self, clause: &[i32]) -> bool {
        if clause.len() == 1 { self.add_unit_clause(clause[0]); return true; }
        let mut _start = Instant::now();
        const INTER_CNF: &str = "intermediate.cnf"; const INTER_NNF: &str = "intermediate.nnf";
        let (replace, _) = match self.closest_unsplitable_and(&clause) {
            Some(and_node) => and_node,
            None => { return false; }
        };
        //if replace == self.root { return false; }

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
        let sup_ddnnf = build_ddnnf(INTER_NNF, Some(last_lit_number));
        //println!("elapsed time: ddnnf subgraph: {}", _start.elapsed().as_secs_f64());

        // Remove all edges of the old graph. We keep the nodes.
        // Most are useless but some are used in other parts of the graph. Hence, we keep them
        let mut dfs = DfsPostOrder::new(&self.graph, adjusted_replace);
        while let Some(nx) = dfs.next(&self.graph) {
            let mut children = self.graph.neighbors_directed(nx, Outgoing).detach();
            while let Some(edge_to_child) = children.next_edge(&self.graph) {
                self.graph.remove_edge(edge_to_child);
            }
        }

        // add the new subgraph as unconnected additional graph
        let sub = sup_ddnnf.inter_graph;
        let mut dfs = DfsPostOrder::new(&sub.graph, sub.root);
        let mut cache = HashMap::new();

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
                        let offset_lit = if feature.is_positive() { feature + 1_000_000 } else { feature - 1_000_000 };
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
                self.graph.add_edge(new_nx, *cache.get(&child).unwrap(), ());
            }
        }

        // remove the reference to the starting node with the new subgraph
        let new_sub_root = *cache.get(&sub.root).unwrap();
        let mut parents = self.graph.neighbors_directed(adjusted_replace, Incoming).detach();
        while let Some((parent_edge, parent_node)) = parents.next(&self.graph) {
            self.graph.remove_edge(parent_edge);
            self.graph.add_edge(parent_node, new_sub_root, ());
        }

        // clean up temp files
        fs::remove_file(INTER_CNF).unwrap();
        fs::remove_file(INTER_NNF).unwrap();
        true
    }

    /// Adds the necessary reference / removes them to extend a dDNNF by a unit clause
    fn add_unit_clause(&mut self, feature: i32) {
        if feature.unsigned_abs() <= self.number_of_variables {
            // it's an old feature
            match self.literals_nx.get(&-feature) {
                // Add a unit clause by removing the existence of its contradiction
                Some(&node) => {
                    self.graph.remove_node(node);
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
                self.graph.add_edge(self.root, new_or, ());
                
                let new_positive_lit = self.graph.add_node(TId::Literal { feature: self.number_of_variables as i32 });
                let new_negative_lit = self.graph.add_node(TId::Literal { feature: -(self.number_of_variables as i32) });
                self.graph.add_edge(new_or, new_positive_lit, ());
                self.graph.add_edge(new_or, new_negative_lit, ());
            }
            
            // Add the actual new feature
            let new_feature = self.graph.add_node(TId::Literal { feature });
            self.graph.add_edge(self.root, new_feature, ());
            self.number_of_variables = feature.unsigned_abs();
        }
    }
}

#[cfg(test)]
mod test {
    use std::{collections::HashSet, fs::{File, self}, io::Write};

    use serial_test::serial;

    use crate::parser::{build_ddnnf, d4v2_wrapper::compile_cnf};

    #[test]
    #[serial]
    fn closest_unsplittable_and() {
        let mut ddnnf = build_ddnnf("tests/data/VP9_d4.nnf", Some(42));

        let input = vec![
            vec![], vec![4], vec![5], vec![4, 5],
            vec![42], vec![-5], vec![-8]
        ];
        let output = vec![
            None, Some(vec![-5, 4]), Some(vec![-4, 5]), Some(vec![-5, -4, -3, 4, 5]),
            Some(vec![-41, 42]), Some(vec![-5, -4, -3, 3, 4, 5]), Some(vec![-9, -8, -7, 7, 8, 9])
        ];

        for (index, inp) in input.iter().enumerate() {
            match ddnnf.inter_graph.closest_unsplitable_and(inp) {
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
            let cnf = ddnnf.inter_graph.transform_to_cnf_distributive(ddnnf.inter_graph.root, &[]);
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

    #[test]
    #[serial]
    fn incremental_adding_clause() {
        let ddnnf_file_paths = vec![
            ("tests/data/VP9_w.dimacs", "tests/data/VP9_wo_-4-5.dimacs", 42, vec![-4, -5])
        ];

        for (path_w_clause, path_wo_clause, features, clause) in ddnnf_file_paths {
            let mut ddnnf_w = build_ddnnf(path_w_clause, Some(features));

            let mut expected_results = Vec::new();
            for f in 1..=features {
                expected_results.push(ddnnf_w.execute_query(&[f as i32]));
            }
            
            let mut ddnnf_wo = build_ddnnf(path_wo_clause, Some(features));
            ddnnf_wo.inter_graph.add_clause(&clause);
            ddnnf_wo.rebuild();

            let mut results_after_addition = Vec::new();
            for f in 1..=features {
                results_after_addition.push(ddnnf_wo.execute_query(&[f as i32]));
            }

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
}