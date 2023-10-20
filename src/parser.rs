pub mod c2d_lexer;
use c2d_lexer::{lex_line_c2d, C2DToken, TId};

pub mod d4_lexer;
use d4_lexer::{lex_line_d4, D4Token};

pub mod from_cnf;
use from_cnf::{check_for_cnf_header, CNFToken};

pub(crate) mod intermediate_representation;

pub(crate) mod d4v2_wrapper;
use crate::parser::d4v2_wrapper::compile_cnf;
use crate::parser::intermediate_representation::IntermediateGraph;
use crate::parser::util::{calc_and_count, calc_or_count};

pub mod persisting;
pub mod util;

use core::panic;
use std::{
    cell::RefCell,
    cmp::max,
    collections::{HashMap, HashSet},
    ffi::OsStr,
    fs::{self},
    io::{BufRead, BufReader},
    path::Path,
    process,
    rc::Rc,
};

use crate::ddnnf::Ddnnf;

use petgraph::{
    graph::{EdgeIndex, NodeIndex},
    stable_graph::StableGraph,
    visit::DfsPostOrder,
    Direction::{Incoming, Outgoing},
};

use self::util::open_file_savely;

type DdnnfGraph = StableGraph<TId, ()>;

/// Parses a ddnnf, referenced by the file path. The file gets parsed and we create
/// the corresponding data structure.
///
/// # Examples
///
/// ```
/// extern crate ddnnf_lib;
/// use ddnnf_lib::parser;
/// use ddnnf_lib::Ddnnf;
///
/// let file_path = "./tests/data/small_ex_c2d.nnf";
///
/// let ddnnfx: Ddnnf = parser::build_ddnnf(file_path, None);
/// ```
///
/// # Panics
///
/// The function panics for an invalid file path.
#[inline]
pub fn build_ddnnf(mut path: &str, mut total_features: Option<u32>) -> Ddnnf {
    let mut cnf_path = None;
    if let Some(extension) = Path::new(path).extension().and_then(OsStr::to_str) {
        if extension == "dimacs" || extension == "cnf" {
            let file = open_file_savely(path);
            let lines = BufReader::new(file).lines();

            for line in lines {
                let line = line.expect("Unable to read line");
                match check_for_cnf_header(line.as_str()).unwrap().1 {
                    CNFToken::Header {
                        features,
                        clauses: _,
                    } => {
                        let ddnnf_file = ".intermediate.nnf";
                        compile_cnf(path, ddnnf_file);
                        cnf_path = Some(path);
                        path = ddnnf_file;
                        total_features = Some(features as u32);

                        break;
                    }
                    CNFToken::Comment | CNFToken::Clause => (),
                }
            }
        }
    }

    let file = open_file_savely(path);
    let lines = BufReader::new(file)
        .lines()
        .map(|line| line.expect("Unable to read line"))
        .collect::<Vec<String>>();

    if path == ".intermediate.nnf" {
        fs::remove_file(path).unwrap();
    }

    distribute_building(lines, total_features, cnf_path)
}

/// Chooses, depending on the first read line, which building implmentation to choose.
/// Either the first line is a header and therefore the c2d format or total_features
/// is supplied and its the d4 format.
#[inline]
pub fn distribute_building(
    lines: Vec<String>,
    total_features: Option<u32>,
    cnf_path: Option<&str>,
) -> Ddnnf {
    use C2DToken::*;

    match lex_line_c2d(lines[0].trim()) {
        Ok((
            _,
            Header {
                nodes: _,
                edges: _,
                variables,
            },
        )) => build_c2d_ddnnf(lines, variables as u32),
        Ok(_) | Err(_) => {
            // tried to parse the c2d standard, but failes
            match total_features {
                Some(o) => {
                    // we try to parse the d4 standard
                    build_d4_ddnnf(lines, Some(o), cnf_path)
                }
                None => {
                    // unknown standard or combination -> we assume d4 and choose total_features
                    // Bold, Yellow, Foreground Color (see https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797)
                    println!("\x1b[1;38;5;226mWARNING: The first line of the file isn't a header and the option 'total_features' is not set. \
                        Hence, we can't determine the number of variables and as a result, we might not be able to construct a valid ddnnf. \
                        Nonetheless, we build a ddnnf with our limited information, but we discourage using ddnnife in this manner.\n\x1b[0m"
                    );
                    build_d4_ddnnf(lines, None, cnf_path)
                }
            }
        }
    }
}

/// Parses a ddnnf, referenced by the file path.
/// This function uses C2DTokens which specify a d-DNNF in c2d format.
/// The file gets parsed and we create the corresponding data structure.
#[inline]
fn build_c2d_ddnnf(lines: Vec<String>, variables: u32) -> Ddnnf {
    use C2DToken::*;
    let mut ddnnf_graph = DdnnfGraph::new();
    let mut node_indices = Vec::with_capacity(lines.len());
    let mut literals_nx: HashMap<i32, NodeIndex> = HashMap::new();

    // opens the file with a BufReader and
    // works off each line of the file data seperatly
    // skip the first line, because we already looked at the header
    for line in lines.into_iter().skip(1) {
        node_indices.push(match lex_line_c2d(line.as_ref()).unwrap().1 {
            And { children } => {
                let from = ddnnf_graph.add_node(TId::And);
                for child in children {
                    ddnnf_graph.add_edge(from, node_indices[child], ());
                }
                from
            }
            Or {
                decision: _,
                children,
            } => {
                let from = ddnnf_graph.add_node(TId::Or);
                for child in children {
                    ddnnf_graph.add_edge(from, node_indices[child], ());
                }
                from
            }
            Literal { feature } => {
                let literal_node = ddnnf_graph.add_node(TId::Literal { feature });
                literals_nx.insert(feature, literal_node);
                literal_node
            }
            True => ddnnf_graph.add_node(TId::True),
            False => ddnnf_graph.add_node(TId::False),
            _ => panic!("Tried to parse the header of the .nnf at the wrong time"),
        });
    }

    let root = node_indices[node_indices.len() - 1];
    let literal_diffs = get_literal_diffs(&ddnnf_graph, root);
    let intermediate_graph = IntermediateGraph::new(
        ddnnf_graph,
        root,
        variables,
        literals_nx,
        literal_diffs,
        None,
    );
    Ddnnf::new(intermediate_graph, variables)
}

/// Parses a ddnnf, referenced by the file path.
/// This function uses D4Tokens which specify a d-DNNF in d4 format.
/// The file gets parsed and we create the corresponding data structure.
#[inline]
fn build_d4_ddnnf(
    lines: Vec<String>,
    total_features_opt: Option<u32>,
    cnf_path: Option<&str>,
) -> Ddnnf {
    let mut ddnnf_graph = DdnnfGraph::new();

    let mut total_features = total_features_opt.unwrap_or(0);
    let literal_occurences: Rc<RefCell<Vec<bool>>> =
        Rc::new(RefCell::new(vec![
            false;
            max(100_000, total_features as usize)
        ]));

    let mut indices: Vec<NodeIndex> = Vec::new();

    // With the help of the literals node state, we can add the required nodes
    // for the balancing of the or nodes to archieve smoothness
    let literals_nx: Rc<RefCell<HashMap<i32, NodeIndex>>> = Rc::new(RefCell::new(HashMap::new()));

    let get_literal_indices =
        |ddnnf_graph: &mut DdnnfGraph, literals: Vec<i32>| -> Vec<NodeIndex> {
            let mut lit_nx = literals_nx.borrow_mut();
            let mut literal_nodes = Vec::new();

            for literal in literals {
                literal_nodes.push(match lit_nx.get(&literal) {
                    Some(x) => *x,
                    None => {
                        let nx = ddnnf_graph.add_node(TId::Literal { feature: literal });
                        lit_nx.insert(literal, nx);
                        nx
                    }
                })
            }
            literal_nodes
        };

    // while parsing:
    // remove the weighted edges and substitute it with the corresponding
    // structure that uses AND-Nodes and Literal-Nodes. Example:
    //
    //                   n1                    n1
    //                 /   \                   |
    //              Ln|    |Lm     into       AND
    //                \   /                 /  |  \
    //                 n2                 Ln  n2  Lm
    //
    let resolve_weighted_edge = |ddnnf_graph: &mut DdnnfGraph,
                                 from: NodeIndex,
                                 to: NodeIndex,
                                 edge: EdgeIndex,
                                 weights: Vec<i32>| {
        let literal_nodes = get_literal_indices(ddnnf_graph, weights);
        if !literal_nodes.is_empty() {
            let and_node = ddnnf_graph.add_node(TId::And);
            ddnnf_graph.remove_edge(edge);

            ddnnf_graph.add_edge(from, and_node, ());
            for node in literal_nodes {
                ddnnf_graph.add_edge(and_node, node, ());
            }
            ddnnf_graph.add_edge(and_node, to, ());
        }
    };

    // opens the file with a BufReader and
    // works off each line of the file data seperatly
    for line in lines {
        let next: D4Token = lex_line_d4(line.as_ref()).unwrap().1;

        use D4Token::*;
        match next {
            Edge { from, to, features } => {
                for f in &features {
                    literal_occurences.borrow_mut()[f.unsigned_abs() as usize] = true;
                    total_features = max(total_features, f.unsigned_abs());
                }
                let from_n = indices[from as usize - 1];
                let to_n = indices[to as usize - 1];
                let edge = ddnnf_graph.add_edge(from_n, to_n, ());
                resolve_weighted_edge(&mut ddnnf_graph, from_n, to_n, edge, features);
            }
            And => indices.push(ddnnf_graph.add_node(TId::And)),
            Or => indices.push(ddnnf_graph.add_node(TId::Or)),
            True => indices.push(ddnnf_graph.add_node(TId::True)),
            False => indices.push(ddnnf_graph.add_node(TId::False)),
        }
    }

    let or_triangles: Rc<RefCell<Vec<Option<NodeIndex>>>> =
        Rc::new(RefCell::new(vec![None; (total_features + 1) as usize]));

    let add_literal_node = |ddnnf_graph: &mut DdnnfGraph, f_u32: u32, attach: NodeIndex| {
        let f = f_u32 as i32;
        let mut ort = or_triangles.borrow_mut();

        if ort[f_u32 as usize].is_some() {
            ddnnf_graph.add_edge(attach, ort[f_u32 as usize].unwrap(), ());
        } else {
            let or = ddnnf_graph.add_node(TId::Or);
            ort[f_u32 as usize] = Some(or);

            let pos_lit = get_literal_indices(ddnnf_graph, vec![f])[0];
            let neg_lit = get_literal_indices(ddnnf_graph, vec![-f])[0];

            ddnnf_graph.add_edge(attach, or, ());
            ddnnf_graph.add_edge(or, pos_lit, ());
            ddnnf_graph.add_edge(or, neg_lit, ());
        }
    };

    let balance_or_children =
        |ddnnf_graph: &mut DdnnfGraph,
         from: NodeIndex,
         children: Vec<(NodeIndex, HashSet<u32>)>| {
            for (child_nx, child_literals) in children {
                let and_node = ddnnf_graph.add_node(TId::And);

                // place the newly created and node between the or node and its child
                ddnnf_graph.remove_edge(ddnnf_graph.find_edge(from, child_nx).unwrap());
                ddnnf_graph.add_edge(from, and_node, ());
                ddnnf_graph.add_edge(and_node, child_nx, ());

                for literal in child_literals {
                    add_literal_node(ddnnf_graph, literal, and_node);
                }
            }
        };

    // the root node is per default the first node
    let mut root = NodeIndex::new(0);

    // add literals that are not mentioned in the ddnnf to the new root node
    for i in 1..=total_features {
        if !literal_occurences.borrow()[i as usize] {
            // add a new root which holds the unmentioned variables within the total_features range
            if root == NodeIndex::new(0) {
                root = ddnnf_graph.add_node(TId::And);
                ddnnf_graph.add_edge(root, NodeIndex::new(0), ());
            }

            add_literal_node(&mut ddnnf_graph, i, root);
        }
    }

    // Starting from an initial AND node, we delete all parent AND nodes.
    // We can do this because the start node has a FALSE node as children. Hence, it count is 0!
    let delete_parent_and_chain = |ddnnf_graph: &mut DdnnfGraph, start: NodeIndex| {
        let mut current_vec = Vec::new();
        let mut current = start;
        loop {
            if ddnnf_graph[current] == TId::And {
                // remove the AND node and all parent nodes that are also AND nodes
                let mut parents = ddnnf_graph.neighbors_directed(current, Incoming).detach();
                while let Some(parent) = parents.next_node(ddnnf_graph) {
                    current_vec.push(parent);
                }
                ddnnf_graph.remove_node(current);
            }

            match current_vec.pop() {
                Some(head) => current = head,
                None => break,
            }
        }
    };

    // second dfs:
    // Remove the True and False node if any is part of the dDNNF.
    // Those nodes can influence the core and dead features by reducing the amount of those,
    // we can identify simply by literal occurences.
    // Further, we decrease the size and complexity of the dDNNF.
    let mut dfs = DfsPostOrder::new(&ddnnf_graph, root);
    while let Some(nx) = dfs.next(&ddnnf_graph) {
        let mut neighbours = ddnnf_graph.neighbors_directed(nx, Outgoing).detach();
        loop {
            let next = neighbours.next(&ddnnf_graph);
            match next {
                Some((n_edge, n_node)) => {
                    if !ddnnf_graph.contains_node(n_node) {
                        continue;
                    }

                    if ddnnf_graph[n_node] == TId::True {
                        match ddnnf_graph[nx] {
                            TId::And => {
                                ddnnf_graph.remove_edge(n_edge).unwrap();
                            }
                            TId::Or => (), // should never happen
                            _ => {
                                // Bold, Red, Foreground Color (see https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797)
                                eprintln!("\x1b[1;38;5;196mERROR: Unexpected Nodetype while encoutering a True node. Only OR and AND nodes can have children. Aborting...");
                                process::exit(1);
                            }
                        };
                    }

                    if ddnnf_graph[n_node] == TId::False {
                        match ddnnf_graph[nx] {
                            TId::Or => {
                                ddnnf_graph.remove_edge(n_edge).unwrap();
                            }
                            TId::And => delete_parent_and_chain(&mut ddnnf_graph, nx),
                            _ => {
                                eprintln!("\x1b[1;38;5;196mERROR: Unexpected Nodetype while encoutering a False node. Only OR and AND nodes can have children. Aborting...");
                                process::exit(1);
                            }
                        };
                    }
                }
                None => break,
            }
        }
    }

    // third dfs:
    // Look at each or node. For each outgoing edge:
    // 1. Compute all literals that occur in the children of that edge
    // 2. Determine which literals occur only in the other paths
    // 3. Add those literals in the path we are currently looking at
    // Example:
    //
    //                                              OR
    //                  OR                       /      \
    //                /    \                   /         \
    //              Ln     AND      into     AND        AND
    //                    /   \             /   \      /   \
    //                   Lm   -Ln          Ln   OR    |   -Ln
    //                                         /  \  /
    //                                       -Lm   Lm
    //
    let mut literal_diff: HashMap<NodeIndex, HashSet<i32>> = get_literal_diffs(&ddnnf_graph, root);
    let mut dfs = DfsPostOrder::new(&ddnnf_graph, root);
    while let Some(nx) = dfs.next(&ddnnf_graph) {
        // edges between going from an and node to another node do not
        // have any weights attached to them. Therefore, we can skip them
        if ddnnf_graph[nx] == TId::Or {
            let children_diff = ddnnf_graph
                .neighbors_directed(nx, Outgoing)
                .map(|c| {
                    (
                        c,
                        HashSet::from_iter(
                            literal_diff
                                .get(&c)
                                .unwrap()
                                .clone()
                                .drain()
                                .map(|l| l.unsigned_abs()),
                        ),
                    )
                })
                .collect();

            balance_or_children(&mut ddnnf_graph, nx, diff(children_diff));
        }
    }

    extend_literal_diffs(&ddnnf_graph, &mut literal_diff, root);
    let intermediate_graph = IntermediateGraph::new(
        ddnnf_graph,
        root,
        total_features,
        literals_nx.borrow().clone(),
        literal_diff,
        cnf_path,
    );
    Ddnnf::new(intermediate_graph, total_features)
}

// Computes the difference between the children of a Node
fn diff(literals: Vec<(NodeIndex, HashSet<u32>)>) -> Vec<(NodeIndex, HashSet<u32>)> {
    let mut res: Vec<(NodeIndex, HashSet<u32>)> = Vec::new();
    for i in 0..literals.len() {
        let mut val: HashSet<u32> = HashSet::default();
        for (j, i_res) in literals.iter().enumerate() {
            if i != j {
                val.extend(i_res.1.clone());
            }
        }
        val = &val - &literals[i].1;
        if !val.is_empty() {
            res.push((literals[i].0, val));
        }
    }
    res
}

/// Computes the combined literals used in its children
pub fn get_literal_diffs(
    di_graph: &DdnnfGraph,
    root: NodeIndex,
) -> HashMap<NodeIndex, HashSet<i32>> {
    let mut safe: HashMap<NodeIndex, HashSet<i32>> = HashMap::new();
    let mut dfs = DfsPostOrder::new(di_graph, root);
    while let Some(nx) = dfs.next(di_graph) {
        get_literals(di_graph, &mut safe, nx);
    }
    safe
}

/// Computes the combined literals used in its children
pub fn extend_literal_diffs(
    di_graph: &DdnnfGraph,
    current_safe: &mut HashMap<NodeIndex, HashSet<i32>>,
    root: NodeIndex,
) {
    let mut dfs = DfsPostOrder::new(di_graph, root);
    while let Some(nx) = dfs.next(di_graph) {
        if !current_safe.contains_key(&nx) {
            get_literals(di_graph, current_safe, nx);
        }
    }
}

// determine what literal-nodes the current node is or which occur in its children
fn get_literals(
    di_graph: &DdnnfGraph,
    safe: &mut HashMap<NodeIndex, HashSet<i32>>,
    deciding_node_child: NodeIndex,
) -> HashSet<i32> {
    let lookup = safe.get(&deciding_node_child);
    if let Some(x) = lookup {
        return x.clone();
    }

    use c2d_lexer::TokenIdentifier::*;
    let mut res = HashSet::new();
    match di_graph[deciding_node_child] {
        And | Or => {
            di_graph
                .neighbors_directed(deciding_node_child, Outgoing)
                .for_each(|n| res.extend(get_literals(di_graph, safe, n)));
        }
        Literal { feature } => {
            res.insert(feature);
        }
        _ => (),
    }
    safe.insert(deciding_node_child, res.clone());
    res
}
