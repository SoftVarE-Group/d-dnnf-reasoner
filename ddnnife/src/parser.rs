pub mod c2d_lexer;
pub mod d4_lexer;
pub mod from_cnf;
pub mod graph;
pub mod persisting;
pub mod util;

use crate::ddnnf::{Ddnnf, extended_ddnnf::Attribute, node::Node};
use c2d_lexer::{C2DToken, TId, lex_line_c2d};
use core::panic;
use csv::ReaderBuilder;
use d4_lexer::{D4Token, lex_line_d4};
use graph::DdnnfGraph;
use itertools::Itertools;
use log::{error, warn};
use num::BigInt;
use petgraph::{
    Direction::{Incoming, Outgoing},
    graph::{EdgeIndex, NodeIndex},
    stable_graph::StableGraph,
    visit::DfsPostOrder,
};
use std::cell::RefMut;
use std::{
    cell::RefCell,
    cmp::max,
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
    process,
    rc::Rc,
};

/// Parses a ddnnf, referenced by the file path. The file gets parsed and we create
/// the corresponding data structure.
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use ddnnife::parser;
/// use ddnnife::Ddnnf;
///
/// let file_path = Path::new("./tests/data/small_ex_c2d.nnf");
///
/// let ddnnfx: Ddnnf = parser::build_ddnnf(file_path, None);
/// ```
///
/// # Panics
///
/// The function panics for an invalid file path.
#[inline]
#[allow(unused_mut)]
pub fn build_ddnnf(path: &Path, mut total_features: Option<u32>) -> Ddnnf {
    let mut ddnnf = match File::open(path) {
        Ok(file) => file,
        Err(error) => {
            error!("Failed to open {path:?}: {error}");
            process::exit(1);
        }
    };

    let lines = BufReader::new(ddnnf)
        .lines()
        .map(|result| match result {
            Ok(line) => line,
            Err(error) => {
                error!("Failed to read from {path:?}: {error}");
                process::exit(1);
            }
        })
        .collect::<Vec<String>>();

    distribute_building(lines, total_features)
}

/// Chooses, depending on the first read line, which building implmentation to choose.
/// Either the first line is a header and therefore the c2d format or total_features
/// is supplied and its the d4 format.
#[inline]
pub fn distribute_building(lines: Vec<String>, total_features: Option<u32>) -> Ddnnf {
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
                    build_d4_ddnnf(lines, Some(o))
                }
                None => {
                    // unknown standard or combination -> we assume d4 and choose total_features
                    warn!(
                        "The first line of the file isn't a header and the option 'total_features' is not set. \
                        Hence, we can't determine the number of variables and as a result, we might not be able to construct a valid ddnnf. \
                        Nonetheless, we build a ddnnf with our limited information, but we discourage using ddnnife in this manner."
                    );
                    build_d4_ddnnf(lines, None)
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
    let mut ddnnf_graph = DdnnfGraph::new();
    let mut node_indices = Vec::with_capacity(lines.len());
    let mut literals_nx: HashMap<i32, NodeIndex> = HashMap::new();

    // opens the file with a BufReader and
    // works off each line of the file data seperatly
    // skip the first line, because we already looked at the header
    for line in lines.into_iter().skip(1) {
        node_indices.push(match lex_line_c2d(line.as_ref()).unwrap().1 {
            C2DToken::And { children } => {
                let from = ddnnf_graph.add_node(TId::And);
                for child in children {
                    ddnnf_graph.add_edge(from, node_indices[child], ());
                }
                from
            }
            C2DToken::Or {
                decision: _,
                children,
            } => {
                let from = ddnnf_graph.add_node(TId::Or);
                for child in children {
                    ddnnf_graph.add_edge(from, node_indices[child], ());
                }
                from
            }
            C2DToken::Literal { feature } => {
                let literal_node = ddnnf_graph.add_node(TId::Literal { feature });
                literals_nx.insert(feature, literal_node);
                literal_node
            }
            C2DToken::True => ddnnf_graph.add_node(TId::True),
            C2DToken::False => ddnnf_graph.add_node(TId::False),
            _ => panic!("Tried to parse the header of the .nnf at the wrong time"),
        });
    }

    let root = node_indices[node_indices.len() - 1];
    Ddnnf::new(ddnnf_graph, root, variables)
}

/// Parses a ddnnf, referenced by the file path.
/// This function uses D4Tokens which specify a d-DNNF in d4 format.
/// The file gets parsed and we create the corresponding data structure.
#[inline]
fn build_d4_ddnnf(lines: Vec<String>, total_features_opt: Option<u32>) -> Ddnnf {
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

    // while parsing:
    // remove the weighted edges and substitute it with the corresponding
    // structure that uses AND-Nodes and Literal-Nodes. Example:
    //
    //                   n1                       n1
    //                 /   \                   /    \
    //              Ln|    |Lm     into     AND    AND
    //                \   /                /   \  /   \
    //                 n2                 Ln    n2    Lm
    //
    //
    let resolve_weighted_edge = |ddnnf_graph: &mut DdnnfGraph,
                                 from: NodeIndex,
                                 to: NodeIndex,
                                 edge: EdgeIndex,
                                 weights: Vec<i32>| {
        let literal_nodes =
            get_literal_indices(ddnnf_graph, weights, &mut literals_nx.borrow_mut());

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

            add_literal_node(
                &mut ddnnf_graph,
                i,
                root,
                &mut literals_nx.borrow_mut(),
                total_features,
            );
        }
    }

    // Second DFS: Remove all chains of AND nodes with False children and OR nodes with True children.
    let mut dfs = DfsPostOrder::new(&ddnnf_graph, root);

    // Visit each node.
    while let Some(node) = dfs.next(&ddnnf_graph) {
        let node_type = ddnnf_graph[node];

        // Only consider AND and OR nodes.
        if node_type != TId::And && node_type != TId::Or {
            continue;
        }

        // Check the child nodes.
        let mut neighbours = ddnnf_graph.neighbors_directed(node, Outgoing).detach();
        while let Some((_edge, neighbor)) = neighbours.next(&ddnnf_graph) {
            // Make sure it still exists before accessing.
            if !ddnnf_graph.contains_node(neighbor) {
                continue;
            }

            let child_type = ddnnf_graph[neighbor];

            // A False node as a child of an AND node leads to a chain of AND node deletions.
            if child_type == TId::False && node_type == TId::And {
                delete_chain(&mut ddnnf_graph, node);
            }

            // A True node as a child of an OR node leads to a chain of OR node deletions.
            if child_type == TId::True && node_type == TId::Or {
                delete_chain(&mut ddnnf_graph, node);
            }
        }
    }

    // Remove all True and False nodes.
    ddnnf_graph.retain_nodes(|graph, node| {
        let node = graph[node];
        node != TId::True && node != TId::False
    });

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

            balance_or_children(
                &mut ddnnf_graph,
                nx,
                diff(children_diff),
                &mut literals_nx.borrow_mut(),
                total_features,
            );
        }
    }

    extend_literal_diffs(&ddnnf_graph, &mut literal_diff, root);

    Ddnnf::new(ddnnf_graph, root, total_features)
}

fn get_literal_indices(
    ddnnf_graph: &mut StableGraph<TId, ()>,
    literals: Vec<i32>,
    lit_nx: &mut RefMut<HashMap<i32, NodeIndex>>,
) -> Vec<NodeIndex> {
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
}

/// Starting from an initial node, removes all parent nodes of the same type.
fn delete_chain(ddnnf_graph: &mut DdnnfGraph, start: NodeIndex) {
    // The type of node to remove.
    let node_type = ddnnf_graph[start];

    // A stack of nodes to remove.
    let mut to_remove = vec![start];

    // Continue as long as there are nodes to remove.
    while let Some(current) = to_remove.pop() {
        // Check all parent nodes of the current node.
        let mut parents = ddnnf_graph.neighbors_directed(current, Incoming).detach();
        while let Some(parent) = parents.next_node(ddnnf_graph) {
            // Schedule any parent node of the same type for removal.
            if ddnnf_graph[parent] == node_type {
                to_remove.push(parent);
            }
        }

        // Finally remove the node itself.
        ddnnf_graph.remove_node(current);
    }
}

fn balance_or_children(
    ddnnf_graph: &mut DdnnfGraph,
    from: NodeIndex,
    children: Vec<(NodeIndex, HashSet<u32>)>,
    lit_nx: &mut RefMut<HashMap<i32, NodeIndex>>,
    total_features: u32,
) {
    for (child_nx, child_literals) in children {
        let and_node = ddnnf_graph.add_node(TId::And);

        // place the newly created and node between the or node and its child
        ddnnf_graph.remove_edge(ddnnf_graph.find_edge(from, child_nx).unwrap());
        ddnnf_graph.add_edge(from, and_node, ());
        ddnnf_graph.add_edge(and_node, child_nx, ());

        for literal in child_literals {
            add_literal_node(ddnnf_graph, literal, and_node, lit_nx, total_features);
        }
    }
}

fn add_literal_node(
    ddnnf_graph: &mut DdnnfGraph,
    f_u32: u32,
    attach: NodeIndex,
    lit_nx: &mut RefMut<HashMap<i32, NodeIndex>>,
    total_features: u32,
) {
    let or_triangles: Rc<RefCell<Vec<Option<NodeIndex>>>> =
        Rc::new(RefCell::new(vec![None; (total_features + 1) as usize]));

    let f = f_u32 as i32;
    let mut ort = or_triangles.borrow_mut();

    if ort[f_u32 as usize].is_some() {
        ddnnf_graph.add_edge(attach, ort[f_u32 as usize].unwrap(), ());
    } else {
        let or = ddnnf_graph.add_node(TId::Or);
        ort[f_u32 as usize] = Some(or);

        let pos_lit = get_literal_indices(ddnnf_graph, vec![f], lit_nx)[0];
        let neg_lit = get_literal_indices(ddnnf_graph, vec![-f], lit_nx)[0];

        ddnnf_graph.add_edge(attach, or, ());
        ddnnf_graph.add_edge(or, pos_lit, ());
        ddnnf_graph.add_edge(or, neg_lit, ());
    }
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

// multiplies the count of all child Nodes of an And Node
#[inline]
fn calc_and_count(nodes: &mut [Node], indices: &[usize]) -> BigInt {
    indices.iter().map(|&index| &nodes[index].count).product()
}

// adds up the count of all child Nodes of an And Node
#[inline]
fn calc_or_count(nodes: &mut [Node], indices: &[usize]) -> BigInt {
    indices.iter().map(|&index| &nodes[index].count).sum()
}

/// Is used to parse the queries in the config files
/// The format is:
/// -> A feature is either positiv or negative i32 value with a leading "-"
/// -> Multiple features in the same line form a query
/// -> Queries are seperated by a new line ("\n")
///
/// # Example
/// ```
/// use std::path::Path;
/// use ddnnife::parser::parse_queries_file;
///
/// let config_path = Path::new("./tests/data/auto1.config");
/// let queries: Vec<(usize, Vec<i32>)> = parse_queries_file(config_path);
///
/// assert_eq!((0, vec![1044, 885]), queries[0]);
/// assert_eq!((1, vec![1284, -537]), queries[1]);
/// assert_eq!((2, vec![-1767, 675]), queries[2]);
/// ```
/// # Panic
///
/// Panics for a path to a non existing file
pub fn parse_queries_file(path: &Path) -> Vec<(usize, Vec<i32>)> {
    let file = open_file_savely(path);

    let lines = BufReader::new(file)
        .lines()
        .map(|line| line.expect("Unable to read line"));
    let mut parsed_queries: Vec<(usize, Vec<i32>)> = Vec::new();

    for (line_number, line) in lines.enumerate() {
        // takes a line of the file and parses the i32 values
        let res: Vec<i32> = line
            .split_whitespace()
            .map(|elem| elem.parse::<i32>()
                .unwrap_or_else(|_| panic!("Unable to parse {:?} into an i32 value while trying to parse the querie file at {:?}.\nCheck the help page with \"-h\" or \"--help\" for further information.\n", elem, path)))
            .collect();
        parsed_queries.push((line_number, res));
    }
    parsed_queries
}

pub fn build_attributes(path: &Path) -> HashMap<String, Attribute> {
    let file = open_file_savely(path);
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(file));

    let header = reader
        .headers()
        .expect("File must contain a header.")
        .clone();
    let records = reader
        .records()
        .map(|result| result.unwrap_or_else(|res| panic!("{:?} is not a valid record", res)))
        .collect_vec();

    let (id_idx, _) = header
        .iter()
        .find_position(|&column| column == "(Feature ID,Integer)")
        .unwrap_or_else(|| panic!("File does not contain the column \"(Feature ID,Integer)\"."));
    let record_ids = records.iter().map(|record| {
        record
            .get(id_idx)
            .unwrap_or_else(|| {
                panic!(
                    "Record {:?} has no value in column \"(Feature ID,Integer)\".",
                    record
                )
            })
            .parse::<u32>()
            .unwrap_or_else(|_| {
                panic!(
                    "Record {:?} has no integer value in column \"(Feature ID,Integer)\".",
                    record
                )
            })
    });
    let mut next_expected_id = 1;
    for id in record_ids {
        if id != next_expected_id {
            panic!(
                "Ids in column \"(Feature ID,Integer)\" must be in ascending order starting with \"1\" and increasing by one."
            )
        }
        next_expected_id += 1;
    }

    let regex = regex::Regex::new(r"^\((?<attr_name>[\w\s]+),(?<attr_type>[\w\s]+)\)$").unwrap();
    header
        .iter()
        .map(|col_name| {
            let groups = regex.captures(col_name).unwrap_or_else(|| {
                panic!("Colum \"{col_name}\" must have format \"(<attr_name>,<attr_type>)\".")
            });
            (
                groups["attr_name"].to_string(),
                groups["attr_type"].to_string(),
            )
        })
        .enumerate()
        .filter(|(_, (attr_name, _))| attr_name != "Feature ID" && attr_name != "Feature Name")
        .map(|(attr_idx, (attr_name, attr_type))| {
            let attr_vals = records
                .iter()
                .map(|record| record.get(attr_idx))
                .map(|entry| match entry {
                    Some("") => None,
                    other => other,
                });

            let attribute = match attr_type.as_str() {
                "Integer" => Attribute::new_integer_attr(
                    attr_vals
                        .map(|opt_val| opt_val.map(|val| val.parse().unwrap()))
                        .collect(),
                    Some(0),
                ),
                "Float" => Attribute::new_float_attr(
                    attr_vals
                        .map(|opt_val| opt_val.map(|val| val.parse().unwrap()))
                        .collect(),
                    Some(0.0),
                ),
                "Bool" => Attribute::new_bool_attr(
                    attr_vals
                        .map(|opt_val| opt_val.map(|val| val.parse().unwrap()))
                        .collect(),
                    Some(false),
                ),
                "String" => Attribute::new_string_attr(
                    attr_vals
                        .map(|opt_val| opt_val.map(|val| val.to_string()))
                        .collect(),
                    Some("".to_string()),
                ),
                _ => panic!("Unexpected attribute type."),
            };

            (attr_name.to_string(), attribute)
        })
        .collect()
}

/// Tries to open a file.
/// If an error occurs the program prints the error and exists.
pub fn open_file_savely(path: &Path) -> File {
    // opens the file with a BufReader and
    // works off each line of the file data seperatly
    match File::open(path) {
        Ok(x) => x,
        Err(err) => {
            // Bold, Red, Foreground Color (see https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797)
            error!(
                "The following error code occured while trying to open the file {}:\n{}\nAborting...",
                path.to_str().expect("Failed to serialize path."),
                err
            );
            process::exit(1);
        }
    }
}
