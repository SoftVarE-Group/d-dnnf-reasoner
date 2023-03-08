pub mod c2d_lexer;
use c2d_lexer::{lex_line, TId, C2DToken};

pub mod d4_lexer;
use colour::{red, green, yellow};
use d4_lexer::{lex_line_d4, D4Token};

pub mod persisting;

use core::panic;
use std::{
    cell::RefCell,
    rc::Rc, process
};

use rug::{Integer, Complete};
use rustc_hash::{FxHashMap, FxHashSet};

pub mod bufreader_for_big_files;
use bufreader_for_big_files::BufReaderMl;

use crate::ddnnf::{Ddnnf, node::Node, node::NodeType};

use petgraph::graph::DiGraph;
use petgraph::{
    graph::{EdgeIndex, NodeIndex},
    visit::DfsPostOrder,
    Direction::Outgoing,
};

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
/// let file_path = "./tests/data/small_test.dimacs.nnf";
///
/// let ddnnfx: Ddnnf = parser::build_ddnnf_tree_with_extras(file_path);
/// ```
///
/// # Panics
///
/// The function panics for an invalid file path.
#[inline]
pub fn build_ddnnf_tree_with_extras(path: &str) -> Ddnnf {
    use C2DToken::*;

    let mut buf_reader = BufReaderMl::open(path).expect("Unable to open file");

    let first_line = buf_reader.next().expect("Unable to read line").unwrap();
    let nodes; let variables;

    match lex_line(first_line.trim()) {
        Ok((_, Header { nodes: n, edges: _, variables: v })) => { nodes = n; variables = v; },
        Ok(_) | Err(_) => { // tried to parse the d4 standard without -o option which results in trying to parse the c2d standard
            red!("error: "); print!("the first line of the file isn't a header!\n\nYou probably should use the option: ");
            yellow!("'--ommited_features <OMMITED FEATURES>'");
            print!(".\nFor more information try "); green!("--help\n");
            process::exit(1);
        },
    }

    let mut parsed_nodes: Vec<Node> = Vec::with_capacity(nodes);

    let mut literals: FxHashMap<i32, usize> = FxHashMap::default();
    let mut true_nodes = Vec::new();

    // opens the file with a BufReaderMl which is similar to a regular BufReader
    // works off each line of the file data seperatly
    for line in buf_reader {
        let line = line.expect("Unable to read line");

        let next: Node = match lex_line(line.as_ref()).unwrap().1 {
            And { children } => Node::new_and(
                calc_and_count(&mut parsed_nodes, &children),
                children,
            ),
            Or { decision, children } => {
                Node::new_or(
                    decision,
                    calc_or_count(&mut parsed_nodes, &children),
                    children,
                )
            }
            Literal { feature } => Node::new_literal(feature),
            True => Node::new_bool(true),
            False => Node::new_bool(false),
            _ => panic!(
                "Tried to parse the header of the .nnf at the wrong time"
            ),
        };

        // fill the parent node pointer, save literals
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

    Ddnnf::new(parsed_nodes, literals, true_nodes, variables as u32, nodes)
}

/// Parses a ddnnf, referenced by the file path.
/// This function uses D4Tokens which specify a d-DNNF in d4 format.
/// The file gets parsed and we create the corresponding data structure.
#[inline]
pub fn build_d4_ddnnf_tree(path: &str, ommited_features: u32) -> Ddnnf {
    let buf_reader = BufReaderMl::open(path).expect("Unable to open file");

    let mut ddnnf_graph = DiGraph::<TId, ()>::new();

    let literal_occurences: Rc<RefCell<Vec<bool>>> =
        Rc::new(RefCell::new(vec![false; (ommited_features + 1) as usize]));

    let mut indices: Vec<NodeIndex> = Vec::new();

    // With the help of the literals node state, we can add the required nodes
    // for the balancing of the or nodes to archieve smoothness
    let nx_literals: Rc<RefCell<FxHashMap<NodeIndex, i32>>> =
        Rc::new(RefCell::new(FxHashMap::default()));
    let literals_nx: Rc<RefCell<FxHashMap<i32, NodeIndex>>> =
        Rc::new(RefCell::new(FxHashMap::default()));

    let get_literal_indices = |ddnnf_graph: &mut DiGraph<TId, ()>,
                               literals: Vec<i32>|
     -> Vec<NodeIndex> {
        let mut nx_lit = nx_literals.borrow_mut();
        let mut lit_nx = literals_nx.borrow_mut();

        let mut literal_nodes = Vec::new();

        for literal in literals {
            if literal.is_positive() {
                literal_nodes.push(match lit_nx.get(&literal) {
                    Some(x) => *x,
                    None => {
                        let nx = ddnnf_graph.add_node(TId::PositiveLiteral);
                        nx_lit.insert(nx, literal);
                        lit_nx.insert(literal, nx);
                        nx
                    }
                })
            } else {
                literal_nodes.push(match lit_nx.get(&literal) {
                    Some(x) => *x,
                    None => {
                        let nx = ddnnf_graph.add_node(TId::NegativeLiteral);
                        nx_lit.insert(nx, literal);
                        lit_nx.insert(literal, nx);
                        nx
                    }
                })
            }
        }
        literal_nodes
    };

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
    let resolve_weighted_edge =
    |ddnnf_graph: &mut DiGraph<TId, ()>,
        from: NodeIndex,
        to: NodeIndex,
        edge: EdgeIndex,
        weights: Vec<i32>| {
        let and_node = ddnnf_graph.add_node(TId::And);
        let literal_nodes = get_literal_indices(
            ddnnf_graph,
            weights,
        );

        ddnnf_graph.remove_edge(edge);

        ddnnf_graph.add_edge(from, and_node, ());
        for node in literal_nodes {
            ddnnf_graph.add_edge(and_node, node, ());
        }
        ddnnf_graph.add_edge(and_node, to, ());
    };

    // opens the file with a BufReaderMl which is similar to a regular BufReader
    // works off each line of the file data seperatly
    for line in buf_reader {
        let line = line.expect("Unable to read line");

        let next: D4Token = lex_line_d4(line.as_ref()).unwrap().1;

        use D4Token::*;
        match next {
            Edge { from, to, features } => {
                for f in &features {
                    literal_occurences.borrow_mut()[f.unsigned_abs() as usize] = true;
                }
                let from_n = indices[from as usize - 1];
                let to_n = indices[to as usize - 1];
                let edge = ddnnf_graph.add_edge(
                    from_n,
                    to_n,
                    ()
                );
                resolve_weighted_edge(&mut ddnnf_graph, from_n, to_n, edge, features);
            }
            And => indices.push(ddnnf_graph.add_node(TId::And)),
            Or => indices.push(ddnnf_graph.add_node(TId::Or)),
            True => indices.push(ddnnf_graph.add_node(TId::True)),
            False => indices.push(ddnnf_graph.add_node(TId::False)),
        }
    }

    let or_triangles: Rc<RefCell<Vec<Option<NodeIndex>>>> =
        Rc::new(RefCell::new(vec![None; (ommited_features + 1) as usize]));

    let add_literal_node = 
        |ddnnf_graph: &mut DiGraph<TId,()>, f_u32: u32, attach: NodeIndex| {
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
        |ddnnf_graph: &mut DiGraph<TId, ()>,
         from: NodeIndex,
         children: Vec<(NodeIndex, FxHashSet<u32>)>| {
            for child in children {
                let and_node = ddnnf_graph.add_node(TId::And);

                // place the newly created and node between the or node and its child
                ddnnf_graph
                    .remove_edge(ddnnf_graph.find_edge(from, child.0).unwrap());
                ddnnf_graph.add_edge(from, and_node, ());
                ddnnf_graph.add_edge(and_node, child.0, ());

                for literal in child.1 {
                    add_literal_node(ddnnf_graph, literal, and_node);
                }
            }
        };

    // add a new root which hold the unmentioned variables within the ommited_features range
    let root = ddnnf_graph.add_node(TId::And);
    ddnnf_graph.add_edge(root, NodeIndex::new(0), ());

    // add literals that are not mentioned in the ddnnf to the new root node
    for i in 1..ommited_features + 1 {
        if !literal_occurences.borrow()[i as usize] {
            add_literal_node(&mut ddnnf_graph, i, root);
        }
    }

    // snd dfs:
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
    let mut safe: FxHashMap<NodeIndex, FxHashSet<u32>> = FxHashMap::default();
    let mut dfs = DfsPostOrder::new(&ddnnf_graph, root);
    while let Some(nx) = dfs.next(&ddnnf_graph) {
        // edges between going from an and node to another node do not
        // have any weights attached to them. Therefore, we can skip them
        if ddnnf_graph[nx] == TId::Or {
            let diffrences = get_literal_diff(
                &ddnnf_graph,
                &mut safe,
                &nx_literals.borrow(),
                nx,
            );
            balance_or_children(&mut ddnnf_graph, nx, diffrences);
        }
    }

    // perform a depth first search to get the nodes ordered such
    // that child nodes are listed before their parents
    // transform that interim representation into a node vector
    dfs = DfsPostOrder::new(&ddnnf_graph, root);
    let mut nd_to_usize: FxHashMap<NodeIndex, usize> = FxHashMap::default();

    let mut parsed_nodes: Vec<Node> = Vec::with_capacity(ddnnf_graph.node_count());
    let mut literals: FxHashMap<i32, usize> = FxHashMap::default();
    let mut true_nodes = Vec::new();
    let nx_lit = nx_literals.borrow();

    while let Some(nx) = dfs.next(&ddnnf_graph) {
        nd_to_usize.insert(nx, parsed_nodes.len());
        let neighs = ddnnf_graph
            .neighbors(nx)
            .map(|n| *nd_to_usize.get(&n).unwrap())
            .collect::<Vec<usize>>();
        let next: Node = match ddnnf_graph[nx] {
            // extract the parsed Token
            TId::PositiveLiteral |
            TId::NegativeLiteral => {
                Node::new_literal(nx_lit.get(&nx).unwrap().to_owned())
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

    let len = parsed_nodes.len();
    Ddnnf::new(parsed_nodes, literals, true_nodes, ommited_features, len)
}

// determine the differences in literal-nodes occuring in the child nodes
fn get_literal_diff(
    di_graph: &DiGraph<TId, ()>,
    safe: &mut FxHashMap<NodeIndex, FxHashSet<u32>>,
    nx_literals: &FxHashMap<NodeIndex, i32>,
    or_node: NodeIndex,
) -> Vec<(NodeIndex, FxHashSet<u32>)> {
    let mut inter_res = Vec::new();
    let neighbors = di_graph.neighbors_directed(or_node, Outgoing);

    for neighbor in neighbors {
        inter_res
            .push((neighbor, get_literals(di_graph, safe, nx_literals, neighbor)));
    }

    let mut res: Vec<(NodeIndex, FxHashSet<u32>)> = Vec::new();
    for i in 0..inter_res.len() {
        let mut val: FxHashSet<u32> = FxHashSet::default();
        for (j, i_res) in inter_res.iter().enumerate() {
            if i != j {
                val.extend(&i_res.1);
            }
        }
        val = &val - &inter_res[i].1;
        if !val.is_empty() {
            res.push((inter_res[i].0, val));
        }
    }
    res
}

// determine what literal-nodes the current node is or which occur in its children
fn get_literals(
    di_graph: &DiGraph<TId, ()>,
    safe: &mut FxHashMap<NodeIndex, FxHashSet<u32>>,
    nx_literals: &FxHashMap<NodeIndex, i32>,
    or_child: NodeIndex,
) -> FxHashSet<u32> {
    let lookup = safe.get(&or_child);
    if let Some(x) = lookup {
        return x.clone();
    }

    let mut res = FxHashSet::default();
    use c2d_lexer::TokenIdentifier::*;
    match di_graph[or_child] {
        And | Or => {
            di_graph.neighbors_directed(or_child, Outgoing).for_each(|n| {
                res.extend(get_literals(di_graph, safe, nx_literals, n))
            });
            safe.insert(or_child, res.clone());
        }
        PositiveLiteral | NegativeLiteral => {
            res.insert(nx_literals.get(&or_child).unwrap().unsigned_abs() as u32);
            safe.insert(or_child, res.clone());
        }
        _ => (),
    }
    res
}

// multiplies the count of all child Nodes of an And Node
#[inline]
fn calc_and_count(nodes: &mut [Node], indices: &[usize]) -> Integer {
    Integer::product(indices.iter().map(|&index| &nodes[index].count))
        .complete()
}

// adds up the count of all child Nodes of an And Node
#[inline]
fn calc_or_count(
    nodes: &mut [Node],
    indices: &[usize],
) -> Integer {
    Integer::sum(indices.iter().map(|&index| &nodes[index].count)).complete()
}

/// Is used to parse the queries in the config files
/// The format is:
/// -> A feature is either positiv or negative i32 value with a leading "-"
/// -> Multiple features in the same line form a query
/// -> Queries are seperated by a new line ("\n")
///
/// # Example
/// ```
/// use ddnnf_lib::parser::parse_queries_file;
///
/// let config_path = "./tests/data/auto1.config";
/// let queries: Vec<(usize, Vec<i32>)> = parse_queries_file(config_path);
///
/// assert_eq!((0, vec![1044, 885]), queries[0]);
/// assert_eq!((1, vec![1284, -537]), queries[1]);
/// assert_eq!((2, vec![-1767, 675]), queries[2]);
/// ```
/// # Panic
///
/// Panics for a path to a non existing file
pub fn parse_queries_file(path: &str) -> Vec<(usize, Vec<i32>)> {
    let buf_reader = BufReaderMl::open(path).expect("Unable to open file");
    let mut parsed_queries: Vec<(usize, Vec<i32>)> = Vec::new();

    // opens the file with a BufReaderMl which is similar to a regular BufReader
    // works off each line of the file data seperatly
    for (line_number, line) in buf_reader.enumerate() {
        let l = line.expect("Unable to read line");

        // takes a line of the file and parses the i32 values
        let res: Vec<i32> = l.as_ref().split_whitespace().into_iter()
        .map(|elem| elem.to_string().parse::<i32>()
            .unwrap_or_else(|_| panic!("Unable to parse {:?} into an i32 value while trying to parse the querie file at {:?}.\nCheck the help page with \"-h\" or \"--help\" for further information.\n", elem, path))
        ).collect();
        parsed_queries.push((line_number, res));
    }
    parsed_queries
}