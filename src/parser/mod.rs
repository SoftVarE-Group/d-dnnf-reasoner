pub mod lexer;
use lexer::{lex_line, TId};

pub mod d4_lexer;
use d4_lexer::{lex_line_d4, D4Token};

use core::panic;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc
};

use rug::{Complete, Integer};

pub mod bufreader_for_big_files;
use bufreader_for_big_files::BufReaderMl;

use crate::data_structure::{
    Ddnnf, Node,
    NodeType::{And, False, Literal, Or, True},
};

use petgraph::{
    graph::{NodeIndex, EdgeIndex},
    visit::{DfsPostOrder}, Direction::Outgoing,
};
use petgraph::{Graph};

/// Parses a ddnnf, referenced by the file path. The file gets parsed and we create
/// the corresponding data structure.
///
/// # Examples
///
/// ```
/// extern crate ddnnf_lib;
/// use ddnnf_lib::parser;
/// use ddnnf_lib::data_structure::Ddnnf;
///
/// let file_path = "./tests/data/small_test.dimacs.nnf";
///
/// let ddnnf: Ddnnf = parser::build_ddnnf_tree(file_path);
/// let ddnnfx: Ddnnf = parser::build_ddnnf_tree_with_extras(file_path);
/// ```
///
/// # Panics
///
/// The function panics for an invalid file path.
#[inline]
pub fn build_ddnnf_tree(path: &str) -> Ddnnf {
    let mut buf_reader = BufReaderMl::open(path).expect("Unable to open file");

    let first_line = buf_reader.next().expect("Unable to read line").unwrap();
    let header = lex_line(first_line.trim()).unwrap().1 .1;

    let mut parsed_nodes: Vec<Node> = Vec::with_capacity(header[0] as usize);

    // opens the file with a BufReaderMl which is similar to a regular BufReader
    // works off each line of the file data seperatly
    for line in buf_reader {
        let line = line.expect("Unable to read line");

        let next = match lex_line(line.trim()).unwrap().1 {
            // extract the parsed Token
            (TId::PositiveLiteral, v) => Node::new_literal(v[0] as i32),
            (TId::NegativeLiteral, v) => Node::new_literal(-(v[0] as i32)),
            (TId::And, v) => Node::new_and(
                v.clone(),
                calc_overall_and_count(&mut parsed_nodes, &v),
            ),
            (TId::Or, v) => {
                let v_num = v.clone().remove(0) as i32;
                let mut children = v.clone();
                children.remove(0);
                Node::new_or(
                    v_num,
                    children,
                    calc_overall_or_count(&mut parsed_nodes, &v),
                )
            }
            (TId::True, _) => Node::new_bool(True),
            (TId::False, _) => Node::new_bool(False),
            _ => panic!(
                "Tried to parse the header of the .nnf at the wrong time"
            ),
        };

        parsed_nodes.push(next);
    }

    Ddnnf::new(parsed_nodes, HashMap::new(), header[2] as u32, header[0])
}

#[inline]
/// Adds the parent connection as well as the hashmpa for the literals and their corresponding position in the vector
/// Works analog to build_ddnnf_tree()
pub fn build_ddnnf_tree_with_extras(path: &str) -> Ddnnf {
    let mut buf_reader = BufReaderMl::open(path).expect("Unable to open file");

    let first_line = buf_reader.next().expect("Unable to read line").unwrap();
    let header = lex_line(first_line.trim()).unwrap().1 .1;

    let mut parsed_nodes: Vec<Node> = Vec::with_capacity(header[0]);

    let mut literals: HashMap<i32, usize> = HashMap::new();

    // opens the file with a BufReaderMl which is similar to a regular BufReader
    // works off each line of the file data seperatly
    for line in buf_reader {
        let line = line.expect("Unable to read line");

        let next: Node = match lex_line(line.as_ref()).unwrap().1 {
            // extract the parsed Token
            (TId::PositiveLiteral, v) => Node::new_literal(v[0] as i32),
            (TId::NegativeLiteral, v) => Node::new_literal(-(v[0] as i32)),
            (TId::And, v) => Node::new_and(
                v.clone(),
                calc_overall_and_count(&mut parsed_nodes, &v),
            ),
            (TId::Or, v) => {
                let v_num = v.clone().remove(0) as i32;
                let mut children = v.clone();
                children.remove(0);
                Node::new_or(
                    v_num,
                    children,
                    calc_overall_or_count(&mut parsed_nodes, &v),
                )
            }
            (TId::True, _) => Node::new_bool(True),
            (TId::False, _) => Node::new_bool(False),
            _ => panic!(
                "Tried to parse the header of the .nnf at the wrong time"
            ),
        };

        // fill the parent node pointer
        if next.node_type == And || next.node_type == Or {
            let next_childs: Vec<usize> = next.children.clone().unwrap();

            let next_indize: usize = parsed_nodes.len();

            for i in next_childs {
                parsed_nodes[i].parents.push(next_indize);
            }
        }

        // fill the HashMap with the literals
        if next.node_type == Literal {
            if literals.contains_key(&next.var_number.unwrap()) {
                panic!(
                    "Literal {:?} occured at least twice. This violates the standard!",
                    next.var_number.unwrap()
                );
            }
            literals.insert(next.var_number.unwrap(), parsed_nodes.len());
        }

        parsed_nodes.push(next);
    }

    Ddnnf::new(parsed_nodes, literals, header[2] as u32, header[0])
}

/// Parses a ddnnf, referenced by the file path.
/// This function uses D4Tokens which specify a d-DNNF in d4 format.
/// The file gets parsed and we create the corresponding data structure.
/// TODO actually return a smooth! d-DNNF
#[inline]
pub fn build_d4_ddnnf_tree(path: &str, ommited_features: u32) -> Ddnnf {
    let buf_reader = BufReaderMl::open(path).expect("Unable to open file");

    let mut ddnnf_graph = Graph::<TId, Vec<i32>>::new();

    // We have to remember the state of our literals because only two occurence of
    // each literal (one positive and one negative) are allowed
    #[derive(PartialEq, Debug, Clone)]
    enum LiteralOcc {
        Neither,
        Positive,
        Negative,
        Both,
    }

    let literal_occurences: Rc<RefCell<Vec<LiteralOcc>>> =
        Rc::new(RefCell::new(vec![
            LiteralOcc::Neither;
            (ommited_features + 1) as usize
        ]));

    let lit_occ_change = |f: &i32| {
        use LiteralOcc::*;
        let mut lo = literal_occurences.borrow_mut();
        let f_abs = f.abs() as usize;
        match (lo[f_abs].clone(), f > &0) {
            (Neither, true) => lo[f_abs] = Positive,
            (Neither, false) => lo[f_abs] = Negative,
            (Positive, true) => (),
            (Positive, false) => lo[f_abs] = Both,
            (Negative, true) => lo[f_abs] = Both,
            (Negative, false) => (),
            (Both, _) => (),
        }
    };

    let mut indices: HashMap<i32, NodeIndex> = HashMap::new();

    // opens the file with a BufReaderMl which is similar to a regular BufReader
    // works off each line of the file data seperatly
    for line in buf_reader {
        let line = line.expect("Unable to read line");

        let next: D4Token = lex_line_d4(line.as_ref()).unwrap().1;

        use D4Token::*;
        match next {
            Edge { from, to, features } => {
                for f in &features {
                    lit_occ_change(f);
                }
                ddnnf_graph.add_edge(
                    indices.get(&from).unwrap().to_owned(),
                    indices.get(&to).unwrap().to_owned(),
                    features,
                );
            }
            And { position } => {
                indices.insert(position, ddnnf_graph.add_node(TId::And));
            }
            Or { position } => {
                indices.insert(position, ddnnf_graph.add_node(TId::Or));
            }
            True { position } => {
                let tx = ddnnf_graph.add_node(TId::True);
                indices.insert(position, tx);
            }
            False { position } => {
                indices.insert(position, ddnnf_graph.add_node(TId::False));
            }
        }
    }

    let root = ddnnf_graph.add_node(TId::And);
    let old_root: NodeIndex = NodeIndex::new(0);
    ddnnf_graph.add_edge(root, old_root, Vec::new());

    // With the help of the literals node state, we can add the required nodes
    // for the balancing of the or nodes to archieve smoothness
    let nx_literals: Rc<RefCell<HashMap<NodeIndex, i32>>> =
        Rc::new(RefCell::new(HashMap::new()));
    let literals_nx: Rc<RefCell<HashMap<i32, NodeIndex>>> =
        Rc::new(RefCell::new(HashMap::new()));

    let add_literal_node =
        |ddnnf_graph: &mut Graph<TId, Vec<i32>>, f_u32: u32, attach: NodeIndex| {
        let f = f_u32 as i32; // we enforce f as u32 to avoid signum issues
        let mut nx_lit = nx_literals.borrow_mut();
        let mut lit_nx = literals_nx.borrow_mut();

        lit_occ_change(&(f as i32));
        let or = ddnnf_graph.add_node(TId::Or);
        let pos_lit = match lit_nx.get(&f) {
            Some(x) => *x,
            None => {
                let nx = ddnnf_graph.add_node(TId::PositiveLiteral);
                nx_lit.insert(nx, f);
                lit_nx.insert(f, nx);
                nx
            }
        };
        let neg_lit = match lit_nx.get(&-f) {
            Some(x) => *x,
            None => {
                let nx = ddnnf_graph.add_node(TId::NegativeLiteral);
                nx_lit.insert(nx, -f);
                lit_nx.insert(-f, nx);
                nx
            }
        };

        ddnnf_graph.add_edge(attach, or, Vec::new());
        ddnnf_graph.add_edge(or, pos_lit, Vec::new());
        ddnnf_graph.add_edge(or, neg_lit, Vec::new());
    };

    let get_literal_indices = 
    |ddnnf_graph: &mut Graph<TId, Vec<i32>>, literals: Vec<i32>| -> Vec<NodeIndex> {
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

    let resolve_weighted_edge =
    |ddnnf_graph: &mut Graph<TId, Vec<i32>>, from: NodeIndex, to: NodeIndex, edge: EdgeIndex| {
        let and_node = ddnnf_graph.add_node(TId::And);
        let literal_nodes = get_literal_indices(ddnnf_graph, ddnnf_graph.edge_weight(edge).unwrap().clone());

        ddnnf_graph.remove_edge(edge);

        ddnnf_graph.add_edge(from, and_node, Vec::new());
        for node in literal_nodes {
            ddnnf_graph.add_edge(and_node, node, Vec::new());
        }
        ddnnf_graph.add_edge(and_node, to, Vec::new());
    };

    let balance_or_children =
    |ddnnf_graph: &mut Graph<TId, Vec<i32>>, from: NodeIndex, children: Vec<(NodeIndex, HashSet<u32>)>| {
        if children.is_empty() { return; }

        for child in children {
            if child.1.is_empty() { continue; }

            let and_node = ddnnf_graph.add_node(TId::And);

            // place the newly created and node between the or node and its child
            ddnnf_graph.remove_edge(ddnnf_graph.find_edge(from, child.0).unwrap());
            ddnnf_graph.add_edge(from, and_node, Vec::new());
            ddnnf_graph.add_edge(and_node, child.0, Vec::new());

            for literal in child.1 {
                add_literal_node(ddnnf_graph, literal, and_node);
            }
        }
    };

    // add literals that are not mentioned in the ddnnf to the new root node
    for i in 1..ommited_features + 1 {
        if literal_occurences.borrow()[i as usize] == LiteralOcc::Neither {
            add_literal_node(&mut ddnnf_graph, i, root);
        }
    }

    // first dfs:
    // remove the weighted edges and substitute it with the corresponding
    // structure that uses AND-Nodes and Literal-Nodes. Example: 
    //
    //                  n1                       n1
    //                /    \                   /    \
    //              Ln|    |Lm      into     AND    AND
    //                \    /                /   \  /   \
    //                  n2                 Ln    n2    Lm
    //
    //
    let mut dfs = DfsPostOrder::new(&ddnnf_graph, root);
    while let Some(nx) = dfs.next(&ddnnf_graph) {
        // edges between going from an and node to another node do not
        // have any weights attached to them. Therefore, we can skip them
        if ddnnf_graph[nx] != TId::Or { continue; }
        let mut neighbors = ddnnf_graph.neighbors_directed(nx, Outgoing).detach();

        while let Some(neighbor) = neighbors.next(&ddnnf_graph) {
            if ddnnf_graph.edge_weight(neighbor.0).unwrap().is_empty() { continue; }
            resolve_weighted_edge(&mut ddnnf_graph, nx, neighbor.1, neighbor.0);
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
    //                  OR                       /  \
    //                /    \                   /      \
    //              Ln     AND      into     AND      AND
    //                    /   \             /   \    /   \
    //                   Lm   -Ln          Ln   OR   |  -Ln
    //                                         /  \  /
    //                                       -Lm   Lm       
    //

    let mut safe: HashMap<NodeIndex, HashSet<u32>> = HashMap::new();
    dfs = DfsPostOrder::new(&ddnnf_graph, root);
    while let Some(nx) = dfs.next(&ddnnf_graph) {
        // edges between going from an and node to another node do not
        // have any weights attached to them. Therefore, we can skip them
        if ddnnf_graph[nx] != TId::Or { continue; }
        let diffrences = get_literal_diff(&ddnnf_graph, &mut safe, &nx_literals.borrow(), nx);
        balance_or_children(&mut ddnnf_graph, nx, diffrences);
    }

    // perform a depth first search to get the nodes ordered such
    // that child nodes are listed before their parents
    // transform that interim representation into a node vector
    let mut dfs = DfsPostOrder::new(&ddnnf_graph, root);
    let mut nd_to_usize: HashMap<NodeIndex, usize> = HashMap::new();

    let mut parsed_nodes: Vec<Node> = Vec::new();
    let mut literals: HashMap<i32, usize> = HashMap::new();

    while let Some(nx) = dfs.next(&ddnnf_graph) {
        nd_to_usize.insert(nx, parsed_nodes.len());
        let neighs = ddnnf_graph
            .neighbors(nx)
            .map(|n| *nd_to_usize.get(&n).unwrap())
            .collect::<Vec<usize>>();
        let next: Node = match ddnnf_graph[nx] {
            // extract the parsed Token
            TId::PositiveLiteral => {
                Node::new_literal(*nx_literals.borrow().get(&nx).unwrap())
            }
            TId::NegativeLiteral => {
                Node::new_literal(*nx_literals.borrow().get(&nx).unwrap())
            }
            TId::And => Node::new_and(
                neighs.clone(),
                calc_overall_and_count(&mut parsed_nodes, &neighs),
            ),

            TId::Or => Node::new_or(
                0,
                neighs.clone(),
                calc_overall_or_count_multiple_children(&mut parsed_nodes, &neighs),
            ),
            TId::True => Node::new_bool(True),
            TId::False => Node::new_bool(False),
            other => panic!(
                "There should not be any other Nodetypes as And, Or, and Literals. Found Nodetype: {:?}", other
            ),
        };

        // fill the parent node pointer
        if next.node_type == And || next.node_type == Or {
            let next_childs: Vec<usize> = next.children.clone().unwrap();

            let next_indize: usize = parsed_nodes.len();

            for i in next_childs {
                parsed_nodes[i].parents.push(next_indize);
            }
        }

        // fill the HashMap with the literals
        if next.node_type == Literal {
            if literals.contains_key(&next.var_number.unwrap()) {
                panic!(
                    "Literal {:?} occured at least twice. This violates the standard!",
                    next.var_number.unwrap()
                );
            }
            literals.insert(next.var_number.unwrap(), parsed_nodes.len());
        }

        parsed_nodes.push(next);
    }

    let len = parsed_nodes.len();
    Ddnnf::new(parsed_nodes, literals, ommited_features, len)
}

// determine the differences in literal-nodes occuring in the child nodes
fn get_literal_diff(graph: &Graph<TId, Vec<i32>>, safe: &mut HashMap<NodeIndex, HashSet<u32>>, nx_literals: &HashMap<NodeIndex, i32>, or_node: NodeIndex) -> Vec<(NodeIndex, HashSet<u32>)> {
    let mut inter_res = Vec::new();
    let neighbors = graph.neighbors_directed(or_node, Outgoing);
    
    for neighbor in neighbors {
        inter_res.push((neighbor, get_literals(graph, safe, nx_literals, neighbor)));
    }

    let mut res: Vec<(NodeIndex, HashSet<u32>)> = Vec::new();
    for i in 0..inter_res.len() {
        let mut val: HashSet<u32> = HashSet::new();
        for (j, i_res) in inter_res.iter().enumerate() {
            if i == j { continue; }
            val.extend(&i_res.1);
        }
        res.push((inter_res[i].0, &val - &inter_res[i].1));
    }
    res
}

// determine what literal-nodes the current node is or which occur in its children
fn get_literals(graph: &Graph<TId, Vec<i32>>, safe: &mut HashMap<NodeIndex, HashSet<u32>>, nx_literals: &HashMap<NodeIndex, i32>, or_child: NodeIndex) -> HashSet<u32> {
    let lookup = safe.get(&or_child);
    if let Some(x) = lookup {
        return x.clone();
    }

    let mut res = HashSet::new();
    use lexer::TokenIdentifier::*;
    match graph[or_child] {
        And | Or
            => { graph.neighbors_directed(or_child, Outgoing)
            .for_each(|n| res.extend(get_literals(graph, safe, nx_literals, n))); safe.insert(or_child, res.clone()); },
        PositiveLiteral | NegativeLiteral
            => { res.insert(nx_literals.get(&or_child).unwrap().abs() as u32); safe.insert(or_child, res.clone()); },
        _ => { safe.insert(or_child, res.clone()); },
    }
    res
}

// multiplies the overall_count of all child Nodes of an And Node
#[inline]
fn calc_overall_and_count(nodes: &mut Vec<Node>, indizes: &[usize]) -> Integer {
    Integer::product(indizes.iter().map(|indize| &nodes[*indize].count))
        .complete()
}

// adds up the overall_count of all child Nodes of an And Node
#[inline]
fn calc_overall_or_count(nodes: &mut Vec<Node>, indizes: &[usize]) -> Integer {
    Integer::from(&nodes[indizes[1]].count + &nodes[indizes[2]].count)
}

#[inline]
fn calc_overall_or_count_multiple_children(
    nodes: &mut Vec<Node>,
    indizes: &[usize],
) -> Integer {
    Integer::sum(indizes.iter().map(|indize| &nodes[*indize].count))
        .complete()
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
/// // an example for a config file (and the actuall start of "./tests/data/axTLS.config"):
/// // -62 86
/// // 61 86
/// // 36 -83
///
/// let config_path = "./tests/data/axTLS.config";
/// let queries: Vec<Vec<i32>> = parse_queries_file(config_path);
///
/// assert_eq!(vec![-62, 86], queries[0]);
/// assert_eq!(vec![61, 86], queries[1]);
/// assert_eq!(vec![36, -83], queries[2]);
/// ```
/// # Panic
///
/// Panics for a path to a non existing file
pub fn parse_queries_file(path: &str) -> Vec<Vec<i32>> {
    let buf_reader = BufReaderMl::open(path).expect("Unable to open file");
    let mut parsed_queries: Vec<Vec<i32>> = Vec::new();

    // opens the file with a BufReaderMl which is similar to a regular BufReader
    // works off each line of the file data seperatly
    for line in buf_reader {
        let l = line.expect("Unable to read line");

        // takes a line of the file and parses the i32 values
        let res: Vec<i32> = l.as_ref().split_whitespace().into_iter()
        .map(|elem| elem.to_string().parse::<i32>()
            .unwrap_or_else(|_| panic!("Unable to parse {:?} into an i32 value while trying to parse the querie file at {:?}.\nCheck the help page with \"-h\" or \"--help\" for further information.\n", elem, path))
        ).collect();
        parsed_queries.push(res);
    }
    parsed_queries
}

/// Parses i32 values out of a Vector of Strings
///
/// # Example
/// ```
/// use ddnnf_lib::parser::parse_features;
///
/// let valid_in = Some(vec![String::from("3"), String::from("-12")]);
/// assert_eq!(vec![3,-12], parse_features(valid_in));
/// ```
/// # Panic
///
/// Panics for String values that can not be parsed into an i32 and for None
pub fn parse_features(input: Option<Vec<String>>) -> Vec<i32> {
    input.unwrap().into_iter()
    .map(|elem| elem.parse::<i32>()
        .unwrap_or_else(|_| panic!("Unable to parse {:?} into an i32 value.\nCheck the help page with \"-h\" or \"--help\" for further information.\n", elem))
    ).collect()
}
