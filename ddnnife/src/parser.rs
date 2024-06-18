pub mod c2d_lexer;
use c2d_lexer::{lex_line_c2d, C2DToken, TId};

pub mod d4_lexer;
use d4_lexer::{lex_line_d4, D4Token};

pub mod from_cnf;
use from_cnf::{check_for_cnf_header, CNFToken};

pub mod persisting;

use core::panic;
use num::BigInt;
use std::{
    cell::RefCell,
    cmp::max,
    collections::{BTreeSet, HashMap, HashSet},
    ffi::OsStr,
    fs::{self, File},
    io::{BufRead, BufReader},
    path::Path,
    process,
    rc::Rc,
};

use crate::ddnnf::{node::Node, node::NodeType, Ddnnf};

use petgraph::{
    graph::{EdgeIndex, NodeIndex},
    stable_graph::StableGraph,
    visit::DfsPostOrder,
    Direction::{Incoming, Outgoing},
};

/// Parses a ddnnf, referenced by the file path. The file gets parsed and we create
/// the corresponding data structure.
///
/// # Examples
///
/// ```
/// use ddnnife::parser;
/// use ddnnife::Ddnnf;
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
    let mut clauses = BTreeSet::new();
    if let Some(extension) = Path::new(path).extension().and_then(OsStr::to_str) {
        if extension == "dimacs" || extension == "cnf" {
            #[cfg(feature = "d4")]
            {
                let file = open_file_savely(path);
                let lines = BufReader::new(file).lines();
                for line in lines {
                    let line = line.expect("Unable to read line");
                    match check_for_cnf_header(line.as_str()).unwrap().1 {
                        CNFToken::Header {
                            total_features: total_features_header,
                            total_clauses: _,
                        } => {
                            let ddnnf_file = ".intermediate.nnf";
                            d4_oxide::compile_ddnnf(path.to_string(), ddnnf_file.to_string());
                            path = ddnnf_file;
                            total_features = Some(total_features_header as u32);
                        }
                        CNFToken::Clause { features } => {
                            clauses.insert(features);
                        }
                        CNFToken::Comment => (),
                    }
                }
            }

            #[cfg(not(feature = "d4"))]
            {
                panic!("CNF to d-DNNF compilation is only possible when including d4.");
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

    if clauses.is_empty() {
        distribute_building(lines, total_features, None)
    } else {
        distribute_building(lines, total_features, Some(clauses))
    }
}

/// Chooses, depending on the first read line, which building implmentation to choose.
/// Either the first line is a header and therefore the c2d format or total_features
/// is supplied and its the d4 format.
#[inline]
pub fn distribute_building(
    lines: Vec<String>,
    total_features: Option<u32>,
    clauses: Option<BTreeSet<BTreeSet<i32>>>,
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
        )) => build_c2d_ddnnf(lines, variables as u32, clauses),
        Ok(_) | Err(_) => {
            // tried to parse the c2d standard, but failes
            match total_features {
                Some(o) => {
                    // we try to parse the d4 standard
                    build_d4_ddnnf(lines, Some(o), clauses)
                }
                None => {
                    // unknown standard or combination -> we assume d4 and choose total_features
                    // Bold, Yellow, Foreground Color (see https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797)
                    println!("\x1b[1;38;5;226mWARNING: The first line of the file isn't a header and the option 'total_features' is not set. \
                        Hence, we can't determine the number of variables and as a result, we might not be able to construct a valid ddnnf. \
                        Nonetheless, we build a ddnnf with our limited information, but we discourage using ddnnife in this manner.\n\x1b[0m"
                    );
                    build_d4_ddnnf(lines, None, clauses)
                }
            }
        }
    }
}

/// Parses a ddnnf, referenced by the file path.
/// This function uses C2DTokens which specify a d-DNNF in c2d format.
/// The file gets parsed and we create the corresponding data structure.
#[inline]
fn build_c2d_ddnnf(
    lines: Vec<String>,
    variables: u32,
    clauses: Option<BTreeSet<BTreeSet<i32>>>,
) -> Ddnnf {
    use C2DToken::*;

    let mut parsed_nodes: Vec<Node> = Vec::with_capacity(lines.len());

    let mut literals: HashMap<i32, usize> = HashMap::new();
    let mut true_nodes = Vec::new();

    // opens the file with a BufReader and
    // works off each line of the file data seperatly
    // skip the first line, because we already looked at the header
    for line in lines.into_iter().skip(1) {
        let next: Node = match lex_line_c2d(line.as_ref()).unwrap().1 {
            And { children } => {
                Node::new_and(calc_and_count(&mut parsed_nodes, &children), children)
            }
            Or { decision, children } => Node::new_or(
                decision,
                calc_or_count(&mut parsed_nodes, &children),
                children,
            ),
            Literal { feature } => Node::new_literal(feature),
            True => Node::new_bool(true),
            False => Node::new_bool(false),
            _ => panic!("Tried to parse the header of the .nnf at the wrong time"),
        };

        // fill the parent node pointer, save literals
        match &next.ntype {
            NodeType::And { children } | NodeType::Or { children } => {
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

    Ddnnf::new(parsed_nodes, literals, true_nodes, variables, clauses)
}

/// Parses a ddnnf, referenced by the file path.
/// This function uses D4Tokens which specify a d-DNNF in d4 format.
/// The file gets parsed and we create the corresponding data structure.
#[inline]
fn build_d4_ddnnf(
    lines: Vec<String>,
    total_features_opt: Option<u32>,
    clauses: Option<BTreeSet<BTreeSet<i32>>>,
) -> Ddnnf {
    let mut ddnnf_graph = StableGraph::<TId, ()>::new();

    let mut total_features = total_features_opt.unwrap_or(0);
    let literal_occurences: Rc<RefCell<Vec<bool>>> =
        Rc::new(RefCell::new(vec![
            false;
            max(100_000, total_features as usize)
        ]));

    let mut indices: Vec<NodeIndex> = Vec::new();

    // With the help of the literals node state, we can add the required nodes
    // for the balancing of the or nodes to archieve smoothness
    let nx_literals: Rc<RefCell<HashMap<NodeIndex, i32>>> = Rc::new(RefCell::new(HashMap::new()));
    let literals_nx: Rc<RefCell<HashMap<i32, NodeIndex>>> = Rc::new(RefCell::new(HashMap::new()));

    let get_literal_indices =
        |ddnnf_graph: &mut StableGraph<TId, ()>, literals: Vec<i32>| -> Vec<NodeIndex> {
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
    let resolve_weighted_edge = |ddnnf_graph: &mut StableGraph<TId, ()>,
                                 from: NodeIndex,
                                 to: NodeIndex,
                                 edge: EdgeIndex,
                                 weights: Vec<i32>| {
        let and_node = ddnnf_graph.add_node(TId::And);
        let literal_nodes = get_literal_indices(ddnnf_graph, weights);

        ddnnf_graph.remove_edge(edge);

        ddnnf_graph.add_edge(from, and_node, ());
        for node in literal_nodes {
            ddnnf_graph.add_edge(and_node, node, ());
        }
        ddnnf_graph.add_edge(and_node, to, ());
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

    let add_literal_node =
        |ddnnf_graph: &mut StableGraph<TId, ()>, f_u32: u32, attach: NodeIndex| {
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
        |ddnnf_graph: &mut StableGraph<TId, ()>,
         from: NodeIndex,
         children: Vec<(NodeIndex, HashSet<u32>)>| {
            for child in children {
                let and_node = ddnnf_graph.add_node(TId::And);

                // place the newly created and node between the or node and its child
                ddnnf_graph.remove_edge(ddnnf_graph.find_edge(from, child.0).unwrap());
                ddnnf_graph.add_edge(from, and_node, ());
                ddnnf_graph.add_edge(and_node, child.0, ());

                for literal in child.1 {
                    add_literal_node(ddnnf_graph, literal, and_node);
                }
            }
        };

    // add a new root which hold the unmentioned variables within the total_features range
    let root = ddnnf_graph.add_node(TId::And);
    ddnnf_graph.add_edge(root, NodeIndex::new(0), ());

    // add literals that are not mentioned in the ddnnf to the new root node
    for i in 1..=total_features {
        if !literal_occurences.borrow()[i as usize] {
            add_literal_node(&mut ddnnf_graph, i, root);
        }
    }

    // Starting from an initial AND node, we delete all parent AND nodes.
    // We can do this because the start node has a FALSE node as children. Hence, it count is 0!
    let delete_parent_and_chain = |ddnnf_graph: &mut StableGraph<TId, ()>, start: NodeIndex| {
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
                            TId::And => ddnnf_graph.remove_edge(n_edge).unwrap(),
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
                            TId::Or => ddnnf_graph.remove_edge(n_edge).unwrap(),
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
    let mut safe: HashMap<NodeIndex, HashSet<u32>> = HashMap::new();
    let mut dfs = DfsPostOrder::new(&ddnnf_graph, root);
    while let Some(nx) = dfs.next(&ddnnf_graph) {
        // edges between going from an and node to another node do not
        // have any weights attached to them. Therefore, we can skip them
        if ddnnf_graph[nx] == TId::Or {
            let diffrences = get_literal_diff(&ddnnf_graph, &mut safe, &nx_literals.borrow(), nx);
            balance_or_children(&mut ddnnf_graph, nx, diffrences);
        }
    }

    // perform a depth first search to get the nodes ordered such
    // that child nodes are listed before their parents
    // transform that interim representation into a node vector
    dfs = DfsPostOrder::new(&ddnnf_graph, root);
    let mut nd_to_usize: HashMap<NodeIndex, usize> = HashMap::new();

    let mut parsed_nodes: Vec<Node> = Vec::with_capacity(ddnnf_graph.node_count());
    let mut literals: HashMap<i32, usize> = HashMap::new();
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
            TId::PositiveLiteral | TId::NegativeLiteral => {
                Node::new_literal(nx_lit.get(&nx).unwrap().to_owned())
            }
            TId::And => Node::new_and(calc_and_count(&mut parsed_nodes, &neighs), neighs),

            TId::Or => Node::new_or(0, calc_or_count(&mut parsed_nodes, &neighs), neighs),
            TId::True => Node::new_bool(true),
            TId::False => Node::new_bool(false),
            TId::Header => panic!("The d4 standard does not include a header!"),
        };

        match &next.ntype {
            NodeType::And { children } | NodeType::Or { children } => {
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

    Ddnnf::new(parsed_nodes, literals, true_nodes, total_features, clauses)
}

// determine the differences in literal-nodes occuring in the child nodes
fn get_literal_diff(
    di_graph: &StableGraph<TId, ()>,
    safe: &mut HashMap<NodeIndex, HashSet<u32>>,
    nx_literals: &HashMap<NodeIndex, i32>,
    or_node: NodeIndex,
) -> Vec<(NodeIndex, HashSet<u32>)> {
    let mut inter_res = Vec::new();
    let neighbors = di_graph.neighbors_directed(or_node, Outgoing);

    for neighbor in neighbors {
        inter_res.push((
            neighbor,
            get_literals(di_graph, safe, nx_literals, neighbor),
        ));
    }

    let mut res: Vec<(NodeIndex, HashSet<u32>)> = Vec::new();
    for i in 0..inter_res.len() {
        let mut val: HashSet<u32> = HashSet::new();
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
    di_graph: &StableGraph<TId, ()>,
    safe: &mut HashMap<NodeIndex, HashSet<u32>>,
    nx_literals: &HashMap<NodeIndex, i32>,
    or_child: NodeIndex,
) -> HashSet<u32> {
    let lookup = safe.get(&or_child);
    if let Some(x) = lookup {
        return x.clone();
    }

    let mut res = HashSet::new();
    use c2d_lexer::TokenIdentifier::*;
    match di_graph[or_child] {
        And | Or => {
            di_graph
                .neighbors_directed(or_child, Outgoing)
                .for_each(|n| res.extend(get_literals(di_graph, safe, nx_literals, n)));
            safe.insert(or_child, res.clone());
        }
        PositiveLiteral | NegativeLiteral => {
            res.insert(nx_literals.get(&or_child).unwrap().unsigned_abs());
            safe.insert(or_child, res.clone());
        }
        _ => (),
    }
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
/// use ddnnife::parser::parse_queries_file;
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

/// Tries to open a file.
/// If an error occurs the program prints the error and exists.
pub fn open_file_savely(path: &str) -> File {
    // opens the file with a BufReader and
    // works off each line of the file data seperatly
    match File::open(path) {
        Ok(x) => x,
        Err(err) => {
            // Bold, Red, Foreground Color (see https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797)
            eprintln!("\x1b[1;38;5;196mERROR: The following error code occured while trying to open the file \"{}\":\n{}\nAborting...", path, err);
            process::exit(1);
        }
    }
}
