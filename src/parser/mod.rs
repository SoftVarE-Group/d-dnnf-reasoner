pub mod lexer;
use lexer::{lex_line, TId};

use std::collections::HashMap;

use rug::{Complete, Integer};

pub mod bufreader_for_big_files;
use bufreader_for_big_files::BufReaderMl;

use crate::data_structure::{
    Ddnnf, Node,
    NodeType::{And, False, Literal, Or, True},
};

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
