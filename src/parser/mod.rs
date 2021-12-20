pub mod lexer;
use lexer::{
    TId,
    lex_line
};

use std::{
    collections::{HashSet, HashMap}
};

use rug::{Complete, Integer};

pub mod bufreader_for_big_files;
use bufreader_for_big_files::*;

use crate::data_structure::{
    *,
    NodeType::*
};

/// Parses a ddnnf, referenced by the file path. The file gets parsed and we create
/// the corresponding data structure.
/// 
/// # Examples
/// 
/// ```
/// mod parser;
/// use crate::parser::build_ddnnf_tree;
/// 
/// let file_path = "example_input/automotive.dimacs.nnf"
/// 
/// let ddnnf: Ddnnf = build_ddnnf_tree(file_path);
/// ```
/// 
/// # Panics
/// 
/// The function panics for an invalid file path.

#[inline]
pub fn build_ddnnf_tree(path: &str) -> Ddnnf{
    let mut buf_reader = BufReaderMl::open(path).expect("Unable to open file");
    
    let first_line = buf_reader.next().expect("Unable to read line").unwrap();
    let header = lex_line(first_line.trim()).unwrap().1.1;

    let mut parsed_nodes: Vec<Node> = Vec::with_capacity(header[0]);

    // opens the file with a BufReaderMl which is similar to a regular BufReader
    // works off each line of the file data seperatly
    for line in buf_reader {
        let line = line.expect("Unable to read line");

        let next = match lex_line(line.trim()).unwrap().1{ // extract the parsed Token 
            (TId::PositiveLiteral, v) => Node::new_literal(v[0] as i32),
            (TId::NegativeLiteral, v) => Node::new_literal(-(v[0] as i32)),
            (TId::And, v) => Node::new_and(And, v.clone(), calc_overall_and_count(&mut parsed_nodes, &v)),
            (TId::Or, v) => {
                let v_num = v.clone().remove(0) as i32;
                let mut children = v.clone();
                children.remove(0);
                Node::new_or(Or, v_num, children, calc_overall_or_count(&mut parsed_nodes, &v))
            },
            (TId::True, _) => Node::new_bool(True),
            (TId::False, _) => Node::new_bool(False),
            _ => panic!("Tried to parse the header of the .nnf at the wrong time")
        };

        parsed_nodes.push(next);
    }

    Ddnnf::new(parsed_nodes, HashMap::new(), header[2] as u32, header[0])
}

#[inline]
// adds the parent connection as well as the hashmpa for the literals and their corresponding position in the vector
pub fn build_ddnnf_tree_with_extras(path: &str) -> Ddnnf{
    let mut buf_reader = BufReaderMl::open(path).expect("Unable to open file");
    
    let first_line = buf_reader.next().expect("Unable to read line").unwrap();
    let header = lex_line(first_line.trim()).unwrap().1.1;

    let mut parsed_nodes: Vec<Node> = Vec::with_capacity(header[0]);

    let mut literals: HashMap<i32, usize> = HashMap::new();

    // opens the file with a BufReaderMl which is similar to a regular BufReader
    // works off each line of the file data seperatly
    for line in buf_reader {
        let line = line.expect("Unable to read line");

        let next: Node = match lex_line(line.as_ref()).unwrap().1{ // extract the parsed Token 
            (TId::PositiveLiteral, v) => Node::new_literal(v[0] as i32),
            (TId::NegativeLiteral, v) => Node::new_literal(-(v[0] as i32)),
            (TId::And, v) => Node::new_and(And, v.clone(), calc_overall_and_count(&mut parsed_nodes, &v)),
            (TId::Or, v) => {
                let v_num = v.clone().remove(0) as i32;
                let mut children = v.clone();
                children.remove(0);
                Node::new_or(Or, v_num, children, calc_overall_or_count(&mut parsed_nodes, &v))
            },
            (TId::True, _) => Node::new_bool(True),
            (TId::False, _) => Node::new_bool(False),
            _ => panic!("Tried to parse the header of the .nnf at the wrong time")
        };

        // fill the parent node pointer
        if next.node_type == And || next.node_type == Or{
            let next_childs: Vec<usize> = next.children.clone().unwrap();

            let next_indize: usize = parsed_nodes.len();

            for i in next_childs{
                parsed_nodes[i].parents.push(next_indize);
            }
        }

        // fill the HashMap with the literals
        if next.node_type == Literal {
            if literals.contains_key(&next.var_number.unwrap()) {
                panic!("Literal {:?} occured at least twice. This violates the standard!", next.var_number.unwrap());
            }
            literals.insert(next.var_number.unwrap(), parsed_nodes.len());
        }

        parsed_nodes.push(next);
    }

    Ddnnf::new(parsed_nodes, literals, header[2] as u32, header[0])
}

// multiplies the overall_count of all child Nodes of an And Node
#[inline]
fn calc_overall_and_count(nodes: &mut Vec<Node>, indizes: &[usize]) -> Integer{
    Integer::product(indizes.iter().map(|indize| &nodes[*indize].count)).complete()
}

// adds up the overall_count of all child Nodes of an And Node
#[inline]
fn calc_overall_or_count(nodes: &mut Vec<Node>, indizes: &[usize]) -> Integer{ 
    Integer::from(&nodes[indizes[1]].count + &nodes[indizes[2]].count)
}

pub fn parse_features(input: Option<Vec<String>>) -> Vec<i32>{
    input.unwrap().into_iter()
    .map(|elem| elem.to_string().parse::<i32>()
        .expect(&format!("Unable to parse {:?} into an i32 value.\nCheck the help page with \"-h\" or \"--help\" for further information.\n", elem))
    ).collect()
}

pub fn parse_queries_file(path: &str) -> Vec<Vec<i32>>{
    let buf_reader = BufReaderMl::open(path).expect("Unable to open file");
    let mut parsed_queries: Vec<Vec<i32>> = Vec::new();

    // opens the file with a BufReaderMl which is similar to a regular BufReader
    // works off each line of the file data seperatly
    for line in buf_reader {
        let l = line.expect("Unable to read line");

        // takes a line of the file and parses the i32 values
        let res: Vec<i32> = l.as_ref().split_whitespace().into_iter()
        .map(|elem| elem.to_string().parse::<i32>()
            .expect(&format!("Unable to parse {:?} into an i32 value while trying to parse the querie file at {:?}.\nCheck the help page with \"-h\" or \"--help\" for further information.\n", elem, path))
        ).collect();
        parsed_queries.push(res);
    }
    parsed_queries
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn token_parsing_test(){
        let mut ddnnf: Ddnnf = build_ddnnf_tree("example_input/test.dimacs.nnf");

        assert_eq!(ddnnf.number_of_variables, 5);

        let or_node = ddnnf.nodes.pop().unwrap();
        let or_node_childs = or_node.children.unwrap();

        assert_eq!(or_node_childs.len().clone(), 2 as usize);
        assert_eq!(or_node.node_type, NodeType::Or);
        assert_eq!(or_node.count, Integer::from(5 as u64));

        assert_eq!(ddnnf.nodes[or_node_childs[0]].node_type, NodeType::And);
        assert_eq!(ddnnf.nodes[or_node_childs[1]].node_type, NodeType::Literal);

        let and_node_childs = ddnnf.nodes.pop().unwrap().children.unwrap();
        assert_eq!(and_node_childs.len(), 4 as usize);
        assert_eq!(ddnnf.nodes[and_node_childs[0]].node_type, NodeType::Literal);
        assert_eq!(ddnnf.nodes[and_node_childs[1]].node_type, NodeType::Or);
    }
}