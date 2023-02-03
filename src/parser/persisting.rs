use std::{io::{LineWriter, Write}, fs::File};

use crate::{Ddnnf, Node, NodeType};

/// Takes a d-DNNF and writes the string representation into a file with the provided name
pub fn write_ddnnf(ddnnf: &mut Ddnnf, path_out: &str) -> std::io::Result<()> {    
    let file = File::create(path_out)?;
    let mut file = LineWriter::with_capacity(1000, file);
    
    file.write_all(format!("nnf {} {} {}\n", ddnnf.nodes.len(), 0, ddnnf.number_of_variables).as_bytes())?;
    for node in &ddnnf.nodes {
        file.write_all(deconstruct_node(node).as_bytes())?;
    }

    Ok(())
}

/// Takes a node of the ddnnf which is in the our representation of a flatted DAG
/// and transforms it into the corresponding String.
/// We use an adjusted version of the c2d format: Or nodes can have multiple children, there are no decision nodes
fn deconstruct_node(node: &Node) -> String {
    let mut str = match &node.ntype {
        NodeType::And { children } => deconstruct_children(String::from("A "), children),
        NodeType::Or { children } => deconstruct_children(String::from("O 0 "), children),
        NodeType::Literal { literal } => format!("L {}", literal),
        NodeType::True => String::from("A 0"),
        NodeType::False => String::from("O 0 0"),
    };
    str.push('\n');
    str
}

#[inline]
fn deconstruct_children(mut str: String, children: &Vec<usize>) -> String {
    str.push_str(&children.len().to_string());
    str.push(' ');

    for n in 0..children.len() {
        str.push_str(&children[n].to_string());

        if n != children.len() - 1 {
            str.push(' ');
        }
    }
    str
}