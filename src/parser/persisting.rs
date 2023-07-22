//! Functions to persist dDNNFs. Either in c2d format or as mermaid markdown graph.

use std::{io::{LineWriter, Write}, fs::File};

use petgraph::stable_graph::NodeIndex;
use rug::Assign;

use crate::{Ddnnf, Node, NodeType};

/// Takes a d-DNNF and writes the string representation into a file with the provided name
pub fn write_ddnnf(ddnnf: &Ddnnf, path_out: &str) -> std::io::Result<()> {    
    let file = File::create(path_out)?;
    let mut lw = LineWriter::with_capacity(1000, file);
    
    lw.write_all(format!("nnf {} {} {}\n", ddnnf.nodes.len(), 0, ddnnf.number_of_variables).as_bytes())?;
    for node in &ddnnf.nodes {
        lw.write_all(deconstruct_node(node).as_bytes())?;
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
        NodeType::Literal { literal } => format!("L {literal}"),
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

/// Takes a Ddnnf, transforms it into a corresponding markdown mermaid representation,
/// and saves it into the provided file name.
/// 
/// We als add a legend that describes the mermaidified nodes
pub fn write_as_mermaid_md(ddnnf: &Ddnnf, features: &[i32], path_out: &str, alternative_start: Option<(NodeIndex, i32)>) -> std::io::Result<()> { 
    let mut ddnnf = ddnnf.clone(); // work on clone because we might manipulate it
    for node in ddnnf.nodes.iter_mut() {
        node.temp.assign(&node.count);
    }
    
    ddnnf.operate_on_partial_config_marker(features, Ddnnf::calc_count_marked_node);
    
    let file = File::create(path_out)?;
    let mut lw = LineWriter::with_capacity(1000, file);
    
    let config = format!("```mermaid\n\t\
    graph TD
        subgraph pad1 [ ]
            subgraph pad2 [ ]
                subgraph legend[Legend]
                    nodes(\"<font color=white> Node Type <font color=cyan> \
                    Node Number <font color=greeny> Count <font color=red> Temp Count \
                    <font color=orange> Query {features:?}\")
                    style legend fill:none, stroke:none
                end
                style pad2 fill:none, stroke:none
            end
            style pad1 fill:none, stroke:none
        end
        classDef marked stroke:#d90000, stroke-width:4px\n\n");
    lw.write_all(config.as_bytes()).unwrap();
    let marking = ddnnf.get_marked_nodes_clone(features);
    match alternative_start {
        Some((start, depth)) => {
            ddnnf.inter_graph = ddnnf.inter_graph.get_partial_graph_til_depth(start, depth);
            ddnnf.rebuild();
        },
        None => (),
    };
    lw.write_all(mermaidify_nodes(&mut ddnnf, &marking).as_bytes())?;
    lw.write_all(b"```").unwrap();

    Ok(())
}

/// Adds the nodes its children to the mermaid graph
fn mermaidify_nodes(ddnnf: &Ddnnf, marking: &[usize]) -> String {
    let mut result = String::new();

    for (position, node) in ddnnf.nodes.iter().enumerate().rev() {
        result = format!("{}{}", result, match &node.ntype {
            NodeType::And { children } | NodeType::Or { children } => {
                let mut mm_node = format!("\t\t{}{}", mermaidify_type(ddnnf, position), marking_insert(marking, position));

                let mut children_series = children.clone();
                children_series.sort_unstable(); // sort by index -> we force printing the dfs

                if !children_series.is_empty() {
                    mm_node.push_str(" --> ");
                    for (i, &child) in children_series.iter().enumerate() {
                        if ddnnf.nodes[child].ntype == NodeType::True {
                            continue;
                        }
                        mm_node.push_str(&child.to_string());
                        if i != children_series.len() - 1 {
                            mm_node.push_str(" & ");
                        } else {
                            mm_node.push_str(";\n");
                        }
                    }
                } else {
                    mm_node.push_str(";\n");
                }
                mm_node
            },
            NodeType::Literal { literal: _ } | NodeType::False => { 
                format!("\t\t{}{};\n", mermaidify_type(ddnnf, position), marking_insert(marking, position))
            },
            _ => String::new()
        });
    }

    result
}

/// Each node in the mermaid graph contains information about
///     1) NodeType,
///     2) Position in the flattened graph,
///     3) Count (of the model)
///     4) Current Temp Count
fn mermaidify_type(ddnnf: &Ddnnf, position: usize) -> String {
    let node = &ddnnf.nodes[position];
    let mut mm_node = format!("{}{}", position, match node.ntype {
        NodeType::And { children: _ } => String::from("(\"∧"),
        NodeType::Or { children: _ } => String::from("(\"∨"),
        NodeType::Literal { literal } => {
            if literal.is_negative() {
                format!("(\"¬L{}", literal.abs())
            } else {
                format!("(\"L{literal}")
            }
        },
        NodeType::True => String::from("(\"T"),
        NodeType::False => String::from("(\"F"),
    });

    let meta_info = format!(" <font color=cyan>{} <font color=greeny>{} <font color=red>{}\")", position, node.count, node.temp);
    mm_node.push_str(&meta_info);
    mm_node
}

fn marking_insert(marking: &[usize], position: usize) -> &str {
    if marking.binary_search(&position).is_ok() {
        ":::marked"
    } else {
        ""
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::parser::build_ddnnf;

    #[test]
    fn partial_mermaid() {
        let mut ddnnf = build_ddnnf("tests/data/vp9.cnf", Some(42));
        let root =  ddnnf.inter_graph.root;
        write_as_mermaid_md(&mut ddnnf, &[], "whole.md", None).unwrap();
        write_as_mermaid_md(&mut ddnnf, &[], "partial_root.md", Some((root, 5))).unwrap();
    }
}