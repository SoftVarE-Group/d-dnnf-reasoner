use std::{
    cmp::max,
    collections::BTreeSet,
    fs::File,
    io::{LineWriter, Write},
};

use crate::{Ddnnf, Node, NodeType};

use crate::util::format_vec;

/// Takes a CNF and writes the string representation into a file with the provided name
pub(crate) fn write_cnf_to_file(
    clauses: &BTreeSet<BTreeSet<i32>>,
    total_features: u32,
    path_out: &str,
) -> std::io::Result<()> {
    let file = File::create(path_out)?;
    let mut lw = LineWriter::with_capacity(1000, file);

    lw.write_all(format!("p cnf {} {}\n", total_features, clauses.len()).as_bytes())?;
    for clause in clauses {
        lw.write_all(format!("{} 0\n", format_vec(clause.iter())).as_bytes())?;
    }

    Ok(())
}

/// Takes a d-DNNF and writes the string representation into a file with the provided name
pub fn write_ddnnf_to_file(ddnnf: &Ddnnf, path_out: &str) -> std::io::Result<()> {
    let file = File::create(path_out)?;
    let mut lw = LineWriter::with_capacity(1000, file);

    lw.write_all(
        format!(
            "nnf {} {} {}\n",
            ddnnf.nodes.len(),
            0,
            ddnnf.number_of_variables
        )
        .as_bytes(),
    )?;
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
        NodeType::Literal { literal } => format!("L {}", literal),
        NodeType::True => String::from("A 0"),
        NodeType::False => String::from("O 0 0"),
    };
    str.push('\n');
    str
}

#[inline]
fn deconstruct_children(mut str: String, children: &[usize]) -> String {
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
pub fn write_as_mermaid_md(
    ddnnf: &mut Ddnnf,
    features: &[i32],
    path_out: &str,
) -> std::io::Result<()> {
    for node in ddnnf.nodes.iter_mut() {
        node.temp.clone_from(&node.count);
    }

    ddnnf.operate_on_partial_config_marker(features, Ddnnf::calc_count_marked_node);

    let file = File::create(path_out)?;
    let mut lw = LineWriter::with_capacity(1000, file);

    let config = format!(
        "```mermaid\n\t\
    graph TD
        subgraph pad1 [ ]
            subgraph pad2 [ ]
                subgraph legend[Legend]
                    nodes(\"<font color=white> Node Type <font color=cyan> \
                    Node Number <font color=greeny> Count <font color=red> Temp Count \
                    <font color=orange> Query {:?}\")
                    style legend fill:none, stroke:none
                end
                style pad2 fill:none, stroke:none
            end
            style pad1 fill:none, stroke:none
        end
        classDef marked stroke:#d90000, stroke-width:4px\n\n",
        features
    );
    lw.write_all(config.as_bytes()).unwrap();
    let marking = ddnnf.get_marked_nodes_clone(features);
    lw.write_all(mermaidify_nodes(ddnnf, &marking).as_bytes())?;
    lw.write_all(b"```").unwrap();

    Ok(())
}

/// Adds the nodes its children to the mermaid graph
fn mermaidify_nodes(ddnnf: &Ddnnf, marking: &[usize]) -> String {
    let mut result = String::new();

    for (position, node) in ddnnf.nodes.iter().enumerate().rev() {
        result = format!(
            "{}{}",
            result,
            match &node.ntype {
                NodeType::And { children } | NodeType::Or { children } => {
                    let mut mm_node = format!(
                        "\t\t{}{} --> ",
                        mermaidify_type(ddnnf, position),
                        marking_insert(marking, position)
                    );

                    let mut children_series = children.clone();
                    children_series.sort_by_key(|c1| compute_depth(ddnnf, *c1));

                    if !children_series.is_empty() {
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
                    }
                    mm_node
                }
                NodeType::Literal { literal: _ } | NodeType::False => {
                    format!(
                        "\t\t{}{};\n",
                        mermaidify_type(ddnnf, position),
                        marking_insert(marking, position)
                    )
                }
                _ => String::new(),
            }
        );
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
    let mut mm_node = format!(
        "{}{}",
        position,
        match node.ntype {
            NodeType::And { children: _ } => String::from("(\"∧"),
            NodeType::Or { children: _ } => String::from("(\"∨"),
            NodeType::Literal { literal } => {
                if literal.is_negative() {
                    format!("(\"¬L{}", literal.abs())
                } else {
                    format!("(\"L{}", literal)
                }
            }
            NodeType::True => String::from("(\"T"),
            NodeType::False => String::from("(\"F"),
        }
    );

    let meta_info = format!(
        " <font color=cyan>{} <font color=greeny>{} <font color=red>{}\")",
        position, node.count, node.temp
    );
    mm_node.push_str(&meta_info);
    mm_node
}

/// Computes the depth of any node in the current graph.
/// Here, the depth is the length of the deepest path starting from position
fn compute_depth(ddnnf: &Ddnnf, position: usize) -> usize {
    match &ddnnf.nodes[position].ntype {
        NodeType::And { children } | NodeType::Or { children } => children
            .iter()
            .fold(0, |acc, &x| max(acc + 1, compute_depth(ddnnf, x) + 1)),
        NodeType::True => 0,
        _ => 1,
    }
}

fn marking_insert(marking: &[usize], position: usize) -> &str {
    if marking.binary_search(&position).is_ok() {
        ":::marked"
    } else {
        ""
    }
}
