use crate::{Ddnnf, DdnnfKind, NodeType};
use std::io::Write;

/// Takes a Ddnnf, transforms it into a corresponding markdown mermaid representation,
/// and saves it into the provided file name.
///
/// We als add a legend that describes the mermaidified nodes
pub fn write_as_mermaid_md(
    ddnnf: &mut Ddnnf,
    features: &[i32],
    mut output: impl Write,
) -> std::io::Result<()> {
    for node in ddnnf.nodes.iter_mut() {
        node.temp.clone_from(&node.count);
    }

    ddnnf.operate_on_partial_config_marker(features, Ddnnf::calc_count_marked_node);

    let config = format!(
        "```mermaid\n\t\
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
        classDef marked stroke:#d90000, stroke-width:4px\n\n"
    );
    output.write_all(config.as_bytes()).unwrap();
    let marking = ddnnf.get_marked_nodes_clone(features);
    output.write_all(mermaidify_nodes(ddnnf, &marking).as_bytes())?;
    output.write_all(b"```").unwrap();

    Ok(())
}

/// Adds the nodes its children to the mermaid graph
fn mermaidify_nodes(ddnnf: &Ddnnf, marking: &[usize]) -> String {
    if ddnnf.is_trivial() {
        return match ddnnf.kind {
            DdnnfKind::Tautology => {
                "0(\"⊤ <font color=cyan>0 <font color=greeny>1 <font color=red>1\");\n"
            }
            DdnnfKind::Contradiction => {
                "0(\"⊥ <font color=cyan>0 <font color=greeny>0 <font color=red>0\");\n"
            }
            DdnnfKind::NonTrivial => unreachable!(),
        }
        .into();
    }

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
                    children_series.sort_unstable();

                    if !children_series.is_empty() {
                        for (i, &child) in children_series.iter().enumerate() {
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
                NodeType::Literal { literal: _ } => {
                    format!(
                        "\t\t{}{};\n",
                        mermaidify_type(ddnnf, position),
                        marking_insert(marking, position)
                    )
                }
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
                    format!("(\"L{literal}")
                }
            }
        }
    );

    let meta_info = format!(
        " <font color=cyan>{} <font color=greeny>{} <font color=red>{}\")",
        position, node.count, node.temp
    );
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
