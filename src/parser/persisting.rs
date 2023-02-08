use std::{io::{LineWriter, Write}, fs::File, cmp::max};

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

/// Takes a Ddnnf, transforms it into a corresponding markdown mermaid representation
/// , and saves it into the provided file name.
/// 
/// We als add a legend that describes the mermaidified nodes
pub fn write_as_mermaid_md(ddnnf: &Ddnnf, path_out: &str) -> std::io::Result<()> { 
    let file = File::create(path_out)?;
    let mut lw = LineWriter::with_capacity(1000, file);
    
    let config = String::from("```mermaid\n\t\
        graph TD\n\t\
        %%{init:{'flowchart':{'nodeSpacing': 50, 'rankSpacing': 100}}}%%\n\t\t\
            subgraph pad1 [ ]\n\t\t\
                subgraph pad2 [ ]\n\t\t\
                    subgraph legend[Legend]\n\t\t\t\
                        nodes(\"<font color=white> Node Type <font color=cyan> Node Number <font color=greeny> Count <font color=red> Temp Count\")\n\t\t\t\t\
                    end\n\t\t\t\
                    style legend stroke:orange,stroke-width:3px,color:orange,fill:none,height:90px,width:365px,x:20px\n\t\t\t\
                end\n\t\t\
                style pad2 fill:none, stroke:none\n\t\t\
                style pad1 fill:none, stroke:none\n\t\t\
            end\n");
    lw.write_all(config.as_bytes()).unwrap();
    lw.write_all(mermaidify_node(ddnnf, ddnnf.number_of_nodes-1).as_bytes())?;
    lw.write_all(b"```").unwrap();

    Ok(())
}

/// Adds the nodes its children to the mermaid graph
fn mermaidify_node(ddnnf: &Ddnnf, position: usize) -> String {
    match &ddnnf.nodes[position].ntype {
        NodeType::And { children } | NodeType::Or { children } => {
            let mut res = String::new();
            let mm_node = format!("\t\t{} --> ", mermaidify_type(&ddnnf, position));

            let mut children_series = children.clone();
            children_series.sort_by(|c1, c2| compute_depth(ddnnf, *c1).cmp(&&compute_depth(ddnnf, *c2)));

            if children_series.len() != 0 {
                res.push_str(&mm_node);

                for child in children_series {
                    if ddnnf.nodes[child].ntype == NodeType::True {
                        continue;
                    }
                    let con = format!("{} & ", mermaidify_type(&ddnnf, child));
                    res.push_str(&con);
                }

                res.drain(res.len()-3..);
                res.push_str(";\n")
            }

            for child in children {
                res.push_str(&mermaidify_node(ddnnf, *child));
            }
            res
        },
        _ => String::new(),
    }
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
                format!("(\"L{}", literal)
            }
        },
        NodeType::True => String::from("(\"T"),
        NodeType::False => String::from("(\"F"),
    });

    let meta_info = format!(" <font color=cyan>{} <font color=greeny>{} <font color=red>{}\")", position, node.count, node.temp);
    mm_node.push_str(&meta_info);
    mm_node
}

/// Computes the depth of any node in the current graph.
/// Here, the depth is the length of the deepest path starting from position
fn compute_depth(ddnnf: &Ddnnf, position: usize) -> usize {
    match &ddnnf.nodes[position].ntype {
        NodeType::And { children } | NodeType::Or { children } => {
            children.into_iter().fold(0, |acc, &x| max(acc+1, compute_depth(ddnnf, x) + 1))
        },
        NodeType::True => 0,
        _ => 1
    }
}