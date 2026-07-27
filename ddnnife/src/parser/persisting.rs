use crate::{Ddnnf, DdnnfKind, Node, NodeType};

impl Ddnnf {
    /// Transforms the d-DNNF into a corresponding markdown mermaid representation.
    ///
    /// Adds a legend describing the mermaidified nodes.
    pub fn mermaidify(&mut self, features: &[i32]) -> String {
        match self.kind {
            DdnnfKind::Tautology => {
                return "0(\"⊤ <font color=cyan>0 <font color=greeny>1 <font color=red>1\");\n"
                    .into();
            }
            DdnnfKind::Contradiction => {
                return "0(\"⊥ <font color=cyan>0 <font color=greeny>0 <font color=red>0\");\n"
                    .into();
            }
            DdnnfKind::NonTrivial => {}
        }

        for node in self.nodes.iter_mut() {
            node.temp.clone_from(&node.count);
        }

        self.operate_on_partial_config_marker(features, Ddnnf::calc_count_marked_node);
        let marking = self.get_marked_nodes_clone(features);

        let mut result = format!(
            r#"```mermaid
graph TD
subgraph pad1 [ ]
    subgraph pad2 [ ]
        subgraph legend[Legend]
            nodes("<font color=white> Node Type <font color=cyan> Node Number <font color=greeny> Count <font color=red> Temp Count <font color=orange> Query {features:?}")
            style legend fill:none, stroke:none
        end
        style pad2 fill:none, stroke:none
    end
    style pad1 fill:none, stroke:none
end
classDef marked stroke:#d90000, stroke-width:4px
"#,
        );

        for (position, node) in self.nodes.iter().enumerate().rev() {
            result.push_str(
                match &node.ntype {
                    NodeType::And { children } | NodeType::Or { children } => {
                        let mut mm_node = format!(
                            "{}{} --> ",
                            node.mermaidify(position),
                            marking_insert(&marking, position)
                        );

                        let mut children_series = children.clone();
                        children_series.sort_unstable();

                        if !children_series.is_empty() {
                            for (i, &child) in children_series.iter().enumerate() {
                                mm_node.push_str(&child.to_string());
                                if i != children_series.len() - 1 {
                                    mm_node.push_str(" & ");
                                }
                            }
                        }

                        mm_node
                    }
                    NodeType::Literal { literal: _ } => format!(
                        "{}{}",
                        node.mermaidify(position),
                        marking_insert(&marking, position)
                    ),
                }
                .as_ref(),
            );

            result.push_str(";\n");
        }

        result.push_str("```\n");
        result
    }
}

impl Node {
    /// Generates the mermaid representation of this node.
    ///
    /// Each node in the mermaid graph contains information about
    ///     1) NodeType,
    ///     2) Position in the flattened graph,
    ///     3) Count (of the model)
    ///     4) Current Temp Count
    fn mermaidify(&self, position: usize) -> String {
        format!(
            "{}(\"{} <font color=cyan>{} <font color=greeny>{} <font color=red>{}\")",
            position,
            match self.ntype {
                NodeType::And { children: _ } => String::from("∧"),
                NodeType::Or { children: _ } => String::from("∨"),
                NodeType::Literal { literal } => {
                    if literal.is_negative() {
                        format!("¬L{}", literal.abs())
                    } else {
                        format!("L{literal}")
                    }
                }
            },
            position,
            self.count,
            self.temp
        )
    }
}

fn marking_insert(marking: &[usize], position: usize) -> &str {
    if marking.binary_search(&position).is_ok() {
        ":::marked"
    } else {
        ""
    }
}
