use num::BigInt;
use NodeType::{And, False, Literal, Or, True};

/// Represents all types of Nodes with its different parts
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct Node {
    pub(crate) marker: bool,
    /// The cardinality of the node for the cardinality of a feature model
    pub count: BigInt,
    /// The cardinality during the different queries
    pub temp: BigInt,
    /// The cardinality during the different queries
    pub partial_derivative: BigInt,
    /// Every node excpet the root has (multiple) parent nodes
    pub(crate) parents: Vec<usize>,
    /// the different kinds of nodes with its additional fields
    pub ntype: NodeType,
}

/// The Type of the Node declares how we handle the computation for the different types of cardinalities
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    /// The cardinality of an And node is always the product of its childs
    And { children: Vec<usize> },
    /// The cardinality of an Or node is the sum of its children
    Or { children: Vec<usize> },
    /// The cardinality is one if not declared otherwise due to some query
    Literal { literal: i32 },
    /// The cardinality is one
    True,
    /// The cardinality is zero
    False,
}

impl Node {
    #[inline]
    /// Creates a new node
    fn new_node(count: BigInt, ntype: NodeType) -> Node {
        Node {
            marker: false,
            count,
            temp: BigInt::ZERO,
            partial_derivative: BigInt::ZERO,
            parents: Vec::new(),
            ntype,
        }
    }

    #[inline]
    /// Creates a new And node
    pub fn new_and(count: BigInt, children: Vec<usize>) -> Node {
        Node::new_node(count, And { children })
    }

    #[inline]
    /// Creates a new Or node
    pub fn new_or(_decision_var: u32, count: BigInt, children: Vec<usize>) -> Node {
        Node::new_node(count, Or { children })
    }

    #[inline]
    /// Creates a new Literal node
    pub fn new_literal(literal: i32) -> Node {
        Node::new_node(BigInt::from(1), Literal { literal })
    }

    #[inline]
    /// Creates either a new True or False node
    pub fn new_bool(b: bool) -> Node {
        if b {
            Node::new_node(BigInt::from(1), True)
        } else {
            Node::new_node(BigInt::ZERO, False)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn build_nodes() {
        assert_eq!(
            Node::new_and(BigInt::from(42), vec![1, 5, 10]),
            Node {
                marker: false,
                count: BigInt::from(42),
                temp: BigInt::ZERO,
                partial_derivative: BigInt::ZERO,
                parents: vec![],
                ntype: And {
                    children: vec![1, 5, 10]
                }
            }
        );
        assert_eq!(
            Node::new_node(
                BigInt::from(42),
                And {
                    children: vec![1, 5, 10]
                }
            ),
            Node {
                marker: false,
                count: BigInt::from(42),
                temp: BigInt::ZERO,
                partial_derivative: BigInt::ZERO,
                parents: vec![],
                ntype: And {
                    children: vec![1, 5, 10]
                }
            }
        );
        assert_eq!(
            Node::new_or(42, BigInt::from(42), vec![1, 5, 10]),
            Node {
                marker: false,
                count: BigInt::from(42),
                temp: BigInt::ZERO,
                partial_derivative: BigInt::ZERO,
                parents: vec![],
                ntype: Or {
                    children: vec![1, 5, 10]
                }
            }
        );
        assert_eq!(
            Node::new_literal(42),
            Node {
                marker: false,
                count: BigInt::from(1),
                temp: BigInt::ZERO,
                partial_derivative: BigInt::ZERO,
                parents: vec![],
                ntype: Literal { literal: 42 }
            }
        );
        assert_eq!(
            Node::new_bool(true),
            Node {
                marker: false,
                count: BigInt::from(1),
                temp: BigInt::ZERO,
                partial_derivative: BigInt::ZERO,
                parents: vec![],
                ntype: True
            }
        );
        assert_eq!(
            Node::new_bool(false),
            Node {
                marker: false,
                count: BigInt::ZERO,
                temp: BigInt::ZERO,
                partial_derivative: BigInt::ZERO,
                parents: vec![],
                ntype: False
            }
        );
    }
}
