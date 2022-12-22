#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Represents all types of Nodes with its different parts
pub struct Node {
    pub(crate) marker: bool,
    /// The cardinality of the node for the cardinality of a feature model
    pub count: Integer,
    /// The cardinality during the different queries
    pub temp: Integer,
    /// Every node excpet the root has (multiple) parent nodes
    pub(crate) parents: Vec<usize>,
    /// the different kinds of nodes with its additional fields
    pub ntype: NodeType,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// The Type of the Node declares how we handle the computation for the different types of cardinalities
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

use NodeType::{And, False, Literal, Or, True};
use rug::Integer;

impl Node {
    #[inline]
    /// Creates a new node
    fn new_node(count: Integer, ntype: NodeType) -> Node {
        Node {
            marker: false,
            count,
            temp: Integer::ZERO,
            parents: Vec::new(),
            ntype
        }
    }

    #[inline]
    /// Creates a new And node
    pub fn new_and(count: Integer, children: Vec<usize>) -> Node {
        Node::new_node(count, And { children })
    }

    #[inline]
    /// Creates a new Or node
    pub fn new_or(
        _decision_var: u32,
        count: Integer,
        children: Vec<usize>,
    ) -> Node {
        Node::new_node(count, Or { children })
    }

    #[inline]
    /// Creates a new Literal node
    pub fn new_literal(literal: i32) -> Node {
        Node::new_node(Integer::from(1), Literal { literal })
    }

    #[inline]
    /// Creates either a new True or False node
    pub fn new_bool(b: bool) -> Node {
        if b {
            Node::new_node(Integer::from(1), True)
        } else {
            Node::new_node(Integer::ZERO, False)
        }
    }
}