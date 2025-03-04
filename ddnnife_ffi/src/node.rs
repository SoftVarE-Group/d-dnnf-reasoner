use num::BigInt;

type Node = ddnnife::Node;
type NodeType = ddnnife::NodeType;

#[uniffi::remote(Enum)]
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

#[uniffi::remote(Record)]
pub struct Node {
    /// The cardinality of the node for the cardinality of a feature model
    pub count: BigInt,
    /// The cardinality during the different queries
    pub temp: BigInt,
    /// The cardinality during the different queries
    pub partial_derivative: BigInt,
    /// the different kinds of nodes with its additional fields
    pub ntype: ddnnife::NodeType,
}
