use crate::{Ddnnf, NodeType};
use serde::Serialize;

/// Various statistics about a d-DNNF.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct Statistics {
    /// The amount of nodes in the d-DNNF per node type.
    pub nodes: NodeCount,
    /// The amount of child connections in a d-DNNF.
    pub child_connections: ChildConnections,
    /// Path length information.
    pub paths: Paths,
}

impl From<&Ddnnf> for Statistics {
    fn from(ddnnf: &Ddnnf) -> Self {
        Self {
            nodes: NodeCount::from(ddnnf),
            child_connections: ChildConnections::from(ddnnf),
            paths: Paths::from(ddnnf),
        }
    }
}

#[cfg_attr(feature = "uniffi", uniffi::export)]
impl Ddnnf {
    /// Generates various statistics about this d-DNNF.
    #[cfg_attr(feature = "uniffi", uniffi::method)]
    pub fn statistics(&self) -> Statistics {
        Statistics::from(self)
    }
}

/// The amount of nodes in a d-DNNF per node type.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct NodeCount {
    pub total: usize,
    pub and: usize,
    pub or: usize,
    pub literal: usize,
    pub r#true: usize,
    pub r#false: usize,
}

impl From<&Ddnnf> for NodeCount {
    fn from(ddnnf: &Ddnnf) -> Self {
        let mut counts = Self {
            total: ddnnf.nodes.len(),
            ..Default::default()
        };

        ddnnf.nodes.iter().for_each(|node| match node.ntype {
            NodeType::And { .. } => counts.and += 1,
            NodeType::Or { .. } => counts.or += 1,
            NodeType::Literal { .. } => counts.literal += 1,
            NodeType::True => counts.r#true += 1,
            NodeType::False => counts.r#false += 1,
        });

        counts
    }
}

/// The amount of child connections in a d-DNNF.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct ChildConnections {
    /// The total amount of child connection.
    pub total: usize,
    /// The number of child connections from AND nodes.
    pub and: usize,
    /// The number of child connections from OR nodes.
    pub or: usize,
}

impl From<&Ddnnf> for ChildConnections {
    fn from(ddnnf: &Ddnnf) -> Self {
        let mut connections = Self::default();

        ddnnf.nodes.iter().for_each(|node| match &node.ntype {
            NodeType::And { children } => connections.and += children.len(),
            NodeType::Or { children } => connections.or += children.len(),
            _ => {}
        });

        connections.total = connections.and + connections.or;

        connections
    }
}

/// Path length information about a d-DNNF.
#[derive(Default, Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct Paths {
    /// The total amount of paths.
    pub amount: usize,
    /// The length of the shortest path.
    pub shortest: usize,
    /// The length of the longest path.
    pub longest: usize,
    /// The mean path length.
    pub mean: f64,
    /// The standard deviation of the path lengths.
    pub deviation: f64,
}

impl From<&Ddnnf> for Paths {
    fn from(ddnnf: &Ddnnf) -> Self {
        let depths = calculate_depths(ddnnf, ddnnf.nodes.len() - 1, 0);

        let amount = depths.len();
        let &shortest = depths.iter().min().unwrap();
        let &longest = depths.iter().max().unwrap();
        let mean = depths.iter().sum::<usize>() as f64 / amount as f64;

        let variance: f64 = depths
            .iter()
            .map(|&depth| (depth as f64 - mean).powi(2))
            .sum();
        let deviation = (variance / amount as f64).sqrt();

        Self {
            amount,
            shortest,
            longest,
            mean,
            deviation,
        }
    }
}

/// Calculates the depths of all possible paths starting from the given node.
fn calculate_depths(ddnnf: &Ddnnf, index: usize, current_depth: usize) -> Vec<usize> {
    match &ddnnf.nodes[index].ntype {
        NodeType::And { children } | NodeType::Or { children } => children
            .iter()
            .flat_map(|&child| calculate_depths(ddnnf, child, current_depth + 1))
            .collect(),
        _ => vec![current_depth],
    }
}
