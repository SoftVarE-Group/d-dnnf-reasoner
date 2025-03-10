use crate::Ddnnf;
use ddnnife::ddnnf::statistics;

type Statistics = statistics::Statistics;
type NodeCount = statistics::NodeCount;
type ChildConnections = statistics::ChildConnections;
type Paths = statistics::Paths;

#[uniffi::export]
impl Ddnnf {
    /// Generates various statistics about this d-DNNF.
    #[uniffi::method]
    pub fn statistics(&self) -> Statistics {
        self.0.statistics()
    }
}

#[uniffi::remote(Record)]
pub struct Statistics {
    /// The amount of nodes in the d-DNNF per node type.
    pub nodes: NodeCount,
    /// The amount of child connections in a d-DNNF.
    pub child_connections: ChildConnections,
    /// Path length information.
    pub paths: statistics::Paths,
}

#[uniffi::remote(Record)]
pub struct NodeCount {
    pub total: usize,
    pub and: usize,
    pub or: usize,
    pub literal: usize,
    pub r#true: usize,
    pub r#false: usize,
}

#[uniffi::remote(Record)]
pub struct ChildConnections {
    /// The total amount of child connection.
    pub total: usize,
    /// The number of child connections from AND nodes.
    pub and: usize,
    /// The number of child connections from OR nodes.
    pub or: usize,
}

#[uniffi::remote(Record)]
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
