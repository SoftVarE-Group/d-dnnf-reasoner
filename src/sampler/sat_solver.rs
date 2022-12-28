use crate::data_structure::NodeType;
use crate::Ddnnf;

/// This is a d-DNNF backed SAT-Solver.
/// It works basically in the same way as the marking algorithm for partial configuration
/// counting but with a few shortcuts. This solver also allows SAT-Solving
/// for sub graphs of the d-DNNF and allows to cache the solvers state between calls.
/// The [SatSolver::is_sat()] function has different variants for when those features
/// are needed or not.
///
/// # Subgraph SAT
///
/// Calls to the subgraph variants take a root node as additional parameter.
/// The solver then calculates the satisfiability of the config for the subgraph rooted at
/// that node. The rest of the d-DNNF will not influence the result.
///
/// If a subgraph variant returns true, then the config is SAT in the subgraph and
/// consequently in the d-DNNF as a whole. If this returns false, then the config is
/// not SAT in the subgraph. In this case, no statement can be made whether
/// the configuration is SAT or not for the entire d-DNNF.
///
/// # Solver State Caching
///
/// This solver allows the caller to cache the state of the solver between calls. This can greatly
/// improve performance when calculating SAT for many configurations that share a common core.
/// For example, if you have a configuration and want to test it with many pair-wise interactions.
/// You can then calculate SAT for the configuration once. After that you can call the solver
/// for the interactions alone while passing the cached state. The solver then only has to consider
/// the literals of the interactions.
///
/// Note: The caching variants mutate the given state. If you need the same state multiple times
/// (which is likely because why would you cache it otherwise?) it is your responsibility as caller
/// to copy the state beforehand.
#[derive(Debug, Clone)]
pub struct SatSolver<'a> {
    ddnnf: &'a Ddnnf,
    new_state: Vec<bool>,
}

const STATE_SIZE: &str =
    "size of sat solver state vec should equal number of nodes in the ddnnf";

const NODE_EXISTS: &str = "node should exist in the ddnnf";

impl<'a> SatSolver<'a> {
    /// Create a new [SatSolver] backed by the given d-DNNF
    pub fn new(ddnnf: &'a Ddnnf) -> Self {
        let mut result = Self {
            ddnnf,
            new_state: vec![],
        };

        // Find and mark all False nodes
        let mut new_state = vec![false; ddnnf.number_of_nodes];
        ddnnf
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(node_id, node)| {
                if let NodeType::False = &node.ntype {
                    Some(node_id)
                } else {
                    None
                }
            })
            .for_each(|node_id| {
                result.mark(node_id, ddnnf.number_of_nodes - 1, &mut new_state);
            });

        result.new_state = new_state;
        result
    }

    /// Create a new state vec for this solver
    pub fn new_state(&self) -> Vec<bool> {
        // clone the state where the false nodes are already marked
        self.new_state.clone()
    }

    /// Calculates if the given config is SAT.
    ///
    /// See [SatSolver] for more details
    pub fn is_sat(&self, config: &[i32]) -> bool {
        let mut state = vec![false; self.ddnnf.number_of_nodes];
        self.is_sat_cached(config, &mut state)
    }

    /// Calculates if the given config is SAT. This is the variant with cached state.
    ///
    /// See [SatSolver] for more details
    pub fn is_sat_cached(
        &self,
        config: &[i32],
        cached_state: &mut Vec<bool>,
    ) -> bool {
        self.is_sat_in_subgraph_cached(
            config,
            self.ddnnf.number_of_nodes - 1,
            cached_state,
        )
    }

    /// Calculates if the given config is SAT. This is the subgraph variant.
    ///
    /// See [SatSolver] for more details
    pub fn is_sat_in_subgraph(&self, config: &[i32], root: usize) -> bool {
        let mut state = vec![false; self.ddnnf.number_of_nodes];
        self.is_sat_in_subgraph_cached(config, root, &mut state)
    }

    /// Calculates if the given config is SAT. This is the subgraph variant with cached state.
    ///
    /// See [SatSolver] for more details
    pub fn is_sat_in_subgraph_cached(
        &self,
        config: &[i32],
        root: usize,
        cached_state: &mut Vec<bool>,
    ) -> bool {
        let &root_not_sat = cached_state.get(root).expect(STATE_SIZE);
        if root_not_sat {
            return false;
        }

        for literal in config {
            // get the node_id of the negated literal
            let negated = self.ddnnf.literals.get(&-literal);
            if let Some(negated) = negated {
                if self.mark(*negated, root, cached_state) {
                    /*
                    One literal propagated up to the root. We now know that this
                    config is not sat.
                    Note: At this point, cached_state is incomplete because we return early.
                    But this is not a problem because a config that is not sat will
                    remain not sat no matter what literals are added.
                    (The fact that this config is not sat is present in the
                     */
                    return false;
                }
            }
        }
        return !*cached_state.get(root).expect(STATE_SIZE);
    }

    /// Mark a node as *not sat* and then propagates up until no node can be marked anymore.
    /// Returns true if the root got marked. Otherwise, false is returned.
    fn mark(&self, node: usize, root: usize, marked: &mut Vec<bool>) -> bool {
        if *marked.get(node).expect(STATE_SIZE) {
            return false; // do nothing if the node was marked already
        }
        marked[node] = true;
        if node == root {
            return true;
        }

        let node = self.ddnnf.nodes.get(node).expect(NODE_EXISTS);

        node.parents.iter().any(|&parent| {
            if self.should_mark_parent(parent, marked) {
                self.mark(parent, root, marked)
            } else {
                false
            }
        })
    }

    /// Finds out if a parent node should be marked or not.
    /// And nodes should be marked if at least one child is marked.
    /// Or nodes should be marked if all children are marked.
    /// Other node types can not be the parent of another node.
    fn should_mark_parent(&self, node: usize, marked: &[bool]) -> bool {
        let node = self.ddnnf.nodes.get(node).expect(NODE_EXISTS);
        match &node.ntype {
            NodeType::And { .. } => {
                // And node should be marked if at least one child is marked.
                // This is the case because otherwise this method would not have been called.
                true
            }
            NodeType::Or { children } => children
                .iter()
                .all(|&child| *marked.get(child).expect(STATE_SIZE)),
            _ => panic!("node type of a parent node should be And or Or"),
        }
    }
}
