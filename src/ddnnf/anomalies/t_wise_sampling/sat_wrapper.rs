use crate::Ddnnf;

#[derive(Debug, Clone)]
pub struct SatWrapper<'a> {
    ddnnf: &'a Ddnnf,
    new_state: Vec<bool>,
}

impl<'a> SatWrapper<'a> {
    /// Create a new [SatWrapper] backed by the given d-DNNF
    pub(super) fn new(ddnnf: &'a Ddnnf) -> Self {
        Self {
            ddnnf,
            new_state: vec![false; ddnnf.nodes.len()],
        }
    }

    /// Create a new state vec for this solver
    pub(super) fn new_state(&self) -> Vec<bool> {
        // clone the state where the false nodes are already marked
        self.new_state.clone()
    }

    /// Calculates if the given config is SAT. This is the variant with cached state.
    ///
    /// See [SatWrapper] for more details
    pub(super) fn is_sat_cached(&self, config: &[i32], cached_state: &mut Vec<bool>) -> bool {
        self.is_sat_in_subgraph_cached(config, self.ddnnf.nodes.len() - 1, cached_state)
    }

    /// Calculates if the given config is SAT. This is the subgraph variant with cached state.
    ///
    /// See [SatWrapper] for more details
    pub(super) fn is_sat_in_subgraph_cached(
        &self,
        config: &[i32],
        root: usize,
        cached_state: &mut Vec<bool>,
    ) -> bool {
        self.ddnnf.sat_propagate(config, cached_state, Some(root))
    }
}
