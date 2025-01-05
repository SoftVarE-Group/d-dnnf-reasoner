pub mod anomalies;
pub mod clause_cache;
pub mod counting;
pub mod heuristics;
pub mod multiple_queries;
pub mod node;
pub mod stream;

use self::{clause_cache::ClauseCache, node::Node};
use crate::parser::from_cnf::reduce_clause;
use crate::parser::intermediate_representation::{
    ClauseApplication, IncrementalStrategy, IntermediateGraph,
};
use itertools::Either;
use num::BigInt;
use std::cmp::max;
use std::collections::{BTreeSet, HashMap, HashSet};

type Clause = BTreeSet<i32>;
type ClauseSet = BTreeSet<Clause>;
type EditOperation = (Vec<Clause>, Vec<Clause>);

/// A Ddnnf holds all the nodes as a vector, also includes meta data and further information that is used for optimations
#[derive(Clone, Debug)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Object))]
pub struct Ddnnf {
    /// An intermediate representation that can be changed without destroying the structure
    pub inter_graph: IntermediateGraph,
    /// The actual nodes of the d-DNNF in postorder
    pub nodes: Vec<Node>,
    /// The saved state to enable undoing and adapting the d-DNNF. Avoid exposing this field outside of this source file!
    cached_state: Option<ClauseCache>,
    /// Literals for upwards propagation
    pub literals: HashMap<i32, usize>, // <var_number of the Literal, and the corresponding indize>
    pub true_nodes: Vec<usize>, // Indices of true nodes. In some cases those nodes needed to have special treatment
    /// The core/dead features of the model corresponding with this ddnnf
    pub core: HashSet<i32>,
    /// An interim save for the marking algorithm
    pub md: Vec<usize>,
    pub number_of_variables: u32,
    /// The number of threads
    pub max_worker: u16,
}

#[cfg_attr(feature = "uniffi", uniffi::export)]
impl Default for Ddnnf {
    #[cfg_attr(feature = "uniffi", uniffi::constructor)]
    fn default() -> Self {
        Ddnnf {
            inter_graph: IntermediateGraph::default(),
            nodes: Vec::new(),
            cached_state: None,
            literals: HashMap::new(),
            true_nodes: Vec::new(),
            core: HashSet::new(),
            md: Vec::new(),
            number_of_variables: 0,
            max_worker: 4,
        }
    }
}

#[cfg_attr(feature = "uniffi", uniffi::export)]
impl Ddnnf {
    /// Loads a d-DNNF from file.
    #[cfg_attr(feature = "uniffi", uniffi::constructor)]
    fn from_file(path: String, features: Option<u32>) -> Self {
        crate::parser::build_ddnnf(&path.clone(), features)
    }

    /// Loads a d-DNNF from file, using the projected d-DNNF compilation.
    ///
    /// Panics when not including d4 as it is required for projected compilation.
    #[cfg_attr(feature = "uniffi", uniffi::constructor)]
    fn from_file_projected(path: String, features: Option<u32>) -> Self {
        #[cfg(feature = "d4")]
        return crate::parser::build_ddnnf_projected(&path.clone(), features);
        #[cfg(not(feature = "d4"))]
        panic!("d4 is required for projected compilation.");
    }

    /// Returns the current count of the root node in the d-DNNF.
    ///
    /// This value is the same during all computations.
    #[cfg_attr(feature = "uniffi", uniffi::method)]
    pub fn rc(&self) -> BigInt {
        self.nodes[self.nodes.len() - 1].count.clone()
    }

    /// Returns the core features of this d-DNNF.
    ///
    /// This is only calculated once at creation of the d-DNNF.
    #[cfg_attr(feature = "uniffi", uniffi::method)]
    pub fn get_core(&self) -> HashSet<i32> {
        self.core.clone()
    }
}

impl Ddnnf {
    /// Creates a new ddnnf including dead and core features
    pub fn new(mut inter_graph: IntermediateGraph, number_of_variables: u32) -> Ddnnf {
        let dfs_ig = inter_graph.rebuild(None);
        let clauses: BTreeSet<BTreeSet<i32>> = inter_graph
            .cnf_clauses
            .iter()
            .map(|clause| clause.iter().copied().collect())
            .collect();

        let mut ddnnf = Ddnnf {
            inter_graph,
            nodes: dfs_ig.0,
            literals: dfs_ig.1,
            true_nodes: dfs_ig.2,
            cached_state: None,
            core: HashSet::new(),
            md: Vec::new(),
            number_of_variables,
            max_worker: 4,
        };

        ddnnf.calculate_core();

        if !clauses.is_empty() {
            ddnnf.update_cached_state(Either::Right(clauses), Some(number_of_variables));
        }

        ddnnf
    }

    /// Checks if the creation of a cached state is valid.
    /// That is only the case if the input format was CNF.
    pub fn can_save_state(&self) -> bool {
        self.cached_state.is_some()
    }

    /// Either initialises the ClauseCache by saving the clauses and its corresponding clauses
    /// or updates the state accordingly.
    pub fn update_cached_state(
        &mut self,
        clause_info: Either<EditOperation, ClauseSet>,
        total_features: Option<u32>,
    ) -> bool {
        match self.cached_state.as_mut() {
            Some(state) => match clause_info.left() {
                Some((add, rmv)) => {
                    if total_features.is_none()
                        || !state.apply_edits_and_replace(add, rmv, total_features.unwrap())
                    {
                        return false;
                    }
                    // The old d-DNNF got replaced by the new one.
                    // Consequently, the current higher level d-DNNF becomes the older one.
                    // We swap their field data to keep the order without needing to deal with recursivly building up
                    // obselete d-DNNFs that trash the RAM.
                    self.swap();
                }
                None => return false,
            },
            None => match clause_info.right() {
                Some(clauses) => {
                    let mut state = ClauseCache::default();
                    state.initialize(clauses, total_features.unwrap());
                    self.cached_state = Some(state);
                }
                None => return false,
            },
        }
        true
    }

    fn swap(&mut self) {
        if let Some(cached_state) = self.cached_state.as_mut() {
            if let Some(save_state) = cached_state.old_state.as_mut() {
                std::mem::swap(&mut self.nodes, &mut save_state.nodes);
                std::mem::swap(&mut self.literals, &mut save_state.literals);
                std::mem::swap(&mut self.true_nodes, &mut save_state.true_nodes);
                std::mem::swap(&mut self.core, &mut save_state.core);
                std::mem::swap(&mut self.md, &mut save_state.md);
                std::mem::swap(
                    &mut self.number_of_variables,
                    &mut save_state.number_of_variables,
                );
                std::mem::swap(&mut self.max_worker, &mut save_state.max_worker);
            }
        }
    }

    // Performes an undo operation resulting in swaping the current d-DNNF with its older version.
    // Hence, the perviously older version becomes the current one and the current one becomes the older version.
    // A second undo operation in a row is equivalent to a redo. Can fail if there is no old d-DNNF available.
    pub fn undo_on_cached_state(&mut self) -> bool {
        match self.cached_state.as_mut() {
            Some(state) => {
                state.setup_for_undo();
                self.swap();
                //std::mem::swap(self, &mut state.to_owned().get_old_state().unwrap());
                true
            }
            None => false,
        }
    }

    /// We invalidate all collected data that belongs to the dDNNF and build it again
    /// by doing a DFS. That is necessary if we altered the intermedidate graph in any way.
    pub fn rebuild(&mut self) {
        let dfs_ig = self.inter_graph.rebuild(None);
        self.nodes = dfs_ig.0;
        self.literals = dfs_ig.1;
        self.true_nodes = dfs_ig.2;

        self.get_core();
        self.md.clear();
        // The highest absolute value of literals must be is also the number of variables because
        // there are no gaps in the feature to number mapping
        self.number_of_variables = self
            .literals
            .keys()
            .fold(0, |acc, x| max(acc, x.unsigned_abs()));

        let clauses: BTreeSet<BTreeSet<i32>> = self
            .inter_graph
            .cnf_clauses
            .iter()
            .map(|clause| clause.iter().copied().collect())
            .collect();

        if !clauses.is_empty() {
            self.update_cached_state(Either::Right(clauses), Some(self.number_of_variables));
        }
    }

    /// Takes a list of clauses. Each clause consists out of one or multiple variables that are conjuncted.
    /// The clauses are disjuncted.
    /// Example: [[1, -2, 3], [4]] would represent (1 ∨ ¬2 ∨ 3) ∧ (4)
    pub fn prepare_and_apply_incremental_edit(
        &mut self,
        edit_operations: Vec<(Vec<i32>, ClauseApplication)>,
    ) -> IncrementalStrategy {
        let mut edit_lits = HashSet::new();
        let mut op_add = Vec::new();
        let mut op_rmv = Vec::new();
        for (clause, application) in edit_operations {
            match reduce_clause(&clause, &HashSet::new()) {
                Some(reduced_clause) => {
                    if reduced_clause.is_empty() {
                        continue;
                    }
                    edit_lits.extend(reduced_clause.iter());
                    match application {
                        ClauseApplication::Add => op_add.push(reduced_clause),
                        ClauseApplication::Remove => op_rmv.push(reduced_clause),
                    }
                }
                None => panic!("dDNNF becomes UNSAT for clause: {:?}!", clause),
            }
        }
        let strategy = self
            .inter_graph
            .apply_incremental_edit((edit_lits, (op_add, op_rmv)));
        self.rebuild();
        strategy
    }

    // Returns the current temp count of the root node in the ddnnf.
    // That value is changed during computations
    fn rt(&self) -> BigInt {
        self.nodes[self.nodes.len() - 1].temp.clone()
    }

    /// Computes the total number of nodes in the dDNNF
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Computes the total number of edges in the dDNNF
    pub fn edge_count(&self) -> usize {
        use crate::NodeType::*;
        let mut total_edges = 0;

        for node in self.nodes.iter() {
            match &node.ntype {
                And { children } | Or { children } => {
                    total_edges += children.len();
                }
                _ => (),
            }
        }
        total_edges
    }

    /// Computes the sharing of nodes in the dDNNF.
    /// We define sharing as #nodes / #nodes as tree
    pub fn sharing(&self) -> f64 {
        use crate::NodeType::*;
        let mut sub_nodes = vec![0_u64; self.node_count()];

        for (index, node) in self.nodes.iter().enumerate() {
            match &node.ntype {
                And { children } | Or { children } => {
                    sub_nodes[index] = children.iter().fold(0, |acc, &i| acc + sub_nodes[i]) + 1
                }
                _ => sub_nodes[index] = 1,
            }
        }
        self.node_count() as f64 / sub_nodes.last().unwrap().to_owned() as f64
    }

    /// Determines the positions of the inverted featueres
    pub fn map_features_opposing_indexes(&self, features: &[i32]) -> Vec<usize> {
        let mut indexes = Vec::with_capacity(features.len());
        for number in features {
            if let Some(i) = self.literals.get(&-number).cloned() {
                indexes.push(i);
            }
        }
        indexes
    }

    /// Executes a query.
    /// We use the in our opinion best type of query depending on the amount of features.
    ///
    /// # Example
    /// ```
    /// use ddnnife::Ddnnf;
    /// use ddnnife::parser::*;
    /// use num::BigInt;
    ///
    /// // create a ddnnf
    /// let file_path = "./tests/data/small_ex_c2d.nnf";
    /// let mut ddnnf: Ddnnf = build_ddnnf(file_path, None);
    ///
    /// assert_eq!(BigInt::from(1), ddnnf.execute_query(&vec![3,4]));
    /// assert_eq!(BigInt::from(2), ddnnf.execute_query(&vec![3]));
    pub fn execute_query(&mut self, features: &[i32]) -> BigInt {
        match features.len() {
            0 => self.rc(),
            1 => self.card_of_feature_with_marker(features[0]),
            2..=20 => {
                self.operate_on_partial_config_marker(features, Ddnnf::calc_count_marked_node)
            }
            _ => self.operate_on_partial_config_default(features, Ddnnf::calc_count),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::parser::build_ddnnf;

    #[test]
    fn features_opposing_indexes() {
        let ddnnf = build_ddnnf("tests/data/small_ex_c2d.nnf", None);

        assert_eq!(
            vec![4, 2, 9],
            ddnnf.map_features_opposing_indexes(&[1, 2, 3, 4])
        );
        assert_eq!(
            vec![0, 1, 5, 8],
            ddnnf.map_features_opposing_indexes(&[-1, -2, -3, -4])
        );
    }
}
