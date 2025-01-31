mod fixed_fifo;

use std::{
    cmp::max,
    collections::{HashMap, HashSet},
    io::Write,
    ops::Not,
    panic,
    sync::Arc,
};

use itertools::{Either, Itertools};
use petgraph::{
    algo::is_cyclic_directed,
    stable_graph::NodeIndex,
    visit::{Dfs, DfsPostOrder, NodeIndexable},
    Direction::{Incoming, Outgoing},
};

use self::fixed_fifo::{ConflictFn, FixedFifo};
use super::{calc_and_count, calc_or_count, from_cnf::apply_decisions, DdnnfGraph};
use crate::util::format_vec;
use crate::{
    c2d_lexer::TId,
    parser::{
        build_ddnnf, extend_literal_diffs,
        from_cnf::{get_all_clauses_cnf, simplify_clauses},
    },
    Node, NodeType,
};

const MAX_CACHED_SUB_DAGS: usize = 10;

// (LITERALS, (CLAUSES TO ADD, CLAUSES TO REMOVE))
type IncrementalEdit = (HashSet<i32>, (Vec<Vec<i32>>, Vec<Vec<i32>>));
type SubDAGInfo = (NodeIndex, DdnnfGraph, NodeIndex);
type CachedSubDag = (IncrementalEdit, Either<SubDAGInfo, IntermediateGraph>);

fn get_push_retain_fn_for_cachedsubdag() -> Arc<ConflictFn<CachedSubDag>> {
    Arc::new(|((_, _), cache_entry), ((_, _), cache_entry_other)| {
        use crate::c2d_lexer::TokenIdentifier::*;
        let ddnnf_graph = match cache_entry {
            Either::Left((_, graph, _)) => graph,
            Either::Right(_) => return false,
        };
        let ddnnf_graph_other = match cache_entry_other {
            Either::Left((_, graph, _)) => graph,
            Either::Right(_) => return false,
        };

        let mut lits = HashSet::new();
        for node in ddnnf_graph.node_indices() {
            if let Literal { feature } = ddnnf_graph[node] {
                lits.insert(feature);
            }
        }

        for node in ddnnf_graph_other.node_indices() {
            if let Literal { feature } = ddnnf_graph[node] {
                if lits.contains(&feature) {
                    return false;
                }
            }
        }

        true
    })
}

/// The IntermediateGraph enables us to modify the dDNNF. The structure of a vector of nodes does not allow
/// for that because deleting or removing nodes would mess up the indices.
#[derive(Debug, Clone)]
pub struct IntermediateGraph {
    graph: DdnnfGraph,
    cache: FixedFifo<CachedSubDag>,
    pub root: NodeIndex,
    number_of_variables: u32,
    literals_nx: HashMap<i32, NodeIndex>,
    literal_children: HashMap<NodeIndex, HashSet<i32>>,
    pub cnf_clauses: Vec<Vec<i32>>,
}

impl Default for IntermediateGraph {
    fn default() -> Self {
        IntermediateGraph {
            graph: DdnnfGraph::default(),
            cache: FixedFifo::new(MAX_CACHED_SUB_DAGS, get_push_retain_fn_for_cachedsubdag()),
            root: NodeIndex::default(),
            number_of_variables: 0,
            literals_nx: HashMap::default(),
            literal_children: HashMap::default(),
            cnf_clauses: Vec::default(),
        }
    }
}

/// A clause can either be added or removed
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ClauseApplication {
    /// Add a clause to the d-DNNF
    Add,
    /// Remove a clause from the d-DNNF
    Remove,
}

impl Not for ClauseApplication {
    type Output = ClauseApplication;

    fn not(self) -> Self::Output {
        use ClauseApplication::*;
        match self {
            Add => Remove,
            Remove => Add,
        }
    }
}

/// There are four different strategies that can be used to incrementally add a clause.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum IncrementalStrategy {
    /// Adding this clause does not alter the d-DNNF. Thus, we don't have to do anything
    Tautology,
    /// A clause that contains exactly one literal. We have a special handling for those clauses.
    UnitClause,
    /// The core algorithm: we identify the affected sub-DAG and
    /// replace it with its substitute that contains the incremental clause
    SubDAGReplacement,
    /// There is no better solution, than compile the whole CNF with the changes
    /// because almost everything is affected by the new clause.
    Recompile,
    /// Revert the adding / removing of a clause.
    /// Undo is used implicitly instead of any other technique if the cache allows for it.
    Undo,
    /// The error state represents results that occur due to bad queries.
    Error,
}

impl IntermediateGraph {
    /// Creates a new IntermediateGraph
    pub fn new(
        graph: DdnnfGraph,
        root: NodeIndex,
        number_of_variables: u32,
        literals_nx: HashMap<i32, NodeIndex>,
        literal_children: HashMap<NodeIndex, HashSet<i32>>,
        cnf_path: Option<&str>,
    ) -> IntermediateGraph {
        debug_assert!(!is_cyclic_directed(&graph));
        let mut inter_graph = IntermediateGraph {
            graph,
            cache: FixedFifo::new(MAX_CACHED_SUB_DAGS, get_push_retain_fn_for_cachedsubdag()),
            root,
            number_of_variables,
            literals_nx,
            literal_children,
            cnf_clauses: match cnf_path {
                Some(path) => get_all_clauses_cnf(path),
                None => Vec::new(),
            },
        };
        inter_graph.cnf_clauses = simplify_clauses(inter_graph.cnf_clauses);
        inter_graph
    }

    // move all values from other to self
    fn move_inter_graph(&mut self, other: IntermediateGraph) {
        self.graph = other.graph;
        self.root = other.root;
        self.number_of_variables = other.number_of_variables;
        self.literals_nx = other.literals_nx;
        self.literal_children = other.literal_children;
        self.cnf_clauses = other.cnf_clauses;
    }

    /// Starting for the IntermediateGraph, we do a PostOrder walk through the graph the create the
    /// list of nodes which we use for counting operations and other types of queries.
    pub fn rebuild(
        &mut self,
        alt_root: Option<NodeIndex>,
    ) -> (Vec<Node>, HashMap<i32, usize>, Vec<usize>) {
        // always make sure that there are no cycles
        debug_assert!(!is_cyclic_directed(&self.graph));

        // perform a depth first search to get the nodes ordered such
        // that child nodes are listed before their parents
        // transform that interim representation into a node vector
        let mut dfs = DfsPostOrder::new(&self.graph, alt_root.unwrap_or(self.root));
        let mut nd_to_usize: HashMap<NodeIndex, usize> = HashMap::new();

        let mut parsed_nodes: Vec<Node> = Vec::with_capacity(self.graph.node_count());
        let mut literals: HashMap<i32, usize> = HashMap::new();
        let mut true_nodes = Vec::new();

        while let Some(nx) = dfs.next(&self.graph) {
            nd_to_usize.insert(nx, parsed_nodes.len());
            let neighs = self
                .graph
                .neighbors(nx)
                .map(|n| *nd_to_usize.get(&n).unwrap())
                .collect::<Vec<usize>>();
            let next: Node = match self.graph[nx] {
                // extract the parsed Token
                TId::Literal { feature } => Node::new_literal(feature),
                TId::And => Node::new_and(calc_and_count(&mut parsed_nodes, &neighs), neighs),
                TId::Or => Node::new_or(0, calc_or_count(&mut parsed_nodes, &neighs), neighs),
                TId::True => Node::new_bool(true),
                TId::False => Node::new_bool(false),
                TId::Header => {
                    panic!("The d4 standard does not include a header!")
                }
            };

            match &next.ntype {
                // build additional references from the child to its parent
                NodeType::And { children } | NodeType::Or { children } => {
                    let next_indize: usize = parsed_nodes.len();
                    for &i in children {
                        parsed_nodes[i].parents.push(next_indize);
                    }
                }
                // create mapping from literal to its node index
                NodeType::Literal { literal } => {
                    literals.insert(*literal, parsed_nodes.len());
                }
                NodeType::True => {
                    true_nodes.push(parsed_nodes.len());
                }
                _ => (),
            }

            parsed_nodes.push(next);
        }

        (parsed_nodes, literals, true_nodes)
    }

    /// For a given clause we search for the node that contains all literals of that clause
    /// and therefore, all other clauses that contain those literals while having as little children
    /// as possible.
    /// Further, the node has to be the endpoint of bridge.
    pub fn closest_unsplitable_bridge(
        &mut self,
        edit: &IncrementalEdit,
    ) -> Option<(NodeIndex, HashSet<i32>)> {
        let (edit_lits, (_, _)) = edit;
        if edit_lits.is_empty() {
            return None;
        }
        let mut closest_node = (self.root, self.literal_children.get(&self.root).unwrap());

        let mut bridge_end_point = Vec::new();
        for bridge_endpoint in self.find_bridges() {
            match self.graph[bridge_endpoint] {
                TId::And | TId::Or => {
                    let diffs = self.literal_children.get(&bridge_endpoint).unwrap();
                    // we only consider and nodes that include all literals of the clause
                    if edit_lits
                        .iter()
                        .all(|e| diffs.contains(e) && diffs.contains(&-e))
                        && closest_node.1.len() > diffs.len()
                    {
                        closest_node = (bridge_endpoint, diffs);
                    }
                    bridge_end_point.push(bridge_endpoint);
                }
                _ => (),
            }
        }

        let devided_and = self.divide_bridge(closest_node.0, &bridge_end_point, edit);
        Some((
            devided_and,
            self.literal_children.get(&devided_and).unwrap().clone(),
        ))
    }

    // Evgeny: https://stackoverflow.com/questions/23179579/finding-bridges-in-graph-without-recursion
    #[inline]
    fn find_bridges(&self) -> HashSet<NodeIndex> {
        let mut bridges = HashSet::new();

        // calculate neighbours beforehand, because we have to index them multiple times
        let mut neighbours = vec![Vec::new(); self.graph.node_bound()];

        #[allow(clippy::needless_range_loop)]
        for i in 0..self.graph.node_count() {
            let neighbours_inc = self
                .graph
                .neighbors_directed(self.graph.from_index(i), Incoming);
            let neighbours_out = self
                .graph
                .neighbors_directed(self.graph.from_index(i), Outgoing);
            neighbours[i] = neighbours_inc
                .chain(neighbours_out)
                .map(|nx| self.graph.to_index(nx))
                .collect::<Vec<usize>>();
        }

        let mut visited = vec![false; self.graph.node_bound()];
        let mut tin = vec![0usize; self.graph.node_bound()]; // time-in
        let mut low = vec![0usize; self.graph.node_bound()]; // f-up (minimum discovery time)
        let mut timer = 0usize;
        let mut stack: Vec<(usize, usize, usize)> = Vec::new(); // (node, parent) pairs

        stack.push((self.root.index(), self.graph.node_bound() + 1, 0));

        while let Some((v, parent, i)) = stack.pop() {
            if i == 0 {
                visited[v] = true;
                timer += 1;
                tin[v] = timer;
                low[v] = timer;
            }

            if i < neighbours[v].len() {
                let to = neighbours[v][i];
                stack.push((v, parent, i + 1));
                if to != parent {
                    if visited[to] {
                        low[v] = low[v].min(tin[to]);
                    } else {
                        stack.push((to, v, 0));
                    }
                }
            }
            if i > 0 && i <= neighbours[v].len() {
                let to = neighbours[v][i - 1];
                if to != parent {
                    low[v] = low[v].min(low[to]);
                    if low[to] > tin[v] {
                        bridges.insert(self.graph.from_index(to));
                    }
                }
            }
        }

        bridges
    }

    // Change the structure by splitting one AND/OR node into two.
    // In the following, we focus on the new AND'/OR' instead of AND/OR, resulting
    // in a smaller portion of the dDNNF, we have to travers.
    //
    // We can exclude all children fulfill:
    //      1) don't contain any of the variables of the clause
    //      2) the edge beteen the AND an its child is a bridge
    //
    //
    // Example (with c1 and c2 as the relevant_children):
    //
    //           AND                        AND
    //      ______|______      =>    _______|_______
    //     /   /  |  \   \          |      |   \    \
    //    c1  c2 c3  ... cn        AND'   c3   ...  cn
    //                            /   \
    //                           c1   c2
    //
    fn divide_bridge(
        &mut self,
        initial_start: NodeIndex,
        bridges: &[NodeIndex],
        edit: &IncrementalEdit,
    ) -> NodeIndex {
        let (edit_lits, (op_add, op_rmv)) = edit;
        if op_add.is_empty() && op_rmv.is_empty() {
            return initial_start;
        }

        // Find those children that are relevant to the provided clause.
        // Those all contain at least one of the literals of the clause.
        let mut relevant_children = Vec::new();
        let mut contains_at_least_one_irrelevant = false;
        let mut children = self
            .graph
            .neighbors_directed(initial_start, Outgoing)
            .detach();
        while let Some(child) = children.next_node(&self.graph) {
            let lits = self.literal_children.get(&child).unwrap();
            if edit_lits
                .iter()
                .any(|f| lits.contains(f) || lits.contains(&-f))
                || bridges.contains(&child)
            {
                relevant_children.push(child);
            } else {
                contains_at_least_one_irrelevant = true;
            }
        }

        // If there is no irrelevant child, it makes no sense to divide the AND node.
        if !contains_at_least_one_irrelevant {
            return initial_start;
        }

        // Create the new node and adjust the edges.
        let new_node = self.graph.add_node(self.graph[initial_start]);
        let mut new_node_lits = HashSet::new();
        for child in relevant_children {
            if let Some(edge) = self.graph.find_edge(initial_start, child) {
                self.graph.remove_edge(edge);
            }
            self.graph.add_edge(new_node, child, ());

            new_node_lits.extend(self.literal_children.get(&child).unwrap());
        }
        self.graph.add_edge(initial_start, new_node, ());

        // Add child literals of new_and to mapping
        self.literal_children.insert(new_node, new_node_lits);

        new_node
    }

    // Experimental method to use the initial CNF for recompiling
    pub fn transform_to_cnf_from_starting_cnf(
        &mut self,
        edit: &IncrementalEdit,
    ) -> (Vec<String>, NodeIndex, HashMap<u32, u32>) {
        let (_, (op_add, op_rmv)) = edit;

        if self.cnf_clauses.is_empty() {
            return (Vec::new(), NodeIndex::new(0), HashMap::new());
        }

        self.adjust_intern_cnf(op_add, op_rmv);

        // 1) Find the closest node and the decisions to that point
        let (closest_node, relevant_literals) = match self.closest_unsplitable_bridge(edit) {
            Some((nx, lits)) => (nx, lits),
            None => (self.root, HashSet::new()), // Just do the whole CNF without any adjustments
        };

        // Recompile the whole dDNNF if at least 80% are children of the closest_node we could replace
        if relevant_literals.len() as f32
            > self.literal_children.get(&self.root).unwrap().len() as f32 * 0.85
        {
            return (Vec::new(), self.root, HashMap::new());
        }

        use crate::c2d_lexer::TokenIdentifier::*;
        let mut accumlated_decisions = HashSet::new(); //self.get_decisions_target_nx(closest_node);
        let mut lits = HashSet::new();
        let mut dfs = Dfs::new(&self.graph, self.root);
        while let Some(nx) = dfs.next(&self.graph) {
            if let Literal { feature } = self.graph[nx] {
                lits.insert(feature);
            }
        }
        for &num in &lits {
            if !lits.contains(&(-num)) {
                accumlated_decisions.insert(num);
            }
        }

        // 2) Filter for clauses that contain at least one of the relevant literals
        //    aka the literals that are children of the closest AND node
        let mut relevant_clauses: Vec<Vec<i32>> = if op_add.is_empty() && op_rmv.is_empty() {
            self.cnf_clauses.clone()
        } else {
            self.cnf_clauses
                .clone()
                .into_iter()
                .filter(|initial_clause| {
                    initial_clause.iter().any(|elem| {
                        relevant_literals.contains(elem) || relevant_literals.contains(&-elem)
                    } || initial_clause.len() == 1) // Add unit clauses. Those decisions are implicit in the dDNNF.
                })
                .collect_vec()
        };

        // 2.5 add each feature as optional to account for features that become optional
        //     due to the newly added clause.
        let mut variables = HashSet::new();
        for var in relevant_literals.iter() {
            variables.insert(var.unsigned_abs());
        }

        // 2.75) Repeatedly apply the summed up decions to the remaining clauses
        (relevant_clauses, accumlated_decisions) =
            apply_decisions(relevant_clauses, accumlated_decisions);

        // 3) Handle consequences of decisions
        let mut red_variables = HashSet::new();
        for clause in relevant_clauses.iter() {
            for variable in clause {
                red_variables.insert(variable.unsigned_abs());
            }
        }

        for variable in variables.symmetric_difference(&red_variables).cloned() {
            let v_i32 = &(variable as i32);
            if !accumlated_decisions.contains(v_i32)
                && !accumlated_decisions.contains(&-v_i32)
                && (relevant_literals.contains(v_i32) || relevant_literals.contains(&-v_i32))
            {
                relevant_clauses.push(vec![variable as i32, -(variable as i32)]);
            }

            if accumlated_decisions.contains(v_i32) {
                relevant_clauses.push(vec![*v_i32]);
            } else if accumlated_decisions.contains(&-v_i32) {
                relevant_clauses.push(vec![-v_i32]);
            }
        }

        if relevant_clauses.is_empty() {
            return (Vec::new(), NodeIndex::new(0), HashMap::new());
        }

        // 4) Adjust indices of the literals
        //    (to avoid unwanted gaps in the CNF which would be interpreted as optional features)
        let mut index = 1_u32;
        let mut re_index: HashMap<u32, u32> = HashMap::new();
        for clause in relevant_clauses.iter_mut() {
            for elem in clause {
                let elem_signum = elem.signum();
                match re_index.get(&(elem.unsigned_abs())) {
                    Some(val) => {
                        *elem = *val as i32 * elem_signum;
                    }
                    None => {
                        re_index.insert(elem.unsigned_abs(), index);
                        *elem = index as i32 * elem_signum;
                        index += 1;
                    }
                }
            }
        }

        // write the meta information of the header
        let mut cnf = vec![format!("p cnf {} {}\n", index - 1, relevant_clauses.len())];

        for clause in relevant_clauses {
            cnf.push(format!("{} 0\n", format_vec(clause.iter())));
        }

        // Swaps key with value in the key value pairs
        let pairs: Vec<(u32, u32)> = re_index.drain().collect();
        for (key, value) in pairs {
            re_index.insert(value, key);
        }
        (cnf, closest_node, re_index)
    }

    fn adjust_intern_cnf(&mut self, op_add: &[Vec<i32>], op_rmv: &[Vec<i32>]) {
        // make adjustments to the clauses
        if !op_rmv.is_empty() {
            let mut rmv_clause_set = Vec::new();
            for clause in op_rmv {
                rmv_clause_set.push(clause.iter().cloned().collect::<HashSet<_>>());
            }
            self.cnf_clauses.retain(|rel_clause| {
                rmv_clause_set.iter().any(|rmv_clause| {
                    &rel_clause.iter().cloned().collect::<HashSet<_>>() != rmv_clause
                })
            });
        }

        let mut max_lit_add = 0;
        for clause in op_add.iter().cloned() {
            max_lit_add = max(
                max_lit_add,
                clause.iter().map(|lit| lit.unsigned_abs()).max().unwrap(),
            );
            self.cnf_clauses.push(clause);
        }
        self.number_of_variables = max(self.number_of_variables, max_lit_add);
        self.cnf_clauses = simplify_clauses(self.cnf_clauses.clone());
    }

    /// Adjust the clauses according to the edit, write everything to file, compile the new ddnnf,
    /// replace the whole old ddnnf with the new one, and store the old in the cache.
    fn recompile_everything(&mut self, edit: IncrementalEdit) {
        let (edit_lits, (op_add, op_rmv)) = &edit;
        debug_assert!(!edit_lits.is_empty());
        debug_assert!(!op_add.is_empty() || !op_rmv.is_empty());

        self.adjust_intern_cnf(op_add, op_rmv);

        // write clauses to file
        let mut inter_cnf = tempfile::Builder::new().suffix(".cnf").tempfile().unwrap();
        inter_cnf
            .write_all(
                format!(
                    "p cnf {} {}\n",
                    self.number_of_variables,
                    self.cnf_clauses.len()
                )
                .as_bytes(),
            )
            .unwrap();
        for clause in self.cnf_clauses.iter() {
            inter_cnf
                .write_all(format!("{} 0\n", format_vec(clause.iter())).as_bytes())
                .unwrap();
        }

        let sup = build_ddnnf(inter_cnf.path().to_str().unwrap(), None);
        self.switch_sub_dag(edit, Either::Right(sup.inter_graph));

        self.rebuild(None);
    }

    pub fn apply_incremental_edit(&mut self, edit: IncrementalEdit) -> IncrementalStrategy {
        let (edit_lits, (op_add, op_rmv)) = &edit;
        if op_add.is_empty() && op_rmv.is_empty() {
            return IncrementalStrategy::Tautology;
        }

        if let Some(((edit_lits_cache, (op_add_cache, op_rm_cache)), replacement)) =
            self.cache.find_and_remove(|(item, _)| {
                let (edit_lits_entry, (op_add_entry, op_rmv_entry)) = item;
                if edit_lits_entry != edit_lits {
                    return false;
                }
                fn vec_value_eq(vec_fst: &[Vec<i32>], vec_snd: &[Vec<i32>]) -> bool {
                    fn vec_to_hashset(vec: &[i32]) -> HashSet<i32> {
                        vec.iter().cloned().collect()
                    }
                    vec_fst.iter().all(|inner_list| {
                        vec_snd.iter().any(|other_inner_list| {
                            vec_to_hashset(inner_list) == vec_to_hashset(other_inner_list)
                        })
                    })
                }
                // check for identical values cross (ClauseApplication is inverted when finding a suiting cached state)
                vec_value_eq(op_add, op_rmv_entry) && vec_value_eq(op_rmv, op_add_entry)
            })
        {
            self.switch_sub_dag((edit_lits_cache, (op_rm_cache, op_add_cache)), replacement);
            return IncrementalStrategy::Undo;
        }

        let mut adds_new_feature = false;
        if !op_add.is_empty() {
            let clause_max_var = op_add
                .iter()
                .flatten()
                .map(|f| f.unsigned_abs())
                .max()
                .unwrap();

            if self.number_of_variables < clause_max_var {
                adds_new_feature = true;
            }
        }

        // single clause should be added that only contains one feature
        if op_add.len() == 1 && !adds_new_feature && op_add[0].len() == 1 {
            self.add_unit_clause(op_add[0][0]);
            return IncrementalStrategy::UnitClause;
        }

        let mut inter_cnf = tempfile::Builder::new().suffix(".cnf").tempfile().unwrap();
        let (cnf, adjusted_replace, re_indices) = self.transform_to_cnf_from_starting_cnf(&edit);

        if cnf.is_empty() {
            return if adjusted_replace == self.root {
                self.recompile_everything(edit);
                IncrementalStrategy::Recompile
            } else {
                IncrementalStrategy::Tautology
            };
        }

        // persist CNF
        let cnf_flat = cnf.join("");
        inter_cnf.write_all(cnf_flat.as_bytes()).unwrap();

        let mut sup = build_ddnnf(inter_cnf.path().to_str().unwrap(), None);

        // reindex the new sub DAG
        let mut dfs = DfsPostOrder::new(&sup.inter_graph.graph, sup.inter_graph.root);
        while let Some(nx) = dfs.next(&sup.inter_graph.graph) {
            if let TId::Literal { feature } = sup.inter_graph.graph[nx] {
                let re_lit = *re_indices.get(&(feature.unsigned_abs())).unwrap() as i32;
                sup.inter_graph.graph[nx] = TId::Literal {
                    feature: re_lit * feature.signum(),
                };
            }
        }

        self.switch_sub_dag(
            edit,
            Either::Left((
                adjusted_replace,
                sup.inter_graph.graph,
                sup.inter_graph.root,
            )),
        );

        IncrementalStrategy::SubDAGReplacement
    }

    fn switch_sub_dag(
        &mut self,
        edit: IncrementalEdit,
        replacement: Either<SubDAGInfo, IntermediateGraph>,
    ) {
        match replacement {
            Either::Left((switching_node, other, other_root)) => {
                // extract sub DAG that we want to cache
                let mut rep_ddnnf_graph = DdnnfGraph::new();
                let mut dfs = DfsPostOrder::new(&self.graph, switching_node);
                let mut rep_cache = HashMap::new();
                let mut lit_cache = HashMap::new();
                while let Some(nx) = dfs.next(&self.graph) {
                    let new_nx = match self.graph[nx] {
                        // map literal nodes to already existing nodes
                        TId::Literal { feature } => match lit_cache.get(&feature) {
                            Some(self_nx) => *self_nx,
                            None => {
                                let lit_nx = rep_ddnnf_graph.add_node(TId::Literal { feature });
                                lit_cache.insert(feature, lit_nx);
                                lit_nx
                            }
                        },
                        // everything else can just be added
                        _ => rep_ddnnf_graph.add_node(self.graph[nx]),
                    };
                    rep_cache.insert(nx, new_nx);

                    let mut children = self.graph.neighbors_directed(nx, Outgoing).detach();
                    while let Some(child_nx) = children.next_node(&self.graph) {
                        rep_ddnnf_graph.add_edge(new_nx, *rep_cache.get(&child_nx).unwrap(), ());
                    }

                    // remove all edges and nodes of the current sub DAG
                    let mut children = self.graph.neighbors_directed(nx, Outgoing).detach();
                    while let Some((child_edge, child_node)) = children.next(&self.graph) {
                        self.graph.remove_edge(child_edge);
                        if self.graph.neighbors_directed(nx, Incoming).count() == 0 {
                            self.graph.remove_node(child_node);
                        }
                    }
                }
                self.cache.retain_push((
                    edit,
                    Either::Left((
                        switching_node,
                        rep_ddnnf_graph,
                        *rep_cache.get(&switching_node).unwrap(),
                    )),
                ));

                // add edges and nodes of the replacement
                let mut dfs = DfsPostOrder::new(&other, other_root);
                let mut cache = HashMap::new();
                while let Some(nx) = dfs.next(&other) {
                    let new_nx = match other[nx] {
                        // map literal nodes to already existing nodes
                        TId::Literal { feature } => match self.literals_nx.get(&feature) {
                            Some(self_nx) => *self_nx,
                            None => self.graph.add_node(TId::Literal { feature }),
                        },
                        // everything else can just be added
                        _ => self.graph.add_node(other[nx]),
                    };
                    cache.insert(nx, new_nx);

                    let mut children = other.neighbors_directed(nx, Outgoing).detach();
                    while let Some(child_nx) = children.next_node(&other) {
                        self.graph
                            .add_edge(new_nx, *cache.get(&child_nx).unwrap(), ());
                    }
                }

                // replace the reference to the starting node with the new subgraph
                let new_sub_root = *cache.get(&other_root).unwrap();
                self.graph.add_edge(switching_node, new_sub_root, ());

                // add the literal_children of the newly created ddnnf
                extend_literal_diffs(&self.graph, &mut self.literal_children, new_sub_root);
            }
            Either::Right(other_ig) => {
                self.cache.retain_push((edit, Either::Right(self.clone())));
                self.move_inter_graph(other_ig)
            }
        }
    }

    /// Adds the necessary reference / removes it to extend a dDNNF by a unit clause
    fn add_unit_clause(&mut self, feature: i32) {
        self.adjust_intern_cnf(&[vec![feature]], &[]);
        if feature.unsigned_abs() <= self.number_of_variables {
            // it's an old feature
            if let Some(&node) = self.literals_nx.get(&-feature) {
                // Add a unit clause by removing the existence of its contradiction
                // Removing it is the same as replacing it by a FALSE node.
                // We resolve the FALSE node immidiatly.
                let mut need_to_be_removed = vec![node];

                while !need_to_be_removed.is_empty() {
                    let mut remove_in_next_step = Vec::new();
                    for node in need_to_be_removed {
                        let mut parents = self.graph.neighbors_directed(node, Incoming).detach();
                        while let Some(parent) = parents.next_node(&self.graph) {
                            if self.graph[parent] == TId::And {
                                remove_in_next_step.push(parent);
                            }
                        }
                        self.graph.remove_node(node);
                    }
                    need_to_be_removed = remove_in_next_step;
                }
            } // If its contradiction does not exist, we don't have to do anything
        } else {
            // one / multiple new features depending on feature number
            // Example: If the current #variables = 42 and the new feature number is 50,
            // we have to add all the features from 43 to 49 (including) as optional features (or triangle at root)
            while self.number_of_variables != feature.unsigned_abs() {
                self.number_of_variables += 1;

                let new_or = self.graph.add_node(TId::Or);
                self.graph.add_edge(self.root, new_or, ());

                let new_positive_lit = self.graph.add_node(TId::Literal {
                    feature: self.number_of_variables as i32,
                });
                let new_negative_lit = self.graph.add_node(TId::Literal {
                    feature: -(self.number_of_variables as i32),
                });
                self.graph.add_edge(new_or, new_positive_lit, ());
                self.graph.add_edge(new_or, new_negative_lit, ());
            }

            // Add the actual new feature
            let new_feature = self.graph.add_node(TId::Literal { feature });
            self.graph.add_edge(self.root, new_feature, ());
            self.number_of_variables = feature.unsigned_abs();
        }
    }

    pub fn get_partial_graph_til_depth(&self, start: NodeIndex, depth: i32) -> IntermediateGraph {
        let mut sub_ig = self.clone();
        let mut distance_mapping = HashMap::new();

        let mut dfs = Dfs::new(&self.graph, start);
        while let Some(nx) = dfs.next(&self.graph) {
            let mut highest_parent_depth = 0;

            let mut parents = self.graph.neighbors_directed(nx, Incoming).detach();
            while let Some(node) = parents.next_node(&self.graph) {
                highest_parent_depth = max(
                    highest_parent_depth,
                    *distance_mapping.get(&node).unwrap_or(&0),
                );
            }

            distance_mapping.insert(nx, highest_parent_depth + 1);
        }

        // remove all nodes from the clone that are not within the depth limit
        let mut dfs = Dfs::new(&self.graph, self.root);
        while let Some(nx) = dfs.next(&self.graph) {
            let remove_node = match distance_mapping.get(&nx) {
                Some(&nx_depth) => nx_depth > depth, // depth does not fit
                None => true, // no entry -> node is neither parent nor any child of start
            };

            if remove_node {
                let mut parents = sub_ig.graph.neighbors_directed(nx, Incoming).detach();
                while let Some(edge_inc) = parents.next_edge(&sub_ig.graph) {
                    sub_ig.graph.remove_edge(edge_inc);
                }

                let mut children = sub_ig.graph.neighbors_directed(nx, Outgoing).detach();
                while let Some(edge_out) = children.next_edge(&sub_ig.graph) {
                    sub_ig.graph.remove_edge(edge_out);
                }

                sub_ig.graph.remove_node(nx);
            }
        }

        sub_ig.root = start;
        sub_ig
    }
}

#[cfg(feature = "d4")]
#[cfg(test)]
mod test {
    use num::BigInt;
    use rand::rngs::StdRng;
    use std::{
        collections::HashSet,
        fs::{self, File},
        io::Write,
    };

    use crate::{
        parser::{
            build_ddnnf,
            from_cnf::{add_clause_cnf, get_all_clauses_cnf, remove_clause_cnf},
            intermediate_representation::ClauseApplication,
        },
        Ddnnf,
    };

    use super::{IncrementalEdit, IncrementalStrategy};

    fn build_edit_from_single(clause: Vec<i32>, app: ClauseApplication) -> IncrementalEdit {
        build_edit(vec![(clause, app)])
    }

    fn build_edit(clause_apps: Vec<(Vec<i32>, ClauseApplication)>) -> IncrementalEdit {
        let mut edit_lits = HashSet::new();
        let mut op_add = Vec::new();
        let mut op_rmv = Vec::new();
        for (clause, application) in clause_apps {
            if clause.is_empty() {
                continue;
            }
            edit_lits.extend(clause.iter());
            match application {
                ClauseApplication::Add => op_add.push(clause),
                ClauseApplication::Remove => op_rmv.push(clause),
            }
        }
        (edit_lits, (op_add, op_rmv))
    }

    fn wrap_build_edit(clause: &[i32]) -> IncrementalEdit {
        (
            HashSet::from_iter(clause.iter().cloned()),
            (Vec::new(), Vec::new()),
        )
    }

    #[test]
    fn closest_unsplittable_and() {
        let bridge_comparison = |mut ddnnf: Ddnnf,
                                 input: Vec<Vec<i32>>,
                                 output: Vec<Option<Vec<i32>>>| {
            for (index, inp) in input.iter().enumerate() {
                match ddnnf
                    .inter_graph
                    .closest_unsplitable_bridge(&wrap_build_edit(inp))
                {
                    Some((_replace_and_node, literals)) => {
                        let mut literals_as_vec = HashSet::<_>::from_iter(literals.iter().copied())
                            .into_iter()
                            .collect::<Vec<i32>>();

                        literals_as_vec.sort();
                        assert_eq!(output[index].clone().unwrap(), literals_as_vec);
                    }
                    None => {
                        assert!(output[index].is_none());
                    }
                }
            }
        };

        let ddnnf_vp9 = build_ddnnf("tests/data/VP9.cnf", Some(42));
        let input_vp9 = vec![
            vec![],
            vec![4],
            vec![5],
            vec![4, 5],
            vec![42],
            vec![-5],
            vec![-8],
        ];
        let output_vp9 = vec![
            None,
            Some(vec![-5, -4, -3, 3, 4, 5]),
            Some(vec![-5, -4, -3, 3, 4, 5]),
            Some(vec![-5, -4, -3, 3, 4, 5]),
            Some(vec![-42, -41, 41, 42]),
            Some(vec![-5, -4, -3, 3, 4, 5]),
            Some(vec![-9, -8, -7, 7, 8, 9]),
        ];
        bridge_comparison(ddnnf_vp9, input_vp9, output_vp9);
    }

    fn check_for_cardinality_correctness(path: &str, break_point: usize) {
        let temp_file = tempfile::Builder::new().suffix(".cnf").tempfile().unwrap();
        let temp_file_path_buf = temp_file.path().to_path_buf();
        let temp_file_path = temp_file_path_buf.to_str().unwrap();
        fs::copy(path, temp_file_path).unwrap();

        let mut clauses = get_all_clauses_cnf(temp_file_path);
        use rand::SeedableRng;
        let mut rng: StdRng = SeedableRng::seed_from_u64(42);
        use rand::prelude::SliceRandom;
        clauses.shuffle(&mut rng);

        let mut ddnnf_w = build_ddnnf(temp_file_path, None);
        let mut card_of_features_w: Vec<BigInt> = Vec::new();
        for feature in 1_i32..ddnnf_w.number_of_variables as i32 {
            card_of_features_w.push(ddnnf_w.execute_query(&[feature]));
        }

        for (index, clause) in clauses.into_iter().enumerate() {
            remove_clause_cnf(temp_file_path, &clause, None);
            if index >= break_point {
                break;
            }

            /************************** Adding the clause **************************/
            let mut ddnnf_inc = build_ddnnf(temp_file_path, None);
            let mut card_of_features_wo = Vec::new();
            for feature in 1_i32..ddnnf_w.number_of_variables as i32 {
                card_of_features_wo.push(ddnnf_w.execute_query(&[feature]));
            }

            ddnnf_inc
                .inter_graph
                .apply_incremental_edit(build_edit_from_single(
                    clause.clone(),
                    ClauseApplication::Add,
                ));
            ddnnf_inc.rebuild();

            assert_eq!(ddnnf_inc.rc(), ddnnf_w.rc());
            for feature in 1_i32..ddnnf_w.number_of_variables as i32 {
                assert_eq!(
                    ddnnf_inc.execute_query(&[feature]),
                    card_of_features_w[feature as usize - 1]
                );
            }

            /************************** Removing the clause **************************/
            let mut ddnnf_w_c = ddnnf_inc.clone();
            ddnnf_w_c
                .inter_graph
                .apply_incremental_edit(build_edit_from_single(
                    clause.clone(),
                    ClauseApplication::Remove,
                ));
            ddnnf_inc.rebuild();

            assert_eq!(ddnnf_w_c.rc(), ddnnf_w.rc());
            for feature in 1_i32..ddnnf_w.number_of_variables as i32 {
                assert_eq!(
                    ddnnf_w_c.execute_query(&[feature]),
                    card_of_features_wo[feature as usize - 1]
                );
            }

            add_clause_cnf(temp_file_path, &clause);
        }
    }

    #[test]
    fn transform_to_cnf_from_starting_cnf_clauses_small_models() {
        let ddnnf_file_paths = vec![
            "tests/data/VP9.cnf",
            "tests/data/X264.cnf",
            "tests/data/HiPAcc.cnf",
        ];

        for path in ddnnf_file_paths {
            check_for_cardinality_correctness(path, usize::MAX);
        }
    }

    #[test]
    fn transform_to_cnf_from_starting_cnf_clauses_medium_models() {
        let ddnnf_file_paths = vec!["tests/data/kc_axTLS.cnf", "tests/data/toybox.cnf"];

        for path in ddnnf_file_paths {
            check_for_cardinality_correctness(path, 100);
        }
    }

    #[test]
    fn transform_to_cnf_from_starting_cnf_clauses_big_models_auto1() {
        check_for_cardinality_correctness("tests/data/auto1.cnf", 10);
    }

    fn assert_cardinalities(ddnnf: &mut Ddnnf, temp_file_path: &str) {
        let mut ddnnf_control = build_ddnnf(temp_file_path, None);
        for feature in 1_i32..ddnnf.number_of_variables as i32 {
            assert_eq!(
                ddnnf_control.execute_query(&[feature]),
                ddnnf.execute_query(&[feature])
            );
        }
    }

    #[test]
    fn undo_clause_remove() {
        let ddnnf_file_paths = vec![
            "tests/data/VP9.cnf",
            "tests/data/X264.cnf",
            "tests/data/HiPAcc.cnf",
        ];

        for path in ddnnf_file_paths {
            println!("***************** Model {path} *****************");

            let temp_file = tempfile::Builder::new().suffix(".cnf").tempfile().unwrap();
            let temp_file_path_buf = temp_file.path().to_path_buf();
            let temp_file_path = temp_file_path_buf.to_str().unwrap();
            fs::copy(path, temp_file_path).unwrap();

            let mut ddnnf = build_ddnnf(temp_file_path, None);
            let mut expected_card_of_features: Vec<BigInt> = Vec::new();
            for feature in 1_i32..ddnnf.number_of_variables as i32 {
                expected_card_of_features.push(ddnnf.execute_query(&[feature]));
            }

            for clause in get_all_clauses_cnf(temp_file_path).into_iter() {
                if clause.len() == 1 {
                    continue;
                } // skip unit clause
                let strat_remove =
                    ddnnf
                        .inter_graph
                        .apply_incremental_edit(build_edit_from_single(
                            clause.clone(),
                            ClauseApplication::Remove,
                        ));
                let strat_add = ddnnf
                    .inter_graph
                    .apply_incremental_edit(build_edit_from_single(
                        clause.clone(),
                        ClauseApplication::Add,
                    ));
                if strat_remove == IncrementalStrategy::SubDAGReplacement {
                    assert_eq!(IncrementalStrategy::Undo, strat_add);
                    assert_cardinalities(&mut ddnnf, temp_file_path);
                }
            }
        }
    }

    #[test]
    fn undo_clause_add() {
        let ddnnf_file_paths = vec![
            "tests/data/VP9.cnf",
            "tests/data/X264.cnf",
            "tests/data/HiPAcc.cnf",
        ];

        for path in ddnnf_file_paths {
            println!("***************** Model {path} *****************");

            let temp_file = tempfile::Builder::new().suffix(".cnf").tempfile().unwrap();
            let temp_file_path_buf = temp_file.path().to_path_buf();
            let temp_file_path = temp_file_path_buf.to_str().unwrap();
            fs::copy(path, temp_file_path).unwrap();

            for clause in get_all_clauses_cnf(temp_file_path).into_iter() {
                if clause.len() == 1 {
                    continue;
                } // skip unit clause
                remove_clause_cnf(temp_file_path, &clause, None);

                let mut ddnnf = build_ddnnf(temp_file_path, None);
                ddnnf
                    .inter_graph
                    .apply_incremental_edit(build_edit_from_single(
                        clause.clone(),
                        ClauseApplication::Add,
                    ));
                assert_eq!(
                    IncrementalStrategy::Undo,
                    ddnnf
                        .inter_graph
                        .apply_incremental_edit(build_edit_from_single(
                            clause.clone(),
                            ClauseApplication::Remove
                        ))
                );
                assert_cardinalities(&mut ddnnf, temp_file_path);

                add_clause_cnf(temp_file_path, &clause);
            }
        }
    }

    #[test]
    fn incremental_adding_clause() {
        let ddnnf_file_paths = vec![(
            "tests/data/VP9.cnf",
            "tests/data/VP9_wo_-4-5.cnf",
            42,
            vec![-4, -5],
        )];

        for (path_w_clause, path_wo_clause, features, clause) in ddnnf_file_paths {
            let mut ddnnf_w = build_ddnnf(path_w_clause, Some(features));

            let mut expected_results = Vec::new();
            for f in 1..=features {
                expected_results.push(ddnnf_w.execute_query(&[f as i32]));
            }

            let mut ddnnf_inc = build_ddnnf(path_wo_clause, Some(features));
            ddnnf_inc
                .inter_graph
                .apply_incremental_edit(build_edit_from_single(clause, ClauseApplication::Add));
            ddnnf_inc.rebuild();

            let mut results_after_addition = Vec::new();
            for f in 1..=features {
                results_after_addition.push(ddnnf_inc.execute_query(&[f as i32]));
            }

            assert_eq!(expected_results.len(), results_after_addition.len());
            assert_eq!(expected_results, results_after_addition);
        }
    }

    #[test]
    fn adding_new_features() {
        let mut ddnnf = build_ddnnf("tests/data/small_ex.cnf", None);
        assert_eq!(4, ddnnf.number_of_variables);
        assert_eq!(BigInt::from(4), ddnnf.rc());

        ddnnf
            .inter_graph
            .apply_incremental_edit(build_edit_from_single(vec![6, -6], ClauseApplication::Add));
        ddnnf.rebuild();
        assert_eq!(6, ddnnf.number_of_variables);
        assert_eq!(BigInt::from(16), ddnnf.rc());

        ddnnf
            .inter_graph
            .apply_incremental_edit(build_edit_from_single(vec![-5, -6], ClauseApplication::Add));
        ddnnf.rebuild();
        assert_eq!(6, ddnnf.number_of_variables);
        assert_eq!(BigInt::from(12), ddnnf.rc());
    }

    #[test]
    fn adding_unit_clause() {
        // small example, missing the "1 0" unit clause
        const CNF_PATH: &str = "tests/data/.small_missing_unit.cnf";
        let cnf_flat = "p cnf 4 2\n2 3 0\n-2 -3 0\n";
        let mut cnf_file = File::create(CNF_PATH).unwrap();
        cnf_file.write_all(cnf_flat.as_bytes()).unwrap();

        let mut ddnnf_sb = build_ddnnf("tests/data/small_ex_c2d.nnf", Some(4));
        let mut ddnnf_missing_clause1 = build_ddnnf(CNF_PATH, None);
        let mut ddnnf_missing_clause2 = ddnnf_missing_clause1.clone();

        ddnnf_missing_clause1.inter_graph.add_unit_clause(1); // add the unit clause directly
        ddnnf_missing_clause2
            .inter_graph
            .apply_incremental_edit(build_edit_from_single(vec![1], ClauseApplication::Add)); // indirectly via adding a clause
        ddnnf_missing_clause1.rebuild();
        ddnnf_missing_clause2.rebuild();

        // We have to sort the results because the inner structure does not have to be identical
        let mut sb_enumeration = ddnnf_sb.enumerate(&mut vec![], 1_000).unwrap();
        sb_enumeration.sort();

        let mut ms1_enumeration = ddnnf_missing_clause1.enumerate(&mut vec![], 1_000).unwrap();
        ms1_enumeration.sort();

        let mut ms2_enumeration = ddnnf_missing_clause2.enumerate(&mut vec![], 1_000).unwrap();
        ms2_enumeration.sort();

        // check whether the dDNNFs contain the same configurations
        assert_eq!(sb_enumeration, ms1_enumeration);
        assert_eq!(sb_enumeration, ms2_enumeration);

        fs::remove_file(CNF_PATH).unwrap();
    }

    #[test]
    fn build_model_clause_by_clause() {
        let ddnnf_file_paths = vec![
            "tests/data/VP9.cnf",
            "tests/data/X264.cnf",
            "tests/data/HiPAcc.cnf",
        ];

        for ddnnf_file_path in ddnnf_file_paths {
            let mut temp_file = tempfile::Builder::new().suffix(".cnf").tempfile().unwrap();
            let temp_file_path_buf = temp_file.path().to_path_buf();
            let temp_file_path = temp_file_path_buf.to_str().unwrap();
            temp_file.write_all("p cnf 0 0\n".as_bytes()).unwrap();

            fn check_empirical_equivalence(ddnnf_target: &mut Ddnnf, other: &mut Ddnnf) {
                for feature in 1_i32..ddnnf_target.number_of_variables as i32 {
                    assert_eq!(
                        ddnnf_target.execute_query(&[feature]),
                        other.execute_query(&[feature])
                    );
                }
            }

            let mut ddnnf_inc = Ddnnf::default();
            let mut ddnnf_rec;
            for clause in get_all_clauses_cnf(ddnnf_file_path) {
                add_clause_cnf(temp_file_path, &clause);

                ddnnf_inc
                    .inter_graph
                    .apply_incremental_edit(build_edit_from_single(clause, ClauseApplication::Add));
                ddnnf_inc.rebuild();
                ddnnf_rec = build_ddnnf(temp_file_path, None);

                check_empirical_equivalence(&mut ddnnf_rec, &mut ddnnf_inc);
            }
        }
    }
}
