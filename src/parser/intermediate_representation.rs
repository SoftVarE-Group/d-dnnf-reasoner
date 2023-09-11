use std::{
    cmp::max,
    collections::{HashMap, HashSet},
    fs::File,
    io::Write,
    panic,
    time::Instant,
};

use itertools::Itertools;
use petgraph::{
    algo::is_cyclic_directed,
    stable_graph::{NodeIndex, StableGraph},
    visit::{Dfs, DfsPostOrder, NodeIndexable},
    Direction::{Incoming, Outgoing},
};

use crate::{
    c2d_lexer::TId,
    parser::{
        build_ddnnf, extend_literal_diffs,
        from_cnf::{get_all_clauses_cnf, simplify_clauses},
        persisting::write_as_mermaid_md,
        util::format_vec,
    },
    Ddnnf, Node, NodeType,
};

use super::{calc_and_count, calc_or_count, from_cnf::apply_decisions};

/// The IntermediateGraph enables us to modify the dDNNF. The structure of a vector of nodes does not allow
/// for that because deleting or removing nodes would mess up the indices.
#[derive(Clone, Debug, Default)]
pub struct IntermediateGraph {
    graph: StableGraph<TId, Option<Vec<i32>>>,
    pub root: NodeIndex,
    number_of_variables: u32,
    literals_nx: HashMap<i32, NodeIndex>,
    literal_children: HashMap<NodeIndex, HashSet<i32>>,
    cnf_clauses: Vec<Vec<i32>>,
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
    /// becuase almost everything is affected by the new clause
    Recompile,
}

const DEBUG: bool = false;

impl IntermediateGraph {
    /// Creates a new IntermediateGraph
    pub fn new(
        graph: StableGraph<TId, Option<Vec<i32>>>,
        root: NodeIndex,
        number_of_variables: u32,
        literals_nx: HashMap<i32, NodeIndex>,
        literal_children: HashMap<NodeIndex, HashSet<i32>>,
        cnf_path: Option<&str>,
    ) -> IntermediateGraph {
        debug_assert!(!is_cyclic_directed(&graph));
        let mut inter_graph = IntermediateGraph {
            graph,
            root,
            number_of_variables,
            literals_nx,
            literal_children,
            cnf_clauses: match cnf_path {
                Some(path) => get_all_clauses_cnf(path),
                None => Vec::new(),
            },
        };
        inter_graph.add_decision_nodes_fu();
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
        let mut dfs =
            DfsPostOrder::new(&self.graph, alt_root.unwrap_or(self.root));
        let mut nd_to_usize: HashMap<NodeIndex, usize> = HashMap::new();

        let mut parsed_nodes: Vec<Node> =
            Vec::with_capacity(self.graph.node_count());
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
                TId::And => Node::new_and(
                    calc_and_count(&mut parsed_nodes, &neighs),
                    neighs,
                ),
                TId::Or => Node::new_or(
                    0,
                    calc_or_count(&mut parsed_nodes, &neighs),
                    neighs,
                ),
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
        clause: &[i32],
    ) -> Option<(NodeIndex, HashSet<i32>)> {
        if clause.is_empty() {
            return None;
        }
        let mut closest_node =
            (self.root, self.literal_children.get(&self.root).unwrap());

        let mut bridge_end_point = Vec::new();
        for bridge_endpoint in self.find_bridges() {
            match self.graph[bridge_endpoint] {
                TId::And | TId::Or => {
                    let diffs =
                        self.literal_children.get(&bridge_endpoint).unwrap();
                    // we only consider and nodes that include all literals of the clause
                    if clause
                        .iter() 
                        .all(|e| diffs.contains(e) && diffs.contains(&-e))
                    {
                        if closest_node.1.len() > diffs.len() {
                            closest_node = (bridge_endpoint, diffs);
                        }
                    }
                    bridge_end_point.push(bridge_endpoint);
                }
                _ => (),
            }
        }

        let devided_and =
            self.divide_bridge(closest_node.0, &bridge_end_point, clause);
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
        clause: &[i32],
    ) -> NodeIndex {
        if clause.is_empty() {
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
            if clause
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
            self.graph.add_edge(new_node, child, None);

            new_node_lits.extend(self.literal_children.get(&child).unwrap());
        }
        self.graph.add_edge(initial_start, new_node, None);

        // Add child literals of new_and to mapping
        self.literal_children.insert(new_node, new_node_lits);

        new_node
    }

    // Experimental method to use the initial CNF for recompiling
    pub fn transform_to_cnf_from_starting_cnf(
        &mut self,
        clause: Vec<i32>,
    ) -> (Vec<String>, NodeIndex, HashMap<u32, u32>) {
        if DEBUG {
            println!("starting 'transform_to_cnf_from_starting_cnf' with clause: {:?}", clause);
        }

        if self.cnf_clauses.is_empty() {
            return (Vec::new(), NodeIndex::new(0), HashMap::new());
        }

        // 1) Find the closest node and the decisions to that point
        let (closest_node, relevant_literals) =
            match self.closest_unsplitable_bridge(&clause) {
                Some((nx, lits)) => (nx, lits.clone()),
                None => (self.root, HashSet::new()), // Just do the whole CNF without any adjustments
            };

        if DEBUG {
            println!(
                "closest node: {:?}, and its relevant literals: {:?}",
                closest_node, relevant_literals
            );
        }

        // Recompile the whole dDNNF if at least 80% are children of the closest_node we could replace
        if relevant_literals.len() as f32 / 2.0
            > self.number_of_variables as f32 * 0.8
        {
            return (Vec::new(), self.root, HashMap::new());
        }

        let mut accumlated_decisions =
            self.get_decisions_target_nx(closest_node);

        // Add unit clauses. Those decisions are implicit in the dDNNF.
        // Hence, we have to search for them
        self.cnf_clauses.iter().for_each(|cnf_clause| {
            if cnf_clause.len() == 1 {
                accumlated_decisions.insert(cnf_clause[0]);
            }
        });

        if DEBUG {
            println!(
                "lits count: {} lits: {:?}",
                relevant_literals.len(),
                relevant_literals
            );
        }

        // 2) Filter for clauses that contain at least one of the relevant literals
        //    aka the literals that are children of the closest AND node
        let mut relevant_clauses: Vec<Vec<i32>> = if clause.is_empty() {
            self.cnf_clauses.clone()
        } else {
            self.cnf_clauses
                .clone()
                .into_iter()
                .filter(|initial_clause| {
                    initial_clause.iter().any(|elem| {
                        relevant_literals.contains(elem)
                            || relevant_literals.contains(&-elem)
                    })
                })
                .collect_vec()
        };
        relevant_clauses.push(clause.clone());

        if DEBUG {
            let mut removed_clauses = self.cnf_clauses.clone();
            removed_clauses.retain(|clause| !relevant_clauses.contains(clause));
            println!("removed clauses: {:?}", removed_clauses);
        }

        // 2.5 add each feature as optional to account for features that become optional
        //     due to the newly added clause.
        let mut variables = HashSet::new();
        for var in relevant_literals.iter() {
            variables.insert(var.unsigned_abs());
        }

        // 3) Repeatedly apply the summed up decions to the remaining clauses
        (relevant_clauses, accumlated_decisions) =
            apply_decisions(relevant_clauses, accumlated_decisions);

        // Continue 2.5
        let mut red_variables = HashSet::new();
        for clause in relevant_clauses.iter() {
            for variable in clause {
                red_variables.insert(variable.unsigned_abs());
            }
        }

        if DEBUG {
            println!(
                "Potential optional features: {:?}",
                variables.symmetric_difference(&red_variables).cloned()
            );
        }

        if DEBUG {
            println!("decisions: {:?}", accumlated_decisions.clone());
        }

        for variable in variables.symmetric_difference(&red_variables).cloned() {
            let v_i32 = &(variable as i32);
            if !accumlated_decisions.contains(v_i32)
                && !accumlated_decisions.contains(&-v_i32)
                && (relevant_literals.contains(v_i32)
                    || relevant_literals.contains(&-v_i32))
            {
                relevant_clauses
                    .push(vec![variable as i32, -(variable as i32)]);
            }

            if accumlated_decisions.contains(v_i32) {
                relevant_clauses.push(vec![*v_i32]);
            } else if accumlated_decisions.contains(&-v_i32) {
                relevant_clauses.push(vec![-v_i32]);
            }
        }

        if DEBUG {
            println!("relevant clause: {:?}", relevant_clauses);
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

        //if DEBUG { println!("reindex: {:?}", re_index); }

        // write the meta information of the header
        let mut cnf =
            vec![format!("p cnf {} {}\n", index - 1, relevant_clauses.len())];

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

    fn recompile_everything(&mut self, clause: Vec<i32>) {
        const INTER_CNF: &str = ".sub.cnf";

        let mut cnf = vec![format!(
            "p cnf {} {}\n",
            self.number_of_variables,
            self.cnf_clauses.len() + 1
        )];
        for clause in self.cnf_clauses.iter() {
            cnf.push(format!("{} 0\n", format_vec(clause.iter())));
        }
        cnf.push(format!("{} 0\n", format_vec(clause.iter())));

        let cnf_flat = cnf.join("");
        let mut cnf_file = File::create(INTER_CNF).unwrap();
        cnf_file.write_all(cnf_flat.as_bytes()).unwrap();

        let sup = build_ddnnf(INTER_CNF, None);
        self.move_inter_graph(sup.inter_graph);
        self.rebuild(None);
        println!("recompiled everything");
    }

    pub fn add_clause(&mut self, clause: Vec<i32>) -> IncrementalStrategy {
        if clause.is_empty() {
            return IncrementalStrategy::Tautology;
        }
        let clause_max_var =
            clause.iter().map(|f| f.unsigned_abs()).max().unwrap();
        if self.number_of_variables < clause_max_var {
            self.number_of_variables = clause_max_var;
            self.recompile_everything(clause);
            return IncrementalStrategy::Recompile;
        }
        //self.cnf_clauses.push(clause.clone());
        if clause.len() == 1 {
            self.add_unit_clause(clause[0]);
            return IncrementalStrategy::UnitClause;
        }

        let mut _start = Instant::now();
        const INTER_CNF: &str = ".sub.cnf";

        _start = Instant::now();
        let (cnf, adjusted_replace, re_indices) =
            self.transform_to_cnf_from_starting_cnf(clause.clone());

        if cnf.is_empty() {
            if adjusted_replace == self.root {
                self.recompile_everything(clause);
                return IncrementalStrategy::Recompile;
            } else {
                return IncrementalStrategy::Tautology;
            }
        }

        println!("Replace CNF clauses: {}", cnf.len());
        _start = Instant::now();
        // persist CNF
        let cnf_flat = cnf.join("");
        let mut cnf_file = File::create(INTER_CNF).unwrap();
        cnf_file.write_all(cnf_flat.as_bytes()).unwrap();

        if DEBUG {
            let mut ddnnf_remove = Ddnnf::default();
            ddnnf_remove.inter_graph = self.clone();
            ddnnf_remove.rebuild();
            write_as_mermaid_md(
                &ddnnf_remove,
                &[],
                "removed_sub.md",
                Some((adjusted_replace, 1_000)),
            )
            .unwrap();
        }

        let mut parents = self
            .graph
            .neighbors_directed(adjusted_replace, Outgoing)
            .detach();
        while let Some(parent_edge) = parents.next_edge(&self.graph) {
            self.graph.remove_edge(parent_edge);
        }

        if DEBUG {
            let mut ddnnf_after_remove = Ddnnf::default();
            ddnnf_after_remove.inter_graph = self.clone();
            ddnnf_after_remove.rebuild();
            write_as_mermaid_md(&ddnnf_after_remove, &[], "after_rm.md", None)
                .unwrap();
        }

        let mut sup = build_ddnnf(INTER_CNF, None);

        if DEBUG {
            sup.rebuild();
            write_as_mermaid_md(&sup, &[], "sub_before_re.md", None).unwrap();
        }

        // add the new subgraph as additional graph (unconnected to self)
        let mut dfs =
            DfsPostOrder::new(&sup.inter_graph.graph, sup.inter_graph.root);
        while let Some(nx) = dfs.next(&sup.inter_graph.graph) {
            match sup.inter_graph.graph[nx] {
                TId::Literal { feature } => {
                    let re_lit = *re_indices
                        .get(&(feature.unsigned_abs()))
                        .unwrap() as i32;
                    sup.inter_graph.graph[nx] = TId::Literal {
                        feature: re_lit * feature.signum(),
                    };
                }
                _ => (),
            }
        }

        if DEBUG {
            sup.rebuild();
            write_as_mermaid_md(&sup, &[], "sub.md", None).unwrap();
        }

        let mut dfs =
            DfsPostOrder::new(&sup.inter_graph.graph, sup.inter_graph.root);
        let mut cache = HashMap::new();
        while let Some(nx) = dfs.next(&sup.inter_graph.graph) {
            let new_nx = match sup.inter_graph.graph[nx] {
                // reindex the literal nodes and attach them to already existing nodes
                TId::Literal { feature } => {
                    match self.literals_nx.get(&feature) {
                        Some(nx) => *nx,
                        None => self.graph.add_node(TId::Literal { feature }),
                    }
                }
                // everything else can just be added
                _ => self.graph.add_node(sup.inter_graph.graph[nx]),
            };
            cache.insert(nx, new_nx);

            let mut children = sup
                .inter_graph
                .graph
                .neighbors_directed(nx, Outgoing)
                .detach();
            while let Some((child_ex, child_nx)) =
                children.next(&sup.inter_graph.graph)
            {
                self.graph.add_edge(
                    new_nx,
                    *cache.get(&child_nx).unwrap(),
                    sup.inter_graph.graph[child_ex].clone(),
                );
            }
        }

        if DEBUG {
            let mut ddnnf_after_add = Ddnnf::default();
            ddnnf_after_add.inter_graph = self.clone();
            ddnnf_after_add.rebuild();
            write_as_mermaid_md(&ddnnf_after_add, &[], "after_adding.md", None)
                .unwrap();
        }

        // replace the reference to the starting node with the new subgraph
        let new_sub_root = *cache.get(&sup.inter_graph.root).unwrap();
        self.graph.add_edge(adjusted_replace, new_sub_root, None);

        // add the literal_children of the newly created ddnnf
        extend_literal_diffs(
            &self.graph,
            &mut self.literal_children,
            new_sub_root,
        );

        if DEBUG {
            let mut ddnnf_at_end = Ddnnf::default();
            ddnnf_at_end.inter_graph = self.clone();
            ddnnf_at_end.rebuild();
            write_as_mermaid_md(&ddnnf_at_end, &[], "at_end.md", None).unwrap();
        }

        // clean up temp files
        //if Path::new(INTER_CNF).exists() { fs::remove_file(INTER_CNF).unwrap(); }
        //if Path::new(INTER_NNF).exists() { fs::remove_file(INTER_NNF).unwrap(); }

        IncrementalStrategy::SubDAGReplacement
    }

    fn add_decision_nodes_fu(&mut self) {
        let mut dfs = Dfs::new(&self.graph, self.root);
        while let Some(nx) = dfs.next(&self.graph) {
            let mut children =
                self.graph.neighbors_directed(nx, Outgoing).detach();

            match self.graph[nx] {
                TId::Or => {
                    let literals_parent =
                        self.literal_children.get(&nx).unwrap();
                    let mut literals_children = Vec::new();
                    let mut deciding_intersection: HashSet<u32> =
                        literals_parent
                            .iter()
                            .map(|l| l.unsigned_abs())
                            .collect();
                    while let Some((edge, child)) = children.next(&self.graph) {
                        let literals_child =
                            self.literal_children.get(&child).unwrap();

                        let diff = literals_parent
                            .difference(literals_child)
                            .map(|literal| -literal)
                            .collect_vec();

                        let unsigned_diff: HashSet<u32> =
                            diff.iter().map(|l| l.unsigned_abs()).collect();
                        deciding_intersection
                            .retain(|&x| unsigned_diff.contains(&x));
                        literals_children.push((diff, edge));
                    }

                    for (mut diff, edge) in literals_children {
                        diff.retain(|literal| {
                            deciding_intersection
                                .contains(&literal.unsigned_abs())
                        });
                        self.graph[edge] = match &self.graph[edge] {
                            Some(existing_diff) => {
                                diff.extend(existing_diff.iter());
                                Some(diff)
                            }
                            None => Some(diff),
                        }
                    }
                }
                TId::And => {
                    let mut literal_children = Vec::new();
                    let mut non_literal_ex = Vec::new();
                    while let Some((edge, child)) = children.next(&self.graph) {
                        match self.graph[child] {
                            TId::Literal { feature } => {
                                literal_children.push(feature)
                            }
                            _ => non_literal_ex.push(edge),
                        }
                    }

                    for edge in non_literal_ex {
                        let new_decisions = match &self.graph[edge] {
                            Some(dec) => {
                                let mut ndec = dec.clone();
                                ndec.extend(literal_children.iter());
                                Some(ndec)
                            }
                            None => Some(literal_children.clone()),
                        };

                        self.graph[edge] = new_decisions;
                    }
                }
                _ => (),
            }
        }
    }

    /// Adds the necessary reference / removes it to extend a dDNNF by a unit clause
    fn add_unit_clause(&mut self, feature: i32) {
        println!("replaced {:?} by unit clause", feature);
        if feature.unsigned_abs() <= self.number_of_variables {
            // it's an old feature
            match self.literals_nx.get(&-feature) {
                // Add a unit clause by removing the existence of its contradiction
                // Removing it is the same as replacing it by a FALSE node.
                // We resolve the FALSE node immidiatly.
                Some(&node) => {
                    let mut need_to_be_removed = vec![node];

                    while !need_to_be_removed.is_empty() {
                        let mut remove_in_next_step = Vec::new();
                        for node in need_to_be_removed {
                            let mut children = self
                                .graph
                                .neighbors_directed(node, Incoming)
                                .detach();
                            while let Some(parent) =
                                children.next_node(&self.graph)
                            {
                                if self.graph[parent] == TId::And {
                                    remove_in_next_step.push(parent);
                                }
                            }
                            self.graph.remove_node(node);
                        }
                        need_to_be_removed = remove_in_next_step;
                    }
                }
                // If its contradiction does not exist, we don't have to do anything
                None => (),
            }
        } else {
            // one / multiple new features depending on feature number
            // Example: If the current #variables = 42 and the new feature number is 50,
            // we have to add all the features from 43 to 49 (including) as optional features (or triangle at root)
            while self.number_of_variables != feature.unsigned_abs() {
                self.number_of_variables += 1;

                let new_or = self.graph.add_node(TId::Or);
                self.graph.add_edge(self.root, new_or, None);

                let new_positive_lit = self.graph.add_node(TId::Literal {
                    feature: self.number_of_variables as i32,
                });
                let new_negative_lit = self.graph.add_node(TId::Literal {
                    feature: -(self.number_of_variables as i32),
                });
                self.graph.add_edge(new_or, new_positive_lit, None);
                self.graph.add_edge(new_or, new_negative_lit, None);
            }

            // Add the actual new feature
            let new_feature = self.graph.add_node(TId::Literal { feature });
            self.graph.add_edge(self.root, new_feature, None);
            self.number_of_variables = feature.unsigned_abs();
        }
    }

    fn get_decisions_target_nx(&self, target: NodeIndex) -> HashSet<i32> {
        let mut nx_decisions = HashMap::new();
        // the root node can't have any decision yet
        nx_decisions.insert(self.root, Vec::new());

        let mut dfs = Dfs::new(&self.graph, self.root);
        while let Some(nx) = dfs.next(&self.graph) {
            let mut children =
                self.graph.neighbors_directed(nx, Outgoing).detach();
            while let Some((edge, child)) = children.next(&self.graph) {
                let mut current_decisions: Vec<i32> =
                    nx_decisions.get(&nx).unwrap().clone();
                //println!("current before: {:?}", current_decisions);
                let edge_decisions =
                    self.graph[edge].clone().unwrap_or(Vec::new());
                current_decisions.extend(edge_decisions.iter());
                //println!("current after: {:?}", current_decisions);
                nx_decisions.insert(child, current_decisions);
            }
        }

        match nx_decisions.get(&target) {
            Some(decision) => decision.clone().into_iter().collect(),
            None => HashSet::new(),
        }
    }

    pub fn get_partial_graph_til_depth(
        &self,
        start: NodeIndex,
        depth: i32,
    ) -> IntermediateGraph {
        let mut sub_ig = self.clone();
        let mut distance_mapping = HashMap::new();

        let mut dfs = Dfs::new(&self.graph, start);
        while let Some(nx) = dfs.next(&self.graph) {
            let mut highest_parent_depth = 0;

            let mut parents =
                self.graph.neighbors_directed(nx, Incoming).detach();
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
                let mut parents =
                    sub_ig.graph.neighbors_directed(nx, Incoming).detach();
                while let Some(edge_inc) = parents.next_edge(&sub_ig.graph) {
                    sub_ig.graph.remove_edge(edge_inc);
                }

                let mut children =
                    sub_ig.graph.neighbors_directed(nx, Outgoing).detach();
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

#[cfg(test)]
mod test {
    use std::{
        collections::HashSet,
        fs::{self, File},
        io::Write,
    };

    use rand::rngs::StdRng;
    use serial_test::serial;

    use crate::{
        parser::{
            build_ddnnf,
            from_cnf::{
                add_clause_cnf, get_all_clauses_cnf, remove_clause_cnf,
            },
            persisting::write_as_mermaid_md,
        },
        Ddnnf,
    };

    const DEBUG: bool = true;

    #[test]
    #[serial]
    fn closest_unsplittable_and() {
        let bridge_comparison =
            |mut ddnnf: Ddnnf,
             input: Vec<Vec<i32>>,
             output: Vec<Option<Vec<i32>>>| {
                for (index, inp) in input.iter().enumerate() {
                    match ddnnf.inter_graph.closest_unsplitable_bridge(inp) {
                        Some((_replace_and_node, literals)) => {
                            let mut literals_as_vec = HashSet::<_>::from_iter(
                                literals.iter().copied(),
                            )
                            .into_iter()
                            .collect::<Vec<i32>>();

                            literals_as_vec.sort();
                            assert_eq!(
                                output[index].clone().unwrap(),
                                literals_as_vec
                            );
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
        let temp_file =
            tempfile::Builder::new().suffix(".cnf").tempfile().unwrap();
        let temp_file_path_buf = temp_file.path().to_path_buf();
        let temp_file_path = temp_file_path_buf.to_str().unwrap();
        fs::copy(path, temp_file_path).unwrap();

        let mut clauses = get_all_clauses_cnf(temp_file_path);
        use rand::SeedableRng;
        let mut rng: StdRng = SeedableRng::seed_from_u64(42);
        use rand::prelude::SliceRandom;
        clauses.shuffle(&mut rng);

        let mut ddnnf_w = build_ddnnf(temp_file_path, None);
        if DEBUG {
            write_as_mermaid_md(&ddnnf_w, &[], "with.md", None).unwrap();
        }
        let mut card_of_features = Vec::new();
        for feature in 1_i32..ddnnf_w.number_of_variables as i32 {
            card_of_features.push(ddnnf_w.execute_query(&[feature]));
        }

        for (index, clause) in clauses.into_iter().enumerate() {
            if index >= break_point {
                break;
            }
            if DEBUG {
                println!("-------------------------------------------------------------");
                println!("Current clause: {:?}", clause);
            }

            remove_clause_cnf(temp_file_path, &clause, None);
            let mut ddnnf_wo = build_ddnnf(temp_file_path, None);
            if DEBUG {
                write_as_mermaid_md(&ddnnf_wo, &[], "before.md", None).unwrap();
            }
            
            ddnnf_wo.inter_graph.add_clause(clause.clone());
            ddnnf_wo.rebuild();
            if DEBUG {
                write_as_mermaid_md(&ddnnf_wo, &[], "after.md", None).unwrap();
            }
            add_clause_cnf(temp_file_path, &clause);

            assert_eq!(ddnnf_wo.rc(), ddnnf_w.rc());
            for feature in 1_i32..ddnnf_w.number_of_variables as i32 {
                assert_eq!(
                    ddnnf_wo.execute_query(&[feature]),
                    card_of_features[feature as usize - 1]
                );
            }
        }
    }

    #[test]
    #[serial]
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
    #[serial]
    fn transform_to_cnf_from_starting_cnf_clauses_medium_models() {
        let ddnnf_file_paths =
            vec!["tests/data/kc_axTLS.cnf", "tests/data/toybox.cnf"];

        for path in ddnnf_file_paths {
            check_for_cardinality_correctness(path, 100);
        }
    }

    #[test]
    #[serial]
    fn transform_to_cnf_from_starting_cnf_clauses_big_models_auto1() {
        check_for_cardinality_correctness("tests/data/auto1.cnf", 20);
    }

    #[test]
    #[serial]
    fn incremental_adding_clause() {
        let ddnnf_file_paths = vec![(
            "tests/data/VP9.cnf",
            "tests/data/VP9_wo_-4-5.cnf",
            42,
            vec![-4, -5],
        )];

        for (path_w_clause, path_wo_clause, features, clause) in
            ddnnf_file_paths
        {
            let mut ddnnf_w = build_ddnnf(path_w_clause, Some(features));

            let mut expected_results = Vec::new();
            for f in 1..=features {
                expected_results.push(ddnnf_w.execute_query(&[f as i32]));
            }

            let mut ddnnf_wo = build_ddnnf(path_wo_clause, Some(features));
            if DEBUG {
                write_as_mermaid_md(&mut ddnnf_wo, &[], "before.md", None)
                    .unwrap();
            }
            ddnnf_wo.inter_graph.add_clause(clause);
            ddnnf_wo.rebuild();
            if DEBUG {
                write_as_mermaid_md(&mut ddnnf_wo, &[], "after.md", None)
                    .unwrap();
            }

            let mut results_after_addition = Vec::new();
            for f in 1..=features {
                results_after_addition
                    .push(ddnnf_wo.execute_query(&[f as i32]));
            }

            assert_eq!(expected_results.len(), results_after_addition.len());
            assert_eq!(expected_results, results_after_addition);
        }
    }

    #[test]
    #[serial]
    fn adding_new_features() {
        let mut ddnnf = build_ddnnf("tests/data/small_ex.cnf", None);
        assert_eq!(4, ddnnf.number_of_variables);
        assert_eq!(4, ddnnf.rc());

        ddnnf.inter_graph.add_clause(vec![6, -6]);
        ddnnf.rebuild();
        assert_eq!(6, ddnnf.number_of_variables);
        assert_eq!(16, ddnnf.rc());

        ddnnf.inter_graph.add_clause(vec![-5, -6]);
        ddnnf.rebuild();
        assert_eq!(6, ddnnf.number_of_variables);
        assert_eq!(12, ddnnf.rc());
    }

    #[test]
    #[serial]
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
        ddnnf_missing_clause2.inter_graph.add_clause(vec![1]); // indirectly via adding a clause
        ddnnf_missing_clause1.rebuild();
        ddnnf_missing_clause2.rebuild();

        // We have to sort the results because the inner structure does not have to be identical
        let mut sb_enumeration =
            ddnnf_sb.enumerate(&mut vec![], 1_000).unwrap();
        sb_enumeration.sort();

        let mut ms1_enumeration =
            ddnnf_missing_clause1.enumerate(&mut vec![], 1_000).unwrap();
        ms1_enumeration.sort();

        let mut ms2_enumeration =
            ddnnf_missing_clause2.enumerate(&mut vec![], 1_000).unwrap();
        ms2_enumeration.sort();

        // check whether the dDNNFs contain the same configurations
        assert_eq!(sb_enumeration, ms1_enumeration);
        assert_eq!(sb_enumeration, ms2_enumeration);

        fs::remove_file(CNF_PATH).unwrap();
    }
}
