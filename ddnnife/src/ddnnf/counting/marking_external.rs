use crate::{Ddnnf, NodeType, int_hash::IntMap};
use bitvec::{bitvec, slice::BitSlice, vec::BitVec};
use num::{BigUint, Zero};

/// Counts for a subset of all nodes.
type CountMap = IntMap<usize, BigUint>;

/// Helper structure for marking nodes when using the marking algorithm.
struct MarkerMap {
    /// Markers for each node, indicating whether it should be recomputed.
    pub markers: BitVec,
    /// For easier access, the node indices to be recomputed.
    pub nodes_to_recount: Vec<usize>,
}

impl MarkerMap {
    /// Creates a new marking map by marking all nodes and their parent nodes recursively.
    pub fn new(ddnnf: &Ddnnf, nodes: &[usize]) -> Self {
        let mut markers = bitvec![0; ddnnf.nodes.len()];
        let mut to_mark = Vec::from(nodes);

        while let Some(i) = to_mark.pop() {
            if markers[i] {
                continue;
            }

            markers.set(i, true);
            to_mark.extend(ddnnf.nodes[i].parents.iter());
        }

        Self {
            nodes_to_recount: markers.iter_ones().collect(),
            markers,
        }
    }
}

impl Ddnnf {
    pub fn operate_on_partial_config_marker_external(
        &self,
        assumptions: &[i32],
        operation: fn(&Self, usize, &BitSlice, &mut CountMap),
    ) -> BigUint {
        // Exit early in case of invalid d-DNNF or assumptions.
        if self.nodes.is_empty() || self.query_is_not_sat(assumptions) {
            return BigUint::ZERO;
        }

        // Simplify the assumptions by using knowledge about core variables.
        let assumptions = self.reduce_query(assumptions);

        // Find all the literals to be marked.
        let indices: Vec<usize> = self.indices_of_inverted_assumptions(&assumptions);

        if indices.is_empty() {
            return self.rc();
        }

        // Mark the literals.
        let marker_map = MarkerMap::new(self, &indices);

        // Re-calculate each marked node.
        let mut counts: CountMap = CountMap::default();
        marker_map
            .nodes_to_recount
            .iter()
            .for_each(|&node_index| operation(self, node_index, &marker_map.markers, &mut counts));

        // Return the new count of the root node.
        let root_node = self.nodes.len() - 1;
        counts
            .get(&root_node)
            .expect("Failed to access root count")
            .clone()
    }

    pub fn calc_count_marked_external(
        &self,
        node_index: usize,
        markers: &BitSlice,
        counts: &mut CountMap,
    ) {
        counts.insert(
            node_index,
            match &self.nodes[node_index].ntype {
                NodeType::And { children } => {
                    let marked_children = children
                        .iter()
                        .filter(|&&child_index| markers[child_index])
                        .collect::<Vec<&usize>>();

                    if marked_children.len() <= children.len() / 2 {
                        marked_children.iter().fold(
                            self.nodes[node_index].count.clone(),
                            |mut acc, &child_index| {
                                let child = &self.nodes[*child_index];

                                if !child.count.is_zero() {
                                    acc /= &child.count;
                                }

                                acc *= &counts[child_index];
                                acc
                            },
                        )
                    } else {
                        children
                            .iter()
                            .map(|child_index| {
                                if markers[*child_index] {
                                    &counts[child_index]
                                } else {
                                    &self.nodes[*child_index].count
                                }
                            })
                            .product()
                    }
                }
                NodeType::Or { children } => children
                    .iter()
                    .map(|child_index| {
                        if markers[*child_index] {
                            &counts[child_index]
                        } else {
                            &self.nodes[*child_index].count
                        }
                    })
                    .sum(),
                NodeType::Literal { .. } => BigUint::ZERO,
            },
        );
    }

    /// Finds the node indices of the inverse assumption literals.
    fn indices_of_inverted_assumptions(&self, assumptions: &[i32]) -> Vec<usize> {
        assumptions
            .iter()
            .filter_map(|literal| self.literals.get(&-literal))
            .copied()
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::MarkerMap;
    use crate::Ddnnf;
    use bitvec::bitvec;
    use std::path::Path;

    #[test]
    fn marking_nodes() {
        let ddnnf = Ddnnf::from_file(Path::new("tests/data/small_ex_c2d.nnf"), None);

        let map = MarkerMap::new(&ddnnf, &[0]);
        let expected_nodes = vec![0, 11];
        assert_eq!(map.nodes_to_recount, expected_nodes);

        let mut expected_markers = bitvec![0; ddnnf.nodes.len()];
        expected_nodes
            .iter()
            .for_each(|&node| expected_markers.set(node, true));

        assert_eq!(map.markers, expected_markers);

        let map = MarkerMap::new(&ddnnf, &[2, 4]);
        let expected_nodes = vec![2, 3, 4, 6, 7, 11];
        assert_eq!(&map.nodes_to_recount, &expected_nodes);

        let mut expected_markers = bitvec![0; ddnnf.nodes.len()];
        expected_nodes
            .iter()
            .for_each(|&node| expected_markers.set(node, true));

        assert_eq!(&map.markers, &expected_markers);
    }
}
