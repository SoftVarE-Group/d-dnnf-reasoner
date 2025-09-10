use crate::NodeType::*;
use crate::ddnnf::anomalies::t_wise_sampling::Config;
use crate::ddnnf::extended_ddnnf::ExtendedDdnnf;
use bimap::BiHashMap;
use itertools::Itertools;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};
use std::ops::Neg;

#[derive(Clone, PartialEq, Debug)]
pub struct OptimalConfig {
    pub(crate) config: Config,
    value: f64,
    n_literals: usize,
}

impl Eq for OptimalConfig {}

impl Hash for OptimalConfig {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.config.hash(state)
    }
}

impl PartialOrd<Self> for OptimalConfig {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OptimalConfig {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.total_cmp(&other.value)
    }
}

impl OptimalConfig {
    pub fn empty(number_of_variables: usize) -> Self {
        OptimalConfig {
            config: Config::from(&[], number_of_variables),
            value: 0.0,
            n_literals: 0,
        }
    }

    pub fn from(literals: &[i32], ext_ddnnf: &ExtendedDdnnf) -> Self {
        OptimalConfig {
            config: Config::from(literals, ext_ddnnf.ddnnf.number_of_variables as usize),
            value: ext_ddnnf.get_objective_fn_val_of_literals(literals),
            n_literals: literals.len(),
        }
    }

    pub fn unify_disjoint(mut self, other: &Self) -> Self {
        debug_assert!(
            self.config
                .get_decided_literals()
                .all(|literal| !other.config.get_decided_literals().contains(&literal))
        );

        self.config.extend(other.config.get_decided_literals());
        self.value += other.value;
        self.n_literals += other.n_literals;
        self
    }
}

impl ExtendedDdnnf {
    pub fn calc_best_config(&self, assumptions: &[i32]) -> Option<OptimalConfig> {
        let root_node_id = self.ddnnf.nodes.len() - 1;
        self.calc_best_config_for_node(root_node_id, assumptions)
    }

    pub fn calc_best_config_for_node(
        &self,
        node_id: usize,
        assumptions: &[i32],
    ) -> Option<OptimalConfig> {
        let node_count = self.ddnnf.nodes.len();
        let mut partial_configs = Vec::with_capacity(node_count);

        for id in 0..=node_id {
            partial_configs.push(self.calc_best_config_for_node_helper(
                id,
                assumptions,
                &partial_configs,
            ));
        }

        partial_configs.remove(node_id)
    }

    pub fn calc_best_config_for_node_helper(
        &self,
        node_id: usize,
        assumptions: &[i32],
        partial_configs: &[Option<OptimalConfig>],
    ) -> Option<OptimalConfig> {
        let number_of_variables = self.ddnnf.number_of_variables as usize;
        let node = self
            .ddnnf
            .nodes
            .get(node_id)
            .unwrap_or_else(|| panic!("Node {node_id} does not exist."));

        match &node.ntype {
            True => Some(OptimalConfig::empty(number_of_variables)),
            False => None,
            Literal { literal } => {
                if assumptions.contains(&literal.neg()) {
                    None
                } else {
                    Some(OptimalConfig::from(&[*literal], self))
                }
            }
            And { children } => {
                let children_configs = children
                    .iter()
                    .map(|&child_node_id| {
                        partial_configs.get(child_node_id).unwrap_or_else(|| {
                            panic!("No partial config for node {child_node_id} present.")
                        })
                    })
                    .collect_vec();

                if children_configs
                    .iter()
                    .any(|child_config_opt| child_config_opt.is_none())
                {
                    return None;
                }

                let unified_config = children_configs
                    .into_iter()
                    .flatten()
                    .fold(OptimalConfig::empty(number_of_variables), |acc, next| {
                        acc.unify_disjoint(next)
                    });

                Some(unified_config)
            }
            Or { children } => children
                .iter()
                .flat_map(|&child_node_id| {
                    partial_configs.get(child_node_id).unwrap_or_else(|| {
                        panic!("No partial config for node {child_node_id} present.")
                    })
                })
                .max()
                .cloned(),
        }
    }

    pub fn calc_top_k_configs(&self, k: usize, assumptions: &[i32]) -> Vec<OptimalConfig> {
        let root_node_id = self.ddnnf.nodes.len() - 1;
        self.calc_top_k_configs_for_node(root_node_id, k, assumptions)
    }

    pub fn calc_top_k_configs_for_node(
        &self,
        node_id: usize,
        k: usize,
        assumptions: &[i32],
    ) -> Vec<OptimalConfig> {
        let node_count = self.ddnnf.nodes.len();
        let mut partial_configs = Vec::with_capacity(node_count);

        for id in 0..=node_id {
            partial_configs.push(self.calc_top_k_configs_for_node_helper(
                id,
                k,
                assumptions,
                &partial_configs,
            ));
        }

        partial_configs.remove(node_id)
    }

    pub fn calc_top_k_configs_for_node_helper(
        &self,
        node_id: usize,
        k: usize,
        assumptions: &[i32],
        partial_configs: &[Vec<OptimalConfig>],
    ) -> Vec<OptimalConfig> {
        let number_of_variables = self.ddnnf.number_of_variables as usize;
        let node = self
            .ddnnf
            .nodes
            .get(node_id)
            .unwrap_or_else(|| panic!("Node {node_id} does not exist."));

        match &node.ntype {
            True => vec![OptimalConfig::empty(number_of_variables)],
            False => vec![],
            Literal { literal } => {
                if assumptions.contains(&literal.neg()) {
                    vec![]
                } else {
                    vec![OptimalConfig::from(&[*literal], self)]
                }
            }
            And { children } => {
                let children_configs = children
                    .iter()
                    .map(|&child_node_id| {
                        partial_configs.get(child_node_id).unwrap_or_else(|| {
                            panic!("No partial configs for node {child_node_id} present.")
                        })
                    })
                    .collect_vec();

                if children_configs
                    .iter()
                    .any(|child_configs| child_configs.is_empty())
                {
                    return vec![];
                }

                merge_top_k_results_and(children_configs, number_of_variables, Some(k))
            }
            Or { children } => {
                let children_configs = children
                    .iter()
                    .map(|&child_node_id| {
                        partial_configs.get(child_node_id).unwrap_or_else(|| {
                            panic!("No partial configs for node {child_node_id} present.")
                        })
                    })
                    .collect_vec();

                merge_top_k_results_or(children_configs, Some(k))
                    .into_iter()
                    .cloned()
                    .collect_vec()
            }
        }
    }
}

pub fn merge_top_k_results_and(
    sorted_lists: Vec<&Vec<OptimalConfig>>,
    number_of_variables: usize,
    max_amount: Option<usize>,
) -> Vec<OptimalConfig> {
    let insert_new_candidate =
        |idx_tuple: Vec<usize>,
         candidates_heap: &mut BinaryHeap<OptimalConfig>,
         candidate_idx_mapping: &mut BiHashMap<OptimalConfig, Vec<usize>>| {
            let candidate = sorted_lists
                .iter()
                .zip(idx_tuple.iter())
                .map(|(list, &i)| &list[i])
                .fold(OptimalConfig::empty(number_of_variables), |acc, next| {
                    acc.unify_disjoint(next)
                });

            candidate_idx_mapping.insert(candidate.clone(), idx_tuple);
            candidates_heap.push(candidate);
        };

    if sorted_lists.iter().any(|list| list.is_empty()) {
        return vec![];
    }

    let mut ordered_cartesian_product = Vec::new();
    let mut candidates_heap = BinaryHeap::new();
    let mut candidate_idx_mapping = BiHashMap::new();

    let max_result_amount = sorted_lists.iter().map(|list| list.len()).product();
    let result_amount = match max_amount {
        None => max_result_amount,
        Some(n) => n.min(max_result_amount),
    };

    let start_idx = vec![0; sorted_lists.len()];
    insert_new_candidate(start_idx, &mut candidates_heap, &mut candidate_idx_mapping);

    for _ in 0..result_amount {
        let best_elem = candidates_heap
            .pop()
            .expect("There must be candidates left.");
        let best_elem_idx_tuple = candidate_idx_mapping
            .get_by_left(&best_elem)
            .expect("This index tuple must have been inserted.");
        ordered_cartesian_product.push(best_elem);

        let new_candidates = (0..sorted_lists.len())
            .filter(|&list_idx| best_elem_idx_tuple[list_idx] + 1 < sorted_lists[list_idx].len())
            .map(|list_idx| {
                let mut new_idx_tuple = best_elem_idx_tuple.clone();
                new_idx_tuple[list_idx] += 1;
                new_idx_tuple
            })
            .filter(|idx_tuple| !candidate_idx_mapping.contains_right(idx_tuple))
            .collect_vec();

        new_candidates.into_iter().for_each(|idx_tuple| {
            insert_new_candidate(idx_tuple, &mut candidates_heap, &mut candidate_idx_mapping)
        });
    }

    ordered_cartesian_product
}

pub fn merge_top_k_results_or(
    sorted_lists: Vec<&Vec<OptimalConfig>>,
    max_amount: Option<usize>,
) -> Vec<&OptimalConfig> {
    let mut results = Vec::new();
    let mut curr_idx_tuple = vec![0; sorted_lists.len()];

    let max_result_amount = sorted_lists.iter().map(|list| list.len()).sum();
    let result_amount = match max_amount {
        None => max_result_amount,
        Some(n) => n.min(max_result_amount),
    };

    for _ in 0..result_amount {
        let (best_elem, list_idx) = sorted_lists
            .iter()
            .enumerate()
            .zip(curr_idx_tuple.iter())
            .filter(|&((_, list), &i)| i < list.len())
            .map(|((list_idx, list), &i)| (&list[i], list_idx))
            .max_by(|(left, _), (right, _)| left.cmp(right))
            .expect("There must be an element left.");

        results.push(best_elem);
        curr_idx_tuple[list_idx] += 1;
    }

    results
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::Ddnnf;
    use crate::ddnnf::extended_ddnnf::test::build_sandwich_ext_ddnnf;
    use num::ToPrimitive;

    pub fn build_sandwich_ext_ddnnf_with_objective_function_values() -> ExtendedDdnnf {
        let mut ext_ddnnf = build_sandwich_ext_ddnnf();
        ext_ddnnf.objective_fn_vals = Some(vec![
            9.0,  // Sandwich
            7.0,  // Bread
            0.0,  // Full Grain
            8.0,  // Flatbread
            4.0,  // Toast
            1.0,  // Cheese
            2.0,  // Gouda
            -2.0, // Sprinkled
            3.0,  // Slice
            4.0,  // Cheddar
            -2.0, // Cream Cheese
            -2.0, // Meat
            5.0,  // Salami
            -9.0, // Ham
            -1.0, // Chicken Breast
            7.0,  // Vegetables
            3.0,  // Cucumber
            6.0,  // Tomatoes
            -5.0, // Lettuce
        ]);

        ext_ddnnf
    }

    #[test]
    fn test_finding_best_configuration_without_assumptions() {
        let assumptions = []; // Nothing
        let ext_ddnnf = build_sandwich_ext_ddnnf_with_objective_function_values();
        let literals = [
            1,   // Sandwich
            2,   // Bread
            -3,  // Full Grain
            4,   // Flatbread
            -5,  // Toast
            6,   // Cheese
            7,   // Gouda
            -8,  // Sprinkled
            9,   // Slice
            10,  // Cheddar
            -11, // Cream Cheese
            12,  // Meat
            13,  // Salami
            -14, // Ham
            -15, // Chicken Breast
            16,  // Vegetables
            17,  // Cucumber
            18,  // Tomatoes
            -19, // Lettuce
        ];
        let expected_optimal_config = OptimalConfig::from(&literals, &ext_ddnnf);

        assert_eq!(
            ext_ddnnf.calc_best_config(&assumptions),
            Some(expected_optimal_config)
        );
    }

    #[test]
    fn test_finding_best_configuration_with_assumptions() {
        let assumptions = [8, -16]; // Sprinkled, -Vegetables
        let ext_ddnnf = build_sandwich_ext_ddnnf_with_objective_function_values();
        let literals = [
            1,   // Sandwich
            2,   // Bread
            -3,  // Full Grain
            4,   // Flatbread
            -5,  // Toast
            6,   // Cheese
            7,   // Gouda
            8,   // Sprinkled
            -9,  // Slice
            10,  // Cheddar
            -11, // Cream Cheese
            12,  // Meat
            13,  // Salami
            -14, // Ham
            -15, // Chicken Breast
            -16, // Vegetables
            -17, // Cucumber
            -18, // Tomatoes
            -19, // Lettuce
        ];
        let expected_optimal_config = OptimalConfig::from(&literals, &ext_ddnnf);

        assert_eq!(
            ext_ddnnf.calc_best_config(&assumptions),
            Some(expected_optimal_config)
        );
    }

    #[test]
    fn test_merge_top_k_results_and() {
        let number_of_variables: usize = 4;

        let mut ddnnf = Ddnnf::default();
        ddnnf.number_of_variables = number_of_variables as u32;
        let ext_ddnnf = ExtendedDdnnf {
            ddnnf,
            attrs: Default::default(),
            objective_fn_vals: Some(vec![3.0, 3.0, 2.0, 2.0]),
        };

        let sorted_lists = vec![
            vec![
                OptimalConfig::from(&[1, 2], &ext_ddnnf),   //1.a: value = 6
                OptimalConfig::from(&[-1, 2], &ext_ddnnf),  //1.b: value = 3
                OptimalConfig::from(&[-1, -2], &ext_ddnnf), //1.c: value = 0
            ],
            vec![
                OptimalConfig::from(&[3, 4], &ext_ddnnf),   //2.a: value = 4
                OptimalConfig::from(&[-3, 4], &ext_ddnnf),  //2.b: value = 2
                OptimalConfig::from(&[-3, -4], &ext_ddnnf), //2.c: value = 0
            ],
        ];

        let expected = vec![
            OptimalConfig::from(&[1, 2, 3, 4], &ext_ddnnf), //1.a + 2.a: value = 10
            OptimalConfig::from(&[1, 2, -3, 4], &ext_ddnnf), //1.a + 2.b: value = 8
            OptimalConfig::from(&[-1, 2, 3, 4], &ext_ddnnf), //1.b + 2.a: value = 7
            OptimalConfig::from(&[1, 2, -3, -4], &ext_ddnnf), //1.a + 2.c: value = 6
            OptimalConfig::from(&[-1, 2, -3, 4], &ext_ddnnf), //1.b + 2.b: value = 5
            OptimalConfig::from(&[-1, -2, 3, 4], &ext_ddnnf), //1.c + 2.a: value = 4
            OptimalConfig::from(&[-1, 2, -3, -4], &ext_ddnnf), //1.b + 2.c: value = 3
            OptimalConfig::from(&[-1, -2, -3, 4], &ext_ddnnf), //1.c + 2.b: value = 2
            OptimalConfig::from(&[-1, -2, -3, -4], &ext_ddnnf), //1.c + 2.c: value = 0
        ];

        assert_eq!(
            merge_top_k_results_and(sorted_lists.iter().collect_vec(), number_of_variables, None),
            expected
        );
    }

    #[test]
    fn test_merge_top_k_results_or() {
        let number_of_variables: usize = 4;

        let mut ddnnf = Ddnnf::default();
        ddnnf.number_of_variables = number_of_variables as u32;
        let ext_ddnnf = ExtendedDdnnf {
            ddnnf,
            attrs: Default::default(),
            objective_fn_vals: Some(vec![3.0, 3.0, 2.0, 2.0]),
        };

        let sorted_lists = vec![
            vec![
                OptimalConfig::from(&[1, 2], &ext_ddnnf),   //1.a: value = 6
                OptimalConfig::from(&[-1, 2], &ext_ddnnf),  //1.b: value = 3
                OptimalConfig::from(&[-1, -2], &ext_ddnnf), //1.c: value = 0
            ],
            vec![
                OptimalConfig::from(&[3, 4], &ext_ddnnf),  //2.a: value = 4
                OptimalConfig::from(&[-3, 4], &ext_ddnnf), //2.b: value = 2
            ],
        ];

        let expected = vec![
            OptimalConfig::from(&[1, 2], &ext_ddnnf),   //1.a: value = 6
            OptimalConfig::from(&[3, 4], &ext_ddnnf),   //2.a: value = 4
            OptimalConfig::from(&[-1, 2], &ext_ddnnf),  //1.b: value = 3
            OptimalConfig::from(&[-3, 4], &ext_ddnnf),  //2.b: value = 2
            OptimalConfig::from(&[-1, -2], &ext_ddnnf), //1.c: value = 0
        ];

        assert_eq!(
            merge_top_k_results_or(sorted_lists.iter().collect_vec(), None),
            expected.iter().collect_vec()
        );
    }

    #[test]
    fn test_finding_top_k_configurations_without_assumptions() {
        let assumptions = []; // Nothing
        let mut ext_ddnnf = build_sandwich_ext_ddnnf_with_objective_function_values();

        let n_config = ext_ddnnf
            .ddnnf
            .execute_query(&assumptions)
            .to_usize()
            .unwrap();
        let best_config_values_brute_force = ext_ddnnf
            .ddnnf
            .enumerate(&mut vec![], n_config)
            .unwrap()
            .into_iter()
            .map(|literals| OptimalConfig::from(&literals[..], &ext_ddnnf))
            .sorted()
            .rev()
            .map(|config| config.value)
            .collect_vec();

        let best_config_values_top_k = ext_ddnnf
            .calc_top_k_configs(n_config + 1337, &assumptions)
            .into_iter()
            .map(|config| config.value)
            .collect_vec();

        assert_eq!(best_config_values_top_k, best_config_values_brute_force);
    }

    #[test]
    fn test_finding_top_k_configurations_with_assumptions() {
        let assumptions = [8, -16]; // Sprinkled, -Vegetables
        let mut ext_ddnnf = build_sandwich_ext_ddnnf_with_objective_function_values();

        let n_config = ext_ddnnf
            .ddnnf
            .execute_query(&assumptions)
            .to_usize()
            .unwrap();
        let best_config_values_brute_force = ext_ddnnf
            .ddnnf
            .enumerate(&mut assumptions.to_vec(), n_config)
            .unwrap()
            .into_iter()
            .map(|literals| OptimalConfig::from(&literals[..], &ext_ddnnf))
            .sorted()
            .rev()
            .map(|config| config.value)
            .collect_vec();

        let best_config_values_top_k = ext_ddnnf
            .calc_top_k_configs(n_config + 1337, &assumptions)
            .into_iter()
            .map(|config| config.value)
            .collect_vec();

        assert_eq!(best_config_values_top_k, best_config_values_brute_force);
    }
}
