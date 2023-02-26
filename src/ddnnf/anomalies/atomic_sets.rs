use itertools::Itertools;
use rug::Integer;

use crate::Ddnnf;

use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq)]
pub struct UnionFind<N: Hash + Eq + Clone> {
    size: usize,
    parents: HashMap<N, N>,
    rank: HashMap<N, usize>,
}

pub trait UnionFindTrait<N: Eq + Hash + Clone> {
    fn find(&mut self, node: N) -> N;
    fn equiv(&mut self, x: N, y: N) -> bool;
    fn union(&mut self, x: N, y: N);
    fn subsets(&mut self) -> Vec<Vec<N>>;
}

impl<T> UnionFind<T>
where T: Eq + Hash + Clone {
    pub fn new() -> UnionFind<T> {
        let parents: HashMap<T, T> = HashMap::new();
        let rank: HashMap<T, usize> = HashMap::new();

        UnionFind {
            size: 0,
            parents,
            rank,
        }
    }

    pub fn entries(&self) -> Vec<T> {
        self.rank.clone().into_keys().collect()
    }
}

impl<T> UnionFindTrait<T> for UnionFind<T>
where T: Eq + Hash + Ord + Clone {
    fn find(&mut self, node: T) -> T {
        if !self.parents.contains_key(&node) {
            self.parents.insert(node.clone(), node.clone());
            self.size += 1;
        }

        if !(node.eq(self.parents.get(&node).unwrap())) {
            let found = self.find((*self.parents.get(&node).unwrap()).clone() );
            self.parents.insert(node.clone(), found);
        }
        (*self.parents.get(&node).unwrap()).clone()
    }

    fn equiv(&mut self, x: T, y: T) -> bool {
        self.find(x) == self.find(y)
    }

    fn union(&mut self, x: T, y: T) {
        let x_root = self.find(x);
        let y_root = self.find(y);

        if !self.rank.contains_key(&x_root) {
            self.rank.insert(x_root.clone(), 0);
        }

        if !self.rank.contains_key(&y_root) {
            self.rank.insert(y_root.clone(), 0);
        }
        if x_root.eq(&y_root) {
            return;
        }
        let x_root_rank: usize = *self.rank.get(&x_root).unwrap();
        let y_root_rank: usize = *self.rank.get(&y_root).unwrap();

        if x_root_rank > y_root_rank {
            self.parents.insert(y_root, x_root);
        } else {
            self.parents.insert(x_root, y_root.clone());
            if x_root_rank == y_root_rank {
                self.rank.insert(y_root, y_root_rank + 1);
            }
        }
    }

    fn subsets(&mut self) -> Vec<Vec<T>> {
        let mut result: HashMap<T, Vec<T>> = HashMap::with_capacity(self.size);

        let rank_cp = self.rank.clone();
    
        for (node, _) in rank_cp.iter() {
            let root = self.find((*node).clone());

            if !result.contains_key(&root) {
                let mut vec = Vec::new();
                vec.push((*node).clone());
                result.insert(root, vec);
            } else {
                let prev = &mut *result.get_mut(&root).unwrap();
                prev.push((*node).clone());
            }
        }
        let mut sets: Vec<Vec<_>> = result.into_values().collect();
        sets.iter_mut().for_each(|set| set.sort_unstable());
        sets
    }
}

impl Ddnnf {
    /// Compute all false-optional features
    /// A feature is false optional iff there is no valid configuration that excludes
    /// the feature while simultanously excluding its parent
    #[allow(dead_code)]
    fn get_false_optional(&mut self) {
        /***
         * Probleme: 
         *      - Wie findet man den parent eines features?
         *      - Welches feature ist das root feature?
         * 
         * Beziehung eines features zu parent:
         * p => c (Damit das child ausgewählt werden kann muss der parent gewählt sein)
         * 
         * F_FM and p and not c is unsatisfiable
         * <=> #SAT(F_FM and p) == #SAT(F_FM and p and c) und f ist nicht mandatory
         * Wie ermittelt man ob ein feature mandatory ist?
         *      -> SAT(F_FM and p and other child von p)
         * 
        */
    }

    /// Compute all atomic sets
    /// A group forms an atomic set iff every valid configuration either includes
    /// or excludes all mebers of that atomic set
    pub(crate) fn get_atomic_sets(&mut self) -> Vec<Vec<u16>> {
        let mut combinations: Vec<(Integer, i32)> = Vec::new();

        for i in 1..=self.number_of_variables as i32 {
            combinations.push((self.execute_query(&[i]), i));
        }
        combinations.sort_unstable();
        
        let mut data_grouped = Vec::new();
        
        let mut current_key = combinations[0].0.clone();
        let mut values_current_key = Vec::new();
        for (key, value) in combinations.into_iter() {
            if current_key == key {
                values_current_key.push(value);
            } else {
                data_grouped.push((current_key, std::mem::take(&mut values_current_key)));
                current_key = key;
                values_current_key.push(value);
            }
        }
        data_grouped.push((current_key, values_current_key));

        let mut atmoic_sets: UnionFind<u16> = UnionFind::new();
        for (key, group) in data_grouped {
            self._incremental_check(key, &group, &mut atmoic_sets);
        }

        atmoic_sets.subsets()
    }

    /// first naive approach to compute atomic sets by incrementally add a feature one by one
    /// while checking if the atomic set property (i.e. the count stays the same) still holds
    fn _incremental_check(&mut self, control: Integer, pot_atomic_set: &Vec<i32>, atomic_sets: &mut UnionFind<u16>) {
        // goes through all combinations of set candidates and checks whether the pair is part of an atomic set
        for pair in pot_atomic_set.to_owned().into_iter().combinations(2) {
            let x = pair[0] as u16; let y = pair[1] as u16;

            // we don't have to check if a pair is part of an atomic set if they already are connected via transitivity
            if atomic_sets.equiv(x, y) {
                continue;
            }

            // we identify a pair of values to be in the same atomic set, then we union them
            if self.execute_query(&pair) == control {
                atomic_sets.union(x, y);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::parser::{build_d4_ddnnf_tree, build_ddnnf_tree_with_extras};

    use super::*;

    #[test]
    fn atomic_sets() {
        let mut vp9: Ddnnf =
            build_d4_ddnnf_tree("tests/data/VP9_d4.nnf", 42);
        let mut axtls: Ddnnf =
            build_ddnnf_tree_with_extras("example_input/axTLS.dimacs.nnf");
        let mut auto1: Ddnnf =
            build_d4_ddnnf_tree("tests/data/auto1_d4.nnf", 2513);
        //let mut auto2: Ddnnf =
        //    build_ddnnf_tree_with_extras("example_input/automotive2_4.dimacs.nnf");

        println!("atomic set vp9: {:?}", vp9.get_atomic_sets());
        println!("atomic set axtls: {:?}", axtls.get_atomic_sets());
        println!("atomic set auto1: {:?}", auto1.get_atomic_sets());
        //println!("atomic set auto2: {:?}", auto2.get_atomic_sets());
    }
}