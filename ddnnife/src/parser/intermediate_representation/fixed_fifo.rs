use std::{
    collections::VecDeque,
    fmt::{self, Debug},
    sync::Arc,
};

pub type ConflictFn<T> = dyn Fn(&T, &T) -> bool;

pub(crate) struct FixedFifo<T> {
    buffer: VecDeque<T>,
    conflict_fn: Arc<ConflictFn<T>>,
    max_size: usize,
}

// Manually implement Clone for FixedFifo because it cannot be derived automatically
impl<T> Clone for FixedFifo<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        FixedFifo {
            buffer: self.buffer.clone(),
            conflict_fn: self.conflict_fn.clone(),
            max_size: self.max_size,
        }
    }
}

impl<T> Debug for FixedFifo<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FixedFifo")
            .field("buffer", &self.buffer)
            .field("max_size", &self.max_size)
            .finish()
    }
}

unsafe impl<T> Send for FixedFifo<T> where T: Send {}
unsafe impl<T> Sync for FixedFifo<T> where T: Sync {}

impl<T> FixedFifo<T> {
    pub(crate) fn new(max_size: usize, conflict_fn: Arc<ConflictFn<T>>) -> Self {
        FixedFifo {
            buffer: VecDeque::with_capacity(max_size),
            conflict_fn,
            max_size,
        }
    }

    pub(crate) fn _new_wo_conflict(max_size: usize) -> Self {
        FixedFifo {
            buffer: VecDeque::with_capacity(max_size),
            conflict_fn: Arc::new(|_, _| true),
            max_size,
        }
    }

    pub(crate) fn _push(&mut self, item: T) {
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
    }

    pub(crate) fn retain_push(&mut self, item: T) {
        self.buffer.retain(|elem| (self.conflict_fn)(&item, elem));

        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
    }

    pub(crate) fn find_and_remove<FR>(&mut self, predicate: FR) -> Option<T>
    where
        FR: Fn(&T) -> bool,
    {
        if let Some(index) = self.buffer.iter().position(predicate) {
            // Use drain to remove and return the matching item.
            let matching_item = self.buffer.drain(index..index + 1).next();
            matching_item
        } else {
            None
        }
    }

    pub(crate) fn _len(&mut self) -> usize {
        self.buffer.len()
    }

    fn _switch_conflict_fn(&mut self, conflict_fn_replacement: Arc<ConflictFn<T>>) {
        self.conflict_fn = conflict_fn_replacement
    }

    pub(crate) fn _get(&self, index: usize) -> Option<&T> {
        self.buffer.get(index)
    }

    pub(crate) fn _get_buffer(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.buffer.clone().into_iter().collect()
    }
}

#[cfg(test)]
mod test {
    use super::FixedFifo;
    use crate::parser::build_ddnnf;
    use crate::parser::intermediate_representation::ClauseApplication;
    use std::{collections::HashSet, sync::Arc};

    #[test]
    fn size_restriction() {
        const MAX_SIZE: usize = 3;
        let mut fsb = FixedFifo::_new_wo_conflict(MAX_SIZE);

        let mut vec: Vec<i32> = vec![];
        assert_eq!(vec, fsb._get_buffer());

        for i in 1..10 {
            fsb._push(i);
            vec.push(i);
            if vec.len() > MAX_SIZE {
                vec.remove(0);
            }
            assert_eq!(vec, fsb._get_buffer());
        }
    }

    #[test]
    fn push_with_conflicts() {
        let mut fsb = FixedFifo::new(10, Arc::new(|_, &add| add % 2 == 0));
        for i in 1..=10 {
            fsb.retain_push(i);
        }
        assert_eq!(vec![2, 4, 6, 8, 10], fsb._get_buffer());

        fsb._switch_conflict_fn(Arc::new(|add, elem| add + elem < 25));
        fsb.retain_push(20);
        assert_eq!(vec![2, 4, 20], fsb._get_buffer());

        let mut fsb_ir_cache: FixedFifo<((HashSet<i32>, (i32, i32)), i32)> = FixedFifo::new(
            10,
            Arc::new(|((edit_lits, _), _), ((edit_lits_other, _), _)| {
                edit_lits.intersection(&edit_lits_other).count() == 0
            }),
        );
        fsb_ir_cache.retain_push(((vec![1, 2, -3].into_iter().collect(), (0, 0)), 0));
        fsb_ir_cache.retain_push(((vec![-1].into_iter().collect(), (0, 0)), 0));
        fsb_ir_cache.retain_push(((vec![4, -4].into_iter().collect(), (0, 0)), 0));
        assert_eq!(3, fsb_ir_cache._len());

        fsb_ir_cache.retain_push(((vec![1, -1].into_iter().collect(), (0, 0)), 0));
        fsb_ir_cache.retain_push(((vec![5, 1].into_iter().collect(), (0, 0)), 0));
        assert_eq!(2, fsb_ir_cache._len());
    }

    #[cfg(feature = "d4")]
    #[test]
    fn metrics() {
        let ddnnf_small_ex = build_ddnnf("tests/data/small_ex_c2d.nnf", None);
        assert_eq!(12, ddnnf_small_ex.node_count());
        assert_eq!(11, ddnnf_small_ex.edge_count());
        assert!((1.0 - ddnnf_small_ex.sharing()).abs() < 1e-7);

        let ddnnf_x264 = build_ddnnf("tests/data/VP9.cnf", None);
        assert_eq!(148, ddnnf_x264.node_count());
        assert_eq!(185, ddnnf_x264.edge_count());
        assert!((148.0 / (185.0 + 1.0) - ddnnf_x264.sharing()).abs() < 1e-7);
    }

    #[test]
    fn rebuild_ddnnf() {
        let mut ddnnfs = Vec::new();
        ddnnfs.push(build_ddnnf("tests/data/auto1_c2d.nnf", None));
        ddnnfs.push(build_ddnnf("tests/data/auto1_d4.nnf", Some(2513)));
        ddnnfs.push(build_ddnnf("tests/data/VP9_d4.nnf", Some(42)));
        ddnnfs.push(build_ddnnf("tests/data/small_ex_c2d.nnf", None));

        for ddnnf in ddnnfs {
            let mut rebuild_clone = ddnnf.clone();
            // rebuild 10 times to ensure that negative effects do not occur no matter
            // how many times we rebuild
            for _ in 0..10 {
                rebuild_clone.rebuild();
            }

            // check each field seperatly due to intermediate_graph.graph not being able to derive PartialEq
            assert_eq!(ddnnf.nodes, rebuild_clone.nodes);
            assert_eq!(ddnnf.literals, rebuild_clone.literals);
            assert_eq!(ddnnf.core, rebuild_clone.core);
            assert_eq!(ddnnf.true_nodes, rebuild_clone.true_nodes);
            assert_eq!(ddnnf.number_of_variables, rebuild_clone.number_of_variables);
        }
    }

    #[test]
    fn incremental_applying_clause() {
        let ddnnf_file_paths = vec![
            ("tests/data/small_ex_c2d.nnf", 4, vec![4]),
            //("tests/data/VP9_d4.nnf", 42, vec![vec![4, 5]])
        ];

        for (path, features, clause) in ddnnf_file_paths {
            let mut ddnnf = build_ddnnf(path, Some(features));
            println!("Card of Features before change:");
            for i in 0..ddnnf.number_of_variables {
                println!("{i}: {:?}", ddnnf.execute_query(&[i as i32]));
            }

            ddnnf.prepare_and_apply_incremental_edit(vec![(clause, ClauseApplication::Add)]);
            println!("Card of Features after change:");
            for i in 0..ddnnf.number_of_variables {
                println!("{i}: {:?}", ddnnf.execute_query(&[i as i32]));
            }
        }
    }
}
