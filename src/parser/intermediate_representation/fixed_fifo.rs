use std::{
    collections::VecDeque,
    fmt::{self, Debug},
    rc::Rc,
};

pub(crate) struct FixedFifo<T> {
    buffer: VecDeque<T>,
    conflict_fn: Rc<dyn Fn(&T, &T) -> bool>,
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

impl<T> FixedFifo<T> {
    pub(crate) fn new(max_size: usize, conflict_fn: Rc<dyn Fn(&T, &T) -> bool>) -> Self {
        FixedFifo {
            buffer: VecDeque::with_capacity(max_size),
            conflict_fn,
            max_size,
        }
    }

    pub(crate) fn _new_wo_conflict(max_size: usize) -> Self {
        FixedFifo {
            buffer: VecDeque::with_capacity(max_size),
            conflict_fn: Rc::new(|_, _| true),
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
        if let Some(index) = self.buffer.iter().position(|item| predicate(item)) {
            // Use drain to remove and return the matching item.
            let matching_item = self.buffer.drain(index..index + 1).next();
            matching_item
        } else {
            None
        }
    }

    pub(crate) fn len(&mut self) -> usize {
        return self.buffer.len();
    }

    fn _switch_conflict_fn(&mut self, conflict_fn_replacement: Rc<dyn Fn(&T, &T) -> bool>) {
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
    use std::rc::Rc;

    use super::FixedFifo;

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
        let mut fsb = FixedFifo::new(10, Rc::new(|_, &add| add % 2 == 0));
        for i in 1..=10 {
            fsb.retain_push(i);
        }
        assert_eq!(vec![2, 4, 6, 8, 10], fsb._get_buffer());

        fsb._switch_conflict_fn(Rc::new(|add, elem| add + elem < 25));
        fsb.retain_push(20);
        assert_eq!(vec![2, 4, 20], fsb._get_buffer());
    }
}
