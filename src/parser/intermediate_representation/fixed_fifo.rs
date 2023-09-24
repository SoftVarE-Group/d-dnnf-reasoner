use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub(crate) struct FixedFifo<T> {
    buffer: VecDeque<T>,
    max_size: usize,
}

impl<T> FixedFifo<T> {
    pub(crate) fn new(max_size: usize) -> Self {
        FixedFifo {
            buffer: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    pub(crate) fn _push(&mut self, item: T) {
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
    }

    pub(crate) fn conflicting_push<F>(&mut self, item: T, _conflict: F)
    where
        F: Fn(&T) -> bool,
    {
        self.buffer.retain(|item| !_conflict(item));
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
    }

    pub(crate) fn find_and_remove<F>(&mut self, predicate: F) -> Option<T>
    where
        F: Fn(&T) -> bool,
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

    #[test]
    fn size_restriction() {
        const MAX_SIZE: usize = 3;
        let mut fsb = FixedFifo::new(MAX_SIZE);

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
        let mut fsb = FixedFifo::new(10);
        for i in 1..=10 {
            fsb.conflicting_push(i, |other| other % 2 != 0);
        }
        assert_eq!(vec![2, 4, 6, 8, 10], fsb._get_buffer());

        fsb.conflicting_push(20, |other| 20 % other == 0);
        assert_eq!(vec![6, 8, 20], fsb._get_buffer());
    }
}
