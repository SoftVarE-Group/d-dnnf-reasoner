use std::iter;
use streaming_iterator::StreamingIterator;

/// This is a [StreamingIterator] that produces t-wise indices. These can be mapped into a list
/// of literals to get t-wise interactions of those literals. [TInteractionIter] provides this
/// functionality.
pub(super) struct TIndicesIter {
    number_of_vars: usize,
    t: usize,
    first: bool,
    tuple: Vec<usize>,
}

impl StreamingIterator for TIndicesIter {
    type Item = [usize];

    fn advance(&mut self) {
        if self.first {
            self.first = false;
            return;
        }

        self.tuple[0] += 1;
        if self.tuple[0] >= self.number_of_vars {
            let mut p: usize = 0;

            // carry over to the next places - like 0999 -> 1000
            while self.tuple[p] >= self.number_of_vars - p && self.tuple[self.t] == 0 {
                self.tuple[p] = 0;
                p += 1; // go to next place
                self.tuple[p] += 1;
            }

            // ensure that we don't have duplicates - like 1000 -> 1234
            if let Some(bound) = self.t.checked_sub(2) {
                for j in (0..=bound).rev() {
                    if self.tuple[j] < self.tuple[j + 1] {
                        self.tuple[j] = self.tuple[j + 1] + 1;
                    }
                }
            }
        }
    }

    fn get(&self) -> Option<&Self::Item> {
        if self.tuple[self.t] == 0 {
            Some(&self.tuple[..self.t]) // return a slice of length t
        } else {
            None
        }
    }
}

impl TIndicesIter {
    pub fn new(number_of_vars: usize, t: usize) -> Self {
        let tuple = (0..t).rev().chain(iter::once(0)).collect();
        Self {
            number_of_vars,
            t,
            first: true,
            tuple,
        }
    }
}

/// This is a [StreamingIterator] to produce t-wise interactions over a slice of literals.
/// This implementation only ever allocates a single [Vec] to hold the current interaction. It is
/// therefore much more performant than variants that implement the [Iterator] trait.
pub(super) struct TInteractionIter<'a> {
    indices_iter: TIndicesIter,
    literals: &'a [i32],
    interaction: Vec<i32>,
}

impl<'a> StreamingIterator for TInteractionIter<'a> {
    type Item = [i32];

    fn advance(&mut self) {
        self.indices_iter.advance();

        if let Some(indices) = self.indices_iter.get() {
            for (value, index) in self.interaction.iter_mut().zip(indices) {
                *value = self.literals[*index];
            }
        }
    }

    fn get(&self) -> Option<&Self::Item> {
        if self.indices_iter.get().is_some() {
            Some(&self.interaction)
        } else {
            None
        }
    }
}

impl<'a> TInteractionIter<'a> {
    pub(super) fn new(literals: &'a [i32], t: usize) -> Self {
        debug_assert!(literals.len() >= t);
        debug_assert!(!literals.contains(&0));
        Self {
            indices_iter: TIndicesIter::new(literals.len(), t),
            literals,
            interaction: vec![0; t],
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_t_indices_iter() {
        let mut iter = TIndicesIter::new(5, 3);

        assert_eq!(Some([2, 1, 0].as_slice()), iter.next());
        assert_eq!(Some([3, 1, 0].as_slice()), iter.next());
        assert_eq!(Some([4, 1, 0].as_slice()), iter.next());
        assert_eq!(Some([3, 2, 0].as_slice()), iter.next());
        assert_eq!(Some([4, 2, 0].as_slice()), iter.next());
        assert_eq!(Some([4, 3, 0].as_slice()), iter.next());
        assert_eq!(Some([3, 2, 1].as_slice()), iter.next());
        assert_eq!(Some([4, 2, 1].as_slice()), iter.next());
        assert_eq!(Some([4, 3, 1].as_slice()), iter.next());
        assert_eq!(Some([4, 3, 2].as_slice()), iter.next());
        assert_eq!(None, iter.next());
    }
}
