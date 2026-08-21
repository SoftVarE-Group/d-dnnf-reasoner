use num::{BigInt, ToPrimitive, integer::binomial};

/// Iterator over indices of `t`-wise interactions.
#[derive(Debug)]
struct TIndicesIter {
    /// The number of elements to be combined.
    n: usize,
    /// The interaction size to generate.
    t: usize,
    /// The currently generated interaction.
    next: Vec<usize>,
    /// Indicates whether the end of iteration is reached.
    finished: bool,
}

impl Iterator for TIndicesIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let result = Some(self.next.clone());

        // Increment the first index.
        self.next[0] += 1;

        // If no overflow occurred there is nothing left to do.
        if self.next[0] < self.n {
            return result;
        }

        // Otherwise, handle overflow incrementally.
        // Handle all indices that exceed the maximum value of their position (`n - i`).
        let mut i = 0;
        while i < self.next.len() - 1 && self.next[i] >= self.n - i {
            // Reset the current (overflowing) position.
            self.next[i] = 0;

            // Increment the next position to account for the overflow.
            self.next[i + 1] += 1;

            // Continue the overflow check at the next position.
            i += 1;
        }

        // Remove duplicates occurring after overflows.
        // Cases such as `[0, 2, 0]` do not represent an actual t-wise interaction.
        // Increment them to the next t-wise interaction such as `[3, 2, 0]`.
        (0..self.t - 1).rev().for_each(|i| {
            let following = self.next[i + 1];
            if self.next[i] < following {
                self.next[i] = following + 1;
            }
        });

        // We are done if the first index would be out of range for the next iteration.
        if self.next[0] == self.n {
            self.finished = true;
        }

        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = binomial(BigInt::from(self.n), BigInt::from(self.t))
            .to_usize()
            .expect("There are too many t-wise iterations");

        (size, Some(size))
    }
}

impl TIndicesIter {
    /// Creates a new iterator for `t`-wise interactions over `n` elements.
    ///
    /// # Panics
    ///
    /// Panics if `n < t`.
    pub fn new(n: usize, t: usize) -> Self {
        assert!(n >= t);

        Self {
            n,
            t,
            next: (0..t).rev().collect(),
            // For `t == 0`, there is nothing to do.
            finished: t == 0,
        }
    }
}

/// This is a [StreamingIterator] to produce t-wise interactions over a slice of literals.
/// This implementation only ever allocates a single [Vec] to hold the current interaction. It is
/// therefore much more performant than variants that implement the [Iterator] trait.
pub struct TInteractionIter<'a> {
    indices_iter: TIndicesIter,
    literals: &'a [i32],
}

impl<'a> Iterator for TInteractionIter<'a> {
    type Item = Vec<i32>;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices_iter.next().map(|indices| {
            indices
                .into_iter()
                .map(|index| self.literals[index])
                .collect()
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices_iter.size_hint()
    }
}

impl<'a> TInteractionIter<'a> {
    /// Creates a new iterator over t-wise interactions of the given literals.
    ///
    /// # Panics
    ///
    /// Panics if the number if literals is not at least `t`.
    pub fn new(literals: &'a [i32], t: usize) -> Self {
        assert!(
            literals.len() >= t,
            "For t-wise iteration, there must be at least t literals."
        );

        debug_assert!(!literals.contains(&0));

        Self {
            indices_iter: TIndicesIter::new(literals.len(), t),
            literals,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_t_indices_iter() {
        let mut iter = TIndicesIter::new(5, 3);

        assert_eq!(Some(vec![2, 1, 0]), iter.next());
        assert_eq!(Some(vec![3, 1, 0]), iter.next());
        assert_eq!(Some(vec![4, 1, 0]), iter.next());
        assert_eq!(Some(vec![3, 2, 0]), iter.next());
        assert_eq!(Some(vec![4, 2, 0]), iter.next());
        assert_eq!(Some(vec![4, 3, 0]), iter.next());
        assert_eq!(Some(vec![3, 2, 1]), iter.next());
        assert_eq!(Some(vec![4, 2, 1]), iter.next());
        assert_eq!(Some(vec![4, 3, 1]), iter.next());
        assert_eq!(Some(vec![4, 3, 2]), iter.next());
        assert_eq!(None, iter.next());
    }
}
