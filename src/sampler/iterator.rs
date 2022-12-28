use std::iter;
use streaming_iterator::StreamingIterator;

/// Creates an iterator for pairs (t=2) of features.
/// The iteration order is lexicographic.
pub fn pair_iter(number_of_vars: i32) -> impl Iterator<Item=Vec<i32>> {
    internal_pair_iter(number_of_vars).map(|(i, j)| vec![i, j])
}

/// Creates an iterator for pair wise (t=2) interactions.
/// The iteration order is lexicographic.
pub fn pair_wise_interaction_iter(
    number_of_vars: i32,
) -> impl Iterator<Item=Vec<i32>> {
    internal_pair_iter(number_of_vars).flat_map(|(i, j)| {
        vec![vec![i, j], vec![i, -j], vec![-i, j], vec![-i, -j]].into_iter()
    })
}

fn internal_pair_iter(number_of_vars: i32) -> impl Iterator<Item=(i32, i32)> {
    (1..=number_of_vars).into_iter().flat_map(move |i| {
        (i + 1..=number_of_vars).into_iter().map(move |j| (i, j))
    })
}

/// Creates an iterator for t-tuples of features.
/// The iteration order is lexicographic.
pub fn t_tuple_iter(
    number_of_vars: i32,
    t: usize,
) -> Box<dyn Iterator<Item=Vec<i32>>> {
    assert!(t > 0, "t must be greater than 0");
    assert!(
        t as i32 <= number_of_vars,
        "t can't be greater than number of vars"
    );

    let mut iter: Box<dyn Iterator<Item=Vec<i32>>>;
    iter = Box::new(iter::once(vec![]));

    for _i in 0..t {
        iter = Box::new(iter.flat_map(move |tuple| {
            let max = match tuple.last() {
                None => 0,
                Some(x) => *x,
            };

            (max + 1..=number_of_vars).into_iter().map(move |j| {
                let mut tuple = tuple.clone();
                tuple.push(j);
                tuple
            })
        }));
    }
    iter
}

/// Creates an iterator for t-tuples over the given literals
/// The iteration order is lexicographic and then mapped to the literals.
pub fn t_wise_over<'a>(
    literals: &'a [i32],
    t: usize,
) -> Box<dyn Iterator<Item=Vec<i32>> + 'a> {
    /*
    The supplied literals may contain a literal and it's negation. We filter interactions out that
    contain a literal and it's negation.
     */
    let conflict_possible =
        t > 1 && literals.iter().any(|literal| literals.contains(&-literal));
    let no_conflicts: fn(&Vec<i32>) -> bool = if conflict_possible {
        |interaction| {
            !interaction
                .iter()
                .any(|literal| interaction.contains(&-literal))
        }
    } else {
        |_| true
    };

    let iter = t_tuple_iter(literals.len() as i32, t)
        .map(move |interaction| {
            interaction
                .into_iter()
                .map(|literal| {
                    literals.get(literal as usize - 1).cloned().expect("")
                })
                .collect()
        })
        .filter(no_conflicts);

    Box::new(iter)
}

/// This is a [StreamingIterator] that produces t-wise indices. These can be mapped into a list
/// of literals to get t-wise interactions of those literals. [TInteractionIter] provides this
/// functionality.
pub struct TIndicesIter {
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
        if self.tuple[0] as usize >= self.number_of_vars {
            let mut p: usize = 0;

            // carry over to the next places - like 0999 -> 1000
            while self.tuple[p] as usize >= self.number_of_vars - p
                && self.tuple[self.t] == 0
            {
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
pub struct TInteractionIter<'a> {
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
    pub fn new(literals: &'a [i32], t: usize) -> Self {
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
    fn test_pair_iter() {
        let mut iter = pair_iter(3);

        assert_eq!(Some(vec![1, 2]), iter.next());
        assert_eq!(Some(vec![1, 3]), iter.next());
        assert_eq!(Some(vec![2, 3]), iter.next());
    }

    #[test]
    fn test_t_wise_iter() {
        let mut iter = t_tuple_iter(5, 1);
        assert_eq!(Some(vec![1]), iter.next());
        assert_eq!(Some(vec![2]), iter.next());
        assert_eq!(Some(vec![3]), iter.next());
        assert_eq!(Some(vec![4]), iter.next());
        assert_eq!(Some(vec![5]), iter.next());
        assert_eq!(None, iter.next());

        let mut iter = t_tuple_iter(5, 3);
        assert_eq!(Some(vec![1, 2, 3]), iter.next());
        assert_eq!(Some(vec![1, 2, 4]), iter.next());
        assert_eq!(Some(vec![1, 2, 5]), iter.next());
        assert_eq!(Some(vec![1, 3, 4]), iter.next());
        assert_eq!(Some(vec![1, 3, 5]), iter.next());
        assert_eq!(Some(vec![1, 4, 5]), iter.next());
        assert_eq!(Some(vec![2, 3, 4]), iter.next());
        assert_eq!(Some(vec![2, 3, 5]), iter.next());
        assert_eq!(Some(vec![2, 4, 5]), iter.next());
        assert_eq!(Some(vec![3, 4, 5]), iter.next());
        assert_eq!(None, iter.next());
    }

    #[test]
    fn test_t_wise_over() {
        let literals = [42, 69, -12, 17, 420];
        let mut iter = t_wise_over(&literals, 1);
        assert_eq!(Some(vec![42]), iter.next());
        assert_eq!(Some(vec![69]), iter.next());
        assert_eq!(Some(vec![-12]), iter.next());
        assert_eq!(Some(vec![17]), iter.next());
        assert_eq!(Some(vec![420]), iter.next());
        assert_eq!(None, iter.next());

        let mut iter = t_wise_over(&literals, 3);
        assert_eq!(Some(vec![42, 69, -12]), iter.next());
        assert_eq!(Some(vec![42, 69, 17]), iter.next());
        assert_eq!(Some(vec![42, 69, 420]), iter.next());
        assert_eq!(Some(vec![42, -12, 17]), iter.next());
        assert_eq!(Some(vec![42, -12, 420]), iter.next());
        assert_eq!(Some(vec![42, 17, 420]), iter.next());
        assert_eq!(Some(vec![69, -12, 17]), iter.next());
        assert_eq!(Some(vec![69, -12, 420]), iter.next());
        assert_eq!(Some(vec![69, 17, 420]), iter.next());
        assert_eq!(Some(vec![-12, 17, 420]), iter.next());
        assert_eq!(None, iter.next());

        let literals = [1, 2, -1, -2];
        let mut iter = t_wise_over(&literals, 2);
        assert_eq!(Some(vec![1, 2]), iter.next());
        assert_eq!(Some(vec![1, -2]), iter.next());
        assert_eq!(Some(vec![2, -1]), iter.next());
        assert_eq!(Some(vec![-1, -2]), iter.next());
    }

    #[test]
    fn test_pair_interaction_iter() {
        let mut iter = pair_wise_interaction_iter(3);

        assert_eq!(Some(vec![1, 2]), iter.next());
        assert_eq!(Some(vec![1, -2]), iter.next());
        assert_eq!(Some(vec![-1, 2]), iter.next());
        assert_eq!(Some(vec![-1, -2]), iter.next());

        assert_eq!(Some(vec![1, 3]), iter.next());
        assert_eq!(Some(vec![1, -3]), iter.next());
        assert_eq!(Some(vec![-1, 3]), iter.next());
        assert_eq!(Some(vec![-1, -3]), iter.next());

        assert_eq!(Some(vec![2, 3]), iter.next());
        assert_eq!(Some(vec![2, -3]), iter.next());
        assert_eq!(Some(vec![-2, 3]), iter.next());
        assert_eq!(Some(vec![-2, -3]), iter.next());

        assert_eq!(None, iter.next());
    }

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
