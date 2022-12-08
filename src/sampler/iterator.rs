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
}
