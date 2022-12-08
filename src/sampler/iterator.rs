/// Creates an iterator for pair wise (t=2) interactions.
/// The iteration order is lexicographic.
fn lexicographic_pair_wise_iter(
    number_of_vars: i32,
) -> impl Iterator<Item=Vec<i32>> {
    (1..=number_of_vars)
        .into_iter()
        .flat_map(move |i| {
            (i + 1..=number_of_vars).into_iter().map(move |j| (i, j))
        })
        .flat_map(|(i, j)| {
            vec![vec![i, j], vec![i, -j], vec![-i, j], vec![-i, -j]].into_iter()
        })
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_pair_wise_lexicographic() {
        let mut iter = lexicographic_pair_wise_iter(3);

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
