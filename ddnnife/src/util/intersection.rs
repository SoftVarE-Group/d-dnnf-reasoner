use std::cmp::Ordering;

/// Calculates the length of the intersection between `a` and `b`, given that both are **sorted**.
pub fn intersection_sorted<T: Ord>(a: &[T], b: &[T]) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut count = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }

    count
}
