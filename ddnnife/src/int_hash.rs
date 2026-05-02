use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};

pub fn map_with_capacity<K, V>(capacity: usize) -> IntMap<K, V>
where
    K: Hash + Eq,
{
    let mut map = IntMap::default();
    map.reserve(capacity);
    map
}

/// A `HashMap` not doing any hashing but using keys as-is.
pub type IntMap<K, V> = HashMap<K, V, BuildIntHasher>;

/// A `HashSet` not doing any hashing but using keys as-is.
pub type IntSet<K> = HashSet<K, BuildIntHasher>;

/// A hasher for integer maps and sets.
///
/// Does not do any hashing but instead returns input values as-is.
/// Currently only accepts `usize`, `u32` and `i32`.
#[derive(Default)]
pub struct IntHasher {
    hash: u64,
}

impl Hasher for IntHasher {
    fn write(&mut self, _bytes: &[u8]) {
        panic!("IntHasher only takes usize, u32 and i32 inputs");
    }

    fn write_usize(&mut self, input: usize) {
        self.hash = input as u64;
    }

    fn write_u32(&mut self, input: u32) {
        self.hash = input as u64;
    }

    fn write_i32(&mut self, input: i32) {
        self.hash = input as u64;
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}

/// A builder for `IntHasher`.
#[derive(Default, Clone, Copy)]
pub struct BuildIntHasher;

impl BuildHasher for BuildIntHasher {
    type Hasher = IntHasher;

    fn build_hasher(&self) -> Self::Hasher {
        IntHasher::default()
    }
}

#[cfg(test)]
mod tests {
    use super::{IntHasher, IntMap, IntSet};
    use std::hash::{Hash, Hasher};

    #[test]
    fn deterministic_same() {
        let mut hasher1 = IntHasher::default();
        42.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = IntHasher::default();
        42.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2, "Same input must produce same hash");
    }

    #[test]
    fn deterministic_not_same() {
        let mut hasher1 = IntHasher::default();
        1.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = IntHasher::default();
        42.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_ne!(
            hash1, hash2,
            "Different inputs must produce different hashes"
        );
    }

    #[test]
    fn map_get() {
        let mut map = IntMap::default();
        map.insert(42, "answer");
        assert_eq!(map.get(&42), Some(&"answer"));
        assert_eq!(map.get(&0), None);
    }

    #[test]
    fn set_contains() {
        let mut set = IntSet::default();
        set.insert(42);
        assert!(set.contains(&42));
        assert!(!set.contains(&0));
    }

    #[test]
    fn set_deterministic_iteration() {
        let mut a = IntSet::default();
        a.insert(3);
        a.insert(1);
        a.insert(2);

        let mut b = IntSet::default();
        b.insert(3);
        b.insert(1);
        b.insert(2);

        let a_vec: Vec<i32> = a.into_iter().collect();
        let b_vec: Vec<i32> = b.into_iter().collect();

        assert_eq!(a_vec, b_vec);
    }
}
