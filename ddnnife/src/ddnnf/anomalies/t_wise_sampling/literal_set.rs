use bitvec::{bitvec, vec::BitVec};

/// A helper data structure to calculate the intersection length between sets
/// of literals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiteralSet(BitVec);

impl LiteralSet {
    /// Creates a new literal set from the given literals.
    ///
    /// # Panics
    ///
    /// Might panic if the number of variables actually encountered is larger than `n_variables`.
    pub fn new<L: Iterator<Item = i32>>(literals: L, n_variables: usize) -> Self {
        let mut set = bitvec![0; n_variables * 2];

        // Mark each literal in the bitset.
        literals
            .filter(|&literal| literal != 0)
            .for_each(|literal| {
                let abs = literal.unsigned_abs() as usize - 1;

                // Generate an index based on the phase of the literal.
                let index = match literal.signum() {
                    // Positive literals go first.
                    1 => abs,
                    // Negative literals come after the positive ones, shifted by half of the sets length.
                    -1 => abs + n_variables,
                    _ => unreachable!(),
                };

                set.set(index, true);
            });

        // Allow the proper usage of the underlying memory later on by clearing any uninitialized bits.
        set.set_uninitialized(false);

        Self(set)
    }

    /// Calculates the length of the intersection between this literal set and another.
    ///
    /// # Note
    ///
    /// Requires both literal sets to be of the **same length**.
    pub fn intersection_length(&self, other: &Self) -> u32 {
        // Operate on words rather than the whole bitset at once.
        self.0
            // Take the underlying memory ...
            .as_raw_slice()
            // word by word.
            .iter()
            // Combine it with the other set.
            .zip(other.0.as_raw_slice().iter())
            // Calculate the bitwise AND between the two words and count the ones.
            .map(|(x, y)| (x & y).count_ones())
            .sum()
    }
}
