use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag to control deterministic behavior at runtime.
///
/// Defaults to `false`, meaning ddnnife will use non-deterministic operations.
#[cfg(not(test))]
static DETERMINISTIC: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
static DETERMINISTIC: AtomicBool = AtomicBool::new(true);

/// Global value for seeding random number generators.
///
/// Only has an effect when [DETERMINISTIC] is `true`.
static DETERMINISTIC_SEED: OnceLock<u64> = OnceLock::new();

/// Enables or disables deterministic operations.
///
/// Actually using deterministic operations requires a seed to be set.
/// See [set_seed].
#[inline]
pub fn set_deterministic(enable: bool) {
    DETERMINISTIC.store(enable, Ordering::Relaxed);
}

/// Returns whether deterministic mode is currently enabled.
#[inline]
pub fn is_deterministic() -> bool {
    DETERMINISTIC.load(Ordering::Relaxed)
}

/// Sets the seed to use for deterministic operations.
/// Does **not** implicitly enable determinism.
///
/// Can only be called once.
///
/// # Panics
///
/// Panics when called more than once.
#[inline]
pub fn set_seed(seed: u64) {
    DETERMINISTIC_SEED
        .set(seed)
        .expect("Seed can only be set once");
}

/// Returns the seed to use for random number generators.
///
/// `None` when no seed has been set yet.
#[inline]
#[cfg(not(test))]
pub fn get_seed() -> Option<u64> {
    DETERMINISTIC_SEED.get().copied()
}

#[inline]
#[cfg(test)]
pub fn get_seed() -> Option<u64> {
    Some(42)
}
