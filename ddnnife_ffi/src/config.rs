use ddnnife::config;

/// Enables or disables deterministic operations.
///
/// Actually using deterministic operations requires a seed to be set.
#[uniffi::export]
pub fn set_deterministic(enable: bool) {
    config::set_deterministic(enable);
}

/// Returns whether deterministic mode is currently enabled.
#[uniffi::export]
pub fn is_deterministic() -> bool {
    config::is_deterministic()
}

/// Sets the seed to use for deterministic operations.
/// Does **not** implicitly enable determinism.
///
/// Can only be called once.
///
/// # Panics
///
/// Panics when called more than once.
#[uniffi::export]
pub fn set_seed(seed: u64) {
    config::set_seed(seed);
}

/// Returns the seed to use for random number generators.
///
/// Not valid when no seed has been set yet.
#[uniffi::export]
pub fn get_seed() -> Option<u64> {
    config::get_seed()
}
