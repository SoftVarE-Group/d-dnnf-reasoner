use crate::config;
use rand::{SeedableRng, rngs::SmallRng};
use std::sync::{LazyLock, OnceLock, RwLock, RwLockWriteGuard};

static RNG: LazyLock<RwLock<SmallRng>> = LazyLock::new(|| RwLock::new(SmallRng::from_os_rng()));
static RNG_DETERMINISTIC: OnceLock<RwLock<SmallRng>> = OnceLock::new();

/// Returns a handle to a random number generator.
///
/// Uses a small but **not** cryptographically safe RNG.
///
/// Depending on the corresponding [config] entries, a deterministic RNG
/// as specified by the seed or an actually random RNG is used.
#[inline]
pub fn rng<'a>() -> RwLockWriteGuard<'a, SmallRng> {
    if config::is_deterministic() {
        RNG_DETERMINISTIC
            .get_or_init(|| {
                RwLock::new(SmallRng::seed_from_u64(
                    config::get_seed()
                        .expect("Using deterministic operations requires a seed to be set"),
                ))
            })
            .write()
            .unwrap()
    } else {
        RNG.write().unwrap()
    }
}
