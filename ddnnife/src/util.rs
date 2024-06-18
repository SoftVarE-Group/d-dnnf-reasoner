use rand::Rng;

#[cfg(any(feature = "deterministic", test))]
use rand::prelude::{SeedableRng, StdRng};

#[cfg(not(any(feature = "deterministic", test)))]
use rand::thread_rng;

pub fn format_vec_separated_by<T: ToString>(
    vals: impl Iterator<Item = T>,
    separator: &str,
) -> String {
    vals.map(|v| v.to_string())
        .collect::<Vec<String>>()
        .join(separator)
}

pub fn format_vec<T: ToString>(vals: impl Iterator<Item = T>) -> String {
    format_vec_separated_by(vals, " ")
}

pub fn format_vec_vec<T>(vals: impl Iterator<Item = T>) -> String
where
    T: IntoIterator,
    T::Item: ToString,
{
    vals.map(|res| format_vec(res.into_iter()))
        .collect::<Vec<String>>()
        .join(";")
}

#[cfg(any(feature = "deterministic", test))]
#[inline]
pub fn rng() -> impl Rng {
    StdRng::seed_from_u64(42)
}

#[cfg(not(any(feature = "deterministic", test)))]
#[inline]
pub fn rng() -> impl Rng {
    thread_rng()
}
