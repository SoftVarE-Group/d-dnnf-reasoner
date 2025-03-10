use num::BigInt;

#[cfg(feature = "d4")]
#[uniffi::export]
pub fn count(input: String) -> BigInt {
    ddnnife::util::count(input)
}

#[cfg(feature = "d4")]
#[uniffi::export]
pub fn count_projected(input: String) -> BigInt {
    ddnnife::util::count_projected(input)
}
