use num::BigInt;
use std::collections::HashSet;

uniffi::custom_type!(usize, u64, {
    remote,
    lower: |custom| custom as u64,
    try_lift: |builtin| Ok(builtin as usize),
});

uniffi::custom_type!(BigInt, Vec<u8>, {
    remote,
    lower: |bigint| bigint.to_signed_bytes_be(),
    try_lift: |vec| Ok(BigInt::from_signed_bytes_be(&vec)),
});

type HashSetu32 = HashSet<u32>;

uniffi::custom_type!(HashSetu32, Vec<u32>, {
    remote,
    lower: |hashset| hashset.into_iter().collect(),
    try_lift: |vec| Ok(vec.into_iter().collect()),
});

type HashSeti32 = HashSet<i32>;

uniffi::custom_type!(HashSeti32, Vec<i32>, {
    remote,
    lower: |hashset| hashset.into_iter().collect(),
    try_lift: |vec| Ok(vec.into_iter().collect()),
});
