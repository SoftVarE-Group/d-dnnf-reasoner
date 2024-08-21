use super::Sample;
use crate::util::format_vec_separated_by;
use std::fmt;

/// An abstraction over the result of sampling as it might be invalid or empty.
#[cfg_attr(feature = "uniffi", derive(uniffi::Enum))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplingResult {
    /// An empty result that is *valid* (a regular sample containing 0 configurations).
    /// This is used to indicate that a subgraph evaluates to true.
    Empty,
    /// An empty result that is *invalid*.
    /// This is used to indicate that a subgraph evaluates to false.
    Void,
    /// A *valid* result having a regular sample.
    ResultWithSample(Sample),
}

impl SamplingResult {
    /// Converts a sampling result into an optional sample.
    pub fn optional(&self) -> Option<&Sample> {
        match self {
            SamplingResult::ResultWithSample(sample) => Some(sample),
            _ => None,
        }
    }

    /// Determines how many configuration the sample contains.
    pub fn len(&self) -> usize {
        match self {
            SamplingResult::Empty | SamplingResult::Void => 0,
            SamplingResult::ResultWithSample(sample) => sample.len(),
        }
    }

    /// Determines whether the sample contains no configurations.
    pub fn is_empty(&self) -> bool {
        match self {
            SamplingResult::Empty | SamplingResult::Void => true,
            SamplingResult::ResultWithSample(sample) => sample.is_empty(),
        }
    }
}

impl From<Sample> for SamplingResult {
    fn from(value: Sample) -> Self {
        if value.is_empty() {
            return SamplingResult::Empty;
        }

        SamplingResult::ResultWithSample(value)
    }
}

impl fmt::Display for SamplingResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplingResult::Empty | SamplingResult::Void => write!(f, ""),
            SamplingResult::ResultWithSample(sample) => {
                write!(f, "{}", format_vec_separated_by(sample.iter(), ";"))
            }
        }
    }
}
