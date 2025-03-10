use crate::ddnnf::Ddnnf;
use ddnnife::ddnnf::anomalies::t_wise_sampling;
use std::collections::HashSet;

#[uniffi::export]
impl Ddnnf {
    /// Generates various statistics about this d-DNNF.
    #[uniffi::method]
    pub fn sample_t_wise(&self, t: usize) -> SamplingResult {
        self.0.sample_t_wise(t).into()
    }
}

pub struct Config(Vec<i32>);

uniffi::custom_newtype!(Config, Vec<i32>);

impl From<t_wise_sampling::Config> for Config {
    fn from(value: t_wise_sampling::Config) -> Self {
        Config(value.literals)
    }
}

#[derive(uniffi::Record)]
pub struct Sample {
    /// Configs that contain all variables of this sample
    pub complete_configs: Vec<Config>,
    /// Configs that do not contain all variables of this sample
    pub partial_configs: Vec<Config>,
    /// The variables that Configs of this sample may contain
    pub vars: HashSet<u32>,
    /// The literals that actually occur in this sample
    pub literals: Vec<i32>,
}

impl From<t_wise_sampling::Sample> for Sample {
    fn from(value: t_wise_sampling::Sample) -> Self {
        Self {
            complete_configs: value
                .complete_configs
                .into_iter()
                .map(Config::from)
                .collect(),
            partial_configs: value
                .partial_configs
                .into_iter()
                .map(Config::from)
                .collect(),
            vars: value.vars,
            literals: value.literals,
        }
    }
}

#[derive(uniffi::Enum)]
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

impl From<t_wise_sampling::SamplingResult> for SamplingResult {
    fn from(value: t_wise_sampling::SamplingResult) -> Self {
        match value {
            t_wise_sampling::SamplingResult::Empty => Self::Empty,
            t_wise_sampling::SamplingResult::Void => Self::Void,
            t_wise_sampling::SamplingResult::ResultWithSample(sample) => {
                Self::ResultWithSample(sample.into())
            }
        }
    }
}
