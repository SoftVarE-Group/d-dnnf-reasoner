mod config;
mod covering_strategies;
mod sample;
mod sample_merger;
mod sampling_result;
mod sat_wrapper;
mod t_iterator;
mod t_wise_sampler;

use crate::util::format_vec;
use crate::Ddnnf;
pub use config::Config;
pub use sample::Sample;
use sample_merger::similarity_merger::SimilarityMerger;
use sample_merger::zipping_merger::ZippingMerger;
use sampling_result::SamplingResult;
use sat_wrapper::SatWrapper;
use std::path::Path;
use std::{fs, io, iter};
use t_wise_sampler::TWiseSampler;

#[cfg_attr(feature = "uniffi", uniffi::export)]
impl Ddnnf {
    /// Generates samples so that all t-wise interactions between literals are covered.
    #[cfg_attr(feature = "uniffi", uniffi::method)]
    pub fn sample_t_wise(&self, t: usize) -> SamplingResult {
        // Setup everything needed for the sampling process.
        let sat_solver = SatWrapper::new(self);
        let and_merger = ZippingMerger {
            t,
            sat_solver: &sat_solver,
            ddnnf: self,
        };
        let or_merger = SimilarityMerger { t };

        TWiseSampler::new(self, and_merger, or_merger).sample(t)
    }
}

pub fn save_sample_to_file(sampling_result: &SamplingResult, file_path: &str) -> io::Result<()> {
    let file_path = Path::new(file_path);
    if let Some(dir) = file_path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut wtr = csv::Writer::from_path(file_path)?;

    match sampling_result {
        /*
        Writing "true" and "false" to the file does not really fit the format of the file but we
        want to somehow distinguish between true and false sampling results.
        True means that the feature model contains no variables and therefore an empty sample
        covers all t-wise interactions.
        False means that the feature model is void.
        */
        SamplingResult::Empty => wtr.write_record(iter::once("true"))?,
        SamplingResult::Void => wtr.write_record(iter::once("false"))?,
        SamplingResult::ResultWithSample(sample) => {
            for (index, config) in sample.iter().enumerate() {
                wtr.write_record([index.to_string(), format_vec(config.get_literals().iter())])?;
            }
        }
    }

    wtr.flush()
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::{parser::build_ddnnf, Ddnnf};

    #[test]
    fn t_wise_sampling_validity() {
        let mut vp9: Ddnnf = build_ddnnf("tests/data/VP9_d4.nnf", Some(42));
        let mut auto1: Ddnnf = build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));

        check_validity_samplingresult(&mut vp9, 1);
        check_validity_samplingresult(&mut vp9, 2);
        check_validity_samplingresult(&mut vp9, 3);
        check_validity_samplingresult(&mut vp9, 4);
        check_validity_samplingresult(&mut auto1, 1);

        fn check_validity_samplingresult(ddnnf: &mut Ddnnf, t: usize) {
            let t_wise_samples = ddnnf.sample_t_wise(t);
            let configs = t_wise_samples
                .optional()
                .unwrap()
                .iter()
                .map(|config| config.get_literals())
                .collect_vec();

            for config in configs.iter() {
                // every config must be complete and satisfiable
                assert_eq!(
                    ddnnf.number_of_variables as usize,
                    config.len(),
                    "config is not complete"
                );
                assert!(ddnnf.sat(config));
            }

            let mut possible_features = (-(ddnnf.number_of_variables as i32)
                ..=ddnnf.number_of_variables as i32)
                .collect_vec();
            possible_features.remove(ddnnf.number_of_variables as usize); // remove the 0
            for combi in possible_features.into_iter().combinations(t) {
                // checks if the pair can be found in at least one of the samples
                let combi_exists = |combi: &[i32]| -> bool {
                    configs.iter().any(|config| {
                        combi
                            .iter()
                            .all(|&f| config[f.unsigned_abs() as usize - 1] == f)
                    })
                };

                assert!(
                    combi_exists(&combi) || !ddnnf.sat(&combi),
                    "combination: {:?} can neither be convered with samples nor is it unsat",
                    combi
                )
            }
        }
    }
}
