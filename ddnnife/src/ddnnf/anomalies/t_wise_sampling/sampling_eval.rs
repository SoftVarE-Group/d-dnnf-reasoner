use std::cmp::min;
use std::collections::HashMap;
use itertools::Itertools;
use rug::ops::Pow;
use crate::ddnnf::anomalies::t_wise_sampling::data_structure::Sample;
use crate::ddnnf::anomalies::t_wise_sampling::t_iterator::TInteractionIter;
use streaming_iterator::StreamingIterator;
use crate::ddnnf::extended_ddnnf::ExtendedDdnnf;

#[derive(Copy, Clone, Debug)]
pub struct SamplingEvalResult {/// The amount of configs in the sample
sample_size: usize,

    /// The combined value of all configs in the sample
    config_values_sum: f64,
    /// The average value of all configs in the sample
    config_values_avg: f64,
    /// The variance of the config values in the sample
    config_values_var: f64,

    additional_interactions_count: usize,
    additional_interactions_value_sum: f64,
    additional_interactions_value_avg: f64,

    /// The total amount of selected features of configs in the sample
    n_selected_features_sum: usize,
    /// The average amount of selected features of configs in the sample
    n_selected_features_avg: f64,
    /// The variance of the amount of selected features of configs in the sample
    n_selected_features_var: f64
}



pub fn eval_sample(sample: &Sample, ext_ddnnf: &ExtendedDdnnf, t: usize) -> Result<SamplingEvalResult, String> {

    if let Err(error) = check_sample_validity(sample, ext_ddnnf, t) {
        return Err(error)
    }

    let sample_size = sample.len();

    let config_values_sum = sample.iter()
        .map(|config| ext_ddnnf.get_objective_fn_val_of_config(config))
        .sum();
    let config_values_avg: f64 = config_values_sum / sample_size as f64;
    let config_values_var = sample.iter()
        .map(|config| ext_ddnnf.get_objective_fn_val_of_config(config))
        .map(|config_val| (config_val - config_values_avg).pow(2))
        .sum::<f64>() / sample_size as f64;

    let n_selected_features_sum = sample.iter()
        .map(|config|
            config.get_decided_literals()
                .filter(|&literal| literal > 0)
                .count())
        .sum();
    let n_selected_features_avg: f64 = n_selected_features_sum as f64 / sample_size as f64;
    let n_selected_features_var = sample.iter()
        .map(|config|
            config.get_decided_literals()
                .filter(|&literal| literal > 0)
                .count())
        .map(|config_n_selected_features| (config_n_selected_features as f64 - n_selected_features_avg).pow(2))
        .sum::<f64>() / sample_size as f64;

    let (additional_interactions_count, additional_interactions_value_sum, additional_interactions_value_avg) = eval_additional_interactions(sample, ext_ddnnf, t);

    Ok(
        SamplingEvalResult {
            sample_size,
            config_values_sum,
            config_values_avg,
            config_values_var,
            additional_interactions_count,
            additional_interactions_value_sum,
            additional_interactions_value_avg,
            n_selected_features_sum,
            n_selected_features_avg,
            n_selected_features_var
        }
    )
}

pub fn eval_additional_interactions(sample: &Sample, ext_ddnnf: &ExtendedDdnnf, t: usize) -> (usize, f64, f64) {
    let mut interaction_counter: HashMap<Vec<i32>, usize> = HashMap::new();

    for config in sample.iter() {
        TInteractionIter::new(config.get_literals(), min(config.get_n_decided_literals(), t))
            .for_each(|interaction| {
                if !interaction_counter.contains_key(interaction) {
                    interaction_counter.insert(interaction.to_vec(), 0);
                }
                *interaction_counter.get_mut(interaction).unwrap() += 1;
            });
    }

    let additional_interactions_count = interaction_counter.values().sum::<usize>() - interaction_counter.len();
    let additional_interactions_value_sum = interaction_counter.iter()
        .map(
            |(interaction, count)|
                ext_ddnnf.get_objective_fn_val_of_literals(&interaction[..]) * (count - 1) as f64
        )
        .sum();
    let additional_interactions_value_avg = additional_interactions_value_sum / additional_interactions_count as f64;

    (additional_interactions_count, additional_interactions_value_sum, additional_interactions_value_avg)
}


pub fn check_sample_validity(sample: &Sample, ext_ddnnf: &ExtendedDdnnf, t: usize) -> Result<(), String> {
    for literals in sample.iter().map(|config| config.get_decided_literals().collect_vec()) {
        if ext_ddnnf.ddnnf.number_of_variables as usize != literals.len() {
            return Err(String::from("Sample contains incomplete config."));
        }

        if !ext_ddnnf.ddnnf.sat_immutable(&literals[..]) {
            return Err(String::from("Sample contains invalid config."));
        }
    }

    let all_literals = (-(ext_ddnnf.ddnnf.number_of_variables as i32)..=ext_ddnnf.ddnnf.number_of_variables as i32)
        .filter(|&literal| literal != 0)
        .collect_vec();

    if TInteractionIter::new(&all_literals[..], t)
        .filter(|interaction| ext_ddnnf.ddnnf.sat_immutable(interaction))
        .any(|interaction| !sample.covers(interaction))
    {
        return Err(String::from("Sample does not have t-wise coverage."));
    }

    Ok(())
}


#[cfg(test)]
mod test {
    use std::time::Instant;
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::{Ddnnf, parser::build_ddnnf};
    use crate::ddnnf::anomalies::t_wise_sampling::sampling_eval::eval_sample;
    use crate::ddnnf::extended_ddnnf::ExtendedDdnnf;

    #[test]
    fn compare_sampling() {
        let mut rng = StdRng::seed_from_u64(1337);
        let n_features = 998;
        let vp9: Ddnnf = build_ddnnf(Path::new("tests/data/busybox_1_28_0.dimacs.nnf"), Some(n_features));
        // let n_features = 42;
        // let vp9: Ddnnf = build_ddnnf(Path::new("tests/data/VP9_d4.nnf"), Some(n_features));
        let mut ext_ddnnf = ExtendedDdnnf {
            ddnnf: vp9,
            attrs: Default::default(),
            objective_fn_vals: Some(vec![rng.gen_range(-100..100) as f64; n_features as usize]),
        };


        for t in 2..=2 {
            println!("\n\n\n#### t = {t}:");

            let now = Instant::now();
            let normal_sample_result = ext_ddnnf.ddnnf.sample_t_wise(t).get_sample().unwrap().clone();
            let normal_elapsed = now.elapsed();
            println!("\nNormal t-wise sampling:");
            println!("time elapsed: {}", normal_elapsed.as_millis());
            println!("{:?}", eval_sample(&normal_sample_result, &ext_ddnnf, t));

            // let now = Instant::now();
            // let optimal_sample_result = ext_ddnnf.sample_t_wise(t).get_sample().unwrap().clone();
            // let optimal_elapsed = now.elapsed();
            // println!("\nOptimized t-wise sampling:");
            // println!("time elapsed: {}", optimal_elapsed.as_millis());
            // println!("{:?}", eval_sample(&optimal_sample_result, &ext_ddnnf, t));

            // let now = Instant::now();
            // let yasa_sample_result = ext_ddnnf.sample_t_wise_yasa(t);
            // let yasa_elapsed = now.elapsed();
            // println!("\nYASA t-wise sampling:");
            // println!("time elapsed: {}", yasa_elapsed.as_millis());
            // println!("{:?}", eval_sample(&yasa_sample_result, &ext_ddnnf, t));
        }

    }
}
