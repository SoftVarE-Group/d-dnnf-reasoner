use super::{Config, Sample, SatWrapper};
use crate::ddnnf::extended_ddnnf::ExtendedDdnnf;

/// Covering strategy that uses the sat state caching.
pub fn cover_with_caching(
    sample: &mut Sample,
    interaction: &[i32],
    sat_solver: &SatWrapper,
    node_id: usize,
    number_of_vars: usize,
) {
    debug_assert!(
        !interaction.contains(&0),
        "Interaction contains undecided literals: {interaction:?}"
    );

    if sample.covers(interaction) {
        return; // already covered
    }

    let mut interaction_sat_state = sat_solver.new_state();

    if !sat_solver.is_sat_in_subgraph_cached(interaction, node_id, &mut interaction_sat_state) {
        return; // interaction invalid
    }

    if let Some(index) = cover(sample, interaction, sat_solver, node_id) {
        // move config to the complete configs if it is complete now
        let config = sample.partial_configs.get(index).expect("");
        if sample.is_config_complete(config) {
            let config = sample.partial_configs.swap_remove(index);
            sample.add_complete(config);
        }
    } else {
        // no config found - create new config
        let mut config = Config::from(interaction, number_of_vars);
        config.set_sat_state(interaction_sat_state);
        sample.add(config);
    }
}

/// Covering strategy that uses the sat state caching.
pub fn cover_with_caching_twise(
    sample: &mut Sample,
    interaction: &[i32],
    sat_solver: &SatWrapper,
    node_id: usize,
    number_of_vars: usize,
) {
    debug_assert!(
        !interaction.contains(&0),
        "Interaction contains undecided literals: {interaction:?}",
    );

    if sample.covers(interaction) {
        return; // already covered
    }

    if let Some(index) = cover(sample, interaction, sat_solver, node_id) {
        // move config to the complete configs if it is complete now
        let config = sample.partial_configs.get(index).expect("");
        if sample.is_config_complete(config) {
            let config = sample.partial_configs.swap_remove(index);
            sample.add_complete(config);
        }
    } else {
        // no config found - create new config
        let mut interaction_sat_state = sat_solver.new_state();
        sat_solver.is_sat_in_subgraph_cached(interaction, node_id, &mut interaction_sat_state);
        let mut config = Config::from(interaction, number_of_vars);
        config.set_sat_state(interaction_sat_state);
        sample.add(config);
    }
}

pub(super) fn cover_with_caching_sorted(
    sample: &mut Sample,
    interaction: &[i32],
    sat_solver: &SatWrapper,
    node_id: usize,
    number_of_vars: usize,
    ext_ddnnf: &ExtendedDdnnf,
) {
    debug_assert!(!interaction.contains(&0));
    if sample.covers(interaction) {
        return; // already covered
    }
    let mut interaction_sat_state = sat_solver.new_state();
    if !sat_solver.is_sat_in_subgraph_cached(interaction, node_id, &mut interaction_sat_state) {
        return; // interaction invalid
    }

    let fond_idx = cover(sample, interaction, sat_solver, node_id);

    if let Some(index) = fond_idx {
        let config = sample.partial_configs.get(index).expect("");
        if sample.is_config_complete(config) {
            // config is complete - remove it from the partial configs and insert it into the complete configs at the right index
            let config = sample.partial_configs.remove(index);
            ext_ddnnf.insert_config_sorted(config, &mut sample.complete_configs);
        } else {
            // config is still partial - shift it up or down until it is at the right index in partial configs
            let config = &sample.partial_configs[index];
            let config_val = ext_ddnnf.get_average_objective_fn_val_of_config(config);
            let mut current_idx = index;

            while current_idx > 0
                && config_val
                    > ext_ddnnf.get_average_objective_fn_val_of_config(
                        &sample.partial_configs[current_idx - 1],
                    )
            {
                sample.partial_configs.swap(current_idx, current_idx - 1);
                current_idx -= 1;
            }

            while current_idx < sample.partial_configs.len() - 1
                && config_val
                    < ext_ddnnf.get_average_objective_fn_val_of_config(
                        &sample.partial_configs[current_idx + 1],
                    )
            {
                sample.partial_configs.swap(current_idx, current_idx + 1);
                current_idx += 1;
            }
        }
    } else {
        // no config found - create new config and insert it into the complete/partial configs at the right index
        let mut config = Config::from(interaction, number_of_vars);
        config.set_sat_state(interaction_sat_state);

        if sample.is_config_complete(&config) {
            ext_ddnnf.insert_config_sorted(config, &mut sample.complete_configs);
        } else {
            ext_ddnnf.insert_config_sorted(config, &mut sample.partial_configs);
        }
    }
}

fn cover(
    sample: &mut Sample,
    interaction: &[i32],
    sat_solver: &SatWrapper,
    node_id: usize,
) -> Option<usize> {
    for (index, config) in sample.partial_configs.iter_mut().enumerate() {
        if config.conflicts_with(interaction) {
            continue;
        }

        config.update_sat_state(sat_solver, node_id);

        // clone sat state so that we don't change the state that is cached in the config
        let mut sat_state = config
            .get_sat_state()
            .cloned()
            .expect("sat state should exist because update_sat_state() was called before");

        if sat_solver.is_sat_in_subgraph_cached(interaction, node_id, &mut sat_state) {
            // we found a config - extend config with interaction and update sat state
            config.extend(interaction.iter().cloned());
            config.set_sat_state(sat_state);
            return Some(index);
        }
    }

    None
}
