use std::iter::{FromIterator};
use std::{path::Path};

use itertools::Itertools;
use rand::seq::SliceRandom;
use rand_distr::{WeightedAliasIndex, Distribution, Binomial};
use rand_pcg::{Pcg32, Lcg64Xsh32};
use rand::{SeedableRng};

use rug::{Assign, Rational, Integer};

use crate::Ddnnf;
use crate::parser::persisting::write_ddnnf;
use super::node::NodeType::*;

impl Ddnnf {
    /// error codes:
    /// E1 Operation is not yet supported
    /// E2 Operation does not exist. Neither now nor in the future
    /// E3 Parse error
    /// E4 Syntax error
    /// E5 Operation was not able to be done, because of wrong input
    /// E6 File or path error
    pub fn handle_stream_msg(&mut self, msg: &str) -> String {
        let args: Vec<&str> = msg.split_whitespace().collect();
        if args.is_empty() {
            return String::from("E4 error: got an empty msg");
        }

        let mut param_index = 1;

        let mut params = Vec::new();
        let mut values = Vec::new();
        let mut seed = 42;
        let mut limit = 1;
        let mut path = Path::new("");

        // go through all possible extra values that can be provided til
        // either there are no more or we can't parse anymore
        while param_index < args.len() {
            param_index += 1;
            match args[param_index-1] {
                "a" | "assumptions" => {
                    params = match get_numbers(&args[param_index..], self.number_of_variables) {
                        Ok(v) => v,
                        Err(e) => return e,
                    };
                    param_index += params.len();
                },
                "v" | "variables" => {
                    values = match get_numbers(&args[param_index..], self.number_of_variables) {
                        Ok(v) => v,
                        Err(e) => return e,
                    };
                    param_index += values.len();
                },
                "seed" | "s" | "limit" | "l" | "path" | "p" => {
                    if param_index < args.len() {
                        match args[param_index-1] {
                            "seed" | "s" => { seed = match args[param_index].parse::<u64>() {
                                    Ok(x) => x,
                                    Err(e) => return format!("E3 error: {}", e),
                                };
                                param_index += 1;
                            },
                            "limit" | "l" => {
                                limit = match args[param_index].parse::<usize>() {
                                    Ok(x) => x,
                                    Err(e) => return format!("E3 error: {}", e),
                                };
                                param_index += 1;
                            },
                            _ => { 
                                // has to be path because of the outer patter match
                                // we use a wildcard to satisfy the rust compiler
                                path = Path::new(args[param_index]);
                                param_index += 1;
                            },
                        }
                    } else {
                        return format!("E4 error: param \"{}\" was used, but no value supplied", args[param_index-1]);
                    }
                },
                other => return format!("E4 error: the option \"{}\" is not valid in this context", other),
            }
        }

        match args[0] {
            "core" => op_with_assumptions_and_vars(
                |d, assumptions, vars| {
                    if vars {
                        let could_be_core = assumptions.pop().unwrap();
                        let without_cf = Ddnnf::execute_query(d, &assumptions);
                        assumptions.push(could_be_core);
                        let with_cf = Ddnnf::execute_query(d, &assumptions);

                        if with_cf == without_cf {
                            Some(could_be_core.to_string())
                        } else {
                            None
                        }
                    } else {
                        if assumptions.is_empty() {
                            let mut core = Vec::from_iter(&d.core);
                            let dead = &d.dead.iter().map(|v| -v).collect::<Vec<i32>>();
                            core.extend(dead);
                            core.sort_by(|a, b| a.abs().cmp(&b.abs()));
                            Some(format_vec(core.iter()))
                        } else {
                            let mut core = Vec::new();
                            let reference = Ddnnf::execute_query(d, &assumptions);
                            for i in 1_i32..=d.number_of_variables as i32 {
                                assumptions.push(i);
                                let inter = Ddnnf::execute_query(d, &assumptions);
                                if reference == inter {
                                    core.push(i);
                                }
                                if inter == 0 {
                                    core.push(-i);
                                }
                                assumptions.pop();
                            }
                            Some(format_vec(core.iter()))
                        }
                    }
                },
                self,
                &mut params,
                &values,
            ),
            "count" => op_with_assumptions_and_vars(
                |d, x, _| Some(Ddnnf::execute_query(d, x)),
                self,
                &mut params,
                &values
            ),
            "sat" => op_with_assumptions_and_vars(
                |d, x, _| Some(Ddnnf::is_sat_for(d, x)),
                self,
                &mut params,
                &values
            ),
            "enum" => {
                let configs = self.enumerate(&mut params, limit);
                match configs {
                    Some(s) => format_vec_vec(s.iter()),
                    None => String::from("E5 error: with the assumptions, the ddnnf is not satisfiable. Hence, there exist no valid sample configurations"),
                }
            },
            "random" => {
                let samples = self.uniform_random_sampling(&mut params, limit, seed);
                match samples {
                    Some(s) => format_vec_vec(s.iter()),
                    None => String::from("E5 error: with the assumptions, the ddnnf is not satisfiable. Hence, there exist no valid sample configurations"),
                }
            }
            "exit" => String::from("exit"),
            "save" => {
                if path.to_str().unwrap() == "" {
                    return String::from("E6 error: no file path was supplied");
                }
                if !path.is_absolute() {
                    return String::from("E6 error: file path is not absolute, but has to be");
                }
                match write_ddnnf(self, path.to_str().unwrap()) {
                    Ok(_) => String::from(""),
                    Err(e) => format!("E6 error: {} while trying to write ddnnf to {}", e, path.to_str().unwrap()),
                }
            },
            "atomic" | "uni_random" | "t-wise_sampling" => {
                String::from("E1 error: not yet supported")
            }
            other => format!("E2 error: the operation \"{}\" is not supported", other),
        }
    }

    /// Creates satisfiable complete configurations for a ddnnf and given assumptions
    /// If the ddnnf on itself or in combination with the assumption is unsatisfiable,
    /// then we can not create any satisfiable configuration and simply return None.
    pub(crate) fn enumerate(&mut self, assumptions: &mut Vec<i32>, _amount: usize) -> Option<Vec<Vec<i32>>> {
        for node in self.nodes.iter_mut() {
            node.temp.assign(&node.count);
        }

        for literal in assumptions.iter() {
            match self.literals.get(&-literal) {
                Some(&x) => self.nodes[x].temp.assign(0),
                None => (),
            }
        }
        
        if self.execute_query(&assumptions) > 0 {
            let mut sample_list = 
                self.enumerate_node(&self.rt(), self.number_of_nodes-1);
            for sample in sample_list.iter_mut() {
                sample.sort_unstable_by_key(|f| f.abs());
            }
            return Some(sample_list);
        }
        None
    }

    // Handles a node appropiate depending on its kind to produce complete
    // satisfiable configurations
    fn enumerate_node(&self, amount: &Integer, index: usize) -> Vec<Vec<i32>> {
        let mut enumeration_list = Vec::new();
        if *amount == 0 { return enumeration_list; }
        match &self.nodes[index].ntype {
            And { children } => {
                let mut enumeration_child_lists = Vec::new();
                for &child in children {
                    let child_enum = self.enumerate_node(&self.nodes[child].temp, child);
                    if !child_enum.is_empty() {
                        enumeration_child_lists.push(child_enum);
                    }
                }
                // combine all combinations of children
                // example:
                //      enumeration_child_lists: Vec<Vec<Vec<i32>>>
                //          [[[1,2],[3]],[[4,5,-5]]]
                //      enumeration_list: Vec<Vec<i32>>
                //          [[1,2,4],[1,2,5],[1,2,-5],[3,4],[3,5],[3,-5]]
                enumeration_list = enumeration_child_lists
                    .into_iter()
                    .multi_cartesian_product()
                    .map(|elem| elem.into_iter().flatten().collect())
                    .collect();
            },
            Or { children } => {
                for &child in children {
                    let mut child_enum = self.enumerate_node(&self.nodes[child].temp, child);
                    if !child_enum.is_empty() {
                        enumeration_list.append(&mut child_enum);
                    }
                }
            },
            Literal { literal } => {
                enumeration_list.push(vec![*literal]);
            },
            _ => (),
        }
        enumeration_list
    }

    /// Generates amount many uniform random samples under a given set of assumptions and a seed.
    /// Each sample is sorted by the number of the features. Each sample is a complete configuration with #SAT of 1.
    /// If the ddnnf itself or in combination with the assumptions is unsatisfiable, None is returned. 
    pub(crate) fn uniform_random_sampling(&mut self, assumptions: &mut Vec<i32>, amount: usize, seed: u64) -> Option<Vec<Vec<i32>>> {
        for node in self.nodes.iter_mut() {
            node.temp.assign(&node.count);
        }
        
        if self.execute_query(&assumptions) > 0 {
            let mut sample_list = self.sample_node(amount, self.number_of_nodes-1, &mut Pcg32::seed_from_u64(seed));
            for sample in sample_list.iter_mut() {
                sample.sort_unstable_by_key(|f| f.abs());
            }
            return Some(sample_list);
        }
        None
    }

    // Performs the operations needed to generate random samples.
    // The algorithm is based upon KUS's uniform random sampling algorithm.
    fn sample_node(&self, amount: usize, index: usize, rng: &mut Lcg64Xsh32) -> Vec<Vec<i32>> {
        let mut sample_list = Vec::new();
        if amount == 0 { return sample_list; }
        match &self.nodes[index].ntype {
            And { children } => {
                for _ in 0..amount {
                    sample_list.push(Vec::new());
                }
                for &child in children {
                    let mut child_sample_list = self.sample_node(amount, child, rng);
                    // shuffle operation from KUS algorithm
                    child_sample_list.shuffle(rng);
                    
                    // stitch operation
                    for (index, sample) in child_sample_list.iter_mut().enumerate() {
                        sample_list[index].append(sample);
                    }
                }
            },
            Or { children } => {
                let mut pick_amount = vec![0; children.len()];
                let mut choices = Vec::new();
                let mut weights = Vec::new();
                
                // compute the probability of getting a sample of a child node
                let parent_count_as_float = Rational::from((&self.nodes[index].temp, 1));
                for child_index in 0..children.len() {
                    let child_count_as_float = Rational::from((&self.nodes[children[child_index]].temp, 1));
                    
                    // can't get a sample of a children with no more valid configuration
                    if child_count_as_float != 0 {
                        let child_amount = (&parent_count_as_float / child_count_as_float).to_f64() * amount as f64;
                        choices.push(child_index);
                        weights.push(child_amount);
                    }
                }

                // choice some sort of weighted distribution depending on the number of children with count > 0
                match weights.len() {
                    1 => pick_amount[choices[0]] += amount,
                    2 => {
                        let binomial_dist = Binomial::new(amount as u64, weights[0]/(weights[0] + weights[1])).unwrap();
                        pick_amount[choices[0]] += binomial_dist.sample(rng) as usize;
                        pick_amount[choices[1]] = amount - pick_amount[choices[0]];
                    },
                    _ => {
                        let weighted_dist = WeightedAliasIndex::new(weights).unwrap();
                        for _ in 0..amount {
                            pick_amount[choices[weighted_dist.sample(rng)]] += 1;
                        }
                    }
                }

                for &choice in choices.iter() {
                    sample_list.append(&mut self.sample_node(pick_amount[choice], children[choice], rng));
                }

                // add empty lists for child nodes that have a count of zero
                while sample_list.len() != amount {
                    sample_list.push(Vec::new());
                }

                sample_list.shuffle(rng);
            },
            Literal { literal } => {
                for _ in 0..amount {
                    sample_list.push(vec![*literal]);
                } 
            },
            _ => (),
        }
        sample_list
    }
}

fn op_with_assumptions_and_vars<T: ToString>(
    operation: fn(&mut Ddnnf, &mut Vec<i32>, bool) -> Option<T>,
    ddnnf: &mut Ddnnf,
    assumptions: &mut Vec<i32>,
    vars: &[i32],
) -> String {
    if vars.is_empty() {
        match operation(ddnnf, assumptions, false) {
            Some(v) => return v.to_string(),
            None => (),
        }
    }

    let mut response = Vec::new();
    for var in vars {
        assumptions.push(*var);
        match operation(ddnnf, assumptions, true) {
            Some(v) => response.push(v.to_string()),
            None => (),
        }
        assumptions.pop();
    }

    response.join(";")
}

fn format_vec<T: ToString>(vals: impl Iterator<Item = T>) -> String {
    vals.map(|v| v.to_string()).collect::<Vec<String>>().join(" ")
}

fn format_vec_vec<T>(vals: impl Iterator<Item = T>) -> String
    where
    T: IntoIterator,
    T::Item: ToString,
{
    vals.map(|res| format_vec(res.into_iter()))
    .collect::<Vec<String>>()
    .join(";")
}

fn get_numbers(params: &[&str], boundary: u32) -> Result<Vec<i32>, String> {
    let mut numbers = Vec::new();
    for param in params.iter() {
        if param.chars().any(|c| c.is_alphabetic()) {
            return Ok(numbers);
        }
        match param.parse::<i32>() {
            Ok(num) => numbers.push(num),
            Err(e) => return Err(format!("E3 error: {}", e)),
        }
    }
    if numbers.is_empty() {
        return Err(String::from(
            "E4 error: option used but there was no value supplied",
        ));
    }

    if numbers.iter().any(|v| v.abs() > boundary as i32) {
        return Err(format!("E3 error: not all parameters are within the boundary of {} to {}", -(boundary as i32), boundary as i32));
    }

    Ok(numbers)
}

#[cfg(test)]
mod test {
    use std::{env, fs, collections::HashSet};

    use super::*;
    use crate::parser::build_d4_ddnnf_tree;

    #[test]
    fn handle_stream_msg_core_test() {
        let mut auto1: Ddnnf =
            build_d4_ddnnf_tree("tests/data/auto1_d4.nnf", 2513);
        let mut vp9: Ddnnf =
            build_d4_ddnnf_tree("tests/data/VP9_d4.nnf", 42);

        let binding = auto1.handle_stream_msg("core");
        let res = binding.split(" ").collect::<Vec<&str>>();
        assert_eq!(
            243, res.len()
        );
        assert!(
            res.contains(&"20") && res.contains(&"-168")
        );
        assert_eq!(
            String::from("20;-58"),
            auto1.handle_stream_msg("core v 20 -20 58 -58")
        );
        assert_eq!(
            String::from(""),
            auto1.handle_stream_msg("core v 1 2 3 -1 -2 -3")
        );
        assert_eq!(
            String::from("67;-58"),
            auto1.handle_stream_msg("core a 20 v 1 67 -58")
        );
        assert_eq!(
            String::from("4;5;6"), // count p 1 2 3 == 0
            auto1.handle_stream_msg("core a 1 2 3 v 4 5 6")
        );

        assert_eq!(
            String::from("1 2 6 10 15 19 25 31 40"),
            vp9.handle_stream_msg("core assumptions 1")
        );
        assert!( // count p 1 2 3 == 0 => all features are core under that assumption
            auto1.handle_stream_msg("core a 1 2 3").split(" ").count() == (auto1.number_of_variables*2) as usize
        );

        assert!(
            auto1.handle_stream_msg("core").split(" ").count() == auto1.core.len() + auto1.dead.len()
        );
        assert!(
            vp9.handle_stream_msg("core").split(" ").count() == vp9.core.len() + vp9.dead.len()
        );
    }

    #[test]
    fn handle_stream_msg_count_test() {
        let mut auto1: Ddnnf =
            build_d4_ddnnf_tree("tests/data/auto1_d4.nnf", 2513);

        assert_eq!(
            String::from(
            vec![
                "1161956426034856869593248790737503394254270990971132154082514918252601863499017129746491423758041981416261653822705296328530201469664767205987091228498329600000000000000000000000",
                "44558490301175088812121229002380743156731839067219819764321860535061593824456157036646879771006312148798957047211355633063364007335639360623647085885718285895752858908864905172260153747031300505600000000000000000000000",
                "387318808678285623197749596912501131418090330323710718027504972750867287833005709915497141252680660472087217940901765442843400489888255735329030409499443200000000000000000000000"
            ].join(";")),
            auto1.handle_stream_msg("count v 1 -2 3")
        );
        assert_eq!(
            String::from("0;0"),
            auto1.handle_stream_msg("count assumptions -1469 -1114 939 1551 variables 1 1529")
        );
        
        assert_eq!(
            auto1.rc().to_string(),
            auto1.handle_stream_msg("count")
        );
        assert_eq!(
            auto1.handle_stream_msg("count v 123 -1111"),
            vec![auto1.handle_stream_msg("count a 123"), auto1.handle_stream_msg("count a -1111")].join(";")
        );
    }

    #[test]
    fn handle_stream_msg_sat_test() {
        let mut auto1: Ddnnf =
            build_d4_ddnnf_tree("tests/data/auto1_d4.nnf", 2513);

        assert_eq!(
            String::from("true;true;false"),
            auto1.handle_stream_msg("sat v 1 -2 58")
        );
        assert_eq!(
            String::from("false;false"),
            auto1.handle_stream_msg("sat a -1469 -1114 939 1551 v 1 1529")
        );
        assert_eq!(
            (auto1.rc() > 0).to_string(),
            auto1.handle_stream_msg("sat")
        );
        assert_eq!(
            auto1.handle_stream_msg("sat v 1 58"),
            vec![
                auto1.handle_stream_msg("sat a 1"),
                auto1.handle_stream_msg("sat a 58")
            ].join(";")
        );
    }

    #[test]
    fn handle_stream_msg_enum_test() {
        let mut _auto1: Ddnnf =
            build_d4_ddnnf_tree("tests/data/auto1_d4.nnf", 2513);
        let mut vp9: Ddnnf =
            build_d4_ddnnf_tree("tests/data/VP9_d4.nnf", 42);

        let binding = vp9.handle_stream_msg("enum a 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39");
        let res: Vec<&str> = binding.split(";").collect();

        assert!(
            res.contains(&"1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39 40 -41 42")
            && res.contains(&"1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39 40 41 -42")
        );

        let binding = vp9.handle_stream_msg("enum a 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 27");
        let res: Vec<&str> = binding.split(";").collect();
        assert_eq!(80, res.len());

        let mut res_set = HashSet::new();
        for config_str in res {
            let config: Vec<i32> = config_str.split(" ").map(|f| f.parse::<i32>().unwrap()).collect();
            assert_eq!(vp9.number_of_variables as usize, config.len(), "the config is partial");
            assert!(vp9.is_sat_for(&config), "the config is not satisfiable");
            res_set.insert(config);
        }

        let binding = vp9.handle_stream_msg("enum");
        let res: Vec<&str> = binding.split(";").collect();
        assert_eq!(216000, res.len());

        let mut res_set = HashSet::new();
        for config_str in res {
            let config: Vec<i32> = config_str.split(" ").map(|f| f.parse::<i32>().unwrap()).collect();
            assert_eq!(vp9.number_of_variables as usize, config.len(), "the config is partial");
            res_set.insert(config);
        }
        assert_eq!(216000, res_set.len(), "at least one config occurs twice or more often");
    }

    #[test]
    fn handle_stream_msg_random_test() {
        let mut auto1: Ddnnf =
            build_d4_ddnnf_tree("tests/data/auto1_d4.nnf", 2513);
        let mut vp9: Ddnnf =
            build_d4_ddnnf_tree("tests/data/VP9_d4.nnf", 42);

        assert_eq!(
            String::from("E3 error: invalid digit found in string"),
            vp9.handle_stream_msg("random seed banana")
        );
        assert_eq!(
            String::from("E3 error: invalid digit found in string"),
            vp9.handle_stream_msg("random limit eight")
        );
        
        assert_eq!(
            String::from("E5 error: with the assumptions, the ddnnf is not satisfiable. Hence, there exist no valid sample configurations"),
            vp9.handle_stream_msg("random a 1 -1")
        );

        let mut binding = vp9.handle_stream_msg("random a 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39 seed 42");
        println!("{}", binding);
        let mut res = binding.split(" ").map(|v| v.parse::<i32>().unwrap()).collect::<Vec<i32>>();
        assert_eq!(1, vp9.execute_query(&res));
        assert_eq!(vp9.number_of_variables as usize, res.len());
        
        binding = vp9.handle_stream_msg("random");
        res = binding.split(" ").map(|v| v.parse::<i32>().unwrap()).collect::<Vec<i32>>();
        assert_eq!(1, vp9.execute_query(&res));
        assert_eq!(vp9.number_of_variables as usize, res.len());

        binding = auto1.handle_stream_msg("random assumptions 1 3 -4 270 122 -2000 limit 135");
        let results = binding.split(";")
            .map(|v| v.split(" ").map(|v_inner| v_inner.parse::<i32>().unwrap()).collect::<Vec<i32>>())
            .collect::<Vec<Vec<i32>>>();
        for result in results.iter() {
            assert_eq!(auto1.number_of_variables as usize, result.len());

            // contains the assumptions
            for elem in vec![1,3,-4,270,122,-2000].iter() {
                assert!(result.contains(elem));
            }
            for elem in vec![-1,-3,4,-270,-122,2000].iter() {
                assert!(!result.contains(elem));
            }

            assert!(auto1.execute_query(result) == 1);
        }
        assert_eq!(135, results.len());
    }

    #[test]
    fn handle_stream_msg_save_test() {
        let mut vp9: Ddnnf =
            build_d4_ddnnf_tree("tests/data/VP9_d4.nnf", 42);
        let binding = env::current_dir().unwrap();
        let working_dir = binding.to_str().unwrap();

        assert_eq!(
            format!("E4 error: the option \"{}\" is not valid in this context", &working_dir),
            vp9.handle_stream_msg(format!("save {}", &working_dir).as_str())
        );
        assert_eq!(
            String::from("E4 error: param \"path\" was used, but no value supplied"),
            vp9.handle_stream_msg("save path")
        );
        assert_eq!(
            String::from("E4 error: param \"p\" was used, but no value supplied"),
            vp9.handle_stream_msg("save p")
        );


        assert_eq!(
            String::from("E6 error: no file path was supplied"),
            vp9.handle_stream_msg("save")
        );
        assert_eq!(
            String::from("E6 error: No such file or directory (os error 2) while trying to write ddnnf to /home/ferris/Documents/crazy_project/out.nnf"),
            vp9.handle_stream_msg("save path /home/ferris/Documents/crazy_project/out.nnf")
        );
        assert_eq!(
            String::from("E6 error: file path is not absolute, but has to be"),
            vp9.handle_stream_msg("save p ./")
        );

        assert_eq!(
            String::from(""),
            vp9.handle_stream_msg(format!("save path {}/tests/data/out.nnf", &working_dir).as_str())
        );
        let _res = fs::remove_file(format!("{}/tests/data/out.nnf", &working_dir));
    }

    #[test]
    fn handle_stream_msg_other_test() {
        let mut auto1: Ddnnf =
            build_d4_ddnnf_tree("tests/data/auto1_d4.nnf", 2513);

        assert_eq!(
            String::from("exit"),
            auto1.handle_stream_msg("exit s 4 l 10")
        );
    }

    #[test]
    fn handle_stream_msg_error_test() {
        let mut auto1: Ddnnf =
            build_d4_ddnnf_tree("tests/data/auto1_d4.nnf", 2513);
        assert_eq!(
            String::from("E4 error: got an empty msg"),
            auto1.handle_stream_msg("")
        );
        assert_eq!(
            String::from("E4 error: the option \"5\" is not valid in this context"),
            auto1.handle_stream_msg("random a 1 2 s 13 5")
        );

        assert_eq!(
            String::from("E4 error: option used but there was no value supplied"),
            auto1.handle_stream_msg("random a")
        );
        assert_eq!(
            String::from("E4 error: option used but there was no value supplied"),
            auto1.handle_stream_msg("count a 1 2 3 v")
        );

        assert_eq!(
            String::from("E1 error: not yet supported"),
            auto1.handle_stream_msg("t-wise_sampling a 1 v 2")
        );
        assert_eq!(
            String::from("E2 error: the operation \"revive_dinosaurs\" is not supported"),
            auto1.handle_stream_msg("revive_dinosaurs a 1 v 2")
        );
        assert_eq!(
            String::from("E4 error: the option \"params\" is not valid in this context"),
            auto1.handle_stream_msg("count assumptions 1 v 2 params 3")
        );
        assert_eq!(
            String::from("E4 error: the option \"god_mode\" is not valid in this context"),
            auto1.handle_stream_msg("count assumptions 1 v 2 god_mode 3")
        );
        assert_eq!(
            String::from("E4 error: the option \"BDDs\" is not valid in this context"),
            auto1.handle_stream_msg("count a 1 2 BDDs 3")
        );  
    }

    #[test]
    fn test_get_numbers() {
        assert_eq!(
            Ok(vec![1, -2, 3]),
            get_numbers(vec!["1", "-2", "3"].as_ref(), 4)
        );
        assert_eq!(
            Ok(vec![1, -2, 3]),
            get_numbers(vec!["1", "-2", "3", "v", "4"].as_ref(), 5)
        );

        assert_eq!(Ok(vec![]), get_numbers(vec!["a", "1", "-2", "3"].as_ref(), 10));
        assert_eq!(
            Ok(vec![]),
            get_numbers(vec!["another_param", "1", "-2", "3"].as_ref(), 10)
        );

        assert_eq!(
            Err(String::from("E3 error: invalid digit found in string")),
            get_numbers(vec!["1", "-2", "3", " ", "4"].as_ref(), 10)
        );
        assert_eq!(
            Err(String::from("E3 error: invalid digit found in string")),
            get_numbers(vec!["1", "-2", "3.0", "v", "4"].as_ref(), 10)
        );
        assert_eq!(
            Err(String::from("E3 error: invalid digit found in string")),
            get_numbers(vec!["1", "-2", "--3", " ", "4"].as_ref(), 10)
        );
        assert_eq!(
            Err(String::from("E4 error: option used but there was no value supplied")),
            get_numbers(vec![].as_ref(), 10)
        );

        assert_eq!(
            Err(String::from("E3 error: not all parameters are within the boundary of -10 to 10")),
            get_numbers(vec!["1", "-2", "-300", "4"].as_ref(), 10)
        );
        assert_eq!(Ok(vec![]), get_numbers(vec!["a", "1", "-2", "30"].as_ref(), 10));
    }
}
