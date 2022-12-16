use std::cmp::max;
use std::iter::FromIterator;
use std::{path::Path, collections::HashSet};

use rand_pcg::{Pcg32, Lcg64Xsh32};
use rand::{Rng, SeedableRng};

use rug::Assign;

use crate::parser::write_ddnnf;

use super::data_structure::*;

impl Ddnnf {
    /// error codes:
    /// E1 Operation is not yet supported
    /// E2 Operation does not exist. Neither now nor in the future
    /// E3 Parse error
    /// E4 Syntax error
    /// E5 file or path error
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
                "p" => {
                    params = match get_numbers(&args[param_index..], self.number_of_variables) {
                        Ok(v) => v,
                        Err(e) => return e,
                    };
                    param_index += params.len();
                },
                "v" => {
                    values = match get_numbers(&args[param_index..], self.number_of_variables) {
                        Ok(v) => v,
                        Err(e) => return e,
                    };
                    param_index += values.len();
                },
                "seed" | "limit" | "path" => {
                    if param_index < args.len() {
                        match args[param_index-1] {
                            "seed" => { seed = match args[param_index].parse::<u64>() {
                                    Ok(x) => x,
                                    Err(e) => return format!("E3 error: {}", e),
                                };
                                param_index += 1;
                            },
                            "limit" => {
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
                |d, x, _| Some(Ddnnf::execute_query(d, x) > 0),
                self,
                &mut params,
                &values
            ),
            "enum" => self.enumerate(&mut params, u64::MAX, usize::MAX),
            "random" => self.enumerate(&mut params, seed, limit),
            "exit" => String::from("exit"),
            "save" => {
                if path.to_str().unwrap() == "" {
                    return String::from("E5 error: no file path was supplied");
                }
                if !path.is_absolute() {
                    return String::from("E5 error: file path is not absolute, but has to be");
                }
                match write_ddnnf(self, path.to_str().unwrap()) {
                    Ok(_) => String::from(""),
                    Err(e) => format!("E5 error: {} while trying to write ddnnf to {}", e, path.to_str().unwrap()),
                }
            },
            "atomic" | "uni_random" | "t-wise_sampling" => {
                String::from("E1 error: not yet supported")
            }
            other => format!("E2 error: the operation \"{}\" is not supported", other),
        }
    }

    fn enumerate(&mut self, assumptions: &mut Vec<i32>,
        seed: u64,
        limit: usize,
    ) -> String {
        let mut sol: HashSet<Vec<i32>> = HashSet::new();
        let assumptions_set: HashSet<i32> = HashSet::from_iter(assumptions.iter().cloned());
        let configs = self.execute_query(&assumptions);
        let mut rng = if seed != u64::MAX {
            Some(Pcg32::seed_from_u64(seed))
        } else {
            None
        };

        self.enumeration_pre_step(assumptions_set.clone(), &mut rng, (limit, max(usize::MAX, configs.to_usize_wrapping())), &mut sol);
        for n in self.nodes.iter_mut() {
            n.marker = false;
        }
        self.md.clear();
        
        format_vec_vec(sol.iter())
    }

    fn enumeration_pre_step(
        &mut self,
        mut assumptions: HashSet<i32>,
        rng: &mut Option<Lcg64Xsh32>,
        limits: (usize, usize),
        sol: &mut HashSet<Vec<i32>>
    ) {
        if sol.len() >= limits.0 || sol.len() >= limits.1 {
            return;
        }

        // reseting all previous assignments of literals
        for n in self.nodes.iter_mut() {
            n.marker = false;
        }
        self.md.clear();
        for (f, index) in self.literals.iter() {
            if assumptions.contains(&-f.to_owned()) {
                self.nodes[*index].temp.assign(0);
                self.nodes[*index].marker = true;
            } else if assumptions.contains(&f.to_owned()){
                self.nodes[*index].temp.assign(1);
                self.nodes[*index].marker = true;
            } else {
                self.nodes[*index].temp.assign(1);
            }
        }

        let mut save = assumptions.clone();
        self.enumeration_step(self.number_of_nodes-1, &mut assumptions);
        
        if self.nodes[self.number_of_nodes-1].temp > 0 {
            if assumptions.len() == self.number_of_variables as usize {
                let mut vec = Vec::from_iter(assumptions.iter().cloned());
                vec.sort_by(|a, b| a.abs().cmp(&b.abs()));
                sol.insert(vec);
                return;
            }

            let mut next = 1;
            while save.contains(&next) || save.contains(&-next) {
                match rng {
                    Some(x) => next = x.gen_range(1..=self.number_of_variables) as i32,
                    None => next += 1,
                }
            }

            if next <= self.number_of_variables as i32 {
                save.insert(next);
                self.enumeration_pre_step(save.clone(), rng, limits, sol); 

                save.remove(&next);
                save.insert(-next);
                self.enumeration_pre_step(save.clone(), rng, limits, sol);
            }
        }
    }

    fn enumeration_step(
        &mut self,
        index: usize,
        set_features: &mut HashSet<i32>
    ) {
        if self.nodes[index].marker {
            return;
        }
        self.nodes[index].marker = true;
        match self.nodes[index].ntype.clone() {
            NodeType::And { children } => {
                for c in &children {
                    self.enumeration_step(*c, set_features);
                    if self.nodes[*c].temp == 0 {
                        self.nodes[index].temp.assign(0);
                        return;
                    }
                }
                self.nodes[index].temp.assign(1);
            },
            NodeType::Or { children } => {
                for c in &children {
                    self.enumeration_step(*c, set_features);
                    if self.nodes[*c].temp > 0 {
                        self.nodes[index].temp.assign(1);
                        return;
                    }
                }
                self.nodes[index].temp.assign(0);
            },
            NodeType::True => self.nodes[index].temp.assign(1),
            _ => (),
        }
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
    use std::{env, fs};

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
            auto1.handle_stream_msg("core p 20 v 1 67 -58")
        );
        assert_eq!(
            String::from("4;5;6"), // count p 1 2 3 == 0
            auto1.handle_stream_msg("core p 1 2 3 v 4 5 6")
        );

        assert_eq!(
            String::from("1 2 6 10 15 19 25 31 40"),
            vp9.handle_stream_msg("core p 1")
        );
        assert!( // count p 1 2 3 == 0 => all features are core under that assumption
            auto1.handle_stream_msg("core p 1 2 3").split(" ").count() == (auto1.number_of_variables*2) as usize
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
            auto1.handle_stream_msg("count p -1469 -1114 939 1551 v 1 1529")
        );
        
        assert_eq!(
            auto1.rc().to_string(),
            auto1.handle_stream_msg("count")
        );
        assert_eq!(
            auto1.handle_stream_msg("count v 123 -1111"),
            vec![auto1.handle_stream_msg("count p 123"), auto1.handle_stream_msg("count p -1111")].join(";")
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
            auto1.handle_stream_msg("sat p -1469 -1114 939 1551 v 1 1529")
        );
        assert_eq!(
            (auto1.rc() > 0).to_string(),
            auto1.handle_stream_msg("sat")
        );
        assert_eq!(
            auto1.handle_stream_msg("sat v 1 58"),
            vec![
                auto1.handle_stream_msg("sat p 1"),
                auto1.handle_stream_msg("sat p 58")
            ].join(";")
        );
    }

    #[test]
    fn handle_stream_msg_enum_test() {
        let mut _auto1: Ddnnf =
            build_d4_ddnnf_tree("tests/data/auto1_d4.nnf", 2513);
        let mut vp9: Ddnnf =
            build_d4_ddnnf_tree("tests/data/VP9_d4.nnf", 42);

        let binding = vp9.handle_stream_msg("enum p 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39");
        let res: Vec<&str> = binding.split(";").collect();

        assert!(
            res.contains(&"1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39 40 -41 42")
            && res.contains(&"1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39 40 41 -42")
        );

        assert_eq!(
            80,
            vp9.handle_stream_msg("enum p 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 27").split(";").count()
        );
        assert_eq!(
            216000,
            vp9.handle_stream_msg("enum").split(";").count()
        );
    }

    #[test]
    fn handle_stream_msg_random_test() {
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

        let mut binding = vp9.handle_stream_msg("random p 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39 seed 69");
        let mut res = binding.split(" ").map(|v| v.parse::<i32>().unwrap()).collect::<Vec<i32>>();
        assert!(
            vp9.execute_query(&res) == 1
        );
        binding = vp9.handle_stream_msg("random");
        res = binding.split(" ").map(|v| v.parse::<i32>().unwrap()).collect::<Vec<i32>>();
        assert!(
            vp9.execute_query(&res) == 1
        );

        assert_eq!(
            35,
            vp9.handle_stream_msg("random p 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 27 limit 35 ").split(";").count()
        );
        // if the limit > the #remaining satisfiable configurations for given assumptions then there will by only #remaining satisfiable configurations
        assert_eq!(
            vp9.handle_stream_msg("count p 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 27").parse::<usize>().unwrap(),
            vp9.handle_stream_msg("random p 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 27 limit 100000000 seed 42").split(";").count()
        );
        let binding2 = vp9.handle_stream_msg("random p 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 27 limit 35 ");
        let inter_res = binding2.split(";").collect::<Vec<&str>>();
        for res in inter_res {
            assert!(
                vp9.execute_query(&res.split(" ").map(|v| v.parse::<i32>().unwrap()).collect::<Vec<i32>>()) == 1
            );
        }

        // make sure that counting still works and the marked nodes aren't bad
        assert_eq!(
            String::from(
            vec![
                "216000", "0", "72000"
            ].join(";")),
            vp9.handle_stream_msg("count v 1 -2 3")
        );
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
            String::from("E5 error: no file path was supplied"),
            vp9.handle_stream_msg("save")
        );
        assert_eq!(
            String::from("E5 error: No such file or directory (os error 2) while trying to write ddnnf to /home/ferris/Documents/crazy_project/out.nnf"),
            vp9.handle_stream_msg("save path /home/ferris/Documents/crazy_project/out.nnf")
        );
        assert_eq!(
            String::from("E5 error: file path is not absolute, but has to be"),
            vp9.handle_stream_msg("save path ./")
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
            auto1.handle_stream_msg("exit seed 4 limit 10")
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
            auto1.handle_stream_msg("random p 1 2 seed 13 5")
        );

        assert_eq!(
            String::from("E4 error: option used but there was no value supplied"),
            auto1.handle_stream_msg("random p")
        );
        assert_eq!(
            String::from("E4 error: option used but there was no value supplied"),
            auto1.handle_stream_msg("count p 1 2 3 v")
        );

        assert_eq!(
            String::from("E1 error: not yet supported"),
            auto1.handle_stream_msg("t-wise_sampling p 1 v 2")
        );
        assert_eq!(
            String::from("E2 error: the operation \"revive_dinosaurs\" is not supported"),
            auto1.handle_stream_msg("revive_dinosaurs p 1 v 2")
        );
        assert_eq!(
            String::from("E4 error: the option \"god_mode\" is not valid in this context"),
            auto1.handle_stream_msg("count p 1 v 2 god_mode 3")
        );
        assert_eq!(
            String::from("E4 error: the option \"BDDs\" is not valid in this context"),
            auto1.handle_stream_msg("count p 1 2 BDDs 3")
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
