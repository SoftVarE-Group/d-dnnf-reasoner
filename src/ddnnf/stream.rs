use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::BufRead;
use std::iter::FromIterator;
use std::process::exit;
use std::sync::Arc;
use std::sync::atomic::{Ordering, AtomicBool};
use std::sync::mpsc::{TryRecvError, Receiver, self};
use std::{thread, io};
use std::path::Path;

use itertools::Itertools;
use nom::IResult;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{digit1, char};
use nom::combinator::{map_res, opt, recognize};
use nom::sequence::{tuple, pair};
use workctl::WorkQueue;

use crate::{Ddnnf, parser::util::*};
use crate::parser::persisting::write_ddnnf;

impl Ddnnf {
    /// Initiate the Stream mode. This enables a commincation channel between stdin and stdout.
    /// Queries from stdin will be computed using max_worker many threads und results will be written
    /// to stdout. 'exit' as Input and breaking the stdin pipe exits Stream mode.
    pub fn init_stream(&self) {
        let mut queue: WorkQueue<(u32, String)> = WorkQueue::new();
        let stop = Arc::new(AtomicBool::new(false)); 
        // Create a MPSC (Multiple Producer, Single Consumer) channel. Every worker
        // is a producer, the main thread is a consumer; the producers put their
        // work into the channel when it's done.
        let (results_tx, results_rx) = mpsc::channel();  
        let mut threads = Vec::new();

        for _ in 0..self.max_worker {
            // copy all the data each worker needs
            let mut t_queue = queue.clone();
            let t_stop = stop.clone();
            let t_results_tx = results_tx.clone();
            let mut ddnnf: Ddnnf = self.clone();

            // spawn a worker thread with its shared and exclusive data
            let handle = thread::spawn(move || {
                loop {
                    if t_stop.load(Ordering::SeqCst) { break; }
                    if let Some((id, buffer)) = t_queue.pull_work() {
                        let response = ddnnf.handle_stream_msg(&buffer);
                        match t_results_tx.send((id, response)) {
                            Ok(_) => (),
                            Err(err) => {
                                eprintln!("Error while worker thread tried to sent \
                                a result to the master: {err}");
                                break;
                            }
                        }       
                    }
                }
            });

            // Add the handle for the newly spawned thread to the list of handles
            threads.push(handle);
        }

        // Main thread reads all the tasks from stdin and puts them in the queue
        let stdin_channel = spawn_stdin_channel();
        let mut remaining_answers = 0;
        let mut id = 0_u32;
        let mut output_id = 0_u32;
        let mut results: BinaryHeap<Reverse<(u32, String)>> = BinaryHeap::new();

        // Check whether next result/s to print is/are already in the result heap.
        // If thats the case, we print it/them. Anotherwise, we do nothing.
        let mut print_result = |results: &mut BinaryHeap<Reverse<(u32, String)>>| {
            while let Some(Reverse((id, res))) = results.peek() {
                if *id == output_id {
                    println!("{res}");
                    results.pop();
                    output_id += 1;
                } else {
                    break;
                }
            }
        };

        // loop til there are no more tasks to do
        loop {
            print_result(&mut results);

            // Check if we got any result
            match results_rx.try_recv() {
                Ok(val) => {
                    results.push(Reverse(val));
                    remaining_answers -= 1;
                },
                Err(err) => {
                    if err == TryRecvError::Disconnected {
                        eprintln!("A worker thread disconnected \
                        while working on a stream task. Aborting...");
                        exit(1);
                    }
                }
            }

            // Check if there is anything we can distribute to the workers
            match stdin_channel.try_recv() {
                Ok(buffer) => {
                    if buffer.as_str() == "exit" { break; }
                    queue.push_work((id, buffer.clone()));
                    id += 1;
                    remaining_answers += 1;
                },
                Err(err) => {
                    if err == TryRecvError::Disconnected { break; }
                },
            }
        }

        // After all tasks are distributed, we wait for the remaining results and print them.
        while remaining_answers != 0 {
            match results_rx.recv() {
                Ok(val) => {
                    results.push(Reverse(val));
                    remaining_answers -= 1;
                },
                Err(err) => {
                    eprintln!("A worker thread send an error ({err}) \
                    while working on a stream task. Aborting...");
                    exit(1);
                }
            }
            print_result(&mut results);
        }

        // Stop busy worker loops
        stop.store(true, Ordering::SeqCst);

        // join threads
        for handle in threads {
            handle.join().unwrap();
        }
    }

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
        let mut limit = None;
        let mut path = Path::new("");

        // go through all possible extra values that can be provided til
        // either there are no more or we can't parse anymore
        while param_index < args.len() {
            param_index += 1;
            match args[param_index-1] {
                "a" | "assumptions" => {
                    params = match get_numbers(&args[param_index..], self.number_of_variables) {
                        Ok(v) => {
                            param_index += v.1;
                            v.0
                        },
                        Err(e) => return e,
                    };
                },
                "v" | "variables" => {
                    values = match get_numbers(&args[param_index..], self.number_of_variables) {
                        Ok(v) => {
                            param_index += v.1;
                            v.0
                        },
                        Err(e) => return e,
                    };
                },
                "seed" | "s" | "limit" | "l" | "path" | "p" => {
                    if param_index < args.len() {
                        match args[param_index-1] {
                            "seed" | "s" => { seed = match args[param_index].parse::<u64>() {
                                    Ok(x) => x,
                                    Err(e) => return format!("E3 error: {e}"),
                                };
                                param_index += 1;
                            },
                            "limit" | "l" => {
                                limit = match args[param_index].parse::<usize>() {
                                    Ok(x) => Some(x),
                                    Err(e) => return format!("E3 error: {e}"),
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
                other => return format!("E4 error: the option \"{other}\" is not valid in this context"),
            }
        }

        match args[0] {
            "core" => op_with_assumptions_and_vars(
                |d, assumptions, vars| {
                    if vars {
                        let could_be_core = assumptions.pop().unwrap();
                        let without_cf = Ddnnf::execute_query(d, assumptions);
                        assumptions.push(could_be_core);
                        let with_cf = Ddnnf::execute_query(d, assumptions);

                        if with_cf == without_cf {
                            Some(could_be_core.to_string())
                        } else {
                            None
                        }
                    } else if assumptions.is_empty() {
                        let mut core = Vec::from_iter(&d.core);
                        core.sort_by_key(|a| a.abs());
                        Some(format_vec(core.iter()))
                    } else {
                        let mut core = Vec::new();
                        let reference = Ddnnf::execute_query(d, assumptions);
                        for i in 1_i32..=d.number_of_variables as i32 {
                            assumptions.push(i);
                            let inter = Ddnnf::execute_query(d, assumptions);
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
                |d, x, _| Some(Ddnnf::sat(d, x)),
                self,
                &mut params,
                &values
            ),
            "enum" => {
                let limit_interpretation = match limit {
                    Some(limit) => limit,
                    None => {
                        if self.rc() > 1_000 {
                            1_000
                        } else {
                            self.rc().to_usize_wrapping()
                        }
                    },
                };
                let configs = self.enumerate(&mut params, limit_interpretation);
                match configs {
                    Some(s) => format_vec_vec(s.iter()),
                    None => String::from("E5 error: with the assumptions, the ddnnf is not satisfiable. Hence, there exist no valid sample configurations"),
                }
            },
            "random" => {
                let limit_interpretation = limit.unwrap_or(1);
                let samples = self.uniform_random_sampling(&params, limit_interpretation, seed);
                match samples {
                    Some(s) => format_vec_vec(s.iter()),
                    None => String::from("E5 error: with the assumptions, the ddnnf is not satisfiable. Hence, there exist no valid sample configurations"),
                }
            }
            "atomic" => {
                if values.iter().any(|&f| f.is_negative()) {
                    return String::from("E5 error: candidates must be positive");
                }
                let candidates = if !values.is_empty()  {
                    Some(values.iter().map(|&f| f as u32).collect_vec())
                } else {
                    None
                };
                format_vec_vec(self.get_atomic_sets(candidates, &params).iter())
            },
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
            "t-wise" => {
                let limit_interpretation = limit.unwrap_or(1);
                self.sample_t_wise(limit_interpretation).to_string()
            }
            other => format!("E2 error: the operation \"{other}\" is not supported"),
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
        if let Some(v) = operation(ddnnf, assumptions, false) { return v.to_string() }
    }

    let mut response = Vec::new();
    for var in vars {
        assumptions.push(*var);
        if let Some(v) = operation(ddnnf, assumptions, true) { response.push(v.to_string()) }
        assumptions.pop();
    }

    response.join(";")
}

// parses numbers and ranges of the form START..[STOP] into a vector of i32
fn get_numbers(params: &[&str], boundary: u32) -> Result<(Vec<i32>, usize), String> {
    let mut numbers = Vec::new();
    let mut parsed_str_count = 0;

    for &param in params.iter() {
        if param.chars().any(|c| c.is_alphabetic()) {
            return Ok((numbers, parsed_str_count));
        }

        fn signed_number(input: &str) -> IResult<&str, &str> {
            recognize(pair(opt(char('-')), digit1))(input)
        }

        // try parsing by starting with the most specific combination and goind
        // to the least specific one
        let range: IResult<&str, Vec<i32>> = 
            alt((
                // limited range
                map_res(
                    tuple((signed_number, tag(".."), signed_number)),
                    |s: (&str, &str, &str) | {
                        match (s.0.parse::<i32>(), s.2.parse::<i32>()) {
                            // start and stop are inclusive (removing the '=' would make stop exclusive)
                            (Ok(start), Ok(stop)) => Ok((start..=stop).collect()),
                            (Ok(_), Err(e)) |
                            (Err(e), _) => Err(e)
                        }
                    }
                ),
                // unlimited range
                map_res(
                    pair(signed_number, tag("..")),
                    |s: (&str, &str) | {
                        match s.0.parse::<i32>() {
                            Ok(start) => Ok((start..=boundary as i32).collect()),
                            Err(e) => Err(e)
                        }
                    }
                ),
                // single number
                map_res(
                    signed_number,
                    |s: &str| {
                        match s.parse::<i32>() {
                            Ok(v) => Ok(vec![v]),
                            Err(e) => Err(e),
                        }
                    }
                )
            ))(param);

        match range {
            // '0' isn't valid in this context and gets removed
            Ok(num) => num.1.into_iter().for_each(|f| if f != 0 { numbers.push(f) }),
            Err(e) => return Err(format!("E3 {e}")),
        }
        parsed_str_count += 1;
    }

    if numbers.is_empty() {
        return Err(String::from(
            "E4 error: option used but there was no value supplied",
        ));
    }

    if numbers.iter().any(|v| v.abs() > boundary as i32) {
        return Err(format!("E3 error: not all parameters are within the boundary of {} to {}", -(boundary as i32), boundary as i32));
    }
    Ok((numbers, parsed_str_count))
}

// spawns a new thread that listens on stdin and delivers its request to the stream message handling
fn spawn_stdin_channel() -> Receiver<String> {
    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        let stdin = io::stdin();
        let lines = stdin.lock().lines();
        for line in lines {
            tx.send(line.unwrap()).unwrap()
        }
    });
    rx
}

#[cfg(test)]
mod test {
    use std::{env, fs::{self}, collections::HashSet};

    use assert_cmd::Command;
    use itertools::Itertools;

    use super::*;
    use crate::parser::build_ddnnf;

    #[test]
    fn handle_stream_msg_core() {
        let mut auto1: Ddnnf =
            build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));
        let mut vp9: Ddnnf =
            build_ddnnf("tests/data/VP9_d4.nnf", Some(42));

        let binding = auto1.handle_stream_msg("core");
        let res = binding.split(" ").collect::<Vec<&str>>();
        assert_eq!(
            279, res.len()
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
            auto1.handle_stream_msg("core v -3..3")
        );
        assert_eq!(
            String::from("67;-58"),
            auto1.handle_stream_msg("core a 20 v 1 -1 67 -67 58 -58")
        );
        assert_eq!(
            String::from("4;5;6"), // count p 1 2 3 == 0
            auto1.handle_stream_msg("core a 1 2 3 v 4 5 6")
        );
        assert_eq!(
            String::from("8640;8640;2880;2880;2880"),
            vp9.handle_stream_msg("count v 1..5 a 24..26")
        );

        assert_eq!(
            String::from("1 2 6 10 15 19 25 31 40"),
            vp9.handle_stream_msg("core assumptions 1")
        );
        assert!( // count p 1 2 3 == 0 => all features are core under that assumption
            auto1.handle_stream_msg("core a 1 2 3").split(" ").count() == (auto1.number_of_variables*2) as usize
        );

        assert_eq!(
            auto1.handle_stream_msg("core").split(" ").count(), auto1.core.len()
        );
        assert_eq!(
            vp9.handle_stream_msg("core").split(" ").count(), vp9.core.len()
        );
    }

    #[test]
    fn handle_stream_msg_count() {
        let mut auto1: Ddnnf =
            build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));

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
    fn handle_stream_msg_sat() {
        let mut auto1: Ddnnf =
            build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));

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
    fn handle_stream_msg_enum() {
        let mut _auto1: Ddnnf =
            build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));
        let mut vp9: Ddnnf =
            build_ddnnf("tests/data/VP9_d4.nnf", Some(42));

        let binding = vp9.handle_stream_msg("enum a 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39 l 10");
        let res: Vec<&str> = binding.split(";").collect_vec();

        assert!(
            res.contains(&"1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39 40 -41 42")
        );
        assert!(
            res.contains(&"1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39 40 41 -42")
        );
        assert_eq!(res.len(), 2, "there should be only 2 configs although we wanted 10, because there are only 2 individual and valid configs");

        let binding = vp9.handle_stream_msg("enum a 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 l 80");
        let res: Vec<&str> = binding.split(";").collect();
        assert_eq!(80, res.len());

        let mut res_set = HashSet::new();
        for config_str in res {
            let config: Vec<i32> = config_str.split(" ").map(|f| f.parse::<i32>().unwrap()).collect();
            assert_eq!(vp9.number_of_variables as usize, config.len(), "the config is partial");
            assert!(vp9.sat(&config), "the config is not satisfiable");
            res_set.insert(config);
        }

        let binding = vp9.handle_stream_msg("enum a 1 l 216000");
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
    fn handle_stream_msg_random() {
        let mut auto1: Ddnnf =
            build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));
        let mut vp9: Ddnnf =
            build_ddnnf("tests/data/VP9_d4.nnf", Some(42));

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
    fn handle_stream_msg_atomic() {
        let mut vp9: Ddnnf =
            build_ddnnf("tests/data/VP9_d4.nnf", Some(42));

        assert_eq!(
            String::from("E4 error: the option \"sets\" is not valid in this context"),
            vp9.handle_stream_msg("atomic sets")
        );
        assert_eq!(
            String::from("E2 error: the operation \"atomic_sets\" is not supported"),
            vp9.handle_stream_msg("atomic_sets")
        );

        // negative assumptions are allowed
        assert_eq!(
            String::from("1 2 3 6 10 15 19 25 30 31 40;4 5 26 27 28 29"),
            vp9.handle_stream_msg("atomic a 1 2 6 -4 30 -5")
        );
        // but negated variables are not allowed, because by definition atomic sets can't contain negated features
        assert_eq!(
            String::from("E5 error: candidates must be positive"),
            vp9.handle_stream_msg("atomic v -1 a 1 2 6 -4 30 -5")
        );

        assert_eq!(
            String::from("1 2 6 10 15 19 25 31 40"),
            vp9.handle_stream_msg("atomic")
        );
        assert_eq!(
            String::from("1 2 6 10"),
            vp9.handle_stream_msg("atomic v 1..10")
        );
        assert_eq!(
            String::from("15 19 25"),
            vp9.handle_stream_msg("atomic v 15..25 a 1 2 6 10 15")
        );
        assert_eq!(
            String::from("1 2 3 6 10 15 19 25 31 40;4 5"),
            vp9.handle_stream_msg("atomic a 1 2 3")
        );
        assert_eq!( // an unsat query results in an atomic set that contains one subset which contains all features
            format_vec((1..=42).into_iter()),
            vp9.handle_stream_msg("atomic a 4 5")
        );
    }

    #[test]
    fn handle_stream_msg_save() {
        let mut vp9: Ddnnf =
            build_ddnnf("tests/data/VP9_d4.nnf", Some(42));
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
    fn handle_stream_msg_other() {
        let mut auto1: Ddnnf =
            build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));

        assert_eq!(
            String::from("exit"),
            auto1.handle_stream_msg("exit s 4 l 10")
        );
    }

    #[test]
    fn handle_stream_msg_error() {
        let mut auto1: Ddnnf =
            build_ddnnf("tests/data/auto1_d4.nnf", Some(2513));
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
            String::from("E2 error: the operation \"t-wise_sampling\" is not supported"),
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
            Ok((vec![1, -2, 3], 3)),
            get_numbers(vec!["1", "-2", "3"].as_ref(), 4)
        );
        assert_eq!(
            Ok((vec![1, -2, 3], 3)),
            get_numbers(vec!["1", "-2", "3", "v", "4"].as_ref(), 5)
        );

        assert_eq!(
            Ok((vec![], 0)),
            get_numbers(vec!["a", "1", "-2", "3"].as_ref(), 10)
        );
        assert_eq!(
            Ok((vec![], 0)),
            get_numbers(vec!["another_param", "1", "-2", "3"].as_ref(), 10)
        );

        assert_eq!(
            Err(String::from("E3 Parsing Error: Error { input: \" \", code: Digit }")),
            get_numbers(vec!["1", "-2", "3", " ", "4"].as_ref(), 10)
        );
        assert_eq!(
            Ok((vec![1, -2, 3], 4)),
            get_numbers(vec!["1", "-2", "0", "3.5", "v", "4"].as_ref(), 10)
        );
        assert_eq!(
            Err(String::from("E3 Parsing Error: Error { input: \"-3\", code: Digit }")),
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
        assert_eq!(
            Ok((vec![], 0)),
            get_numbers(vec!["a", "1", "-2", "30"].as_ref(), 10)
        );
    }

    #[test]
    fn test_number_ranges() {
        assert_eq!(
            Ok((vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)),
            get_numbers(vec!["1..10", "v", "4"].as_ref(), 20)
        );
        assert_eq!(
            Ok(((1..=20).collect_vec(), 1)),
            get_numbers(vec!["1..", "v", "4"].as_ref(), 20)
        );
        assert_eq!(
            Ok((vec![-1, 1, 2, 3, 4, 6, 8, 9, 10], 3)),
            get_numbers(vec!["-1..4", "6", "8..10", "v", "4"].as_ref(), 20)
        );
        assert_eq!(
            Ok((vec![5, 6, 7], 3)),
            get_numbers(vec!["5..2", "5..5", "6..7"].as_ref(), 20)
        );
        assert_eq!(
            Ok((vec![1, 2, 3, 4, 5], 1)),
            get_numbers(vec!["1..5", "a", "6", "7", "3", "4", "5"].as_ref(), 20)
        );
    }

    #[test]
    fn parallel_stream() {
        let mut ddnnf: Ddnnf = build_ddnnf("example_input/auto1_d4_2513.nnf", Some(2513));

        let mut card_of_features_input = String::new();
        let mut should_be = String::new();
        for i in 1..=2513 {
            card_of_features_input += format!("count a {i}\n").as_str(); // form the query for stream
            should_be += format!("{:?}\n", ddnnf.execute_query(&[i])).as_str(); // build the expected result
        }
        card_of_features_input += "exit\n";

        // execute all the 2513 queries with 4 threads
        let cmd_ddnnf = Command::cargo_bin("ddnnife")
            .unwrap()
            .args(["example_input/auto1_d4_2513.nnf", "-t", "2513", "stream", "-j", "4"])
            .write_stdin(card_of_features_input.clone())
            .unwrap();

        // do the same when using a CNF
        let cmd_cnf = Command::cargo_bin("ddnnife")
            .unwrap()
            .args(["example_input/auto1.cnf", "-t", "2513", "stream", "-j", "4"])
            .write_stdin(card_of_features_input.clone())
            .unwrap();

        assert_eq!(should_be, String::from_utf8(cmd_ddnnf.stdout).unwrap());
        assert_eq!(should_be, String::from_utf8(cmd_cnf.stdout).unwrap());
    }
}
