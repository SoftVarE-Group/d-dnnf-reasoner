mod query;

pub use query::Query;

use ddnnife::ddnnf::extended_ddnnf::ExtendedDdnnf;
use ddnnife::util::{format_vec, format_vec_vec};
use ddnnife::Ddnnf;
use ddnnife_cnf::Cnf;
use log::trace;
use std::fs::File;
use std::io::{stdin, Error, Result, Write};

static DEFAULT_LIMIT: Limit = 1;
static DEFAULT_SEED: u64 = 42;
static ERROR_UNSAT: &str = "With the assumptions, the d-DNNF is not satisfiable. Hence, there exist no valid sample configurations.";

type Literal = i32;
type Variable = u32;
type Limit = usize;
type Fitness = f64;

/// Starts the stream mode for the given d-DNNF.
///
/// Continuously waits for queries on stdin and evaluates them.
/// Runs until `exit` is entered.
pub fn stream(ddnnf: &mut Ddnnf) -> Result<()> {
    loop {
        // Wait for the next input line.
        let input = stdin().lines().next().unwrap()?;

        if input == "exit" {
            break;
        }

        // Parse the query and restart in case it is invalid.
        let query = match Query::parse(&input) {
            Err(error) => {
                eprintln!("{error}");
                continue;
            }
            Ok(query) => query,
        };

        trace!("{query:?}");

        // Perform the given operation and print its result.
        match handle_query(query, ddnnf) {
            Ok(result) => println!("{result}"),
            Err(error) => eprintln!("{error}"),
        }
    }

    Ok(())
}

/// Handles a single query.
///
/// Returns the result of the given query
pub fn handle_query(query: Query, ddnnf: &mut Ddnnf) -> Result<String> {
    match query {
        Query::Count {
            mut assumptions,
            variables,
        } => Ok(run_operation(
            |ddnnf, assumptions, _| Some(Ddnnf::execute_query(ddnnf, assumptions)),
            ddnnf,
            &mut assumptions,
            &variables,
        )),
        Query::Core {
            mut assumptions,
            variables,
        } => Ok(run_operation(
            |ddnnf, assumptions, variables| {
                if variables {
                    let could_be_core = assumptions.pop().unwrap();
                    let without_core = Ddnnf::execute_query(ddnnf, assumptions);
                    assumptions.push(could_be_core);
                    let with_core = Ddnnf::execute_query(ddnnf, assumptions);

                    if with_core == without_core {
                        return Some(could_be_core.to_string());
                    }

                    return None;
                }

                let mut result = ddnnf.core_dead_with_assumptions(assumptions);
                result.sort();
                Some(format_vec(result.iter()))
            },
            ddnnf,
            &mut assumptions,
            &variables,
        )),
        Query::Sat {
            mut assumptions,
            variables,
        } => Ok(run_operation(
            |ddnnf, assumptions, _| Some(Ddnnf::sat(ddnnf, assumptions)),
            ddnnf,
            &mut assumptions,
            &variables,
        )),
        Query::Enumerate {
            limit,
            mut assumptions,
        } => match ddnnf.enumerate(&mut assumptions, limit) {
            Some(samples) => Ok(format_vec_vec(samples.iter())),
            None => Err(Error::other(ERROR_UNSAT)),
        },
        Query::Random {
            limit,
            seed,
            assumptions,
        } => match ddnnf.uniform_random_sampling(&assumptions, limit, seed) {
            Some(samples) => Ok(format_vec_vec(samples.iter())),
            None => Err(Error::other(ERROR_UNSAT)),
        },
        Query::TWise { limit, fitness } => Ok(if fitness.is_empty() {
            ddnnf.sample_t_wise(limit)
        } else {
            let ext_ddnnf = ExtendedDdnnf::with_fitness_values(ddnnf.clone(), fitness);
            ext_ddnnf.sample_t_wise(limit)
        }
        .to_string()),
        Query::Atomic {
            candidates,
            assumptions,
            cross,
        } => Ok(format_vec_vec(
            ddnnf
                .get_atomic_sets(candidates, &assumptions, cross)
                .iter(),
        )),
        Query::SaveDdnnf { path } => {
            let mut file = File::create(&path)?;
            file.write_all(ddnnf.to_string().as_bytes())?;
            Ok(format!("d-DNNF successfully saved to {path:?}"))
        }
        Query::SaveCnf { path } => {
            let mut file = File::create(&path)?;
            file.write_all(Cnf::from(&*ddnnf).to_string().as_bytes())?;
            Ok(format!("CNF successfully saved to {path:?}"))
        }
    }
}

/// Runs the given operation under the given assumptions for each variable.
///
/// Runs once in case no variables are given.
/// When multiple variables are given, the respective results are joined by `;`.
///
/// # Note
///
/// A variable can be deselected, we therefore use literals instead of variables.
fn run_operation<T: ToString>(
    operation: fn(&mut Ddnnf, &mut Vec<Literal>, bool) -> Option<T>,
    ddnnf: &mut Ddnnf,
    assumptions: &mut Vec<Literal>,
    variables: &[Literal],
) -> String {
    // If no variables are given, simply perform the operation.
    if variables.is_empty() {
        if let Some(result) = operation(ddnnf, assumptions, false) {
            return result.to_string();
        }
    }

    // In case variables are given, for each variable, we extend the assumptions and run the
    // operation. The results are collected and returned as a single string.
    variables
        .iter()
        .map(|&variable| {
            // Extend the assumptions by the current variable.
            assumptions.push(variable);

            // Perform the operation.
            let result = operation(ddnnf, assumptions, true);

            // Remove the variable from the assumptions.
            assumptions.pop();

            result.map(|result| result.to_string()).unwrap_or_default()
        })
        .collect::<Vec<String>>()
        .join(";")
}

#[cfg(test)]
mod test {
    use super::{handle_query, Query};
    use ddnnife::parser::build_ddnnf;
    use ddnnife::util::format_vec;
    use ddnnife::Ddnnf;
    use num::{BigInt, One};
    use std::collections::HashSet;
    use std::io::{Error, Result};
    use std::path::Path;

    fn handle_string_query(ddnnf: &mut Ddnnf, query: &str) -> Result<String> {
        let query = Query::parse(query);

        if let Err(error) = query {
            return Err(Error::other(error.to_string()));
        }

        handle_query(query.unwrap(), ddnnf)
    }

    #[test]
    fn handle_query_core() {
        let mut auto1: Ddnnf = build_ddnnf(Path::new("tests/data/auto1_d4.nnf"), Some(2513));
        let mut vp9: Ddnnf = build_ddnnf(Path::new("tests/data/VP9_d4.nnf"), Some(42));

        let binding = handle_string_query(&mut auto1, "core").unwrap();
        let res = binding.split(' ').collect::<Vec<&str>>();

        assert_eq!(279, res.len());
        assert!(res.contains(&"20") && res.contains(&"-168"));

        assert_eq!(
            String::from("20;;;-58"),
            handle_string_query(&mut auto1, "core v 20 -20 58 -58").unwrap()
        );

        assert_eq!(
            String::from(";;;;;"),
            handle_string_query(&mut auto1, "core v -3..3").unwrap()
        );

        assert_eq!(
            String::from(";;67;;;-58"),
            handle_string_query(&mut auto1, "core a 20 v 1 -1 67 -67 58 -58").unwrap()
        );

        assert_eq!(
            String::from("4;5;6"), // count p 1 2 3 == 0
            handle_string_query(&mut auto1, "core a 1 2 3 v 4 5 6").unwrap()
        );

        assert_eq!(
            String::from("8640;8640;2880;2880;2880"),
            handle_string_query(&mut vp9, "count v 1..5 a 24..26").unwrap()
        );

        assert_eq!(
            String::from("1 2 6 10 15 19 25 31 40"),
            handle_string_query(&mut vp9, "core assumptions 1").unwrap()
        );

        assert_eq!(
            handle_string_query(&mut auto1, "core a 1 2 3")
                .unwrap()
                .split(' ')
                .count(),
            (auto1.number_of_variables * 2) as usize
        );

        assert_eq!(
            handle_string_query(&mut auto1, "core")
                .unwrap()
                .split(' ')
                .count(),
            auto1.core.len()
        );

        assert_eq!(
            handle_string_query(&mut vp9, "core")
                .unwrap()
                .split(' ')
                .count(),
            vp9.core.len()
        );
    }

    #[test]
    fn handle_query_count() {
        let mut auto1: Ddnnf = build_ddnnf(Path::new("tests/data/auto1_d4.nnf"), Some(2513));

        assert_eq!(
            ["1161956426034856869593248790737503394254270990971132154082514918252601863499017129746491423758041981416261653822705296328530201469664767205987091228498329600000000000000000000000",
                "44558490301175088812121229002380743156731839067219819764321860535061593824456157036646879771006312148798957047211355633063364007335639360623647085885718285895752858908864905172260153747031300505600000000000000000000000",
                "387318808678285623197749596912501131418090330323710718027504972750867287833005709915497141252680660472087217940901765442843400489888255735329030409499443200000000000000000000000"].join(";"),
            handle_string_query(&mut auto1, "count v 1 -2 3").unwrap()
        );

        assert_eq!(
            String::from("0;0"),
            handle_string_query(
                &mut auto1,
                "count assumptions -1469 -1114 939 1551 variables 1 1529"
            )
            .unwrap()
        );

        assert_eq!(
            auto1.rc().to_string(),
            handle_string_query(&mut auto1, "count").unwrap()
        );

        assert_eq!(
            handle_string_query(&mut auto1, "count v 123 -1111").unwrap(),
            [
                handle_string_query(&mut auto1, "count a 123").unwrap(),
                handle_string_query(&mut auto1, "count a -1111").unwrap()
            ]
            .join(";")
        );
    }

    #[test]
    fn handle_query_sat() {
        let mut auto1: Ddnnf = build_ddnnf(Path::new("tests/data/auto1_d4.nnf"), Some(2513));

        assert_eq!(
            String::from("true;true;false"),
            handle_string_query(&mut auto1, "sat v 1 -2 58").unwrap()
        );

        assert_eq!(
            String::from("false;false"),
            handle_string_query(&mut auto1, "sat a -1469 -1114 939 1551 v 1 1529").unwrap()
        );

        assert_eq!(
            (auto1.rc() > BigInt::ZERO).to_string(),
            handle_string_query(&mut auto1, "sat").unwrap()
        );

        assert_eq!(
            handle_string_query(&mut auto1, "sat v 1 58").unwrap(),
            [
                handle_string_query(&mut auto1, "sat a 1").unwrap(),
                handle_string_query(&mut auto1, "sat a 58").unwrap()
            ]
            .join(";")
        );
    }

    #[test]
    fn handle_query_enum() {
        let _auto1: Ddnnf = build_ddnnf(Path::new("tests/data/auto1_d4.nnf"), Some(2513));
        let mut vp9: Ddnnf = build_ddnnf(Path::new("tests/data/VP9_d4.nnf"), Some(42));

        let binding = handle_string_query(&mut vp9, "enum a 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39 l 10").unwrap();
        let res: Vec<&str> = binding.split(';').collect();

        assert!(
            res.contains(&"1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39 40 -41 42")
        );

        assert!(
            res.contains(&"1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39 40 41 -42")
        );

        assert_eq!(res.len(), 2, "there should be only 2 configs although we wanted 10, because there are only 2 individual and valid configs");

        let binding = handle_string_query(
            &mut vp9,
            "enum a 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 l 80",
        )
        .unwrap();

        let res: Vec<&str> = binding.split(';').collect();
        assert_eq!(80, res.len());

        let mut res_set = HashSet::new();
        for config_str in res {
            let config: Vec<i32> = config_str
                .split(' ')
                .map(|f| f.parse::<i32>().unwrap())
                .collect();

            assert_eq!(
                vp9.number_of_variables as usize,
                config.len(),
                "the config is partial"
            );

            assert!(vp9.sat(&config), "the config is not satisfiable");
            res_set.insert(config);
        }

        let binding = handle_string_query(&mut vp9, "enum a 1 l 216000").unwrap();
        let res: Vec<&str> = binding.split(';').collect();
        assert_eq!(216000, res.len());

        let mut res_set = HashSet::new();
        for config_str in res {
            let config: Vec<i32> = config_str
                .split(' ')
                .map(|f| f.parse::<i32>().unwrap())
                .collect();

            assert_eq!(
                vp9.number_of_variables as usize,
                config.len(),
                "the config is partial"
            );

            res_set.insert(config);
        }

        assert_eq!(
            216000,
            res_set.len(),
            "at least one config occurs twice or more often"
        );
    }

    #[test]
    fn handle_query_random() {
        let mut auto1: Ddnnf = build_ddnnf(Path::new("tests/data/auto1_d4.nnf"), Some(2513));
        let mut vp9: Ddnnf = build_ddnnf(Path::new("tests/data/VP9_d4.nnf"), Some(42));

        assert!(handle_string_query(&mut vp9, "random seed banana").is_err());
        assert!(handle_string_query(&mut vp9, "random limit eight").is_err());
        assert!(handle_string_query(&mut vp9, "random a 1 -1").is_err());

        let mut binding = handle_string_query(&mut vp9, "random a 1 2 3 -4 -5 6 7 -8 -9 10 11 -12 -13 -14 15 16 -17 -18 19 20 -21 -22 -23 -24 25 26 -27 -28 -29 -30 31 32 -33 -34 -35 -36 37 38 39 seed 42").unwrap();
        let mut res = binding
            .split(' ')
            .map(|v| v.parse::<i32>().unwrap())
            .collect::<Vec<i32>>();

        assert_eq!(BigInt::from(1), vp9.execute_query(&res));
        assert_eq!(vp9.number_of_variables as usize, res.len());

        binding = handle_string_query(&mut vp9, "random").unwrap();
        res = binding
            .split(' ')
            .map(|v| v.parse::<i32>().unwrap())
            .collect::<Vec<i32>>();

        assert_eq!(BigInt::from(1), vp9.execute_query(&res));
        assert_eq!(vp9.number_of_variables as usize, res.len());

        binding = handle_string_query(
            &mut auto1,
            "random assumptions 1 3 -4 270 122 -2000 limit 135",
        )
        .unwrap();

        let results = binding
            .split(';')
            .map(|v| {
                v.split(' ')
                    .map(|v_inner| v_inner.parse::<i32>().unwrap())
                    .collect::<Vec<i32>>()
            })
            .collect::<Vec<Vec<i32>>>();

        for result in results.iter() {
            assert_eq!(auto1.number_of_variables as usize, result.len());

            // contains the assumptions
            for elem in [1, 3, -4, 270, 122, -2000].iter() {
                assert!(result.contains(elem));
            }
            for elem in [-1, -3, 4, -270, -122, 2000].iter() {
                assert!(!result.contains(elem));
            }

            assert!(auto1.execute_query(result).is_one());
        }

        assert_eq!(135, results.len());
    }

    #[test]
    fn handle_query_atomic() {
        let mut vp9: Ddnnf = build_ddnnf(Path::new("tests/data/VP9_d4.nnf"), Some(42));

        assert!(handle_string_query(&mut vp9, "atomic sets").is_err());
        assert!(handle_string_query(&mut vp9, "atomic_sets").is_err());

        // negative assumptions are allowed
        assert_eq!(
            String::from("1 2 3 6 10 15 19 25 30 31 40;4 5 26 27 28 29"),
            handle_string_query(&mut vp9, "atomic a 1 2 6 -4 30 -5").unwrap()
        );

        // but negated variables are not allowed, because by definition atomic sets can't contain negated features
        assert!(handle_string_query(&mut vp9, "atomic v -1 a 1 2 6 -4 30 -5").is_err());

        assert_eq!(
            String::from("1 2 6 10 15 19 25 31 40"),
            handle_string_query(&mut vp9, "atomic").unwrap()
        );

        assert_eq!(
            String::from("1 2 6 10"),
            handle_string_query(&mut vp9, "atomic v 1..10").unwrap()
        );

        assert_eq!(
            String::from("15 19 25"),
            handle_string_query(&mut vp9, "atomic v 15..25 a 1 2 6 10 15").unwrap()
        );

        assert_eq!(
            String::from("1 2 3 6 10 15 19 25 31 40;4 5"),
            handle_string_query(&mut vp9, "atomic a 1 2 3").unwrap()
        );

        assert_eq!(
            // an unsat query results in an atomic set that contains one subset which contains all features
            format_vec(1..=42),
            handle_string_query(&mut vp9, "atomic a 4 5").unwrap()
        );
    }

    #[test]
    fn handle_query_error() {
        let mut auto1: Ddnnf = build_ddnnf(Path::new("tests/data/auto1_d4.nnf"), Some(2513));
        assert!(handle_string_query(&mut auto1, "").is_err());
        assert!(handle_string_query(&mut auto1, "random a 1 2 s 13 5").is_err());
        assert!(handle_string_query(&mut auto1, "count a 1 count v 2").is_err());
        assert!(handle_string_query(&mut auto1, "count a 1 2 3 a 1").is_err());
        assert!(handle_string_query(&mut auto1, "random a").is_err());
        assert!(handle_string_query(&mut auto1, "count a 1 2 3 v").is_err());
        assert!(handle_string_query(&mut auto1, "t-wise_sampling a 1 v 2").is_err());
        assert!(handle_string_query(&mut auto1, "revive_dinosaurs a 1 v 2").is_err());
        assert!(handle_string_query(&mut auto1, "count assumptions 1 v 2 params 3").is_err());
        assert!(handle_string_query(&mut auto1, "count a 1 2 BDDs 3").is_err());
    }
}
