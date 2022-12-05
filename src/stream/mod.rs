use crate::Ddnnf;

/// error codes:
/// E1 Operation is not yet supported
/// E2 Operation does not exist. Neither now nor in the future
/// E3 Parse error
/// E4 Syntax error

pub fn handle_stream_msg(msg: &str, ddnnf: &mut Ddnnf) -> String {
    let args: Vec<&str> = msg.split_whitespace().collect();
    if args.is_empty() {
        return String::from("E4 error: got an empty msg");
    }

    let mut param_index = 1;
    let mut previous_param_index = 0;
    let mut params = Vec::new();
    let mut values = Vec::new();

    // go through all possible extra values that can be provided til
    // either there are no more or we can't parse anymore
    while param_index < args.len() && param_index != previous_param_index {
        previous_param_index = param_index;

        if param_index < args.len() && args[param_index] == "p" {
            param_index += 1;
            params = match get_numbers(&args[param_index..]) {
                Ok(v) => v,
                Err(e) => return e,
            };
            param_index += params.len();
        }

        if param_index < args.len() && args[param_index] == "v" {
            param_index += 1;
            values = match get_numbers(&args[param_index..]) {
                Ok(v) => v,
                Err(e) => return e,
            };
            param_index += values.len();
        }
    }

    if param_index == previous_param_index {
        return String::from("E4 error: could not use all parameters provided");
    }

    println!("params: {:?}, values: {:?}, param_index: {:?}, previous_param_index: {:?}",
        params, values, param_index, previous_param_index);

    match args[0] {
        "core" => op_with_assumptions_and_vars(
            |d, x| {
                let could_be_core = x.pop().unwrap();
                let without_cf =
                    Ddnnf::card_of_partial_config_with_marker(d, x);
                x.push(could_be_core);
                let with_cf = Ddnnf::card_of_partial_config_with_marker(d, x);

                if with_cf == without_cf {
                    Some(could_be_core)
                } else {
                    None
                }
            },
            ddnnf,
            &mut params,
            &values,
            false
        ),
        "count" => op_with_assumptions_and_vars(
            |d, x| Some(Ddnnf::card_of_partial_config_with_marker(d, x)),
            ddnnf,
            &mut params,
            &values,
            true
        ),
        "sat" => op_with_assumptions_and_vars(
            |d, x| {
                Some(Ddnnf::card_of_partial_config_with_marker(d, x) > 0)
            },
            ddnnf,
            &mut params,
            &values,
            true
        ),
        "exit" => String::from("exit"),
        "atomic" | "random" | "uni_random" | "t-wise_sampling" | "enum" => {
            String::from("E1 error: not yet supported")
        }
        _ => String::from("E2 error: is not supported"),
    }
}

fn op_with_assumptions_and_vars<T: ToString>(
    operation: fn(&mut Ddnnf, &mut Vec<i32>) -> Option<T>,
    ddnnf: &mut Ddnnf,
    assumptions: &mut Vec<i32>,
    vars: &[i32],
    can_handle_empty_vars: bool
) -> String {
    let mut response = Vec::new();
    for var in vars {
        assumptions.push(*var);
        match operation(ddnnf, assumptions) {
            Some(v) => response.push(v.to_string()),
            None => (),
        }
        assumptions.pop();
    }

    if vars.is_empty() {
        if !assumptions.is_empty() || can_handle_empty_vars {
            match operation(ddnnf, assumptions) {
                Some(v) => response.push(v.to_string()),
                None => (),
            }
        } else {
            return String::from("E4 error: can't compute if features are core if no features are supplied");
        }
    }

    response.join(";")
}

fn get_numbers(params: &[&str]) -> Result<Vec<i32>, String> {
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
            "E4 error: option used but there was not value supplied",
        ));
    }

    Ok(numbers)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::parser::build_d4_ddnnf_tree;

    #[test]
    fn handle_stream_msg_test() {
        let mut auto1: Ddnnf =
            build_d4_ddnnf_tree("example_input/auto1_d4.nnf", 2513);

        // core are 20 and 2122. dead are 177 and 2370
        let test0: &str = "core p 1 2 3 2122 177 -1 -2 -3 -20 -177 -2370";
        println!("{}", handle_stream_msg(test0, &mut auto1));

        let test1: &str = "core v 5 6 7 p 1 -2 3";
        handle_stream_msg(test1, &mut auto1);

        let test2: &str = "core p 1 -2 3 v";
        handle_stream_msg(test2, &mut auto1);

        let test3: &str = "core p 1 -2 3";
        handle_stream_msg(test3, &mut auto1);
    }

    #[test]
    fn handle_stream_msg_core_test() {
        let mut auto1: Ddnnf =
            build_d4_ddnnf_tree("example_input/auto1_d4.nnf", 2513);

        assert_eq!(
            String::from("20;-58"),
            handle_stream_msg("core v 20 -20 58 -58", &mut auto1)
        );
        assert_eq!(
            String::from("67;-58"),
            handle_stream_msg("core p 20 v 1 67 -58", &mut auto1)
        );
        assert_eq!(
            String::from(""),
            handle_stream_msg("core p 1", &mut auto1)
        );
        assert_eq!(
            String::from("4;5;6"), // count p 1 2 3 == 0
            handle_stream_msg("core p 1 2 3 v 4 5 6", &mut auto1)
        );

        assert_eq!(
            String::from("E4 error: can't compute if features are core if no features are supplied"),
            handle_stream_msg("core", &mut auto1)
        );
    }

    #[test]
    fn handle_stream_msg_count_test() {
        let mut auto1: Ddnnf =
            build_d4_ddnnf_tree("example_input/auto1_d4.nnf", 2513);

        assert_eq!(
            String::from("1161956426034856869593248790737503394254270990971132154082514918252601863499017129746491423758041981416261653822705296328530201469664767205987091228498329600000000000000000000000;44558490301175088812121229002380743156731839067219819764321860535061593824456157036646879771006312148798957047211355633063364007335639360623647085885718285895752858908864905172260153747031300505600000000000000000000000;387318808678285623197749596912501131418090330323710718027504972750867287833005709915497141252680660472087217940901765442843400489888255735329030409499443200000000000000000000000"),
            handle_stream_msg("count v 1 -2 3", &mut auto1)
        );
        assert_eq!(
            String::from("0;0"),
            handle_stream_msg("count p -1469 -1114 939 1551 v 1 1529", &mut auto1)
        );
        
        assert_eq!(
            auto1.rc().to_string(),
            handle_stream_msg("count", &mut auto1)
        );
        assert_eq!(
            handle_stream_msg("count v 123 -1111", &mut auto1),
            vec![handle_stream_msg("count p 123", &mut auto1), handle_stream_msg("count p -1111", &mut auto1)].join(";")
        );
    }

    #[test]
    fn handle_stream_msg_sat_test() {
        let mut auto1: Ddnnf =
            build_d4_ddnnf_tree("example_input/auto1_d4.nnf", 2513);

        assert_eq!(
            String::from("true;true;false"),
            handle_stream_msg("sat v 1 -2 58", &mut auto1)
        );
        assert_eq!(
            String::from("false;false"),
            handle_stream_msg("sat p -1469 -1114 939 1551 v 1 1529", &mut auto1)
        );
        assert_eq!(
            (auto1.rc() > 0).to_string(),
            handle_stream_msg("sat", &mut auto1)
        );
        assert_eq!(
            handle_stream_msg("sat v 1 58", &mut auto1),
            vec![handle_stream_msg("sat p 1", &mut auto1), handle_stream_msg("sat p 58", &mut auto1)].join(";")
        );
    }

    #[test]
    fn handle_stream_msg_error_test() {
        let mut auto1: Ddnnf =
            build_d4_ddnnf_tree("example_input/auto1_d4.nnf", 2513);
        assert_eq!(
            String::from("E4 error: got an empty msg"),
            handle_stream_msg("", &mut auto1)
        );
        assert_eq!(
            String::from("E4 error: could not use all parameters provided"),
            handle_stream_msg("count p 1 v 2 another_param 3", &mut auto1)
        );

        assert_eq!(
            String::from("E1 error: not yet supported"),
            handle_stream_msg("t-wise_sampling p 1 v 2", &mut auto1)
        );
        assert_eq!(
            String::from("E2 error: is not supported"),
            handle_stream_msg("revive_dinosaurs p 1 v 2", &mut auto1)
        );
    }

    #[test]
    fn test_get_numbers() {
        assert_eq!(
            Ok(vec![1, -2, 3]),
            get_numbers(vec!["1", "-2", "3"].as_ref())
        );
        assert_eq!(
            Ok(vec![1, -2, 3]),
            get_numbers(vec!["1", "-2", "3", "v", "4"].as_ref())
        );

        assert_eq!(Ok(vec![]), get_numbers(vec!["a", "1", "-2", "3"].as_ref()));
        assert_eq!(
            Ok(vec![]),
            get_numbers(vec!["another_param", "1", "-2", "3"].as_ref())
        );

        assert_eq!(
            Err(String::from("E3 error: invalid digit found in string")),
            get_numbers(vec!["1", "-2", "3", " ", "4"].as_ref())
        );
        assert_eq!(
            Err(String::from("E3 error: invalid digit found in string")),
            get_numbers(vec!["1", "-2", "3.0", "v", "4"].as_ref())
        );
        assert_eq!(
            Err(String::from("E3 error: invalid digit found in string")),
            get_numbers(vec!["1", "-2", "--3", " ", "4"].as_ref())
        );
        assert_eq!(
            Err(String::from("E4 error: option used but there was not value supplied")),
            get_numbers(vec![].as_ref())
        );
    }
}
