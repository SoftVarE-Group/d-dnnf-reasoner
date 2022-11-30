use crate::Ddnnf;

/// error codes:
/// 1 Operation is not yet supported
/// 2 Operation does not exist. Neither now nor in the future
/// 3 Parse error
/// 4 Syntax error

pub fn handle_stream_msg(msg: &str, ddnnf: &mut Ddnnf) -> String {
    let args: Vec<&str> = msg.split_whitespace().collect();
    if args.len() == 0 { return String::from("error: got an empty msg"); }

    let mut param_index = 1; let mut previous_param_index = 0;
    let mut params = Vec::new(); let mut values = Vec::new();
    
    // go through all possible extra values that can be provided til
    // either there are no more or we can't parse anymore
    while param_index < args.len() && param_index != previous_param_index {
        previous_param_index = param_index;

        if param_index < args.len() && args[param_index] == "p" {
            param_index += 1;
            params = match get_numbers(&args[param_index..]) {
                Ok(v) => v,
                Err(e) => return e
            };
            param_index += params.len();
        }

        if param_index < args.len() && args[param_index] == "v" {
            param_index += 1;
            values = match get_numbers(&args[param_index..]) {
                Ok(v) => v,
                Err(e) => return e
            };
            param_index += values.len();
        }
    }

    if param_index == previous_param_index {
        return String::from("error: could not use all parameters provided");
    }

    println!("params: {:?}, values: {:?}, param_index: {:?}, previous_param_index: {:?}",
        params, values, param_index, previous_param_index);


    match args[0] {
        "core" => return params.iter()
            .filter(|&p| if p.is_positive() { ddnnf.core.contains(p) } else { ddnnf.dead.contains(&p.abs()) })
            .map(|v| v.to_string())
            .collect::<Vec<String>>().join(" "),
        "count" => return ddnnf.card_of_partial_config_with_marker(&params).to_string(),
        "sat" => (ddnnf.card_of_partial_config_with_marker(&params) != 0).to_string(),
        "exit" => return String::from("exit"),
        "atomic" | "random" | "uni random"|
        "t-wise sampling" | "enum" => return String::from("1 error: not yet supported"), 
        _ => return String::from("2 error: is not supported"),
    }
}

fn get_numbers(params: &[&str]) -> Result<Vec<i32>, String> {
    let mut numbers = Vec::new();
    for param in params.iter() {
        if param.chars().all(|c| c.is_alphabetic()) {
            return Ok(numbers);
        }
        match param.parse::<i32>() {
            Ok(num) => numbers.push(num),
            Err(e) => return Err(format!("3 error: {}", e.to_string())),
        }
    }
    if numbers.len() == 0 {
        return Err(String::from("4 error: option used but there was not value supplied"))
    }

    return Ok(numbers);
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::parser::build_d4_ddnnf_tree;

    #[test]
    fn handle_stream_msg_test() {
        let mut auto1: Ddnnf = build_d4_ddnnf_tree(
            "example_input/auto1_d4.nnf",
            2513,
        );

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
    fn test_get_numbers() {
        let test1 = vec!["1", "-2", "3"];
        let test2 = vec!["1", "-2", "3", "v", "4"];
        let test3 = vec!["1", "-2", "3", " ", "4"];
        
        assert_eq!(get_numbers(&test1), Ok(vec![1,-2,3]));
        assert_eq!(get_numbers(&test2), Ok(vec![1,-2,3]));
        assert_eq!(get_numbers(&test3), Err(String::from("error: invalid digit found in string")));

        let test4 = vec!["a", "1", "-2", "3"];
        let test5 = vec!["1", "-2", "3.0", "v", "4"];
        let test6 = vec!["1", "-2", "--3", " ", "4"];

        assert_eq!(get_numbers(&test4), Ok(vec![]));
        assert_eq!(get_numbers(&test5), Err(String::from("error: invalid digit found in string")));
        assert_eq!(get_numbers(&test6), Err(String::from("error: invalid digit found in string")));
    }
}