//#![warn(clippy::all, clippy::pedantic)]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

extern crate clap;
use clap::{AppSettings, Arg, arg, Command, value_parser};

extern crate colour;
use colour::{green, yellow_ln};

use rustyline::error::ReadlineError;
use rustyline::Editor;

use std::io::{self, Write, BufRead};
use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::thread::{self};
use std::time::Instant;

use ddnnf_lib::ddnnf::Ddnnf;
use ddnnf_lib::parser::{self as dparser, persisting::write_ddnnf};

fn main() {
    let mut matches = Command::new("ddnnife")
    .global_settings(&[AppSettings::ColoredHelp])
    .author("Heiko Raab; heiko.raab@uni-ulm-de\nChico Sundermann; chico.sundermann@uni-ulm.de")
    .version("0.5.0")
    .setting(AppSettings::ArgRequiredElseHelp)
    .arg(Arg::with_name("file_path")
        .value_name("FILE PATH")
        .help("The path to the file in dimacs format. The d-dnnf has to be smooth to work properly!")
        .takes_value(true))
    .arg(arg!(-f --features "The numbers of the features that should be included or excluded (positive number to include, negative to exclude). Can be one or multiple. A feature f has to be ∈ ℤ and the only allowed seperator is a whitespace! This should be the last option, because of the ambiguity of a hyphen as signed int and another option or flag")
        .requires("file_path")
        .value_parser(value_parser!(i32))
        .value_name("FEATURE/S")
        .allow_hyphen_values(true)
        .takes_value(true)
        .multiple(true))
    .arg(arg!(--stream "The stream API")
        .requires("file_path")
        .takes_value(false))
    .arg(arg!(-i --interactive "The interactive mode allows the computation of multiple queries one after another by typing the query into the console. This mode is way slower then loading the queries with a file!\nThis mode also requires a file path to a file in dimacs format. There are the following options in interactive mode:\n[feature numbers with the same format as in -f]: computes the cardinality of partial configurations\nexit: closes the application (CTRL+C and CTRL+D also work)\nhelp: prints help information")
        .requires("file_path")
        .takes_value(false))
    .arg(arg!(-q --queries "Mulitple queries that follow the feature format and the qeries themself are seperatated by \"\\n\". The option takes a file with queries as first argument and an optional second argument for saving the results. Default output file is '{FILE_NAME}-queries.txt'.")
        .requires("file_path")
        .value_name("FILE QUERIES")
        .value_parser(value_parser!(String))
        .takes_value(true)
        .min_values(1)
        .max_values(2))
    .arg(arg!(-c --card_of_fs "Computes the cardinality of features for the feature model, i.e. the cardinality iff we select one feature for all features. Default output file is '{FILE_NAME}-features.csv'.")
        .requires("file_path")
        .value_name("CARDINALITY OF FEATURES")
        .value_parser(value_parser!(String))
        .takes_value(false)
        .min_values(0))
    .arg(arg!(-o --ommited_features "The number of omitted features. This is strictly necessary if the ddnnf has the d4 format respectivily does not contain a header.")
        .requires("file_path")
        .value_parser(value_parser!(u32))
        .value_name("OMMITED FEATURES")
        .takes_value(true))
    .arg(arg!(-s --save_ddnnf "Save the smooth ddnnf in the c2d format. Default output file is '{FILE_NAME}.nnf'. Alternatively, you can choose a name. The .nnf ending is added automatically")
        .requires("file_path")
        .value_parser(value_parser!(String))
        .min_values(0))
    .arg(arg!(-n --threads "Specify how many threads should be used. Default is 4. Possible values are between 1 and 32.")
        .requires("file_path")
        .value_parser(value_parser!(u16).range(1..33))
        .value_name("NUMBER OF THREADS")
        .takes_value(true))
    .arg(arg!(--heuristics "Provides information about the type of nodes, their connection and the different paths.")
        .requires("file_path")
        .takes_value(false))
    .get_matches();

    // create the ddnnf based of the input file that is required
    let time = Instant::now();
    let mut ddnnf: Ddnnf;

    if matches.contains_id("ommited_features") {
        let ommited_features: u32 = *matches.get_one("ommited_features").unwrap();
        ddnnf = dparser::build_d4_ddnnf_tree(
            matches.value_of("file_path").unwrap(),
            ommited_features,
        );

        if !matches.contains_id("stream") {
            let elapsed_time = time.elapsed().as_secs_f32();
            println!(
                "Ddnnf overall count: {:#?}\nElapsed time for parsing and overall count in seconds: {:.3}s.",
                ddnnf.rc(),
                elapsed_time
            );
        }
    } else {
        ddnnf = dparser::build_ddnnf_tree_with_extras(
            matches.value_of("file_path").unwrap(),
        );

        if !matches.contains_id("stream") {
            let elapsed_time = time.elapsed().as_secs_f32();
            println!(
                "Ddnnf overall count: {:#?}\nElapsed time for parsing and overall count in seconds: {:.3}s.",
                ddnnf.rc(),
                elapsed_time
            );
        }
    }

    // print the heuristics
    if matches.contains_id("heuristics") {
        ddnnf.print_all_heuristics();
    }

    // change the number of threads used for cardinality of features and partial configurations
    if matches.contains_id("threads") {
        ddnnf.max_worker = *matches.get_one("threads").unwrap();
    }

    // computes the cardinality for the partial configuration that can be mentioned with parameters
    if matches.contains_id("features") {
        let features: Vec<i32> =
                matches.get_many("features")
                .expect("invalid format for features").copied().collect();
        ddnnf.execute_query_interactive(features);
    }

    // file path without last extension
    let file_path = String::from(Path::new(matches.value_of("file_path").unwrap()).with_extension("").file_name().unwrap().to_str().unwrap());

    // computes the cardinality of partial configurations and saves the results in a .txt file
    // the results do not have to be in the same order if the number of threads is greater than one
    if matches.contains_id("queries") {
        let file_path_in = matches.remove_one::<String>("queries").unwrap();
        let file_path_out = &format!(
            "{}-queries.txt",
            matches.remove_one::<String>("queries").get_or_insert(file_path.clone()).as_str()
        );

        let time = Instant::now();
        ddnnf
            .card_multi_queries(file_path_in.as_str(), file_path_out)
            .unwrap_or_default();
        let elapsed_time = time.elapsed().as_secs_f64();

        println!(
            "Computed values of all queries in {} and the results are saved in {}\n
            It took {} seconds. That is an average of {} seconds per query",
            file_path_in,
            file_path_out,
            elapsed_time,
            elapsed_time / dparser::parse_queries_file(file_path_in.as_str()).len() as f64
        );
    }

    // computes the cardinality of features and saves the results in a .csv file
    // the cardinalities are always sorted from lowest to highest (also for multiple threadss)
    if matches.contains_id("card_of_fs") {
        let file_path = &format!(
            "{}-features.csv",
            matches.get_one::<String>("card_of_fs").get_or_insert(&file_path).as_str()
        );

        let time = Instant::now();
        ddnnf
            .card_of_each_feature(file_path)
            .unwrap_or_default();
        let elapsed_time = time.elapsed().as_secs_f64();

        println!(
            "Computed the Cardinality of all features in {} and the results are saved in {}\n
                It took {} seconds. That is an average of {} seconds per feature",
            &matches.value_of("file_path").unwrap(),
            file_path,
            elapsed_time,
            elapsed_time / ddnnf.number_of_variables as f64
        );
    }

    // switch in the interactive mode
    if matches.contains_id("interactive") {
        let mut rl = Editor::<()>::new();
        if rl.load_history("history.txt").is_err() {
            println!("No previous history.");
        }
        yellow_ln!("\nThis is the d-DNNF repl. Type help for further information about the usage");
        loop {
            let readline = rl.readline(">> ");
            match readline {
                Ok(line) => {
                    rl.add_history_entry(line.as_str());
                    match line.as_str() {
                        "help" => {
                            yellow_ln!("Usage information:");
                            green!("\t[feature numbers with the same format as in -f]: ");
                            println!("computes the cardinality of partial configurations");
                            green!("\texit: ");
                            println!("closes the application (CTRL+C and CTRL+D also work)");
                            green!("\thelp: ");
                            println!("prints this message");
                        }
                        "exit" => break,
                        other => {
                            let features: Option<Vec<i32>> = other.split_whitespace().map(|elem|
                            match elem.to_string().parse::<i32>() {
                                // check if input is within the valid range
                                Ok(s) => if s.abs() > ddnnf.number_of_variables as i32 || s == 0 {
                                    println!("The feature number {} is out of the range of 1 to {}. Please try again.", s, ddnnf.number_of_variables);
                                    None
                                } else {
                                    Some(s)
                                },
                                Err(e) => {
                                    println!("{}. Please try again.", e);
                                    None
                                },
                            }).collect();
                            if let Some(f) = features {
                                ddnnf.execute_query_interactive(f)
                            }
                        }
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("CTRL-C: program closes...");
                    break;
                }
                Err(ReadlineError::Eof) => {
                    println!("CTRL-D: program closes...");
                    break;
                }
                Err(err) => {
                    println!("Error: {:?}", err);
                    break;
                }
            }
        }
        rl.save_history("history.txt").unwrap();
    }


    // switch in the stream mode
    if matches.contains_id("stream") {
        let stdin_channel = spawn_stdin_channel();
        
        let stdout = io::stdout();
        let mut handle_out = stdout.lock();

        loop {
            match stdin_channel.recv() {
                Ok(mut buffer) => {
                    buffer.pop();

                    let response = ddnnf.handle_stream_msg(&buffer);

                    if response.as_str() == "exit" { handle_out.write_all("ENDE \\ü/".as_bytes()).unwrap(); break; }
                    
                    handle_out.write_all(format!("{}\n", response).as_bytes()).unwrap();
                    handle_out.flush().unwrap();
                },
                Err(e) => {
                    handle_out.write_all(format!("error while receiving msg: {}\n", e).as_bytes()).unwrap();
                    handle_out.flush().unwrap();
                },
            }
        }
    }

    // writes the d-DNNF to file
    if matches.contains_id("save_ddnnf") {
        let path = &format!(
            "{}-saved.nnf",
            matches.get_one::<String>("save_ddnnf").get_or_insert(&file_path).as_str()
        );
        write_ddnnf(&mut ddnnf, path).unwrap();
        println!("The smooth d-DNNF was written into the c2d format in {}.", path);
    }
}

fn spawn_stdin_channel() -> Receiver<String> {
    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        let stdin = io::stdin();
        let mut handle_in = stdin.lock();
        loop {
            let mut buffer = String::new();
            handle_in.read_line(&mut buffer).unwrap();
            tx.send(buffer).unwrap();
        }
    });
    rx
}