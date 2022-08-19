//#![warn(clippy::all, clippy::pedantic)]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

extern crate clap;
use clap::{crate_authors, crate_version, App, AppSettings, Arg};

extern crate colour;
use colour::{green, yellow_ln};

use rustyline::error::ReadlineError;
use rustyline::Editor;

use std::process::{self};
use std::time::Instant;

use ddnnf_lib::data_structure::Ddnnf;
use ddnnf_lib::parser::{self as dparser};

fn main() {
    let matches = App::new("ddnnife")
    .global_settings(&[AppSettings::ColoredHelp])
    .author(crate_authors!("\n"))
    .version(crate_version!())
    .setting(AppSettings::ArgRequiredElseHelp)
    .arg(Arg::with_name("FILE PATH")
        .display_order(1)
        .index(1)
        .allow_hyphen_values(true)
        .help("The path to the file in dimacs format. The d-dnnf has to be smooth to work properly!"))
    .arg(Arg::with_name("FEATURE/S")
        .display_order(2)
        .requires("FILE PATH")
        .help("The numbers of the features that should be included or excluded (positive number to include, negative to exclude). Can be one or multiple. A feature f has to be ∈ ℤ and the only allowed seperator is a whitespace!")
        .short("f")
        .long("features")
        .allow_hyphen_values(true)
        .takes_value(true)
        .multiple(true))
    .arg(Arg::with_name("FILE QUERIES")
        .display_order(2)
        .requires("FILE PATH")
        .help("Mulitple queries that follow the feature format and the qeries themself are seperatated by \"\\n\". Give the path to the input file after the flag. The default file name for the output is the \"out.txt\".")
        .short("q")
        .long("queries")
        .allow_hyphen_values(true)
        .takes_value(true))
    .arg(Arg::with_name("INTERACTIVE")
        .display_order(3)
        .requires("FILE PATH")
        .help("The interactive mode allows the computation of multiple queries one after another by typing the query into the console. This mode is way slower then loading the queries with a file!\nThis mode also requires a file path to a file in dimacs format. There are the following options in interactive mode:\n[feature numbers with the same format as in -f]: computes the cardinality of partial configurations\nexit: closes the application (CTRL+C and CTRL+D also work)\nhelp: prints help information")
        .short("i")
        .long("interactive"))
    .arg(Arg::with_name("CARDINALITY OF FEATURES")
        .display_order(4)
        .requires("FILE PATH")
        .help("Computes the cardinality of features for the feature model, i.e. the cardinality iff we select one feature for all features. The default file name for the output is the \"out.csv\".")
        .short("c")
        .long("card_of_fs"))
    .arg(Arg::with_name("OMMITED FEATURES")
        .display_order(5)
        .requires("FILE PATH")
        .help("The number of omitted features. This is strictly necessary if the ddnnf has the d4 format respectivily does not contain a header.")
        .short("o")
        .long("ommited_features")
        .takes_value(true))
    .arg(Arg::with_name("NUMBER OF THREADS")
        .display_order(5)
        .requires("FILE PATH")
        .help("Specify how many threads should be used. Default is 4. Possible values are between 1 and 32.")
        .short("n")
        .long("number_threads")
        .takes_value(true))
    .arg(Arg::with_name("HEURISTICS")
        .display_order(6)
        .requires("FILE PATH")
        .help("Provides information about the type of nodes, their connection and the different paths.")
        .long("heuristics"))
    .arg(Arg::with_name("CUSTOM OUTPUT FILE NAME")
        .display_order(7)
        .requires("FILE PATH")
        .help("Allows a custom file name for output file for the cardinality of features and file queries. The appropiate file ending gets added automcaticly.")
        .short("s")
        .long("save_as")
        .takes_value(true))
    .get_matches();

    if matches.is_present("HEURISTICS") {
        let mut ddnnf: Ddnnf =
            dparser::build_ddnnf_tree(matches.value_of("FILE PATH").unwrap());
        ddnnf.print_all_heuristics();
        process::exit(0);
    }

    // create the ddnnf based of the input file that is required
    let time = Instant::now();
    let mut ddnnf: Ddnnf;

    if matches.is_present("OMMITED FEATURES") {
        let ommited_features =
            match matches
                .value_of("OMMITED FEATURES")
                .unwrap()
                .parse::<u32>()
            {
                Ok(x) => x,
                Err(e) => panic!(
                    "[Error]: {:?}\nInvalid input for the number of ommited features! Aborting...",
                    e
                ),
            };
        ddnnf = dparser::build_d4_ddnnf_tree(
            matches.value_of("FILE PATH").unwrap(),
            ommited_features,
        );

        let elapsed_time = time.elapsed().as_secs_f32();
        println!(
            "Ddnnf overall count: {:#?}\n
            Elapsed time for parsing and overall count in seconds: {:.3}s.",
            ddnnf.nodes[ddnnf.number_of_nodes - 1].count.clone(),
            elapsed_time
        );
    } else {
        ddnnf = dparser::build_ddnnf_tree_with_extras(
            matches.value_of("FILE PATH").unwrap(),
        );
        let elapsed_time = time.elapsed().as_secs_f32();
        println!(
            "Ddnnf overall count: {:#?}\n
            Elapsed time for parsing and overall count in seconds: {:.3}s.",
            ddnnf.nodes[ddnnf.number_of_nodes - 1].count.clone(),
            elapsed_time
        );
    }

    // change the number of threads used for cardinality of features and partial configurations
    if matches.is_present("NUMBER OF THREADS") {
        let threads: u16 = match matches
            .value_of("NUMBER OF THREADS")
            .unwrap()
            .parse::<u16>()
        {
            Ok(x) => {
                if x > 32 {
                    32
                } else {
                    x
                }
            }
            Err(e) => panic!(
                "[Error]: {:?}\nInvalid input for number of threads! Aborting...",
                e
            ),
        };
        ddnnf.max_worker = threads;
    }

    // computes the cardinality for the partial configuration that can be mentioned with parameters
    if matches.is_present("FEATURE/S") {
        let features: Vec<i32> =
            dparser::parse_features(matches.values_of_lossy("FEATURE/S"));
        ddnnf.execute_query(features);
    }

    // computes the cardinality of partial configurations and saves the results in a .txt file
    // the results do not have to be in the same order if the number of threads is greater than one
    if matches.is_present("FILE QUERIES") {
        let file_path_in = matches.value_of("FILE QUERIES").unwrap();
        let file_path_out = &format!(
            "{}{}",
            matches.value_of("CUSTOM OUTPUT FILE NAME").unwrap_or("out"),
            ".txt"
        );

        let time = Instant::now();
        ddnnf
            .card_multi_queries(file_path_in, file_path_out)
            .unwrap_or_default();
        let elapsed_time = time.elapsed().as_secs_f64();

        println!(
            "Computed values of all queries in {} and the results are saved in {}\n
            It took {} seconds. That is an average of {} seconds per query",
            file_path_in,
            file_path_out,
            elapsed_time,
            elapsed_time / dparser::parse_queries_file(file_path_in).len() as f64
        );
    }

    // switch in the interactive mode
    if matches.is_present("INTERACTIVE") {
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
                                Ok(s) => Some(s),
                                Err(e) => {
                                    println!("The followin parsing error occured: {}. Please try again.", e);
                                    None
                                },
                            }).collect();
                            if let Some(f) = features {
                                ddnnf.execute_query(f)
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

    // computes the cardinality of features and saves the results in a .csv file
    // the cardinalities are always sorted from lowest to highest (also for multiple threadss)
    if matches.is_present("CARDINALITY OF FEATURES") {
        let file_path = &format!(
            "{}{}",
            matches.value_of("CUSTOM OUTPUT FILE NAME").unwrap_or("out"),
            ".csv"
        );

        let time = Instant::now();
        ddnnf
            .card_of_each_feature_to_csv(file_path)
            .unwrap_or_default();
        let elapsed_time = time.elapsed().as_secs_f64();

        println!(
            "Computed the Cardinality of all features in {} and the results are saved in {}\n
                It took {} seconds. That is an average of {} seconds per feature",
            &matches.value_of("FILE PATH").unwrap(),
            file_path,
            elapsed_time,
            elapsed_time / ddnnf.number_of_variables as f64
        );
    }
}
