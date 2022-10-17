//#![warn(clippy::all, clippy::pedantic)]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

extern crate clap;
use clap::{AppSettings, Arg, arg, Command, value_parser};

extern crate colour;
use colour::{green, yellow_ln};

use rustyline::error::ReadlineError;
use rustyline::Editor;

use std::process::{self};
use std::time::Instant;

use ddnnf_lib::data_structure::Ddnnf;
use ddnnf_lib::parser::{self as dparser, write_ddnnf};

fn main() {
    let matches = Command::new("ddnnife")
    .global_settings(&[AppSettings::ColoredHelp])
    .author("Heiko Raab; heiko.raab@uni-ulm-de\nChico Sundermann; chico.sundermann@uni-ulm.de")
    .version("0.4.0")
    .setting(AppSettings::ArgRequiredElseHelp)
    .arg(Arg::with_name("file_path")
        .value_name("FILE PATH")
        .allow_hyphen_values(true)
        .help("The path to the file in dimacs format. The d-dnnf has to be smooth to work properly!")
        .takes_value(true))
    .arg(arg!(-f --features "The numbers of the features that should be included or excluded (positive number to include, negative to exclude). Can be one or multiple. A feature f has to be ∈ ℤ and the only allowed seperator is a whitespace!")
        .requires("file_path")
        .value_name("FEATURE/S")
        .allow_hyphen_values(true)
        .takes_value(true)
        .multiple(true))
    .arg(arg!(-q --queries "Mulitple queries that follow the feature format and the qeries themself are seperatated by \"\\n\". Give the path to the input file after the flag. The default file name for the output is the \"out.txt\".")
        .requires("file_path")
        .value_name("FILE QUERIES")
        .allow_hyphen_values(true)
        .takes_value(true))
    .arg(arg!(-i --interactive "The interactive mode allows the computation of multiple queries one after another by typing the query into the console. This mode is way slower then loading the queries with a file!\nThis mode also requires a file path to a file in dimacs format. There are the following options in interactive mode:\n[feature numbers with the same format as in -f]: computes the cardinality of partial configurations\nexit: closes the application (CTRL+C and CTRL+D also work)\nhelp: prints help information")
        .requires("file_path")
        .takes_value(false))
    .arg(arg!(-c --card_of_fs "Computes the cardinality of features for the feature model, i.e. the cardinality iff we select one feature for all features. The default file name for the output is the \"out.csv\".")
        .requires("file_path")
        .value_name("CARDINALITY OF FEATURES")
        .takes_value(false))
    .arg(arg!(-o --ommited_features "The number of omitted features. This is strictly necessary if the ddnnf has the d4 format respectivily does not contain a header.")
        .requires("file_path")
        .value_parser(value_parser!(u32))
        .value_name("OMMITED FEATURES")
        .takes_value(true))
    .arg(arg!(-d --ddnnf "Save the smooth ddnnf in the c2d format. Default output file is out.nnf. Alternatively, you can choose a name. The .nnf ending is added automatically")
        .requires("file_path")
        .takes_value(true)
        .default_value("out"))
    .arg(arg!(-n --threads "Specify how many threads should be used. Default is 4. Possible values are between 1 and 32.")
        .requires("file_path")
        .value_name("NUMBER OF THREADS")
        .takes_value(true))
    .arg(arg!(--heuristics "Provides information about the type of nodes, their connection and the different paths.")
        .requires("file_path")
        .takes_value(false))
    .arg(arg!(-s --save_as "Allows a custom file name for output file for the cardinality of features and file queries. The appropiate file ending gets added automcaticly.")
        .requires("file_path")
        .value_name("CUSTOM OUTPUT FILE NAME")
        .takes_value(true))
    .get_matches();

    if matches.contains_id("heuristics") {
        let mut ddnnf: Ddnnf =
            dparser::build_ddnnf_tree(matches.value_of("file_path").unwrap());
        ddnnf.print_all_heuristics();
        process::exit(0);
    }

    // create the ddnnf based of the input file that is required
    let time = Instant::now();
    let mut ddnnf: Ddnnf;

    if matches.contains_id("ommited_features") {
        let ommited_features: u32 = *matches.get_one("ommited_features")
            .expect("[Error]: Invalid input for the number of ommited features! Aborting...");
        ddnnf = dparser::build_d4_ddnnf_tree(
            matches.value_of("file_path").unwrap(),
            ommited_features,
        );

        let elapsed_time = time.elapsed().as_secs_f32();
        println!(
            "Ddnnf overall count: {:#?}\n
            Elapsed time for parsing and overall count in seconds: {:.3}s.",
            ddnnf.rc(),
            elapsed_time
        );
    } else {
        ddnnf = dparser::build_ddnnf_tree_with_extras(
            matches.value_of("file_path").unwrap(),
        );
        let elapsed_time = time.elapsed().as_secs_f32();
        println!(
            "Ddnnf overall count: {:#?}\n
            Elapsed time for parsing and overall count in seconds: {:.3}s.",
            ddnnf.rc(),
            elapsed_time
        );
    }

    // change the number of threads used for cardinality of features and partial configurations
    if matches.contains_id("threads") {
        let threads: u16 = match matches
            .value_of("threads")
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
    if matches.contains_id("features") {
        let features: Vec<i32> =
            dparser::parse_features(matches.values_of_lossy("features"));
        ddnnf.execute_query(features);
    }

    // computes the cardinality of partial configurations and saves the results in a .txt file
    // the results do not have to be in the same order if the number of threads is greater than one
    if matches.contains_id("queries") {
        let file_path_in = matches.value_of("queries").unwrap();
        let file_path_out = &format!(
            "{}{}",
            matches.value_of("save_as").unwrap_or("out"),
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
    if matches.contains_id("card_of_fs") {
        let file_path = &format!(
            "{}{}",
            matches.value_of("save_as").unwrap_or("out"),
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
            &matches.value_of("file_path").unwrap(),
            file_path,
            elapsed_time,
            elapsed_time / ddnnf.number_of_variables as f64
        );
    }

    if matches.contains_id("ddnnf") {
        let path = &format!(
            "{}{}",
            matches.value_of("ddnnf").unwrap(),
            ".nnf"
        );
        write_ddnnf(ddnnf, path).unwrap();        
    }
}
