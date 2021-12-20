#![allow(dead_code)]
#![allow(unused_imports)]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

extern crate clap;

use clap::{
    App,
    Arg,
    AppSettings,
    crate_authors,
    crate_version,
};

extern crate colour;
use colour::*;

use rug::Integer;

use rustyline::error::ReadlineError;
use rustyline::Editor;

mod parser;
use crate::parser::{
    build_ddnnf_tree_with_extras,
    parse_queries_file,
    parse_features
};

mod data_structure;
use crate::data_structure::*;

use std::{
    time::Instant,
    collections::HashSet
};

fn main() {

    if true{ // RELEASE_MODE? if true then the executable gets created and requires file input and operation input

    let matches = App::new("dknife")
        .global_settings(&[AppSettings::ColoredHelp])
        .author(crate_authors!())
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
        .arg(Arg::with_name("HEURISTICS")
            .display_order(5)
            .requires("FILE PATH")
            .help("Provides information about the type of nodes, their connection and the different paths.")
            .long("heuristics"))
        .arg(Arg::with_name("CUSTOM OUTPUT FILE NAME")
            .display_order(6)
            .requires("FILE PATH")
            .help("Allows a custom file name for output file for the cardinality of features and file queries. The appropiate file ending gets added automcaticly.")
            .short("s")
            .long("save_as")
            .takes_value(true))
        .get_matches();

    // create the ddnnf based of the input file that is required
    let time = Instant::now();
    let mut ddnnf: Ddnnf = build_ddnnf_tree_with_extras(matches.value_of("FILE PATH").unwrap());
    let elapsed_time = time.elapsed().as_secs_f32();

    println!("Ddnnf overall count: {:#?}", ddnnf.nodes[ddnnf.number_of_nodes - 1].count);

    // if you remove the comment below the whole ddnnf gets printed (only usefull for very small inputs)
    //println!("Ddnnf structure: {:#?}", ddnnf);

    println!("Elapsed time for parsing and overall count in seconds: {:.3}s.", elapsed_time);
    
    if matches.is_present("FEATURE/S") {
        let features: Vec<i32> = parse_features(matches.values_of_lossy("FEATURE/S"));
        ddnnf.execute_query(features);
    }

    if matches.is_present("FILE QUERIES"){
        let file_path_in = matches.value_of("FILE QUERIES").unwrap();
        let file_path_out = &format!("{}{}", matches.value_of("CUSTOM OUTPUT FILE NAME").unwrap_or("out"), ".txt");
        
        let time = Instant::now();
        ddnnf.card_multi_queries(file_path_in, file_path_out).unwrap_or_default();
        let elapsed_time = time.elapsed().as_secs_f64();
    
        println!("Computed values of all queries in {} and the results are saved in {}\n
                It took {} seconds. That is an average of {} seconds per query",  
                file_path_in, file_path_out, elapsed_time, elapsed_time/parse_queries_file(file_path_in).len() as f64);
    }

    if matches.is_present("INTERACTIVE"){
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
                            match features{
                                Some(f) => ddnnf.execute_query(f),
                                None => ()
                            }
                            
                        },
                    }
                },
                Err(ReadlineError::Interrupted) => {
                    println!("CTRL-C: program closes...");
                    break
                },
                Err(ReadlineError::Eof) => {
                    println!("CTRL-D: program closes...");
                    break
                },
                Err(err) => {
                    println!("Error: {:?}", err);
                    break
                }
            }
        }
        rl.save_history("history.txt").unwrap();
    }

    if matches.is_present("HEURISTICS"){
        let node_count: u64 = ddnnf.number_of_nodes as u64;

        let (a, o, l, t, f): (u64, u64, u64, u64, u64) = ddnnf.get_nodetype_numbers();
        println!("\nThe d-DNNF consists out of the following node types:\n\
                \t |-> {:?} out of {:?} are And nodes (≈{:.2}% of total)\n\
                \t |-> {:?} out of {:?} are Or nodes (≈{:.2}% of total)\n\
                \t |-> {:?} out of {:?} are Literal nodes (≈{:.2}% of total)\n\
                \t |-> {:?} out of {:?} are True nodes (≈{:.2}% of total)\n\
                \t |-> {:?} out of {:?} are False nodes (≈{:.2}% of total)\n",
                a, node_count, (a as f64/node_count as f64) * 100 as f64, o, node_count, (o as f64/node_count as f64)  * 100 as f64,
                l, node_count, (l as f64/node_count as f64) * 100 as f64,
                t, node_count, (t as f64/node_count as f64)  * 100 as f64, f, node_count, (f as f64/node_count as f64) * 100 as f64);

        
        let (childs_number, mean_and_childs): (u64, u64) = ddnnf.get_child_number();
        println!("\nThe d-DNNF has the following information regarding node count:\n\
                \t |-> The overall count of child connections is {:?}\n\
                \t |-> The overall node count is {:?}.\n\
                \t |-> There are {:.2} times as much connections as nodes\n\
                \t |-> Each of the {:?} And nodes has an average of ≈{:.2} child nodes\n",
                childs_number, node_count, childs_number as f64/node_count as f64, a, mean_and_childs as f64/a as f64);

        let (lowest, highest, mean, s_x, length): (u64, u64, f64, f64, u64) = ddnnf.get_depths();
        println!("\nThe d-DNNF has the following length attributes:\n\
                \t |-> The shortest path is {:?} units long\n\
                \t |-> The longest path is {:?} units long\n\
                \t |-> The mean path is ≈{:.2} units long\n\
                \t |-> The standard derivation is ≈{:.2} units\n\
                \t |-> There are {:?} different paths. (different paths can sometimes just differ by one node)\n",
                lowest, highest, mean, s_x, length);
    }

    if matches.is_present("CARDINALITY OF FEATURES"){
        let file_path = &format!("{}{}", matches.value_of("CUSTOM OUTPUT FILE NAME").unwrap_or("out"), ".csv");
        
        let time = Instant::now();
        ddnnf.card_of_each_feature_to_csv(file_path).unwrap_or_default();
        let elapsed_time = time.elapsed().as_secs_f64();
    
        println!("Computed the Cardinality of all features in {} and the results are saved in {}\n
                It took {} seconds. That is an average of {} seconds per feature", 
                &matches.value_of("FILE PATH").unwrap(), file_path, elapsed_time, elapsed_time/ddnnf.number_of_variables as f64);
    }

    }else{

    //let args: Vec<String> = env::args().collect();
    //let args: Vec<String> = vec![String::from("filler"), String::from("example_input/c2d_manual.dimacs.nnf")];
    //let args: Vec<String> = vec![String::from("filler"), String::from("example_input/automotive2_4.dimacs.nnf")];
    let args: Vec<String> = vec![String::from("filler"), String::from("automotive01.dimacs.nnf")];
    //let args: Vec<String> = vec![String::from("filler"), String::from("example_input/adder.dimacs.nnf")];
    //let args: Vec<String> = vec![String::from("filler"), String::from("example_input/uClinux-base.dimacs.nnf")];

    let time = Instant::now();
    let mut ddnnf: Ddnnf = build_ddnnf_tree_with_extras(&args[1]);
    let elapsed_time = time.elapsed().as_secs_f32();

    println!("Ddnnf overall count: {:#?}", ddnnf.nodes[ddnnf.number_of_nodes - 1].count);

    // if you remove the comment below the whole ddnnf gets printed (only usefull for very small inputs)
    //println!("Ddnnf structure: {:#?}", ddnnf);

    println!("Elapsed time for parsing and overall count in seconds: {:.3}s.", elapsed_time);


    let file_path_in = "in2.txt";
    let file_path_out = &format!("{}{}", "out", ".txt");
    
    let time = Instant::now();
    ddnnf.card_multi_queries(file_path_in, file_path_out).unwrap_or_default();
    let elapsed_time = time.elapsed().as_secs_f64();

    println!("Computed values of all queries in {} and the results are saved in {}\n
    It took {} seconds. That is an average of {} seconds per query",  
    file_path_in, file_path_out, elapsed_time, elapsed_time/parse_queries_file(file_path_in).len() as f64);

    //---- dead and core --------------------------------------------------------------------------------------------------------------------------------

    green_ln!("#core in reality automotive2_4: 1778. Adder: 7. Computed #core: {:?} => {:?}", ddnnf.core.len(), ddnnf.core);
    green_ln!("example for automotive2_4: feature 61 is core but 134002 and 134003 are both occurences of L 61");
    blue_ln!("#dead in reality automotive2_4: 17. Adder: 38. Computed #dead: {:?} => {:?}", ddnnf.dead.len(), ddnnf.dead);

    //---- one feature --------------------------------------------------------------------------------------------------------------------------------

    //for the automotive2_4.dimacs.nnf file
    let feature_number: i32 = 2855; // has a negative literal // 5803 has negative literal and just a few parent nodes
    //let feature_number: i32 = 17698; // has no negative literal

    //for adder
    //let feature_number: i32 = 90;
    //let feature_number: i32 = 384;

    let time = Instant::now();
    ddnnf.card_of_feature(&feature_number);
    let elapsed_time = time.elapsed().as_secs_f32();

    println!("Feature count for feature number {:?}: {:#?}", feature_number, ddnnf.nodes[ddnnf.number_of_nodes - 1].temp);
    println!("Elapsed time for one feature: {:.3} seconds.\n", elapsed_time);

    let time = Instant::now();
    let res: Integer = ddnnf.card_of_feature_with_marker(&feature_number);
    let elapsed_time = time.elapsed().as_secs_f32();
    let count = get_counter();

    println!("Feature count for feature number {:?}: {:#?}", feature_number, res);
    println!("{:?} nodes were marked (note that nodes which are on multiple paths are only marked once). That are ≈{:.2}% of the ({}) total nodes",
        count, count as f64/ddnnf.number_of_nodes as f64 * 100.0, ddnnf.number_of_nodes);
    println!("Elapsed time for one feature: {:.6} seconds.\n", elapsed_time);
    

    //---- partial config --------------------------------------------------------------------------------------------------------------------------------

    //for the automotive2_4.dimacs.nnf file
    let features: Vec<i32> = vec![2264, 17698, 10341, 12843, 13922, -8264, -3826, -9237, -17107, 16763, 16769, 8380, 13240];//
    //let features: Vec<i32> = vec![2]; // for c2d manual and debugging

    let time = Instant::now();
    let res: Integer = ddnnf.card_of_partial_config_rec(&features);
    let elapsed_time = time.elapsed().as_secs_f32();

    println!("The cardinality for the partial configuration {:?} is: {:#?}", features, res);
    red_ln!("Elapsed time for a partial configuration in seconds: {:.6}s. It was computed with recursion using decision nodes", elapsed_time);

    let time = Instant::now();
    let res: Integer = ddnnf.card_of_partial_config(&features);
    let elapsed_time = time.elapsed().as_secs_f32();

    println!("The cardinality for the partial configuration {:?} is: {:#?}", features, res);
    red_ln!("Elapsed time for a partial configuration in seconds: {:.6}s.", elapsed_time);

    /*
    let time = Instant::now();
    let res: Integer = ddnnf.card_of_partial_config_rec_up(&features);
    let elapsed_time = time.elapsed().as_secs_f32();

    println!("The cardinality for the partial configuration {:?} is: {:#?}", features, res);
    println!("Elapsed time for a partial configuration in seconds: {:.6}s. It was computed with recursion and given values uppwards", elapsed_time);
    */
    
    let time = Instant::now();
    let res: Integer = ddnnf.card_of_partial_config_with_marker(&features);
    let elapsed_time = time.elapsed().as_secs_f32();

    println!("The cardinality for the partial configuration {:?} is: {:#?}", features, res);
    println!("{:?} nodes were marked (note that nodes which are on multiple paths are only marked once). That are ≈{:.2}% of the ({}) total nodes",
        get_counter(), get_counter() as f64/ddnnf.number_of_nodes as f64 * 100.0, ddnnf.number_of_nodes);
    println!("Elapsed time for a partial configuration in seconds: {:.6}s.", elapsed_time);

    //---- cardinality of features --------------------------------------------------------------------------------------------------------------------------------

    // create the filename for the csv file and place it in the output folder
    //let name: &Vec<&str> = &args[1].split(|c: char| c == '/' || c == '.').collect();
    //let file_path = &format!("output/{}{}", name[0], ".csv");
    let file_path_ = "out.csv";

    let time = Instant::now();
    ddnnf.card_of_each_feature_to_csv(file_path_).unwrap_or_default();
    let elapsed_time = time.elapsed().as_secs_f64();

    println!("Computed the Cardinality of all features in {} and the results are saved in {}.\n
            It took {} seconds. That are {} seconds per feature", &args[1], file_path_, elapsed_time, elapsed_time/ddnnf.number_of_variables as f64);

    
    //---- heuristics --------------------------------------------------------------------------------------------------------------------------------
    
    
    let node_count: u64 = ddnnf.nodes.len() as u64;
    
    let (a, o, l, t, f): (u64, u64, u64, u64, u64) = ddnnf.get_nodetype_numbers();
    println!("\nThe d-DNNF consists out of the following node types:\n\
            \t |-> {:?} out of {:?} are And nodes (≈{:.2}% of total)\n\
            \t |-> {:?} out of {:?} are Or nodes (≈{:.2}% of total)\n\
            \t |-> {:?} out of {:?} are Literal nodes (≈{:.2}% of total)\n\
            \t |-> {:?} out of {:?} are True nodes (≈{:.2}% of total)\n\
            \t |-> {:?} out of {:?} are False nodes (≈{:.2}% of total)\n",
            a, node_count, (a as f64/node_count as f64) * 100 as f64, o, node_count, (o as f64/node_count as f64)  * 100 as f64,
            l, node_count, (l as f64/node_count as f64) * 100 as f64,
            t, node_count, (t as f64/node_count as f64)  * 100 as f64, f, node_count, (f as f64/node_count as f64) * 100 as f64);

    /*
    let (childs_number, mean_and_childs): (u64, u64) = ddnnf.get_child_number();
    println!("\nThe d-DNNF has the following information regarding node count:\n\
            \t |-> The overall count of child connections is {:?}\n\
            \t |-> The overall node count is {:?}.\n\
            \t |-> There are {:.2} times as much connections as nodes\n\
            \t |-> Each of the {:?} And nodes has an average of ≈{:.2} child nodes\n",
            childs_number, node_count, childs_number as f64/node_count as f64, a, mean_and_childs as f64/a as f64);

    let (lowest, highest, mean, s_x, length): (u64, u64, f64, f64, u64) = ddnnf.get_depths();
    println!("\nThe d-DNNF has the following length attributes:\n\
            \t |-> The shortest path is {:?} units long\n\
            \t |-> The longest path is {:?} units long\n\
            \t |-> The mean path is ≈{:.2} units long\n\
            \t |-> The standard derivation is ≈{:.2} units\n\
            \t |-> There are {:?} different paths. (different paths can sometimes just differ by one node)\n",
            lowest, highest, mean, s_x, length);
    
    println!("The total time in seconds is {:.2}", total_time.elapsed().as_secs_f32());*/
    }
}