use clap::{Parser, Subcommand};
use ddnnife::ddnnf::anomalies::t_wise_sampling::save_sample_to_file;
use ddnnife::ddnnf::Ddnnf;
use ddnnife::parser::{
    self as dparser,
    persisting::{write_as_mermaid_md, write_ddnnf_to_file},
};
use ddnnife::util::format_vec;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(name = "ddnnife", version, arg_required_else_help(true))]
struct Cli {
    /// The path to either a dDNNF file in c2d or d4 format or a CNF file. The ddnnf has to be either fulfill the requirements
    /// of the c2d format and be smooth or produced by the newest d4 compiler version to work properly!
    #[arg(verbatim_doc_comment)]
    file_path: Option<String>,

    /// Allows to load the ddnnf via stdin.
    /// Either 'total_features' has to be set or the file must start with a header of the form 'nnf n v e',
    /// where v is the number of nodes, e is the number of edges,
    /// and n is the number of variables over which the d-dnnf is defined.
    /// Like the c2d and the d4 format specifies, each line must be defided by a new line.
    /// Two following new lines end the reading from stdin.
    #[arg(short, long, verbatim_doc_comment)]
    pipe_ddnnf_stdin: bool,

    /// Choose one of the available
    #[clap(subcommand)]
    operation: Option<Operation>,

    /// The number of total features.
    /// This is strictly necessary if the ddnnf has the d4 format respectivily does not contain a header.
    #[arg(short, long, verbatim_doc_comment)]
    total_features: Option<u32>,

    /// Save the smooth ddnnf in the c2d format. Default output file is '{FILE_NAME}-saved.nnf'.
    /// Alternatively, you can choose a name. The .nnf ending is added automatically.
    #[arg(long, verbatim_doc_comment)]
    save_ddnnf: Option<String>,

    /// Provides information about the type of nodes, their connection, and the different paths.
    #[arg(long, verbatim_doc_comment)]
    heuristics: bool,
}

#[derive(Debug, Clone, Subcommand)]
enum Operation {
    /// Computes the cardinality of features for an assignment.
    #[clap(short_flag = 's')]
    Count {
        /// The numbers of the features that should be included or excluded
        /// (positive number to include, negative to exclude). Can be one or multiple features.
        /// A feature f has to be ∈ ℤ and the only allowed seperator is a whitespace.
        /// The default is no assumption. The Output gets displayed on the terminal.
        #[arg(num_args = 0.., allow_negative_numbers = true, verbatim_doc_comment)]
        features: Option<Vec<i32>>,
    },
    /// Computes the cardinality of a single feature for all features. Is single threaded.
    #[clap(short_flag = 'c')]
    CountFeatures {
        /// Computes the cardinality of features for the feature model,
        /// i.e. the cardinality iff we select one feature for all features.
        /// Default output file is '{FILE_NAME}-features.csv'.
        #[arg(verbatim_doc_comment)]
        custom_output_file: Option<String>,
    },
    /// Computes the cardinality of multiple (partial) configurations.
    #[clap(short_flag = 'q')]
    CountQueries {
        /// Path to a file that may contain multiple queries.
        /// Queries are split by new rows and consist of feature numbers ∈ ℤ that can be negated.
        /// Feature numbers are separated by a space.
        #[arg(verbatim_doc_comment)]
        queries_input_file: String,
        /// Default output file is '{FILE_NAME}-queries.csv'.
        #[arg(verbatim_doc_comment)]
        custom_output_file: Option<String>,
        /// Specify how many threads should be used.
        /// Possible values are between 1 and 32.
        #[arg(short, long, value_parser = clap::value_parser!(u16).range(1..=32), default_value_t = 4, verbatim_doc_comment)]
        jobs: u16,
    },
    /// Computes multiple SAT queries.
    Sat {
        /// Path to a file that may contain multiple queries.
        /// Queries are split by new rows and consist of feature numbers ∈ ℤ that can be negated.
        /// Feature numbers are separated by a space.
        #[arg(verbatim_doc_comment)]
        queries_input_file: String,
        /// Default output file is '{FILE_NAME}-sat.csv'.
        #[arg(verbatim_doc_comment)]
        custom_output_file: Option<String>,
        /// Specify how many threads should be used.
        /// Possible values are between 1 and 32.
        #[arg(short, long, value_parser = clap::value_parser!(u16).range(1..=32), default_value_t = 4, verbatim_doc_comment)]
        jobs: u16,
    },

    /// Starts ddnnife in stream mode.
    Stream {
        /// Specify how many threads should be used.
        /// Possible values are between 1 and 32.
        #[arg(short, long, value_parser = clap::value_parser!(u16).range(1..=32), default_value_t = 1, verbatim_doc_comment)]
        jobs: u16,
    },
    /// Evaluates multiple queries of the stream format from a file.
    StreamQueries {
        /// Path to a file that may contain multiple queries.
        /// Queries are split by new rows and consist of feature numbers ∈ ℤ that can be negated.
        /// Feature numbers are separated by a space.
        #[arg(verbatim_doc_comment)]
        queries_input_file: String,
        /// Default output file is '{FILE_NAME}-stream.csv'.
        #[arg(verbatim_doc_comment)]
        custom_output_file: Option<String>,
    },
    /// Computes t-wise samples
    TWise {
        /// The default ouput file is '{FILE_NAME}-t-wise.csv'.
        #[arg(verbatim_doc_comment)]
        custom_output_file: Option<String>,
        /// The 't' in t-wise sampling refers to the degree of interaction
        /// or combination of input parameters to be considered in each test case.
        /// For example, 2-wise sampling (also known as pairwise testing) considers
        /// all possible combinations of two input parameters.
        /// The higher the value of 't', the more comprehensive the testing will be,
        /// but also the larger the number of test cases required.
        #[clap(short, verbatim_doc_comment, default_value_t = 2)]
        t: usize,
    },
    /// Computes core, dead, false-optional features, and atomic sets.
    Anomalies {
        /// The default ouput file is '{FILE_NAME}-anomalies.csv'.
        #[arg(verbatim_doc_comment)]
        custom_output_file: Option<String>,
    },
    /// Computes all atomic sets for the feature model (under assumptions and for candidates).
    AtomicSets {
        /// The default ouput file is '{FILE_NAME}-atomic.csv'.
        #[arg(verbatim_doc_comment)]
        custom_output_file: Option<String>,
        /// The numbers of the features that should be included or excluded
        /// (positive number to include, negative to exclude).
        /// Can be one or multiple. A feature f has to be ∈ ℤ
        /// and the only allowed seperator is a whitespace.
        /// The default is no assumption.
        #[arg(short, long, allow_negative_numbers = true, num_args = 0.., verbatim_doc_comment)]
        assumptions: Vec<i32>,
        /// Restrictes the potential sets to the result features mentioned.
        /// The default are all features of the model.
        #[clap(short, long, allow_negative_numbers = false, num_args = 0.., verbatim_doc_comment)]
        candidates: Option<Vec<u32>>,
        /// Without the cross flag,
        /// we only consider atomic set candidates of included features.
        /// With the cross flag,
        /// we consider included and excluded feature candidates.
        #[clap(long, verbatim_doc_comment)]
        cross: bool,
    },
    /// Generates uniform random sample
    Urs {
        /// The default ouput file is '{FILE_NAME}-urs.csv'.
        #[arg(verbatim_doc_comment)]
        custom_output_file: Option<String>,
        /// The numbers of the features that should be included or excluded
        /// (positive number to include, negative to exclude).
        /// Can be one or multiple. A feature f has to be ∈ ℤ
        /// and the only allowed seperator is a whitespace.
        /// The default is no assumption.
        #[clap(short, long, allow_negative_numbers = true, num_args = 0.., verbatim_doc_comment)]
        assumptions: Vec<i32>,
        /// Reusing the same seed yields the same urs.
        #[clap(short, long, default_value_t = 42)]
        seed: u64,
        /// The amount of samples ddnnife should generate.
        #[clap(short, long, default_value_t = 1000)]
        number: usize,
    },
    /// Computes the core and dead features.
    #[clap(verbatim_doc_comment)]
    Core {
        /// An leading '-' indicates that the feature is dead.
        /// Contrast to that, if the '-' is missing the feature is core.
        /// The default ouput file is '{FILE_NAME}-core.csv'.
        #[arg(verbatim_doc_comment)]
        custom_output_file: Option<String>,
    },
    /// Transforms the smooth d-DNNF into the mermaid.md format.
    #[clap(verbatim_doc_comment)]
    Mermaid {
        /// Default output file is '{FILE_NAME}-mermaid.md'.
        /// Alternatively, you can choose a name. The .md ending is added automatically.
        #[arg(verbatim_doc_comment)]
        custom_output_file: Option<String>,
        /// The numbers of the features that should be included or excluded
        /// (positive number to include, negative to exclude).
        /// The nodes will be annotaded with their count regarding this query
        /// while using the marking algorithm.
        /// Can be one or multiple. A feature f has to be ∈ ℤ
        /// and the only allowed seperator is a whitespace.
        /// The default is no assumption.
        #[clap(short, long, allow_negative_numbers = true, num_args = 0.., verbatim_doc_comment)]
        assumptions: Vec<i32>,
    },
}

fn main() {
    let cli = Cli::parse();

    // create the ddnnf based of the input file that is required
    let time = Instant::now();
    let mut ddnnf: Ddnnf;

    if cli.pipe_ddnnf_stdin {
        // read model line by line from stdin
        let mut input = Vec::new();
        for line in io::stdin().lock().lines() {
            let read_line = line.unwrap();
            if read_line.is_empty() {
                break;
            }
            input.push(read_line);
        }
        ddnnf = dparser::distribute_building(input, cli.total_features, None);
    } else {
        let ddnnf_path = &cli.file_path.clone().unwrap();
        ddnnf = dparser::build_ddnnf(ddnnf_path, cli.total_features)
    }

    // file path without last extension
    let input_file_path = String::from(
        Path::new(&cli.file_path.unwrap_or(String::from("ddnnf.nnf")))
            .with_extension("")
            .file_name()
            .unwrap()
            .to_str()
            .unwrap(),
    );

    // Uses the supplied file path if there is any.
    // If there is no prefix, we switch to the default fallback.
    let construct_ouput_path = |maybe_prefix: &Option<String>, operation: &str, file_type: &str| {
        format!(
            "{}-{}.{}",
            maybe_prefix.clone().unwrap_or(input_file_path.clone()),
            operation,
            file_type
        )
    };

    // print additional output, iff we are not in the stream mode
    match &cli.operation {
        Some(Operation::Stream { .. }) => (),
        _ => {
            let elapsed_time = time.elapsed().as_secs_f32();
            println!(
                "Ddnnf overall count: {:#?}\nElapsed time for parsing, and overall count in seconds: {:.3}s. \
                (This includes compiling to dDNNF if needed)",
                ddnnf.rc(),
                elapsed_time
            );
        }
    }

    if cli.operation.is_some() {
        let operation = match cli.operation {
            Some(op) => op,
            None => todo!(),
        };

        // change the number of threads used for cardinality of features and partial configurations
        match operation {
            CountQueries { jobs, .. } | Stream { jobs } | Sat { jobs, .. } => {
                ddnnf.max_worker = jobs;
            }
            _ => (),
        }

        // compute the output file path to which the results (if any) will be written
        let output_file_path: String = match &operation {
            CountFeatures {
                custom_output_file, ..
            } => construct_ouput_path(custom_output_file, "features", "csv"),
            CountQueries {
                custom_output_file, ..
            } => construct_ouput_path(custom_output_file, "queries", "csv"),
            Sat {
                custom_output_file, ..
            } => construct_ouput_path(custom_output_file, "sat", "csv"),
            StreamQueries {
                custom_output_file, ..
            } => construct_ouput_path(custom_output_file, "stream", "csv"),
            TWise {
                custom_output_file,
                t,
            } => construct_ouput_path(custom_output_file, format!("{}-wise", t).as_str(), "csv"),
            Anomalies { custom_output_file } => {
                construct_ouput_path(custom_output_file, "anomalies", "txt")
            }
            AtomicSets {
                custom_output_file, ..
            } => construct_ouput_path(custom_output_file, "atomic", "csv"),
            Urs {
                custom_output_file, ..
            } => construct_ouput_path(custom_output_file, "urs", "csv"),
            Core { custom_output_file } => construct_ouput_path(custom_output_file, "core", "csv"),
            Mermaid {
                custom_output_file, ..
            } => construct_ouput_path(custom_output_file, "mermaid", "md"),
            _ => String::new(),
        };

        use Operation::*;
        match &operation {
            AtomicSets {
                custom_output_file: _,
                assumptions,
                candidates,
                cross,
            } => {
                let mut wtr =
                    BufWriter::new(File::create(&output_file_path).expect("Unable to create file"));
                for set in ddnnf.get_atomic_sets(candidates.clone(), assumptions, *cross) {
                    wtr.write_all(format_vec(set.iter()).as_bytes()).unwrap();
                    wtr.write_all("\n".as_bytes()).unwrap();
                }
                wtr.flush().unwrap();
                println!(
                    "\nComputed the atomic sets and saved the results in {}.",
                    output_file_path
                );
            }
            Urs {
                assumptions,
                seed,
                number,
                custom_output_file: _,
            } => {
                let mut wtr =
                    BufWriter::new(File::create(&output_file_path).expect("Unable to create file"));
                for sample in ddnnf
                    .uniform_random_sampling(assumptions, *number, *seed)
                    .unwrap()
                {
                    wtr.write_all(format_vec(sample.iter()).as_bytes()).unwrap();
                    wtr.write_all("\n".as_bytes()).unwrap();
                }
                wtr.flush().unwrap();
                println!(
                    "\nComputed {} uniform random samples and saved the results in {}.",
                    number, output_file_path
                );
            }
            TWise {
                t,
                custom_output_file: _,
            } => {
                let sample_result = ddnnf.sample_t_wise(*t);
                save_sample_to_file(&sample_result, &output_file_path).unwrap();
                println!(
                    "\nComputed {}-wise samples and saved the results in {}.",
                    t, output_file_path
                );
            }
            // computes the cardinality for the partial configuration that can be mentioned with parameters
            Count { features } => {
                let features = features.clone().unwrap_or(vec![]);
                println!(
                    "\nDdnnf count for query {:?} is: {:?}",
                    &features,
                    ddnnf.execute_query(&features)
                );
                let marked_nodes = ddnnf.get_marked_nodes_clone(&features);
                println!("While computing the cardinality of the partial configuration {} out of the {} nodes were marked. \
                    That are {:.2}%", marked_nodes.len(), ddnnf.nodes.len(), marked_nodes.len() as f64 / ddnnf.nodes.len() as f64 * 100.0);
            }
            // computes the cardinality of features and saves the results in a .csv file
            // the cardinalities are always sorted from lowest to highest (also for multiple threads)
            CountFeatures { .. } => {
                let time = Instant::now();
                ddnnf
                    .card_of_each_feature(&output_file_path)
                    .unwrap_or_default();
                let elapsed_time = time.elapsed().as_secs_f64();

                println!(
                    "\nComputed the Cardinality of all features in {} and the results are saved in {}\n\
                    It took {} seconds. That is an average of {} seconds per feature",
                    input_file_path,
                    output_file_path,
                    elapsed_time,
                    elapsed_time / ddnnf.number_of_variables as f64
                );
            }
            CountQueries {
                queries_input_file, ..
            } => {
                compute_queries(
                    &mut ddnnf,
                    queries_input_file,
                    &output_file_path,
                    Ddnnf::execute_query,
                );
            }
            Sat {
                queries_input_file, ..
            } => {
                compute_queries(
                    &mut ddnnf,
                    queries_input_file,
                    &output_file_path,
                    Ddnnf::sat,
                );
            }
            StreamQueries {
                queries_input_file, ..
            } => {
                let mut wtr =
                    BufWriter::new(File::create(&output_file_path).expect("Unable to create file"));

                let file = dparser::open_file_savely(queries_input_file);
                let queries = BufReader::new(file)
                    .lines()
                    .map(|line| line.expect("Unable to read line"));

                for query in queries {
                    wtr.write_all(ddnnf.handle_stream_msg(&query).as_bytes())
                        .unwrap();
                    wtr.write_all("\n".as_bytes()).unwrap();
                }

                wtr.flush().unwrap();
                println!(
                    "\nComputed stream queries and saved the results in {}.",
                    output_file_path
                );
            }
            // switch in the stream mode
            Stream { .. } => {
                ddnnf.init_stream();
            }
            // writes the anomalies of the d-DNNF to file
            // anomalies are: core, dead, false-optional features and atomic sets
            Anomalies {
                custom_output_file: _,
            } => {
                ddnnf.write_anomalies(&output_file_path).unwrap();
                println!("\nThe anomalies of the d-DNNF (i.e. core, dead, false-optional features, and atomic sets) are written into {}.", output_file_path);
            }
            Core {
                custom_output_file: _,
            } => {
                let mut core: Vec<i32> = ddnnf.core.clone().into_iter().collect();
                core.sort_unstable_by_key(|k| k.abs());

                let mut wtr =
                    BufWriter::new(File::create(&output_file_path).expect("Unable to create file"));
                wtr.write_all(format_vec(core.iter()).as_bytes()).unwrap();
                wtr.write_all("\n".as_bytes()).unwrap();
                println!(
                    "\nComputed the core / dead features and saved the results in {}.",
                    output_file_path
                );
            }
            Mermaid {
                custom_output_file: _,
                assumptions,
            } => {
                write_as_mermaid_md(&mut ddnnf, assumptions, &output_file_path).unwrap();
                println!("The smooth d-DNNF was transformed into mermaid markdown format and was written in {}.", output_file_path);
            }
        }
    }

    // writes the d-DNNF to file
    if cli.save_ddnnf.is_some() {
        let path = construct_ouput_path(&cli.save_ddnnf, "saved", "nnf");
        write_ddnnf_to_file(&ddnnf, &path).unwrap();
        println!(
            "\nThe smooth d-DNNF was written into the c2d format in {}.",
            path
        );
    }

    // prints heuristics
    if cli.heuristics {
        ddnnf.print_all_heuristics();
    }
}

fn compute_queries<T: ToString + Ord + Send + 'static>(
    ddnnf: &mut Ddnnf,
    queries_file: &String,
    output_file: &String,
    operation: fn(&mut Ddnnf, query: &[i32]) -> T,
) {
    let time = Instant::now();
    ddnnf
        .operate_on_queries(operation, queries_file, output_file)
        .unwrap_or_default();
    let elapsed_time = time.elapsed().as_secs_f64();

    println!(
        "\nComputed values of all queries in {} and the results are saved in {}\n\
        It took {} seconds. That is an average of {} seconds per query",
        queries_file,
        output_file,
        elapsed_time,
        elapsed_time / dparser::parse_queries_file(queries_file.as_str()).len() as f64
    );
}
