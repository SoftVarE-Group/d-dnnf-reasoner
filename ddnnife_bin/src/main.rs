use clap::{Parser, Subcommand};
use ddnnife::ddnnf::statistics::Statistics;
use ddnnife::ddnnf::Ddnnf;
use ddnnife::parser::{
    self as dparser,
    persisting::{write_as_mermaid_md, write_ddnnf_to_file},
};
use ddnnife::util::format_vec;
use ddnnife_cnf::Cnf;
use log::info;
use std::fs::File;
use std::io::{self, stdout, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(version, arg_required_else_help(true))]
struct Cli {
    /// Input path, stdin when not given.
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Output path, stdout when not given.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Choose one of the available
    #[clap(subcommand)]
    operation: Option<Operation>,

    /// The number of total features.
    /// This is strictly necessary if the ddnnf has the d4 format respectivily does not contain a header.
    #[arg(short, long)]
    total_features: Option<u32>,

    /// Save the smooth ddnnf in the c2d format.
    #[arg(long)]
    save_ddnnf: Option<PathBuf>,

    /// Whether to use projected d-DNNF compilation for the input.
    #[cfg(feature = "d4")]
    #[arg(short, long)]
    projected: bool,

    /// Provides information about the type of nodes, their connection, and the different paths.
    #[arg(long)]
    heuristics: bool,

    /// Logging level for outputting warnings and information such as heuristics.
    #[arg(short, long, default_value_t=log::LevelFilter::Info)]
    logging: log::LevelFilter,
}

#[derive(Debug, Clone, Subcommand)]
enum Operation {
    /// Computes the cardinality of features for an assignment.
    Count {
        /// The numbers of the features that should be included or excluded
        /// (positive number to include, negative to exclude). Can be one or multiple features.
        /// A feature f has to be ∈ ℤ and the only allowed seperator is a whitespace.
        /// The default is no assumption. The Output gets displayed on the terminal.
        #[arg(num_args = 0.., allow_negative_numbers = true)]
        features: Option<Vec<i32>>,
    },
    /// Computes the cardinality of a single feature for all features. Is single threaded.
    CountFeatures,
    /// Computes the cardinality of multiple (partial) configurations.
    CountQueries {
        /// Path to a file that may contain multiple queries.
        /// Queries are split by new rows and consist of feature numbers ∈ ℤ that can be negated.
        /// Feature numbers are separated by a space.
        queries_input_file: PathBuf,
        /// Specify how many threads should be used.
        /// Possible values are between 1 and 32.
        #[arg(short, long, value_parser = clap::value_parser!(u16).range(1..=32), default_value_t = 4)]
        jobs: u16,
    },
    /// Computes multiple SAT queries.
    Sat {
        /// Path to a file that may contain multiple queries.
        /// Queries are split by new rows and consist of feature numbers ∈ ℤ that can be negated.
        /// Feature numbers are separated by a space.
        queries_input_file: PathBuf,
        /// Specify how many threads should be used.
        /// Possible values are between 1 and 32.
        #[arg(short, long, value_parser = clap::value_parser!(u16).range(1..=32), default_value_t = 4)]
        jobs: u16,
    },

    /// Starts ddnnife in stream mode.
    Stream {
        /// Specify how many threads should be used.
        /// Possible values are between 1 and 32.
        #[arg(short, long, value_parser = clap::value_parser!(u16).range(1..=32), default_value_t = 1)]
        jobs: u16,
    },
    /// Evaluates multiple queries of the stream format from a file.
    StreamQueries {
        /// Path to a file that may contain multiple queries.
        /// Queries are split by new rows and consist of feature numbers ∈ ℤ that can be negated.
        /// Feature numbers are separated by a space.
        queries_input_file: PathBuf,
    },
    /// Computes t-wise samples
    TWise {
        /// The 't' in t-wise sampling refers to the degree of interaction
        /// or combination of input parameters to be considered in each test case.
        /// For example, 2-wise sampling (also known as pairwise testing) considers
        /// all possible combinations of two input parameters.
        /// The higher the value of 't', the more comprehensive the testing will be,
        /// but also the larger the number of test cases required.
        #[clap(short, default_value_t = 2)]
        t: usize,
    },
    /// Computes core, dead, false-optional features, and atomic sets.
    Anomalies,
    /// Computes all atomic sets for the feature model (under assumptions and for candidates).
    AtomicSets {
        /// The numbers of the features that should be included or excluded
        /// (positive number to include, negative to exclude).
        /// Can be one or multiple. A feature f has to be ∈ ℤ
        /// and the only allowed seperator is a whitespace.
        /// The default is no assumption.
        #[arg(short, long, allow_negative_numbers = true, num_args = 0..)]
        assumptions: Vec<i32>,
        /// Restrictes the potential sets to the result features mentioned.
        /// The default are all features of the model.
        #[clap(short, long, allow_negative_numbers = false, num_args = 0..)]
        candidates: Option<Vec<u32>>,
        /// Without the cross flag,
        /// we only consider atomic set candidates of included features.
        /// With the cross flag,
        /// we consider included and excluded feature candidates.
        #[clap(long)]
        cross: bool,
    },
    /// Generates uniform random sample
    Urs {
        /// The numbers of the features that should be included or excluded
        /// (positive number to include, negative to exclude).
        /// Can be one or multiple. A feature f has to be ∈ ℤ
        /// and the only allowed seperator is a whitespace.
        /// The default is no assumption.
        #[clap(short, long, allow_negative_numbers = true, num_args = 0..)]
        assumptions: Vec<i32>,
        /// Reusing the same seed yields the same urs.
        #[clap(short, long, default_value_t = 42)]
        seed: u64,
        /// The amount of samples ddnnife should generate.
        #[clap(short, long, default_value_t = 1000)]
        number: usize,
    },
    /// Computes the core and dead features.
    Core,
    /// Transforms the smooth d-DNNF into the mermaid.md format.
    Mermaid {
        /// The numbers of the features that should be included or excluded
        /// (positive number to include, negative to exclude).
        /// The nodes will be annotaded with their count regarding this query
        /// while using the marking algorithm.
        /// Can be one or multiple. A feature f has to be ∈ ℤ
        /// and the only allowed seperator is a whitespace.
        /// The default is no assumption.
        #[clap(short, long, allow_negative_numbers = true, num_args = 0..)]
        assumptions: Vec<i32>,
    },
    /// Outputs statistics about the d-DNNF as JSON.
    Statistics {
        /// Whether to pretty-print the JSON output
        #[arg(short, long)]
        pretty: bool,
    },
    /// Converts a d-DNNF into a CNF by using Tseitin transformations.
    ToCnf,
}

fn main() {
    let cli = Cli::parse();

    pretty_env_logger::formatted_builder()
        .filter_level(cli.logging)
        .init();

    // create the ddnnf based of the input file that is required
    let time = Instant::now();

    let mut ddnnf = if let Some(path) = &cli.input {
        // Read from file.
        #[cfg(feature = "d4")]
        {
            if cli.projected {
                dparser::build_ddnnf_projected(path, cli.total_features)
            } else {
                dparser::build_ddnnf(path, cli.total_features)
            }
        }

        #[cfg(not(feature = "d4"))]
        dparser::build_ddnnf(path, cli.total_features)
    } else {
        // Read from stdin.
        let mut input = Vec::new();
        for line in io::stdin().lock().lines() {
            let read_line = line.unwrap();
            if read_line.is_empty() {
                break;
            }
            input.push(read_line);
        }

        dparser::distribute_building(input, cli.total_features, None)
    };

    // print additional output, iff we are not in the stream mode
    match &cli.operation {
        Some(Operation::Stream { .. }) => (),
        _ => {
            let elapsed_time = time.elapsed().as_secs_f32();
            info!("Ddnnf overall count: {:#?}", ddnnf.rc());
            info!(
                "Elapsed time for parsing, and overall count in seconds: {:.3}s. (This includes compiling to dDNNF if needed)",
                elapsed_time
            );
        }
    }

    if let Some(operation) = cli.operation {
        // change the number of threads used for cardinality of features and partial configurations
        match operation {
            Operation::CountQueries { jobs, .. }
            | Operation::Stream { jobs }
            | Operation::Sat { jobs, .. } => {
                ddnnf.max_worker = jobs;
            }
            _ => (),
        }

        let mut writer: Box<dyn Write> = if let Some(path) = &cli.output {
            Box::new(BufWriter::new(
                File::create(path).expect("Unable to create file"),
            ))
        } else {
            Box::new(BufWriter::new(stdout()))
        };

        match &operation {
            Operation::AtomicSets {
                assumptions,
                candidates,
                cross,
            } => {
                for set in ddnnf.get_atomic_sets(candidates.clone(), assumptions, *cross) {
                    writer.write_all(format_vec(set.iter()).as_bytes()).unwrap();
                    writer.write_all("\n".as_bytes()).unwrap();
                }
            }
            Operation::Urs {
                assumptions,
                seed,
                number,
            } => {
                for sample in ddnnf
                    .uniform_random_sampling(assumptions, *number, *seed)
                    .unwrap()
                {
                    writer
                        .write_all(format_vec(sample.iter()).as_bytes())
                        .unwrap();

                    writer.write_all("\n".as_bytes()).unwrap();
                }
            }
            Operation::TWise { t } => {
                writer
                    .write_all(ddnnf.sample_t_wise(*t).to_string().as_bytes())
                    .unwrap();
            }
            // computes the cardinality for the partial configuration that can be mentioned with parameters
            Operation::Count { features } => {
                let features = features.clone().unwrap_or(vec![]);
                let count = ddnnf.execute_query(&features);

                writer.write_all(count.to_string().as_ref()).unwrap();

                let marked_nodes = ddnnf.get_marked_nodes_clone(&features);
                info!("While computing the cardinality of the partial configuration {} out of the {} nodes were marked. \
                    That are {:.2}%", marked_nodes.len(), ddnnf.nodes.len(), marked_nodes.len() as f64 / ddnnf.nodes.len() as f64 * 100.0);
            }
            // computes the cardinality of features and saves the results in a .csv file
            // the cardinalities are always sorted from lowest to highest (also for multiple threads)
            Operation::CountFeatures => {
                let time = Instant::now();

                let mut csv_writer = csv::Writer::from_writer(writer);

                ddnnf
                    .card_of_each_feature()
                    .for_each(|(variable, cardinality, ratio)| {
                        csv_writer
                            .write_record(vec![
                                variable.to_string(),
                                cardinality.to_string(),
                                format!("{:.10e}", ratio),
                            ])
                            .unwrap();
                    });

                writer = csv_writer.into_inner().unwrap();

                let elapsed_time = time.elapsed().as_secs_f64();

                info!(
                    "Runtime: {} seconds. That is an average of {} seconds per feature.",
                    elapsed_time,
                    elapsed_time / ddnnf.number_of_variables as f64
                );
            }
            Operation::CountQueries {
                queries_input_file, ..
            } => {
                compute_queries(
                    &mut ddnnf,
                    queries_input_file,
                    &mut writer,
                    Ddnnf::execute_query,
                );
            }
            Operation::Sat {
                queries_input_file, ..
            } => {
                compute_queries(&mut ddnnf, queries_input_file, &mut writer, Ddnnf::sat);
            }
            Operation::StreamQueries {
                queries_input_file, ..
            } => {
                let file = dparser::open_file_savely(queries_input_file);
                let queries = BufReader::new(file)
                    .lines()
                    .map(|line| line.expect("Unable to read line"));

                for query in queries {
                    writer
                        .write_all(ddnnf.handle_stream_msg(&query).as_bytes())
                        .unwrap();
                    writer.write_all("\n".as_bytes()).unwrap();
                }

                writer.flush().unwrap();
            }
            // switch in the stream mode
            Operation::Stream { .. } => {
                ddnnf.init_stream();
            }
            // writes the anomalies of the d-DNNF to file
            // anomalies are: core, dead, false-optional features and atomic sets
            Operation::Anomalies => {
                ddnnf.write_anomalies(&mut writer).unwrap();
            }
            Operation::Core => {
                let mut core: Vec<i32> = ddnnf.core.clone().into_iter().collect();
                core.sort_unstable_by_key(|k| k.abs());

                writer
                    .write_all(format_vec(core.iter()).as_bytes())
                    .unwrap();
            }
            Operation::Mermaid { assumptions } => {
                write_as_mermaid_md(&mut ddnnf, assumptions, &mut writer).unwrap();
            }
            Operation::Statistics { pretty } => {
                let statistics = Statistics::from(&ddnnf);

                if *pretty {
                    serde_json::to_writer_pretty(&mut writer, &statistics)
                } else {
                    serde_json::to_writer(&mut writer, &statistics)
                }
                .expect("Unable to serialize statistics.");
            }
            Operation::ToCnf => {
                writer
                    .write_all(Cnf::from(&ddnnf).to_string().as_bytes())
                    .unwrap();
            }
        }

        writer.flush().unwrap();
    }

    // writes the d-DNNF to file
    if let Some(path) = cli.save_ddnnf {
        write_ddnnf_to_file(&ddnnf, &path).unwrap();
        info!(
            "The smooth d-DNNF was written into the c2d format in {}.",
            path.to_str().expect("Failed to serialize path.")
        );
    }
}

fn compute_queries<T: ToString + Ord + Send + 'static>(
    ddnnf: &mut Ddnnf,
    queries_file: &Path,
    output: impl Write,
    operation: fn(&mut Ddnnf, query: &[i32]) -> T,
) {
    let time = Instant::now();

    ddnnf
        .operate_on_queries(operation, queries_file, output)
        .unwrap_or_default();

    let elapsed_time = time.elapsed().as_secs_f64();

    info!(
        "Runtime: {} seconds. That is an average of {} seconds per query.",
        elapsed_time,
        elapsed_time / dparser::parse_queries_file(queries_file).len() as f64
    );
}
