#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

extern crate clap;
use clap::{/*crate_authors,*/ crate_version, App, AppSettings, Arg};
use ddnnf_lib::parser::bufreader_for_big_files::BufReaderMl;

use std::collections::HashMap;
use std::time::Instant;

use std::fs::File;
use std::io::{BufWriter, Write};

pub use ddnnf_lib::parser as dparser;
pub use dparser::lexer;
pub use lexer::{TId, Token};

pub use ddnnf_lib::Node;

/// For some reason dsharp is not able to smooth a d-dnnf while still satisfy
/// the standard. Therefore, we have to adjust the resulting d-dnnf to enforce all rules.
/// dsharp violates the standard by generating two instances of the same literal for some
/// literals. We can resolve that issue by replacing the first occurence with a dummy value.
/// The dummy ensures the validity of the counting algorithm for other literals and configurations
/// that do not contain the doubled literal. We choose a true node as dummy.

fn main() {
    let matches = App::new("dhone")
    .global_settings(&[AppSettings::ColoredHelp])
    //.author(crate_authors!())
    .version(crate_version!())
    .setting(AppSettings::ArgRequiredElseHelp)
    .arg(Arg::with_name("FILE PATH")
        .display_order(1)
        .index(1)
        .allow_hyphen_values(true)
        .help("The path to the file in dimacs format."))
    .arg(Arg::with_name("CUSTOM OUTPUT FILE NAME")
        .display_order(2)
        .requires("FILE PATH")
        .help("Name of the output file. If this parameter is not set, we name the output file res.dimacs.nnf")
        .short("s")
        .long("save_as")
        .takes_value(true))
    .get_matches();

    if matches.is_present("FILE PATH") {
        let time = Instant::now();

        let token_stream = preprocess(matches.value_of("FILE PATH").unwrap());

        let output_file = matches
            .value_of("CUSTOM OUTPUT FILE NAME")
            .unwrap_or("res.dimacs.nnf");

        let file: File =
            File::create(output_file).expect("Unable to create file");
        let mut buf_writer = BufWriter::new(file);

        for token in token_stream {
            buf_writer
                .write_all(lexer::serialize_token(token).as_bytes())
                .expect("Unable to write data");
        }

        let elapsed_time = time.elapsed().as_secs_f32();
        println!(
            "Elapsed time for preprocessing {}: {:.3}s.",
            matches.value_of("FILE PATH").unwrap(),
            elapsed_time
        );
    }
}

#[inline]
/// Preprocesses a ddnnf and replaces duplicates of literals (if occuring)
/// with true nodes. There the first occurence always stays the same and all other
/// occurences are swaped with dummy values. We choose true nodes because they do not change
/// the endresult in any way
/// # Examples
///
/// ```
/// extern crate ddnnf_lib;
/// use ddnnf_lib::parser;
///
/// let file_path = "./tests/data/small_test.dimacs.nnf";
/// preprocess(file_path);
/// ```
///
/// # Panics
///
/// The function panics for an invalid file path.
fn preprocess(path: &str) -> Vec<Token> {
    let mut token_stream: Vec<Token> = get_token_stream(path);

    let mut literals: HashMap<i32, usize> = HashMap::with_capacity(1000);

    // indices of nodes that should be replaced with true nodes
    let mut changes: Vec<usize> = Vec::new();

    for (index, token) in token_stream.iter().enumerate() {
        // if we find a literal we save the number and its position
        let unique: i32 = match token {
            (TId::PositiveLiteral, v) => {
                if literals.get(&(v[0] as i32)).is_some() {
                    v[0] as i32
                } else {
                    literals.insert(v[0] as i32, index);
                    0
                }
            }
            (TId::NegativeLiteral, v) => {
                if literals.get(&-(v[0] as i32)).is_some() {
                    -(v[0] as i32)
                } else {
                    literals.insert(-(v[0] as i32), index);
                    0
                }
            }
            _ => 0,
        };

        // if the literal already exists in a previous line then we replace all further
        // occurences with a true node
        if unique != 0 {
            let pos: usize = *literals.get(&unique).unwrap();
            changes.push(pos);
        }
    }

    for change in changes {
        token_stream[change] = (TId::True, vec![]);
    }
    token_stream
}

// generates a token stream from a file path
fn get_token_stream(path: &str) -> Vec<Token> {
    let buf_reader = BufReaderMl::open(path).expect("Unable to open file");
    // we do not know the capacity beforehand without applying semmantics but we know that the file will often be quite big
    let mut parsed_tokens: Vec<Token> = Vec::with_capacity(10000);

    // opens the file with a BufReaderMl which is similar to a regular BufReader
    // works off each line of the file data seperatly
    for line in buf_reader {
        let line = line.expect("Unable to read line");
        parsed_tokens.push(dparser::lexer::lex_line(line.as_ref()).unwrap().1);
    }

    parsed_tokens
}
