//! A collection of small utility functions.

use std::{fs::File, io::{BufReader, BufRead}, process};

use rug::{Integer, Complete};

use crate::Node;


// multiplies the count of all child Nodes of an And Node
#[inline]
pub(super) fn calc_and_count(nodes: &mut [Node], indices: &[usize]) -> Integer {
    Integer::product(indices.iter().map(|&index| &nodes[index].count))
        .complete()
}

// adds up the count of all child Nodes of an And Node
#[inline]
pub(super) fn calc_or_count(
    nodes: &mut [Node],
    indices: &[usize],
) -> Integer {
    Integer::sum(indices.iter().map(|&index| &nodes[index].count)).complete()
}

/// Is used to parse the queries in the config files
/// The format is:
/// -> A feature is either positiv or negative i32 value with a leading "-"
/// -> Multiple features in the same line form a query
/// -> Queries are seperated by a new line ("\n")
///
/// # Example
/// ```
/// use ddnnf_lib::parser::util::parse_queries_file;
///
/// let config_path = "./tests/data/auto1.config";
/// let queries: Vec<(usize, Vec<i32>)> = parse_queries_file(config_path);
///
/// assert_eq!((0, vec![1044, 885]), queries[0]);
/// assert_eq!((1, vec![1284, -537]), queries[1]);
/// assert_eq!((2, vec![-1767, 675]), queries[2]);
/// ```
/// # Panic
///
/// Panics for a path to a non existing file
pub fn parse_queries_file(path: &str) -> Vec<(usize, Vec<i32>)> {
    let file = open_file_savely(path);

    let lines = BufReader::new(file)
        .lines()
        .map(|line| line.expect("Unable to read line"));
    let mut parsed_queries: Vec<(usize, Vec<i32>)> = Vec::new();

    for (line_number, line) in lines.enumerate() {
        // takes a line of the file and parses the i32 values
        let res: Vec<i32> = line.split_whitespace()
            .map(|elem| elem.parse::<i32>()
            .unwrap_or_else(|_| panic!("Unable to parse {elem} into an i32 value while trying to parse the querie file at {path}.\nCheck the help page with \"-h\" or \"--help\" for further information.\n"))
        ).collect();
        parsed_queries.push((line_number, res));
    }
    parsed_queries
}

/// Tries to open a file.
/// If an error occurs the program prints the error and exists.
pub fn open_file_savely(path: &str) -> File {
    // opens the file with a BufReader and
    // works off each line of the file data seperatly
    match File::open(path) {
        Ok(x) => x,
        Err(err) => {
            // Bold, Red, Foreground Color (see https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797)
            eprintln!("\x1b[1;38;5;196mERROR: The following error code occured while trying to open the file \"{}\":\n{}\nAborting...", path, err);
            process::exit(1);
        }
    }
}

/// Builds a string from a vector without commas or brackets.
/// Example: format_vec([1, 2, 3]) => "1 2 3".
pub fn format_vec<T: ToString>(vals: impl Iterator<Item = T>) -> String {
    vals.map(|v| v.to_string()).collect::<Vec<String>>().join(" ")
}

/// Does the same as [format_vec] and divides the vectors by a semicolon.
pub fn format_vec_vec<T>(vals: impl Iterator<Item = T>) -> String
    where
    T: IntoIterator,
    T::Item: ToString,
{
    vals.map(|res| format_vec(res.into_iter()))
    .collect::<Vec<String>>()
    .join(";")
}