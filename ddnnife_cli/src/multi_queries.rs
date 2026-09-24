use ddnnife::{Ddnnf, util::format_vec_separated_by};
use log::info;
use std::{
    fmt::Display,
    fs::File,
    io::{self, BufRead, BufReader, Write},
    path::Path,
    time::Instant,
};

/// Computes multiple queries as specified in the given file and writes the results to the given output.
///
/// The output format is `l1 l2 ... ln,result` where `l1` to `ln` are the literals of a query.
/// Each result is on a new line.
pub fn compute_queries<T: Display>(
    ddnnf: &mut Ddnnf,
    queries_file: &Path,
    mut output: impl Write,
    operation: fn(&mut Ddnnf, query: &[i32]) -> T,
) -> io::Result<()> {
    let queries: Vec<Vec<i32>> = BufReader::new(File::open(queries_file)?)
        .lines()
        .map(|line| line.expect("Failed to read query"))
        .map(|line| {
            line.split_whitespace()
                .map(|literal| literal.parse::<i32>().expect("Failed to read literal"))
                .collect()
        })
        .collect();

    let time = Instant::now();

    queries.iter().try_for_each(|query| {
        let literals = format_vec_separated_by(query.iter(), " ");
        let result = operation(ddnnf, query);
        writeln!(output, "{literals},{result}")
    })?;

    let elapsed_time = time.elapsed().as_secs_f64();

    info!(
        "Runtime: {} seconds. That is an average of {} seconds per query.",
        elapsed_time,
        elapsed_time / queries.len() as f64
    );

    Ok(())
}

#[cfg(test)]
mod test {
    use super::compute_queries;
    use crate::Ddnnf;
    use num::BigInt;
    use std::{fs, io::BufWriter, path::Path};

    fn card_multi_queries(ddnnf: &mut Ddnnf, queries: &Path, expected: &Path) {
        let mut buffer = BufWriter::new(Vec::new());
        compute_queries(ddnnf, queries, &mut buffer, Ddnnf::execute_query).unwrap();
        let output = String::from_utf8(buffer.into_inner().unwrap()).unwrap();
        let expected = fs::read_to_string(expected).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn vp9_count() {
        card_multi_queries(
            &mut Ddnnf::from_file(Path::new("tests/data/VP9_d4.nnf"), Some(42)),
            Path::new("tests/data/VP9.config"),
            Path::new("tests/data/VP9_sb_pc.csv"),
        );
    }

    #[test]
    fn vp9_sat() {
        let mut ddnnf = Ddnnf::from_file(Path::new("tests/data/VP9_d4.nnf"), Some(42));
        let queries_file = Path::new("tests/data/VP9.config");

        let mut buffer = BufWriter::new(Vec::new());
        compute_queries(&mut ddnnf, queries_file, &mut buffer, Ddnnf::sat).unwrap();
        let output = String::from_utf8(buffer.into_inner().unwrap()).unwrap();

        for line in output.lines() {
            // a line has the format "[QUERY],[RESULT]"
            let split_query_res: Vec<&str> = line.split(',').collect();

            // takes a query of the file and parses the i32 values
            let query: Vec<i32> = split_query_res[0]
                .split_whitespace()
                .map(|elem| elem.parse::<i32>().unwrap())
                .collect();
            let res = split_query_res[1].parse::<bool>().unwrap();

            assert_eq!(ddnnf.sat(&query), res);
            assert_eq!(
                ddnnf.sat(&query),
                (ddnnf.execute_query(&query) > BigInt::ZERO)
            );
        }
    }

    #[test]
    fn auto1_c2d_count() {
        card_multi_queries(
            &mut Ddnnf::from_file(Path::new("tests/data/auto1_c2d.nnf"), Some(42)),
            Path::new("tests/data/auto1.config"),
            Path::new("tests/data/auto1_sb_pc.csv"),
        );
    }

    #[test]
    fn auto1_d4_count() {
        card_multi_queries(
            &mut Ddnnf::from_file(Path::new("tests/data/auto1_d4.nnf"), Some(42)),
            Path::new("tests/data/auto1.config"),
            Path::new("tests/data/auto1_sb_pc.csv"),
        );
    }
}
