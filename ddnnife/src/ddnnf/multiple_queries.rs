use crate::{Ddnnf, parser};
use std::path::Path;
use std::{error::Error, io::Write};

impl Ddnnf {
    /// Computes the given operation for all queries in path_in.
    /// The results are saved in the path_out. The .csv ending always gets added to the user input.
    pub fn operate_on_queries<T: ToString + Ord + Send + 'static>(
        &mut self,
        operation: fn(&mut Ddnnf, &[i32]) -> T,
        path_in: &Path,
        mut output: impl Write,
    ) -> Result<(), Box<dyn Error>> {
        let work_queue: Vec<(usize, Vec<i32>)> = parser::parse_queries_file(path_in);

        for (_, work) in &work_queue {
            let cardinality = operation(self, work);
            let mut features_str = work
                .iter()
                .fold(String::new(), |acc, &num| acc + &num.to_string() + " ");
            features_str.pop();
            let data = &format!("{},{}\n", features_str, cardinality.to_string());
            output.write_all(data.as_bytes())?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::Ddnnf;
    use crate::parser::build_ddnnf;
    use file_diff::diff_files;
    use itertools::Itertools;
    use num::BigInt;
    use std::{
        fs::{self, File},
        io::{BufRead, BufReader, BufWriter},
        path::Path,
    };

    #[test]
    fn card_multi_queries() {
        let mut ddnnf: Ddnnf = build_ddnnf(Path::new("./tests/data/VP9_d4.nnf"), Some(42));

        let output =
            BufWriter::new(File::create("./tests/data/pcs.csv").expect("Unable to create file"));

        ddnnf
            .operate_on_queries(
                Ddnnf::execute_query,
                Path::new("./tests/data/VP9.config"),
                output,
            )
            .unwrap();

        let mut actual = File::open("./tests/data/pcs.csv").unwrap();
        let mut expected = File::open("./tests/data/VP9_sb_pc.csv").unwrap();

        // diff_files is true if the files are identical
        assert!(
            diff_files(&mut actual, &mut expected),
            "partial config results differ from the expected results"
        );

        fs::remove_file("./tests/data/pcs.csv").unwrap();
    }

    #[test]
    fn sat_multi_queries() {
        let mut ddnnf: Ddnnf = build_ddnnf(Path::new("./tests/data/VP9_d4.nnf"), Some(42));

        let output =
            BufWriter::new(File::create("./tests/data/sat.csv").expect("Unable to create file"));

        ddnnf
            .operate_on_queries(Ddnnf::sat_mut, Path::new("./tests/data/VP9.config"), output)
            .unwrap();

        let sat_results = File::open("./tests/data/sat.csv").unwrap();
        let lines = BufReader::new(sat_results)
            .lines()
            .map(|line| line.expect("Unable to read line"));

        for line in lines {
            // a line has the format "[QUERY],[RESULT]"
            let split_query_res = line.split(',').collect_vec();

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

        fs::remove_file("./tests/data/sat.csv").unwrap();
    }
}
