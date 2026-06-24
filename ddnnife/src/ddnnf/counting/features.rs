use super::super::Ddnnf;
use log::info;
use num::{BigInt, BigRational, ToPrimitive};
use std::error::Error;
use std::path::Path;

impl Ddnnf {
    /// Calculates the count of multiple literals (iterables) each under fixed assumptions.
    /// Results in one count per iterable, each valid under the given assumptions together with the iterable.
    ///
    /// Uses partial derivatives and therefore does not involve re-counting.
    pub fn count_iterables(&mut self, assumptions: &[i32], iterables: &[i32]) -> Vec<BigInt> {
        // Calculate the original count under the given assumptions.
        let original = self.execute_query(assumptions);
        info!("Count under assumptions: {original}");

        // Calculate the partial derivatives under the given assumptions.
        self.annotate_partial_derivatives_assumptions(assumptions);

        // Calculate the count for each iterable.
        iterables
            .iter()
            .map(|iterable| {
                // We cannot flip an already "flipped" variable again
                if assumptions.contains(iterable) {
                    return original.clone();
                }
                // Check whether there is a node for the inverse literal.
                match self.literals.get(&-iterable) {
                    // When there is a node, use its partial derivative to reduce the original count.
                    Some(&node) => &original - &self.nodes[node].partial_derivative,
                    // Otherwise simply return the original count.
                    None => original.clone(),
                }
            })
            .collect()
    }

    pub fn card_of_each_feature(&mut self) -> impl Iterator<Item = (i32, BigInt, f64)> + '_ {
        self.annotate_partial_derivatives();
        let rc = self.rc();
        (1_i32..=self.number_of_variables as i32).map(move |variable| {
            let cardinality = self.card_of_feature_with_partial_derivatives(variable);
            let ratio = BigRational::from((cardinality.clone(), rc.clone()))
                .to_f64()
                .unwrap();
            (variable, cardinality, ratio)
        })
    }

    #[inline]
    /// Computes the cardinality of features for all features in a model.
    /// The results are saved in the file_path.
    /// The function exclusively uses the marking based function.
    /// Here the number of threads influence the speed by using a shared work queue.
    /// # Example
    /// ```
    /// use ddnnife::Ddnnf;
    /// use ddnnife::parser::*;
    /// use std::fs;
    /// use std::path::Path;
    ///
    /// // create a ddnnf
    /// // and run the queries
    /// let mut ddnnf: Ddnnf = build_ddnnf(Path::new("./tests/data/small_ex_c2d.nnf"), None);
    /// ddnnf.card_of_each_feature_csv(Path::new("./tests/data/smt_out.csv"))
    ///      .unwrap_or_default();
    /// let _rm = fs::remove_file("./tests/data/smt_out.csv");
    ///
    /// ```
    pub fn card_of_each_feature_csv(&mut self, file_path: &Path) -> Result<(), Box<dyn Error>> {
        self.annotate_partial_derivatives();

        // start the csv writer with the file_path
        let mut wtr = csv::Writer::from_path(file_path)?;

        self.card_of_each_feature()
            .for_each(|(variable, cardinality, ratio)| {
                wtr.write_record(vec![
                    variable.to_string(),
                    cardinality.to_string(),
                    format!(
                        "{:.10e}",
                        ratio.to_f64().expect("Failed to convert rational!")
                    ),
                ])
                .unwrap();
            });

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::parser::build_ddnnf;
    use file_diff::diff_files;
    use std::fs::{self, File};
    use std::path::Path;
    use std::str::FromStr;

    use super::*;

    #[test]
    fn count_multiple() {
        let mut ddnnf = build_ddnnf(Path::new("./tests/data/auto1_c2d.nnf"), None);
        let result = ddnnf.count_iterables(&[], &[3, 5]);
        let expected = vec![
            BigInt::from_str(
                "387318808678285623197749596912501131418090330323710718027504972750867287833005709915497141252680660472087217940901765442843400489888255735329030409499443200000000000000000000000",
            ).unwrap(),
            BigInt::from_str(
                "19558927176703111970630256604805514155483434271585300434767867070716031103221451966433957821099471645560819328082732687237237092292630699169456731901173780106030980101589212366582299152026173440000000000000000000000000",
            ).unwrap(),
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn count_multiple_assumptions() {
        let mut ddnnf = build_ddnnf(Path::new("./tests/data/auto1_c2d.nnf"), None);
        let result = ddnnf.count_iterables(&[1, -2, 3], &[6, 7, 42, -42]);
        let expected = vec![
            BigInt::from_str(
                "384036445892876423001158498633581630304377700405713169569644761117385361664929390339942080733590146400289868636317852176378625909465473907063530151791820800000000000000000000000",
            ).unwrap(),
            BigInt::from_str(
                "19365940433914281159887479845625056570904516516184828136213103529095442189043204367066948281569355315080124708098083707795934801510156013623320772608000000000000000000000000000",
            ).unwrap(),
            BigInt::ZERO,
            BigInt::from_str(
                "387318808678285623197749596912501131418090330323710718027504972750867287833005709915497141252680660472087217940901765442843400489888255735329030409499443200000000000000000000000",
            ).unwrap(),
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn card_multi_queries() {
        let mut ddnnf: Ddnnf = build_ddnnf(Path::new("./tests/data/VP9_d4.nnf"), Some(42));
        ddnnf.max_worker = 1;
        ddnnf
            .card_of_each_feature_csv(Path::new("./tests/data/fcs.csv"))
            .unwrap();

        ddnnf.max_worker = 4;
        ddnnf
            .card_of_each_feature_csv(Path::new("./tests/data/fcm.csv"))
            .unwrap();

        let mut is_single = File::open("./tests/data/fcs.csv").unwrap();
        let mut is_multi = File::open("./tests/data/fcm.csv").unwrap();
        let mut should_be = File::open("./tests/data/VP9_sb_fs.csv").unwrap();

        // diff_files is true if the files are identical
        assert!(
            diff_files(&mut is_single, &mut is_multi),
            "card of features results of single und multi variant have differences"
        );
        is_single = File::open("./tests/data/fcs.csv").unwrap();
        assert!(
            diff_files(&mut is_single, &mut should_be),
            "card of features results differ from the expected results"
        );

        fs::remove_file("./tests/data/fcs.csv").unwrap();
        fs::remove_file("./tests/data/fcm.csv").unwrap();
    }

    #[test]
    fn test_card_of_features_pd() {
        let pd_file = Path::new("./tests/data/cof_pd.csv");
        let should_file = Path::new("./tests/data/VP9_sb_fs.csv");

        let mut ddnnf: Ddnnf = build_ddnnf(Path::new("./tests/data/VP9_d4.nnf"), Some(42));
        ddnnf.max_worker = 1;
        ddnnf.card_of_each_feature_csv(pd_file).unwrap();

        let mut pd: File = File::open(pd_file).unwrap();
        let mut should_be = File::open(should_file).unwrap();

        assert!(
            diff_files(&mut pd, &mut should_be),
            "card of features results differ from the expected results"
        );

        fs::remove_file(pd_file).unwrap();
    }
}
