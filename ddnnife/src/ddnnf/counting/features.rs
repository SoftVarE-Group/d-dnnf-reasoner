use super::super::Ddnnf;
use num::{BigInt, BigRational, ToPrimitive};
use std::error::Error;
use std::path::Path;

impl Ddnnf {
    pub fn card_of_each_feature(&mut self) -> impl Iterator<Item = (i32, BigInt, f64)> + '_ {
        self.annotate_partial_derivatives();
        let rc = self.rc();
        (1_i32..self.number_of_variables as i32 + 1).map(move |variable| {
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

    use super::*;

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
