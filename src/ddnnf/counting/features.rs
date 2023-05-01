use std::{sync::mpsc, error::Error, thread};

use rug::Float;
use workctl::WorkQueue;

use super::super::Ddnnf;

impl Ddnnf {
    #[inline]
    /// Computes the cardinality of features for all features in a model.
    /// The results are saved in the file_path. The .csv ending always gets added to the user input.
    /// The function exclusively uses the marking based function.
    /// Here the number of threads influence the speed by using a shared work queue.
    /// # Example
    /// ```
    /// extern crate ddnnf_lib;
    /// use ddnnf_lib::Ddnnf;
    /// use ddnnf_lib::parser::*;
    /// use rug::Integer;
    /// use std::fs;
    ///
    /// // create a ddnnf
    /// // and run the queries
    /// let mut ddnnf: Ddnnf = build_ddnnf("./tests/data/small_ex_c2d.nnf", None);
    /// ddnnf.card_of_each_feature("./tests/data/smt_out.csv")
    ///      .unwrap_or_default();
    /// let _rm = fs::remove_file("./tests/data/smt_out.csv");
    ///
    /// ```
    pub fn card_of_each_feature(
        &mut self,
        file_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        if self.max_worker == 1 {
            self.card_of_each_feature_single(file_path)
        } else {
            self.card_of_each_feature_multi(file_path)
        }
    }

    /// Computes the cardinality of features for all features in a model
    /// in a single threaded environment
    fn card_of_each_feature_single(
        &mut self,
        file_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        // start the csv writer with the file_path
        let mut wtr = csv::Writer::from_path(file_path)?;

        for work in 1_i32..self.number_of_variables as i32 + 1 {
            let cardinality = self.card_of_feature_with_marker(work);
            wtr.write_record(vec![
                work.to_string(),
                cardinality.to_string(),
                format!("{:.20}", Float::with_val(200, cardinality) / self.rc()
            )])?;
        }

        Ok(())
    }

    /// Computes the cardinality of features for all features in a model
    /// in a multi threaded environment.
    /// Here we have to take into account:
    ///     1) using channels for communication
    ///     2) cloning the ddnnf
    ///     3) sorting our results
    fn card_of_each_feature_multi(
        &mut self,
        file_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        let mut queue: WorkQueue<i32> =
            WorkQueue::with_capacity(self.number_of_variables as usize);

        // Create a MPSC (Multiple Producer, Single Consumer) channel. Every worker
        // is a producer, the main thread is a consumer; the producers put their
        // work into the channel when it's done.
        let (results_tx, results_rx) = mpsc::channel();

        let mut threads = Vec::new();

        for e in 1..&self.number_of_variables + 1 {
            queue.push_work(e as i32); // x..y are all numbers between x and y including x, excluding y
        }

        for _ in 0..self.max_worker {
            let mut t_queue = queue.clone();
            let t_results_tx = results_tx.clone();
            let mut ddnnf: Ddnnf = self.clone();

            // thread::spawn takes a closure (an anonymous function that "closes"
            // over its environment). The move keyword means it takes ownership of
            // those variables, meaning they can't be used again in the main thread.
            let handle = thread::spawn(move || {
                // Loop while there's expected to be work, looking for work.
                // If work is available, do that work.
                while let Some(work) = t_queue.pull_work() {
                    let result = ddnnf.card_of_feature_with_marker(work);

                    // Send the work and the result of that work.
                    //
                    // Sending could fail. If so, there's no use in
                    // doing any more work, so abort.
                    match t_results_tx.send((work, result)) {
                        Ok(_) => (),
                        Err(_) => {
                            break;
                        }
                    }
                }

                // Signal to the operating system that now is a good time
                // to give another thread a chance to run.
                std::thread::yield_now();
            });

            // Add the handle for the newly spawned thread to the list of handles
            threads.push(handle);
        }

        let mut results: Vec<(i32, String, String)> =
            Vec::with_capacity(self.number_of_variables as usize);

        // Get completed work from the channel while there's work to be done.
        for _ in 0..self.number_of_variables {
            match results_rx.recv() {
                // If the control thread successfully receives, a job was completed.
                Ok((feature, cardinality)) => {
                    // Caching the results
                    results.push((
                        feature,
                        cardinality.to_string(),
                        format!(
                            "{:.20}",
                            Float::with_val(200, cardinality) / self.rc()
                        ),
                    ));
                }
                Err(_) => {
                    panic!("All workers died unexpectedly.");
                }
            }
        }

        // Just make sure that all the other threads are done.
        for handle in threads {
            handle.join().unwrap();
        }

        results.sort_unstable_by_key(|k| k.0);
        // start the csv writer with the file_path
        let mut wtr = csv::Writer::from_path(file_path)?;

        for element in results {
            wtr.write_record(vec![
                element.0.to_string(),
                element.1,
                element.2,
            ])?;
        }

        // Flush everything into the file that is still in a buffer
        // Now we finished writing the csv file
        wtr.flush()?;

        // If everything worked as expected, then we can return Ok(()) and we are happy :D
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::fs::{self, File};

    use file_diff::diff_files;

    use crate::parser::build_ddnnf;

    use super::*;

    #[test]
    fn card_multi_queries() {
        let mut ddnnf: Ddnnf = build_ddnnf("./tests/data/VP9_d4.nnf", Some(42));
        ddnnf.max_worker = 1;
        ddnnf.card_of_each_feature("./tests/data/fcs.csv").unwrap();

        ddnnf.max_worker = 4;
        ddnnf.card_of_each_feature("./tests/data/fcm.csv").unwrap();

        let mut is_single = File::open("./tests/data/fcs.csv").unwrap();
        let mut is_multi = File::open("./tests/data/fcm.csv").unwrap();
        let mut should_be = File::open("./tests/data/VP9_sb_fs.csv").unwrap();

        // diff_files is true if the files are identical
        assert!(diff_files(&mut is_single, &mut is_multi), "card of features results of single und multi variant have differences");
        is_single = File::open("./tests/data/fcs.csv").unwrap();
        assert!(diff_files(&mut is_single, &mut should_be), "card of features results differ from the expected results");

        fs::remove_file("./tests/data/fcs.csv").unwrap();
        fs::remove_file("./tests/data/fcm.csv").unwrap();
    }

    #[test]
    fn test_equality_single_and_multi() {
        let mut ddnnf: Ddnnf = build_ddnnf("./tests/data/VP9_d4.nnf", Some(42));
        ddnnf.max_worker = 1;

        ddnnf.card_of_each_feature_single("./tests/data/fcs1.csv").unwrap();
        ddnnf.card_of_each_feature_single("./tests/data/fcm1.csv").unwrap();

        ddnnf.max_worker = 4;
        ddnnf.card_of_each_feature_multi("./tests/data/fcm4.csv").unwrap();

        let mut is_single = File::open("./tests/data/fcs1.csv").unwrap();
        let mut is_multi = File::open("./tests/data/fcm1.csv").unwrap();
        let mut is_multi4 = File::open("./tests/data/fcm4.csv").unwrap();
        let mut should_be = File::open("./tests/data/VP9_sb_fs.csv").unwrap();
    
        // diff_files is true if the files are identical
        assert!(diff_files(&mut is_single, &mut is_multi), "card of features results of single und multi variant have differences");
        is_single = File::open("./tests/data/fcs1.csv").unwrap();
        is_multi = File::open("./tests/data/fcm1.csv").unwrap();
        assert!(diff_files(&mut is_multi, &mut is_multi4), "card of features for multiple threads differs when using multiple threads");
        assert!(diff_files(&mut is_single, &mut should_be), "card of features results differ from the expected results");

        fs::remove_file("./tests/data/fcs1.csv").unwrap();
        fs::remove_file("./tests/data/fcm1.csv").unwrap();
        fs::remove_file("./tests/data/fcm4.csv").unwrap();
    }
}