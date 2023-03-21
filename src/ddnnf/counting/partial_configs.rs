use std::{error::Error, sync::mpsc, thread, io::{BufWriter, Write}, fs::File};

use workctl::{WorkQueue};

use crate::{Ddnnf, parser};

impl Ddnnf{
    #[inline]
    /// Computes the cardinality of partial configurations for all queries in path_in.
    /// The results are saved in the path_out. The .csv ending always gets added to the user input.
    /// The function uses the marking approach for configurations with <= 10 features. For larger
    /// queries we use the "standard" approach without marking.
    /// Here the number of threads influence the speed by using a shared work queue.
    /// The result can be in a different order for iff multiple threads are used.
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
    /// let mut ddnnf: Ddnnf = build_ddnnf("./tests/data/small_test.dimacs.nnf", None);
    /// ddnnf.card_multi_queries(
    ///     "./tests/data/small_test.config",
    ///     "./tests/data/smt_out.csv",)
    ///     .unwrap_or_default();
    /// let _rm = fs::remove_file("./tests/data/smt_out.csv");
    ///
    /// ```
    pub fn card_multi_queries(
        &mut self,
        path_in: &str,
        path_out: &str,
    ) -> Result<(), Box<dyn Error>> {
        if self.max_worker == 1 {
            self.card_multi_queries_single(path_in, path_out)
        } else {
            self.card_multi_queries_multi(path_in, path_out)
        }
    }

    /// Computes the cardinality of partial configurations for all queries in path_in
    /// in a multi threaded environment
    /// Here we have to take into account:
    ///     1) using channels for communication
    ///     2) cloning the ddnnf
    ///     3) sorting our results
    fn card_multi_queries_single(
        &mut self,
        path_in: &str,
        path_out: &str,
    ) -> Result<(), Box<dyn Error>> {
        // start the file writer with the file_path
        let f = File::create(path_out).expect("Unable to create file");
        let mut wtr = BufWriter::new(f);

        let work_queue: Vec<(usize, Vec<i32>)> = parser::parse_queries_file(path_in);

        for work in &work_queue {
            let cardinality = self.execute_query(&work.1);
            let mut features_str =
            work.1.iter().fold(String::new(), |acc, &num| {
                acc + &num.to_string() + " "
            });
            features_str.pop();
            let data = &format!("{},{}\n", features_str, cardinality);
            wtr.write_all(data.as_bytes())?;
        }

        Ok(())
    }

    /// Computes the cardinality of partial configurations for all queries in path_in
    /// in a multi threaded environment
    fn card_multi_queries_multi(
        &mut self,
        path_in: &str,
        path_out: &str,
    ) -> Result<(), Box<dyn Error>> {
        let work: Vec<(usize, Vec<i32>)> = parser::parse_queries_file(path_in);
        let mut queue = WorkQueue::with_capacity(work.len());

        for e in work.clone() {
            queue.push_work(e);
        }

        let (results_tx, results_rx) = mpsc::channel();

        let mut threads = Vec::new();

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
                    // Send the work and the result of that work.
                    //
                    // Sending could fail. If so, there's no use in
                    // doing any more work, so abort.
                    let work_c = work.clone();
                    match t_results_tx.send((work_c.0, work_c.1, ddnnf.execute_query(&work.1))) {
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

        // start the csv writer with the file_path
        let mut wtr = csv::Writer::from_path(path_out)?;
        let mut results = Vec::new();

        // Get completed work from the channel while there's work to be done.
        for _ in 0..work.len() {
            match results_rx.recv() {
                // If the control thread successfully receives, a job was completed.
                Ok(result) => {
                    results.push(result);
                }
                // If the control thread is the one left standing, that's pretty
                // problematic.
                Err(_) => {
                    panic!("All workers died unexpectedly.");
                }
            }
        }

        results.sort_unstable();

        for result in results {
            let mut features_str = result.1.iter().fold(String::new(), |acc, &num| {
                acc + &num.to_string() + " "
            });
            features_str.pop();
            wtr.write_record(vec![features_str, result.2.to_string()])?;
        }

        // Just make sure that all the other threads are done.
        for handle in threads {
            handle.join().unwrap();
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
    use std::fs;

    use file_diff::diff_files;

    use crate::parser::build_ddnnf;

    use super::*;

    #[test]
    fn card_multi_queries() {
        let mut ddnnf: Ddnnf = build_ddnnf("./tests/data/VP9_d4.nnf", Some(42));
        ddnnf.max_worker = 1;
        ddnnf.card_multi_queries("./tests/data/VP9.config", "./tests/data/pcs.csv").unwrap();

        ddnnf.max_worker = 4;
        ddnnf.card_multi_queries("./tests/data/VP9.config", "./tests/data/pcm.csv").unwrap();

        let mut is_single = File::open("./tests/data/pcs.csv").unwrap();
        let mut is_multi = File::open("./tests/data/pcm.csv").unwrap();
        let mut should_be = File::open("./tests/data/VP9_sb_pc.csv").unwrap();

        // diff_files is true if the files are identical
        assert!(diff_files(&mut is_single, &mut is_multi), "partial config results of single und multi variant have differences");
        is_single = File::open("./tests/data/pcs.csv").unwrap();
        assert!(diff_files(&mut is_single, &mut should_be), "partial config results differ from the expected results");

        fs::remove_file("./tests/data/pcs.csv").unwrap();
        fs::remove_file("./tests/data/pcm.csv").unwrap();
    }

    #[test]
    fn test_equality_single_and_multi() {
        let mut ddnnf: Ddnnf = build_ddnnf("./tests/data/VP9_d4.nnf", Some(42));
        ddnnf.max_worker = 1;

        ddnnf.card_multi_queries_single("./tests/data/VP9.config", "./tests/data/pcs1.csv").unwrap();
        ddnnf.card_multi_queries_multi("./tests/data/VP9.config", "./tests/data/pcm1.csv").unwrap();

        ddnnf.max_worker = 4;
        ddnnf.card_multi_queries_multi("./tests/data/VP9.config", "./tests/data/pcm4.csv").unwrap();

        let mut is_single = File::open("./tests/data/pcs1.csv").unwrap();
        let mut is_multi = File::open("./tests/data/pcm1.csv").unwrap();
        let mut is_multi4 = File::open("./tests/data/pcm4.csv").unwrap();
        let mut should_be = File::open("./tests/data/VP9_sb_pc.csv").unwrap();
    
        // diff_files is true if the files are identical
        assert!(diff_files(&mut is_single, &mut is_multi), "partial config results of single und multi variant have differences");
        is_single = File::open("./tests/data/pcs1.csv").unwrap();
        is_multi = File::open("./tests/data/pcm1.csv").unwrap();
        assert!(diff_files(&mut is_multi, &mut is_multi4), "partial config for multiple threads differs when using multiple threads");
        assert!(diff_files(&mut is_single, &mut should_be), "partial config results differ from the expected results");

        fs::remove_file("./tests/data/pcs1.csv").unwrap();
        fs::remove_file("./tests/data/pcm1.csv").unwrap();
        fs::remove_file("./tests/data/pcm4.csv").unwrap();
    }
}