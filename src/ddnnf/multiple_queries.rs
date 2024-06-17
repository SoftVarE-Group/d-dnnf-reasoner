use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    sync::mpsc,
    thread,
};

use workctl::WorkQueue;

use crate::{parser, Ddnnf};

impl Ddnnf {
    #[inline]
    /// Computes the given operation for all queries in path_in.
    /// The results are saved in the path_out. The .csv ending always gets added to the user input.
    /// Here, the number of threads influence the speed by using a shared work queue.
    pub fn operate_on_queries<T: ToString + Ord + Send + 'static>(
        &mut self,
        operation: fn(&mut Ddnnf, &[i32]) -> T,
        path_in: &str,
        path_out: &str,
    ) -> Result<(), Box<dyn Error>> {
        if self.max_worker == 1 {
            self.queries_single_thread(operation, path_in, path_out)
        } else {
            self.queries_multi_thread(operation, path_in, path_out)
        }
    }

    /// Computes the operation for all queries in path_in
    /// in a multi threaded environment
    /// Here we have to take into account:
    ///     1) using channels for communication
    ///     2) cloning the ddnnf
    ///     3) sorting our results
    fn queries_single_thread<T: ToString>(
        &mut self,
        operation: fn(&mut Ddnnf, &[i32]) -> T,
        path_in: &str,
        path_out: &str,
    ) -> Result<(), Box<dyn Error>> {
        // start the file writer with the file_path
        let f = File::create(path_out).expect("Unable to create file");
        let mut wtr = BufWriter::new(f);

        let work_queue: Vec<(usize, Vec<i32>)> = parser::parse_queries_file(path_in);

        for (_, work) in &work_queue {
            let cardinality = operation(self, work);
            let mut features_str = work
                .iter()
                .fold(String::new(), |acc, &num| acc + &num.to_string() + " ");
            features_str.pop();
            let data = &format!("{},{}\n", features_str, cardinality.to_string());
            wtr.write_all(data.as_bytes())?;
        }

        Ok(())
    }

    /// Computes the cardinality of partial configurations for all queries in path_in
    /// in a multi threaded environment
    fn queries_multi_thread<T: ToString + Ord + Send + 'static>(
        &mut self,
        operation: fn(&mut Ddnnf, &[i32]) -> T,
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
                while let Some((index, work)) = t_queue.pull_work() {
                    // Send the work and the result of that work.
                    //
                    // Sending could fail. If so, there's no use in
                    // doing any more work, so abort.
                    let work_c = work.clone();
                    match t_results_tx.send((index, work_c, operation(&mut ddnnf, &work))) {
                        Ok(_) => (),
                        Err(_) => {
                            break;
                        }
                    }
                }

                // Signal to the operating system that now is a good time
                // to give another thread a chance to run.
                thread::yield_now();
            });

            // Add the handle for the newly spawned thread to the list of handles
            threads.push(handle);
        }

        // start the file writer with the file_path
        let f = File::create(path_out).expect("Unable to create file");
        let mut wtr = BufWriter::new(f);
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

        for (_, query, result) in results {
            let mut features_str = query
                .iter()
                .fold(String::new(), |acc, &num| acc + &num.to_string() + " ");
            features_str.pop();
            let data = &format!("{},{}\n", features_str, result.to_string());
            wtr.write_all(data.as_bytes())?;
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
    use std::{
        fs,
        io::{BufRead, BufReader},
    };

    use file_diff::diff_files;
    use itertools::Itertools;
    use num::BigInt;

    use crate::parser::build_ddnnf;

    use super::*;

    #[test]
    fn card_multi_queries() {
        let mut ddnnf: Ddnnf = build_ddnnf("./tests/data/VP9_d4.nnf", Some(42));
        ddnnf.max_worker = 1;
        ddnnf
            .queries_multi_thread(
                Ddnnf::execute_query,
                "./tests/data/VP9.config",
                "./tests/data/pcs.csv",
            )
            .unwrap();

        ddnnf.max_worker = 4;
        ddnnf
            .queries_multi_thread(
                Ddnnf::execute_query,
                "./tests/data/VP9.config",
                "./tests/data/pcm.csv",
            )
            .unwrap();

        let mut is_single = File::open("./tests/data/pcs.csv").unwrap();
        let mut is_multi = File::open("./tests/data/pcm.csv").unwrap();
        let mut should_be = File::open("./tests/data/VP9_sb_pc.csv").unwrap();

        // diff_files is true if the files are identical
        assert!(
            diff_files(&mut is_single, &mut is_multi),
            "partial config results of single und multi variant have differences"
        );
        is_single = File::open("./tests/data/pcs.csv").unwrap();
        assert!(
            diff_files(&mut is_single, &mut should_be),
            "partial config results differ from the expected results"
        );

        fs::remove_file("./tests/data/pcs.csv").unwrap();
        fs::remove_file("./tests/data/pcm.csv").unwrap();
    }

    #[test]
    fn sat_multi_queries() {
        let mut ddnnf: Ddnnf = build_ddnnf("./tests/data/VP9_d4.nnf", Some(42));
        ddnnf
            .operate_on_queries(
                Ddnnf::sat,
                "./tests/data/VP9.config",
                "./tests/data/sat.csv",
            )
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

    #[test]
    fn test_equality_single_and_multi() {
        let mut ddnnf: Ddnnf = build_ddnnf("./tests/data/VP9_d4.nnf", Some(42));
        ddnnf.max_worker = 1;

        ddnnf
            .queries_single_thread(
                Ddnnf::execute_query,
                "./tests/data/VP9.config",
                "./tests/data/pcs1.csv",
            )
            .unwrap();
        ddnnf
            .queries_multi_thread(
                Ddnnf::execute_query,
                "./tests/data/VP9.config",
                "./tests/data/pcm1.csv",
            )
            .unwrap();

        ddnnf.max_worker = 4;
        ddnnf
            .queries_multi_thread(
                Ddnnf::execute_query,
                "./tests/data/VP9.config",
                "./tests/data/pcm4.csv",
            )
            .unwrap();

        let mut is_single = File::open("./tests/data/pcs1.csv").unwrap();
        let mut is_multi = File::open("./tests/data/pcm1.csv").unwrap();
        let mut is_multi4 = File::open("./tests/data/pcm4.csv").unwrap();
        let mut should_be = File::open("./tests/data/VP9_sb_pc.csv").unwrap();

        // diff_files is true if the files are identical
        assert!(
            diff_files(&mut is_single, &mut is_multi),
            "partial config results of single und multi variant have differences"
        );
        is_single = File::open("./tests/data/pcs1.csv").unwrap();
        is_multi = File::open("./tests/data/pcm1.csv").unwrap();
        assert!(
            diff_files(&mut is_multi, &mut is_multi4),
            "partial config for multiple threads differs when using multiple threads"
        );
        assert!(
            diff_files(&mut is_single, &mut should_be),
            "partial config results differ from the expected results"
        );

        fs::remove_file("./tests/data/pcs1.csv").unwrap();
        fs::remove_file("./tests/data/pcm1.csv").unwrap();
        fs::remove_file("./tests/data/pcm4.csv").unwrap();
    }
}
