use std::{error::Error, sync::mpsc, thread, io::{BufWriter, Write}, fs::File};

use workctl::WorkQueue;

use crate::{Ddnnf, parser};

impl Ddnnf{
    #[inline]
    /// Computes the cardinality of partial configurations for all queries in path_in.
    /// The results are saved in the path_out. The .txt ending always gets added to the user input.
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
    /// let mut ddnnf: Ddnnf = build_ddnnf_tree_with_extras("./tests/data/small_test.dimacs.nnf");
    /// ddnnf.card_multi_queries(
    ///     "./tests/data/small_test.config",
    ///     "./tests/data/smt_out.txt",)
    ///     .unwrap_or_default();
    /// let _rm = fs::remove_file("./tests/data/smt_out.txt");
    ///
    /// ```
    pub fn card_multi_queries(
        &mut self,
        path_in: &str,
        path_out: &str,
    ) -> Result<(), Box<dyn Error>> {
        let work: Vec<Vec<i32>> = parser::parse_queries_file(path_in);
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
                    match t_results_tx.send((work.clone(), ddnnf.execute_query(&work))) {
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

        // start the file writer with the file_path
        let f = File::create(path_out).expect("Unable to create file");
        let mut wtr = BufWriter::new(f);

        // Get completed work from the channel while there's work to be done.
        for _ in 0..work.len() {
            match results_rx.recv() {
                // If the control thread successfully receives, a job was completed.
                Ok((features, cardinality)) => {
                    let mut features_str =
                        features.iter().fold(String::new(), |acc, &num| {
                            acc + &num.to_string() + " "
                        });
                    features_str.pop();
                    let data = &format!("{},{}\n", features_str, cardinality);
                    wtr.write_all(data.as_bytes())?;
                }
                // If the control thread is the one left standing, that's pretty
                // problematic.
                Err(_) => {
                    panic!("All workers died unexpectedly.");
                }
            }
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