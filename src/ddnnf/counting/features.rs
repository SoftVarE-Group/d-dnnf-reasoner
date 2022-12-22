use std::{sync::mpsc, error::Error, thread};

use rug::Float;
use workctl::WorkQueue;

use super::super::Ddnnf;

impl Ddnnf {
    #[inline]
    /// Computes the cardinality of features for all features in a modell.
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
    /// let mut ddnnf: Ddnnf = build_ddnnf_tree_with_extras("./tests/data/small_test.dimacs.nnf");
    /// ddnnf.card_of_each_feature_to_csv("./tests/data/smt_out.csv")
    ///      .unwrap_or_default();
    /// let _rm = fs::remove_file("./tests/data/smt_out.csv");
    ///
    /// ```
    pub fn card_of_each_feature_to_csv(
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
                    let result = ddnnf.card_of_feature_with_marker(work).1;

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