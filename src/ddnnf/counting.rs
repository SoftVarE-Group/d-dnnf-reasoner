// The Problem:
//  We have lots of computations that each require a huge data structure (our Ddnnf).
//  Each thread needs a Ddnnf and they are not allowed to share one.
//  Furthermore, we don't want to clone the initial ddnnf for each thread because
//  the cost for that operation is even bigger than one computation.
//
// The Solution:
//  Create a queue which can be safely shared between threads from which those
//  threads can pull work (the feature that should be computed now) each time they finish their current work.
//
// We assume that we have MAX_WORKER processor cores which will do work for us.
// You could use the num_cpus crate to find this for a particular machine.
pub mod features;

// Modules that provide the basic counting logic.
pub mod default_count;
pub mod marking;
