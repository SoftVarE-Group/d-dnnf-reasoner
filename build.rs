use std::{process::Command, env};

fn main() {
    // Set up a local copy of the repository. We choose the forked version of d4v2 due to a missing library in the origin repo.
    Command::new("mkdir").arg("d4v2").output().unwrap();
    Command::new("git").args(["clone", "https://github.com/SoftVarE-Group/d4v2.git", "d4v2/."]).output().unwrap();
    
    // build the d4v2 binary
    env::set_current_dir(&env::current_dir().unwrap().canonicalize().unwrap().join("d4v2").as_path()).unwrap();
    match Command::new("./build.sh").output() {
        Ok(_) => (),
        Err(err) => eprintln!("An error occured while trying to build the d4v2 binary: {}", err),
    }

    // debug build
    #[cfg(debug_assertions)]
    Command::new("cp").args(["./build/d4", "../target/debug/d4v2"]).output().unwrap();

    // release build
    #[cfg(not(debug_assertions))]
    Command::new("cp").args(["./build/d4", "../target/release/d4v2"]).output().unwrap();
}