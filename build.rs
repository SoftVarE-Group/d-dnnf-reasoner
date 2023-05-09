use std::{process::Command, env, fs::File, io::Write};

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

    // Include the binary data of d4v2 in the source files.
    // This allows us to staticlly bind it to the ddnnife binary.
    const EXTERNAL_BINARY_DATA: &[u8] = include_bytes!("d4v2/build/d4");
    let mut out = File::create("../src/parser/d4v2.bin").unwrap();
    out.write_all(EXTERNAL_BINARY_DATA).unwrap();
}