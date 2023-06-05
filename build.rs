use std::{process::{Command, exit}, env, fs::File, io::Write, path::Path};

fn main() {
    // Include or exclude d4v2 depending on the enviroment variable
    println!("cargo:rerun-if-changed=EXCLUDE_D4V2");

    #[cfg(windows)] let mut out = File::create("src\\bin\\d4v2.bin").unwrap();
    #[cfg(unix)] let mut out = File::create("src/bin/d4v2.bin").unwrap();

    match std::env::var("EXCLUDE_D4V2") {
        Ok(val) => {
            if val == "TRUE" {
                println!("Not including d4v2");
                // create an empty binary file so the constant referencing it has a value
                out.write_all(b"").unwrap();
                exit(0);
            }
        },
        Err(_) => (), // not set -> we expect that the user wants to use d4
    }
    
    // Force rerunning this build script if src/parser/d4v2.bin gets deleted or changed in any way
    // to make sure it works properly
    #[cfg(windows)] println!("cargo:rerun-if-changed=src\\bin\\d4v2.bin");
    #[cfg(unix)] println!("cargo:rerun-if-changed=src/bin/d4v2.bin");

    if !Path::new("d4v2/build/d4").exists() {
        println!("cloning and building d4...");

        // Set up a local copy of the repository. We choose the forked version of d4v2 due to a missing library in the origin repo.
        Command::new("rm").args(["-rf", "d4v2"]).output().unwrap();
        Command::new("mkdir").arg("d4v2").output().expect("wasn't able to create the 'd4v2' folder");
        Command::new("git").args(["clone", "https://github.com/SoftVarE-Group/d4v2.git", "d4v2/."]).output().expect("could not clone the d4v2 repository");
    
        // build the d4v2 binary
        let current_dir = &env::current_dir().unwrap();
        env::set_current_dir(current_dir.canonicalize().unwrap().join("d4v2").as_path()).unwrap();
        match Command::new("./build.sh").output() {
            Ok(_) => (),
            Err(err) => eprintln!("An error occured while trying to build the d4v2 binary: {err}"),
        }
        env::set_current_dir(current_dir).unwrap();
    }

    // Include the binary data of d4v2 in the source files.
    // This allows us to staticlly bind it to the ddnnife binary.
    out.write_all(&std::fs::read("d4v2/build/d4").unwrap()).unwrap();
}