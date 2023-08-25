use std::os::unix::fs::PermissionsExt;
use std::{
    fs,
    process::{self, Command},
};
use tempfile::NamedTempFile;

#[cfg(windows)]
const D4V2: &[u8] = include_bytes!("..\\bin\\d4v2.bin"); // relative from source file
#[cfg(unix)]
const D4V2: &[u8] = include_bytes!("../bin/d4v2.bin");

/// Using the d4v2 CNF to dDNNF compiler from cril,
/// we take a CNF from path_in and write the dDNNF to path_out
pub fn compile_cnf(path_in: &str, path_out: &str) {
    // If the byte array is empty, we did not include d4v2. Consequently, we can't compile and have to exit.
    if D4V2.is_empty() {
        // Bold, Red, Foreground Color (see https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797)
        eprintln!("\x1b[1;38;5;196m\nERROR: d4v2 is not part of that binary! \
            Hence, CNF files can not be handled by this binary file. \
            Compile again with 'EXCLUDE_D4V2=FALSE cargo build --release' to use d4v2.\nAborting...");
        process::exit(1);
    }

    // Write d4 to a temporary file.
    // Will be removed by system once out of scope.
    let d4_file = NamedTempFile::new().expect("Failed to create temporary file to run d4.");
    fs::write(&d4_file, D4V2).expect("Failed to write d4 to file.");

    // Make the file executable.
    let mut permissions = d4_file
        .as_file()
        .metadata()
        .expect("Failed to get d4 file metadata.")
        .permissions();

    #[cfg(unix)]
    permissions.set_mode(0o755);

    #[cfg(windows)]
    permissions.set_readonly(false);

    d4_file
        .as_file()
        .set_permissions(permissions)
        .expect("Failed to make d4 file executable.");

    // Execute the command to compile a dDNNF from a CNF file.
    Command::new(&d4_file.into_temp_path())
        .args([
            "-i",
            path_in,
            "-m",
            "ddnnf-compiler",
            "--dump-ddnnf",
            path_out,
        ])
        .output()
        .unwrap();
}
