# ddnnife a d-dnnf-reasoner

ddnnife takes a smooth d-DNNF as input following the [standard format specified by c2d](http://reasoning.cs.ucla.edu/c2d/). After parsing and storing the d-DNNF, ddnnife can be used to compute the cardinality of single features, all features, or partial configurations.

## Requirements for building

First, if not done already you have to [install rust](https://www.rust-lang.org/tools/install). The recommended way is the following using curl and rustup:
```properties
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
After that, we recommend to enter '1' to proceed with installation (without customizations).

Additionally, we use rug for the computations. Make sure to install everything mentioned on rugs [crates.io page](https://crates.io/crates/rug) to use rug and our software. There is says:

*Rug [...] depends on the [GMP](https://gmplib.org/), [MPFR](https://www.mpfr.org/) and [MPC](https://www.multiprecision.org/mpc/) libraries through the low-level FFI bindings in the [gmp-mpfr-sys crate](https://crates.io/crates/gmp-mpfr-sys), which needs some setup to build; the [gmp-mpfr-sys documentation](https://docs.rs/gmp-mpfr-sys/1.4.7/gmp_mpfr_sys/index.html) has some details on usage under [different os][...].*

To build on GNU/Linux, make sure you have ```diffutils, ggc, m4 and make``` installed on your system. For example on Ubuntu:
```properties
sudo apt-get update && apt-get install diffutils gcc m4 make
```


## Build the binaries:
Make sure that the current working directory is the one inlcuding the Cargo.toml file.

### Both
```properties
cargo build --release
```

### The preprocessor (dhone)
```properties
cargo build --release --bin dhone
```

### ddnnife
```properties
cargo build --release --bin ddnnife
```

## Usage:
Simply execute the binaries with the -h, --help flag or no parameter at all.

Note: In this and the following code examples we added the ```./target/release/``` directories as prefix because thats where the binaries are placed when building they are built according to the previous chapter and the working directory is not switched.

Help for dhone
```properties
./target/release/dhone -h
```

and ddnnife
```properties
./target/release/ddnnife -h
```

### Examples
Preprocesses the ddnnf: berkeleydb_dsharp.dimacs.nnf which may need preprocessing because it was created with dsharp (in this case it is actually necessary).
```properties
./target/release/dhone example_input/berkeleydb_dsharp.dimacs.nnf -s example_input/berkeleydb_prepo.dimacs.nnf
```
We only compute the cardinality of a feature model for automotive01.
```properties
./target/release/ddnnife example_input/automotive01.dimacs.nnf
```

Compute the cardinality of features for busybox-1.18.0.dimacs.nnf with 2 threads and save the result as busybox_feat.csv in the current working directory.
```properties
./target/release/ddnnife example_input/busybox-1.18.0.dimacs.nnf -c -s busybox_feat -n 2
```

Compute the cardinality of features for automotive01 when compiled with d4. Here we need the -o Option that allows us to specify the total number of features. That information is needed but not contained in ddnnfs using the d4 standard. Furthermore, the parsing takes more time because we have to smooth the ddnnf.
```properties
./target/release/ddnnife example_input/auto1_d4.nnf -o 2513 -c
```

Compute the cardinality of partial configurations for X264.dimacs.nnf with 4 threads (default) and save the result as out.txt in the current working directory (default).
```properties
./target/release/ddnnife example_input/X264.dimacs.nnf -q example_input/X264.config
```
## Create documentation and open it in browser

```properties
cargo doc --open
```

Besides the generated documentation there are further comments in the code itself.

## Run tests:
```properties
cargo test --release
```

Test coverage can be determined with tarpaulin. tarpaulin is not included in rustup.
Make sure to execute the following commands in the folder that also contains the Cargo.toml file.

usage:
1) install tarpaulin
2) run tests and save results as .html
3) open the report with a browser (here we use google-chrome)
```properties
cargo install cargo-tarpaulin
cargo tarpaulin -o Html
google-chrome tarpaulin-report.html
```

## cpu native flag
In the ./cargo/config.toml file under "[build]" are rustflags which optimise the binary for the cpu of the system that compiles the code. If we want to share a binary between different pcs then we would recommend to remove/comment that line.
