# d-dknnife a d-dnnf-reasoner

## requirements for building

We use rug for the computations. Make sure to install everything mentioned [here](crates.io/crates/rug) to use rug and our software.

## build the binaries:

### the preprocessor (dhone)
```properties
sh:~$ cargo build --release --bin dhone
```

### d-dknnife
```properties
sh:~$ cargo build --release --bin d-dknnife
```

## usage:
Simply execute the binaries with the -h, --help flag or no parameter at all
```properties
sh:~$ ./dhone -h
sh:~$ ./d-dknnife -h
```

## run tests:
```properties
sh:~$ cargo test
```

Test coverage can be determined with tarpaulin. tarpaulin is not included in rustup.

usage:
1) install tarpaulin
2) run tests and save results as .html
3) open the report with a browser (here we use google-chrome)
```properties
sh:~$ cargo install cargo-tarpaulin
sh:~$ cargo tarpaulin -o Html
sh:~$ google-chrome tarpaulin-report.html
```


## create documentation and open it in browser
Just public functions, enums, structs are documented and also just those of the parser module

```properties
sh:~$ cargo doc --open
```

## cpu native flag
In the ./cargo/config.toml file under "[build]" are rustflags which optimise the binary for the cpu of the system that compiles the code. If we want to share a binary between different pcs then we would recommend to remove/comment that line.
