# dknife a d-dnnf-reasoner

## requirements for building

We use rug for the computations. Make sure to install everything mentioned [here](crates.io/crates/rug) to use rug and our software.

## build the binary:

```properties
sh:~$ cargo build --release
```

## usage:
Simply execute the binary with the -h or --help flag
```properties
sh:~$ ./d-dnnf_reasoner -h
```

## run tests:
```properties
sh:~$ cargo test
```
or  (can be a bit faster than the option above in execution time)
```properties
sh:~$ cargo test --release
```

## create documentation and open it in browser
Just public functions, enums, structs are documented and also just those of the parser module

```properties
sh:~$ cargo doc --open
```

## cpu native flag
In the ./cargo/config.toml file under "[build]" are rustflags which optimise the binary for the cpu of the system that compiles the code. If we want to share a binary between different pcs then we would recommend to remove/comment that line.
