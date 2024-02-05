# d-ddnnf-reasoner

This is a built version of ddnnife and dhone.
The project is licensed under the LGPL-3.0 and the source can be found at https://github.com/SoftVarE-Group/d-dnnf-reasoner.

## Dependencies

The Mt-KaHyPar dependency is bundled with this build, other dependencies have to be installed:

- TBB (https://github.com/oneapi-src/oneTBB)
- hwloc (https://open-mpi.org/projects/hwloc)

## Usage

The binaries `ddnnife` and `dhone` are inside `bin`.
The Mt-KaHyPar library has to be available for `ddnnife` to run.
This can either be accomplished by moving the `lib` directories contents to the global library path (such as `/usr/lib`)
or by setting the `LD_LIBRARY_PATH` environment variable to include the `lib` directory.
Then, the linker will be able to find `libmtkahypar.so` required by `ddnnife`.

To show the help message, use:

```
ddnnife --help
```

For full usage instructions, see the README at https://github.com/SoftVarE-Group/d-dnnf-reasoner
