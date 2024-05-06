# d-ddnnf-reasoner

This is a built version of ddnnife and dhone.
The project is licensed under the LGPL-3.0 and the source can be found at https://github.com/SoftVarE-Group/d-dnnf-reasoner.

## Dependencies

All dependencies are bundled with this build inside the `lib` directory.
To use the binaries, they have to be visible to the linker.
This can either be accomplished by moving the `lib` directories contents to the global library path (such as `/usr/lib`)
or by setting the `DYLD_LIBRARY_PATH` environment variable to include the `lib` directory.

## Gatekeeper

The execution of downloaded binaries might be blocked by Gatekeeper.
To resolve this, the attribute `com.apple.quarantine` has to be removed from the binaries inside `bin` and the dependencies inside `lib`:

```
xattr -d com.apple.quarantine bin/*
xattr -d com.apple.quarantine lib/*
```

## Usage

The binaries `ddnnife` and `dhone` are inside `bin`.
To show the help message, use:

```
ddnnife --help
```

For full usage instructions, see the README at https://github.com/SoftVarE-Group/d-dnnf-reasoner
