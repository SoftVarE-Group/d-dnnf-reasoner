# Bindings

`ddnnife` bindings are provided via [`uniffi`][uniffi].
Each language binding is located in its own subdirectory and should stay separated from the Rust codebase.

For building instructions, see the specific binding's README.

## `ddnnife_bindgen`

This is a version of `uniffi-bindgen` specific to this project.
Its task is to actually generate the bindings to the Rust library.
The binary of this crate is called `uniffi-bindgen` to work with build tools.

[uniffi]: https://mozilla.github.io/uniffi-rs
