{ fenix }:
{
  # Rust target triple mappings for the target platforms.
  map = {
    aarch64-darwin.default = "aarch64-apple-darwin";
    aarch64-linux = {
      default = "aarch64-unknown-linux-gnu";
      static = "aarch64-unknown-linux-musl";
    };
    x86_64-darwin.default = "x86_64-apple-darwin";
    x86_64-linux = {
      default = "x86_64-unknown-linux-gnu";
      static = "x86_64-unknown-linux-musl";
    };
    x86_64-windows.default = "x86_64-pc-windows-gnu";
  };

  # Constructs a Rust toolchain for the build system with a (possibly different) target system.
  toolchain =
    system: target:
    fenix.packages.${system}.combine [
      fenix.packages.${system}.stable.defaultToolchain
      fenix.packages.${system}.targets.${target}.stable.rust-std
    ];
}
