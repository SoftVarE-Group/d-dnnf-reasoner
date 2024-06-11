{
  description = "Packages and development environments for ddnnife";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils/v1.0.0";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    d4 = {
      url = "github:SoftVarE-Group/d4v2/mt-kahypar";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      fenix,
      d4,
      ...
    }:
    let
      lib = nixpkgs.lib;
      systems = lib.systems.doubles.unix;
    in
    flake-utils.lib.eachSystem systems (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        toolchain = {
          default = fenix.packages.${system}.stable.defaultToolchain;
          static =
            with fenix.packages.${system};
            combine [
              stable.defaultToolchain
              targets.x86_64-unknown-linux-musl.stable.rust-std
            ];
        };

        rust = {
          default = pkgs.makeRustPlatform {
            cargo = toolchain.default;
            rustc = toolchain.default;
          };
          static = pkgs.pkgsStatic.makeRustPlatform {
            cargo = toolchain.static;
            rustc = toolchain.static;
          };
        };

        crate = {
          name = "ddnnife";

          meta = with lib; {
            description = "A d-DNNF reasoner";
            homepage = "https://github.com/SoftVarE-Group/d-dnnf-reasoner";
            license = licenses.lgpl3Plus;
            platforms = systems;
          };

          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;

          nativeBuildInputs = [ pkgs.gnum4 ];
        };

        crate-d4 = crate // {
          buildFeatures = [ "d4" ];

          buildInputs = [
            pkgs.boost.dev
            pkgs.gmp.dev
            d4.packages.${system}.mt-kahypar
          ] ++ lib.optionals pkgs.stdenv.isDarwin [ pkgs.libiconv ];

          nativeBuildInputs = [
            pkgs.gnum4
            pkgs.pkg-config
          ];

          # FIXME: Tests are currently unable to run on x86_64-darwin.
          doCheck = system != "x86_64-darwin";
        };

        ddnnife = rust.default.buildRustPackage crate;
        ddnnife-static = rust.static.buildRustPackage crate;
        ddnnife-d4 = rust.default.buildRustPackage crate-d4;

        runTest =
          name: crate: testCommand:
          rust.default.buildRustPackage (
            crate
            // {
              inherit name;
              installPhase = "touch $out";
              dontBuild = true;
              doCheck = true;
              checkType = "debug";
              checkPhase = ''
                set -x
                ${testCommand}
                set +x
              '';
            }
          );
      in
      {
        formatter = pkgs.nixfmt-rfc-style;

        checks = {
          format = runTest "format" crate "cargo fmt --check";
          lint = runTest "lint" crate-d4 "cargo clippy --all-features";
        };

        packages = {
          default = self.packages.${system}.ddnnife-d4;

          inherit ddnnife;
          inherit ddnnife-static;
          inherit ddnnife-d4;

          container = pkgs.dockerTools.buildLayeredImage {
            name = "ddnnife";
            contents = [ self.packages.${system}.ddnnife-d4 ];
            config = {
              Entrypoint = [ "/bin/ddnnife" ];
              Labels = {
                "org.opencontainers.image.source" = "https://github.com/SoftVarE-Group/d-dnnf-reasoner";
                "org.opencontainers.image.description" = "A d-DNNF reasoner";
                "org.opencontainers.image.licenses" = "LGPL-3.0-or-later";
              };
            };
          };

          documentation = pkgs.stdenv.mkDerivation {
            name = "documentation";
            src = ./doc;
            installPhase = ''
              mkdir $out
              cp ${system}.md $out/README.md
            '';
          };

          documentation-d4 = pkgs.stdenv.mkDerivation {
            name = "documentation";
            src = ./doc;
            installPhase = ''
              mkdir $out
              cp ${system}-d4.md $out/README.md
            '';
          };

          all = pkgs.buildEnv {
            name = "ddnnife";
            paths = [
              self.packages.${system}.ddnnife
              self.packages.${system}.documentation
            ];
          };

          all-d4 = pkgs.buildEnv {
            name = "ddnnife";
            paths = [
              self.packages.${system}.ddnnife-d4
              self.packages.${system}.documentation-d4
              d4.packages.${system}.dependencies
            ];
          };
        };
      }
    );
}
