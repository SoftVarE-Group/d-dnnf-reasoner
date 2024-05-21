{
  description = "Packages and development environments for ddnnife";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils/v1.0.0";
    crane = {
      url = "github:ipetkov/crane/v0.16.3";
      inputs.nixpkgs.follows = "nixpkgs";
    };
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

  outputs = { self, nixpkgs, flake-utils, crane, fenix, d4, ... }:
    let
      lib = nixpkgs.lib;
      systems = lib.systems.doubles.unix;
    in flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        toolchain = fenix.packages.${system}.stable.defaultToolchain;
        craneLib = (crane.mkLib pkgs).overrideToolchain toolchain;

        src = craneLib.path ./.;

        crateArgs = {
          inherit src;

          strictDeps = true;
          doCheck = false;

          buildInputs = lib.optionals pkgs.stdenv.isDarwin [ pkgs.libiconv ];

          nativeBuildInputs = [ pkgs.gnum4 ];

          meta = with lib; {
            description = "A d-DNNF reasoner";
            homepage = "https://github.com/SoftVarE-Group/d-dnnf-reasoner";
            license = licenses.lgpl3Plus;
            platforms = systems;
          };
        };

        crateArgs-d4 = crateArgs // {
          cargoExtraArgs = "--features d4";

          buildInputs =
            [ pkgs.boost.dev pkgs.gmp.dev d4.packages.${system}.mt-kahypar ]
            ++ lib.optionals pkgs.stdenv.isDarwin [ pkgs.libiconv ];

          nativeBuildInputs = [ pkgs.gnum4 pkgs.pkg-config ];

          # FIXME: libcxx symbols missing (https://github.com/NixOS/nixpkgs/issues/166205)
          # Should be fixed in 24.05.
          env = lib.optionalAttrs pkgs.stdenv.cc.isClang {
            NIX_LDFLAGS = "-l${pkgs.stdenv.cc.libcxx.cxxabi.libName}";
          };
        };

        cargoArtifacts = craneLib.buildDepsOnly crateArgs;
        cargoArtifacts-d4 = craneLib.buildDepsOnly crateArgs-d4;

        ddnnife =
          craneLib.buildPackage (crateArgs // { inherit cargoArtifacts; });

        ddnnife-d4 = craneLib.buildPackage
          (crateArgs-d4 // { cargoArtifacts = cargoArtifacts-d4; });
      in {
        formatter = pkgs.nixfmt;

        checks = {
          format = craneLib.cargoFmt { inherit src; };

          lint = craneLib.cargoClippy
            (crateArgs-d4 // { cargoArtifacts = cargoArtifacts-d4; });

          test = craneLib.cargoNextest (crateArgs-d4 // {
            doCheck = true;
            cargoArtifacts = cargoArtifacts-d4;
            cargoNextestExtraArgs =
              "--test-threads 1 --no-fail-fast --hide-progress-bar";
          });
        };

        packages = {
          default = self.packages.${system}.ddnnife-d4;

          inherit ddnnife;
          inherit ddnnife-d4;

          container = pkgs.dockerTools.buildLayeredImage {
            name = "ddnnife";
            contents = [ self.packages.${system}.ddnnife-d4 ];
            config = {
              Entrypoint = [ "/bin/ddnnife" ];
              Labels = {
                "org.opencontainers.image.source" =
                  "https://github.com/SoftVarE-Group/d-dnnf-reasoner";
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
            name = "documentation-d4";
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
            ] ++ lib.optionals pkgs.stdenv.isDarwin [ pkgs.libiconv ];
          };

          all-d4 = pkgs.buildEnv {
            name = "ddnnife-d4";
            paths = [
              self.packages.${system}.ddnnife-d4
              self.packages.${system}.documentation-d4
              d4.packages.${system}.dependencies
            ] ++ lib.optionals pkgs.stdenv.isDarwin [ pkgs.libiconv ];
          };
        };

        devShells.default = craneLib.devShell { inputsFrom = [ ddnnife-d4 ]; };
      });
}
