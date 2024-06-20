{
  description = "Packages and development environments for ddnnife";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane = {
      url = "github:ipetkov/crane/v0.17.3";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    d4 = {
      url = "github:SoftVarE-Group/d4v2/mt-kahypar";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      fenix,
      crane,
      d4,
      ...
    }:
    let
      lib = nixpkgs.lib;

      # All supported build systems, the Windows build is cross-compiled.
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      # Rust target triple mappings for the target platforms.
      rust = {
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

      # Determines the build system from the given package set.
      # Can be used for pointing to Windows cross-builds.
      buildSystem =
        pkgs:
        if pkgs.stdenv.hostPlatform.isWindows then pkgs.stdenv.buildPlatform.system else pkgs.stdenv.system;

      # If building for Windows, appends -windows, otherwise nothing.
      windowsSuffix = pkgs: name: if pkgs.stdenv.hostPlatform.isWindows then "${name}-windows" else name;

      # Constructs a Rust toolchain from the build to the target system.
      toolchain =
        system: target:
        fenix.packages.${system}.combine [
          fenix.packages.${system}.stable.defaultToolchain
          fenix.packages.${system}.targets.${target}.stable.rust-std
        ];

      # A ddnnife build for the specified platform.
      # Build and host packages can be used for a cross-build.
      # This outputs a set of the corresponding crane library, the crate build definition,
      # the package and dependeny artifacts.
      ddnnife =
        buildPkgs: hostPkgs: withD4:
        let
          buildSystem = buildPkgs.stdenv.system;
          hostSystem = hostPkgs.stdenv.system;
          static = !withD4 && (rust.${hostSystem} ? static);
          windows = hostPkgs.stdenv.hostPlatform.isWindows;
          mt-kahypar = d4.packages.${buildSystem}.${windowsSuffix hostPkgs "mt-kahypar"};
          target = if static then rust.${hostSystem}.static else rust.${hostSystem}.default;
          craneLib = (crane.mkLib buildPkgs).overrideToolchain (toolchain buildSystem target);
          metadata = craneLib.crateNameFromCargoToml { cargoToml = ./ddnnife/Cargo.toml; };

          crate =
            {
              pname = metadata.pname;
              version = metadata.version;
              src = ./.;
              strictDeps = true;
              CARGO_BUILD_TARGET = target;

              # Darwin builds fail without libiconv.
              buildInputs = lib.optionals hostPkgs.stdenv.isDarwin [ hostPkgs.libiconv ];
            }
            // lib.optionalAttrs static {
              # A static build needs to set the C compiler for C dependencies to be compiled correctly.
              TARGET_CC =
                let
                  cc = hostPkgs.pkgsStatic.stdenv.cc;
                in
                "${cc}/bin/${cc.targetPrefix}cc";
            }
            // lib.optionalAttrs withD4 {
              cargoExtraArgs = "--features d4";

              buildInputs = [
                hostPkgs.boost.dev
                hostPkgs.gmp.dev
                mt-kahypar.dev
              ] ++ lib.optionals hostPkgs.stdenv.isDarwin [ hostPkgs.libiconv ];

              nativeBuildInputs = [
                buildPkgs.m4
                buildPkgs.pkg-config
              ];

              # FIXME: Tests with d4 are currently unable to run on x86_64-darwin.
              doCheck = buildSystem != "x86_64-darwin";
            }
            // lib.optionalAttrs windows (
              let
                # The default MinGW GCC in nix comes with mcfgthreads which seems to be unable
                # to produce static Rust binaries with C dependencies.
                cc = hostPkgs.buildPackages.wrapCC (
                  hostPkgs.buildPackages.gcc-unwrapped.override ({
                    threadsCross = {
                      model = "win32";
                      package = null;
                    };
                  })
                );
              in
              {
                TARGET_CC = "${cc}/bin/${cc.targetPrefix}cc";
                TARGET_CXX = "${cc}/bin/${cc.targetPrefix}cc";

                depsBuildBuild = [
                  cc
                  hostPkgs.windows.pthreads
                ];

                # Would need Wine support to run.
                doCheck = false;
              }
            )
            // lib.optionalAttrs (windows && withD4) {
              # The Windows cross-build won't find the correct include and library directories by default.
              CXXFLAGS = "-I ${hostPkgs.boost.dev}/include -I ${hostPkgs.gmp.dev}/include -I ${mt-kahypar.dev}/include";
              CARGO_BUILD_RUSTFLAGS = "-L ${mt-kahypar}/lib";
            };

          artifacts = craneLib.buildDepsOnly crate;
        in
        {
          inherit craneLib;
          inherit crate;
          inherit artifacts;
          package = craneLib.buildPackage (crate // { cargoArtifacts = artifacts; });
        };

      # A simple README explaining how to setup the built directories to run the binaries.
      documentation =
        pkgs:
        pkgs.stdenv.mkDerivation {
          name = "ddnnife-documentation";
          src = ./doc;
          installPhase = ''
            mkdir $out
            cp ${pkgs.stdenv.system}.md $out/README.md
          '';
        };

      # The d4-enabled binaries with all dependencies.
      bundled-d4 =
        pkgs:
        let
          system = buildSystem pkgs;
          windowsSuffix' = windowsSuffix pkgs;
        in
        pkgs.buildEnv {
          name = "ddnnife";
          paths = [
            self.packages.${system}.${windowsSuffix' "ddnnife-d4"}
            d4.packages.${system}.${windowsSuffix' "dependencies"}
            (documentation pkgs)
          ];
        };
    in
    {
      formatter = lib.genAttrs systems (system: nixpkgs.legacyPackages.${system}.nixfmt-rfc-style);
      packages = lib.genAttrs systems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pkgs-windows = pkgs.pkgsCross.mingwW64;
        in
        {
          default = self.packages.${system}.ddnnife-d4;

          ddnnife = (ddnnife pkgs pkgs false).package;
          ddnnife-d4 = (ddnnife pkgs pkgs true).package;

          ddnnife-windows = (ddnnife pkgs pkgs-windows false).package;
          ddnnife-d4-windows = (ddnnife pkgs pkgs-windows true).package;

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

          bundled-d4 = bundled-d4 pkgs;
          bundled-d4-windows = bundled-d4 pkgs-windows;
        }
      );
      checks = lib.genAttrs systems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          generated = ddnnife pkgs pkgs false;
          generated-d4 = ddnnife pkgs pkgs true;
          craneLib = generated-d4.craneLib;
        in
        {
          format = craneLib.cargoFmt generated.crate;
          lint = craneLib.cargoClippy (
            generated-d4.crate
            // {
              cargoArtifacts = generated-d4.artifacts;
              cargoClippyExtraArgs = "--all-features -- --deny warnings";
            }
          );
          deny = craneLib.cargoDeny generated.crate;
        }
      );
    };
}
