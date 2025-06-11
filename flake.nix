{
  description = "Packages and development environments for ddnnife";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane.url = "github:ipetkov/crane/v0.20.3";
    d4 = {
      url = "github:SoftVarE-Group/d4v2/2.2.0";
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

      # Determines the build system from the given package set.
      # Can be used for pointing to Windows cross-builds.
      buildSystem =
        pkgs:
        if pkgs.stdenv.hostPlatform.isWindows then pkgs.stdenv.buildPlatform.system else pkgs.stdenv.system;

      # If building for Windows, appends -windows, otherwise nothing.
      windowsSuffix = pkgs: name: if pkgs.stdenv.hostPlatform.isWindows then "${name}-windows" else name;

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
            self.packages.${system}."${windowsSuffix' "ddnnife"}-d4"
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

          defaultAttrs = {
            buildPkgs = pkgs;
            inherit fenix;
            inherit crane;
            component = "ddnnife_bin";
          };

          staticAttrs.hostPkgs = pkgs.pkgsStatic;
          windowsAttrs.hostPkgs = pkgs-windows;

          d4Attrs = defaultAttrs // {
            d4 = true;
            mt-kahypar = d4.packages.${system}.mt-kahypar;
          };

          libAttrs = {
            name = "libddnnife";
            component = "ddnnife_ffi";
            test = false;
          };
        in
        {
          default = self.packages.${system}.ddnnife-d4;

          ddnnife = import ./nix/ddnnife.nix defaultAttrs;
          ddnnife-static = import ./nix/ddnnife.nix (defaultAttrs // staticAttrs);

          ddnnife-d4 = import ./nix/ddnnife.nix d4Attrs;
          ddnnife-d4-bundled = bundled-d4 pkgs;

          ddnnife-windows = import ./nix/ddnnife.nix (defaultAttrs // windowsAttrs);
          ddnnife-windows-d4 = import ./nix/ddnnife.nix (
            d4Attrs // windowsAttrs // { mt-kahypar = d4.packages.${system}.mt-kahypar-windows; }
          );
          ddnnife-windows-d4-bundled = bundled-d4 pkgs-windows;

          libddnnife = import ./nix/ddnnife.nix (defaultAttrs // libAttrs);
          libddnnife-d4 = import ./nix/ddnnife.nix (d4Attrs // libAttrs);

          dependencies-d4 = d4.packages.${system}.dependencies;
          dependencies-d4-windows = d4.packages.${system}.dependencies-windows;

          bindgen = import ./nix/ddnnife.nix (
            defaultAttrs
            // {
              name = "bindgen";
              component = "ddnnife_bindgen";
              test = false;
            }
          );

          python = import ./nix/ddnnife.nix (
            defaultAttrs
            // {
              pythonLib = true;
              component = "ddnnife";
              test = false;
            }
          );

          documentation = import ./nix/ddnnife.nix (
            defaultAttrs
            // {
              documentation = true;
              component = "ddnnife";
            }
          );

          container = pkgs.dockerTools.buildLayeredImage {
            name = "ddnnife";
            contents = [
              self.packages.${system}.ddnnife-d4
              (pkgs.runCommand "create-tmp" { } "install -dm 1777 $out/tmp")
            ];
            config = {
              Entrypoint = [ "/bin/ddnnife" ];
              Labels = {
                "org.opencontainers.image.source" = "https://github.com/SoftVarE-Group/d-dnnf-reasoner";
                "org.opencontainers.image.description" = "A d-DNNF reasoner";
                "org.opencontainers.image.licenses" = "LGPL-3.0-or-later";
              };
            };
          };
        }
      );
      checks = lib.genAttrs systems (
        system:
        let
          defaultAttrs = {
            buildPkgs = nixpkgs.legacyPackages.${system};
            inherit fenix;
            inherit crane;
          };

          d4Attrs = defaultAttrs // {
            d4 = true;
            mt-kahypar = d4.packages.${system}.mt-kahypar;
          };
        in
        {
          format = import ./nix/ddnnife.nix (defaultAttrs // { format = true; });
          lint = import ./nix/ddnnife.nix (d4Attrs // { lint = true; });
          deny = import ./nix/ddnnife.nix (defaultAttrs // { deny = true; });
        }
      );
      devShells = lib.genAttrs systems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          d4Pkgs = d4.packages.${system};
        in
        {
          default = pkgs.mkShell {
            nativeBuildInputs = [
              pkgs.cmake
              pkgs.pkg-config
            ];

            buildInputs = [
              pkgs.boost.dev
              pkgs.pkgsStatic.gmp.dev
              pkgs.pkgsStatic.mpfr.dev
              d4Pkgs.mt-kahypar.dev
            ];
          };
        }
      );
    };
}
