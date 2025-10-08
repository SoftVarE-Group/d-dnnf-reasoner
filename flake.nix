{
  description = "Packages and development environments for ddnnife";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane.url = "github:ipetkov/crane/v0.20.3";
  };

  outputs =
    {
      self,
      nixpkgs,
      fenix,
      crane,
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
    in
    {
      formatter = lib.genAttrs systems (system: nixpkgs.legacyPackages.${system}.nixfmt-rfc-style);
      packages = lib.genAttrs systems (
        system:
        let
          # Shorthands for different package sets.
          pkgs = nixpkgs.legacyPackages.${system};
          pkgsStatic = pkgs.pkgsStatic;
          pkgsWindows = pkgs.pkgsCross.mingwW64;
          pkgsSelf = self.packages.${system};

          rustAttrs = {
            craneLibDefault = crane.mkLib pkgs;
            inherit fenix;
          };

          defaultAttrs = rustAttrs // {
            component = "ddnnife_cli";
          };

          libAttrs = defaultAttrs // {
            name = "libddnnife";
            component = "ddnnife_ffi";
            test = false;
          };

          kotlinAttrs = rustAttrs // {
            bindgen = pkgsSelf.bindgen;
            ddnnife-kotlin = pkgsSelf.kotlin;
            libddnnife = pkgsSelf.libddnnife;
          };

          pythonAttrs = defaultAttrs // {
            pythonLib = true;
            test = false;
          };
        in
        {
          default = pkgsSelf.ddnnife;

          ddnnife = pkgs.callPackage ./nix/ddnnife.nix defaultAttrs;
          ddnnife-static = pkgsStatic.callPackage ./nix/ddnnife.nix defaultAttrs;
          ddnnife-windows = pkgsWindows.callPackage ./nix/ddnnife.nix defaultAttrs;

          libddnnife = pkgs.callPackage ./nix/ddnnife.nix libAttrs;
          libddnnife-windows = pkgsWindows.callPackage ./nix/ddnnife.nix libAttrs;

          bindgen = pkgs.callPackage ./nix/bindgen.nix rustAttrs;

          kotlin = pkgs.callPackage ./nix/kotlin.nix kotlinAttrs;
          kotlin-windows = pkgsWindows.callPackage ./nix/kotlin.nix (
            kotlinAttrs
            // {
              libddnnife = pkgsSelf.libddnnife-windows;
              ddnnife-kotlin = pkgsSelf.kotlin-windows;
            }
          );

          kotlin-bundled = pkgs.callPackage ./nix/kotlin.nix (
            kotlinAttrs
            // {
              bundled = true;
              libddnnife = null;
            }
          );

          python = pkgs.callPackage ./nix/ddnnife.nix pythonAttrs;
          python-static = pkgsStatic.callPackage ./nix/ddnnife.nix pythonAttrs;
          python-windows = pkgsWindows.callPackage ./nix/ddnnife.nix pythonAttrs;

          documentation = pkgs.callPackage ./nix/ddnnife.nix (
            defaultAttrs
            // {
              documentation = true;
              component = "ddnnife";
            }
          );

          benchmark = pkgs.callPackage ./nix/ddnnife.nix (rustAttrs // { benchmark = true; });

          container = pkgs.dockerTools.buildLayeredImage {
            name = "ddnnife";
            contents = [
              pkgsSelf.ddnnife
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
          pkgs = nixpkgs.legacyPackages.${system};

          defaultAttrs = {
            craneLibDefault = crane.mkLib pkgs;
            inherit fenix;
          };
        in
        {
          format = pkgs.callPackage ./nix/ddnnife.nix (defaultAttrs // { format = true; });
          lint = pkgs.callPackage ./nix/ddnnife.nix (defaultAttrs // { lint = true; });
          deny = pkgs.callPackage ./nix/ddnnife.nix (defaultAttrs // { deny = true; });
        }
      );
    };
}
