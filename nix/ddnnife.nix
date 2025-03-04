{
  buildPkgs,
  hostPkgs ? buildPkgs,
  fenix,
  crane,
  mt-kahypar ? null,
  component ? "",
  name ? "ddnnife",
  d4 ? false,
  pythonLib ? false,
  test ? true,
  deny ? false,
  documentation ? false,
  format ? false,
  lint ? false,
}:
let
  lib = buildPkgs.lib;

  buildSystem = buildPkgs.stdenv.system;
  hostSystem = hostPkgs.stdenv.system;

  # The default MinGW GCC in Nix comes with mcfgthreads which seems to be unable
  # to produce static Rust binaries with C dependencies.
  cc-windows = hostPkgs.buildPackages.wrapCC (
    hostPkgs.buildPackages.gcc-unwrapped.override ({
      threadsCross = {
        model = "win32";
        package = null;
      };
    })
  );

  rust = import ./rust.nix { inherit fenix; };
  static = hostPkgs.hostPlatform.isStatic;
  target = if static then rust.map.${hostSystem}.static else rust.map.${hostSystem}.default;
  craneLib = (crane.mkLib buildPkgs).overrideToolchain (rust.toolchain buildSystem target);

  metadata = craneLib.crateNameFromCargoToml { cargoToml = ../ddnnife/Cargo.toml; };

  features = lib.optionalString d4 "--features d4";

  craneAction =
    if deny then
      "cargoDeny"
    else if documentation then
      "cargoDoc"
    else if format then
      "cargoFmt"
    else if lint then
      "cargoClippy"
    else
      "buildPackage";

  crate =
    {
      meta = {
        mainProgram = "ddnnife";
        description = "A d-DNNF reasoner.";
        homepage = "https://github.com/SoftVarE-Group/d-dnnf-reasoner";
        license = lib.licenses.lgpl3Plus;
        platforms = lib.platforms.unix ++ lib.platforms.windows;
      };

      # The build differs between the variants and the dep build should therefore be named differently.
      pname = lib.concatStringsSep "-" ([ "ddnnife" ] ++ lib.optionals d4 [ "d4" ]);

      version = metadata.version;

      src = ./..;
      strictDeps = true;

      buildInputs =
        lib.optionals d4 [
          hostPkgs.boost.dev
          mt-kahypar.dev
        ]
        ++ lib.optionals (d4 && hostPkgs.stdenv.hostPlatform.isLinux) [
          hostPkgs.pkgsStatic.gmp.dev
          hostPkgs.pkgsStatic.mpfr.dev
        ]
        ++ lib.optionals (d4 && hostPkgs.stdenv.hostPlatform.isDarwin) [
          (hostPkgs.gmp.override {
            withStatic = true;
          })
          hostPkgs.mpfr.dev
        ]
        ++ lib.optionals (d4 && hostPkgs.stdenv.hostPlatform.isWindows) [
          (hostPkgs.gmp.override {
            stdenv = hostPkgs.overrideCC hostPkgs.stdenv cc-windows;
            withStatic = true;
          })
          hostPkgs.mpfr.dev
        ]
        ++ lib.optionals hostPkgs.stdenv.isDarwin [ hostPkgs.libiconv ];

      nativeBuildInputs =
        lib.optionals d4 [
          buildPkgs.cmake
          buildPkgs.pkg-config
        ]
        ++ lib.optionals pythonLib [ buildPkgs.maturin ];

      cargoExtraArgs = features;

      CARGO_BUILD_TARGET = target;
      TARGET_CC = "${hostPkgs.stdenv.cc}/bin/${hostPkgs.stdenv.cc.targetPrefix}cc";

      doCheck = test;
    }
    // lib.optionalAttrs hostPkgs.stdenv.hostPlatform.isWindows {
      TARGET_CC = "${cc-windows}/bin/${cc-windows.targetPrefix}cc";
      TARGET_CXX = "${cc-windows}/bin/${cc-windows.targetPrefix}cc";

      depsBuildBuild = [
        cc-windows
        hostPkgs.windows.pthreads
      ];

      CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER = (
        buildPkgs.writeShellScript "wine-wrapped" ''
          export WINEPREFIX=''$(mktemp -d)
          export WINEDEBUG=-all
          ${buildPkgs.wineWow64Packages.minimal}/bin/wine $@
        ''
      );
    }
    // lib.optionalAttrs (d4 && hostPkgs.stdenv.system == "x86_64-darwin") {
      # FIXME: Tests with d4 are currently unable to run on x86_64-darwin.
      doCheck = false;
    }
    // lib.optionalAttrs (d4 && hostPkgs.stdenv.hostPlatform.isWindows) {
      # The Windows cross-build won't find the correct include and library directories by default.
      CXXFLAGS = "-I ${hostPkgs.boost.dev}/include -I ${mt-kahypar.dev}/include";
      CARGO_BUILD_RUSTFLAGS = "-L ${mt-kahypar}/lib";

      # FIXME: Tests with d4 are currently unable to run on x86_64-windows.
      doCheck = false;
    };

  cargoArtifacts = craneLib.buildDepsOnly (
    crate
    // {
      # The FFI crate should not be part of the pre-built dependencies as it is not compatible with every target.
      cargoExtraArgs = "--workspace --exclude ddnnife_ffi";
    }
  );
in
craneLib.${craneAction} (
  crate
  // {
    pname = name;

    cargoExtraArgs = lib.concatStringsSep " " (
      lib.optionals (component != "") [ "--package ${component}" ] ++ [ features ]
    );

    cargoTestExtraArgs = "--workspace";

    inherit cargoArtifacts;
  }
  // lib.optionalAttrs hostPkgs.stdenv.isAarch64 {
    # FIXME: Doc-tests currently fail on aarch64-{darwin, linux}.
    cargoTestExtraArgs = "--workspace --all-targets";
  }
  // lib.optionalAttrs pythonLib {
    buildPhaseCargoCommand = ''
      cd bindings/python
      maturin build --offline
      cd ../..
    '';

    installPhaseCommand = ''
      mkdir -p $out
      cp target/wheels/* $out/
    '';

    doNotPostBuildInstallCargoBinaries = true;
  }
  // lib.optionalAttrs lint { cargoClippyExtraArgs = "--all-features -- --deny warnings"; }
)
