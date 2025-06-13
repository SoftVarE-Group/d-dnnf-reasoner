{
  stdenv,
  lib,
  buildPackages,
  overrideCC,
  windows,
  libiconv,
  cmake,
  pkg-config,
  gmp,
  mpfr,
  writeShellScript,
  wineWow64Packages,
  rust,
  mt-kahypar ? null,
  component ? "",
  name ? "ddnnife",
  d4 ? false,
  craneAction ? "buildPackage",
}:
let
  # The default MinGW GCC in Nix comes with mcfgthreads which seems to be unable
  # to produce static Rust binaries with C dependencies.
  cc-windows = buildPackages.wrapCC (
    buildPackages.gcc-unwrapped.override ({
      threadsCross = {
        model = "win32";
        package = null;
      };
    })
  );

  craneLib = rust.craneLib;
  metadata = craneLib.crateNameFromCargoToml { cargoToml = ../ddnnife/Cargo.toml; };

  features = lib.optionalString d4 "--features d4";

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
          boost.dev
          mt-kahypar.dev
        ]
        ++ lib.optionals (d4 && stdenv.hostPlatform.isUnix) [
          (gmp.override {
            withStatic = true;
          })
          mpfr.dev
        ]
        ++ lib.optionals (d4 && stdenv.hostPlatform.isWindows) [
          (gmp.override {
            stdenv = overrideCC stdenv cc-windows;
            withStatic = true;
          })
          mpfr.dev
        ]
        ++ lib.optionals stdenv.isDarwin [ libiconv ];

      nativeBuildInputs = lib.optionals d4 [
        cmake
        pkg-config
      ];

      cargoExtraArgs = features;
      cargoTestExtraArgs = "--workspace";
      cargoClippyExtraArgs = "--all-features -- --deny warnings";

      CARGO_BUILD_TARGET = target;
      TARGET_CC = "${stdenv.cc}/bin/${stdenv.cc.targetPrefix}cc";

      doCheck = test;
    }
    // lib.optionalAttrs stdenv.hostPlatform.isWindows {
      TARGET_CC = "${cc-windows}/bin/${cc-windows.targetPrefix}cc";
      TARGET_CXX = "${cc-windows}/bin/${cc-windows.targetPrefix}cc";

      depsBuildBuild = [
        cc-windows
        windows.pthreads
      ];

      CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER = (
        writeShellScript "wine-wrapped" ''
          export WINEPREFIX=''$(mktemp -d)
          export WINEDEBUG=-all
          ${wineWow64Packages.minimal}/bin/wine $@
        ''
      );
    }
    // lib.optionalAttrs (d4 && stdenv.system == "x86_64-darwin") {
      # FIXME: Tests with d4 are currently unable to run on x86_64-darwin.
      doCheck = false;
    }
    // lib.optionalAttrs (d4 && stdenv.hostPlatform.isWindows) {
      # The Windows cross-build won't find the correct include and library directories by default.
      CXXFLAGS = "-I ${boost.dev}/include -I ${mt-kahypar.dev}/include";
      CARGO_BUILD_RUSTFLAGS = "-L ${mt-kahypar}/lib";

      # FIXME: Tests with d4 are currently unable to run on x86_64-windows.
      doCheck = false;
    };

  cargoArtifacts = craneLib.buildDepsOnly (
    crate
    // {
      # The FFI crates should not be part of the pre-built dependencies.
      cargoExtraArgs = "${features} --workspace --exclude ddnnife_bindgen --exclude ddnnife_ffi";
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

    inherit cargoArtifacts;
  }
  // lib.optionalAttrs stdenv.isAarch64 {
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
