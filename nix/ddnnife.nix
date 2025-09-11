{
  name ? "ddnnife",
  component ? null,
  pythonLib ? false,
  test ? true,
  deny ? false,
  format ? false,
  lint ? false,
  documentation ? false,
  craneLibDefault,
  buildPackages,
  fenix,
  lib,
  maturin,
  pkgs,
  stdenv,
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

  cc = if stdenv.targetPlatform.isWindows then cc-windows else stdenv.cc;

  target = stdenv.targetPlatform.rust.rustcTarget;

  toolchain =
    pkgs:
    let
      system = pkgs.stdenv.buildPlatform.system;
    in
    fenix.packages.${system}.combine [
      fenix.packages.${system}.stable.defaultToolchain
      fenix.packages.${system}.targets.${target}.stable.rust-std
    ];

  craneLib = craneLibDefault.overrideToolchain (p: toolchain p);

  metadata = craneLib.crateNameFromCargoToml { cargoToml = ../ddnnife/Cargo.toml; };

  src = lib.fileset.toSource {
    root = ./..;
    fileset = lib.fileset.unions (
      [
        (craneLib.fileset.commonCargoSources ./..)
        ../ddnnife/tests/data
        ../ddnnife_cli/tests/data
        ../example_input
      ]
      ++ lib.optionals pythonLib [
        ../bindings/python
      ]
    );
  };

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

  # The FFI crates should not be part of the default built.
  cargoExtraArgs = "--workspace --exclude ddnnife_bindgen --exclude ddnnife_ffi";

  crate = {
    meta = {
      mainProgram = "ddnnife";
      description = "A d-DNNF reasoner.";
      homepage = "https://github.com/SoftVarE-Group/d-dnnf-reasoner";
      license = lib.licenses.lgpl3Plus;
      platforms = lib.platforms.unix ++ lib.platforms.windows;
    };

    pname = metadata.pname;
    version = metadata.version;

    inherit src;
    strictDeps = true;

    CARGO_BUILD_TARGET = target;
    TARGET_CC = lib.getExe cc;

    doCheck = test;
  }
  // lib.optionalAttrs stdenv.targetPlatform.isWindows {
    depsBuildBuild = [
      cc
      pkgs.windows.pthreads
    ];

    CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER = (
      buildPackages.writeShellScript "wine-wrapped" ''
        export WINEPREFIX=''$(mktemp -d)
        export WINEDEBUG=-all
        ${lib.getExe buildPackages.wineWow64Packages.minimal} $@
      ''
    );
  };

  cargoArtifacts = craneLib.buildDepsOnly (crate // { inherit cargoExtraArgs; });
in
craneLib.${craneAction} (
  crate
  // {
    pname = name;

    inherit cargoArtifacts;

    cargoExtraArgs = lib.optionalString (component != null) "--package ${component}";
    cargoTestExtraArgs = cargoExtraArgs;
    cargoClippyExtraArgs = "--all-features -- --deny warnings";
  }
  // lib.optionalAttrs pythonLib {
    nativeBuildInputs = [
      maturin
    ];

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
)
