{
  bindgen,
  craneLibDefault,
  ddnnife-kotlin,
  fenix,
  gradle,
  lib,
  # A single library passed results in a target-specific build.
  libddnnife ? null,
  # A bundled build takes the libraries at `./libddnnife`.
  bundled ? false,
  pkgs,
  stdenv,
}:
let
  libPrefix = if stdenv.hostPlatform.isWindows then "" else "lib";
  libFile = "${libPrefix}ddnnife${stdenv.hostPlatform.extensions.sharedLibrary}";

  toolchain = pkgs: fenix.packages.${pkgs.stdenv.buildPlatform.system}.stable.defaultToolchain;
  craneLib = craneLibDefault.overrideToolchain (p: toolchain p);

  metadata = craneLib.crateNameFromCargoToml { cargoToml = ../ddnnife_ffi/Cargo.toml; };
  version = metadata.version;

  src = lib.fileset.toSource {
    root = ./..;
    fileset = lib.fileset.unions (
      [
        (craneLib.fileset.commonCargoSources ./..)
        ../bindings/kotlin
        ../example_input
      ]
      ++ lib.optionals bundled [ (lib.fileset.maybeMissing ./libddnnife) ]
    );
  };
in
stdenv.mkDerivation {
  pname = "ddnnife-kotlin";
  inherit version;

  inherit src;

  nativeBuildInputs = [
    gradle
    (toolchain pkgs)
  ];

  mitmCache = gradle.fetchDeps {
    pkg = ddnnife-kotlin;
    data = ./kotlin-dependencies.json;
  };

  __darwinAllowLocalNetworking = true;

  gradleBuildTask = "shadowJar";

  gradleFlags = [
    "--project-dir=bindings/kotlin"
    "-Pbindgen=${bindgen}/bin/uniffi-bindgen"
  ]
  ++ lib.optionals (!bundled) [ "-Plibrary=${libddnnife}/lib/${libFile}" ]
  ++ lib.optionals bundled [ "-Plibraries=../../nix/libddnnife" ]
  ++ lib.optionals stdenv.hostPlatform.isWindows [
    "-PgeneratePrefix=win32-x86-64"
    "-PgenerateLib=${libFile}"
  ];

  doCheck = true;

  installPhase =
    let
      jar = "ddnnife-${version}-all.jar";
    in
    "install -D bindings/kotlin/build/libs/${jar} $out/${jar}";
}
