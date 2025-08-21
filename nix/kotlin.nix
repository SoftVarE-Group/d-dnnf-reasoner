{
  bindgen,
  craneLibDefault,
  ddnnife-kotlin,
  fenix,
  gradle,
  lib,
  libddnnife,
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
    fileset = lib.fileset.unions [
      (craneLib.fileset.commonCargoSources ./..)
      ../bindings/kotlin
      ../example_input
    ];
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
    "-Plibrary=${libddnnife}/lib/${libFile}"
    "-Pbindgen=${bindgen}/bin/uniffi-bindgen"
  ]
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
