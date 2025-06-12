{
  pkgs,
  stdenv,
  lib,
  gradle,
  rust,
  bindgen,
  libddnnife,
  ddnnife-kotlin,
}:
let
  libPrefix = if stdenv.hostPlatform.isWindows then "" else "lib";
  libFile = "${libPrefix}ddnnife${stdenv.hostPlatform.extensions.sharedLibrary}";

  metadata = rust.craneLib.crateNameFromCargoToml { cargoToml = ../ddnnife_ffi/Cargo.toml; };
  version = metadata.version;
in
stdenv.mkDerivation {
  pname = "ddnnife-kotlin";
  inherit version;

  src = ../.;

  nativeBuildInputs = [
    gradle
    rust.toolchainDefault
  ];

  mitmCache = gradle.fetchDeps {
    pkg = ddnnife-kotlin;
    data = ./kotlin-dependencies.json;
  };

  __darwinAllowLocalNetworking = true;

  gradleBuildTask = "shadowJar";

  gradleFlags =
    [
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
    "install -D bindings/kotlin/build/libs/${jar} $out/share/ddnnife/${jar}";
}
