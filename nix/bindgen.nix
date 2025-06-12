{
  stdenv,
  rust,
}:
let
  craneLib = rust.craneLib;
  metadata = rust.craneLib.crateNameFromCargoToml { cargoToml = ../ddnnife_bindgen/Cargo.toml; };
in
craneLib.buildPackage {
  pname = metadata.pname;
  version = metadata.version;

  src = craneLib.cleanCargoSource ./..;
  strictDeps = true;

  cargoExtraArgs = "-p ddnnife_bindgen";
}
