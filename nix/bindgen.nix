{
  crane,
  fenix,
  stdenv,
  pkgs,
}:
let
  toolchain = pkgs: fenix.packages.${pkgs.stdenv.buildPlatform.system}.stable.defaultToolchain;
  craneLib = (crane.mkLib pkgs).overrideToolchain (p: toolchain p);
  metadata = craneLib.crateNameFromCargoToml { cargoToml = ../ddnnife_bindgen/Cargo.toml; };
in
craneLib.buildPackage {
  pname = metadata.pname;
  version = metadata.version;

  src = craneLib.cleanCargoSource ./..;
  strictDeps = true;

  cargoExtraArgs = "-p ddnnife_bindgen";
}
