{
  pkgs,
  stdenv,
  fenix,
  crane,
}:
let
  rust = import ./rust.nix { inherit fenix; };
  target =
    if stdenv.buildPlatform.isStatic then
      rust.map.${stdenv.system}.static
    else
      rust.map.${stdenv.system}.default;

  toolchain = rust.toolchain stdenv.buildPlatform.system target;
  craneLib = (crane.mkLib pkgs).overrideToolchain toolchain;
  metadata = craneLib.crateNameFromCargoToml { cargoToml = ../ddnnife_bindgen/Cargo.toml; };
in
craneLib.buildPackage {
  pname = metadata.pname;
  version = metadata.version;

  src = craneLib.cleanCargoSource ./..;
  strictDeps = true;

  cargoExtraArgs = "-p ddnnife_bindgen";
}
