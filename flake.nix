{
  description = "Build a cargo project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs = {
        nixpkgs.follows = "nixpkgs";
      };
    };

    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      crane,
      rust-overlay,
      flake-utils,
      advisory-db,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowBroken = true;
          };
          overlays = [ (import rust-overlay) ];
        };

        inherit (pkgs) lib;

        rust-toolchain = pkgs.rust-bin.selectLatestNightlyWith (
          toolchain:
          toolchain.default.override {
            extensions = [ "rust-src" ];
            targets = [
              "aarch64-apple-darwin"
              "x86_64-apple-darwin"
              "aarch64-unknown-linux-gnu"
              "x86_64-unknown-linux-gnu"
            ];
          }
        );

        craneLib = (crane.mkLib pkgs).overrideToolchain rust-toolchain;
        src = craneLib.cleanCargoSource (craneLib.path ./.);

        commonArgs = {
          inherit src;

          buildInputs =
            [ ]
            ++ lib.optionals pkgs.stdenv.isDarwin [
              # Additional darwin specific inputs can be set here
              pkgs.libiconv
            ];
        };

        # Build *just* the cargo dependencies, so we can reuse
        # all of that work (e.g. via cachix) when running in CI
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        rs-poker = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            doCheck = false;
          }
        );
      in
      {
        checks =
          {
            # Build the crate as part of `nix flake check` for convenience
            inherit rs-poker;

            # Run clippy (and deny all warnings) on the crate source,
            # again, resuing the dependency artifacts from above.
            rs-poker-clippy = craneLib.cargoClippy (
              commonArgs
              // {
                inherit cargoArtifacts;
                cargoClippyExtraArgs = "--all-targets -- --deny warnings";
              }
            );

            rs-poker-doc = craneLib.cargoDoc (commonArgs // { inherit cargoArtifacts; });

            # Check formatting
            rs-poker-fmt = craneLib.cargoFmt {
              inherit src;
              cargoClippyExtraArgs = "--all --check";
            };

            # Audit dependencies
            rs-poker-audit = craneLib.cargoAudit { inherit src advisory-db; };

            # Run tests since rs-poker has doCheck = false;
            rs-poker-nextest = craneLib.cargoNextest (
              commonArgs
              // {
                inherit cargoArtifacts;
                partitions = 1;
                partitionType = "count";
                cargoNextestExtraArgs = "--all-targets";
              }
            );
          }
          // lib.optionalAttrs (system == "x86_64-linux") {
            rs-poker-coverage = craneLib.cargoTarpaulin (commonArgs // { inherit cargoArtifacts; });
          };

        packages = {
          inherit rs-poker;
          default = rs-poker;
        };

        apps.default = flake-utils.lib.mkApp { drv = rs-poker; };

        devShells.default = pkgs.mkShell {
          inputsFrom = builtins.attrValues self.checks.${system};

          nativeBuildInputs = with pkgs; [
            rust-toolchain
            rust-analyzer
            pkg-config
            git
            cmake
            openssl
          ];

          shellHook = ''
            BASE=$(git rev-parse --show-toplevel || echo ".")

            # This keeps cargo self contained in this dir
            export CARGO_HOME=$BASE/.nix-cargo
            mkdir -p $CARGO_HOME
            export PATH=$CARGO_HOME/bin:$PATH
          '';
        };
      }
    );
}
