{
  description = "Develop Python on Nix with uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
            ];

            env = lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              # Python libraries often load native shared objects using dlopen(3).
              # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
              LD_LIBRARY_PATH =
                lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc
                  pkgs.zlib
                  pkgs.glib
                  pkgs.SDL2
                  pkgs.libx11
                  pkgs.libxext
                  pkgs.libxrandr
                  pkgs.libxcursor
                  pkgs.libxi
                  pkgs.libxinerama
                  pkgs.libxxf86vm
                  pkgs.libxkbcommon
                  pkgs.wayland
                  pkgs.libGL
                  pkgs.libpulseaudio
                ]
                + ":/run/opengl-driver/lib:/run/opengl-driver-32/lib";
            };

            shellHook = ''
              unset PYTHONPATH
              uv sync
              . .venv/bin/activate
            '';
          };
        }
      );
    };
}
