{
  inputs = {
    mach-nix.url = "mach-nix";
  };

  outputs = {self, nixpkgs, mach-nix }@inp:
    let
      l = nixpkgs.lib // builtins;
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];
      forAllSystems = f: l.genAttrs supportedSystems
        (system: f system (import nixpkgs {inherit system;}));
    in
    {
      # enter this python environment by executing `nix shell .`
      defaultPackage = forAllSystems (system: pkgs: mach-nix.lib."${system}".mkPython {
        requirements = ''
	  tensorflow
          gymnasium
          pybullet
          scipy
          pygame
          numpy
	'';
	providers.pygame = "nixpkgs";
	providers.pybullet = "conda/nixpkgs";
      });
    };
}
