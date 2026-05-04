# CMSC 714 Research Project

## Local setup (Docker — recommended)

The repo ships a `Dockerfile` that builds an Ubuntu 22.04 image with SimGrid
4.1 + FFTW + the C++ build chain pre-installed. This is the supported way to
develop on macOS/Windows; native Linux users can also use it for
toolchain/version consistency across teammates.

```
make docker-build   # one-time, ~10 min
make docker-shell   # drop into a shell, repo mounted at /work
make docker-make    # one-shot: run `make` inside the container
```

Inside `docker-shell`, `make` builds binaries and `./run_benchmarks.sh ...`
runs them as documented below — `smpicxx` and `smpirun` are on the container
PATH. Re-run `make docker-build` only when the `Dockerfile` itself changes.

## SimGrid install (native, if not using Docker)

Use the latest version of SimGrid on Linux.
Source:
https://github.com/simgrid/simgrid/releases/download/v4.1/simgrid-4.1.tar.gz

Install:
https://simgrid.org/doc/latest/Installing_SimGrid.html#installing-from-source



## simgrid docs:

https://simgrid.org/doc/latest/index.html

mpi specifically: https://simgrid.org/doc/latest/Tutorial_MPI_Applications.html

## simgrid usage

compile with `smpicxx` or `smpicc` and run with `smpirun` which takes the same args as `mpirun` but with `-platform example.xml` to use network topology defined in xml file

## run_benchmarks example usage (after running make)
Usage: ./run_benchmarks.sh \<binary> \<np> [program arguments...]

Examples:

`
./run_benchmarks.sh bin/stencil 64
`

`
./run_benchmarks.sh bin/3d_stencil/3d_stencil_torus_hypercube 128 96 96 96
`