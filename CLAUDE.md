# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

CMSC 714 research project: measure how MPI applications and collectives perform across different
simulated network topologies (ring, torus, hypercube, fat_tree, dragonfly) at varying node counts
(16, 32, 64, 128, 256). Everything is run under SimGrid's SMPI, not on real hardware — see README.md
for SimGrid install pointers. There is no test suite; "correctness" is reproducible CSV timings.

## Build

```
make            # builds every src/**/*.cpp into bin/<same-relative-path>
make clean      # remove bin/
make clean-results
make clean-all  # remove bin/ and results/
```

The Makefile auto-discovers sources via `find src -name '*.cpp'` and mirrors the directory layout
under `bin/`. Toolchain is hard-wired:
- `CXX = smpicxx` (SimGrid's MPI compiler wrapper — must be on PATH)
- `LDFLAGS = -lfftw3 -lfftw3_mpi -lm` (FFTW3 + FFTW3-MPI required even though only `2d_fft_mpi.cpp`
  uses them; every binary links them)

### Local environment (Docker)

The team runs SimGrid locally, not on a shared cluster. The repo's `Dockerfile`
provides Ubuntu 22.04 + SimGrid 4.1 + FFTW; this is the supported macOS/Windows path
and the recommended path on Linux for toolchain consistency:

```
make docker-build   # one-time, ~10 min (builds SimGrid from source inside the image)
make docker-shell   # interactive shell inside the container, repo mounted at /work
make docker-make    # one-shot: run `make` inside the container
```

`smpicxx` and `smpirun` are only available inside the container. Editor diagnostics on
the host (clang not finding `mpi.h`) are expected noise — actual builds happen via
`make` inside `docker-shell` (or via `make docker-make`).

## Run a single benchmark

```
./run_benchmarks.sh <binary> <np> [program args...]
# e.g.
./run_benchmarks.sh bin/stencil 64
./run_benchmarks.sh bin/3d_stencil/3d_stencil_torus_hypercube 128 96 96 96
```

The script iterates over **every** `platforms/<np>/*.xml` (i.e. all topologies at that node count),
invokes `smpirun -np <np> -platform <xml> --cfg=smpi/host-speed:auto --cfg=smpi/coll-selector:ompi
<binary> <args>`, and writes one CSV row per topology to `results/<np>/<basename>_results.csv`.

If the env var `RESULTS_SUBDIR` is set, output goes to `results/$RESULTS_SUBDIR/<np>/...` —
`run_3d_stencil_scaling.sh` uses this to keep `strong/` and `weak/` studies separate.

## Run the 3D-stencil scaling sweep

```
./run_3d_stencil_scaling.sh [strong|weak|all]   # default: all
```

Loops `np ∈ {16,32,64,128,256}` × the four `3d_stencil_*` binaries × every topology XML for that np.
Strong = fixed `STRONG_N=256` cubic grid (override via env). Weak = per-np grid from
`weak_edge_for_np()` chosen so `NZ % np == 0` (the ring variant requires this).

## How a new benchmark plugs in

1. Drop a `.cpp` under `src/` (or `src/<subdir>/`). `make` picks it up automatically; the binary
   path mirrors the source path.
2. **Print exactly one timing line ending in `... s`** on rank 0. `run_benchmarks.sh` extracts the
   number with `grep -oP '[\d.e+-]+(?= s)' | tail -1`. If the program prints multiple `... s`
   lines, only the last is recorded; if none, the row is logged as `ERROR`. Existing examples:
   `"Stencil avg time per iter: ... s"`, `"Alltoall avg time: ... s"`, `"RingAllreduce time: ... s"`.
3. To benchmark at a new node count `N`, add `platforms/N/<topology>.xml` files **and** extend
   `NP_LIST` in `run_3d_stencil_scaling.sh` (and `weak_edge_for_np`) if the new count should
   participate in the scaling sweep.

## Platform XML files

`platforms/<np>/*.xml` define SimGrid network topologies. The host count in each XML must equal the
`<np>` directory name. Filenames are the topology label that ends up in the CSV's `topology` column,
so don't rename them casually. Two generators exist:
- `generate_hypercube.py <n>` — prints a hypercube XML to stdout (n must be a power of 2).
- `generate_ring.py` — n is hardcoded at the top of the file; edit it before running.
The torus / fat_tree / dragonfly XMLs are hand-authored — there is no generator for them.

## Source layout — what's what

- `src/stencil.cpp` — 2D 5-point stencil, `MPI_Cart_create` + `MPI_Sendrecv` halo exchange.
- `src/3d_stencil/3d_stencil_*.cpp` — four topology-tuned 3D 7-point stencil variants. The
  decomposition / neighbor-mapping differs per topology; that's the point of the experiment, so
  don't "unify" them. Args: `prog [N]` (cubic) or `prog NX NY NZ`.
- `src/2d_fft_mpi.cpp` — slab-decomposed 2D FFT: row-FFT → all-to-all transpose → row-FFT →
  transpose back. The all-to-all is the topology-sensitive step. Arg: `N` (must be divisible by P).
- `src/alltoall.cpp`, `src/ring_allreduce.cpp` — collective microbenchmarks.
