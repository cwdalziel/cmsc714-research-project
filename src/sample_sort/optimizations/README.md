# Sample Sort — Topology-Specific Optimizations

One scaffold per topology. **Each currently runs the agnostic algorithm
unchanged** — they exist so that:

1. The build/run/CSV pipeline is verified for all 6 binaries before any
   optimization work starts.
2. Each `.cpp` has a clearly-marked `STEP 7 OPTIMIZATION POINT` comment
   block describing the topology-specific approach to replace the
   placeholder `MPI_Alltoallv` with.

The scaffolds **all pass `--verify` today** (sanity check against the
agnostic algorithm). Their timings are ≈ identical to agnostic until you
replace step 7.

## Files

| File | Strong np=256 baseline | Weak np=256 baseline | Replacement strategy (see file) |
|------|----------------------:|---------------------:|---------------------------------|
| `sample_sort_ring.cpp` | 0.0826 s | 0.1868 s | Ring rotation: `p-1` rounds of single-hop neighbor `MPI_Sendrecv`s |
| `sample_sort_torus.cpp` | 0.0069 s | 0.0488 s | Row-then-column phases on a 2D Cartesian sub-grid |
| `sample_sort_hypercube.cpp` | 0.0054 s | 0.0342 s | Recursive halving / hyperquicksort over `log p` dims |
| `sample_sort_fat_tree.cpp` | 0.0041 s | 0.1488 s | Investigate np=256 weak surprise first; chunked or non-blocking alltoallv |
| `sample_sort_dragonfly.cpp` | 0.0092 s | 0.1018 s | Hierarchical (intra-group then inter-group) |

(Baseline numbers from `../RESULTS.md`. Optimized variants need to beat
these on their matching topology.)

## How to work on one variant

1. Open the file and read the `STEP 7 OPTIMIZATION POINT` comment block —
   that's the only section to change.
2. Replace the placeholder `MPI_Alltoallv` call with the topology-specific
   redistribution.
3. From the repo root, build inside the container:
   ```
   make docker-make
   ```
4. Smoke-test correctness on a small problem before benchmarking:
   ```
   make docker-shell
   smpirun -np 16 -platform platforms/16/<topology>.xml \
       --cfg=smpi/host-speed:auto --cfg=smpi/coll-selector:ompi \
       ./bin/sample_sort/optimizations/sample_sort_<topology> 1024 --verify
   ```
   If `VERIFY: ok`, the algorithm is correct.
5. Re-run the scaling sweep — `run_sample_sort_scaling.sh` automatically
   picks up the binary and produces a fresh CSV per (np × mode):
   ```
   ./run_sample_sort_scaling.sh
   ```
   New CSVs land at `results/{strong,weak}/<np>/sample_sort_<topology>_results.csv`
   alongside the existing agnostic CSVs (no clobbering).
6. Compare the new CSVs against the agnostic baseline numbers in
   `../RESULTS.md`; document the speedup.

## Order to tackle them in (suggested)

Biggest expected gain first → most dramatic chart story:

1. **ring** — agnostic is catastrophic; the rotation pattern should
   provide a large improvement.
2. **hypercube** — recursive halving is mathematically clean and the
   baseline is already strong, so beating it is a meaningful technical
   result.
3. **torus** — well-studied 2D-grid optimization; many references.
4. **dragonfly** — hierarchical sample sort, conceptually similar to
   torus but with non-uniform group structure.
5. **fat_tree** — last; the np=256 weak surprise is the interesting story
   here, not raw speedup. Investigate `platforms/256/fat_tree.xml` and
   alternative coll-selectors *before* writing code.
