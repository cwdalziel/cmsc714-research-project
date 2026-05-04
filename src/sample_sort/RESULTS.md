# Sample Sort — Agnostic Baseline Results

**Status:** Phase 1 complete (2026-05-04). The agnostic sample-sort baseline
has been benchmarked across all 5 topologies × 5 process counts × 2 scaling
regimes. Raw CSVs live at
`../../results/{strong,weak}/<np>/sample_sort_results.csv`.

## Setup

- Binary: `bin/sample_sort/agnostic/sample_sort` (from `agnostic/sample_sort.cpp`)
- SimGrid 4.1 inside the project's Docker image (`../../Dockerfile`)
- **Strong scaling:** `total_N = 1<<24 = 16,777,216` keys for every np
- **Weak scaling:** `total_N = (1<<20) × np = 1,048,576 × np` (1M keys/rank)
- Driver: `../../run_sample_sort_scaling.sh all`
- Topologies sourced from `../../platforms/<np>/*.xml`
- Timings are SimGrid-simulated wall time on rank 0, fenced with
  `MPI_Barrier` + `MPI_Wtime` at start and end. All runs verified globally
  sorted via `--verify`.

## Strong scaling (TOTAL_N = 16M for every np)

| np  | torus  | fat_tree | hypercube | dragonfly | ring       |
|----:|-------:|---------:|----------:|----------:|-----------:|
|  16 | 0.0100 |   0.0106 |    0.0121 |    0.0172 |     0.0136 |
|  32 | 0.0054 |   0.0055 |    0.0068 |    0.0128 |     0.0091 |
|  64 | 0.0040 |   0.0053 |    0.0049 |    0.0079 |     0.0127 |
| 128 | **0.0022** | 0.0037 | 0.0027 |    0.0041 |     0.0115 |
| 256 | 0.0069 |   0.0041 |    0.0054 |    0.0092 | **0.0826** |

(seconds; **bold** = standout)

## Weak scaling (per-rank = 1M keys; total = 1M × np)

| np  | torus  | fat_tree   | hypercube  | dragonfly | ring       |
|----:|-------:|-----------:|-----------:|----------:|-----------:|
|  16 | 0.0105 |     0.0115 |     0.0123 |    0.0170 |     0.0146 |
|  32 | 0.0094 |     0.0105 |     0.0147 |    0.0264 |     0.0182 |
|  64 | 0.0178 |     0.0183 |     0.0197 |    0.0293 |     0.0420 |
| 128 | 0.0111 |     0.0161 |     0.0301 |    0.0345 |     0.0544 |
| 256 | 0.0488 | **0.1488** | **0.0342** |    0.1018 | **0.1868** |

## Findings

### Predicted: ring catastrophe at scale

Strong np=256 has ring at `0.083 s` while every other topology clusters
around `0.004–0.009 s` — a **10–20× gap**. Below np=256 ring is bad but
not catastrophic; the gap *widens* with scale because ring's bisection
bandwidth is `O(1)` while the all-to-all volume grows. Textbook prediction
confirmed.

### Surprise #1 — strong-scaling sweet spot at np=128, then performance regresses

Strong scaling is supposed to be monotonic in `np`: more processes, faster.
We see exactly that through np=128 (torus: `0.010 → 0.0054 → 0.0040 → 0.0022`),
then *worse* at np=256 (torus rebounds to `0.0069`). This is the signature
of all-to-all message overhead going `O(p²)`: at np=256 there are ~65k
message pairs, each tiny, and per-message latency starts dominating.
The crossover point between bandwidth-bound and latency-bound regimes
is between np=128 and np=256 for these problem sizes.

### Surprise #2 — fat tree underperforms at np=256 weak

The simple "fat tree always wins all-to-all" intuition fails at scale.
At weak np=256:

- fat_tree: `0.149 s`
- hypercube: `0.034 s` — **4× faster than fat tree**
- ring: `0.187 s` — only marginally worse than fat tree

Possible causes worth investigating before writing the fat-tree
optimization variant:

- Link bandwidths in `../../platforms/256/fat_tree.xml` may saturate at
  this rank count (check the XML; the topology was hand-authored, not
  generator-produced)
- SMPI's collective selector (`--cfg=smpi/coll-selector:ompi`) picks an
  OpenMPI-style alltoall implementation; that algorithm may interact
  poorly with the simulated fat-tree topology — try `--cfg=smpi/coll-selector:mpich`
  for a cross-check
- Hypercube's `log p`-step recursive structure scales gracefully even when
  raw bisection bandwidth isn't dominant — at this scale, diameter
  (number of hops) may matter more than bisection (cross-section bandwidth)

This is the most interesting result. It elevates the project from
"confirms theory" to "investigates theory."

## Per-topology baselines (targets for Phase 3 optimization)

Optimized variants need to beat these numbers, or explain why they can't:

| Topology  | Strong np=256 | Weak np=256 | Optimization angle                              |
|-----------|--------------:|------------:|-------------------------------------------------|
| ring      |        0.0826 |      0.1868 | Replace global alltoall with `p-1` neighbor exchanges. **Largest expected gain.** |
| fat_tree  |        0.0041 |      0.1488 | Investigate the weak np=256 surprise first; chunked alltoall or comm/comp overlap |
| dragonfly |        0.0092 |      0.1018 | Hierarchical: intra-group sample sort, then inter-group redistribution |
| torus     |        0.0069 |      0.0488 | Row-then-column sub-alltoalls on a 2D process grid |
| hypercube |        0.0054 |      0.0342 | Recursive halving across `log p` hypercube dimensions |

## Next steps (Phase 3)

Suggested order (biggest expected gain first → most dramatic chart story):

1. `optimizations/sample_sort_ring.cpp` — pairwise neighbor exchange
2. `optimizations/sample_sort_hypercube.cpp` — recursive halving
3. `optimizations/sample_sort_torus.cpp` — row+column phases
4. `optimizations/sample_sort_dragonfly.cpp` — hierarchical (group-local + inter-group)
5. `optimizations/sample_sort_fat_tree.cpp` — last; only if a real win
   exists once the np=256 weak surprise is understood

After each new variant, the existing
`../../run_sample_sort_scaling.sh` driver picks it up automatically
(missing-binary skip logic). Compare same-topology speedup vs. the agnostic
numbers in this file.
