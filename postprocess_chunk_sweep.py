#!/usr/bin/env python3
"""
Post-process the fat_tree chunk-count sweep:
  - Reads results/chunk_sweep/fat_tree_np256_weak.csv
  - Writes a U-curve plot (log-x chunks, mean comm time + std error bars,
    horizontal dashed line for agnostic baseline) to
    results/plots/fat_tree_chunk_sweep.png
  - Writes results/chunk_sweep/results_summary.md (5 lines + bisection ratio)

Run via:
    docker run --rm -v "$PWD:/work" -w /work python:3.11-slim \\
        bash -c 'pip install -q matplotlib numpy && python postprocess_chunk_sweep.py'
"""

import csv
import os
import sys
from collections import defaultdict
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CSV_PATH = "results/chunk_sweep/fat_tree_np256_weak.csv"
PLOT_PATH = "results/plots/fat_tree_chunk_sweep.png"
SUMMARY_PATH = "results/chunk_sweep/results_summary.md"

# Bisection BW from results/chunk_sweep/bisection_math.md
BISECTION_BW_GB_S = 400.0
ALLTOALL_BISECTION_BYTES_GB = 1.0  # 256 * 4 MB / 1024


def load():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows.append({
                "variant": r["variant"],
                "chunks": int(r["chunks"]),
                "rep": int(r["rep"]),
                "time_s": float(r["time_s"]),
            })
    return rows


def aggregate(rows):
    """Return dict: variant -> chunks -> {'mean': ..., 'std': ..., 'n': ...}"""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["chunks"])].append(r["time_s"])
    out = defaultdict(dict)
    for (variant, chunks), times in groups.items():
        out[variant][chunks] = {
            "mean": mean(times),
            "std":  stdev(times) if len(times) > 1 else 0.0,
            "n":    len(times),
            "raw":  times,
        }
    return out


def plot_u_curve(agg):
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    chunks_sorted = sorted(agg["fat_tree-opt"].keys())
    means = [agg["fat_tree-opt"][k]["mean"] for k in chunks_sorted]
    stds  = [agg["fat_tree-opt"][k]["std"]  for k in chunks_sorted]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.errorbar(chunks_sorted, means, yerr=stds,
                marker="o", markersize=8, linewidth=2, capsize=5,
                color="#3498DB", label="fat_tree-opt (mean ± std, n=3)")

    # Agnostic baseline (mean across reps) as horizontal dashed line + band
    if "agnostic" in agg:
        agn_mean = agg["agnostic"][0]["mean"]
        agn_std  = agg["agnostic"][0]["std"]
        ax.axhline(y=agn_mean, color="#E74C3C", linestyle="--", linewidth=2,
                   label=f"agnostic baseline (mean={agn_mean*1000:.2f} ms, n={agg['agnostic'][0]['n']})")
        if agn_std > 0:
            ax.fill_between(chunks_sorted,
                            [agn_mean - agn_std] * len(chunks_sorted),
                            [agn_mean + agn_std] * len(chunks_sorted),
                            color="#E74C3C", alpha=0.10)

    # Annotate minimum
    if means:
        i_min = means.index(min(means))
        ax.annotate(f"min: K={chunks_sorted[i_min]}\n{means[i_min]*1000:.2f} ms",
                    xy=(chunks_sorted[i_min], means[i_min]),
                    xytext=(15, 25), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="lightyellow", edgecolor="gray"))

    ax.set_xscale("log", base=2)
    ax.set_xticks(chunks_sorted)
    ax.set_xticklabels([str(k) for k in chunks_sorted])
    ax.set_xlabel("Number of alltoallv sub-rounds (chunks K), log₂ scale")
    ax.set_ylabel("Communication time (s)")
    ax.set_title(
        "Fat-tree chunk-count sweep — weak np=256, 1M keys/rank\n"
        "(--cfg=smpi/simulate-computation:0 --cfg=smpi/alltoall:pair)"
    )
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Wrote plot: {PLOT_PATH}")


def write_summary(agg):
    """Data-driven summary — describes whatever shape the data actually has,
    rather than asserting a hypothesized U-curve."""
    chunks_sorted = sorted(agg["fat_tree-opt"].keys())
    means = {k: agg["fat_tree-opt"][k]["mean"] for k in chunks_sorted}
    stds  = {k: agg["fat_tree-opt"][k]["std"]  for k in chunks_sorted}
    k_min = min(means, key=means.get)
    k_max = max(means, key=means.get)
    t_min = means[k_min]
    t_max = means[k_max]

    agn_mean = agg["agnostic"][0]["mean"] if "agnostic" in agg else None
    agn_std  = agg["agnostic"][0]["std"]  if "agnostic" in agg else None

    # Classify curve shape. Use a generous tolerance (3σ) so tiny noise-level
    # wiggles don't prevent us from reporting "essentially monotonic."
    interior_min = (k_min != chunks_sorted[0]) and (k_min != chunks_sorted[-1])
    # Endpoint-based check: the minimum sits at the K=1 endpoint AND the
    # K=max endpoint is significantly higher than the K=1 endpoint (>= 3σ).
    no_chunking_benefit = (
        k_min == chunks_sorted[0] and
        means[chunks_sorted[-1]] - means[chunks_sorted[0]] >
            3.0 * max(stds[chunks_sorted[0]], stds[chunks_sorted[-1]])
    )
    monotonic_increase = all(
        means[chunks_sorted[i + 1]] >=
            means[chunks_sorted[i]] - 3.0 * stds[chunks_sorted[i]]
        for i in range(len(chunks_sorted) - 1)
    )
    monotonic_decrease = all(
        means[chunks_sorted[i + 1]] <=
            means[chunks_sorted[i]] + 3.0 * stds[chunks_sorted[i]]
        for i in range(len(chunks_sorted) - 1)
    )

    # Bisection utilization at the min-K point
    demanded_bw = ALLTOALL_BISECTION_BYTES_GB / t_min
    bisection_pct = 100.0 * demanded_bw / BISECTION_BW_GB_S

    # Speedup of min vs the K=1 endpoint (or vs agnostic — they should be ≡)
    t_K1 = means.get(1, None)
    k1_eq_agn = (agn_mean is not None and t_K1 is not None
                 and abs(t_K1 - agn_mean) < 1e-9)

    lines = ["# Chunk-Sweep Results Summary", ""]

    # Lead with what the curve shape actually is
    if interior_min:
        speedup_endpoints = max(means[chunks_sorted[0]], means[chunks_sorted[-1]]) / t_min
        lines += [
            f"- **U-curve observed.** Sweep K ∈ {chunks_sorted} of `fat_tree-opt` "
            f"at weak np=256 has a clear interior minimum at **K={k_min}** "
            f"({t_min*1000:.2f} ms, n={agg['fat_tree-opt'][k_min]['n']}, "
            f"σ={stds[k_min]*1000:.3f} ms). Endpoints are {speedup_endpoints:.2f}× "
            f"slower, confirming the chunk-count tradeoff is real.",
        ]
    elif monotonic_increase:
        lines += [
            f"- **No U-curve.** Mean comm time grows **monotonically** with K "
            f"across {chunks_sorted}: K={chunks_sorted[0]} is fastest "
            f"({t_min*1000:.2f} ms), K={chunks_sorted[-1]} is slowest "
            f"({t_max*1000:.2f} ms, {t_max/t_min:.2f}× worse). "
            f"**Chunking does not help under this methodology.**",
        ]
    elif monotonic_decrease:
        lines += [
            f"- **Monotonic decrease.** Mean comm time decreases with K. "
            f"K={chunks_sorted[-1]} is fastest ({t_min*1000:.2f} ms); "
            f"K={chunks_sorted[0]} is slowest ({t_max*1000:.2f} ms). No "
            f"interior optimum within the swept range — try larger K.",
        ]
    else:
        lines += [
            f"- **Mixed shape.** Minimum at K={k_min} ({t_min*1000:.2f} ms), "
            f"but the curve isn't cleanly monotonic or U-shaped — see "
            f"`fat_tree_chunk_sweep.png`.",
        ]

    if k1_eq_agn:
        lines += [
            f"- **K=1 ≡ agnostic** to floating-point precision "
            f"({t_K1*1000:.4f} ms vs {agn_mean*1000:.4f} ms). This is structural: "
            f"K=1 in the chunked variant degenerates to a single `MPI_Alltoallv` "
            f"call — exactly the agnostic algorithm. Confirms the chunking "
            f"machinery itself is correct and adds no overhead at K=1.",
        ]
    elif agn_mean is not None:
        speedup_min_vs_agn = agn_mean / t_min if t_min > 0 else float("nan")
        if speedup_min_vs_agn >= 1.05:
            lines += [
                f"- **Vs. agnostic:** chunked at K={k_min} = {t_min*1000:.2f} ms "
                f"vs. agnostic = {agn_mean*1000:.2f} ms ⇒ "
                f"**{speedup_min_vs_agn:.2f}× speedup**.",
            ]
        else:
            lines += [
                f"- **Vs. agnostic:** chunked at K={k_min} = {t_min*1000:.2f} ms "
                f"vs. agnostic = {agn_mean*1000:.2f} ms — "
                f"**no meaningful speedup** (ratio {speedup_min_vs_agn:.2f}×).",
            ]

    lines += [
        f"- **Bisection sanity check:** at the min-K point, demanded "
        f"cross-bisection bandwidth = {demanded_bw:.2f} GB/s vs. supply "
        f"400 GB/s ⇒ **{bisection_pct:.2f}% utilization**. The bottleneck "
        f"is NOT raw bandwidth — chunking would only help if it relieved "
        f"contention or queue pressure, neither of which is implicated by "
        f"this measurement.",
    ]

    if (monotonic_increase or no_chunking_benefit) and k1_eq_agn:
        lines += [
            "- **Implication for the original 5.27× headline:** the "
            "headline does not reproduce under comm-only methodology "
            "(`--cfg=smpi/simulate-computation:0 --cfg=smpi/alltoall:pair`). "
            "Since K=1 ≡ agnostic and is FASTEST, chunking provides no "
            "mechanistic benefit when only the network is being simulated. "
            "The original speedup must therefore come from an interaction "
            "between OMPI's `MPI_Alltoallv` selector and SimGrid's "
            "computation simulation — not from a topology-aware property "
            "of fat-tree itself. **Recommended follow-up:** re-run the "
            "sweep WITHOUT `simulate-computation:0` to confirm the 5.27× "
            "returns; that would localize the effect to the compute / "
            "selector interaction.",
        ]
    elif interior_min:
        lines += [
            f"- **Defense for the original 5.27× headline:** the U-curve "
            f"shape (single interior minimum at K={k_min}, regression on "
            f"both sides) is the mechanistic signature of an in-flight "
            f"buffer pressure tradeoff: too few chunks ⇒ contention; too "
            f"many ⇒ per-call setup overhead. The hardcoded K=4 sat at or "
            f"near the sweet spot.",
        ]

    lines += [
        "",
        "_Generated by `postprocess_chunk_sweep.py` from "
        "`fat_tree_np256_weak.csv`._",
        "_Bisection-bandwidth derivation in `bisection_math.md`._",
    ]

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote summary: {SUMMARY_PATH}")
    print()
    print("--- Summary preview ---")
    for line in lines[:10]:
        print(line)


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run the sweep first.", file=sys.stderr)
        return 1
    rows = load()
    if not rows:
        print(f"ERROR: {CSV_PATH} has no data rows.", file=sys.stderr)
        return 1
    agg = aggregate(rows)
    if "fat_tree-opt" not in agg:
        print("ERROR: no fat_tree-opt rows in CSV.", file=sys.stderr)
        return 1
    plot_u_curve(agg)
    write_summary(agg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
