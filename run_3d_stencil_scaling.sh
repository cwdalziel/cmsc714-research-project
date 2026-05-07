#!/bin/bash
#
# Run strong and weak scaling benchmarks for every 3d_stencil binary by calling
# run_benchmarks.sh for each (binary, np) pair.
#
# Strong scaling: fixed cubic grid N^3 for all np (default N=512).
# Weak scaling:   fixed per-rank volume via NX,NY fixed and NZ = np * WEAK_Z_PER_RANK
#                 (keeps ring's NZ % np == 0 and applies identical controls to all binaries).
#
# CSV output layout:
#   results/strong/<np>/<basename>_results.csv
#   results/weak/<np>/<basename>_results.csv
#
# Requires: run_benchmarks.sh in the same directory; binaries under bin/3d_stencil/;
#           platforms/<np>/*.xml for each np.
#
# Usage: ./run_3d_stencil_scaling.sh [strong|weak|all]
#        Default: all  (runs strong then weak; strong alone is many smpirun batches:
#        len(STENCIL_BINS) × len(NP_LIST) calls to run_benchmarks.sh)
#
# Loop order is by np first, then binary: at each process count you run all stencil
# variants before increasing np (less confusing than one full np-sweep per binary).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_BENCH="$SCRIPT_DIR/run_benchmarks.sh"
BIN_DIR="bin/3d_stencil"

STENCIL_BINS=(
    "3d_stencil_agnostic_no_reorder"
    # "3d_stencil_torus"
    # "3d_stencil_hypercube"
    # "3d_stencil_dragonfly"
    # "3d_stencil_fat_tree"
    # "3d_stencil_ring"
)

NP_LIST=(16 32 64 128 256)

# Strong scaling: same global cubic grid at every np.
# Bumped default to 512 so timings are less dominated by very small-message effects.
STRONG_N="${STRONG_N:-512}"

# Weak scaling controls (consistent across all implementations):
# Per-rank volume = WEAK_NX * WEAK_NY * WEAK_Z_PER_RANK
WEAK_NX="${WEAK_NX:-192}"
WEAK_NY="${WEAK_NY:-192}"
WEAK_Z_PER_RANK="${WEAK_Z_PER_RANK:-4}"

weak_grid_for_np() {
    local np="$1"
    if [ "$np" -le 0 ]; then
        echo "run_3d_stencil_scaling.sh: invalid np=$np" >&2
        exit 1
    fi
    local nz=$(( np * WEAK_Z_PER_RANK ))
    echo "$WEAK_NX $WEAK_NY $nz"
}

run_strong() {
    export RESULTS_SUBDIR=strong
    echo "======== Strong scaling (NX=NY=NZ=$STRONG_N for all np) ========"
    echo "Batches: ${#NP_LIST[@]} np × ${#STENCIL_BINS[@]} binaries (each batch = all topologies in platforms/<np>/)."
    for np in "${NP_LIST[@]}"; do
        for bin in "${STENCIL_BINS[@]}"; do
            local path="$BIN_DIR/$bin"
            if [ ! -f "$path" ]; then
                echo "Skip missing binary: $path" >&2
                continue
            fi
            echo ""
            echo "--- STRONG  np=$np  $bin  grid=$STRONG_N ---"
            bash "$RUN_BENCH" "$path" "$np" "$STRONG_N"
        done
    done
    unset RESULTS_SUBDIR
}

run_weak() {
    export RESULTS_SUBDIR=weak
    echo "======== Weak scaling (constant work/rank: NX=$WEAK_NX NY=$WEAK_NY NZ=np*$WEAK_Z_PER_RANK) ========"
    echo "Batches: ${#NP_LIST[@]} np × ${#STENCIL_BINS[@]} binaries."
    for np in "${NP_LIST[@]}"; do
        for bin in "${STENCIL_BINS[@]}"; do
            local path="$BIN_DIR/$bin"
            if [ ! -f "$path" ]; then
                echo "Skip missing binary: $path" >&2
                continue
            fi
            local nx ny nz
            read -r nx ny nz <<< "$(weak_grid_for_np "$np")"
            echo ""
            echo "--- WEAK  np=$np  $bin  grid=${nx}x${ny}x${nz} ---"
            bash "$RUN_BENCH" "$path" "$np" "$nx" "$ny" "$nz"
        done
    done
    unset RESULTS_SUBDIR
}

if [ ! -f "$RUN_BENCH" ]; then
    echo "Error: $RUN_BENCH not found" >&2
    exit 1
fi

MODE="${1:-all}"
case "$MODE" in
    strong) run_strong ;;
    weak)   run_weak ;;
    all)
        echo "Running BOTH studies: strong first (long), then weak. For only weak: $0 weak"
        echo ""
        run_strong
        echo ""
        echo "======== Strong finished; starting weak scaling ========"
        echo ""
        run_weak
        ;;
    *)
        echo "Usage: $0 [strong|weak|all]" >&2
        exit 1
        ;;
esac

echo ""
echo "Done. Results under results/strong/<np>/ and results/weak/<np>/ (when both ran)."
