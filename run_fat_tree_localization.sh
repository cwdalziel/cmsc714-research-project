#!/bin/bash
#
# Localization experiment: figure out WHERE the 5.27x weak-np=256 speedup
# came from. Vary one knob at a time:
#   - simulate-computation: ON (default) vs OFF
#   - alltoallv algorithm: OMPI-selector-default vs forced "pair"
#
# Compares agnostic vs fat_tree-opt K=4 in each cell. If forcing the alltoallv
# algorithm collapses the gap, the 5.27x came from OMPI selector picking
# different algorithms for one big alltoallv vs four small ones.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NP=256
TOTAL_N=$((1024*1024*256))   # weak: 1M keys/rank * 256
PLATFORM="platforms/$NP/fat_tree.xml"
OUTDIR="results/chunk_sweep"
OUTFILE="$OUTDIR/fat_tree_localization.csv"
mkdir -p "$OUTDIR"

if [ ! -f "$OUTFILE" ]; then
    echo "condition,variant,sim_compute,alltoallv,time_s" > "$OUTFILE"
fi

run_one() {
    local condition="$1"
    local variant="$2"
    local sim_compute="$3"   # "on" or "off"
    local alltoallv="$4"     # "default" or "pair"
    local binary="$5"
    local prog_extra="$6"

    # Skip if already measured
    if awk -F, -v c="$condition" -v v="$variant" \
        'NR>1 && $1==c && $2==v && $5 ~ /^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$/ {ok=1; exit} END{exit !ok}' \
        "$OUTFILE"; then
        printf "  %-30s %-15s (skip; in CSV)\n" "$condition" "$variant"
        return 0
    fi

    local extra_smpi=""
    [ "$sim_compute" = "off" ] && extra_smpi="$extra_smpi --cfg=smpi/simulate-computation:0"
    [ "$alltoallv" = "pair" ] && extra_smpi="$extra_smpi --cfg=smpi/alltoallv:pair"

    printf "  %-30s %-15s ... " "$condition" "$variant"
    local start=$(date +%s)
    local OUTPUT
    OUTPUT=$(smpirun -np "$NP" -platform "$PLATFORM" \
                     --cfg=smpi/host-speed:auto \
                     --cfg=smpi/coll-selector:ompi \
                     $extra_smpi \
                     "$binary" "$TOTAL_N" --seed=1 $prog_extra 2>&1) || true
    local elapsed=$(( $(date +%s) - start ))

    local TIME
    TIME=$(echo "$OUTPUT" | grep -oP 'Sample sort time: \K[\d.eE+-]+' | tail -1)

    if [ -z "$TIME" ] || ! [[ "$TIME" =~ ^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
        echo "FAILED (wall=${elapsed}s)"
        echo "$OUTPUT" | tail -3 | sed 's/^/      /'
        return 1
    fi
    echo "$TIME s (wall=${elapsed}s)"
    echo "$condition,$variant,$sim_compute,$alltoallv,$TIME" >> "$OUTFILE"
}

AGN=bin/sample_sort/agnostic/sample_sort
FT4="bin/sample_sort/optimizations/sample_sort_fat_tree"
FT4_ARGS="--chunks=4"

echo "============ Fat-tree 5.27x localization, np=$NP weak ============"
echo

# Cell A: original methodology (compute sim ON, OMPI selector). Should reproduce ~149/28 ms.
echo "[A] compute-sim ON, alltoallv: OMPI-default"
run_one "A_compute-on_ompi-default" "agnostic"      on default "$AGN" ""
run_one "A_compute-on_ompi-default" "fat_tree-K4"   on default "$FT4" "$FT4_ARGS"

# Cell B: force alltoallv:pair, compute sim still ON. Tests if the gap survives when both use same algo.
echo
echo "[B] compute-sim ON, alltoallv: pair (forced)"
run_one "B_compute-on_alltoallv-pair" "agnostic"    on pair "$AGN" ""
run_one "B_compute-on_alltoallv-pair" "fat_tree-K4" on pair "$FT4" "$FT4_ARGS"

# Cell C: compute-sim OFF, OMPI selector default. (chunk sweep used the methodology of cell D, not C.)
echo
echo "[C] compute-sim OFF, alltoallv: OMPI-default"
run_one "C_compute-off_ompi-default" "agnostic"    off default "$AGN" ""
run_one "C_compute-off_ompi-default" "fat_tree-K4" off default "$FT4" "$FT4_ARGS"

# Cell D: chunk-sweep methodology — already known from chunk sweep (12.1 ms ≡ agnostic ≡ K=1)
# Skip: would just duplicate existing results/chunk_sweep/fat_tree_np256_weak.csv data.
echo
echo "[D] compute-sim OFF, alltoallv: pair  (skipped — duplicates chunk sweep K=1 ≈ agnostic ≈ 12.1 ms)"

echo
echo "Done. Results in $OUTFILE"
echo
echo "=== Compact view ==="
awk -F, '
    NR>1 {
        gap[$1] = ""
        if ($2 == "agnostic")    agn[$1]   = $5
        if ($2 == "fat_tree-K4") chunk[$1] = $5
    }
    END {
        printf "%-32s %12s %12s %10s\n", "condition", "agnostic", "K=4", "ratio"
        for (c in gap) {
            if (agn[c] != "" && chunk[c] != "")
                printf "%-32s %12.4f %12.4f %9.2fx\n", c, agn[c], chunk[c], agn[c]/chunk[c]
            else
                printf "%-32s %12s %12s %10s\n", c, (agn[c]?agn[c]:"-"), (chunk[c]?chunk[c]:"-"), "?"
        }
    }
' "$OUTFILE" | sort
