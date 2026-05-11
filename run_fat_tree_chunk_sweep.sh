#!/bin/bash
#
# Chunk-count sweep for the fat_tree-opt sample-sort variant at weak np=256.
# Defends the headline 5.27x weak-scaling result against "why 4 chunks?"
#
# Each (chunks, rep) is independently restartable: the CSV is appended-to,
# and a row that already exists with a valid time is skipped.
#
# Usage:
#   ./run_fat_tree_chunk_sweep.sh
#
# Output:
#   results/chunk_sweep/fat_tree_np256_weak.csv  (variant,chunks,rep,seed,time_s)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NP=256
PER_RANK=$((1024 * 1024))                # 1M keys per rank
TOTAL_N=$((PER_RANK * NP))               # 256M keys total
PLATFORM="platforms/$NP/fat_tree.xml"
OUTDIR="results/chunk_sweep"
OUTFILE="$OUTDIR/fat_tree_np256_weak.csv"

CHUNK_VALUES=(1 2 4 8 16)
REPS=(1 2 3)

if [ ! -f "$PLATFORM" ]; then
    echo "Error: $PLATFORM not found" >&2
    exit 1
fi
if [ ! -f bin/sample_sort/optimizations/sample_sort_fat_tree ]; then
    echo "Error: bin/sample_sort/optimizations/sample_sort_fat_tree not built" >&2
    exit 1
fi
if [ ! -f bin/sample_sort/agnostic/sample_sort ]; then
    echo "Error: bin/sample_sort/agnostic/sample_sort not built" >&2
    exit 1
fi

mkdir -p "$OUTDIR"

if [ ! -f "$OUTFILE" ]; then
    echo "variant,chunks,rep,seed,time_s" > "$OUTFILE"
fi

# Skip-if-already-present check: row with this (variant, chunks, rep) already
# has a valid floating-point time_s.
already_done() {
    local variant="$1" chunks="$2" rep="$3"
    awk -F, -v v="$variant" -v c="$chunks" -v r="$rep" '
        NR>1 && $1==v && $2==c && $3==r &&
        $5 ~ /^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$/ {found=1; exit}
        END {exit !found}
    ' "$OUTFILE"
}

run_one() {
    local variant="$1" binary="$2" chunks="$3" rep="$4" seed="$5"

    if already_done "$variant" "$chunks" "$rep"; then
        printf "  %-14s chunks=%-3s rep=%d  (skip; already in CSV)\n" "$variant" "$chunks" "$rep"
        return 0
    fi

    local extra=""
    if [ "$variant" = "fat_tree-opt" ]; then
        extra="--chunks=$chunks"
    fi

    printf "  %-14s chunks=%-3s rep=%d seed=%d ... " "$variant" "$chunks" "$rep" "$seed"

    local OUTPUT
    OUTPUT=$(smpirun -np "$NP" \
                     -platform "$PLATFORM" \
                     --cfg=smpi/host-speed:auto \
                     --cfg=smpi/coll-selector:ompi \
                     --cfg=smpi/simulate-computation:0 \
                     --cfg=smpi/alltoall:pair \
                     "$binary" "$TOTAL_N" --seed="$seed" $extra 2>&1) || true

    local TIME
    TIME=$(echo "$OUTPUT" | grep -oP 'Sample sort time: \K[\d.eE+-]+' | tail -1)

    if [ -z "$TIME" ] || ! [[ "$TIME" =~ ^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
        echo "FAILED"
        echo "    last 5 lines of output:"
        echo "$OUTPUT" | tail -5 | sed 's/^/      /'
        return 1
    fi
    echo "$TIME s"
    echo "$variant,$chunks,$rep,$seed,$TIME" >> "$OUTFILE"
}

echo "======== Fat-tree chunk-count sweep ========"
echo "  np=$NP  per_rank=$PER_RANK  TOTAL_N=$TOTAL_N"
echo "  platform=$PLATFORM"
echo "  output=$OUTFILE"
echo "  smpirun flags: --cfg=smpi/simulate-computation:0 --cfg=smpi/alltoall:pair (+ host-speed:auto, coll-selector:ompi)"
echo

echo "--- fat_tree-opt: sweep K in ${CHUNK_VALUES[*]}, ${#REPS[@]} reps each ---"
for chunks in "${CHUNK_VALUES[@]}"; do
    for rep in "${REPS[@]}"; do
        run_one "fat_tree-opt" "bin/sample_sort/optimizations/sample_sort_fat_tree" "$chunks" "$rep" "$rep"
    done
done

echo
echo "--- agnostic reference: ${#REPS[@]} reps ---"
for rep in "${REPS[@]}"; do
    # chunks=0 is sentinel meaning "no chunking applies" (agnostic has no chunk param)
    run_one "agnostic" "bin/sample_sort/agnostic/sample_sort" "0" "$rep" "$rep"
done

echo
echo "Done. Raw timings in $OUTFILE"
echo "Total rows: $(($(wc -l < "$OUTFILE") - 1))"
