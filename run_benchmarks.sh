#!/bin/bash

# Usage: ./run_benchmarks.sh <binary> <np>
# Example: ./run_benchmarks.sh bin/stencil 64

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <binary> <np>"
    exit 1
fi

BINARY=$1
NP=$2
FLOPS=1Gf
PLATFORM_DIR=platforms/$NP   # look for platform files for this node count
RESULTS_DIR=results/$NP      # separate results folder per node count

if [ ! -f "$BINARY" ]; then
    echo "Error: binary '$BINARY' not found"
    exit 1
fi

if [ ! -d "$PLATFORM_DIR" ]; then
    echo "Error: platform directory '$PLATFORM_DIR' not found"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

BENCHMARK=$(basename "$BINARY")
OUTFILE="$RESULTS_DIR/${BENCHMARK}_results.csv"

echo "topology,time_s" > "$OUTFILE"
echo "Running $BENCHMARK across all topologies with $NP ranks..."
echo "-----------------------------------------------------------"

for PLATFORM in "$PLATFORM_DIR"/*.xml; do
    TOPOLOGY=$(basename "$PLATFORM" .xml)
    printf "  %-20s ... " "$TOPOLOGY"
    OUTPUT=$(smpirun -np "$NP" \
                     -platform "$PLATFORM" \
                     --cfg=smpi/host-speed:auto \
                     --cfg=smpi/coll-selector:ompi \
                     "$BINARY" 2>&1)

    TIME=$(echo "$OUTPUT" | grep -oP '[\d.e+-]+(?= s)' | tail -1)

    if [ -z "$TIME" ]; then
        echo "FAILED"
        echo "$TOPOLOGY,ERROR" >> "$OUTFILE"
        echo "    Output was: $OUTPUT"
    else
        echo "$TIME s"
        echo "$TOPOLOGY,$TIME" >> "$OUTFILE"
    fi
done

echo "-----------------------------------------------------------"
echo "Results saved to $OUTFILE"