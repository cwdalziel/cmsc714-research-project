#!/bin/bash

# Usage: ./run_benchmarks.sh <binary>
# Example: ./run_benchmarks.sh bin/alltoall

if [ -z "$1" ]; then
    echo "Usage: $0 <binary>"
    exit 1
fi
FLOPS=
BINARY=$1
NP=64
PLATFORM_DIR=platforms
RESULTS_DIR=results

# Validate binary exists
if [ ! -f "$BINARY" ]; then
    echo "Error: binary '$BINARY' not found"
    exit 1
fi

# Validate platform directory exists
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
                     --cfg=smpi/host-speed:8Gf \
                     --cfg=smpi/display-timing:yes \
                     --cfg=smpi/coll-selector:ompi \
                     "$BINARY" 2>&1)

    # Extract the timing line from the output
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