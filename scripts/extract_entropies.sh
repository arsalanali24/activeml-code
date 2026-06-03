#!/bin/bash
# Run this after all three jobs finish
# Usage: bash extract_entropies.sh

RESULTS_DIR=~/activeml/results
mkdir -p $RESULTS_DIR

for mult in 5 3 1; do
    logfile=$(ls ~/activeml/logs/fecl4_m${mult}_*.out 2>/dev/null | tail -1)

    if [ -z "$logfile" ]; then
        echo "mult=${mult}: no log file yet"
        continue
    fi

    echo "=== Multiplicity ${mult} ==="
    echo "Log: $logfile"

    # Check if job succeeded
    grep -c "Happy landing" $logfile > /dev/null 2>&1
    if grep -q "Happy landing" $logfile; then
        echo "Status: SUCCESS"
    else
        echo "Status: FAILED or still running"
        grep -i "error\|fatal\|abort" $logfile | tail -5
        continue
    fi

    # Extract DMRG energy
    echo "DMRG energy:"
    grep "DMRGSCF.*Total energy" $logfile | tail -1

    # Extract s1 entropies
    echo "Single-site entropies:"
    grep -A 50 "Single-orbital entropy" $logfile | \
        grep -E "^\s+[0-9]" | head -20

    echo ""
done
