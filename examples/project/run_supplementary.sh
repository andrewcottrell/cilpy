#!/usr/bin/env bash
# run_supplementary.sh — queues every supplementary run in priority order,
# so a partial finish is still usable.
#
# Usage (from the cilpy repo root), inside tmux/screen:
#   bash examples/project/run_supplementary.sh 2>&1 | tee supplementary.log
#
# Each step writes to ./out, which is then moved to
# examples/project/raw/out_supp_<set>_<part>[_taut<N>]. A step whose folder
# already exists is skipped, so the script can be restarted after a stop.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
RUNS="${RUNS:-30}"
ITERS="${ITERS:-1000}"
WORKERS="${WORKERS:-8}"

RAW_DIR="$SCRIPT_DIR/raw"
mkdir -p "$RAW_DIR"

if [ -e out ]; then
    echo "out/ already exists in $REPO_ROOT. Move it aside first." >&2
    exit 1
fi

echo ">>> Running check_install.py ..."
$PYTHON examples/project/check_install.py

step() {  # step <set> <part> [tau_t]
    local set_name="$1" part="$2" tau="${3:-}"
    local dest="$RAW_DIR/out_supp_${set_name}_${part}${tau:+_taut$tau}"
    if [ -e "$dest" ]; then
        echo ">>> skipping $set_name $part ${tau:+tau_t=$tau}: $dest exists"
        return
    fi
    echo ""
    echo "============================================================"
    echo "  $set_name $part ${tau:+tau_t=$tau}    $(date)"
    echo "============================================================"
    $PYTHON examples/project/run_supplementary.py \
        --set "$set_name" --part "$part" ${tau:+--tau-t "$tau"} \
        --runs "$RUNS" --iters "$ITERS" --workers "$WORKERS"
    mv out "$dest"
    echo "  Moved out/ -> $dest"
}

step rho0    static
step rho0    dynamic 25
step sampled static
step sampled dynamic 25
step rho0    dynamic 10
step rho0    dynamic 50
step sampled dynamic 10
step sampled dynamic 50

echo ""
echo "ALL SUPPLEMENTARY RUNS COMPLETE   $(date)"
echo "Tables:   examples/project/results/results_supp_*.csv"
echo "Raw runs: examples/project/raw/out_supp_*/"
