#!/usr/bin/env bash
# run_all_overnight.sh — One command to launch every experiment campaign.
#
# Usage (from the cilpy repo root):
#   nohup bash examples/project/run_all_overnight.sh > overnight.log 2>&1 &
#
# Or inside tmux/screen:
#   bash examples/project/run_all_overnight.sh 2>&1 | tee overnight.log
#
# The script runs sequentially (no parallelism between campaigns) so that
# output directories don't collide. Total wall time: many hours.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
RUNS=30
ITERS=1000
WORKERS="${WORKERS:-8}"
# QUICK=1 bash examples/project/run_all_overnight.sh  -> 3 runs x 100 iterations
# of everything, as an end-to-end smoke test (minutes, not hours).
if [ "${QUICK:-0}" = "1" ]; then
    RUNS=3
    ITERS=100
fi

# Full campaign output (with per-iteration fronts) is kept here, out of git.
# The runner always writes to ./out; each campaign is moved here afterwards.
RAW_DIR="$SCRIPT_DIR/raw"

# Keep the 22 Sept campaign's raw output (shared penalty swarm, rho = 1):
# move it aside once instead of overwriting it.
if [ -d "$RAW_DIR/out_static" ] && [ ! -d "$SCRIPT_DIR/raw_22sept" ]; then
    mv "$RAW_DIR" "$SCRIPT_DIR/raw_22sept"
    echo "Moved the old raw/ to raw_22sept/"
fi
mkdir -p "$RAW_DIR"
rm -rf out

echo "============================================================"
echo "  OVERNIGHT EXPERIMENT CAMPAIGN"
echo "  Started: $(date)"
echo "  Repo:    $REPO_ROOT"
echo "  Python:  $($PYTHON --version 2>&1)"
echo "  Runs:    $RUNS   Iterations: $ITERS"
echo "============================================================"

# -------------------------------------------------------------------
# 0. Sanity checks
# -------------------------------------------------------------------
echo ""
echo ">>> Running check_install.py ..."
$PYTHON examples/project/check_install.py
echo ""
echo ">>> Running pytest (quick) ..."
$PYTHON -m pytest test/ -q --tb=short
echo ""
echo "All checks passed."

# -------------------------------------------------------------------
# 1. Static MO campaign (SOSC)
# -------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  [1/4] STATIC MO CAMPAIGN"
echo "  $(date)"
echo "============================================================"
$PYTHON examples/project/run_static_mo_experiments.py \
    --runs "$RUNS" --iters "$ITERS" --workers "$WORKERS"
rm -rf "$RAW_DIR/out_static"
mv out "$RAW_DIR/out_static"
$PYTHON examples/project/validate_results.py --out-dir "$RAW_DIR/out_static" --runs "$RUNS" || true
$PYTHON examples/project/compare_baselines.py --static-only \
    --ccpso-out-dir "$RAW_DIR/out_static" || true

# -------------------------------------------------------------------
# 2-4. Dynamic campaign at tau_t = 10, 25, 50 (SODC, DOSC, DODC)
# -------------------------------------------------------------------
# The refresh-mode campaign runs every dynamic experiment under both refresh
# policies. Its clean arm IS the main dynamic campaign (same configurations,
# same seeds), so aggregate_campaign.py derives both the main table and the
# paired clean-vs-clear table from it; no separate dynamic run is needed.
STEP=2
for TAU_T in 10 25 50; do
    echo ""
    echo "============================================================"
    echo "  [$STEP/4] DYNAMIC CAMPAIGN  tau_t=$TAU_T"
    echo "  $(date)"
    echo "============================================================"
    $PYTHON examples/project/run_refresh_mode_experiments.py \
        --runs "$RUNS" --iters "$ITERS" --tau-t "$TAU_T" --n-t 10 --workers "$WORKERS"
    REFRESH_DIR="$RAW_DIR/out_refresh_taut${TAU_T}"
    rm -rf "$REFRESH_DIR"
    mv out "$REFRESH_DIR"

    $PYTHON examples/project/aggregate_campaign.py \
        --out-dir "$REFRESH_DIR" --tau-t "$TAU_T"
    $PYTHON examples/project/analyse_dynamic_significance.py \
        --out-dir "$REFRESH_DIR" --tau-t "$TAU_T" || true

    $PYTHON examples/project/run_dnsga2_baseline.py \
        --runs "$RUNS" --n-gen "$ITERS" --tau-t "$TAU_T" --n-t 10
    STEP=$((STEP + 1))
done

$PYTHON examples/project/compare_baselines.py --dynamic-only --tau-t 10 25 50 \
    --ccpso-out-dir "$RAW_DIR/out_refresh_taut{tau}" || true

# -------------------------------------------------------------------
# Stripped copies for version control
# -------------------------------------------------------------------
echo ""
echo ">>> slim_campaign.py ..."
$PYTHON examples/project/slim_campaign.py

echo ""
echo "============================================================"
echo "  ALL CAMPAIGNS COMPLETE"
echo "  Finished: $(date)"
echo "============================================================"
echo ""
echo "Output locations (all under examples/project/):"
echo "  results/results_static_mo.csv                 static table"
echo "  results/comparison_static.csv                 against NSGA-II"
echo "  results/results_dynamic_taut{10,25,50}.csv    main dynamic tables"
echo "  results/results_refresh_taut{10,25,50}.csv    clean against clear, paired"
echo "  results/significance_dynamic_taut{10,25,50}.csv"
echo "  results/results_dnsga2_taut{10,25,50}.csv     DNSGA-II baseline"
echo "  results/comparison_dynamic.csv               against DNSGA-II"
echo "  raw/out_static/, raw/out_refresh_taut*/      per-iteration CSVs (not in git)"
echo "  data/out_*/                                  the same without fronts (in git)"
echo ""
echo "Done."
