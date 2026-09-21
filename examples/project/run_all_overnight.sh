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
WORKERS=8

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
# 1. Static MO campaign
# -------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  [1/10] STATIC MO CAMPAIGN"
echo "  $(date)"
echo "============================================================"
$PYTHON examples/project/run_static_mo_experiments.py \
    --runs "$RUNS" --iters "$ITERS" --workers "$WORKERS"

# -------------------------------------------------------------------
# 2-4. Dynamic campaign at tau_t = 10, 25, 50
# -------------------------------------------------------------------
for TAU_T in 10 25 50; do
    echo ""
    echo "============================================================"
    echo "  DYNAMIC CAMPAIGN  tau_t=$TAU_T"
    echo "  $(date)"
    echo "============================================================"

    # -- Dynamic (main + archive sentries) --
    echo ">>> [tau_t=$TAU_T] run_dynamic_experiements.py ..."
    $PYTHON examples/project/run_dynamic_experiements.py \
        --runs "$RUNS" --iters "$ITERS" --tau-t "$TAU_T" --n-t 10 --workers "$WORKERS"

    # Move output to a tau-specific folder before starting the next tau.
    # The aggregate function already ran inside the script, so the CSV
    # is in the CWD. The per-run files in out/ need to be preserved for
    # aggregate_campaign.py to read later.
    DYNAMIC_DIR="out_dynamic_taut${TAU_T}"
    rm -rf "$DYNAMIC_DIR"
    cp -r out "$DYNAMIC_DIR"
    echo "  Copied out/ -> $DYNAMIC_DIR"

    # -- Refresh-mode ablation --
    echo ">>> [tau_t=$TAU_T] run_refresh_mode_experiments.py ..."
    # Clear out/ so only refresh files are in it.
    rm -rf out
    $PYTHON examples/project/run_refresh_mode_experiments.py \
        --runs "$RUNS" --iters "$ITERS" --tau-t "$TAU_T" --n-t 10 --workers "$WORKERS"

    REFRESH_DIR="out_refresh_taut${TAU_T}"
    rm -rf "$REFRESH_DIR"
    mv out "$REFRESH_DIR"
    echo "  Moved out/ -> $REFRESH_DIR"

    # -- Cross-campaign aggregation (produces both main + paired tables) --
    echo ">>> [tau_t=$TAU_T] aggregate_campaign.py ..."
    $PYTHON examples/project/aggregate_campaign.py \
        --out-dir "$REFRESH_DIR" --tau-t "$TAU_T"

    # -- DNSGA-II baseline --
    echo ">>> [tau_t=$TAU_T] run_dnsga2_baseline.py ..."
    $PYTHON examples/project/run_dnsga2_baseline.py \
        --runs "$RUNS" --tau-t "$TAU_T" --n-t 10
done

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  ALL CAMPAIGNS COMPLETE"
echo "  Finished: $(date)"
echo "============================================================"
echo ""
echo "Output locations:"
echo "  results_static_mo.csv           — static MO table"
echo "  results_dynamic_taut{10,25,50}.csv  — main dynamic tables"
echo "  results_refresh_taut{10,25,50}.csv  — refresh-mode paired tables"
echo "  results_dnsga2_taut{10,25,50}.csv   — DNSGA-II baseline"
echo ""
echo "  out_dynamic_taut{10,25,50}/     — per-iteration CSVs (dynamic)"
echo "  out_refresh_taut{10,25,50}/     — per-iteration CSVs (refresh)"
echo ""
echo "Done."
