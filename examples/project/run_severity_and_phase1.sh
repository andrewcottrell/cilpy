#!/usr/bin/env bash
# run_severity_and_phase1.sh — the two supplementary campaigns of 15 Oct 2026.
#
#   1. Change-severity sweep: n_t in {5, 20} at tau_t = 25 (n_t = 10 is the
#      main campaign). DTNK, DTNK5, DTNK6 with the four MGPSO-based algorithms
#      (clean refresh), FDA1 and FDA3 with MGPSO, and DNSGA-II on all five.
#      DTNK4 is left out: its shape depends on floor(tau / tau_t) only, so
#      n_t has no effect on it.
#   2. Phase 1 re-run: the single-objective CCLS on CEC2006 G01, G02, G04,
#      G05, G06 and CMPB in all four classes, with sentry change detection in
#      the objective PSO.
#
# Output files of the sweep carry an _nt<n_t> suffix (results_dynamic_taut25_nt5.csv
# etc.), so none of the main campaign's files are touched.
#
# Usage (from the cilpy repo root):
#   nohup bash examples/project/run_severity_and_phase1.sh > severity.log 2>&1 &
#   QUICK=1 bash examples/project/run_severity_and_phase1.sh   # smoke test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
RUNS=30
ITERS=1000
WORKERS="${WORKERS:-8}"
TAU_T=25
SEVERITIES="5 20"
SWEEP_PROBLEMS="FDA1 FDA3 DTNK DTNK5 DTNK6"
RAW_DIR="$SCRIPT_DIR/raw"
SUFFIX=""
if [ "${QUICK:-0}" = "1" ]; then
    RUNS=2
    ITERS=60
    # Keep the smoke test away from the real raw/ and results/ folders.
    SUFFIX="_quick"
fi

if [ -d out ]; then
    echo "An out/ folder exists in $REPO_ROOT; move or delete it first."
    exit 1
fi
mkdir -p "$RAW_DIR"

echo "============================================================"
echo "  SEVERITY SWEEP AND PHASE 1 RE-RUN"
echo "  Started: $(date)"
echo "  Runs: $RUNS   Iterations: $ITERS   Workers: $WORKERS"
echo "============================================================"

$PYTHON -m pytest test/ -q --tb=short

# -------------------------------------------------------------------
# 1. Change-severity sweep
# -------------------------------------------------------------------
for N_T in $SEVERITIES; do
    OUT="$RAW_DIR/out_severity_taut${TAU_T}_nt${N_T}${SUFFIX}"
    echo ""
    echo "============================================================"
    echo "  SEVERITY n_t=$N_T  (tau_t=$TAU_T)   $(date)"
    echo "============================================================"
    $PYTHON examples/project/run_dynamic_experiments.py \
        --runs "$RUNS" --iters "$ITERS" --tau-t "$TAU_T" --n-t "$N_T" \
        --workers "$WORKERS" --problems $SWEEP_PROBLEMS --out-dir out
    rm -rf "$OUT"
    mv out "$OUT"

    if [ "${QUICK:-0}" != "1" ]; then
        $PYTHON examples/project/analyse_dynamic_significance.py \
            --out-dir "$OUT" --tau-t "$TAU_T" --n-t "$N_T" \
            --problems DTNK DTNK5 DTNK6 || true
        $PYTHON examples/project/run_dnsga2_baseline.py \
            --runs "$RUNS" --n-gen "$ITERS" --tau-t "$TAU_T" --n-t "$N_T" \
            --problems $SWEEP_PROBLEMS
        $PYTHON examples/project/compare_baselines.py --dynamic-only \
            --tau-t "$TAU_T" --n-t "$N_T" --ccpso-out-dir "$OUT" || true
    fi
done

# -------------------------------------------------------------------
# 2. Phase 1 re-run
# -------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  PHASE 1 RE-RUN   $(date)"
echo "============================================================"
OUT="$RAW_DIR/out_phase1${SUFFIX}"
QUICK_FLAG=""
[ "${QUICK:-0}" = "1" ] && QUICK_FLAG="--quick"
$PYTHON examples/project/run_phase1.py --runs "$RUNS" --iters "$ITERS" \
    --workers "$WORKERS" $QUICK_FLAG --out-dir out
rm -rf "$OUT"
mv out "$OUT"
if [ "${QUICK:-0}" = "1" ]; then
    rm -f "$SCRIPT_DIR/results/results_phase1.csv" \
          "$SCRIPT_DIR/results/results_dynamic_taut${TAU_T}_nt"*.csv
    echo ""
    echo "Smoke test passed. Delete raw/*_quick when done."
    exit 0
fi

$PYTHON examples/project/slim_campaign.py \
    "out_severity_taut${TAU_T}_nt5" "out_severity_taut${TAU_T}_nt20" out_phase1

echo ""
echo "============================================================"
echo "  DONE   $(date)"
echo "============================================================"
echo "  results/results_dynamic_taut${TAU_T}_nt{5,20}.csv       sweep, MGPSO-based"
echo "  results/significance_dynamic_taut${TAU_T}_nt{5,20}.csv"
echo "  results/results_dnsga2_taut${TAU_T}_nt{5,20}.csv        sweep, DNSGA-II"
echo "  results/comparison_dynamic_nt{5,20}.csv"
echo "  results/results_phase1.csv                             Phase 1 table"
echo "  raw/out_severity_*, raw/out_phase1                     per-iteration CSVs"
echo "  data/out_severity_*, data/out_phase1                   the same without fronts"
