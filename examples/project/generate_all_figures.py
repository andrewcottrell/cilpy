# examples/project/generate_all_figures.py
"""Generates the full report figure set from campaign output folders.

Produces, into ``figures/`` by default:

Dynamic (one campaign folder per change frequency):
  <problem>_overlay_taut<T>.png    front motion across environments, true
                                   front and obtained archive side by side
  <problem>_snapshots_taut<T>.png  per-environment panels, shared axes
  <problem>_trace_taut<T>.png      IGD against iteration, changes marked
  <problem>_refresh_taut<T>.png    clean vs clear IGD traces (ablation)

Static (from the static campaign folder):
  static_fronts.png                obtained vs true front for every
                                   constrained problem and algorithm

Only figures whose source CSVs exist are attempted, so a partial campaign
still produces whatever it can. Existing files are skipped unless
``--force`` is given, which makes re-runs cheap while iterating.

Usage (from the repository root):

    # everything it can find
    python examples/project/generate_all_figures.py

    # specific frequencies / a subset of problems
    python examples/project/generate_all_figures.py --tau-t 10 50
    python examples/project/generate_all_figures.py --problems DTNK DTNK2

    # only the report-critical figures
    python examples/project/generate_all_figures.py --report-set
"""

import argparse
import ast
import csv
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

csv.field_size_limit(10 ** 9)

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTTER = os.path.join(HERE, "plot_front_tracking.py")

DYNAMIC_PROBLEMS = ["FDA1", "FDA3", "DTNK", "DTNK3", "DTNK2"]
# The report only needs the constrained dynamic problems in every mode.
REPORT_DYNAMIC = ["DTNK", "DTNK3", "DTNK2"]
STATIC_PROBLEMS = ["BNH", "SRN", "TNK", "CONSTR"]      # OSY: 6-D, no ref front
STATIC_ALGORITHMS = ["MGPSO", "MGPSO_feasarch", "CCPSO_filter", "CCPSO_strict"]

DEFAULT_ALGORITHM = "CCPSO_strict"


def find_csv(out_dir, problem, algorithm, suffixes=("_clean", "")):
    for suffix in suffixes:
        path = os.path.join(out_dir, f"{problem}_{algorithm}{suffix}.out.csv")
        if os.path.exists(path):
            return path
    return None


def run_plot(csv_path, mode, tau_t, out_png, extra=None, force=False):
    if os.path.exists(out_png) and not force:
        print(f"  skip (exists): {os.path.basename(out_png)}")
        return True
    cmd = [sys.executable, PLOTTER, csv_path, "--tau-t", str(tau_t),
           "--mode", mode, "--out", out_png]
    if extra:
        cmd += extra
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {os.path.basename(out_png)}")
        print("    " + (result.stderr.strip().splitlines() or ["?"])[-1])
        return False
    print(f"  wrote {os.path.basename(out_png)}")
    return True


# ---------------------------------------------------------------------------
# Dynamic figures
# ---------------------------------------------------------------------------

def dynamic_figures(out_dir, tau_t, problems, fig_dir, algorithm,
                    max_iter, force, report_set):
    print(f"\n=== dynamic figures: {out_dir} (tau_t={tau_t}) ===")
    made = 0
    for problem in problems:
        clean = find_csv(out_dir, problem, algorithm)
        if clean is None:
            print(f"  {problem}: no CSV for {algorithm}, skipping")
            continue

        # Overlay: the clearest single figure for front motion.
        made += run_plot(
            clean, "overlay", tau_t,
            os.path.join(fig_dir, f"{problem}_overlay_taut{tau_t}.png"),
            ["--panels", "8"], force)

        # Trace: sawtooth recovery; scale the window to the frequency so a
        # comparable number of environments is shown at every tau_t.
        window = max_iter or max(150, tau_t * 12)
        made += run_plot(
            clean, "trace", tau_t,
            os.path.join(fig_dir, f"{problem}_trace_taut{tau_t}.png"),
            ["--max-iter", str(window)], force)

        if not report_set:
            made += run_plot(
                clean, "snapshots", tau_t,
                os.path.join(fig_dir, f"{problem}_snapshots_taut{tau_t}.png"),
                ["--panels", "8"], force)

        # Refresh ablation: clean vs clear on shared axes.
        clear = os.path.join(out_dir, f"{problem}_{algorithm}_clear.out.csv")
        if os.path.exists(clear):
            made += run_plot(
                clean, "compare", tau_t,
                os.path.join(fig_dir, f"{problem}_refresh_taut{tau_t}.png"),
                ["--compare-with", clear, "--max-iter", str(window)], force)
    return made


# ---------------------------------------------------------------------------
# Static figure
# ---------------------------------------------------------------------------

def _final_front(path):
    """Final-iteration front of run 1, from a per-iteration CSV."""
    last = None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.split("[")[0].strip() for h in next(reader)]
        i_front = header.index("front")
        for row in reader:
            if row[0] != "1":
                continue
            if row[i_front]:
                last = row[i_front]
    return np.asarray(ast.literal_eval(last), dtype=float) if last else None


def static_figure(out_dir, fig_dir, force):
    out_png = os.path.join(fig_dir, "static_fronts.png")
    if os.path.exists(out_png) and not force:
        print(f"\n=== static figure ===\n  skip (exists): static_fronts.png")
        return 0
    print(f"\n=== static figure: {out_dir} ===")

    from cilpy.problem.multi_objective import BNH, SRN, TNK, CONSTR
    classes = {"BNH": BNH, "SRN": SRN, "TNK": TNK, "CONSTR": CONSTR}

    available = [p for p in STATIC_PROBLEMS
                 if os.path.exists(os.path.join(out_dir, f"{p}_MGPSO.out.csv"))]
    if not available:
        print("  no static CSVs found, skipping")
        return 0

    rows, cols = len(STATIC_ALGORITHMS), len(available)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows),
                             squeeze=False)

    for j, problem in enumerate(available):
        ref = classes[problem]().true_pareto_front(600)
        for i, algorithm in enumerate(STATIC_ALGORITHMS):
            ax = axes[i][j]
            path = os.path.join(out_dir, f"{problem}_{algorithm}.out.csv")
            ax.plot(ref[:, 0], ref[:, 1], ".", ms=2, color="0.65")
            if os.path.exists(path):
                front = _final_front(path)
                if front is not None and len(front):
                    ax.plot(front[:, 0], front[:, 1], "o", ms=3,
                            color="crimson")
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="0.5")
            if i == 0:
                ax.set_title(problem, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{algorithm}\n$f_2$", fontsize=8)
            ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("$f_1$", fontsize=8)

    fig.suptitle("Static constrained problems: obtained front (red) "
                 "vs true Pareto front (grey), run 1", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"  wrote static_fronts.png")
    return 1


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau-t", type=int, nargs="+", default=[10, 25, 50])
    ap.add_argument("--dynamic-dir", default="out_refresh_taut{tau}",
                    help="template for dynamic campaign folders")
    ap.add_argument("--static-dir", default="out_static",
                    help="static campaign output folder")
    ap.add_argument("--problems", nargs="+", default=None,
                    help=f"default: {DYNAMIC_PROBLEMS}")
    ap.add_argument("--algorithm", default=DEFAULT_ALGORITHM)
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="trace window; default scales with tau_t")
    ap.add_argument("--report-set", action="store_true",
                    help="only the report-critical figures")
    ap.add_argument("--force", action="store_true",
                    help="regenerate figures that already exist")
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    problems = args.problems or (REPORT_DYNAMIC if args.report_set
                                 else DYNAMIC_PROBLEMS)

    total = 0
    for tau_t in args.tau_t:
        out_dir = args.dynamic_dir.format(tau=tau_t)
        if not os.path.isdir(out_dir):
            print(f"\n=== dynamic figures: {out_dir} ===\n  no such folder, "
                  f"skipping")
            continue
        total += dynamic_figures(out_dir, tau_t, problems, args.fig_dir,
                                 args.algorithm, args.max_iter, args.force,
                                 args.report_set)

    if os.path.isdir(args.static_dir):
        total += static_figure(args.static_dir, args.fig_dir, args.force)
    else:
        print(f"\n=== static figure ===\n  '{args.static_dir}' not found; "
              f"pass --static-dir <folder> to include it")

    print(f"\nDone. Figures in {args.fig_dir}/")


if __name__ == "__main__":
    main()
