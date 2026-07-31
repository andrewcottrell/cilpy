# examples/project/plot_front_tracking.py
"""Plots the obtained archive against the true Pareto front over time.

Reads a per-iteration campaign CSV (the ``front`` column holds the full
archive as objective vectors at every iteration) and regenerates the true
Pareto front for the corresponding environment from the problem definition,
so the two can be overlaid.

Three outputs, selected by ``--mode``:

* ``snapshots`` (default) -- a grid of panels, one per environment,
  overlaying the archive on the true front. The report figure: it shows at
  a glance whether the algorithm tracks the front as it moves.
* ``trace`` -- IGD against iteration for one run, with environment changes
  marked. Shows the sawtooth recovery pattern and makes the distinction
  between MIGD (mean over all iterations) and MIGD_bc (mean at the last
  iteration of each environment) visually obvious.
* ``compare`` -- IGD traces of two CSVs on shared axes, for the refresh
  mode ablation (clean vs clear).
* ``animate`` -- optional GIF of the archive tracking the front, for the
  demo rather than the report. Requires pillow.

The problem is inferred from the filename, or given with ``--problem``.

Examples (from the repository root):

    python examples/project/plot_front_tracking.py \\
        out_refresh_taut10/DTNK3_CCPSO_strict_clean.out.csv --tau-t 10

    python examples/project/plot_front_tracking.py \\
        out_refresh_taut10/DTNK_CCPSO_strict_clean.out.csv --tau-t 10 --mode trace

    python examples/project/plot_front_tracking.py \\
        out_refresh_taut10/DTNK3_CCPSO_strict_clean.out.csv \\
        --compare-with out_refresh_taut10/DTNK3_CCPSO_strict_clear.out.csv \\
        --tau-t 10 --mode compare
"""

import argparse
import ast
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cilpy.problem.dynamic_multi_objective import (
    FDA1, FDA3, DTNK, DTNK2, DTNK3,
)

csv.field_size_limit(10 ** 9)   # the front column is large

PROBLEM_CLASSES = {
    "FDA1": FDA1, "FDA3": FDA3,
    "DTNK": DTNK, "DTNK2": DTNK2, "DTNK3": DTNK3,
}


def infer_problem(path):
    """Longest matching problem name in the filename (DTNK3 before DTNK)."""
    base = os.path.basename(path)
    for name in sorted(PROBLEM_CLASSES, key=len, reverse=True):
        if base.startswith(name + "_"):
            return name
    return None


def read_run(path, run_id="1", iterations=None, want_front=True):
    """Streams one run out of a per-iteration CSV.

    Args:
        path: CSV path.
        run_id: Which run to extract.
        iterations: Optional set of iterations to keep (fronts are large;
            restricting this keeps memory flat).
        want_front: Parse the front column.

    Returns:
        dict {iteration: {"igd": float, "front": np.ndarray or None,
                          "size": int}}
    """
    out = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.split("[")[0].strip() for h in next(reader)]
        i_iter = header.index("iteration")
        i_igd = header.index("igd")
        i_size = header.index("archive_size")
        i_front = header.index("front")
        for row in reader:
            if row[0] != run_id:
                continue
            it = int(row[i_iter])
            if iterations is not None and it not in iterations:
                continue
            front = None
            if want_front and row[i_front]:
                front = np.asarray(ast.literal_eval(row[i_front]), dtype=float)
            out[it] = {
                "igd": float(row[i_igd]) if row[i_igd] else np.nan,
                "front": front,
                "size": int(row[i_size]),
            }
    return out


def true_front_at(problem_cls, tau_t, n_t, iteration, n_points=400):
    """Reference front for the environment active at `iteration`."""
    problem = problem_cls(tau_t=tau_t, n_t=n_t)
    # The runner calls begin_iteration() before each step, so at iteration k
    # the counter has been advanced k times.
    problem._tau = iteration
    return problem.true_pareto_front(n_points), problem.t


# ---------------------------------------------------------------------------

def plot_snapshots(path, problem_name, tau_t, n_t, run_id, n_panels, out_png):
    problem_cls = PROBLEM_CLASSES[problem_name]
    # Sample the LAST iteration of successive environments, so each panel
    # shows converged quality for that environment.
    picks = [(k + 1) * tau_t - 1 for k in range(n_panels)]
    data = read_run(path, run_id, iterations=set(picks))
    picks = [p for p in picks if p in data]

    # Pre-compute every panel's content so that SHARED axis limits can be
    # applied. Autoscaling each panel independently makes a translating
    # front appear stationary, since the axes move with it.
    panels = []
    xs, ys = [], []
    for it in picks:
        ref, t = true_front_at(problem_cls, tau_t, n_t, it)
        front = data[it]["front"]
        panels.append((it, t, ref, front))
        xs.append(ref[:, 0]); ys.append(ref[:, 1])
        if front is not None and len(front):
            xs.append(front[:, 0]); ys.append(front[:, 1])
    x_all = np.concatenate(xs); y_all = np.concatenate(ys)
    pad_x = 0.05 * (x_all.max() - x_all.min())
    pad_y = 0.05 * (y_all.max() - y_all.min())
    xlim = (x_all.min() - pad_x, x_all.max() + pad_x)
    ylim = (y_all.min() - pad_y, y_all.max() + pad_y)

    cols = min(4, len(picks))
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.4 * rows),
                             squeeze=False)

    for ax, (it, t, ref, front) in zip(axes.ravel(), panels):
        ax.plot(ref[:, 0], ref[:, 1], ".", ms=2.5, color="0.62",
                label="true POF")
        if front is not None and len(front):
            ax.plot(front[:, 0], front[:, 1], "o", ms=3.2, color="crimson",
                    label="archive")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)   # shared: motion visible
        ax.set_title(f"iter {it}   $t$ = {t:.1f}   "
                     f"IGD = {data[it]['igd']:.4f}", fontsize=9)
        ax.set_xlabel("$f_1$"); ax.set_ylabel("$f_2$")
        ax.legend(fontsize=7, loc="upper right")

    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")

    fig.suptitle(f"{problem_name}: archive vs true Pareto front "
                 f"($\\tau_t$ = {tau_t}, run {run_id}); "
                 f"axes shared across panels", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


def plot_overlay(path, problem_name, tau_t, n_t, run_id, n_envs, out_png):
    """All environments on one axis, coloured by time: shows the motion."""
    problem_cls = PROBLEM_CLASSES[problem_name]
    picks = [(k + 1) * tau_t - 1 for k in range(n_envs)]
    data = read_run(path, run_id, iterations=set(picks))
    picks = [p for p in picks if p in data]

    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for k, it in enumerate(picks):
        colour = cmap(k / max(1, len(picks) - 1))
        ref, t = true_front_at(problem_cls, tau_t, n_t, it)
        axes[0].plot(ref[:, 0], ref[:, 1], ".", ms=2.5, color=colour,
                     label=f"$t$={t:.1f}")
        front = data[it]["front"]
        if front is not None and len(front):
            axes[1].plot(front[:, 0], front[:, 1], "o", ms=3.0, color=colour,
                         label=f"$t$={t:.1f}")

    for ax, title in zip(axes, ["true Pareto front per environment",
                                "obtained archive per environment"]):
        ax.set_xlabel("$f_1$"); ax.set_ylabel("$f_2$")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, ncol=2)
    # Shared limits between the two panels for direct comparison.
    xlim = (min(a.get_xlim()[0] for a in axes), max(a.get_xlim()[1] for a in axes))
    ylim = (min(a.get_ylim()[0] for a in axes), max(a.get_ylim()[1] for a in axes))
    for ax in axes:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    fig.suptitle(f"{problem_name}: front motion over {len(picks)} "
                 f"environments ($\\tau_t$ = {tau_t}, run {run_id})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


def plot_trace(path, problem_name, tau_t, run_id, out_png, max_iter=200):
    iterations = set(range(1, max_iter + 1))
    data = read_run(path, run_id, iterations=iterations, want_front=False)
    its = sorted(data)
    igd = [data[i]["igd"] for i in its]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(its, igd, "-", lw=1.2, color="crimson", label="IGD")

    for k in range(1, max_iter // tau_t + 1):
        ax.axvline(k * tau_t, color="0.75", lw=0.8, ls="--", zorder=0)
    bc = [i for i in its if i % tau_t == tau_t - 1]
    ax.plot(bc, [data[i]["igd"] for i in bc], "o", ms=4, color="navy",
            label="last iteration of environment (MIGD$_{bc}$ samples)")

    ax.axhline(np.nanmean(igd), color="darkgreen", lw=1.0, ls=":",
               label=f"MIGD = {np.nanmean(igd):.4f}")
    ax.set_xlabel("iteration"); ax.set_ylabel("IGD")
    ax.set_title(f"{problem_name}: IGD over time "
                 f"($\\tau_t$ = {tau_t}, run {run_id}); "
                 f"dashed lines mark environment changes", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


def plot_compare(path_a, path_b, label_a, label_b, problem_name, tau_t,
                 run_id, out_png, max_iter=200):
    iterations = set(range(1, max_iter + 1))
    a = read_run(path_a, run_id, iterations, want_front=False)
    b = read_run(path_b, run_id, iterations, want_front=False)
    its = sorted(set(a) & set(b))

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(its, [a[i]["igd"] for i in its], "-", lw=1.2, color="crimson",
            label=f"{label_a} (MIGD = "
                  f"{np.nanmean([a[i]['igd'] for i in its]):.4f})")
    ax.plot(its, [b[i]["igd"] for i in its], "-", lw=1.2, color="darkorange",
            label=f"{label_b} (MIGD = "
                  f"{np.nanmean([b[i]['igd'] for i in its]):.4f})")
    for k in range(1, max_iter // tau_t + 1):
        ax.axvline(k * tau_t, color="0.8", lw=0.7, ls="--", zorder=0)
    ax.set_xlabel("iteration"); ax.set_ylabel("IGD")
    ax.set_title(f"{problem_name}: archive refresh policy "
                 f"($\\tau_t$ = {tau_t}, run {run_id})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")


def animate(path, problem_name, tau_t, n_t, run_id, out_gif, max_iter=120):
    from matplotlib.animation import FuncAnimation, PillowWriter
    problem_cls = PROBLEM_CLASSES[problem_name]
    iterations = set(range(1, max_iter + 1))
    data = read_run(path, run_id, iterations)
    its = sorted(data)

    fig, ax = plt.subplots(figsize=(5, 4))

    def draw(it):
        ax.clear()
        ref, t = true_front_at(problem_cls, tau_t, n_t, it)
        ax.plot(ref[:, 0], ref[:, 1], ".", ms=2.5, color="0.62")
        front = data[it]["front"]
        if front is not None and len(front):
            ax.plot(front[:, 0], front[:, 1], "o", ms=3.5, color="crimson")
        ax.set_title(f"{problem_name}  iter {it}  $t$={t:.1f}  "
                     f"IGD={data[it]['igd']:.4f}", fontsize=9)
        ax.set_xlabel("$f_1$"); ax.set_ylabel("$f_2$")

    anim = FuncAnimation(fig, draw, frames=its, interval=120)
    anim.save(out_gif, writer=PillowWriter(fps=8))
    print(f"wrote {out_gif}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_path")
    p.add_argument("--mode", choices=["snapshots", "overlay", "trace",
                                      "compare", "animate"],
                   default="snapshots")
    p.add_argument("--compare-with", help="second CSV for --mode compare")
    p.add_argument("--problem", help="override problem inferred from filename")
    p.add_argument("--tau-t", type=int, required=True)
    p.add_argument("--n-t", type=int, default=10)
    p.add_argument("--run", default="1")
    p.add_argument("--panels", type=int, default=8)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--out", help="output image path")
    args = p.parse_args()

    problem_name = args.problem or infer_problem(args.csv_path)
    if problem_name not in PROBLEM_CLASSES:
        print(f"Could not infer problem from '{args.csv_path}'. "
              f"Use --problem with one of {sorted(PROBLEM_CLASSES)}.")
        sys.exit(1)

    stem = os.path.splitext(os.path.basename(args.csv_path))[0]
    out = args.out

    if args.mode == "snapshots":
        plot_snapshots(args.csv_path, problem_name, args.tau_t, args.n_t,
                       args.run, args.panels,
                       out or f"{stem}_snapshots.png")
    elif args.mode == "overlay":
        plot_overlay(args.csv_path, problem_name, args.tau_t, args.n_t,
                     args.run, args.panels,
                     out or f"{stem}_overlay.png")
    elif args.mode == "trace":
        plot_trace(args.csv_path, problem_name, args.tau_t, args.run,
                   out or f"{stem}_trace.png", args.max_iter)
    elif args.mode == "compare":
        if not args.compare_with:
            print("--mode compare requires --compare-with")
            sys.exit(1)
        plot_compare(args.csv_path, args.compare_with, "clean", "clear",
                     problem_name, args.tau_t, args.run,
                     out or f"{stem}_compare.png", args.max_iter)
    else:
        animate(args.csv_path, problem_name, args.tau_t, args.n_t, args.run,
                out or f"{stem}.gif", args.max_iter)


if __name__ == "__main__":
    main()
