"""Visualise DTNK4 feasible region in decision space across environments.

Shows the orbiting scalloped crescent — the 'spinning spiky circle' — as it
moves through (x1, x2) space over successive environments.

Usage:
    python examples/project/plot_dtnk4_decision_space.py
    python examples/project/plot_dtnk4_decision_space.py --envs 12 --tau-t 50
    python examples/project/plot_dtnk4_decision_space.py --overlay   # all on one axis
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cilpy.problem.dynamic_multi_objective import DTNK4


def feasibility_mask(problem, X, Y):
    s = problem._SCALE
    cx, cy = problem._centre()
    phi = problem._scallop_phase()
    dx = cx + 0.5 * s
    dy = cy + 0.5 * s

    rx, ry = X - cx, Y - cy
    angle = np.arctan2(rx, ry)
    dist_sq = rx ** 2 + ry ** 2
    ring_r = s ** 2 + 0.1 * s ** 2 * np.cos(16.0 * (angle - phi))
    g1 = -(dist_sq - ring_r)
    g2 = (X - dx) ** 2 + (Y - dy) ** 2 - 0.5 * s ** 2

    return (g1 <= 0) & (g2 <= 0)


def plot_panels(n_envs, tau_t, n_t, out_png, grid_n=500):
    problem = DTNK4(tau_t=tau_t, n_t=n_t)
    x = np.linspace(0, math.pi, grid_n)
    X, Y = np.meshgrid(x, x)

    cols = min(4, n_envs)
    rows = int(np.ceil(n_envs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.6 * rows),
                             squeeze=False)

    cmap = plt.get_cmap("viridis")
    for k in range(n_envs):
        it = (k + 1) * tau_t - 1
        problem._tau = it
        mask = feasibility_mask(problem, X, Y)

        ax = axes[k // cols][k % cols]
        ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5],
                    colors=[cmap(k / max(1, n_envs - 1))], alpha=0.7)
        ax.contour(X, Y, mask.astype(float), levels=[0.5],
                   colors=["k"], linewidths=0.6)
        cx, cy = problem._centre()
        ax.plot(cx, cy, "w+", ms=8, mew=1.5)
        ax.set_xlim(0, math.pi); ax.set_ylim(0, math.pi)
        ax.set_aspect("equal")
        ax.set_title(f"env {k}  (iter {it})", fontsize=9)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

    for ax in axes.ravel()[n_envs:]:
        ax.axis("off")

    fig.suptitle(f"DTNK4 feasible region in decision space "
                 f"($\\tau_t$ = {tau_t})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def plot_overlay(n_envs, tau_t, n_t, out_png, grid_n=500):
    problem = DTNK4(tau_t=tau_t, n_t=n_t)
    x = np.linspace(0, math.pi, grid_n)
    X, Y = np.meshgrid(x, x)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    cmap = plt.get_cmap("viridis")

    for k in range(n_envs):
        it = (k + 1) * tau_t - 1
        problem._tau = it
        mask = feasibility_mask(problem, X, Y)
        colour = cmap(k / max(1, n_envs - 1))
        ax.contour(X, Y, mask.astype(float), levels=[0.5],
                   colors=[colour], linewidths=1.0)
        cx, cy = problem._centre()
        ax.plot(cx, cy, "+", color=colour, ms=6, mew=1.2)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(0, n_envs - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="environment index")
    cbar.set_ticks(range(0, n_envs, max(1, n_envs // 5)))

    ax.set_xlim(0, math.pi); ax.set_ylim(0, math.pi)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(f"DTNK4: orbiting scalloped crescent, "
                 f"{n_envs} environments ($\\tau_t$ = {tau_t})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--tau-t", type=int, default=50)
    ap.add_argument("--n-t", type=int, default=10)
    ap.add_argument("--overlay", action="store_true",
                    help="all environments on one axis")
    ap.add_argument("--out", help="output path")
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    if args.overlay:
        out = args.out or os.path.join(
            args.fig_dir, f"DTNK4_decision_overlay_taut{args.tau_t}.png")
        plot_overlay(args.envs, args.tau_t, args.n_t, out)
    else:
        out = args.out or os.path.join(
            args.fig_dir, f"DTNK4_decision_panels_taut{args.tau_t}.png")
        plot_panels(args.envs, args.tau_t, args.n_t, out)


if __name__ == "__main__":
    main()
