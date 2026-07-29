# examples/project/aggregate_campaign.py
"""Produces BOTH result tables from a single refresh-mode campaign.

The refresh-mode campaign's ``clean`` arm is exactly the main dynamic
campaign: the same 14 (problem, algorithm) experiments, run with identical
seeds, and therefore identical results. Running both campaigns duplicates
14 of 39 experiments per frequency for no gain. This script lets you run
only the refresh-mode campaign and derive both tables from its output:

1. ``results_dynamic_mo.csv``   -- the main dynamic table (MIGD,
   MIGD_bc, front feasibility per problem and algorithm), computed from
   the ``clean`` arm and with the ``_clean`` suffix stripped from the
   algorithm names so the table matches the main campaign's format.

2. ``results_refresh_mode.csv`` -- the paired clean-vs-clear comparison
   with Wilcoxon signed-rank tests.

Usage (from the repository root), pointing at a campaign's output folder:

    python examples/project/aggregate_campaign.py --out-dir out_refresh_taut10 --tau-t 10
    python examples/project/aggregate_campaign.py --out-dir out_refresh_taut25 --tau-t 25
    python examples/project/aggregate_campaign.py --out-dir out_refresh_taut50 --tau-t 50

Output filenames carry the frequency, e.g. ``results_dynamic_taut10.csv``
and ``results_refresh_taut10.csv``, so successive frequencies do not
overwrite one another.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

PROBLEMS = ["FDA1", "FDA3", "DTNK", "DTNK3", "DTNK2"]
ALGORITHMS = ["MGPSO", "MGPSO_feasarch", "CCPSO_filter", "CCPSO_strict"]
REFRESH_MODES = ("clean", "clear")


def _mean_std(values):
    if len(values) == 0:
        return "", ""
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.4f}", f"{arr.std():.4f}"


def _igd_series(path):
    """{run_id: [(iteration, igd), ...]} from a per-iteration CSV."""
    series = defaultdict(list)
    if not os.path.exists(path):
        return series
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.split("[")[0].strip() for h in next(reader)]
        try:
            idx = header.index("igd")
        except ValueError:
            return series
        for row in reader:
            if row[idx] != "":
                series[row[0]].append((int(row[1]), float(row[idx])))
    return series


def _migd_per_run(path, tau_t):
    """{run_id: (MIGD, MIGD_bc)}."""
    out = {}
    for run_id, pairs in _igd_series(path).items():
        values = [v for _, v in pairs]
        before = [v for it, v in pairs if it % tau_t == tau_t - 1]
        if values:
            out[run_id] = (
                float(np.mean(values)),
                float(np.mean(before)) if before else float("nan"),
            )
    return out


def _front_feasibility(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.split("[")[0].strip() for h in next(reader)]
        try:
            idx = header.index("final_front_feasibility_pct")
        except ValueError:
            return []
        return [float(r[idx]) for r in reader if r[idx] != ""]


# ---------------------------------------------------------------------------

def main_table(out_dir, tau_t, out_path):
    """Main dynamic table, from the clean arm, names de-suffixed."""
    rows = []
    for problem in PROBLEMS:
        for algorithm in ALGORITHMS:
            iter_path = os.path.join(
                out_dir, f"{problem}_{algorithm}_clean.out.csv")
            summary_path = os.path.join(
                out_dir, f"{problem}_{algorithm}_clean.summary.out.csv")
            if not os.path.exists(iter_path):
                continue

            per_run = _migd_per_run(iter_path, tau_t)
            migd = [v[0] for v in per_run.values()]
            migd_bc = [v[1] for v in per_run.values()]
            feas = _front_feasibility(summary_path)

            migd_m, migd_s = _mean_std(migd)
            bc_m, bc_s = _mean_std(migd_bc)
            feas_m, feas_s = _mean_std(feas)
            rows.append([problem, algorithm, migd_m, migd_s,
                         bc_m, bc_s, feas_m, feas_s])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "problem", "algorithm", "migd_mean", "migd_std",
            "migd_before_change_mean", "migd_before_change_std",
            "front_feasibility_pct_mean", "front_feasibility_pct_std",
        ])
        w.writerows(rows)

    print(f"\n=== MAIN DYNAMIC TABLE (tau_t = {tau_t}) ===")
    print(f"{'problem':8} {'algorithm':16} {'MIGD':>18} "
          f"{'MIGD(before chg)':>20} {'front feas%':>16}")
    for r in rows:
        print(f"{r[0]:8} {r[1]:16} {r[2] + ' ± ' + r[3]:>18} "
              f"{r[4] + ' ± ' + r[5]:>20} {r[6] + ' ± ' + r[7]:>16}")
    print(f"-> {out_path}")


def paired_table(out_dir, tau_t, out_path):
    """Paired clean-vs-clear comparison with Wilcoxon signed-rank tests."""
    rows = []
    print(f"\n=== REFRESH MODE COMPARISON (tau_t = {tau_t}) ===")
    print(f"{'problem':7} {'algorithm':16} {'clean MIGD':>18} "
          f"{'clear MIGD':>18} {'paired p':>10} {'better':>7}")

    for problem in PROBLEMS:
        for algorithm in ALGORITHMS:
            per_mode = {
                mode: _migd_per_run(
                    os.path.join(out_dir,
                                 f"{problem}_{algorithm}_{mode}.out.csv"),
                    tau_t)
                for mode in REFRESH_MODES
            }
            if not per_mode["clean"] or not per_mode["clear"]:
                continue

            shared = sorted(set(per_mode["clean"]) & set(per_mode["clear"]),
                            key=int)
            clean = np.array([per_mode["clean"][r][0] for r in shared])
            clear = np.array([per_mode["clear"][r][0] for r in shared])
            clean_bc = np.array([per_mode["clean"][r][1] for r in shared])
            clear_bc = np.array([per_mode["clear"][r][1] for r in shared])

            p_value, verdict = float("nan"), "n/a"
            if wilcoxon is not None and len(shared) >= 5:
                if np.any(clean != clear):
                    _, p_value = wilcoxon(clean, clear)
                    verdict = ("tie" if p_value >= 0.05 else
                               ("clean" if np.median(clean) < np.median(clear)
                                else "clear"))
                else:
                    p_value, verdict = 1.0, "identical"

            rows.append([
                problem, algorithm, len(shared),
                f"{clean.mean():.4f}", f"{clean.std():.4f}",
                f"{clear.mean():.4f}", f"{clear.std():.4f}",
                f"{clean_bc.mean():.4f}", f"{clear_bc.mean():.4f}",
                f"{p_value:.4g}", verdict,
            ])
            print(f"{problem:7} {algorithm:16} "
                  f"{clean.mean():9.4f} ± {clean.std():6.4f} "
                  f"{clear.mean():9.4f} ± {clear.std():6.4f} "
                  f"{p_value:10.4g} {verdict:>7}")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "problem", "algorithm", "paired_runs",
            "clean_migd_mean", "clean_migd_std",
            "clear_migd_mean", "clear_migd_std",
            "clean_migd_bc_mean", "clear_migd_bc_mean",
            "wilcoxon_p", "better",
        ])
        w.writerows(rows)
    print(f"-> {out_path}")
    if wilcoxon is None:
        print("NOTE: scipy unavailable; paired tests skipped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True,
                        help="campaign output folder, e.g. out_refresh_taut10")
    parser.add_argument("--tau-t", type=int, required=True,
                        help="change frequency the campaign was run at")
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        print(f"No such directory: {args.out_dir}")
        sys.exit(1)

    main_table(args.out_dir, args.tau_t,
               f"results_dynamic_taut{args.tau_t}.csv")
    paired_table(args.out_dir, args.tau_t,
                 f"results_refresh_taut{args.tau_t}.csv")


if __name__ == "__main__":
    main()