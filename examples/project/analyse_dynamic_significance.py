# examples/project/analyse_dynamic_significance.py
"""Mann--Whitney U tests for the main dynamic comparisons.

The refresh-mode ablation already reports paired Wilcoxon tests. This
script supplies the significance tests for the *main* dynamic comparisons,
which are unpaired (different algorithms, different seeds):

1. CCPSO vs the MGPSO-FA ablation -- isolates the contribution of the
   co-evolutionary mechanism from archive-level constraint handling alone.
   This is the headline claim of the dynamic chapter.
2. CCPSO filter vs CCPSO strict -- the archive admission comparison, for
   the dynamic counterpart of the static-campaign finding.

Both are computed on per-run MIGD, extracted from the per-iteration CSVs
of a campaign output folder.

Usage (from the repository root):

    python examples/project/analyse_dynamic_significance.py \\
        --out-dir out_refresh_taut10 --tau-t 10

    # all three frequencies at once
    for t in 10 25 50; do
      python examples/project/analyse_dynamic_significance.py \\
          --out-dir out_refresh_taut$t --tau-t $t
    done

Results are written to significance_dynamic_taut<tau>.csv.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

csv.field_size_limit(10 ** 9)

PROBLEMS = ["DTNK", "DTNK3", "DTNK2"]
CATEGORY = {"DTNK": "SODC", "DTNK3": "DOSC", "DTNK2": "DODC"}


def migd_per_run(path, tau_t):
    """[MIGD per run] from a per-iteration CSV, ignoring the front column."""
    series = defaultdict(list)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.split("[")[0].strip() for h in next(reader)]
        i_it, i_igd = header.index("iteration"), header.index("igd")
        for row in reader:
            if row[i_igd]:
                series[row[0]].append(float(row[i_igd]))
    return [float(np.mean(v)) for v in series.values() if v]


def find(out_dir, problem, algorithm):
    """Handles both plain and _clean-suffixed filenames."""
    for suffix in ("_clean", ""):
        path = os.path.join(out_dir, f"{problem}_{algorithm}{suffix}.out.csv")
        if os.path.exists(path):
            return path
    return None


def compare(out_dir, tau_t, problem, algo_a, algo_b):
    pa, pb = find(out_dir, problem, algo_a), find(out_dir, problem, algo_b)
    if not pa or not pb:
        return None
    a, b = migd_per_run(pa, tau_t), migd_per_run(pb, tau_t)
    if len(a) < 5 or len(b) < 5:
        return None
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    med_a, med_b = float(np.median(a)), float(np.median(b))
    better = ("tie" if p >= 0.05 else (algo_a if med_a < med_b else algo_b))
    improvement = 100.0 * (med_b - med_a) / med_b if med_b else float("nan")
    return {
        "n_a": len(a), "n_b": len(b),
        "median_a": med_a, "median_b": med_b,
        "p": float(p), "better": better, "improvement_pct": improvement,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tau-t", type=int, required=True)
    args = ap.parse_args()

    if not os.path.isdir(args.out_dir):
        print(f"No such directory: {args.out_dir}")
        sys.exit(1)

    comparisons = [
        ("CCPSO_strict", "MGPSO_feasarch"),
        ("CCPSO_filter", "MGPSO_feasarch"),
        ("CCPSO_filter", "CCPSO_strict"),
    ]

    rows = []
    print(f"\nMann--Whitney U on per-run MIGD  "
          f"(tau_t = {args.tau_t}, alpha = 0.05)\n")
    for algo_a, algo_b in comparisons:
        print(f"--- {algo_a} vs {algo_b} ---")
        print(f"{'problem':7} {'cat':5} {'median A':>10} {'median B':>10} "
              f"{'improv':>8} {'p':>12} {'better':>15}")
        for problem in PROBLEMS:
            res = compare(args.out_dir, args.tau_t, problem, algo_a, algo_b)
            if res is None:
                print(f"{problem:7} {CATEGORY[problem]:5}  (files missing)")
                continue
            print(f"{problem:7} {CATEGORY[problem]:5} {res['median_a']:10.4f} "
                  f"{res['median_b']:10.4f} {res['improvement_pct']:7.1f}% "
                  f"{res['p']:12.4g} {res['better']:>15}")
            rows.append([
                args.tau_t, problem, CATEGORY[problem], algo_a, algo_b,
                res["n_a"], res["n_b"],
                f"{res['median_a']:.6f}", f"{res['median_b']:.6f}",
                f"{res['improvement_pct']:.2f}", f"{res['p']:.6g}",
                res["better"],
            ])
        print()

    out_path = f"significance_dynamic_taut{args.tau_t}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau_t", "problem", "category", "algorithm_a",
                    "algorithm_b", "n_a", "n_b", "median_a", "median_b",
                    "improvement_pct", "p_value", "better"])
        w.writerows(rows)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
