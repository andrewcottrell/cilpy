# examples/project/compare_baselines.py
"""Head-to-head comparison: CCPSO against the external pymoo baselines.

Reads aggregate tables already produced by the campaigns and prints (and
writes) side-by-side comparisons. Runs nothing: it is pure analysis, so it
is cheap to re-run whenever a table is regenerated.

Static comparison
-----------------
    CCPSO   <- results_static_mo.csv        (run_static_mo_experiments.py)
    NSGA-II <- results_nsga2.csv            (run_nsga2_baseline.py)

Compared on mean IGD, and on feasible-front hypervolume for OSY, which has
no reference front.

Dynamic comparison
------------------
    CCPSO    <- results_dynamic_taut<T>.csv  (aggregate_campaign.py)
    DNSGA-II <- results_dnsga2_taut<T>.csv   (run_dnsga2_baseline.py)

Compared on MIGD and MIGD_bc, per change frequency.

Significance
------------
Aggregate tables carry means and standard deviations, not per-run values,
so where per-run data is available the script uses it for a Mann--Whitney
U test; otherwise it reports the effect size only and says so. Per-run
CCPSO values are read from the campaign's out/ folders when a folder is
supplied with --ccpso-out-dir.

Usage (from the repository root):

    python examples/project/compare_baselines.py                    # both, defaults
    python examples/project/compare_baselines.py --static-only
    python examples/project/compare_baselines.py --dynamic-only --tau-t 10 25 50
    python examples/project/compare_baselines.py --ccpso-algorithm CCPSO_filter
    python examples/project/compare_baselines.py --ccpso-out-dir out_static

Writes comparison_static.csv and comparison_dynamic.csv.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

csv.field_size_limit(10 ** 9)

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    mannwhitneyu = None

STATIC_PROBLEMS = ["SCH1", "ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6",
                   "BNH", "SRN", "TNK", "CONSTR", "OSY"]
DYNAMIC_PROBLEMS = ["FDA1", "FDA3", "DTNK", "DTNK3", "DTNK2"]
NO_REFERENCE_FRONT = {"OSY"}          # hypervolume instead of IGD

# Identical to the campaign aggregation; never change between algorithms.
HV_REF_POINTS = {
    "SCH1": (110.0, 110.0),
    "ZDT1": (1.1, 1.1), "ZDT2": (1.1, 1.1), "ZDT3": (1.1, 1.1),
    "ZDT4": (1.1, 1.1), "ZDT6": (1.1, 1.1),
    "BNH": (150.0, 60.0), "SRN": (250.0, 100.0),
    "TNK": (1.3, 1.3), "CONSTR": (1.1, 10.0), "OSY": (0.0, 100.0),
}


def read_table(path):
    """Reads an aggregate CSV into {(problem, algorithm): {column: value}}.

    Tolerates the trailing whitespace some campaign headers carry.
    """
    table = {}
    if not os.path.exists(path):
        return table
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            clean = {(k or "").strip(): (v.strip() if isinstance(v, str) else v)
                     for k, v in row.items()}
            key = (clean.get("problem", ""), clean.get("algorithm", ""))
            table[key] = clean
    return table


def number(record, *names):
    """First parseable value among the given column names."""
    for name in names:
        if record and name in record and record[name] not in ("", None):
            try:
                return float(record[name])
            except ValueError:
                continue
    return float("nan")


def per_run_igd(out_dir, problem, algorithm):
    """Per-run final IGD from a campaign summary file, if available."""
    if not out_dir:
        return []
    for suffix in ("_clean", ""):
        path = os.path.join(
            out_dir, f"{problem}_{algorithm}{suffix}.summary.out.csv")
        if os.path.exists(path):
            with open(path, newline="") as f:
                reader = csv.reader(f)
                header = [h.split("[")[0].strip() for h in next(reader)]
                if "final_igd" not in header:
                    return []
                idx = header.index("final_igd")
                return [float(r[idx]) for r in reader if r[idx] != ""]
    return []


def per_run_hv(out_dir, problem, algorithm):
    """Per-run feasible-front hypervolume from a CCPSO summary file.

    The campaign stores the feasible front itself, so hypervolume is
    recomputed here against the same fixed reference point used in the
    campaign aggregation.
    """
    if not out_dir:
        return []
    import ast
    try:
        from cilpy.compare.metrics import hypervolume
    except ImportError:
        return []
    reference = HV_REF_POINTS.get(problem)
    if reference is None:
        return []
    for suffix in ("_clean", ""):
        path = os.path.join(
            out_dir, f"{problem}_{algorithm}{suffix}.summary.out.csv")
        if os.path.exists(path):
            values = []
            with open(path, newline="") as f:
                reader = csv.reader(f)
                header = [h.split("[")[0].strip() for h in next(reader)]
                if "final_feasible_front" not in header:
                    return []
                idx = header.index("final_feasible_front")
                for row in reader:
                    if not row[idx]:
                        values.append(0.0)
                        continue
                    try:
                        front = ast.literal_eval(row[idx])
                    except (ValueError, SyntaxError):
                        continue
                    values.append(hypervolume(front, reference)
                                  if front else 0.0)
            return values
    return []


def per_run_migd(out_dir, problem, algorithm, tau_t):
    """Per-run MIGD from a campaign per-iteration file, if available."""
    if not out_dir:
        return []
    for suffix in ("_clean", ""):
        path = os.path.join(out_dir, f"{problem}_{algorithm}{suffix}.out.csv")
        if os.path.exists(path):
            series = defaultdict(list)
            with open(path, newline="") as f:
                reader = csv.reader(f)
                header = [h.split("[")[0].strip() for h in next(reader)]
                if "igd" not in header:
                    return []
                idx = header.index("igd")
                for row in reader:
                    if row[idx]:
                        series[row[0]].append(float(row[idx]))
            return [float(np.mean(v)) for v in series.values() if v]
    return []


def baseline_per_run(per_run_dir, filename, column):
    """Per-run baseline values written by the pymoo baseline scripts."""
    path = os.path.join(per_run_dir, filename)
    if not os.path.exists(path):
        return []
    values = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get(column) or "").strip()
            if raw:
                try:
                    values.append(float(raw))
                except ValueError:
                    pass
    return values


def test_difference(ours, theirs):
    """Two-sided Mann--Whitney U, or nan when a sample is unavailable."""
    if mannwhitneyu is None or len(ours) < 5 or len(theirs) < 5:
        return float("nan")
    if np.allclose(ours, theirs):
        return 1.0
    try:
        _, p = mannwhitneyu(ours, theirs, alternative="two-sided")
        return float(p)
    except ValueError:
        return float("nan")


def format_p(p_value):
    if np.isnan(p_value):
        return "--"
    return f"{p_value:.2g}" if p_value >= 1e-4 else "<1e-4"


def verdict(ours, theirs, p_value, lower_is_better=True):
    """Winner label, taking significance into account where available."""
    if np.isnan(ours) or np.isnan(theirs):
        return "n/a"
    better = (ours < theirs) if lower_is_better else (ours > theirs)
    if p_value is not None and not np.isnan(p_value) and p_value >= 0.05:
        return "tie"
    return "CCPSO" if better else "baseline"


def improvement(ours, theirs, lower_is_better=True):
    """Percentage improvement of CCPSO over the baseline."""
    if np.isnan(ours) or np.isnan(theirs) or theirs == 0:
        return float("nan")
    return (100.0 * (theirs - ours) / abs(theirs) if lower_is_better
            else 100.0 * (ours - theirs) / abs(theirs))


# ---------------------------------------------------------------------------

def compare_static(ccpso_path, baseline_path, ccpso_algorithm, out_dir,
                   per_run_dir, out_csv):
    ccpso = read_table(ccpso_path)
    baseline = read_table(baseline_path)
    if not ccpso or not baseline:
        print(f"\n=== STATIC ===\n  missing input: "
              f"{ccpso_path if not ccpso else baseline_path}")
        return

    print(f"\n=== STATIC: {ccpso_algorithm} vs NSGA-II ===")
    print(f"{'problem':8} {'metric':6} {'CCPSO':>11} {'NSGA-II':>11} "
          f"{'improv':>9} {'p':>10} {'better':>9}")

    rows = []
    for problem in STATIC_PROBLEMS:
        ours_row = ccpso.get((problem, ccpso_algorithm))
        theirs_row = baseline.get((problem, "NSGA2"))
        if ours_row is None or theirs_row is None:
            continue

        baseline_file = f"perrun_nsga2_{problem}.csv"
        if problem in NO_REFERENCE_FRONT:
            metric = "HV"
            ours = number(ours_row, "hv_feasible_mean (fixed ref)", "hv_mean")
            theirs = number(theirs_row, "hv_mean")
            lower_better = False
            ours_samples = per_run_hv(out_dir, problem, ccpso_algorithm)
            theirs_samples = baseline_per_run(
                per_run_dir, baseline_file, "hv_feasible")
        else:
            metric = "IGD"
            ours = number(ours_row, "igd_mean")
            theirs = number(theirs_row, "igd_mean")
            lower_better = True
            ours_samples = per_run_igd(out_dir, problem, ccpso_algorithm)
            theirs_samples = baseline_per_run(
                per_run_dir, baseline_file, "final_igd")

        p_value = test_difference(ours_samples, theirs_samples)
        gain = improvement(ours, theirs, lower_better)
        who = verdict(ours, theirs, p_value, lower_better)
        print(f"{problem:8} {metric:6} {ours:11.4f} {theirs:11.4f} "
              f"{gain:8.1f}% {format_p(p_value):>10} {who:>9}")
        rows.append([problem, metric, f"{ours:.6f}", f"{theirs:.6f}",
                     f"{gain:.2f}",
                     "" if np.isnan(p_value) else f"{p_value:.6g}",
                     len(ours_samples), len(theirs_samples), who])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "metric", "ccpso", "nsga2",
                         "ccpso_improvement_pct", "p_value",
                         "n_ccpso", "n_baseline", "better"])
        writer.writerows(rows)
    print(f"-> {out_csv}")

    wins = sum(1 for r in rows if r[-1] == "CCPSO")
    print(f"   CCPSO better on {wins} of {len(rows)} problems")


def compare_dynamic(tau_values, ccpso_template, baseline_template,
                    ccpso_algorithm, baseline_algorithm, out_dir_template,
                    per_run_dir, out_csv):
    print(f"\n=== DYNAMIC: {ccpso_algorithm} vs {baseline_algorithm} ===")
    rows = []
    for tau_t in tau_values:
        ccpso = read_table(ccpso_template.format(tau=tau_t))
        baseline = read_table(baseline_template.format(tau=tau_t))
        if not ccpso or not baseline:
            print(f"\n  tau_t = {tau_t}: missing input, skipping")
            continue

        out_dir = (out_dir_template.format(tau=tau_t)
                   if out_dir_template else None)

        print(f"\n  tau_t = {tau_t}")
        print(f"  {'problem':8} {'CCPSO MIGD':>11} {'base MIGD':>11} "
              f"{'improv':>9} {'p':>11} {'better':>9}")

        for problem in DYNAMIC_PROBLEMS:
            ours_row = ccpso.get((problem, ccpso_algorithm))
            theirs_row = None
            for name in (baseline_algorithm, "DNSGA2_A", "DNSGA2_B"):
                if (problem, name) in baseline:
                    theirs_row = baseline[(problem, name)]
                    break
            if ours_row is None or theirs_row is None:
                continue

            ours = number(ours_row, "migd_mean")
            theirs = number(theirs_row, "migd_mean")
            ours_bc = number(ours_row, "migd_before_change_mean")
            theirs_bc = number(theirs_row, "migd_before_change_mean")

            ours_samples = per_run_migd(
                out_dir, problem, ccpso_algorithm, tau_t)
            variant = theirs_row.get("algorithm", "DNSGA2_A").split("_")[-1]
            theirs_samples = baseline_per_run(
                per_run_dir,
                f"perrun_dnsga2_{variant}_taut{tau_t}_{problem}.csv",
                "migd")

            p_value = test_difference(ours_samples, theirs_samples)
            gain = improvement(ours, theirs, True)
            who = verdict(ours, theirs, p_value, True)
            print(f"  {problem:8} {ours:11.4f} {theirs:11.4f} "
                  f"{gain:8.1f}% {format_p(p_value):>11} {who:>9}")
            rows.append([tau_t, problem, f"{ours:.6f}", f"{theirs:.6f}",
                         f"{ours_bc:.6f}", f"{theirs_bc:.6f}",
                         f"{gain:.2f}",
                         "" if np.isnan(p_value) else f"{p_value:.6g}",
                         len(ours_samples), len(theirs_samples), who])

    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tau_t", "problem", "ccpso_migd",
                             "baseline_migd", "ccpso_migd_bc",
                             "baseline_migd_bc",
                             "ccpso_improvement_pct", "p_value",
                             "n_ccpso", "n_baseline", "better"])
            writer.writerows(rows)
        print(f"\n-> {out_csv}")
        wins = sum(1 for r in rows if r[-1] == "CCPSO")
        print(f"   CCPSO better on {wins} of {len(rows)} comparisons")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--static-ccpso", default="results_static_mo.csv")
    ap.add_argument("--static-baseline", default="results_nsga2.csv")
    ap.add_argument("--dynamic-ccpso", default="results_dynamic_taut{tau}.csv")
    ap.add_argument("--dynamic-baseline",
                    default="results_dnsga2_taut{tau}.csv")
    ap.add_argument("--tau-t", type=int, nargs="+", default=[10, 25, 50])
    ap.add_argument("--ccpso-algorithm", default="CCPSO_strict")
    ap.add_argument("--baseline-algorithm", default="DNSGA2_A")
    ap.add_argument("--ccpso-out-dir", default=None,
                    help="campaign out/ folder, enables per-run statistics "
                         "e.g. out_static or out_refresh_taut{tau}")
    ap.add_argument("--per-run-dir", default="per_run",
                    help="folder of per-run baseline CSVs written by the "
                         "pymoo baseline scripts")
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--dynamic-only", action="store_true")
    args = ap.parse_args()

    if not args.dynamic_only:
        compare_static(args.static_ccpso, args.static_baseline,
                       args.ccpso_algorithm, args.ccpso_out_dir,
                       args.per_run_dir, "comparison_static.csv")

    if not args.static_only:
        compare_dynamic(args.tau_t, args.dynamic_ccpso,
                        args.dynamic_baseline, args.ccpso_algorithm,
                        args.baseline_algorithm, args.ccpso_out_dir,
                        args.per_run_dir, "comparison_dynamic.csv")

    print("\nMann--Whitney U, two-sided, alpha = 0.05. A '--' p-value means "
          "per-run\nvalues were unavailable for one side: pass "
          "--ccpso-out-dir for CCPSO, and\nensure the baseline scripts "
          "wrote to --per-run-dir.")


if __name__ == "__main__":
    main()
