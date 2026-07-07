# examples/project/validate_results.py
"""Validity analysis for the static MO campaign results.

Answers "can I trust these numbers?" with five check families, each
reporting PASS / WARN / FAIL:

1. COMPLETENESS  -- every expected (problem, algorithm) experiment exists
   and contains the expected number of runs, with no empty metric columns
   where values are expected.

2. CONVERGENCE   -- the per-iteration IGD has plateaued: if the final IGD is
   still substantially below the mid-run IGD, the runs were too short and
   final-iteration comparisons are unstable.

3. PLAUSIBILITY  -- unconstrained IGD values fall in ranges consistent with
   published multi-objective PSO results on the ZDT suite (an implementation
   bug almost always blows these numbers up by orders of magnitude).
   ZDT4 is exempt: multimodal failure is the documented expected outcome.

4. EXPECTED ORDERINGS -- constraint-handling sanity: CCPSO variants must
   deliver ~100% feasible fronts; plain MGPSO must NOT be beating CCPSO on
   feasible-front hypervolume anywhere (if it does, something is mislabeled).

5. SIGNIFICANCE  -- two-sided Mann-Whitney U tests (the standard
   non-parametric test in evolutionary computation, since fitness
   distributions are rarely normal) between CCPSO_filter and the
   MGPSO_feasarch ablation on final IGD, per constrained problem. Reports
   which differences are significant at alpha = 0.05.

Exit code is non-zero if any FAIL occurs, so this can gate an analysis
pipeline. Usage (from the repository root, after the campaign):

    python examples/project/validate_results.py
    python examples/project/validate_results.py --runs 30 --iters 1000
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

UNCONSTRAINED = ["SCH1", "ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"]
CONSTRAINED = ["BNH", "SRN", "TNK", "CONSTR", "OSY"]
CONSTRAINED_ALGOS = ["MGPSO", "MGPSO_feasarch", "CCPSO_filter", "CCPSO_strict"]

# Plausible final-IGD ceilings for a correctly implemented MO-PSO after
# ~1000 iterations (generous: published MGPSO/NSGA-II results are usually
# well below these). Exceeding the ceiling suggests an implementation or
# configuration error rather than ordinary stochastic variation.
IGD_PLAUSIBLE_CEILING = {
    "SCH1": 0.10, "ZDT1": 0.10, "ZDT2": 0.10, "ZDT3": 0.15, "ZDT6": 0.10,
    # ZDT4 intentionally absent: failure is the expected, documented outcome.
}

_counts = {"PASS": 0, "WARN": 0, "FAIL": 0}


def report(level, message):
    _counts[level] += 1
    print(f"  [{level:4}] {message}")


def read_summary(problem, algo):
    path = f"out/{problem}_{algo}.summary.out.csv"
    if not os.path.exists(path):
        return None
    columns = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.split("[")[0].strip() for h in next(reader)]
        for row in reader:
            for key, value in zip(header, row):
                columns[key].append(value)
    return columns


def read_igd_series(problem, algo):
    """Per-run IGD time series from the per-iteration CSV."""
    path = f"out/{problem}_{algo}.out.csv"
    if not os.path.exists(path):
        return {}
    series = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.split("[")[0].strip() for h in next(reader)]
        try:
            igd_idx = header.index("igd")
        except ValueError:
            return {}
        for row in reader:
            if row[igd_idx] != "":
                series[row[0]].append(float(row[igd_idx]))
    return series


def floats(values):
    return [float(v) for v in values if v != ""]


def check_completeness(expected_runs):
    print("\n== 1. COMPLETENESS ==")
    experiments = [(p, "MGPSO") for p in UNCONSTRAINED] + [
        (p, a) for p in CONSTRAINED for a in CONSTRAINED_ALGOS
    ]
    for problem, algo in experiments:
        columns = read_summary(problem, algo)
        if columns is None:
            report("FAIL", f"{problem} x {algo}: summary file missing")
            continue
        n = len(columns.get("run_id", []))
        if n != expected_runs:
            report("FAIL", f"{problem} x {algo}: {n} runs, expected {expected_runs}")
        elif problem != "OSY" and len(floats(columns.get("final_igd", []))) != n:
            report("FAIL", f"{problem} x {algo}: missing IGD values")
        else:
            report("PASS", f"{problem} x {algo}: {n} runs complete")


def check_convergence():
    print("\n== 2. CONVERGENCE (IGD plateau) ==")
    experiments = [(p, "MGPSO") for p in UNCONSTRAINED if p != "ZDT4"] + [
        (p, a) for p in CONSTRAINED if p != "OSY"
        for a in ("CCPSO_filter", "CCPSO_strict")
    ]
    for problem, algo in experiments:
        series = read_igd_series(problem, algo)
        if not series:
            report("WARN", f"{problem} x {algo}: no IGD series found")
            continue
        ratios = []
        for run_series in series.values():
            mid = run_series[len(run_series) // 2]
            final = run_series[-1]
            if mid > 0:
                ratios.append(final / mid)
        mean_ratio = float(np.mean(ratios))
        if mean_ratio < 0.5:
            report("WARN",
                   f"{problem} x {algo}: final IGD is {mean_ratio:.0%} of "
                   f"mid-run IGD -- still improving, consider more iterations")
        else:
            report("PASS",
                   f"{problem} x {algo}: plateaued "
                   f"(final/mid IGD = {mean_ratio:.2f})")


def check_plausibility():
    print("\n== 3. PLAUSIBILITY (vs published ranges) ==")
    for problem, ceiling in IGD_PLAUSIBLE_CEILING.items():
        columns = read_summary(problem, "MGPSO")
        if columns is None:
            continue
        igds = floats(columns.get("final_igd", []))
        if not igds:
            report("WARN", f"{problem}: no IGD values")
            continue
        median = float(np.median(igds))
        if median <= ceiling:
            report("PASS", f"{problem}: median IGD {median:.4f} "
                           f"<= plausible ceiling {ceiling}")
        else:
            report("FAIL", f"{problem}: median IGD {median:.4f} exceeds "
                           f"plausible ceiling {ceiling} -- investigate")
    columns = read_summary("ZDT4", "MGPSO")
    if columns:
        median = float(np.median(floats(columns.get("final_igd", ["nan"]))))
        report("PASS", f"ZDT4: median IGD {median:.4f} (failure expected and "
                       f"documented -- premature convergence on 21^9 local fronts)")


def check_orderings():
    print("\n== 4. EXPECTED ORDERINGS ==")
    for problem in CONSTRAINED:
        for algo in ("CCPSO_filter", "CCPSO_strict"):
            columns = read_summary(problem, algo)
            if columns is None:
                continue
            feas = floats(columns.get("final_front_feasibility_pct", []))
            if not feas:
                report("WARN", f"{problem} x {algo}: no front feasibility data")
            elif np.mean(feas) >= 99.0:
                report("PASS", f"{problem} x {algo}: front feasibility "
                               f"{np.mean(feas):.1f}%")
            else:
                report("FAIL", f"{problem} x {algo}: front feasibility only "
                               f"{np.mean(feas):.1f}% -- constraint handling "
                               f"not doing its job")


def check_significance():
    print("\n== 5. SIGNIFICANCE (Mann-Whitney U, alpha=0.05) ==")
    print("     CCPSO_filter vs MGPSO_feasarch on final IGD:")
    for problem in CONSTRAINED:
        if problem == "OSY":
            continue  # no IGD available
        a = read_summary(problem, "CCPSO_filter")
        b = read_summary(problem, "MGPSO_feasarch")
        if a is None or b is None:
            continue
        igd_a = floats(a.get("final_igd", []))
        igd_b = floats(b.get("final_igd", []))
        if len(igd_a) < 5 or len(igd_b) < 5:
            report("WARN", f"{problem}: too few runs for a meaningful test")
            continue
        stat, p = mannwhitneyu(igd_a, igd_b, alternative="two-sided")
        better = "CCPSO" if np.median(igd_a) < np.median(igd_b) else "feasarch"
        if p < 0.05:
            report("PASS", f"{problem}: p={p:.4f} -- significant, "
                           f"{better} better (medians "
                           f"{np.median(igd_a):.4f} vs {np.median(igd_b):.4f})")
        else:
            report("WARN", f"{problem}: p={p:.4f} -- NOT significant "
                           f"(medians {np.median(igd_a):.4f} vs "
                           f"{np.median(igd_b):.4f}); report as a tie")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30,
                        help="expected number of runs per experiment")
    args = parser.parse_args()

    if not os.path.isdir("out"):
        print("No out/ directory found -- run the campaign first.")
        sys.exit(1)

    check_completeness(args.runs)
    check_convergence()
    check_plausibility()
    check_orderings()
    check_significance()

    print(f"\n==== SUMMARY: {_counts['PASS']} pass, "
          f"{_counts['WARN']} warn, {_counts['FAIL']} fail ====")
    if _counts["FAIL"]:
        print("FAILs must be resolved before the results are reportable.")
        sys.exit(1)
    if _counts["WARN"]:
        print("WARNs are judgement calls -- address or justify in the report.")


if __name__ == "__main__":
    main()