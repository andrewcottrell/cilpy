"""The runner records the solver's problem evaluations per run."""
import csv
import os

from cilpy.problem.multi_objective import TNK
from cilpy.runner import ExperimentRunner
from cilpy.solver.mgpso import MGPSO


def test_summary_records_evaluations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    problem = TNK()
    ExperimentRunner(
        problems=[problem],
        solver_configurations=[{"class": MGPSO, "params": {
            "name": "M", "swarm_size": 10, "w": 0.72, "c1": 1.49,
            "c2": 1.49, "c3": 1.49}}],
        num_runs=2,
        max_iterations=5,
    ).run_experiments()
    with open(os.path.join("out", "TNK_M.summary.out.csv"), newline="") as f:
        reader = csv.reader(f)
        header = [h.split("[")[0].strip() for h in next(reader)]
        rows = list(reader)
    idx = header.index("evaluations")
    # Static problem: one probe at construction plus 2 x 10 particles at
    # construction and in each of the 5 iterations, and nothing else.
    assert [int(r[idx]) for r in rows] == [int(rows[0][idx])] * 2
    assert int(rows[0][idx]) >= 6 * 20
    # The wrapper is removed after each run.
    assert "evaluate" not in vars(problem)
