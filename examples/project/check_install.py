# examples/project/check_install.py
"""Pre-flight check: verifies the repository has the CURRENT versions of all
project components before launching a campaign. Run this on any machine
BEFORE starting experiments:

    python examples/project/check_install.py

Exits non-zero with a clear message if any stale file is detected. This
exists because the dynamic tau_t sweeps were once run against an old
runner/mgpso pair, silently producing MIGD values ~10x too large (frozen
t=0 reference fronts and never-rescored personal bests).
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

failures = []

def check(label, condition):
    print(f"  [{'OK' if condition else 'STALE'}] {label}")
    if not condition:
        failures.append(label)

print("== Component version check ==")

# --- MGPSO: change response machinery (Milestone 5) ---
from cilpy.solver import mgpso
check("mgpso: sentinel change detection (_environment_changed)",
      hasattr(mgpso.MGPSO, "_environment_changed"))
check("mgpso: full change response with pbest re-scoring (_respond_to_change)",
      hasattr(mgpso.MGPSO, "_respond_to_change"))
check("mgpso: feasible-only archive admission flag",
      "feasible_archive_only" in inspect.signature(mgpso.MGPSO.__init__).parameters)
check("mgpso: cached/vectorised archive (performance fix)",
      hasattr(mgpso._Archive, "_matrix"))

# --- Runner: dynamic support + honest reporting ---
from cilpy import runner as runner_mod
runner_src = inspect.getsource(runner_mod)
check("runner: per-iteration reference-front refresh for dynamic problems",
      "true_pareto_front()" in runner_src and "is_dynamic()" in runner_src)
check("runner: problem time reset between runs",
      "reset_time" in runner_src)
check("runner: feasible-front column for honest hypervolume",
      "final_feasible_front" in runner_src)

# --- CCLS: multi-objective support ---
from cilpy.solver import ccls
ccls_src = inspect.getsource(ccls)
# Behavioural test: with a large multiplier, a DEEPLY FEASIBLE solution
# must receive ZERO penalty (clamped form), not a large negative one
# (signed form, which collapses the archive to a single point).
from cilpy.problem.multi_objective import CONSTR as _CONSTR
_proxy = ccls._LagrangianMinProblem(_CONSTR(), penalty_rho=0.0)
_proxy.set_fixed_multipliers([1000.0, 1000.0], [])
_x = [1.0, 0.0]                      # feasible, well inside the region
_raw = _CONSTR().evaluate(_x).fitness
_pen = _proxy.evaluate(_x).fitness
check("ccls: violation-clamped multiplier term in MO proxy "
      "(feasible solutions unpenalised)",
      all(abs(p - r) < 1e-9 for p, r in zip(_pen, _raw)))
check("ccls: archive_strategy flag",
      "archive_strategy" in ccls_src)
check("ccls: violation-aware multiplier anchor",
      hasattr(ccls.CoevolutionaryLagrangianSolver, "_select_multiplier_anchor"))

# --- Problems ---
try:
    from cilpy.problem.dynamic_multi_objective import FDA1, FDA3, DTNK, DTNK2, DTNK3
    check("problems: all five dynamic MO problems incl. DTNK3 (DOSC)", True)
except ImportError:
    check("problems: all five dynamic MO problems incl. DTNK3 (DOSC)", False)

# --- Functional smoke: correct dynamic behaviour end to end ---
print("\n== Functional smoke test (few seconds) ==")
import numpy as np, random
random.seed(0); np.random.seed(0)
p = FDA1(dimension=5, tau_t=3, n_t=10)
ref0 = p.true_pareto_front(50)
s = mgpso.MGPSO(problem=p, name="chk", swarm_size=8)
old_pbest = None
changed_detected = False
for it in range(1, 8):
    p.begin_iteration()
    if it == 4:
        old_pbest = [pt.y_eval.fitness for sw in s._swarms for pt in sw.particles]
    s.step()
    if it == 4:
        new_pbest = [pt.y_eval.fitness for sw in s._swarms for pt in sw.particles]
        changed_detected = any(
            o != n for o, n in zip(old_pbest, new_pbest)
        )
check("smoke: pbests re-scored after an environment change", changed_detected)
p.reset_time()
check("smoke: reset_time() returns problem to t=0", p.t == 0.0)

print()
if failures:
    print(f"FAILED: {len(failures)} stale component(s). Extract the latest "
          f"project tarball over the repository root before running anything.")
    sys.exit(1)
print("ALL CURRENT — safe to launch campaigns.")