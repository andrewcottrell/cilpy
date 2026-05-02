# Experimental Changes Log

## 2026-05-01

### Motivation
The competitive coevolutionary Lagrangian solver was consistently selecting infeasible solutions for CEC2006 problems. The multiplier search used infinite bounds, which makes GA initialization and mutation unstable, and the Lagrangian penalized negative (feasible) constraint values. These low-risk changes stabilize multiplier evolution and focus penalties on violations without changing the overall competitive coevolutionary structure.

### Changes Made
- Updated [cilpy/solver/ccls.py](../../cilpy/solver/ccls.py):
  - Clamp Lagrange multiplier bounds to a finite range via `max_multiplier`.
  - Penalize only positive inequality constraint violations using `max(0, g)`.
  - Add a `penalty_scale` hook for controlled penalty amplification.
  - Square inequality violations in the Lagrangian penalty to increase pressure on infeasible solutions.
  - Raise default `max_multiplier` and `penalty_scale` to strengthen penalties.

### How To Revert
1. If using git, run:
   - `git checkout -- cilpy/solver/ccls.py`
   - `git checkout -- docs/dev/experiment_changes.md`
2. If not using git, manually revert the edits in [cilpy/solver/ccls.py](../../cilpy/solver/ccls.py) and delete this file.

### Notes
Defaults are conservative (`max_multiplier=100.0`, `penalty_scale=1.0`). If feasibility is still poor, increase `max_multiplier` or `penalty_scale` when constructing `CoevolutionaryLagrangianSolver`.
