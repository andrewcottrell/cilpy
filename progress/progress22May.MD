# Progress Report - 2026-05-22

## Scope
- Codebase: cilpy repo and selected project files in /home/acottrell/Documents/Honours/Project
- Sources: git history, core .py files, logs/STATUS.md, logs/project_context.md, devlog.MD

## Current state summary
- Phase 1 (SCSO) complete with 30-run results and analysis.
- Phase 2 (DOSC) and Phase 3 (SODC) not started; Phase 4 (multi-objective MGPSO) deferred.
- Core contribution is CoevolutionaryLagrangianSolver (CCLS) in cilpy/solver/ccls.py.
- Dynamic handling in CCLS step() is not implemented (TODO) and blocks Phase 2/3.

## Git history snapshot (recent, with dates)
- 2026-05-12: progress (analysis script, benchmark example tweaks, logs refreshed).
- 2026-05-11: parameters tuned (STATUS/project_context/addtoreport logs updated).
- 2026-05-05: reverted from feasibility rules (CCLS changes, PSO/GA updates, example scripts).
- 2026-05-02: benchmarking and feasibility enforcement (CEC2006 fitness bounds added, Deb feasibility handler added).
- 2026-05-02: troubleshooting (PSO/GA comparator updates, CCLS adjustments, diagnostics).
- 2026-05-01: benchmark problems and tests added (CEC2006 work-in-progress; example scripts).

## Code-level changes (from git diffs)

### CCLS (cilpy/solver/ccls.py)
- Multiplier initialization seeded from constraint violations at lower bounds, not zeros.
- penalty_rho default in _LagrangianMinProblem set to 0.1 and penalty_rho_equality defaults to penalty_rho.
- Lagrangian penalty uses max(0, g) for inequality violations and abs(h) for equality violations.
- Dynamic handling TODO remains in step().

### PSO and GA (cilpy/solver/pso.py, cilpy/solver/ga.py)
- Both accept optional constraint_handler and pass it to Solver.
- PSO gbest initialization now uses comparator (constraint-aware) instead of raw fitness.

### Deb feasibility handler (cilpy/solver/chm/deb_feasibility.py)
- Added DebFeasibilityHandler implementation for feasibility-first comparisons.

### CEC2006 problems (cilpy/problem/constrained.py)
- Added get_fitness_bounds for G01-G06 to support P_RED metric.

## Current implementation snapshot (key points)

### CoevolutionaryLagrangianSolver
- Two solver populations: objective solver minimizes L(x, mu*, lambda*), multiplier solver maximizes L(x*, mu, lambda).
- _LagrangianMinProblem returns Lagrangian value plus penalty terms, and passes through constraints for feasibility metrics.
- _LagrangianMaxProblem returns negative Lagrangian to enable maximization with minimizer solvers.

### Experiment configuration (examples/ccpso_benchmark_exa.py)
- CCPSO configured with penalty_rho=0.5, penalty_rho_equality=0.5, max_multiplier=10000.
- Swarm size 200 for both sub-solvers.
- Current script runs only C01 (other problems defined but not used).
- Note: benchmark options in the script are not necessarily the settings used for prior runs/results.
- Historical runs include CMPB_SOSC, CMPB_SODC, and CMPB_DOSC.

### Analysis tooling (examples/analysis.py)
- Aggregates out/*.out.csv and out/*.summary.out.csv to produce P_RED, feasibility, fitness, and convergence summaries.
- No out/ directory present in repo currently.

## Results snapshot (Phase 1 final 30-run results)
| Problem | Mean Fitness | Std | Mean Feas% | Std Feas% | Mean P_RED | Known Opt |
| --- | --- | --- | --- | --- | --- | --- |
| G01 | -5.8617 | 0.7575 | 99.97 | 0.18 | 0.6108 | -15.0 |
| G02 | -0.0716 | 0.0107 | 33.35 | 38.81 | 0.9108 | -0.8036 |
| G04 | -27888.15 | 4629.05 | 14.28 | 28.55 | 0.4479 | -30665.5 |
| G05 | 506.65 | 277.97 | 0.00 | 0.00 | 0.9480 | 5126.5 |
| G06 | 1241000.0 | 0.00 | 0.00 | 0.00 | 179.258 | -6961.8 |
| CMPB_SOSC | -55.7979 | 3.0019 | 99.60 | 1.39 | 0.2049 | - |
| CMPB_SODC | -68.1754 | 0.9370 | 100.00 | 0.00 | 0.0271 | - |
| CMPB_DOSC | -38.5721 | 21.9889 | 96.33 | 17.65 | 0.3928 | - |

## Known limitations and risks
- Dynamic handling in CCLS step() is not implemented, blocking DOSC/SODC phases.
- Premature convergence (G01, CMPB_SOSC): diversity collapse and stagnation.
- Multiplier cap saturation (G05, G06): 0% feasibility with max_multiplier=10000.
- Bimodal feasibility on G02/G04 indicates sensitivity to initialization.
- G03 excluded due to equality encoding interacting poorly with multiplier dynamics.
- Tests for solvers are placeholders (test/solver/test_de.py, test_ga.py, test_pso.py are TODO only).
- docs/index.md still has TODO for specifying comparisons.

## Report status (from logs/STATUS.md and logs/project_context.md)
- Report is near-complete with minor TODOs remaining (table environment changes, c1/c2 formatting, abstract scope tweak, and one TODO comment removal).
- Phase 1 results already written; Phase 2 and beyond pending dynamic handling.

## Consolidated logs
- devlog/status files have been updated and combined in:
  - /home/acottrell/Documents/Honours/Project/cilpy/logs/devlog_status_combined.md

## Sources reviewed
- /home/acottrell/Documents/Honours/Project/cilpy/logs/devlog_status_combined.md
- git history and diffs for cilpy repo
- cilpy/solver/ccls.py, cilpy/solver/pso.py, cilpy/solver/ga.py
- cilpy/solver/chm/deb_feasibility.py, cilpy/problem/constrained.py
- examples/analysis.py, examples/ccpso_benchmark_exa.py
