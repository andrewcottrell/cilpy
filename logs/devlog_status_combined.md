# Consolidated Devlog and Status

Last updated: 2026-05-22

## Sources merged
- /home/acottrell/Documents/Honours/Project/devlog.MD
- /home/acottrell/Documents/Honours/Project/cilpy/logs/STATUS.md
- /home/acottrell/Documents/Honours/Project/cilpy/logs/project_context.md

---

## Dev Log

### 30 April 2026
- fixed co1 in problems

### 1 May 2026
- implemented G02 - G06 in constrained.py
- chasing bugs as to why GA and PSO are not finding solutions to G01 problem (seems to be a library issue)

### 2 May 2026
- added Deb feasibility handler and constraint-aware comparisons in GA/PSO
- added fitness bounds for CEC2006 problems to support P_RED
- troubleshooting CCLS and experiment scripts

### 5 May 2026
- reverted from feasibility rules and refocused on CCLS Lagrangian handling
- updated benchmark scripts

### 11 May 2026
- tuned CCPSO parameters; added STATUS/project context notes

### 12 May 2026
- added analysis script; refreshed progress notes and benchmarks

### 22 May 2026
- audited git history and code state for progress consolidation
- updated logs to note that current benchmark script options may not match prior runs
- confirmed CMPB runs completed for SOSC, SODC, and DOSC even if current script settings differ

---

## Project Context (condensed)

### Overview
- Honours project extends cilpy (v0.0.2) for constrained and dynamic single-objective optimization with a competitive coevolutionary framework.
- Multi-objective MGPSO extension is deferred until the coevolutionary component is validated.

### Phases
| Phase | Problem Type | Objective | Constraint | Status |
| --- | --- | --- | --- | --- |
| 1 - SCSO | Static obj, static constraint | Single | Static | Complete (30-run results) |
| 2 - DOSC | Dynamic obj, static constraint | Single | Static | Not started |
| 3 - SODC | Static obj, dynamic constraint | Single | Dynamic | Not started |
| 4 - MO | Any | Multi | TBD | Deferred |

### Core approach (CCLS)
- CoevolutionaryLagrangianSolver uses two solver populations: objective solver (minimizes Lagrangian) and multiplier solver (maximizes Lagrangian).
- Extensions used in this project:
  - multiplier seeding from constraint violations at lower bounds
  - bounded multiplier search space (max_multiplier)
  - augmented Lagrangian penalty (penalty_rho, penalty_rho_equality)
  - passthrough of constraints for feasibility tracking

---

## Project Status (condensed)

### CCLS snapshot
- CCLS is the primary research contribution in cilpy/solver/ccls.py.
- _LagrangianMinProblem evaluates L(x, mu*, lambda*) and adds penalty terms for violations.
- _LagrangianMaxProblem evaluates -L(x*, mu, lambda) for solver compatibility.
- Dynamic handling in CCLS step() is not implemented yet (TODO).

### Benchmarking configuration
- num_runs=30, max_iterations=1000, swarm_size=200
- penalty_rho=0.5, penalty_rho_equality=0.5, max_multiplier=10000
- Objective solver PSO (w=0.72, c1=1.49, c2=1.49)
- Multiplier solver PSO (w=0.4, c1=1.2, c2=1.2)
- Note: current benchmark script options may not reflect historical runs (see examples/ccpso_benchmark_exa.py)
- Historical runs include CMPB_SOSC, CMPB_SODC, and CMPB_DOSC

### Phase 1 results (final 30 runs)
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

### Known limitations and failure modes
- Premature convergence (G01, CMPB_SOSC): diversity collapse and stagnation.
- Multiplier cap saturation (G05, G06): 0% feasibility with max_multiplier=10000.
- Bimodal behavior (G02, G04): high feasibility variance across runs.
- Dynamic handling in CCLS step() not implemented.
- G03 excluded due to equality encoding interfering with multiplier dynamics.

### Open issues and next steps
- Implement dynamic change detection/response in CCLS step() and hook begin_iteration().
- Evaluate CCLS on CMPB_DOSC and CMPB_SODC once dynamic handling is in place.
- Consider QPSO or inertia decay/restart strategies to reduce premature convergence.
- Consider adaptive max_multiplier scheduling for G05/G06.

### Out of scope
- Deb feasibility rules and repair mechanisms are not part of the research contribution.
- The compare module remains a placeholder.

---

## Recent Code Changes (Git history snapshot)
- 2026-05-12: Added analysis tooling; updated benchmark example and logs.
- 2026-05-11: Tuned CCPSO parameters; expanded status/context notes.
- 2026-05-05: Reverted feasibility-rule focus in CCLS; updated benchmark scripts.
- 2026-05-02: Added Deb feasibility handler; GA/PSO accept constraint handlers; CEC2006 fitness bounds added for P_RED.
