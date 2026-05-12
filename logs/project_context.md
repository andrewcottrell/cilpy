# Project Context — Honours Project 2026

## Project Title

**Co-evolutionary Multi-guide Particle Swarm Optimization Approach for Constrained Multi-objective Optimization Problems**

**Student:** A. Cottrell  
**Institution:** Stellenbosch University, Computer Science Division  
**Student number:** 26989395

---

## Overview

This is a year-long honours project extending the **cilpy** library (v0.0.2, originally by Willie Loftie-Eaton) to support solving of static and dynamic constrained multi-objective optimisation problems. The approach combines the **competitive co-evolutionary framework** (Pampàra & Engelbrecht, 2025) with a **multi-guide PSO (MGPSO)** (Scheepers et al., 2019).

The project is structured in phases of increasing complexity, working from single-objective static problems up to dynamic multi-objective problems.

---

## Key Deadlines

| Date | Milestone |
|------|-----------|
| 15 May 2026 | Report submission (draft 2) — **NEARLY COMPLETE** |
| Late May 2026 | Demo 1 — effectively a status update; target is single-objective optimisation |
| TBD | Demo 2 / final submission |

---

## Project Phases

| Phase | Problem Type | Objective | Constraint | Status |
|-------|-------------|-----------|------------|--------|
| **1 — SCSO** | Static objective, static constraint | Single | Static | **Complete — results in report** |
| **2 — DOSC** | Dynamic objective, static constraint | Single | Dynamic | Not started |
| **3 — SODC** | Static objective, dynamic constraint | Single | Dynamic | Not started |
| **4 — MO** | Any | Multi | Static + Dynamic | Deferred (MGPSO) |

**Demo 1 target:** Complete and validate Phase 1 (SCSO single-objective). ✓ Done.

---

## Core Approach

### Constraint Handling — Competitive Co-evolutionary Lagrangian Solver (CCLS)

The `CoevolutionaryLagrangianSolver` (CCLS) is the main research contribution for the single-objective phases. It uses **augmented Lagrangian relaxation** — no external constraint handlers (Deb feasibility rules or alpha constraint method) are used. The constraint handling emerges entirely from the coevolutionary multiplier dynamics plus the augmented penalty term.

**Four extensions made to the original cilpy CCLS:**

1. **Violation-aware initialisation** — multipliers seeded from constraint violations at lower bound rather than zeros
2. **Bounded multiplier search space** — `max_multiplier` cap replacing the original unbounded upper limit; found to be highly problem-dependent (CEC2006 requires up to 10000)
3. **Augmented Lagrangian penalty term** — `penalty_rho` added to close the duality gap on non-convex problems; theoretically justified as augmented Lagrangian extension
4. **Constraint value passthrough** — `get_result()` re-evaluates against original problem enabling feasibility tracking

Two sub-solvers run simultaneously each iteration:

- **Objective solver (P1):** PSO minimising `L(x, μ*) = f(x) + μ*·g(x) + ρ·Σmax(0,g(x))²`
- **Multiplier solver (P2):** PSO maximising `L(x*, μ) = f(x*) + μ·g(x*)`

**Theoretical grounding:** Augmented Lagrangian / Lagrangian duality. The saddle point theorem guarantees convergence to the constrained optimum when multipliers converge. For non-convex problems the duality gap exists; `penalty_rho` closes it.

### Multi-objective Phase — MGPSO (Deferred)

When Phase 4 begins, the approach will use a **multi-guide PSO** with:
- One sub-swarm per objective function
- A shared archive of non-dominated solutions that each sub-swarm contributes to
- Archive guide selection via tournament selection based on crowding distance
- The velocity update gains a third attractor term (archive guide)
- The co-evolutionary constraint handling wraps around the MGPSO

---

## cilpy Library Structure

```
cilpy/
├── cilpy/
│   ├── problem/
│   │   ├── __init__.py          # Problem & Evaluation interfaces
│   │   ├── unconstrained.py     # Sphere, Quadratic (Schwefel 1.2), Ackley
│   │   ├── constrained.py       # G01–G06 (CEC2006), C01, C02
│   │   ├── multi_objective.py   # SCH1 (Schaffer)
│   │   ├── mpb.py               # Moving Peaks Benchmark
│   │   └── cmpb.py              # Constrained Moving Peaks Benchmark (CMPB)
│   ├── solver/
│   │   ├── __init__.py          # Solver interface
│   │   ├── pso.py               # PSO, QPSO
│   │   ├── ga.py                # GA
│   │   ├── de.py                # DE/rand/1/bin
│   │   ├── ccls.py              # CoevolutionaryLagrangianSolver (CCLS) ← main contribution
│   │   └── chm/
│   │       ├── __init__.py      # ConstraintHandler interface, DefaultComparator
│   │       ├── deb_feasibility.py
│   │       └── alpha_constraint.py
│   └── runner.py                # ExperimentRunner
├── examples/
├── test/
├── out/                         # CSV benchmark results
└── docs/
```

---

## Benchmarking

### Production config (used for final 30-run results)
- `num_runs=30`, `max_iterations=1000`, `swarm_size=200` (both sub-solvers)
- `penalty_rho=0.5`, `penalty_rho_equality=0.5`, `max_multiplier=10000.0`
- Objective solver: `w=0.72, c1=1.49, c2=1.49` | Multiplier solver: `w=0.4, c1=1.2, c2=1.2`

### Phase 1 final results (30 runs, max_multiplier=10000)

| Problem | Mean Fitness | Std | Mean Feas% | Std Feas% | Mean P_RED | Known Opt |
|---------|-------------|-----|------------|-----------|------------|-----------|
| G01 | -5.8617 | 0.7575 | 99.97 | 0.18 | 0.6108 | -15.0 |
| G02 | -0.0716 | 0.0107 | 33.35 | 38.81 | 0.9108 | -0.8036 |
| G04 | -27888.15 | 4629.05 | 14.28 | 28.55 | 0.4479 | -30665.5 |
| G05 | 506.65 | 277.97 | 0.00 | 0.00 | 0.9480 | 5126.5 |
| G06 | 1241000.0 | 0.00 | 0.00 | 0.00 | 179.258 | -6961.8 |
| CMPB_SOSC | -55.7979 | 3.0019 | 99.60 | 1.39 | 0.2049 | — |
| CMPB_SODC | -68.1754 | 0.9370 | 100.00 | 0.00 | 0.0271 | — |
| CMPB_DOSC | -38.5721 | 21.9889 | 96.33 | 17.65 | 0.3928 | — |

G03 excluded — equality encoding (`|h| − ε ≤ 0`) interferes with multiplier dynamics.

### Key diagnostic findings

**Two distinct failure modes identified:**

**1. Premature convergence** (G01, CMPB_SOSC): solver finds a feasible region and stagnates. Diversity collapses to near-zero. Swarm converges to same local basin nearly every run (low std). Root cause: PSO diversity collapse, not constraint handling failure.

**2. Multiplier cap saturation** (G05, G06): 0% feasibility across all 30 runs. Constraint violation magnitudes exceed `max_multiplier=10000`. G06 P_RED of 179.258 (outside expected [0,1] range) confirms complete failure. Requires much higher cap or adaptive scheduling.

**Bimodal behaviour** (G02, G04): High std on feasibility (38.81, 28.55) reveals runs either find a feasible region or fail entirely depending on initialisation. Solver sensitive to initialisation on non-convex problems.

**CMPB headline result**: CMPB_SODC — 100% feasibility, P_RED 0.027, zero std. Best result across all problems. Validates that extended CCLS handles dynamic constraints correctly.

### analysis.py outputs
- `analysis_fitness_summary.csv` — mean/std/min/max final fitness per problem
- `analysis_feasibility_summary.csv` — mean/std/min/max feasibility per problem
- `analysis_p_red_summary.csv` — mean/std/min/max P_RED per problem
- `analysis_convergence_detail.csv` — per-run final state including diversity

---

## Report Status (as of 2026-05-11)

**Title:** Co-evolutionary Multi-Guide Particle Swarm Optimization for Constrained Multi-objective Optimization Problems

**All sections written. Remaining before submission:**
- Remove TODO comment (line 290)
- Change `\begin{table}[h]` to `\begin{table*}[t]` and `\end{table}` to `\end{table*}`
- Fix `c1`, `c2` to `$c_1$`, `$c_2$` in Section III (lines 358, 360)
- Fix "observed on G01 and the CMPB problems" → "observed on G01 and CMPB\_SOSC" in results
- Abstract tweak: replace "The resulting implementation provides a framework for solving dynamic constrained multi-objective problems" with "This report presents Phase 1 of this implementation, validating the constraint handling framework on single-objective problems"

**Section status:**
- Abstract ✓ (minor tweak needed)
- Section I (Introduction) ✓
- Section II (Background) ✓ — PSO, MGPSO, CCLS background with algorithms 1, 2, 3
- Section III (Implementation) ✓ — four CCLS extensions, algorithm, config, benchmarks
- Section IV (Results) ✓ — table, CMPB discussion, CEC2006 two failure modes, future work
- References ✓

**Supervisor feedback addressed:**
- Introduction expanded with narrative, gap, sub-goals, scope statement ✓
- Pseudocode added (Algorithms 1, 2, 3) ✓
- cilpy discussed in Section III ✓
- Sub-goals explicitly stated ✓
- Evaluation/comparison goal included ✓

---

## Key References

- Pampàra, G. & Engelbrecht, A. (2025). A co-evolutionary meta-heuristic framework for dynamic constrained optimization problems. *Soft Computing*. doi: 10.1007/s00500-025-10873-9.
- Scheepers, C., Engelbrecht, A. P., & Cleghorn, C. W. (2019). Multi-guide particle swarm optimization for multi-objective optimization: empirical and stability analysis. *Swarm Intelligence*, 13, 245–276. doi: 10.1007/s11721-019-00171-0.

---

## Notes for Future Sessions

- Claude has no memory between sessions. Paste this document at the start of any new session to restore full context.
- The STATUS.md file contains detailed API documentation for cilpy interfaces, the full problem catalogue, CCLS internals, ExperimentRunner details. Upload it alongside this file for full technical context.
- The report .tex file can be uploaded when working on the report.
- Next phase after report submission: Demo 1 prep (late May), then Phase 2 (DOSC) — requires implementing dynamic change detection in CCLS step().
