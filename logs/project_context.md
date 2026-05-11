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
| 15 May 2026 | Report submission (draft 2) |
| Late May 2026 | Demo 1 — effectively a status update; target is single-objective optimisation |
| TBD | Demo 2 / final submission |

---

## Project Phases

| Phase | Problem Type | Objective | Constraint | Status |
|-------|-------------|-----------|------------|--------|
| **1 — SCSO** | Static objective, static constraint | Single | Static | **In progress** |
| **2 — DOSC** | Dynamic objective, static constraint | Single | Dynamic | Not started |
| **3 — SODC** | Static objective, dynamic constraint | Single | Dynamic | Not started |
| **4 — MO** | Any | Multi | Static + Dynamic | Deferred (MGPSO) |

**Demo 1 target:** Complete and validate Phase 1 (SCSO single-objective).

---

## Core Approach

### Constraint Handling — Competitive Co-evolutionary Lagrangian Solver (CCLS)

The `CoevolutionaryLagrangianSolver` (CCLS) is the main research contribution for the single-objective phases. It uses **pure Lagrangian relaxation** — no external constraint handlers (Deb feasibility rules or alpha constraint method) are used. The constraint handling emerges entirely from the coevolutionary multiplier dynamics.

Two sub-solvers run simultaneously each iteration:

- **Objective solver (P1):** A PSO that minimises the Lagrangian `L(x, μ*) = f(x) + μ*·g(x)` over the solution space, with multipliers held fixed at the current best from P2.
- **Multiplier solver (P2):** A PSO that maximises `L(x*, μ)` over the multiplier space, with the best solution held fixed at the current best from P1.

Each step: get best x* and μ* → update proxy problems → objective solver steps → multiplier solver steps. This drives both populations toward the saddle point simultaneously.

The augmented penalty term `penalty_rho` is added on top of the Lagrangian to improve convergence on non-convex problems (closing the duality gap). This makes it an **augmented Lagrangian** approach, which is more robust than pure Lagrangian for the non-convex CEC2006 benchmarks.

**Theoretical grounding:** Lagrangian relaxation / Lagrangian duality. The saddle point theorem guarantees that if multipliers converge to their optimal values, the Lagrangian optimum coincides with the true constrained optimum. For non-convex problems a duality gap exists, which `penalty_rho` helps close.

The `DebFeasibilityHandler` and `AlphaConstraintHandler` exist in the cilpy codebase but are **explicitly out of scope** for this project's research contribution.

### Multi-objective Phase — MGPSO (Deferred)

When Phase 4 begins, the approach will use a **multi-guide PSO** with:
- One sub-swarm per objective function
- A shared archive of non-dominated solutions that each sub-swarm contributes to
- Archive guide selection via tournament selection based on crowding distance, drawing swarms toward sparsely populated regions of the Pareto front
- The velocity update gains a third attractor term (archive guide) in addition to personal best and neighbourhood best
- The co-evolutionary constraint handling wraps around the MGPSO in the same way it wraps around PSO for single-objective

Multi-objective problems will also be either static or dynamic (the DCOP four-category taxonomy applies).

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

### Sanity-check config (1 run, used for diagnosis)
- `num_runs=1`, `max_iterations=1000`, `swarm_size=200` (both sub-solvers)
- `penalty_rho=0.5` (default), `penalty_rho_equality=0.5`, `max_multiplier=100.0` (default)
- Objective solver: `w=0.72, c1=1.49, c2=1.49` | Multiplier solver: `w=0.4, c1=1.2, c2=1.2`

Full 30-run production config is the same but `num_runs=30`.

### Phase 1 results status

| Problem | Best fitness (1 run) | Feasibility | P_RED | Notes |
|---------|----------------------|-------------|-------|-------|
| G01 | ~-6 | 100% | ~0.63 | Feasible but far from optimum (-15); premature convergence |
| G02 | ~-0.067 | ~93% | — | Feasible but far from optimum (-0.8036); premature convergence |
| G03 | skipped | — | — | Equality encoding concern |
| G04 | ~-22335 | ~10-15% | ~0.219 | Local saddle point; see diagnosis below |
| G05 | ~640 | 0% | — | 0% feasible; multiplier dynamics failing |
| G06 | ~868090 | 0% | — | 0% feasible; multiplier dynamics failing |
| CMPB_SOSC | ~-37.5 | 100% | ~0.07 | Diversity collapsed (2.7e-14); premature convergence |

### Key diagnostic findings

**Two distinct failure modes identified:**

**1. Feasible but suboptimal** (G01, G02, CMPB_SOSC): solver finds a feasible region and stagnates. Diversity collapses. Personal bests cluster and Lagrangian gradient vanishes. Root cause: premature convergence in objective PSO.

**2. Unable to reach feasibility** (G05, G06 at default config): `max_multiplier=100` too low for the constraint magnitudes of these problems. Multiplier PSO hits the cap and cannot enforce constraints sufficiently.

**G04 special case**: Required `max_multiplier=10000` to achieve any feasibility (default 100 → 0% feasible). With 10000, solver reaches 10-15% feasibility and fitness ~-22335 (optimum -30665) but then stagnates at a **local saddle point** — best solution frozen from ~iteration 690 regardless of `penalty_rho` or inertia tuning. Confirmed not a constraint enforcement issue — it is a local optimum in the Lagrangian landscape.

**Hyperparameter sensitivity findings:**
- `max_multiplier` is problem-dependent. G04/G05/G06 need >> 100; default 100 works for G01/G02/CMPB_SOSC.
- `penalty_rho` is theoretically justified as part of the augmented Lagrangian (closes duality gap on non-convex problems) but should be a stabiliser, not the primary enforcement mechanism. If results only work with very high `penalty_rho`, the multiplier population is not doing its job.
- Lowering objective solver inertia (`w=0.4`) maintains diversity but reduces momentum — solver oscillates around feasible boundary rather than converging into it. Original split config (obj: `w=0.72`, mul: `w=0.4`) is the correct design intent.

### Benchmark strategy (agreed)
- CEC2006 G01, G04 as representatives (different constraint structures/dimensionality)
- G03 explicitly scoped out — equality encoding limitation, documented as such
- G05, G06 need `max_multiplier` investigation before inclusion
- CMPB_SOSC is the primary novel benchmark — best result (~0.07 P_RED)
- Results narrative: CCLS correctly handles constraints; premature convergence in the underlying PSO is the dominant limitation; motivates diversity maintenance / restart mechanisms as future work

---

## Report Status

**Title:** Co-evolutionary Multi-Guide Particle Swarm Optimization for Constrained Multi-objective Optimization Problems

**Supervisor feedback (April 2026 draft):**
1. Introduction needs expanding — currently as short as the abstract. Should follow: context → constrained MOP problem → what currently exists (MGPSO for box-constrained MOPs; coevolutionary framework for single-objective DCOPs; cilpy as the existing library) → project goal.
2. Sub-goals must be explicitly stated: (a) implementation of multi-objective benchmark problems; (b) implementation of performance metrics for multi-objective and constrained multi-objective optimisation.
3. Evaluation/comparison goal missing — does the project also aim to evaluate the resulting algorithm against other approaches?
4. Section III (implementation) and Section IV (results/discussion) not yet written.
5. Minor: Background opening paragraph duplicated; draft-mode text left in introduction ("For demo one..."); TODO comment left in source.
6. Suggestions to add pseudocode and a discussion of cilpy.

**Current report content:**
- Abstract: done (needs minor update to reflect phased scope)
- Section I (Introduction): needs full rewrite per feedback
- Section II (Background): largely done — CMOP definitions, PSO/MGPSO, co-evolutionary framework, all with correct math and referencing. Fix: remove duplicated opening paragraph, remove TODO comment.
- Section III (Implementation): not written — enough information now exists to write this
- Section IV (Results/Discussion): not written — enough information now exists to write Phase 1 results

**Key mismatch to manage:** Report title/abstract describe a multi-objective system, but current implementation is single-objective CCLS only. Section III must frame this honestly as a phased project — Phase 1 (SOSC/CCLS) is the demo 1 target; MGPSO is the planned extension.

**What can be written now (as of 2026-05-10):**
- Introduction: full rewrite possible — all context, motivation, sub-goals, and project structure are known
- Section III (Implementation): CCLS architecture, two-solver step procedure, augmented Lagrangian formulation, cilpy library structure, ExperimentRunner, benchmark problems used (G01, G02, G04, CMPB_SOSC), experimental config, metrics (P_RED, feasibility, diversity)
- Section IV (Results): Phase 1 single-objective results — two failure modes (premature convergence vs inability to reach feasibility), G04 local saddle point finding, CMPB_SOSC as best result, hyperparameter sensitivity (max_multiplier problem-dependence, penalty_rho role)
- Discussion: premature convergence as dominant limitation, augmented Lagrangian justification, max_multiplier sensitivity as finding, future work (diversity maintenance, restart mechanisms, QPSO as objective solver, adaptive penalty scheduling)

---

## Key References

- Pampàra, G. & Engelbrecht, A. (2025). A co-evolutionary meta-heuristic framework for dynamic constrained optimization problems. *Soft Computing*. doi: 10.1007/s00500-025-10873-9.
- Scheepers, C., Engelbrecht, A. P., & Cleghorn, C. W. (2019). Multi-guide particle swarm optimization for multi-objective optimization: empirical and stability analysis. *Swarm Intelligence*, 13, 245–276. doi: 10.1007/s11721-019-00171-0.

---

## Notes for Future Sessions

- Claude has no memory between sessions. Paste this document at the start of any new session to restore full context.
- The STATUS.md file (last updated 2026-05-10) contains detailed API documentation for cilpy interfaces, the full problem catalogue, CCLS internals, ExperimentRunner details, and the open issues list. Upload it alongside this file for full technical context.
- The report .tex file and supervisor feedback PDF can also be uploaded when working on the report.
