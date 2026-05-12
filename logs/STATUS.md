# cilpy — Project Status

> Honours Project Extension: Competitive Coevolutionary Framework for Constrained Optimisation  
> Last updated: 2026-05-11

---

## 1. Project Context

This project extends the **cilpy** library (v0.0.2, originally by Willie Loftie-Eaton) to handle constrained and dynamic single-objective optimisation problems using a **competitive coevolutionary framework** — specifically the `CoevolutionaryLagrangianSolver` (CCLS).

The second half of the project will introduce a **multi-guide PSO with archive guide** for multi-objective problems, but this is deferred until the coevolutionary component is validated.

The project explicitly **excludes** feasibility rules as a constraint-handling mechanism. The scope is entirely around the coevolutionary Lagrangian approach. The `DebFeasibilityHandler` and `AlphaConstraintHandler` remain in the codebase but are not part of the research contribution.

---

## 2. Development Phases

| Phase | Problem Type | Objective | Constraint | Status |
|-------|-------------|-----------|------------|--------|
| **1 — SCSO** | Static obj, Static constraint | Single | Static | **Complete — 30-run results in report** |
| **2 — DOSC** | Dynamic obj, Static constraint | Single | Static | Not started |
| **3 — SODC** | Static obj, Dynamic constraint | Single | Dynamic | Not started |
| **4 — MO** | Any | Multi | TBD | Deferred (multi-guide PSO) |

**Current goal**: Report submission 15 May 2026. Phase 1 complete. Next: Demo 1 (late May), then Phase 2 (DOSC).

---

## 3. Directory Structure

```
cilpy/
├── cilpy/
│   ├── problem/
│   │   ├── __init__.py          # Problem & Evaluation interfaces
│   │   ├── unconstrained.py     # Sphere, Quadratic (Schwefel 1.2), Ackley
│   │   ├── constrained.py       # G01–G06 (CEC2006), C01, C02
│   │   ├── multi_objective.py   # SCH1 (Schaffer)
│   │   ├── mpb.py               # Moving Peaks Benchmark
│   │   └── cmpb.py              # Constrained Moving Peaks Benchmark
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
├── examples/                    # Runnable experiment scripts
├── test/                        # pytest suite
├── out/                         # CSV benchmark results
└── docs/                        # MkDocs documentation
```

---

## 4. Core Interfaces

### 4.1 `Evaluation` (dataclass)
*`cilpy/problem/__init__.py`*

```python
@dataclass
class Evaluation(Generic[FitnessType]):
    fitness: FitnessType                              # float (single-obj) or List[float] (multi-obj)
    constraints_inequality: Optional[List[float]]     # g(x) ≤ 0 convention; positive = violation
    constraints_equality: Optional[List[float]]       # h(x) = 0 convention; non-zero = violation
```

---

### 4.2 `Problem` (abstract base class)
*`cilpy/problem/__init__.py`*

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Problem identifier |
| `dimension` | `int` | Number of decision variables |
| `bounds` | `Tuple[List[float], List[float]]` | `(lower_bounds, upper_bounds)` per dimension |

| Method | Signature | Returns | Notes |
|--------|-----------|---------|-------|
| `__init__` | `(dimension, bounds, name)` | — | Must call `super().__init__()` |
| `evaluate` | `(solution: SolutionType)` | `Evaluation[FitnessType]` | Core evaluation |
| `is_dynamic` | `()` | `Tuple[bool, bool]` | `(obj_dynamic, constraint_dynamic)` |
| `is_multi_objective` | `()` | `bool` | — |
| `begin_iteration` | `()` | `None` | Called by runner before each step; override for dynamic problems |
| `get_fitness_bounds` | `()` | `Tuple[FitnessType, FitnessType]` | `(f_min, f_max)`; required for P_RED metric |

---

### 4.3 `Solver` (abstract base class)
*`cilpy/solver/__init__.py`*

| Attribute | Type | Description |
|-----------|------|-------------|
| `problem` | `Problem` | The problem being solved |
| `name` | `str` | Solver identifier |
| `comparator` | `ConstraintHandler` | Comparison strategy (defaults to `DefaultComparator`) |

| Method | Signature | Returns | Notes |
|--------|-----------|---------|-------|
| `__init__` | `(problem, name, constraint_handler=None, **kwargs)` | — | Must call `super().__init__()` |
| `step` | `()` | `None` | One algorithm iteration |
| `get_result` | `()` | `List[Tuple[SolutionType, Evaluation]]` | Best solution(s) so far |
| `get_population` | `()` | `List[SolutionType]` | Full population (optional; needed for diversity metric) |
| `get_population_evaluations` | `()` | `List[Evaluation]` | Population evaluations (optional; needed for feasibility metric) |

---

### 4.4 `ConstraintHandler` (abstract base class)
*`cilpy/solver/chm/__init__.py`*

| Method | Signature | Returns |
|--------|-----------|---------|
| `is_better` | `(eval_a: Evaluation, eval_b: Evaluation)` | `bool` — True if `eval_a` is better |

**Implementations**:

| Class | File | Strategy |
|-------|------|----------|
| `DefaultComparator` | `chm/__init__.py` | Raw fitness comparison (lower is better), ignores constraints |
| `DebFeasibilityHandler` | `chm/deb_feasibility.py` | Deb's rules: feasible > infeasible; lower violation when both infeasible |
| `AlphaConstraintHandler` | `chm/alpha_constraint.py` | Satisfaction level μ(x) ∈ [0,1] gated by threshold α |

> Note: `DebFeasibilityHandler` and `AlphaConstraintHandler` are **out of scope** for this project's research contribution. They exist in the library for completeness.

---

## 5. Problem Catalogue

### 5.1 Unconstrained (`cilpy/problem/unconstrained.py`)

| Class | `__init__` args | Dim | Domain | Optimum | `get_fitness_bounds` |
|-------|-----------------|-----|--------|---------|----------------------|
| `Sphere` | `dimension` | N | [−100, 100] | f(0)=0 | ✓ |
| `Quadratic` | `dimension` | N | [−100, 100] | f(0)=0 | ✓ |
| `Ackley` | `dimension` | N | [−32, 32] | f(0)=0 | ✓ |

### 5.2 Constrained CEC2006 (`cilpy/problem/constrained.py`)

| Class | `__init__` args | Dim | Ineq | Eq | f* | `get_fitness_bounds` |
|-------|-----------------|-----|------|----|----|----------------------|
| `G01` | *(none)* | 13 | 9 | 0 | −15.0 | `(−15.0, 0.0)` |
| `G02` | *(none)* | 20 | 2 | 0 | −0.8036 | `(−0.803619, 0.0)` |
| `G03` | *(none)* | 10 | 1* | 0 | −1.0005 | `(−1.0, 0.0)` |
| `G04` | *(none)* | 5 | 6 | 0 | −30665.54 | `(−30665.539, −20000.0)` |
| `G05` | *(none)* | 4 | 2+3* | 0 | 5126.497 | `(5126.498, 10000.0)` |
| `G06` | *(none)* | 2 | 2 | 0 | −6961.814 | `(−6961.814, 0.0)` |
| `C01` | *(none)* | 2 | 2 | 0 | 0.25 | ✗ |
| `C02` | *(none)* | 2 | 2 | 0 | 1.0 | ✗ |

\* G03 equality is converted to inequality `|h(x)| − ε ≤ 0`. G05 equality constraints are similarly encoded as inequality constraints in the `constraints_inequality` list.

> **G03 currently skipped** in benchmark runs. G06 not yet included in the latest benchmark run.

### 5.3 Dynamic (`cilpy/problem/mpb.py`, `cmpb.py`)

**`MovingPeaksBenchmark`**

| `__init__` arg | Type | Default | Description |
|----------------|------|---------|-------------|
| `dimension` | `int` | — | Search space dimension |
| `num_peaks` | `int` | — | Number of landscape peaks |
| `domain` | `Tuple[float,float]` | — | `(lower, upper)` scalar bound |
| `peak_height` | `float` | — | Initial peak heights |
| `peak_width` | `float` | — | Initial peak widths |
| `change_frequency` | `int` | — | Iterations between landscape changes |
| `shift_severity` | `float` | — | How far peaks move per change |
| `... ` | | | (see mpb.py for full parameter set) |

28 named configurations available via `generate_mpb_configs(dimension)`:
- **STA**: Static (no changes) — used as the static surrogate for SCSO/SODC
- **P/A/C + 1/2/3 + L/C/R**: Progressive/Abrupt/Chaotic × location/height/both × Linear/Circular/Random movement

**`ConstrainedMovingPeaksBenchmark`**

Composed of two independent MPB instances: objective landscape `f` and constraint landscape `g`.  
Optimises `min(g − f)` subject to `g − f ≤ 0`.

| `__init__` arg | Type | Description |
|----------------|------|-------------|
| `f_params` | `dict` | MPB config dict for the objective landscape |
| `g_params` | `dict` | MPB config dict for the constraint landscape |
| `name` | `str` | Problem instance name |

`is_dynamic()` reflects which landscapes are dynamic (can be mixed).

---

## 6. Solver Catalogue

### 6.1 `PSO` (`cilpy/solver/pso.py`)

| `__init__` arg | Type | Default | Description |
|----------------|------|---------|-------------|
| `problem` | `Problem` | — | Problem instance |
| `name` | `str` | — | Solver name |
| `swarm_size` | `int` | — | Number of particles |
| `w` | `float` | — | Inertia weight |
| `c1` | `float` | — | Cognitive (personal best) coefficient |
| `c2` | `float` | — | Social (global best) coefficient |
| `constraint_handler` | `ConstraintHandler` | `None` | Comparison strategy |

Velocity update: `v = w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)`

### 6.2 `QPSO` (`cilpy/solver/pso.py`)

Extends PSO with a quantum subgroup for dynamic problems.

| Additional arg | Type | Description |
|----------------|------|-------------|
| `split_ratio` | `float` | Fraction of particles using standard PSO update |
| `r_cloud` | `float` | Radius of quantum hypersphere around gbest |

Quantum particles sample uniformly from a sphere of radius `r_cloud` centred on gbest.

### 6.3 `GA` (`cilpy/solver/ga.py`)

| `__init__` arg | Type | Default | Description |
|----------------|------|---------|-------------|
| `problem` | `Problem` | — | Problem instance |
| `name` | `str` | — | Solver name |
| `population_size` | `int` | — | Number of individuals |
| `crossover_rate` | `float` | — | Probability of crossover |
| `mutation_rate` | `float` | — | Per-gene mutation probability |
| `tournament_size` | `int` | `2` | Tournament selection size |
| `constraint_handler` | `ConstraintHandler` | `None` | Comparison strategy |

Single-point crossover; Gaussian mutation with σ = 0.1 × domain_range; elitism (best carried over).

### 6.4 `DE` (`cilpy/solver/de.py`)

Strategy: DE/rand/1/bin.

| `__init__` arg | Type | Default | Description |
|----------------|------|---------|-------------|
| `problem` | `Problem` | — | Problem instance |
| `name` | `str` | — | Solver name |
| `population_size` | `int` | — | Population size |
| `crossover_rate` | `float` | — | CR parameter |
| `f_weight` | `float` | — | Differential weight F |
| `constraint_handler` | `ConstraintHandler` | `None` | Comparison strategy |

Mutation: `donor = x_r1 + F·(x_r2 − x_r3)`

---

## 7. CCLS — CoevolutionaryLagrangianSolver

*`cilpy/solver/ccls.py`* — **Primary research contribution.**

### 7.1 Theory

Transforms constrained problem into min-max:

```
Original:   min f(x)   s.t.  g_i(x) ≤ 0,  h_j(x) = 0
Lagrangian: L(x, μ, λ) = f(x) + Σ μ_i·g_i(x) + Σ λ_j·h_j(x)
Min-max:    min_x  max_{μ≥0, λ}  L(x, μ, λ)
```

Two cooperating sub-populations:
- **Objective solver** — minimises `L(x, μ*, λ*)` with fixed multipliers from the multiplier solver
- **Multiplier solver** — maximises `L(x*, μ, λ)` with fixed solution from the objective solver

### 7.2 `CoevolutionaryLagrangianSolver` constructor

| Arg | Type | Required | Description |
|-----|------|----------|-------------|
| `name` | `str` | ✓ | Solver identifier |
| `problem` | `Problem` | ✓ | Original constrained problem |
| `objective_solver_class` | `Type[Solver]` | ✓ | Solver class for solution space (e.g. `PSO`, `GA`) |
| `multiplier_solver_class` | `Type[Solver]` | ✓ | Solver class for multiplier space |
| `objective_solver_params` | `dict` | ✓ | Init kwargs for objective solver (excluding `problem`) |
| `multiplier_solver_params` | `dict` | ✓ | Init kwargs for multiplier solver (excluding `problem`) |
| `penalty_rho` | `float` | `**kwargs`, default `0.0` | Penalty coefficient on inequality violations in `L_min`; `rho·Σmax(0,g_i)` |
| `penalty_rho_equality` | `float` | `**kwargs`, default `= penalty_rho` | Penalty on equality violations; `rho·Σ\|h_j\|` |
| `max_multiplier` | `float` | `**kwargs`, default `None` | Upper bound for multiplier values (clamps μ and λ); `None` = unbounded |
| `constraint_handler` | `ConstraintHandler` | `**kwargs`, default `None` | Passed to objective solver only |

> **Note**: `penalty_rho`, `penalty_rho_equality`, `max_multiplier`, and `constraint_handler` are passed as `**kwargs` to `CoevolutionaryLagrangianSolver.__init__`, not as top-level positional arguments.

### 7.3 Internal proxy problems

**`_LagrangianMinProblem`** (objective space)
- Evaluates `L(x, μ*, λ*) + rho·Σmax(0,g_i) + rho_eq·Σ|h_j|`
- Multipliers initialised from lower-bounds evaluation at construction
- Updated each step via `set_fixed_multipliers(inequality_multipliers, equality_multipliers)`
- Passes through original constraint values in returned `Evaluation` (for feasibility tracking)

**`_LagrangianMaxProblem`** (multiplier space)
- Dimension = num_inequality + num_equality constraints
- Bounds: μ_i ∈ [0, max_multiplier], λ_j ∈ [−max_multiplier, max_multiplier]
- Returns `−L(x*, μ, λ)` (negated for minimising solvers)
- Updated each step via `set_fixed_solution(solution)`

### 7.4 Step procedure

```
1. Get best_solution x* from objective_solver
2. Get best_multipliers [μ*, λ*] from multiplier_solver
3. Update min_problem fixed multipliers ← [μ*, λ*]
4. Update max_problem fixed solution ← x*
5. objective_solver.step()
6. multiplier_solver.step()
```

### 7.5 `get_result()`

Returns the objective solver's best `x*` evaluated against the **original constrained problem** (not the proxy), so returned fitness and constraint values are true values.

### 7.6 Population delegation

`get_population()` and `get_population_evaluations()` delegate to the **objective solver**. Population evaluations reflect Lagrangian fitness (not original), so feasibility tracking uses the constraint fields passed through from `_LagrangianMinProblem`.

### 7.7 Minimal usage example (CCPSO on G01)

```python
from cilpy.problem.constrained import G01
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
from cilpy.solver.pso import PSO
from cilpy.runner import ExperimentRunner

runner = ExperimentRunner(
    problems=[G01()],
    solver_configurations=[{
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": "CCPSO",
            "penalty_rho": 0.5,
            "max_multiplier": 100.0,
            "objective_solver_class": PSO,
            "multiplier_solver_class": PSO,
            "objective_solver_params": {
                "name": "obj_pso",
                "swarm_size": 200,
                "w": 0.72,
                "c1": 1.49,
                "c2": 1.49,
            },
            "multiplier_solver_params": {
                "name": "mul_pso",
                "swarm_size": 200,
                "w": 0.4,
                "c1": 1.2,
                "c2": 1.2,
            },
        },
    }],
    num_runs=30,
    max_iterations=1000,
)
runner.run_experiments()
```

### 7.8 Dynamic support

Dynamic handling via `begin_iteration()` and the `TODO` block in `step()` is **not yet implemented**. The proxy problems delegate `is_dynamic()` to the original problem, but the CCLS step does not yet re-initialise or respond to landscape changes. This is required before Phase 2 (DOSC) can begin.

---

## 8. ExperimentRunner

*`cilpy/runner.py`*

| `__init__` arg | Type | Description |
|----------------|------|-------------|
| `problems` | `Sequence[Problem]` | Problem instances to benchmark |
| `solver_configurations` | `List[dict]` | List of `{"class": SolverClass, "params": {...}}` dicts |
| `num_runs` | `int` | Independent repetitions per problem–solver pair |
| `max_iterations` | `int` | `solver.step()` calls per run |

Optional top-level key in solver config dict:
```python
{
    "class": ...,
    "params": {...},
    "constraint_handler": {"class": HandlerClass, "params": {...}}  # optional
}
```

**Output per problem–solver pair**:
- `out/{problem}_{solver}.out.csv` — per-iteration log
- `out/{problem}_{solver}.summary.out.csv` — per-run P_RED summary

**CSV columns (iteration log)**:

| Column | Description |
|--------|-------------|
| `run_id` | Run number (1 to `num_runs`) |
| `iteration` | Iteration number (1 to `max_iterations`) |
| `accuracy` | Best fitness so far |
| `feasibility` | % of population that is feasible |
| `diversity` | `(1/N)·√(Σ||x_i − centroid||²)` |
| `relative_error` | `(f_max − accuracy) / (f_max − f_min)` — requires `get_fitness_bounds()` |

**Summary CSV columns**:

| Column | Description |
|--------|-------------|
| `problem_name` | — |
| `solver_name` | — |
| `run_id` | — |
| `relative_error_distance` | P_RED = `√(Σ(1−b_i)²) / √N` where b_i = relative_error at iter i |

`feasibility` is measured against **original constraint values** because `_LagrangianMinProblem` passes them through in `Evaluation.constraints_inequality`.

---

## 9. Benchmarking Status

### Production config (final)
`num_runs=30`, `max_iterations=1000`, `swarm_size=200` both sub-solvers.  
`penalty_rho=0.5`, `penalty_rho_equality=0.5`, `max_multiplier=10000.0`  
Objective solver: `w=0.72, c1=1.49, c2=1.49` | Multiplier solver: `w=0.4, c1=1.2, c2=1.2`

### Phase 1 — Final 30-run results

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

G03 excluded — equality encoding interferes with multiplier dynamics.

### Two failure modes identified

**1. Premature convergence** (G01, CMPB_SOSC): feasible but suboptimal; diversity collapses; same local basin every run.  
**2. Multiplier cap saturation** (G05, G06): 0% feasibility; constraint magnitudes exceed max_multiplier=10000; G06 P_RED=179 confirms complete failure.  
**Bimodal** (G02, G04): high std on feasibility; runs either find feasible region or fail entirely depending on initialisation.

### analysis.py outputs
- `analysis_fitness_summary.csv`
- `analysis_feasibility_summary.csv`
- `analysis_p_red_summary.csv`
- `analysis_convergence_detail.csv`

---

## 10. Open Issues & Next Steps

### Immediate (report submission 15 May)
- [ ] Remove TODO comment line 290 in report.tex
- [ ] Fix table environment: `\begin{table}[h]` → `\begin{table*}[t]`, `\end{table}` → `\end{table*}`
- [ ] Fix `c1`, `c2` → `$c_1$`, `$c_2$` in Section III
- [ ] Fix "observed on G01 and the CMPB problems" → "observed on G01 and CMPB\_SOSC"
- [ ] Abstract tweak to reflect Phase 1 scope

### Phase 1 complete — findings
- max_multiplier is problem-dependent; G05/G06 need values >> 10000
- Premature convergence is dominant limitation; motivates inertia decay / restart mechanisms
- Augmented Lagrangian extension (penalty_rho) justified and working
- CMPB_SODC is strongest result: 100% feasibility, P_RED 0.027

### Phase 2 prep (DOSC) — next after Demo 1
- [ ] Implement dynamic change detection / response in `CoevolutionaryLagrangianSolver.step()`
- [ ] Hook into `problem.begin_iteration()` to trigger landscape updates
- [ ] Test CCLS on CMPB_DOSC (`f_params=mpb_configs["A2R"]`, `g_params=mpb_configs["STA"]`)
- [ ] Consider QPSO as the objective solver for better dynamic tracking
- [ ] Implement inertia decay schedule to address premature convergence finding

### Phase 3 prep (SODC)
- [ ] Test CCLS on CMPB_SODC (`f_params=mpb_configs["STA"]`, `g_params=mpb_configs["A2R"]`)
- [ ] Verify multiplier population responds to constraint landscape changes

### Deferred
- [ ] Multi-guide PSO with archive (for multi-objective; existing external code to integrate)
- [ ] Adaptive max_multiplier scheduling (addresses G05/G06 failure mode)

---

## 11. Out-of-Scope Items (Removed from Research Contribution)

- **Feasibility rules** (`DebFeasibilityHandler`): Removed from scope. The constraint handling used in CCLS relies on the Lagrangian reformulation itself, not an external comparator applied to the objective solver.
- **Repair mechanisms**: Not planned.
- **Compare module**: Statistical analysis tooling in `cilpy/compare/` is a placeholder and not part of this project.
