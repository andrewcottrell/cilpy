"""
Diagnostic script: PSO + AlphaConstraintHandler on G04.

Purpose: Isolate whether the constraint handler + solver work correctly
on a well-behaved problem (G04: 52% feasible, no equality constraints)
before introducing CCLS complexity.

Run with:
    python diagnose_g04.py

What to look for:
  - PASS: feasibility % climbs toward 100% and best fitness approaches -30665.5
  - FAIL: feasibility stays near 0%  -> constraint handler or G04 evaluate() bug
  - FAIL: feasibility rises but fitness is wrong -> sign error in objective
  - FAIL: diversity collapses in <20 iters -> PSO diversity problem (expected for CCLS)
"""

import numpy as np
from cilpy.problem import Problem, Evaluation
from cilpy.solver.pso import PSO
from cilpy.solver.ga import GA
from cilpy.solver.chm.alpha_constraint import AlphaConstraintHandler

# ── G04 implementation (CEC2006) ──────────────────────────────────────────────
# f* = -30665.5387, at x* = (78, 33, 29.995, 45, 36.776)
# Feasible ratio rho = 52.1% — ideal first diagnostic target.
# Bounds: 78<=x1<=102, 33<=x2<=45, 27<=xi<=45 (i=3,4,5)
# 6 inequality constraints, no equality constraints.

class G04(Problem):
    def __init__(self):
        lower = [78, 33, 27, 27, 27]
        upper = [102, 45, 45, 45, 45]
        super().__init__(dimension=5, bounds=(lower, upper), name="G04")

    def evaluate(self, solution):
        x = np.array(solution, dtype=float)
        x1, x2, x3, x4, x5 = x

        fitness = (5.3578547 * x3**2
                   + 0.8356891 * x1 * x5
                   + 37.293239 * x1
                   - 40792.141)

        g1 =  85.334407 + 0.0056858*x2*x5 + 0.0006262*x1*x4 - 0.0022053*x3*x5 - 92
        g2 = -85.334407 - 0.0056858*x2*x5 - 0.0006262*x1*x4 + 0.0022053*x3*x5
        g3 =  80.51249  + 0.0071317*x2*x5 + 0.0029955*x1*x2 + 0.0021813*x3**2 - 110
        g4 = -80.51249  - 0.0071317*x2*x5 - 0.0029955*x1*x2 - 0.0021813*x3**2 + 90
        g5 =  9.300961  + 0.0047026*x3*x5 + 0.0012547*x1*x3 + 0.0019085*x3*x4 - 25
        g6 = -9.300961  - 0.0047026*x3*x5 - 0.0012547*x1*x3 - 0.0019085*x3*x4 + 20

        return Evaluation(
            fitness=fitness,
            constraints_inequality=[g1, g2, g3, g4, g5, g6],
        )

    def is_dynamic(self):
        return (False, False)

    def is_multi_objective(self):
        return False

    def get_fitness_bounds(self):
        # rough bounds for relative error metric
        return (-30665.5387, 0.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_feasible(evaluation):
    if evaluation.constraints_inequality:
        return all(v <= 0 for v in evaluation.constraints_inequality)
    return True


def max_violation(evaluation):
    if not evaluation.constraints_inequality:
        return 0.0
    return max(max(v, 0) for v in evaluation.constraints_inequality)


def population_diversity(population):
    if not population or len(population) < 2:
        return 0.0
    arr = np.array(population)
    centroid = arr.mean(axis=0)
    return float(np.mean(np.linalg.norm(arr - centroid, axis=1)))


def check_random_feasibility(problem, n_samples=5000):
    """Estimate feasible ratio by random sampling — should be ~52% for G04."""
    lb, ub = np.array(problem.bounds[0]), np.array(problem.bounds[1])
    feasible = 0
    for _ in range(n_samples):
        x = lb + np.random.rand(problem.dimension) * (ub - lb)
        ev = problem.evaluate(x.tolist())
        if is_feasible(ev):
            feasible += 1
    return feasible / n_samples


def compute_b_inequality(problem, n_samples=2000):
    """
    Estimate a good b_inequality for AlphaConstraintHandler by sampling
    the max constraint violation across random points.
    Rule of thumb: b = 90th percentile of max violation across random solutions.
    """
    lb, ub = np.array(problem.bounds[0]), np.array(problem.bounds[1])
    violations = []
    for _ in range(n_samples):
        x = lb + np.random.rand(problem.dimension) * (ub - lb)
        ev = problem.evaluate(x.tolist())
        violations.append(max_violation(ev))
    return float(np.percentile(violations, 90))


# ── Diagnostic runner ─────────────────────────────────────────────────────────

def run_diagnostic(solver_class, solver_params, handler, problem,
                   max_iterations=500, log_every=50):
    solver = solver_class(
        problem=problem,
        constraint_handler=handler,
        **solver_params,
    )

    print(f"\n{'='*60}")
    print(f"Solver : {solver_params['name']}")
    print(f"Handler: {handler.__class__.__name__}  "
          f"alpha={handler.alpha}, b_ineq={handler.b_inequality:.3f}")
    print(f"{'='*60}")
    print(f"{'Iter':>6}  {'BestFit':>14}  {'Feasible?':>9}  "
          f"{'MaxViol':>10}  {'Diversity':>10}  {'Pop%Feasible':>12}")
    print("-" * 70)

    for i in range(1, max_iterations + 1):
        solver.step()

        if i % log_every == 0 or i == 1:
            # Best result
            results = solver.get_result()
            best_sol, best_eval = results[0]
            best_fit = best_eval.fitness
            feasible = is_feasible(best_eval)
            viol = max_violation(best_eval)

            # Population stats
            try:
                pop = solver.get_population()
                pop_evals = solver.get_population_evaluations()
                diversity = population_diversity(pop)
                pct_feasible = (
                    100 * sum(1 for e in pop_evals if is_feasible(e)) / len(pop_evals)
                )
            except Exception:
                diversity = float("nan")
                pct_feasible = float("nan")

            print(f"{i:>6}  {best_fit:>14.4f}  {'YES' if feasible else 'NO':>9}  "
                  f"{viol:>10.4f}  {diversity:>10.4f}  {pct_feasible:>11.1f}%")

    # Final verdict
    results = solver.get_result()
    best_sol, best_eval = results[0]
    known_optimum = -30665.5387
    error = abs(best_eval.fitness - known_optimum) if is_feasible(best_eval) else float("inf")

    print()
    print("── Final Result ──")
    print(f"  Best fitness : {best_eval.fitness:.6f}")
    print(f"  Known optimum: {known_optimum:.6f}")
    print(f"  Error        : {error:.6f}")
    print(f"  Feasible     : {is_feasible(best_eval)}")
    print(f"  Max violation: {max_violation(best_eval):.6f}")
    print(f"  Best solution: {[round(v,4) for v in best_sol]}")

    if is_feasible(best_eval) and error < 1000:
        print("  ✓ PASS — solver finds feasible region and reasonable fitness")
    elif is_feasible(best_eval):
        print("  ~ PARTIAL — feasible but far from optimum (tune params or iterations)")
    else:
        print("  ✗ FAIL — solver did not find a feasible solution")
        print("    → Check: constraint signs in G04.evaluate()")
        print("    → Check: b_inequality scale vs actual violations")
        print("    → Check: alpha threshold (try lower, e.g. 0.1)")

    return best_eval


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    problem = G04()

    # 1. Sanity check: random feasibility (should be ~52%)
    print("\n── Sanity check: random feasibility sampling ──")
    rho = check_random_feasibility(problem)
    print(f"  Estimated feasible ratio: {rho*100:.1f}%  (expected ~52%)")
    if rho < 0.30:
        print("  ✗ WARNING: feasibility much lower than expected — check G04.evaluate() signs")
    else:
        print("  ✓ Feasibility ratio looks correct")

    # 2. Estimate b_inequality from random samples
    print("\n── Estimating b_inequality from random violation distribution ──")
    b_auto = compute_b_inequality(problem)
    print(f"  90th-percentile max violation: {b_auto:.4f}")
    print(f"  Using b_inequality = {b_auto:.4f}  (vs common default of 5.0)")

    # 3. Show what happens with a BAD b_inequality (default 5.0)
    handler_bad = AlphaConstraintHandler(alpha=0.5, b_inequality=5.0, b_equality=1.0)
    run_diagnostic(
        solver_class=PSO,
        solver_params=dict(name="PSO_alpha_bad_b", swarm_size=40,
                           w=0.7298, c1=1.49618, c2=1.49618),
        handler=handler_bad,
        problem=problem,
        max_iterations=300,
        log_every=50,
    )

    # 4. Same solver with a properly scaled b_inequality
    handler_good = AlphaConstraintHandler(alpha=0.5, b_inequality=b_auto, b_equality=1.0)
    run_diagnostic(
        solver_class=PSO,
        solver_params=dict(name="PSO_alpha_good_b", swarm_size=40,
                           w=0.7298, c1=1.49618, c2=1.49618),
        handler=handler_good,
        problem=problem,
        max_iterations=300,
        log_every=50,
    )

    # 5. GA with good b_inequality
    run_diagnostic(
        solver_class=GA,
        solver_params=dict(name="GA_alpha_good_b", population_size=60,
                           mutation_rate=0.05, crossover_rate=0.8),
        handler=handler_good,
        problem=problem,
        max_iterations=300,
        log_every=50,
    )

    # 6. Lower alpha (less strict feasibility threshold) — helps when
    #    feasible solutions are hard to reach initially
    handler_low_alpha = AlphaConstraintHandler(alpha=0.1, b_inequality=b_auto, b_equality=1.0)
    run_diagnostic(
        solver_class=PSO,
        solver_params=dict(name="PSO_alpha=0.1", swarm_size=40,
                           w=0.7298, c1=1.49618, c2=1.49618),
        handler=handler_low_alpha,
        problem=problem,
        max_iterations=300,
        log_every=50,
    )

    print("\n── Interpretation guide ──────────────────────────────────────────")
    print("bad_b PASS, good_b PASS → b_inequality doesn't matter for G04 specifically")
    print("bad_b FAIL, good_b PASS → b_inequality scale is your root cause")
    print("Both FAIL               → bug in G04.evaluate() or handler logic")
    print("PSO FAIL, GA PASS       → PSO diversity collapse (expected in CCLS)")
    print("All PASS                → G01-G06 failures are CCLS-specific (multiplier bounds)")