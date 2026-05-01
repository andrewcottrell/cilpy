from cilpy.problem.constrained import G06
from cilpy.solver.pso import PSO, QPSO
from cilpy.solver.ccls import _LagrangianMinProblem, _LagrangianMaxProblem

problem = G06()
initial_solution = [problem.bounds[0][i] for i in range(problem.dimension)]

min_prob = _LagrangianMinProblem(problem)
max_prob = _LagrangianMaxProblem(problem, initial_solution)
#max_prob.bounds = ([0.0, 0.0], [0.01, 0.01])

obj_pso = QPSO(
    problem=min_prob,
    name="obj",
    swarm_size=30,
    w=0.7298,
    c1=1.49618,
    c2=1.49618,
    split_ratio=0.5,
    r_cloud=5.0,
)
mul_pso = PSO(problem=max_prob, name="mul", swarm_size=30, w=0.7298, c1=1.49618, c2=1.49618)

print("Initial obj gbest fitness:", obj_pso.gbest_evaluation.fitness)
print("Initial multipliers:", mul_pso.gbest_position)

for i in range(20):
    best_sol, _ = obj_pso.get_result()[0]
    best_mul, _ = mul_pso.get_result()[0]

    min_prob.set_fixed_multipliers(best_mul[:max_prob.num_inequality], [])
    max_prob.set_fixed_solution(best_sol)

    obj_pso.step()
    mul_pso.step()

    best_sol, best_eval = obj_pso.get_result()[0]
    best_mul, _ = mul_pso.get_result()[0]
    orig_eval = problem.evaluate(best_sol)

    pop = obj_pso.get_population()
    diversity = sum(abs(p[0] - obj_pso.gbest_position[0]) for p in pop) / len(pop)

    print(f"\nIter {i+1}:")
    print(f"  Multipliers:  {[f'{m:.4f}' for m in best_mul]}")
    print(f"  True f(x):    {orig_eval.fitness:.4f}")
    print(f"  Constraints:  {[f'{c:.4f}' for c in orig_eval.constraints_inequality]}")
    print(f"  Feasible:     {all(c <= 0 for c in orig_eval.constraints_inequality)}")
    print(f"  Diversity:    {diversity:.4f}")

print("\nObj PSO gbest after 20 iters:", obj_pso.gbest_evaluation.fitness)
print("Mul PSO gbest after 20 iters:", mul_pso.gbest_evaluation.fitness)