from cilpy.solver.chm import ConstraintHandler
from cilpy.problem import Evaluation

class DebFeasibilityHandler(ConstraintHandler[float]):
    def is_better(self, eval_a: Evaluation[float], eval_b: Evaluation[float]) -> bool:
        feasible_a = self._is_feasible(eval_a)
        feasible_b = self._is_feasible(eval_b)

        if feasible_a and feasible_b:
            return eval_a.fitness < eval_b.fitness  # both feasible: compare fitness
        if feasible_a:
            return True                              # a feasible, b not: a wins
        if feasible_b:
            return False                             # b feasible, a not: b wins
        # both infeasible: lower total violation wins
        return self._total_violation(eval_a) < self._total_violation(eval_b)

    def _is_feasible(self, ev: Evaluation[float]) -> bool:
        ineq = ev.constraints_inequality or []
        eq   = ev.constraints_equality   or []
        return all(g <= 0 for g in ineq) and all(abs(h) <= 1e-6 for h in eq)

    def _total_violation(self, ev: Evaluation[float]) -> float:
        ineq = sum(max(0.0, g) for g in (ev.constraints_inequality or []))
        eq   = sum(abs(h)      for h in (ev.constraints_equality   or []))
        return ineq + eq