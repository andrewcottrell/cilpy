# cilpy/solver/ccls.py
"""A co-evolutionary framework for constrained optimization problems.

This module provides the `CoevolutionaryLagrangianSolver`, a meta-solver that
tackles constrained optimization problems by reformulating them using Lagrangian
relaxation. This approach avoids traditional penalty functions by transforming
the problem into an unconstrained min-max optimization task, which is then
solved by two cooperating populations of solvers.

How the Cooperative Co-evolutionary Framework Works:
-----------------------------------------------------
A constrained optimization problem can be stated as:
  Minimize: f(x)
  Subject to: g_i(x) <= 0  (inequality constraints)
              h_j(x) == 0  (equality constraints)

The Lagrangian function combines the objective and constraints into a single
equation:
  L(x, mu, lambda) = f(x) + Sum(mu_i * g_i(x)) + Sum(lambda_j * h_j(x))

Here, mu and lambda are the Lagrangian multipliers. The solution to the
original problem can be found by solving the min-max problem:
  min_x max_{mu,lambda} L(x, mu, lambda)

This framework implements this min-max search using two populations:
1.  An 'Objective Solver' Population: This population searches the original
    problem's solution space (for 'x'). Its goal is to MINIMIZE the Lagrangian
    function, L(x, mu*, lambda*), where the multipliers (mu*, lambda*) are the best ones
    found so far by the other population.
2.  A 'Multiplier Solver' Population: This population searches the space of
    Lagrangian multipliers (for 'mu' and 'lambda'). Its goal is to MAXIMIZE the
    Lagrangian function, L(x*, mu, lambda), where the solution (x*) is the best one
    found so far by the objective population.

At each step, the best individual from each population is used to define the
fitness landscape for the other. This cooperative process simultaneously drives
the solution 'x' towards feasibility and optimality while evolving the
multipliers to appropriately penalize constraint violations.

This implementation acts as a high-level "coordinator" or "meta-solver". It is
configured with two standard solver instances from the library (e.g., two GAs,
two PSOs for CCPSO) which manage the search process for their respective
populations.
"""

from typing import List, Optional, Tuple, Type

from cilpy.problem import Problem, Evaluation, SolutionType
from cilpy.solver import Solver


def _total_violation(evaluation: Evaluation, tolerance: float = 1e-6) -> float:
    """Total constraint violation of an evaluation.

    Sum of positive inequality violations and absolute equality violations
    beyond tolerance. Zero if and only if the solution is feasible.
    """
    violation = 0.0
    for g in evaluation.constraints_inequality or []:
        if g > 0:
            violation += g
    for h in evaluation.constraints_equality or []:
        if abs(h) > tolerance:
            violation += abs(h)
    return violation


class _LagrangianMinProblem(Problem):
    """An internal proxy problem for the objective-space solver ('min' swarm).

    This class wraps the original constrained problem, presenting it to the
    objective solver as an unconstrained problem. Its fitness function is the
    Lagrangian L(x, mu*, lambda*), where the multipliers (mu*, lambda*) are fixed for the
    current generation, having been provided by the multiplier swarm.

    Attributes:
        original_problem (Problem): A reference to the user-defined constrained
            problem.
        fixed_multipliers_inequality (List[float]): The best inequality
            multipliers (mu*) from the multiplier swarm, fixed for this evaluation.
        fixed_multipliers_equality (List[float]): The best equality multipliers
            (lambda*) from the multiplier swarm, fixed for this evaluation.
    """

    def __init__(self, original_problem: Problem, penalty_rho: float = 0.1, penalty_rho_equality: float = None):
        """Initializes the proxy problem for the objective space.

        Args:
            original_problem: The original constrained problem instance that
                will be wrapped.
            penalty_rho: Penalty coefficient for inequality constraint violations
                (default 0.1). Helps accelerate feasibility discovery.
            penalty_rho_equality: Penalty coefficient for equality constraints.
                Defaults to penalty_rho if not specified.
        """
        super().__init__(
            original_problem.dimension, original_problem.bounds, "LagrangianMinProblem"
        )
        self.original_problem = original_problem

        # Evaluate at lower bounds to initialize multipliers based on constraint violations
        initial_point = [original_problem.bounds[0][i] for i in range(original_problem.dimension)]
        initial_eval = original_problem.evaluate(initial_point)

        # Initialize inequality multipliers based on initial violations
        ineq_constraints = initial_eval.constraints_inequality or []
        self.fixed_multipliers_inequality = [
            max(0.0, min(g, 1.0)) for g in ineq_constraints
        ]

        # Initialize equality multipliers based on initial violations
        eq_constraints = initial_eval.constraints_equality or []
        self.fixed_multipliers_equality = [
            min(abs(h), 1.0) for h in eq_constraints
        ]

        self.penalty_rho = penalty_rho
        self.penalty_rho_equality = penalty_rho_equality if penalty_rho_equality is not None else penalty_rho
        

    def set_fixed_multipliers(self, inequality_multipliers, equality_multipliers):
        """Updates the fixed Lagrangian multipliers for the next generation.

        This method is called by the main `CoevolutionaryLagrangianSolver` before
        the objective solver performs its next step.

        Args:
            inequality_multipliers (List[float]): The new set of fixed mu* values.
            equality_multipliers (List[float]): The new set of fixed lambda* values.
        """
        self.fixed_multipliers_inequality = inequality_multipliers
        self.fixed_multipliers_equality = equality_multipliers

    def evaluate(self, solution: list[float]) -> Evaluation:
        """Calculates the Lagrangian value L(x, mu*, lambda*).

        This evaluation treats the problem as unconstrained, returning only a
        single fitness value representing the Lagrangian.

        Args:
            solution (SolutionType): The candidate solution 'x' to evaluate.

        Returns:
            Evaluation[float]: An Evaluation object where `fitness` is the
                Lagrangian value. The constraint fields are empty.
        """
        # Evaluate the original problem to get f(x), g(x), and h(x)
        original_eval = self.original_problem.evaluate(solution)
        fx = original_eval.fitness
        gx = original_eval.constraints_inequality or []
        hx = original_eval.constraints_equality or []

        if isinstance(fx, (list, tuple)):
            # Multi-objective: a single objective-independent penalty P(x)
            # is added to EVERY objective, L_m(x) = f_m(x) + P(x).
            #
            # The multiplier term uses the violation-clamped ("plus
            # function") form mu*^T max(0, g) rather than the signed
            # Lagrangian mu*^T g. The signed form rewards deeply feasible
            # solutions with large NEGATIVE penalties, which -- added to all
            # objectives -- lets the single most-interior solution
            # Pareto-dominate the entire archive and collapses the front to
            # one point. Clamping makes P(x) = 0 for every feasible
            # solution, so feasible solutions compete purely on their true
            # objectives (Pareto structure preserved), while infeasible
            # solutions are inflated on all objectives and get dominated out
            # as the multipliers grow.
            penalty = 0.0
            penalty += sum(
                mu * max(0.0, g)
                for mu, g in zip(self.fixed_multipliers_inequality, gx)
            )
            penalty += sum(
                la * abs(h)
                for la, h in zip(self.fixed_multipliers_equality, hx)
            )
            if self.penalty_rho > 0.0 and gx:
                penalty += self.penalty_rho * sum(max(0.0, g) for g in gx)
            if self.penalty_rho_equality > 0.0 and hx:
                penalty += self.penalty_rho_equality * sum(abs(h) for h in hx)
            lagrangian_value = [fm + penalty for fm in fx]
        else:
            # Single-objective: the original signed Lagrangian, unchanged
            # from the validated Phase 1 implementation.
            lagrangian_value = fx
            lagrangian_value += sum(
                mu * g for mu, g in zip(self.fixed_multipliers_inequality, gx)
            )
            lagrangian_value += sum(
                la * h for la, h in zip(self.fixed_multipliers_equality, hx)
            )
            if self.penalty_rho > 0.0 and gx:
                lagrangian_value += self.penalty_rho * sum(
                    max(0.0, g) for g in gx
                )
            if self.penalty_rho_equality > 0.0 and hx:
                lagrangian_value += self.penalty_rho_equality * sum(
                    abs(h) for h in hx
                )

        # This problem is now unconstrained from the solver's perspective,
        # but the raw constraint values are passed through so that archive
        # admission policies and feasibility reporting remain possible.
        return Evaluation(
            fitness=lagrangian_value,
            constraints_inequality=original_eval.constraints_inequality,
            constraints_equality=original_eval.constraints_equality,
        )

    def get_fitness_bounds(self) -> Tuple[float, float]:
        """Delegates to the original problem to satisfy the interface."""
        return self.original_problem.get_fitness_bounds()

    def is_dynamic(self) -> tuple[bool, bool]:
        """Reports the dynamism of the proxy landscape.

        For multi-objective problems, the objective component is reported as
        dynamic even when the original problem is static: the Lagrangian
        landscape L_m(x) = f_m(x) + P(x) genuinely changes every iteration as
        the multiplier swarm updates mu*. This signals archive-based solvers
        (e.g. MGPSO) to re-evaluate their archive each step, preventing stale
        penalised fitness values computed under old multipliers from
        distorting Pareto dominance in the archive.

        For single-objective problems, the original behaviour (delegation) is
        preserved, matching the validated Phase 1 setup.

        Returns:
            Tuple[bool, bool]: (objectives_dynamic, constraints_dynamic).
        """
        orig_obj_dyn, orig_con_dyn = self.original_problem.is_dynamic()
        if self.original_problem.is_multi_objective():
            return (True, orig_con_dyn)
        return (orig_obj_dyn, orig_con_dyn)

    def is_multi_objective(self) -> bool:
        """Delegates the check for multi-objective to the original problem.

        Returns:
            bool: A boolean indicating if the original problem is
            multi-objective.
        """
        return self.original_problem.is_multi_objective()


class _LagrangianMaxProblem(Problem):
    """An internal proxy problem for the multiplier-space solver ('max' swarm).

    This class wraps the original problem to create the search space for the
    Lagrangian multipliers (mu and lambda). Its fitness function is L(x*, mu, lambda), where
    the solution 'x*' is fixed for the current generation. Since library solvers
    typically minimize, this class returns -L(x*, mu, lambda) to achieve maximization.

    Attributes:
        original_problem (Problem): A reference to the user-defined constrained
            problem.
        fixed_solution_eval (Evaluation): The evaluation result of the best
            solution (x*) from the objective swarm.
        num_inequality (int): The number of inequality constraints.
    """

    def __init__(
        self,
        original_problem: Problem[SolutionType, float],
        fixed_solution: SolutionType,
        max_multiplier: Optional[float] = None,
    ):
        """Initializes the proxy problem for the multiplier space.

        Args:
            original_problem (Problem): The original constrained problem.
            fixed_solution (SolutionType): An initial solution 'x' used to
                determine the number of constraints and thus the dimension
                of the multiplier search space.
        """
        num_inequality = len(
            original_problem.evaluate(fixed_solution).constraints_inequality or []
        )
        num_equality = len(
            original_problem.evaluate(fixed_solution).constraints_equality or []
        )
        dimension = num_inequality + num_equality

        # Multipliers for inequality constraints (mu) must be >= 0
        # Multipliers for equality constraints (lambda) are unrestricted
        if max_multiplier is None:
            lower_bounds = [0.0] * num_inequality + [-float("inf")] * num_equality
            upper_bounds = [float("inf")] * (num_inequality + num_equality)
        else:
            lower_bounds = [0.0] * num_inequality + [
                -max_multiplier
            ] * num_equality
            upper_bounds = [max_multiplier] * (num_inequality + num_equality)

        super().__init__(
            dimension, (lower_bounds, upper_bounds), "LagrangianMaxProblem"
        )
        self.original_problem = original_problem
        self.fixed_solution_eval = original_problem.evaluate(fixed_solution)
        self.num_inequality = num_inequality

    def set_fixed_solution(self, solution):
        """Updates the fixed solution 'x*' for the next generation.

        This method is called by the main `CoevolutionaryLagrangianSolver` before
        the multiplier solver performs its next step.

        Args:
            solution (SolutionType): The new fixed solution 'x*'.
        """
        self.fixed_solution_eval = self.original_problem.evaluate(solution)

    def evaluate(self, solution: list[float]) -> Evaluation:
        """Calculates L(x*, mu, lambda) for maximization.

        The `solution` argument here is a vector of concatenated Lagrangian
        multipliers [mu_1, ..., mu_n, lambda_1, ..., lambda_m].

        Args:
            solution (List[float]): A candidate vector of multipliers.

        Returns:
            Evaluation[float]: An Evaluation object where `fitness` is the
                negated Lagrangian value.
        """
        # Unpack multipliers
        inequality_multipliers = solution[: self.num_inequality]
        equality_multipliers = solution[self.num_inequality :]

        fx = self.fixed_solution_eval.fitness
        gx = self.fixed_solution_eval.constraints_inequality or []
        hx = self.fixed_solution_eval.constraints_equality or []

        # f(x*) is constant with respect to the multipliers, so it never
        # affects the argmax of L(x*, mu, lambda) -- it only shifts the
        # reported value. For multi-objective problems fx is a vector and
        # cannot be added to a scalar, so the constant term is dropped.
        if isinstance(fx, (list, tuple)):
            fx = 0.0

        # Calculate L(x*, mu, lambda)
        lagrangian_value = fx
        lagrangian_value += sum(s * g for s, g in zip(inequality_multipliers, gx))
        lagrangian_value += sum(l * h for l, h in zip(equality_multipliers, hx))

        # Return the negative value because we want to MAXIMIZE L
        return Evaluation(fitness=-lagrangian_value)

    def get_fitness_bounds(self) -> Tuple[float, float]:
        """Delegates to the original problem to satisfy the interface."""
        return self.original_problem.get_fitness_bounds()

    def is_dynamic(self) -> tuple[bool, bool]:
        """Delegates the check for dynamic properties to the original problem.

        Returns:
            Tuple[bool, bool]: A tuple indicating if the original problem's
                objectives or constraints are dynamic.
        """
        return self.original_problem.is_dynamic()

    def is_multi_objective(self) -> bool:
        """Delegates the check for multi-objective to the original problem.

        Returns:
            bool: A boolean indicating if the original problem is
            multi-objective.
        """
        return self.original_problem.is_multi_objective()


class CoevolutionaryLagrangianSolver(Solver):
    """A meta-solver for constrained optimization using a co-evolutionary
    Lagrangian framework.

    This solver transforms a constrained problem into an unconstrained min-max
    problem, which it solves using two cooperating ("co-evolving") populations
    of standard solvers. It is generic and can be configured with any two
    solver classes from the `cilpy` library.

    Attributes:
        objective_solver (Solver): The subordinate solver instance that searches
            the solution space of the original problem.
        multiplier_solver (Solver): The subordinate solver instance that
            searches the space of the Lagrangian multipliers.
        min_problem (_LagrangianMinProblem): The internal proxy problem for the
            objective solver.
        max_problem (_LagrangianMaxProblem): The internal proxy problem for the
            multiplier solver.
    """

    def __init__(
        self,
        name: str,
        problem: Problem,
        objective_solver_class: Type[Solver],
        multiplier_solver_class: Type[Solver],
        objective_solver_params: dict,
        multiplier_solver_params: dict,
        **kwargs,
    ):
        """Initializes the CoevolutionaryLagrangianSolver.

        Args:
            problem: The constrained optimization problem to solve.
            name: The name of this solver instance.
            objective_solver_class: The class of the solver to use for the
                objective space (e.g., `GA`, `PSO`).
            multiplier_solver_class: The class of the solver to use for the
                multiplier space.
            objective_solver_params: A dictionary of parameters to initialize
                the objective solver.
            multiplier_solver_params: A dictionary of parameters to initialize
                the multiplier solver.
        """

        super().__init__(problem, name=name)

        # 1. Create the proxy problems
        # We need an initial solution to dimension the multiplier problem
        initial_solution = [problem.bounds[0][i] for i in range(problem.dimension)]
        penalty_rho = float(kwargs.get("penalty_rho", 0.0))
        penalty_rho_equality = float(kwargs.get("penalty_rho_equality", penalty_rho))
        max_multiplier = kwargs.get("max_multiplier")
        if max_multiplier is not None:
            max_multiplier = float(max_multiplier)
            if max_multiplier <= 0.0:
                max_multiplier = None

        constraint_handler = kwargs.get("constraint_handler", None)

        # Archive management strategy for multi-objective objective solvers:
        #   "filter" (default) -- the archive may contain infeasible
        #       solutions during the run (they are gradually dominated out
        #       as the multipliers grow); get_result() filters the final
        #       front so that only feasible solutions are reported.
        #   "strict" -- infeasible solutions are never admitted to the
        #       archive in the first place; no filtering is needed.
        # Ignored for single-objective problems.
        self.archive_strategy = str(kwargs.get("archive_strategy", "filter"))
        if self.archive_strategy not in ("filter", "strict"):
            raise ValueError(
                f"archive_strategy must be 'filter' or 'strict', "
                f"got '{self.archive_strategy}'."
            )

        self.min_problem = _LagrangianMinProblem(
            problem,
            penalty_rho=penalty_rho,
            penalty_rho_equality=penalty_rho_equality,
        )
        self.max_problem = _LagrangianMaxProblem(
            problem, initial_solution, max_multiplier=max_multiplier
        )

        obj_params = dict(objective_solver_params)
        obj_params.setdefault("name", f"{name}_objective")
        if problem.is_multi_objective() and self.archive_strategy == "strict":
            obj_params.setdefault("feasible_archive_only", True)
        self.objective_solver = objective_solver_class(
            problem=self.min_problem,
            constraint_handler=constraint_handler,
            **obj_params
        )
        mul_params = dict(multiplier_solver_params)
        mul_params.setdefault("name", f"{name}_multiplier")
        self.multiplier_solver = multiplier_solver_class(
            problem=self.max_problem, **mul_params
        )

    def _select_multiplier_anchor(self) -> list:
        """Selects the solution x* against which the multipliers evolve.

        Single-objective: the best solution of the objective solver, exactly
        as in the original framework -- if it violates constraints the
        multipliers grow; once it is feasible they relax toward the saddle
        point.

        Multi-objective: the "best solution" is an entire archive, so a
        representative must be chosen. The rule used is:

        * If the archive is non-empty, the *most-violating* archive member is
          selected. The multipliers then respond to the worst remaining
          violation on the current front, and pressure persists until the
          entire front is feasible. When every archive member is feasible,
          all violations are non-positive and the multipliers relax, matching
          the single-objective saddle-point dynamics.

        * If the archive is empty (possible under the "strict" admission
          strategy before any feasible solution has been found), the
          *least-violating population member* is selected instead. This is
          the current feasibility frontier: penalising its remaining
          violations applies pressure exactly where the swarm is closest to
          success, while avoiding the multiplier-cap saturation that anchoring
          on an arbitrary wildly-infeasible particle would cause.
        """
        results = self.objective_solver.get_result()

        if not self.problem.is_multi_objective():
            return results[0][0]

        if results:
            # Most-violating archive member (ties broken by first found).
            worst_idx = max(
                range(len(results)),
                key=lambda i: _total_violation(results[i][1]),
            )
            return results[worst_idx][0]

        # Empty archive: least-violating member of the population.
        population = self.objective_solver.get_population()
        evaluations = self.objective_solver.get_population_evaluations()
        best_idx = min(
            range(len(population)),
            key=lambda i: _total_violation(evaluations[i]),
        )
        return population[best_idx]

    def step(self) -> None:
        """Performs one co-evolutionary step.

        This process involves:
        1. Getting the best individuals from each population.
        2. Updating the fitness landscape of each sub-problem using the best
           individual from the other population.
        3. Advancing each subordinate solver by one iteration.
        """

        # 1. Get the best individuals from each population
        best_solution = self._select_multiplier_anchor()
        best_multipliers, _ = self.multiplier_solver.get_result()[0]

        num_inequality = self.max_problem.num_inequality
        inequality_multipliers = best_multipliers[:num_inequality]
        equality_multipliers = best_multipliers[num_inequality:]

        # 2. Update the fitness landscapes for the sub-solvers
        # The 'min' problem gets the best multipliers from the 'max' solver
        self.min_problem.set_fixed_multipliers(
            inequality_multipliers, equality_multipliers
        )
        # The 'max' problem gets the best solution from the 'min' solver
        self.max_problem.set_fixed_solution(best_solution)

        # 3. Perform one step of each sub-solver
        self.objective_solver.step()
        self.multiplier_solver.step()

        # TODO: Handle dynamic changes
        # is_obj_dyn, is_con_dyn = self.problem.is_dynamic()
        # if is_obj_dyn or is_con_dyn:
        #     pass

    def get_result(self) -> list[tuple[list[float], Evaluation]]:
        """Returns the best solution(s), evaluated on the ORIGINAL problem.

        Archive members are stored by the objective solver with *penalised*
        Lagrangian fitness, so every returned solution is re-evaluated
        against the original constrained problem, restoring the true
        objective values and constraint information.

        Single-objective: a single (solution, evaluation) tuple, as before.

        Multi-objective: the full archive. Under the "filter" strategy the
        front is filtered to feasible solutions only; if no archive member is
        feasible, the unfiltered archive is returned so that the state of the
        search remains diagnosable (its feasibility is visible in the
        evaluations). Under the "strict" strategy the archive only ever
        contains feasible solutions, so filtering is a no-op.

        Returns:
            List[Tuple[SolutionType, Evaluation]]: One tuple per solution.
        """
        if not self.problem.is_multi_objective():
            best_solution, _ = self.objective_solver.get_result()[0]
            final_evaluation = self.problem.evaluate(best_solution)
            return [(best_solution, final_evaluation)]

        re_evaluated = [
            (solution, self.problem.evaluate(solution))
            for solution, _ in self.objective_solver.get_result()
        ]
        feasible = [
            (s, e) for s, e in re_evaluated if _total_violation(e) == 0.0
        ]
        return feasible if feasible else re_evaluated

    def get_population(self) -> List[List[float]]:
        """Delegates getting population to the objective solver."""
        return self.objective_solver.get_population()

    def get_population_evaluations(self) -> List[Evaluation[float]]:
        """Delegates getting population evaluations to the objective solver."""
        return self.objective_solver.get_population_evaluations()