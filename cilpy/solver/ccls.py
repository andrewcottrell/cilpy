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

from typing import List, Tuple, Type

from cilpy.problem import Problem, Evaluation, SolutionType
from cilpy.solver import Solver


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

    def __init__(self, original_problem: Problem, penalty_scale: float = 1.0):
        """Initializes the proxy problem for the objective space.

        Args:
            original_problem: The original constrained problem instance that
                will be wrapped.
        """
        super().__init__(
            original_problem.dimension, original_problem.bounds, "LagrangianMinProblem"
        )
        self.original_problem = original_problem
        self.penalty_scale = penalty_scale
        self.fixed_multipliers_inequality = [0.0] * len(
            self.original_problem.evaluate(
                self.original_problem.bounds[0]
            ).constraints_inequality
            or []
        )
        self.fixed_multipliers_equality = [0.0] * len(
            self.original_problem.evaluate(
                self.original_problem.bounds[0]
            ).constraints_equality
            or []
        )

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

        # Calculate L(x, mu*, lambda*), penalizing only violations.
        lagrangian_value = fx
        lagrangian_value += self.penalty_scale * sum(
            mu * (max(0.0, g) ** 2)
            for mu, g in zip(self.fixed_multipliers_inequality, gx)
        )
        lagrangian_value += self.penalty_scale * sum(
            la * h for la, h in zip(self.fixed_multipliers_equality, hx)
        )

        # This problem is now unconstrained from the solver's perspective
        return Evaluation(fitness=lagrangian_value)

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
        max_multiplier: float = 10000.0,
        penalty_scale: float = 200.0,
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
        # Multipliers for equality constraints (lambda) are unrestricted,
        # but both are clamped to finite bounds for stable evolution.
        lower_bounds = [0.0] * num_inequality + [-max_multiplier] * num_equality
        upper_bounds = [max_multiplier] * (num_inequality + num_equality)

        super().__init__(
            dimension, (lower_bounds, upper_bounds), "LagrangianMaxProblem"
        )
        self.original_problem = original_problem
        self.fixed_solution_eval = original_problem.evaluate(fixed_solution)
        self.num_inequality = num_inequality
        self.penalty_scale = penalty_scale

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

        # Calculate L(x*, mu, lambda), penalizing only violations.
        lagrangian_value = fx
        lagrangian_value += self.penalty_scale * sum(
            s * (max(0.0, g) ** 2) for s, g in zip(inequality_multipliers, gx)
        )
        lagrangian_value += self.penalty_scale * sum(
            l * h for l, h in zip(equality_multipliers, hx)
        )

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
        max_multiplier: float = 10000.0,
        penalty_scale: float = 200.0,
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
        self.min_problem = _LagrangianMinProblem(
            problem, penalty_scale=penalty_scale
        )
        self.max_problem = _LagrangianMaxProblem(
            problem,
            initial_solution,
            max_multiplier=max_multiplier,
            penalty_scale=penalty_scale,
        )

        # 2. Instantiate the internal solvers
        self.objective_solver = objective_solver_class(
            problem=self.min_problem, **objective_solver_params
        )
        self.multiplier_solver = multiplier_solver_class(
            problem=self.max_problem, **multiplier_solver_params
        )

    def step(self) -> None:
        """Performs one co-evolutionary step.

        This process involves:
        1. Getting the best individuals from each population.
        2. Updating the fitness landscape of each sub-problem using the best
           individual from the other population.
        3. Advancing each subordinate solver by one iteration.
        """

        # 1. Get the best individuals from each population
        best_solution, _ = self.objective_solver.get_result()[0]
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
        """Returns the best solution found in the objective space.

        The returned solution is evaluated against the *original* constrained
        problem, providing the user with the true fitness and constraint
        violation information.

        Returns:
            List[Tuple[SolutionType, Evaluation[float]]]: A list containing a
                single tuple of (best_solution, evaluation_on_original_problem).
        """
        best_solution, _ = self.objective_solver.get_result()[0]
        final_evaluation = self.problem.evaluate(best_solution)
        return [(best_solution, final_evaluation)]

    def get_population(self) -> List[List[float]]:
        """Delegates getting population to the objective solver."""
        return self.objective_solver.get_population()

    def get_population_evaluations(self) -> List[Evaluation[float]]:
        """Delegates getting population evaluations to the objective solver."""
        return self.objective_solver.get_population_evaluations()
