# cilpy/solver/__init__.py
"""The solver module: Defines the optimization algorithm interface.

This module provides the abstract "contract" for all optimization algorithms
(solvers) within the `cilpy` library.

The core component is the `Solver` abstract base class. Any algorithm that
inherits from this class and implements its methods can be used by the
`ExperimentRunner` to solve any `cilpy` problem.
"""

from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, Optional

from cilpy.problem import Problem, Evaluation, SolutionType, FitnessType
from cilpy.solver.chm import ConstraintHandler, DefaultComparator


class Solver(ABC, Generic[SolutionType, FitnessType]):
    """An abstract interface for a problem solver.

    This class is the blueprint for all optimization algorithms in `cilpy`. To
    create a new solver, inherit from this class and implement its abstract
    methods. The interface is generic to support different solution types
    (e.g., `List[float]`, `np.ndarray`) and fitness structures.

    Attributes:
        problem (cilpy.problem.Problem): The problem instance that the solver is
            optimizing.
        name (str): The name of the solver instance.
        comparator (cilpy.solver.chm.ConstraintHandler): The constraint-handling
            comparator used to comparesolutions.

    Example:
        A minimal implementation for a Random Search solver.

        .. code-block:: python

            import random
            from cilpy.problem import Problem, Evaluation
            from cilpy.solver import Solver

            class RandomSearch(Solver[list[float], float]):
                def __init__(self, problem: Problem, name: str):
                    super().__init__(problem, name)
                    self.best_solution = None
                    self.best_eval = Evaluation(fitness=float('inf'))

                def step(self) -> None:
                    # Generate one random solution
                    lower, upper = self.problem.bounds
                    solution = [random.uniform(l, u) for l, u in zip(lower, upper)]
                    evaluation = self.problem.evaluate(solution)

                    # Update best if necessary
                    if evaluation.fitness < self.best_eval.fitness:
                        self.best_solution = solution
                        self.best_eval = evaluation

                def get_result(self) -> list[tuple[list[float], Evaluation[float]]]:
                    return [(self.best_solution, self.best_eval)]
    """

    @abstractmethod
    def __init__(
        self,
        problem: Problem[SolutionType, FitnessType],
        name: str,
        constraint_handler: Optional[ConstraintHandler[FitnessType]] = None,
        **kwargs,
    ):
        """Initializes the solver.

        Subclasses must call `super().__init__(...)` and can use `**kwargs` to
        accept algorithm-specific hyperparameters.

        Args:
            problem: The optimization problem to solve, which must
                implement the `Problem` interface.
            name: The name of the solver instance.
            constraint_handler: An optional strategy object for handling
                constraints. If None, a default fitness-only comparator is used.
            **kwargs: A dictionary for algorithm-specific parameters. For
                example, a GA might accept `population_size=100` or
                `mutation_rate=0.1`.
        """
        self.problem = problem
        self.name = name
        self.comparator = constraint_handler or DefaultComparator()

    @abstractmethod
    def step(self) -> None:
        """Performs one iteration of the optimization algorithm.

        This method contains the core logic of the solver. The
        `ExperimentRunner` will call this method repeatedly in a loop. A single
        step could be one generation in a GA, one iteration in a PSO, or the
        evaluation of one new solution in a simpler algorithm.
        """
        pass

    @abstractmethod
    def get_result(self) -> List[Tuple[SolutionType, Evaluation[FitnessType]]]:
        """Returns the best solution(s) found so far.

        This method provides the current result of the optimization process. It
        is called by the `ExperimentRunner` after each step to log progress.

        Returns:
            A list of tuples, where each tuple contains `(solution,
            evaluation)`.
            - For single-objective solvers, this list should contain a single
              tuple with the best solution found.
            - For multi-objective solvers, this list should contain the set
              of non-dominated solutions (the Pareto front archive).

            Example return for a single-objective solver:
            `[([0.1, -0.5], Evaluation(fitness=0.26))]`
        """
        pass

    def respond_to_change(self) -> None:
        """Tells the solver that the fitness landscape has changed.

        Called by whatever changed the landscape, for example a
        co-evolutionary framework that has just updated the problem this
        solver is optimizing. Solvers that keep a memory of past evaluations
        (personal bests, an archive) should override this method and
        re-evaluate that memory, so that values computed under the old
        landscape do not guide the search.

        This method is optional. The default does nothing, which suits
        solvers without such a memory.
        """
        pass

    def get_population(self) -> List[SolutionType]:
        """
        Returns the entire current population or set of candidate solutions.

        This method is optional and should be implemented by population-based
        algorithms. It is required for certain performance metrics like
        diversity measurement.

        Raises:
            NotImplementedError: If the solver is not swarm based.

        Returns:
            All individuals in the solver's population.
        """
        raise NotImplementedError(
            f"""The solver '{self.name}' does not have a population. Implement
             get_population() to use metrics that require it."""
        )

    def get_population_evaluations(self) -> List[Evaluation[FitnessType]]:
        """
        Returns the evaluations of the entire current population or set of
        candidate solutions.

        This method is optional and should be implemented by population-based
        algorithms. It is required for certain performance metrics like
        percentage of feasible solutions.

        Raises:
            NotImplementedError: If the solver is not swarm based.

        Returns:
            Evaluations for all individuals in the solver's population.
        """
        raise NotImplementedError(
            f"""The solver '{self.name}' does not have a population. Implement
             get_population_evaluations() to use metrics that require it."""
        )
