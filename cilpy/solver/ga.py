# cilpy/solver/ga.py
import copy
import random
from functools import cmp_to_key
from typing import List, Optional, Tuple

from cilpy.problem import Problem, Evaluation
from cilpy.solver import Solver
from cilpy.solver.chm import ConstraintHandler


class GA(Solver[List[float], float]):
    """
    A canonical Genetic Algorithm (GA) for single-objective optimization.

    The algorithm uses:
    - Tournament selection to choose parents.
    - Single-point crossover for reproduction.
    - Gaussian mutation to introduce genetic diversity.
    - Elitism to preserve the best solution across generations.
    """

    def __init__(
        self,
        problem: Problem[List[float], float],
        name: str,
        population_size: int,
        crossover_rate: float,
        mutation_rate: float,
        tournament_size: int = 2,
        constraint_handler: Optional[ConstraintHandler[float]] = None,
        **kwargs,
    ):
        """
        Initializes the Genetic Algorithm solver.

        Args:
            problem: The optimization problem to solve.
            name: the name of the solver
            population_size: The number of individuals in the population.
            crossover_rate: The probability of crossover (pc) occurring between
                two parents.
            mutation_rate: The probability of mutation (pm) for each gene in an
                offspring.
            tournament_size: The number of individuals to select for each
                tournament. Defaults to 2.
            **kwargs: Additional keyword arguments (not used in this canonical
                GA).
        """
        super().__init__(problem, name, constraint_handler=constraint_handler)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        # Initialize population
        self.population = self._initialize_population()
        self.evaluations = [self.problem.evaluate(i) for i in self.population]

    def _initialize_population(self) -> List[List[float]]:
        """Creates the initial population with random solutions."""
        population = []
        lower_bounds, upper_bounds = self.problem.bounds
        for _ in range(self.population_size):
            individual = [
                random.uniform(lower_bounds[i], upper_bounds[i])
                for i in range(self.problem.dimension)
            ]
            population.append(individual)
        return population

    def _selection(self) -> List[List[float]]:
        """
        Performs tournament selection to choose parents using the provided
        comparator.
        """
        parents = []
        for _ in range(self.population_size):
            tournament_indices = random.sample(
                list(range(self.population_size)), self.tournament_size
            )

            # Find the winner of the tournament through pairwise comparison
            winner_idx = tournament_indices[0]
            for i in range(1, len(tournament_indices)):
                competitor_idx = tournament_indices[i]
                # If the competitor is better than the current winner, update the winner
                if self.comparator.is_better(
                    self.evaluations[competitor_idx], self.evaluations[winner_idx]
                ):
                    winner_idx = competitor_idx

            parents.append(self.population[winner_idx])
        return parents

    def _reproduction(self, parents: List[List[float]]) -> List[List[float]]:
        """Creates offspring through single-point crossover."""
        offspring = []
        for i in range(0, self.population_size, 2):
            p1 = parents[i]
            # Ensure there's a second parent for crossover
            p2 = parents[i + 1] if i + 1 < self.population_size else parents[0]

            # Crossover is only possible if the dimension is > 1.
            if self.problem.dimension > 1 and random.random() < self.crossover_rate:
                crossover_point = random.randint(1, self.problem.dimension - 1)
                c1 = p1[:crossover_point] + p2[crossover_point:]
                c2 = p2[:crossover_point] + p1[crossover_point:]
                offspring.extend([c1, c2])
            else:
                offspring.extend([copy.deepcopy(p1), copy.deepcopy(p2)])
        return offspring[: self.population_size]

    def _mutation(self, offspring: List[List[float]]) -> List[List[float]]:
        """Applies Gaussian mutation to offspring."""
        lower_bounds, upper_bounds = self.problem.bounds
        for individual in offspring:
            for i in range(self.problem.dimension):
                if random.random() < self.mutation_rate:
                    # Add noise from a Gaussian distribution with mean 0
                    mutation_value = random.gauss(
                        0, (upper_bounds[i] - lower_bounds[i]) * 0.1
                    )
                    individual[i] += mutation_value
                    # Clamp the value to within the problem bounds
                    individual[i] = max(
                        lower_bounds[i], min(individual[i], upper_bounds[i])
                    )
        return offspring

    def step(self) -> None:
        """Performs one generation of the Genetic Algorithm."""
        # 1. Selection
        parents = self._selection()

        # 2. Reproduction
        offspring = self._reproduction(parents)

        # 3. Mutation
        mutated_offspring = self._mutation(offspring)

        # 4. Evaluate new offspring
        offspring_evaluations = [
            self.problem.evaluate(ind) for ind in mutated_offspring
        ]

        # 5. Combine (create next generation) with elitism
        # Find the best individual from the current generation using the
        # comparator
        best_current_idx = 0
        for i in range(1, self.population_size):
            if self.comparator.is_better(
                self.evaluations[i], self.evaluations[best_current_idx]
            ):
                best_current_idx = i
        best_individual = self.population[best_current_idx]
        best_evaluation = self.evaluations[best_current_idx]

        # The new population is the mutated offspring
        self.population = mutated_offspring
        self.evaluations = offspring_evaluations

        # Find the worst individual in the new generation using the comparator
        worst_new_idx = 0
        for i in range(1, self.population_size):
            # The worst is the one that is not better than the current worst
            if self.comparator.is_better(
                self.evaluations[worst_new_idx], self.evaluations[i]
            ):
                worst_new_idx = i

        # Replace the worst new individual with the best from the previous generation
        self.population[worst_new_idx] = best_individual
        self.evaluations[worst_new_idx] = best_evaluation

    def get_result(self) -> List[Tuple[List[float], Evaluation[float]]]:
        """Returns the best solution found in the current population."""
        best_idx = 0
        for i in range(1, self.population_size):
            if self.comparator.is_better(
                self.evaluations[i], self.evaluations[best_idx]
            ):
                best_idx = i

        best_solution = self.population[best_idx]
        best_evaluation = self.evaluations[best_idx]
        return [(best_solution, best_evaluation)]

    def get_population(self) -> List[List[float]]:
        """
        Returns the entire current GA population.

        This overrides the default Solver method to provide statistics on all
        individuals in the current generation.

        Returns:
            A list containing every individual.
        """
        return self.population

    def get_population_evaluations(self) -> List[Evaluation[float]]:
        """
        Returns the evaluations of the entire current GA population.

        This overrides the default Solver method to provide statistics on all
        individuals in the current generation.

        Returns:
            A list containing the `Evaluation` object for every individual.
        """
        return self.evaluations


class RIGA(GA):
    """
    A Random Immigrants Genetic Algorithm (RIGA) for dynamic optimization.

    This algorithm extends the canonical GA by introducing "random immigrants"
    in each generation to maintain diversity. A fixed percentage of the
    population is replaced by newly generated random individuals, which helps
    the algorithm avoid premature convergence and adapt to changing fitness
    landscapes.
    """

    def __init__(
        self,
        problem: Problem[List[float], float],
        name: str,
        population_size: int,
        crossover_rate: float,
        mutation_rate: float,
        immigrant_rate: float,
        tournament_size: int = 2,
        **kwargs,
    ):
        """
        Initializes the Random Immigrants Genetic Algorithm solver.

        Args:
            problem: The dynamic optimization problem to solve.
            name: the name of the solver
            population_size: The number of individuals in the population.
            crossover_rate: The probability of crossover.
            mutation_rate: The probability of mutation.
            immigrant_rate: The proportion of the population to be replaced by
                random immigrants in each generation (p_im).
            tournament_size: The number of individuals for tournament selection.
                Defaults to 2.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            problem,
            name,
            population_size,
            crossover_rate,
            mutation_rate,
            tournament_size,
            **kwargs,
        )
        self.immigrant_rate = immigrant_rate

    def _generate_immigrants(self, num_immigrants: int) -> List[List[float]]:
        """Generates a specified number of new random individuals."""
        immigrants = []
        lower_bounds, upper_bounds = self.problem.bounds
        for _ in range(num_immigrants):
            individual = [
                random.uniform(lower_bounds[i], upper_bounds[i])
                for i in range(self.problem.dimension)
            ]
            immigrants.append(individual)
        return immigrants

    def step(self) -> None:
        """Performs one generation of the RIGA."""
        # --- Perform a standard GA step first ---
        super().step()

        # --- Introduce Immigrants ---
        num_immigrants = int(self.population_size * self.immigrant_rate)
        if num_immigrants == 0:
            return

        # 1. Generate immigrants
        immigrants = self._generate_immigrants(num_immigrants)
        immigrant_evals = [self.problem.evaluate(ind) for ind in immigrants]

        # 2. Combine with the current population by replacing the worst
        def compare_individuals(idx1, idx2):
            eval1 = self.evaluations[idx1]
            eval2 = self.evaluations[idx2]
            if self.comparator.is_better(eval1, eval2):
                return -1  # eval1 comes first
            elif self.comparator.is_better(eval2, eval1):
                return 1  # eval2 comes first
            return 0

        # Find the indices of the `num_immigrants` worst individuals
        sorted_indices = sorted(
            range(self.population_size), key=cmp_to_key(compare_individuals)
        )
        worst_indices = sorted_indices[-num_immigrants:]

        # Replace them with the new immigrants
        for i, idx in enumerate(worst_indices):
            self.population[idx] = immigrants[i]
            self.evaluations[idx] = immigrant_evals[i]


class HyperMGA(GA):
    """
    A Hyper-mutation Genetic Algorithm (HyperM GA) for dynamic optimization.

    This algorithm extends the canonical GA to adapt to changing environments.
    It detects a change in the problem landscape by monitoring the fitness of
    the best solution. If the fitness degrades, it triggers a "hyper-mutation"
    phase with a significantly higher mutation rate to re-introduce diversity.
    """

    def __init__(
        self,
        problem: Problem[List[float], float],
        name: str,
        population_size: int,
        crossover_rate: float,
        mutation_rate: float,
        hyper_mutation_rate: float,
        hyper_total: int,
        tournament_size: int = 2,
        **kwargs,
    ):
        """
        Initializes the Hyper-mutation Genetic Algorithm solver.

        Args:
            problem: The dynamic optimization problem to solve.
            name: the name of the solver
            population_size: The number of individuals in the population.
            crossover_rate: The base probability of crossover.
            mutation_rate: The standard mutation rate (pm).
            hyper_mutation_rate: The higher mutation rate (p_hyper) used when
                the environment changes.
            hyper_total: The threshold for the hyper-mutation counter to stop
                hyper-mutation.
            tournament_size: The number of individuals for tournament selection.
                Defaults to 2.
            **kwargs: Additional keyword arguments.
        """
        # Call parent init with the standard mutation rate
        super().__init__(
            problem,
            name,
            population_size,
            crossover_rate,
            mutation_rate,
            tournament_size,
            **kwargs,
        )

        self.hyper_mutation_rate = hyper_mutation_rate
        self.hyper_total = hyper_total

        # State tracking variables
        self.f_best: float = float("inf")  # Using 'inf' to represent 'undefined'
        self.hyper_count = 0

        # M_norm is the standard mutation method from the parent GA class
        # We create a new, separate method for hyper-mutation
        self.m_current = self._mutation  # M_current starts as M_norm

    def _hyper_mutation(self, offspring: List[List[float]]) -> List[List[float]]:
        """Applies Gaussian mutation using the hyper_mutation_rate."""
        lower_bounds, upper_bounds = self.problem.bounds
        for individual in offspring:
            for i in range(self.problem.dimension):
                if random.random() < self.hyper_mutation_rate:
                    # Add noise from a Gaussian distribution
                    mutation_value = random.gauss(
                        0, (upper_bounds[i] - lower_bounds[i]) * 0.1
                    )
                    individual[i] += mutation_value
                    # Clamp the value to within the problem bounds
                    individual[i] = max(
                        lower_bounds[i], min(individual[i], upper_bounds[i])
                    )
        return offspring

    def step(self) -> None:
        """Performs one generation of the HyperM GA."""
        # --- Change Detection and State Management ---
        # Evaluate population to get f_test
        self.evaluations = [self.problem.evaluate(ind) for ind in self.population]
        f_test = min(e.fitness for e in self.evaluations)

        # if f_best = undefined then f_best = f_test
        if self.f_best == float("inf"):
            self.f_best = f_test

        # if f_test is less fit than f_best then M_current = M_hyper
        if f_test > self.f_best:
            self.m_current = (
                self._hyper_mutation
            )  # Environment changed, switch to hyper-mutation

        # if hyper_count > hyper_total then M_current = M_norm; hyper_count = 0
        if self.hyper_count > self.hyper_total:
            self.m_current = self._mutation  # Stop hyper-mutation
            self.hyper_count = 0

        # --- Standard GA Operators ---
        # 1. Selection
        parents = self._selection()

        # 2. Reproduction (Crossover)
        offspring = self._reproduction(parents)

        # 3. Mutation (using the currently selected operator)
        mutated_offspring = self.m_current(offspring)

        # 4. Evaluate new offspring
        offspring_evaluations = [
            self.problem.evaluate(ind) for ind in mutated_offspring
        ]

        # 5. Combine (create next generation) with elitism
        best_current_idx = 0
        for i in range(1, self.population_size):
            if self.comparator.is_better(
                self.evaluations[i], self.evaluations[best_current_idx]
            ):
                best_current_idx = i
        best_individual = self.population[best_current_idx]
        best_evaluation = self.evaluations[best_current_idx]

        self.population = mutated_offspring
        self.evaluations = offspring_evaluations

        worst_new_idx = 0
        for i in range(1, self.population_size):
            if self.comparator.is_better(
                self.evaluations[worst_new_idx], self.evaluations[i]
            ):
                worst_new_idx = i

        self.population[worst_new_idx] = best_individual
        self.evaluations[worst_new_idx] = best_evaluation

        # Update f_best for the next iteration
        self.f_best = min(e.fitness for e in self.evaluations)

        # if hyper_count < hyper_total & M_current = M_hyper then hyper_count++
        if (
            self.m_current == self._hyper_mutation
            and self.hyper_count <= self.hyper_total
        ):
            self.hyper_count += 1
