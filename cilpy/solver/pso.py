# cilpy/solver/pso.py

import copy
import random
from typing import List, Optional, Tuple

from cilpy.problem import Problem, Evaluation
from cilpy.solver import Solver
from cilpy.solver.chm import ConstraintHandler


def _as_list(value) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def _evaluations_differ(fresh: Evaluation, old: Evaluation, tolerance: float) -> bool:
    """True if the fitness or any constraint value moved by more than
    *tolerance*."""
    for new_v, old_v in (
        (fresh.fitness, old.fitness),
        (fresh.constraints_inequality, old.constraints_inequality),
        (fresh.constraints_equality, old.constraints_equality),
    ):
        new_l, old_l = _as_list(new_v), _as_list(old_v)
        if len(new_l) != len(old_l):
            return True
        if any(abs(a - b) > tolerance for a, b in zip(new_l, old_l)):
            return True
    return False


class PSO(Solver[List[float], float]):
    """
    A canonical Particle Swarm Optimization (PSO) solver.

    The implementation uses a global best (gbest) topology (star topology), and
    uses inertial weight.
    """

    def __init__(
        self,
        problem: Problem[List[float], float],
        name: str,
        swarm_size: int,
        w: float,
        c1: float,
        c2: float,
        constraint_handler: Optional[ConstraintHandler[float]] = None,
        detect_changes: bool = False,
        **kwargs,
    ):
        """
        Initializes the Particle Swarm Optimization solver.

        Args:
            problem: The optimization problem to solve.
            swarm_size: The number of particles in the swarm.
            w: The inertia weight, controlling the influence of the previous
               velocity.
            c1: The cognitive coefficient, scaling the influence of the
               particle's personal best.
            c2: The social coefficient, scaling the influence of the swarm's
               global best.
            detect_changes: If True and the problem is dynamic, a sentry (the
               global best position) is evaluated again at the start of every
               iteration; when its objective or constraint values have moved,
               the personal bests are re-evaluated (`respond_to_change`).
               Off by default, so a PSO used as a penalty swarm is unchanged.
            **kwargs: Additional keyword arguments (not used in this canonical
               PSO).
        """
        super().__init__(problem, name, constraint_handler=constraint_handler)
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Change detection with a sentry. The sentry is evaluated on the
        # original problem when the PSO solves a proxy (the co-evolutionary
        # Lagrangian), so a change of the multipliers is not taken for a
        # change of the environment.
        self._detect_changes = bool(detect_changes) and any(
            self.problem.is_dynamic()
        )
        self._sentry_problem = getattr(self.problem, "original_problem", self.problem)
        self._sentry_x: Optional[List[float]] = None
        self._sentry_eval: Optional[Evaluation] = None
        self._environment_checked = False
        self.n_changes_detected = 0

        lower_bounds, upper_bounds = self.problem.bounds

        # Initialize swarm
        self.population = self._initialize_positions()
        self.velocities = [
            [
                (
                    random.uniform(
                        -abs(upper_bounds[i] - lower_bounds[i]),
                        abs(upper_bounds[i] - lower_bounds[i]),
                    )
                    * 0.25
                )
                for i in range(self.problem.dimension)
            ]
            for _ in range(self.swarm_size)
        ]

        # Evaluate initial population and set personal bests
        self.evaluations = [self.problem.evaluate(pos) for pos in self.population]
        self.pbest_positions = copy.deepcopy(self.population)
        self.pbest_evaluations = copy.deepcopy(self.evaluations)

        # Initialize global best using comparator (respects constraints)
        best_initial_idx = 0
        for i in range(1, self.swarm_size):
            if self.comparator.is_better(
                self.evaluations[i], self.evaluations[best_initial_idx]
            ):
                best_initial_idx = i
        self.gbest_position = copy.deepcopy(self.population[best_initial_idx])
        self.gbest_evaluation = copy.deepcopy(self.evaluations[best_initial_idx])

    def _initialize_positions(self) -> List[List[float]]:
        """Creates the initial particle positions."""
        positions = []
        lower_bounds, upper_bounds = self.problem.bounds
        for _ in range(self.swarm_size):
            pos = [
                random.uniform(lower_bounds[i], upper_bounds[i])
                for i in range(self.problem.dimension)
            ]
            positions.append(pos)
        return positions

    def check_environment(self, tolerance: float = 1e-12) -> None:
        """Detects an environment change with the sentry and responds to it,
        at most once per iteration. Called by `step`, or earlier by a wrapping
        solver (the co-evolutionary solver calls it before it updates the
        multipliers)."""
        if self._environment_checked or not self._detect_changes:
            return
        self._environment_checked = True

        fresh = None
        if self._sentry_x is not None:
            fresh = self._sentry_problem.evaluate(list(self._sentry_x))
            if _evaluations_differ(fresh, self._sentry_eval, tolerance):
                self.n_changes_detected += 1
                self.respond_to_change()

        # The sentry follows the global best position.
        if fresh is not None and self._sentry_x == self.gbest_position:
            self._sentry_eval = fresh
        else:
            self._sentry_x = copy.deepcopy(self.gbest_position)
            self._sentry_eval = self._sentry_problem.evaluate(list(self._sentry_x))

    def step(self) -> None:
        """Performs one iteration of the PSO algorithm."""
        self.check_environment()
        self._environment_checked = False
        lower_bounds, upper_bounds = self.problem.bounds

        for i in range(self.swarm_size):
            # 1. Update Velocity
            for d in range(self.problem.dimension):
                r1 = random.random()
                r2 = random.random()

                cognitive_component = (
                    self.c1 * r1 * (self.pbest_positions[i][d] - self.population[i][d])
                )
                social_component = (
                    self.c2 * r2 * (self.gbest_position[d] - self.population[i][d])
                )
                inertia_component = self.w * self.velocities[i][d]

                self.velocities[i][d] = (
                    inertia_component + cognitive_component + social_component
                )

            # 2. Update Position
            for d in range(self.problem.dimension):
                self.population[i][d] += self.velocities[i][d]
                # Clamp position to stay within bounds
                self.population[i][d] = max(
                    lower_bounds[d], min(self.population[i][d], upper_bounds[d])
                )

            # 3. Evaluate new position
            self.evaluations[i] = self.problem.evaluate(self.population[i])

            # 4. Update Personal Best (pbest)
            if self.comparator.is_better(
                self.evaluations[i], self.pbest_evaluations[i]
            ):
                self.pbest_positions[i] = copy.deepcopy(self.population[i])
                self.pbest_evaluations[i] = copy.deepcopy(self.evaluations[i])

                # 5. Update Global Best (gbest) - only check if pbest was updated
                if self.comparator.is_better(
                    self.pbest_evaluations[i], self.gbest_evaluation
                ):
                    self.gbest_position = copy.deepcopy(self.pbest_positions[i])
                    self.gbest_evaluation = copy.deepcopy(self.pbest_evaluations[i])

    def respond_to_change(self) -> None:
        """Re-evaluates the swarm's memory after a landscape change.

        The personal best positions are kept, but re-scored against the
        current landscape, and the global best is recomputed from them.
        """
        self.pbest_evaluations = [
            self.problem.evaluate(pos) for pos in self.pbest_positions
        ]
        best = 0
        for i in range(1, len(self.pbest_positions)):
            if self.comparator.is_better(
                self.pbest_evaluations[i], self.pbest_evaluations[best]
            ):
                best = i
        self.gbest_position = copy.deepcopy(self.pbest_positions[best])
        self.gbest_evaluation = copy.deepcopy(self.pbest_evaluations[best])

    def get_result(self) -> List[Tuple[List[float], Evaluation[float]]]:
        """Returns the global best solution found by the swarm."""
        return [(self.gbest_position, self.gbest_evaluation)]

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


class QPSO(PSO):
    """
    A Quantum Particle Swarm Optimization (QPSO) algorithm for dynamic problems.

    QPSO enhances diversity by splitting the swarm into two subgroups:
    1. Neutral Particles: Behave according to the canonical PSO update rules.
    2. Quantum Particles: Move randomly within a "quantum cloud" (a
      hypersphere) centered around the current global best position.

    This dual mechanism balances exploitation and exploration, making the
    algorithm well-suited for dynamic optimization problems.
    """

    def __init__(
        self,
        problem: Problem[List[float], float],
        name: str,
        swarm_size: int,
        w: float,
        c1: float,
        c2: float,
        split_ratio: float,
        r_cloud: float,
        **kwargs,
    ):
        """
        Initializes the Quantum Particle Swarm Optimization solver.

        Args:
            problem: The dynamic optimization problem.
            name: the name of the solver
            swarm_size: The total number of particles in the swarm.
            w: The inertia weight for neutral particles.
            c1: The cognitive coefficient for neutral particles.
            c2: The social coefficient for neutral particles.
            split_ratio: The proportion of the swarm designated as neutral
               particles. The rest will be quantum particles.
            r_cloud: The radius of the quantum cloud for quantum particles.
            **kwargs: Additional keyword arguments.
        """
        # Initialize the base PSO class which sets up a full swarm
        super().__init__(problem, name, swarm_size, w, c1, c2, **kwargs)

        self.split_ratio = split_ratio
        self.r_cloud = r_cloud

        # --- Split the swarm into neutral and quantum subgroups ---
        num_neutral = int(self.swarm_size * self.split_ratio)
        self.neutral_indices = list(range(num_neutral))
        self.quantum_indices = list(range(num_neutral, self.swarm_size))

    def step(self) -> None:
        """Performs one iteration of the QPSO algorithm."""
        lower_bounds, upper_bounds = self.problem.bounds

        # --- 1. Update Neutral Particles ---
        for i in self.neutral_indices:
            # Update velocity using standard PSO equation
            for d in range(self.problem.dimension):
                r1, r2 = random.random(), random.random()
                cognitive = (
                    self.c1 * r1 * (self.pbest_positions[i][d] - self.population[i][d])
                )
                social = self.c2 * r2 * (self.gbest_position[d] - self.population[i][d])
                self.velocities[i][d] = (
                    (self.w * self.velocities[i][d]) + cognitive + social
                )

            # Update position
            for d in range(self.problem.dimension):
                self.population[i][d] += self.velocities[i][d]
                self.population[i][d] = max(
                    lower_bounds[d], min(self.population[i][d], upper_bounds[d])
                )

            # Evaluate and update pbest/gbest
            self._evaluate_and_update_bests(i)

        # --- 2. Update Quantum Particles ---
        for i in self.quantum_indices:
            # Update position by sampling the quantum cloud
            for d in range(self.problem.dimension):
                # Sample a uniform distribution centered on gbest with radius r_cloud
                self.population[i][d] = random.uniform(
                    self.gbest_position[d] - self.r_cloud,
                    self.gbest_position[d] + self.r_cloud,
                )
                # Ensure the particle stays within bounds
                self.population[i][d] = max(
                    lower_bounds[d], min(self.population[i][d], upper_bounds[d])
                )

            # Evaluate and update pbest/gbest
            self._evaluate_and_update_bests(i)

    def _evaluate_and_update_bests(self, particle_idx: int):
        """
        Evaluates a particle and updates its personal best and the global best.
        """
        i = particle_idx
        # Evaluate new position
        self.evaluations[i] = self.problem.evaluate(self.population[i])

        # Update Personal Best (pbest)
        if self.comparator.is_better(self.evaluations[i], self.pbest_evaluations[i]):
            self.pbest_positions[i] = copy.deepcopy(self.population[i])
            self.pbest_evaluations[i] = copy.deepcopy(self.evaluations[i])

            # 5. Update Global Best (gbest) - only check if pbest was updated
            if self.comparator.is_better(
                self.pbest_evaluations[i], self.gbest_evaluation
            ):
                self.gbest_position = copy.deepcopy(self.pbest_positions[i])
                self.gbest_evaluation = copy.deepcopy(self.pbest_evaluations[i])
