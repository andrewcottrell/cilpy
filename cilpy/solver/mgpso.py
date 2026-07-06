# cilpy/solver/mgpso.py
"""Multi-Guide Particle Swarm Optimization (MGPSO).

This module provides the `MGPSO` solver, a multi-objective particle swarm
optimizer that assigns one sub-swarm to each objective function and shares a
bounded archive of non-dominated solutions between them.

Algorithm summary (Scheepers, Engelbrecht and Cleghorn, 2019):

1.  For a problem with ``nm`` objectives, ``nm`` sub-swarms are created. Each
    sub-swarm optimizes exactly one objective using canonical PSO dynamics
    (personal best and neighbourhood best guides).
2.  All sub-swarms contribute their solutions to a single shared archive that
    retains only mutually non-dominated solutions. When the archive exceeds
    its capacity, the most crowded solution (smallest crowding distance) is
    evicted, preserving spread along the Pareto front.
3.  The velocity update gains a third attractor, the *archive guide*, selected
    from the archive by tournament selection on crowding distance. A
    per-particle trade-off coefficient ``lambda ~ U(0, 1)``, sampled once at
    initialisation, balances the influence of the neighbourhood guide against
    the archive guide:

        v = w*v + c1*r1*(y - x) + lam*c2*r2*(y_hat - x)
                                + (1 - lam)*c3*r3*(a_hat - x)

Control parameters can either be fixed (pass ``w``, ``c1``, ``c2``, ``c3``) or,
by default, re-sampled each iteration per particle subject to the MGPSO
order-1/order-2 stability condition (Scheepers et al., 2019, Eq. 18).

Constraint handling: personal-best and neighbourhood-best comparisons are
delegated to the solver's `ConstraintHandler` comparator, so a feasibility
based comparator (e.g. Deb's rules) can be plugged in without modifying the
algorithm. Archive insertion uses Pareto dominance on the objective vector;
for constrained problems the intended usage in this project is to wrap the
problem in the co-evolutionary Lagrangian framework (see
`cilpy.solver.ccls`), which folds constraint penalties into the objectives.

References:
    C. Scheepers, A. P. Engelbrecht, and C. W. Cleghorn, "Multi-guide particle
    swarm optimization for multi-objective optimization: empirical and
    stability analysis," Swarm Intelligence, vol. 13, pp. 245-276, 2019.
    doi: 10.1007/s11721-019-00171-0.
"""

import copy
from typing import List, Optional, Tuple

import numpy as np

from cilpy.problem import Problem, Evaluation
from cilpy.solver import Solver
from cilpy.solver.chm import ConstraintHandler


def dominates(fitness_a: List[float], fitness_b: List[float]) -> bool:
    """Pareto dominance for minimization.

    Solution `a` dominates solution `b` if `a` is no worse than `b` in all
    objectives and strictly better in at least one.

    Args:
        fitness_a: Objective vector of the first solution.
        fitness_b: Objective vector of the second solution.

    Returns:
        True if `fitness_a` dominates `fitness_b`.
    """
    no_worse = all(a <= b for a, b in zip(fitness_a, fitness_b))
    strictly_better = any(a < b for a, b in zip(fitness_a, fitness_b))
    return no_worse and strictly_better


class _Archive:
    """Bounded archive of mutually non-dominated solutions.

    Stores `(position, Evaluation)` pairs. The full `Evaluation` is retained
    (rather than only the fitness vector) so that constraint information
    remains available for feasibility reporting and for future
    constrained-dominance extensions.

    When the archive exceeds `capacity` after an insertion, the solution with
    the smallest crowding distance is evicted.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.positions: List[np.ndarray] = []
        self.evaluations: List[Evaluation] = []

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def entries(self) -> List[Tuple[np.ndarray, Evaluation]]:
        return list(zip(self.positions, self.evaluations))

    def insert(self, position: np.ndarray, evaluation: Evaluation) -> None:
        """Attempts to insert a solution into the archive.

        The solution is rejected if it is dominated by any archive member.
        Otherwise, all archive members it dominates are removed and the
        solution is added. If the archive then exceeds capacity, the most
        crowded member is evicted.
        """
        fitness = evaluation.fitness

        # Reject if dominated by any existing member.
        for existing in self.evaluations:
            if dominates(existing.fitness, fitness):
                return

        # Remove members dominated by the new solution.
        survivors = [
            (p, e)
            for p, e in zip(self.positions, self.evaluations)
            if not dominates(fitness, e.fitness)
        ]
        self.positions = [p for p, _ in survivors]
        self.evaluations = [e for _, e in survivors]

        self.positions.append(position.copy())
        self.evaluations.append(copy.deepcopy(evaluation))

        if len(self) > self.capacity:
            self._evict_most_crowded()

    def _crowding_distances(self) -> np.ndarray:
        """Crowding distance of every archive member, in objective space.

        Boundary solutions receive infinite distance (Deb et al., 2002).
        """
        n = len(self)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([np.inf])

        obj_matrix = np.array([e.fitness for e in self.evaluations])
        nm = obj_matrix.shape[1]
        distances = np.zeros(n)

        for m in range(nm):
            col = obj_matrix[:, m]
            sorted_idx = np.argsort(col)
            sorted_col = col[sorted_idx]

            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf

            f_range = sorted_col[-1] - sorted_col[0]
            if f_range == 0:
                continue

            for k in range(1, n - 1):
                distances[sorted_idx[k]] += (
                    sorted_col[k + 1] - sorted_col[k - 1]
                ) / f_range

        return distances

    def _evict_most_crowded(self) -> None:
        """Removes the member with the smallest crowding distance."""
        distances = self._crowding_distances()
        most_crowded_idx = int(np.argmin(distances))
        self.positions.pop(most_crowded_idx)
        self.evaluations.pop(most_crowded_idx)

    def tournament_select(self, tournament_size: int) -> Optional[np.ndarray]:
        """Selects an archive guide via crowding-distance tournament.

        A tournament of `tournament_size` random members is held and the
        member with the *largest* crowding distance wins, drawing particles
        toward sparsely populated regions of the front.

        Returns:
            The winning position, or None if the archive is empty.
        """
        n = len(self)
        if n == 0:
            return None

        k = min(tournament_size, n)
        indices = np.random.choice(n, size=k, replace=False)
        distances = self._crowding_distances()
        winner = indices[int(np.argmax(distances[indices]))]
        return self.positions[winner].copy()


class _Particle:
    """A single MGPSO particle.

    Holds position, velocity, personal best (position and its `Evaluation` on
    the sub-swarm's assigned objective), the fixed per-particle lambda, and
    the current control parameters used by stability-guided sampling.
    """

    MAX_SAMPLE_ATTEMPTS = 10

    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        self.lower = lower
        self.upper = upper
        self.nx = lower.shape[0]

        self.x = np.random.uniform(lower, upper)
        self.v = np.zeros(self.nx)

        # Personal best on the assigned objective (scalar Evaluation).
        self.y = self.x.copy()
        self.y_eval: Optional[Evaluation] = None

        # Lambda: sampled once, fixed for the particle's lifetime.
        self.lam = np.random.uniform(0.0, 1.0)

        # Current control parameters (reused if stability sampling fails).
        self.w = 0.5
        self.c1 = 1.0
        self.c2 = 1.0
        self.c3 = 1.0

    def _satisfies_stability(
        self, w: float, c1: float, c2: float, c3: float
    ) -> bool:
        """Checks the MGPSO order-1/order-2 stability condition.

        (Scheepers et al., 2019, Eq. 18):

            0 < rho < 4(1 - w^2) /
                (1 - w + (c1^2 + lam^2 c2^2 + (1-lam)^2 c3^2)(1+w) / (3 rho^2))

        where rho = c1 + lam*c2 + (1 - lam)*c3 and |w| < 1.
        """
        if abs(w) >= 1.0:
            return False

        lam = self.lam
        rho = c1 + lam * c2 + (1.0 - lam) * c3
        if rho <= 0:
            return False

        numerator = 4.0 * (1.0 - w**2)
        denom_term = (
            c1**2 + lam**2 * c2**2 + (1.0 - lam) ** 2 * c3**2
        ) * (1.0 + w)
        denominator = 1.0 - w + denom_term / (3.0 * rho**2)

        return rho < numerator / denominator

    def sample_stable_params(self) -> None:
        """Samples w ~ U(0,1) and c1, c2, c3 ~ U(0,2) until stable.

        Limited to `MAX_SAMPLE_ATTEMPTS`; the previous parameters are reused
        if every attempt fails (the fallback used in the original paper).
        """
        for _ in range(self.MAX_SAMPLE_ATTEMPTS):
            w = np.random.uniform(0.0, 1.0)
            c1 = np.random.uniform(0.0, 2.0)
            c2 = np.random.uniform(0.0, 2.0)
            c3 = np.random.uniform(0.0, 2.0)
            if self._satisfies_stability(w, c1, c2, c3):
                self.w, self.c1, self.c2, self.c3 = w, c1, c2, c3
                return

    def set_fixed_params(self, w: float, c1: float, c2: float, c3: float) -> None:
        self.w, self.c1, self.c2, self.c3 = w, c1, c2, c3

    def update(self, y_hat: np.ndarray, a_hat: Optional[np.ndarray]) -> None:
        """One velocity and position update (Scheepers et al., 2019, Eq. 3).

        Args:
            y_hat: The sub-swarm neighbourhood best position.
            a_hat: The archive guide position, or None if the archive is
                empty, in which case the personal best is used as fallback.
        """
        if a_hat is None:
            a_hat = self.y

        r1 = np.random.uniform(0.0, 1.0, self.nx)
        r2 = np.random.uniform(0.0, 1.0, self.nx)
        r3 = np.random.uniform(0.0, 1.0, self.nx)
        lam = self.lam

        self.v = (
            self.w * self.v
            + self.c1 * r1 * (self.y - self.x)
            + lam * self.c2 * r2 * (y_hat - self.x)
            + (1.0 - lam) * self.c3 * r3 * (a_hat - self.x)
        )
        self.x = np.clip(self.x + self.v, self.lower, self.upper)


class _Subswarm:
    """A sub-swarm responsible for a single objective.

    Uses a global (star) neighbourhood topology: every particle shares the
    same neighbourhood best.
    """

    def __init__(
        self,
        n_particles: int,
        lower: np.ndarray,
        upper: np.ndarray,
        objective_idx: int,
    ):
        self.objective_idx = objective_idx
        self.particles = [_Particle(lower, upper) for _ in range(n_particles)]
        self.evaluations: List[Evaluation] = []

        self.y_hat: np.ndarray = self.particles[0].x.copy()
        self.y_hat_eval: Optional[Evaluation] = None

    def _scalar_eval(self, evaluation: Evaluation) -> Evaluation:
        """Projects a full multi-objective Evaluation onto this sub-swarm's
        assigned objective, preserving constraint information so that
        constraint-aware comparators remain applicable.
        """
        return Evaluation(
            fitness=float(evaluation.fitness[self.objective_idx]),
            constraints_inequality=evaluation.constraints_inequality,
            constraints_equality=evaluation.constraints_equality,
        )

    def evaluate(self, problem: Problem) -> None:
        """Evaluates every particle against the full problem."""
        self.evaluations = [
            problem.evaluate(list(p.x)) for p in self.particles
        ]

    def update_bests(self, comparator: ConstraintHandler) -> None:
        """Updates personal bests and the neighbourhood best.

        Comparisons are made on the assigned objective via the solver's
        comparator, so constraint-handling strategies apply transparently.
        """
        for particle, evaluation in zip(self.particles, self.evaluations):
            scalar = self._scalar_eval(evaluation)

            if particle.y_eval is None or comparator.is_better(
                scalar, particle.y_eval
            ):
                particle.y = particle.x.copy()
                particle.y_eval = scalar

            if self.y_hat_eval is None or comparator.is_better(
                scalar, self.y_hat_eval
            ):
                self.y_hat = particle.x.copy()
                self.y_hat_eval = scalar

    def update_archive(self, archive: _Archive) -> None:
        """Offers every particle's current solution to the shared archive."""
        for particle, evaluation in zip(self.particles, self.evaluations):
            archive.insert(particle.x, evaluation)

    def update_particles(
        self,
        archive: _Archive,
        tournament_size: int,
        fixed_params: Optional[Tuple[float, float, float, float]],
    ) -> None:
        """Updates velocity and position of every particle."""
        for particle in self.particles:
            if fixed_params is None:
                particle.sample_stable_params()
            else:
                particle.set_fixed_params(*fixed_params)
            a_hat = archive.tournament_select(tournament_size)
            particle.update(self.y_hat, a_hat)


class MGPSO(Solver[List[float], List[float]]):
    """Multi-Guide Particle Swarm Optimization solver.

    One sub-swarm is created per objective; all sub-swarms share a bounded
    non-dominated archive whose members act as additional velocity guides.

    Attributes:
        n_objectives (int): The number of objectives, determined by probing
            the problem once at initialisation.
        swarm_size (int): Number of particles *per sub-swarm*. The total
            population is `n_objectives * swarm_size`, which is also the
            archive capacity.

    Example:
        .. code-block:: python

            from cilpy.problem.multi_objective import ZDT1
            from cilpy.solver.mgpso import MGPSO
            from cilpy.runner import ExperimentRunner

            runner = ExperimentRunner(
                problems=[ZDT1()],
                solver_configurations=[{
                    "class": MGPSO,
                    "params": {"name": "MGPSO", "swarm_size": 50},
                }],
                num_runs=30,
                max_iterations=1000,
            )
            runner.run_experiments()
    """

    def __init__(
        self,
        problem: Problem[List[float], List[float]],
        name: str,
        swarm_size: int,
        tournament_size: int = 3,
        w: Optional[float] = None,
        c1: Optional[float] = None,
        c2: Optional[float] = None,
        c3: Optional[float] = None,
        constraint_handler: Optional[ConstraintHandler] = None,
        **kwargs,
    ):
        """Initializes the MGPSO solver.

        Args:
            problem: The multi-objective optimization problem to solve. Its
                `evaluate` method must return an `Evaluation` whose fitness is
                a list of floats.
            name: The name of the solver instance.
            swarm_size: The number of particles in each sub-swarm.
            tournament_size: Archive-guide tournament size (2 or 3 in the
                original paper). Defaults to 3.
            w: Inertia weight. If `w`, `c1`, `c2`, and `c3` are all provided,
                fixed control parameters are used; otherwise parameters are
                re-sampled per particle each iteration subject to the MGPSO
                stability condition.
            c1: Cognitive (personal best) acceleration coefficient.
            c2: Social (neighbourhood best) acceleration coefficient.
            c3: Archive-guide acceleration coefficient.
            constraint_handler: Optional comparator used for personal and
                neighbourhood best updates. Defaults to fitness-only
                comparison.
            **kwargs: Additional keyword arguments (unused).
        """
        super().__init__(problem, name, constraint_handler=constraint_handler)

        self.swarm_size = swarm_size
        self.tournament_size = tournament_size

        fixed = (w, c1, c2, c3)
        if all(p is not None for p in fixed):
            self._fixed_params: Optional[Tuple[float, float, float, float]] = (
                float(w), float(c1), float(c2), float(c3)
            )
        else:
            self._fixed_params = None

        lower_list, upper_list = self.problem.bounds
        lower = np.asarray(lower_list, dtype=float)
        upper = np.asarray(upper_list, dtype=float)

        # Cilpy problems do not expose an objective count; probe once.
        probe = self.problem.evaluate(list(lower))
        if not isinstance(probe.fitness, (list, tuple, np.ndarray)):
            raise ValueError(
                f"MGPSO requires a multi-objective problem, but "
                f"'{self.problem.name}' returned a scalar fitness."
            )
        self.n_objectives = len(probe.fitness)

        self._archive = _Archive(capacity=self.n_objectives * self.swarm_size)
        self._swarms = [
            _Subswarm(self.swarm_size, lower, upper, objective_idx=m)
            for m in range(self.n_objectives)
        ]

        # Evaluate the initial population so that get_result() and the
        # population accessors are valid before the first step() call.
        self._dynamic = any(self.problem.is_dynamic())
        for swarm in self._swarms:
            swarm.evaluate(self.problem)
            swarm.update_bests(self.comparator)
            swarm.update_archive(self._archive)

    def _refresh_archive(self) -> None:
        """Re-evaluates archive members against the current environment.

        Used for dynamic problems, where stored fitness values become stale
        when the landscape changes. Dominated members are pruned after
        re-evaluation.
        """
        positions = [p.copy() for p in self._archive.positions]
        self._archive.positions = []
        self._archive.evaluations = []
        for position in positions:
            evaluation = self.problem.evaluate(list(position))
            self._archive.insert(position, evaluation)

    def step(self) -> None:
        """Performs one MGPSO iteration.

        Phase 1: every sub-swarm evaluates its particles, updates personal
        and neighbourhood bests, and offers solutions to the shared archive.
        Phase 2: every sub-swarm updates particle velocities and positions
        using the three-guide velocity equation.
        """
        if self._dynamic:
            self._refresh_archive()

        for swarm in self._swarms:
            swarm.update_particles(
                self._archive, self.tournament_size, self._fixed_params
            )

        for swarm in self._swarms:
            swarm.evaluate(self.problem)
            swarm.update_bests(self.comparator)
            swarm.update_archive(self._archive)

    def get_result(self) -> List[Tuple[List[float], Evaluation[List[float]]]]:
        """Returns the archive of non-dominated solutions (the Pareto front).

        Returns:
            A list of `(solution, Evaluation)` tuples, one per archive member.
        """
        return [
            (position.tolist(), copy.deepcopy(evaluation))
            for position, evaluation in self._archive.entries
        ]

    def get_population(self) -> List[List[float]]:
        """Returns the positions of all particles across all sub-swarms."""
        return [
            particle.x.tolist()
            for swarm in self._swarms
            for particle in swarm.particles
        ]

    def get_population_evaluations(self) -> List[Evaluation[List[float]]]:
        """Returns the evaluations of all particles across all sub-swarms."""
        return [
            evaluation
            for swarm in self._swarms
            for evaluation in swarm.evaluations
        ]