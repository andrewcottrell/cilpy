# cilpy/problem/dynamic_multi_objective.py
"""Dynamic Multi-Objective Benchmark Problems.

Time model (Farina, Deb and Amato, 2004): a discrete iteration counter tau
is advanced by `begin_iteration()`, and the environment time is

    t = (1 / n_t) * floor(tau / tau_t)

where `tau_t` is the change frequency (iterations per environment) and
`n_t` the change severity (larger n_t = smaller steps = milder changes).

Problems provided, by dynamic category:

- `FDA1` : dynamic objectives, box constraints only. The Pareto SET moves
  through decision space while the Pareto FRONT is time-invariant
  (f2 = 1 - sqrt(f1)). Tests tracking in decision space. [Farina et al., 2004]
- `FDA3` : dynamic objectives; the Pareto front itself moves and the
  solution density along it changes. Tests tracking in objective space.
  [Farina et al., 2004]
  (FDA2 is omitted deliberately: the original formulation contains a known
  error and multiple incompatible corrections circulate; FDA1 + FDA3 cover
  the same phenomena unambiguously.)
- `DTNK` : SODC -- static TNK objectives, dynamic constraint: the radius of
  the periodic ring constraint oscillates, r(t) = 1 + 0.2 sin(0.5 pi t),
  moving the feasible boundary that the Pareto front lies on.
- `DTNK2` : DODC -- DTNK's dynamic constraint plus time-translated
  objectives f1 = x1 + 0.2|sin(0.5 pi t)|, f2 = x2 + 0.2|cos(0.5 pi t)|.

All problems implement `true_pareto_front(n_points)` reflecting the CURRENT
environment time, so distance metrics (IGD/GD/spread) computed against it
remain valid across changes. For DTNK/DTNK2 the front is grid-sampled per
environment and cached by t (environments repeat periodically, so the cache
is bounded).

References:
    M. Farina, K. Deb, and P. Amato, "Dynamic multiobjective optimization
    problems: test cases, approximations, and applications," IEEE
    Transactions on Evolutionary Computation, vol. 8, no. 5, pp. 425-442,
    2004. doi: 10.1109/TEVC.2004.831456.

    M. Tanaka, H. Watanabe, Y. Furukawa, and T. Tanino, "GA-based decision
    support system for multicriteria optimization," in Proc. IEEE Int. Conf.
    Systems, Man and Cybernetics, vol. 2, 1995, pp. 1556-1561. (static TNK)
"""

import math
from typing import List, Tuple

import numpy as np

from cilpy.problem import Problem, Evaluation


class _DynamicMOBase(Problem[List[float], List[float]]):
    """Shared time mechanics for dynamic multi-objective problems."""

    def __init__(self, dimension, bounds, name, tau_t: int = 10, n_t: int = 10):
        """
        Args:
            tau_t: Change frequency -- iterations per environment.
            n_t: Change severity -- number of steps discretising t; larger
                values give smaller, more frequent-looking steps in t.
        """
        super().__init__(dimension=dimension, bounds=bounds, name=name)
        self.tau_t = tau_t
        self.n_t = n_t
        self._tau = 0

    @property
    def t(self) -> float:
        """Current environment time."""
        return (self._tau // self.tau_t) / self.n_t

    def begin_iteration(self) -> None:
        """Advances the iteration counter (called by the ExperimentRunner)."""
        self._tau += 1

    def reset_time(self) -> None:
        """Resets time to zero (call between independent runs)."""
        self._tau = 0

    def is_multi_objective(self) -> bool:
        return True


class FDA1(_DynamicMOBase):
    """FDA1: moving Pareto set, time-invariant Pareto front.

        f1(x)   = x1
        g(x, t) = 1 + sum_{i=2..n} (xi - G(t))^2,   G(t) = sin(0.5 pi t)
        f2(x)   = g * (1 - sqrt(f1 / g))

    with x1 in [0, 1], xi in [-1, 1] for i >= 2 (n = 20 by default). The
    Pareto-optimal set is xi = G(t) for i >= 2, which sweeps through the
    decision space, while the front remains f2 = 1 - sqrt(f1). An algorithm
    that fails to track the moving set sees its g value (and hence IGD)
    degrade after every change.

    Reference: [Farina et al., 2004], problem FDA1.
    """

    def __init__(self, dimension: int = 20, tau_t: int = 10, n_t: int = 10):
        lower = [0.0] + [-1.0] * (dimension - 1)
        upper = [1.0] + [1.0] * (dimension - 1)
        super().__init__(dimension, (lower, upper), "FDA1", tau_t, n_t)

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x = np.asarray(solution, dtype=float)
        G = math.sin(0.5 * math.pi * self.t)
        f1 = x[0]
        g = 1.0 + float(np.sum((x[1:] - G) ** 2))
        f2 = g * (1.0 - math.sqrt(f1 / g))
        return Evaluation(fitness=[float(f1), float(f2)])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (True, False)

    def true_pareto_front(self, n_points: int = 500) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


class FDA3(_DynamicMOBase):
    """FDA3: moving Pareto front with time-varying solution density.

        f1(x, t) = x1^F(t),                    F(t) = 10^(2 sin(0.5 pi t))
        g(x, t)  = 1 + G(t) + sum_{i=2..n} (xi - G(t))^2,
                                               G(t) = |sin(0.5 pi t)|
        f2(x, t) = g * (1 - sqrt(f1 / g))

    with x1 in [0, 1], xi in [-1, 1] for i >= 2 (n = 20 by default). The
    front f2 = (1 + G(t)) (1 - sqrt(f1 / (1 + G(t)))) moves vertically with
    G(t), and F(t) redistributes solution density along f1, so both the
    front's location and its sampling difficulty change over time.

    Reference: [Farina et al., 2004], problem FDA3.
    """

    def __init__(self, dimension: int = 20, tau_t: int = 10, n_t: int = 10):
        lower = [0.0] + [-1.0] * (dimension - 1)
        upper = [1.0] + [1.0] * (dimension - 1)
        super().__init__(dimension, (lower, upper), "FDA3", tau_t, n_t)

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x = np.asarray(solution, dtype=float)
        G = abs(math.sin(0.5 * math.pi * self.t))
        F = 10.0 ** (2.0 * math.sin(0.5 * math.pi * self.t))
        f1 = float(x[0]) ** F
        g = 1.0 + G + float(np.sum((x[1:] - G) ** 2))
        f2 = g * (1.0 - math.sqrt(f1 / g))
        return Evaluation(fitness=[float(f1), float(f2)])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (True, False)

    def true_pareto_front(self, n_points: int = 500) -> np.ndarray:
        G = abs(math.sin(0.5 * math.pi * self.t))
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = (1.0 + G) * (1.0 - np.sqrt(f1 / (1.0 + G)))
        return np.column_stack([f1, f2])


class _SampledDynamicFrontMixin:
    """Grid-sampled reference front for 2-variable dynamic problems.

    Fronts are computed for the CURRENT environment time and cached by t.
    Because t is periodic, the number of distinct environments (and hence
    cache entries) is bounded.
    """

    _GRID = 500

    def true_pareto_front(self, n_points: int = 500) -> np.ndarray:
        key = round(self.t, 9)
        cache = getattr(self, "_front_cache", None)
        if cache is None:
            cache = {}
            self._front_cache = cache

        if key not in cache:
            lower, upper = self.bounds
            a = np.linspace(lower[0], upper[0], self._GRID)
            b = np.linspace(lower[1], upper[1], self._GRID)
            X1, X2 = np.meshgrid(a, b)
            X1, X2 = X1.ravel(), X2.ravel()
            F, G = self._batch_evaluate(X1, X2)
            feasible = np.all(G <= 1e-9, axis=1)
            objectives = F[feasible]
            order = np.lexsort((objectives[:, 1], objectives[:, 0]))
            objectives = objectives[order]
            best_f2 = np.inf
            keep = np.zeros(len(objectives), dtype=bool)
            for i, (_, f2) in enumerate(objectives):
                if f2 < best_f2:
                    keep[i] = True
                    best_f2 = f2
            cache[key] = objectives[keep]

        front = cache[key]
        if len(front) > n_points:
            idx = np.linspace(0, len(front) - 1, n_points).astype(int)
            front = front[idx]
        return front.copy()


class DTNK(_SampledDynamicFrontMixin, _DynamicMOBase):
    """DTNK: static objectives with a dynamic constraint (SODC).

    The static TNK problem with an oscillating ring radius:

        min f1(x) = x1,  min f2(x) = x2
        s.t. x1^2 + x2^2 - r(t) - 0.1 cos(16 arctan(x1/x2)) >= 0,
             r(t) = 1 + 0.2 sin(0.5 pi t)
             (x1 - 0.5)^2 + (x2 - 0.5)^2 <= 0.5

    with x1, x2 in [0, pi]. The Pareto front lies ON the first constraint
    boundary, so when r(t) changes, previously optimal (boundary) solutions
    become either infeasible (r grew) or dominated interior points
    (r shrank) -- the canonical SODC stress test for archive maintenance.
    """

    def __init__(self, tau_t: int = 10, n_t: int = 10):
        super().__init__(
            2, ([0.0, 0.0], [math.pi, math.pi]), "DTNK", tau_t, n_t
        )

    def _r(self) -> float:
        return 1.0 + 0.2 * math.sin(0.5 * math.pi * self.t)

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x1, x2 = solution
        angle = math.atan2(x1, x2)
        g1 = -(x1**2 + x2**2 - self._r() - 0.1 * math.cos(16.0 * angle))
        g2 = (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2 - 0.5
        return Evaluation(
            fitness=[x1, x2], constraints_inequality=[g1, g2]
        )

    def _batch_evaluate(self, X1, X2):
        F = np.column_stack([X1, X2])
        angle = np.arctan2(X1, X2)
        G = np.column_stack([
            -(X1**2 + X2**2 - self._r() - 0.1 * np.cos(16.0 * angle)),
            (X1 - 0.5) ** 2 + (X2 - 0.5) ** 2 - 0.5,
        ])
        return F, G

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, True)


class DTNK2(_SampledDynamicFrontMixin, _DynamicMOBase):
    """DTNK2: dynamic objectives AND dynamic constraints (DODC).

    DTNK's oscillating ring constraint combined with time-translated
    objectives:

        min f1(x, t) = x1 + 0.2 |sin(0.5 pi t)|
        min f2(x, t) = x2 + 0.2 |cos(0.5 pi t)|

    subject to DTNK's constraints. Both the feasible region and the mapping
    to objective space move over time, so the front translates while its
    supporting boundary deforms -- the hardest of the four categories.
    """

    def __init__(self, tau_t: int = 10, n_t: int = 10):
        super().__init__(
            2, ([0.0, 0.0], [math.pi, math.pi]), "DTNK2", tau_t, n_t
        )

    def _r(self) -> float:
        return 1.0 + 0.2 * math.sin(0.5 * math.pi * self.t)

    def _shift(self) -> Tuple[float, float]:
        return (
            0.2 * abs(math.sin(0.5 * math.pi * self.t)),
            0.2 * abs(math.cos(0.5 * math.pi * self.t)),
        )

    def evaluate(self, solution: List[float]) -> Evaluation[List[float]]:
        x1, x2 = solution
        s1, s2 = self._shift()
        angle = math.atan2(x1, x2)
        g1 = -(x1**2 + x2**2 - self._r() - 0.1 * math.cos(16.0 * angle))
        g2 = (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2 - 0.5
        return Evaluation(
            fitness=[x1 + s1, x2 + s2], constraints_inequality=[g1, g2]
        )

    def _batch_evaluate(self, X1, X2):
        s1, s2 = self._shift()
        F = np.column_stack([X1 + s1, X2 + s2])
        angle = np.arctan2(X1, X2)
        G = np.column_stack([
            -(X1**2 + X2**2 - self._r() - 0.1 * np.cos(16.0 * angle)),
            (X1 - 0.5) ** 2 + (X2 - 0.5) ** 2 - 0.5,
        ])
        return F, G

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (True, True)