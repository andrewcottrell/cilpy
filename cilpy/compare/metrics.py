# cilpy/compare/metrics.py
"""Performance metrics for multi-objective optimization.

This module provides the standard metrics used to assess the quality of an
approximated Pareto front, together with feasibility measures for constrained
problems. All metrics assume minimization of every objective.

Metric overview:

- `generational_distance` (GD): average distance from each obtained solution
  to the nearest point of the reference front. Measures *convergence*: how
  close the obtained front is to the true front. Blind to coverage.
- `inverted_generational_distance` (IGD): average distance from each
  reference-front point to the nearest obtained solution. Measures
  convergence *and* coverage simultaneously: a front that is close to the
  true front but covers only part of it scores poorly.
- `hypervolume` (HV): volume of objective space dominated by the front and
  bounded by a reference (nadir-like) point. Larger is better. The only
  strictly Pareto-compliant unary metric, and usable without knowing the
  true front.
- `spacing` (Schott): standard deviation of nearest-neighbour distances
  within the front. Measures *uniformity* of the distribution; 0 means
  perfectly evenly spaced.
- `spread` (Deb's Delta): combines extent (distance to the true front's
  extreme points) and uniformity. 0 is ideal; requires a reference front.
- `feasibility_rate`: percentage of evaluations satisfying all constraints.
- `nondominated_filter`: utility to reduce a set of objective vectors to its
  non-dominated subset.

References:
    D. A. Van Veldhuizen and G. B. Lamont, "Multiobjective evolutionary
    algorithm research: A history and analysis," Air Force Institute of
    Technology, Tech. Rep. TR-98-03, 1998. (GD)

    C. A. Coello Coello and M. Reyes Sierra, "A study of the parallelization
    of a coevolutionary multi-objective evolutionary algorithm," in Proc.
    MICAI 2004, LNCS vol. 2972, pp. 688-697, 2004. doi:
    10.1007/978-3-540-24694-7_71. (IGD)

    E. Zitzler and L. Thiele, "Multiobjective evolutionary algorithms: A
    comparative case study and the strength Pareto approach," IEEE
    Transactions on Evolutionary Computation, vol. 3, no. 4, pp. 257-271,
    1999. doi: 10.1109/4235.797969. (Hypervolume)

    J. R. Schott, "Fault tolerant design using single and multicriteria
    genetic algorithm optimization," M.S. thesis, Dept. Aeronautics and
    Astronautics, Massachusetts Institute of Technology, 1995. (Spacing)

    K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist
    multiobjective genetic algorithm: NSGA-II," IEEE Transactions on
    Evolutionary Computation, vol. 6, no. 2, pp. 182-197, 2002.
    doi: 10.1109/4235.996017. (Spread Delta)
"""

from typing import List, Optional, Sequence

import numpy as np

from cilpy.problem import Evaluation


def _as_matrix(front: Sequence[Sequence[float]]) -> np.ndarray:
    """Coerces a front to a 2-D float array of shape (n_solutions, n_objectives)."""
    matrix = np.asarray(front, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def nondominated_filter(front: Sequence[Sequence[float]]) -> np.ndarray:
    """Reduces a set of objective vectors to its non-dominated subset.

    Args:
        front: An (n, nm) collection of objective vectors (minimization).

    Returns:
        The non-dominated subset as an (k, nm) array, k <= n.
    """
    matrix = _as_matrix(front)
    n = matrix.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        others = np.delete(matrix, i, axis=0)
        dominated = np.any(
            np.all(others <= matrix[i], axis=1)
            & np.any(others < matrix[i], axis=1)
        )
        if dominated:
            keep[i] = False
    return matrix[keep]


def generational_distance(
    front: Sequence[Sequence[float]],
    reference_front: Sequence[Sequence[float]],
    p: float = 2.0,
) -> float:
    """Generational Distance (GD): convergence to the reference front.

    GD = ( (1/n) * sum_i d_i^p )^(1/p), where d_i is the Euclidean distance
    from obtained solution i to its nearest reference-front point.

    A GD of 0 means every obtained solution lies exactly on the reference
    front. GD says nothing about how much of the front is covered: a single
    solution sitting on the front achieves GD = 0.

    Args:
        front: The obtained front, shape (n, nm).
        reference_front: A dense sampling of the true Pareto front.
        p: The norm exponent. Defaults to 2 (the common convention).

    Returns:
        The generational distance (lower is better, 0 is ideal).
    """
    obtained = _as_matrix(front)
    reference = _as_matrix(reference_front)
    dists = np.min(
        np.linalg.norm(obtained[:, None, :] - reference[None, :, :], axis=2),
        axis=1,
    )
    return float((np.mean(dists**p)) ** (1.0 / p))


def inverted_generational_distance(
    front: Sequence[Sequence[float]],
    reference_front: Sequence[Sequence[float]],
    p: float = 2.0,
) -> float:
    """Inverted Generational Distance (IGD): convergence and coverage.

    IGD = ( (1/m) * sum_j d_j^p )^(1/p), where d_j is the Euclidean distance
    from reference-front point j to its nearest obtained solution.

    Because every reference point must be near *some* obtained solution for
    the score to be low, IGD penalizes both distance from the front and gaps
    in coverage, making it the standard single-number quality metric when
    the true front is known.

    Args:
        front: The obtained front, shape (n, nm).
        reference_front: A dense sampling of the true Pareto front.
        p: The norm exponent. Defaults to 2.

    Returns:
        The inverted generational distance (lower is better, 0 is ideal).
    """
    obtained = _as_matrix(front)
    reference = _as_matrix(reference_front)
    dists = np.min(
        np.linalg.norm(reference[:, None, :] - obtained[None, :, :], axis=2),
        axis=1,
    )
    return float((np.mean(dists**p)) ** (1.0 / p))


def hypervolume(
    front: Sequence[Sequence[float]],
    reference_point: Sequence[float],
) -> float:
    """Hypervolume (HV) of a bi-objective front (exact, O(n log n)).

    The hypervolume is the area of objective space dominated by the front
    and bounded above by `reference_point`. It rewards both convergence
    (solutions closer to the ideal enclose more area) and diversity
    (well-spread solutions enclose more area than clustered ones), and is
    the only unary metric that is strictly monotonic with respect to Pareto
    dominance. It requires no knowledge of the true front, only a fixed
    reference point that is dominated by every solution of interest.

    The reference point must be kept identical across all runs and
    algorithms being compared; changing it changes the ranking.

    Args:
        front: The obtained front, shape (n, 2). Solutions that do not
            dominate the reference point contribute nothing.
        reference_point: A point [r1, r2] dominated by the region of
            interest (e.g. slightly worse than the nadir point).

    Returns:
        The dominated area (higher is better).

    Raises:
        NotImplementedError: If the front has more than two objectives.
    """
    matrix = _as_matrix(front)
    if matrix.shape[1] != 2:
        raise NotImplementedError(
            "hypervolume() currently supports bi-objective fronts only. "
            "For three or more objectives use a WFG-style implementation."
        )
    r1, r2 = float(reference_point[0]), float(reference_point[1])

    # Keep only points that strictly dominate the reference point.
    mask = (matrix[:, 0] < r1) & (matrix[:, 1] < r2)
    points = nondominated_filter(matrix[mask])
    if points.size == 0:
        return 0.0

    # Sort by f1 ascending; f2 is then strictly descending on a ND set.
    points = points[np.argsort(points[:, 0])]

    area = 0.0
    prev_f2 = r2
    for f1, f2 in points:
        area += (r1 - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return float(area)


def spacing(front: Sequence[Sequence[float]]) -> float:
    """Schott's spacing metric: uniformity of the front's distribution.

    S = sqrt( (1/(n-1)) * sum_i (d_bar - d_i)^2 ), where d_i is the
    Manhattan distance from solution i to its nearest neighbour in the
    front and d_bar is the mean of the d_i.

    A spacing of 0 means all solutions are equidistant from their nearest
    neighbours (perfectly uniform). Spacing measures only *relative*
    uniformity: a tightly clustered front far from the true front can still
    have excellent spacing, so report it alongside a convergence metric.

    Args:
        front: The obtained front, shape (n, nm), n >= 2.

    Returns:
        The spacing value (lower is better, 0 is ideal).
    """
    matrix = _as_matrix(front)
    n = matrix.shape[0]
    if n < 2:
        return 0.0

    dist = np.sum(
        np.abs(matrix[:, None, :] - matrix[None, :, :]), axis=2
    )
    np.fill_diagonal(dist, np.inf)
    d = np.min(dist, axis=1)
    d_bar = np.mean(d)
    return float(np.sqrt(np.sum((d_bar - d) ** 2) / (n - 1)))


def spread(
    front: Sequence[Sequence[float]],
    reference_front: Sequence[Sequence[float]],
) -> float:
    """Deb's Delta spread metric for bi-objective fronts.

    Delta = (d_f + d_l + sum_i |d_i - d_bar|) / (d_f + d_l + (n-1) d_bar)

    where d_f and d_l are the distances between the extreme points of the
    reference front and the boundary solutions of the obtained front, d_i
    are consecutive-neighbour distances along the obtained front (sorted by
    the first objective), and d_bar is their mean.

    Delta = 0 is achieved only by a front that spans the full extent of the
    true front with perfectly uniform spacing; values grow as the front
    shrinks toward a subset of the true front or bunches up. Complements
    IGD by isolating the distribution aspect.

    Args:
        front: The obtained front, shape (n, 2), n >= 2.
        reference_front: A sampling of the true Pareto front, used only for
            its extreme points.

    Returns:
        The spread value (lower is better, 0 is ideal).

    Raises:
        NotImplementedError: If the front has more than two objectives.
    """
    matrix = _as_matrix(front)
    if matrix.shape[1] != 2:
        raise NotImplementedError(
            "spread() currently supports bi-objective fronts only."
        )
    reference = _as_matrix(reference_front)

    matrix = matrix[np.argsort(matrix[:, 0])]
    reference = reference[np.argsort(reference[:, 0])]

    n = matrix.shape[0]
    if n < 2:
        return 1.0

    d_f = float(np.linalg.norm(matrix[0] - reference[0]))
    d_l = float(np.linalg.norm(matrix[-1] - reference[-1]))

    consecutive = np.linalg.norm(np.diff(matrix, axis=0), axis=1)
    d_bar = float(np.mean(consecutive))

    numerator = d_f + d_l + float(np.sum(np.abs(consecutive - d_bar)))
    denominator = d_f + d_l + (n - 1) * d_bar
    if denominator == 0:
        return 0.0
    return numerator / denominator


def feasibility_rate(
    evaluations: Sequence[Evaluation],
    tolerance: float = 1e-6,
) -> float:
    """Percentage of evaluations satisfying all constraints.

    A solution is feasible when every inequality constraint value is <= 0
    and every equality constraint value is within `tolerance` of 0.
    Unconstrained evaluations count as feasible.

    Args:
        evaluations: `Evaluation` objects (e.g. from
            `Solver.get_population_evaluations()` or an archive).
        tolerance: Absolute tolerance for equality constraints.

    Returns:
        Feasibility rate in [0, 100].
    """
    if not evaluations:
        return 0.0

    def _is_feasible(evaluation: Evaluation) -> bool:
        if evaluation.constraints_inequality:
            if any(v > 0 for v in evaluation.constraints_inequality):
                return False
        if evaluation.constraints_equality:
            if any(abs(v) > tolerance for v in evaluation.constraints_equality):
                return False
        return True

    feasible = sum(1 for e in evaluations if _is_feasible(e))
    return 100.0 * feasible / len(evaluations)