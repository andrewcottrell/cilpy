# Comparison Metrics

The `cilpy.compare.metrics` module provides performance metrics for evaluating
and comparing optimization algorithms.

## Available Metrics

- **IGD** (Inverted Generational Distance): measures how close an obtained front
  is to the true Pareto front.
- **GD** (Generational Distance): measures the distance from obtained solutions
  to the reference front.
- **Hypervolume**: the area dominated by the obtained front, bounded by a
  reference point.
- **Hypervolume Ratio**: `HV(front) / HV(reference front)`, clipped to [0, 1].
  Used as the per-iteration score `b` in multi-objective P_RED.
- **Spread (Δ)**: measures the distribution of solutions along the Pareto front.
- **P_RED**: `sqrt(mean((1 - b)^2))` over per-iteration scores, where `b = 1`
  is optimal. Generalises the single-objective relative error distance to
  multi-objective problems via the hypervolume ratio.

::: cilpy.compare.metrics
