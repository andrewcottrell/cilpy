# examples/project/paths.py
"""Locations of the project's experiment outputs.

Everything the campaign scripts read or write lives under this folder, so the
scripts behave the same from any working directory:

* ``results/``          aggregate tables (``results_*.csv``, ``comparison_*.csv``)
* ``results/per_run/``  per-run values of the pymoo baselines
* ``figures/``          report figures
* ``data/``             campaign output with the ``front`` column stripped
                        (small enough to version)
* ``raw/``              full campaign output, including the per-iteration
                        fronts the plotting scripts need (not versioned)

The one exception is the runner's own working folder ``out/``, which
``ExperimentRunner`` creates in the current working directory.
"""

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
PER_RUN_DIR = os.path.join(RESULTS_DIR, "per_run")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(PROJECT_DIR, "raw")


def results_path(filename: str) -> str:
    """Returns the path of an aggregate table, creating ``results/`` if needed."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, filename)
