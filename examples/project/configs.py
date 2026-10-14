# examples/project/configs.py
"""Algorithm configurations shared by every campaign script.

One place defines the algorithms, so the static, dynamic and refresh-mode
campaigns cannot drift apart:

* ``MGPSO``           -- plain MGPSO, no constraint handling.
* ``MGPSO_feasarch``  -- MGPSO-FA: MGPSO whose archive admits feasible
  solutions only; no penalty.
* ``CCMGPSO_strict``  -- co-evolutionary MGPSO, one penalty swarm per
  objective scored with the sampled dual, no fixed penalty (rho = 0),
  feasible-only archive admission.
* ``CCMGPSO_filter``  -- as above, infeasible solutions may enter the archive
  and are filtered out of the returned front.

Every algorithm detects environment changes with the archive members as
sentries. The shared-multiplier CCPSO with a fixed rho = 1 penalty of the
22 Sept campaign is kept as ``legacy_ccpso_config`` for reproducibility only.
"""
from cilpy.problem.dynamic_multi_objective import DTNK, DTNK4, DTNK5, DTNK6, FDA1, FDA3
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver
from cilpy.solver.mgpso import MGPSO
from cilpy.solver.pso import PSO

BASE_SEED = 26989395  # student number: memorable and citable
SWARM_SIZE = 50
FIXED_PARAMS = {"w": 0.72, "c1": 1.49, "c2": 1.49, "c3": 1.49}
MULTIPLIER_PARAMS = {"swarm_size": 30, "w": 0.40, "c1": 1.20, "c2": 1.20}
SENTRY_MODE = "archive"

# Penalty swarm settings of CCMGPSO (14 Oct diagnostics).
MAX_MULTIPLIER = 1000.0
MULTIPLIER_IMMIGRANTS = 0.3
MULTIPLIER_STEPS = 20

# Dynamic problems by category. SOSC is the static campaign.
BOX_DYNAMIC = [FDA1, FDA3]                      # dynamic, box constraints only
CONSTRAINED_DYNAMIC = [DTNK, DTNK5, DTNK6, DTNK4]
CATEGORY = {"DTNK": "SODC", "DTNK5": "DOSC", "DTNK6": "DODC", "DTNK4": "DODC"}
BOX_DYNAMIC_NAMES = ["FDA1", "FDA3"]
CONSTRAINED_DYNAMIC_NAMES = ["DTNK", "DTNK5", "DTNK6", "DTNK4"]
DYNAMIC_NAMES = BOX_DYNAMIC_NAMES + CONSTRAINED_DYNAMIC_NAMES

CONSTRAINED_ALGORITHMS = ["MGPSO", "MGPSO_feasarch", "CCMGPSO_filter", "CCMGPSO_strict"]


def mgpso_config(name="MGPSO", feasible_archive_only=False, refresh_mode="clean"):
    return {
        "class": MGPSO,
        "params": {
            "name": name,
            "swarm_size": SWARM_SIZE,
            "feasible_archive_only": feasible_archive_only,
            "refresh_mode": refresh_mode,
            "sentry_mode": SENTRY_MODE,
            **FIXED_PARAMS,
        },
    }


def ccmgpso_config(strategy, refresh_mode="clean", name=None):
    return {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": name or f"CCMGPSO_{strategy}",
            "objective_solver_class": MGPSO,
            "multiplier_solver_class": PSO,
            "objective_solver_params": {
                "swarm_size": SWARM_SIZE,
                "refresh_mode": refresh_mode,
                "sentry_mode": SENTRY_MODE,
                **FIXED_PARAMS,
            },
            "multiplier_solver_params": dict(MULTIPLIER_PARAMS),
            "multiplier_mode": "per_objective",
            "penalty_rho": 0.0,
            "max_multiplier": MAX_MULTIPLIER,
            "multiplier_immigrants": MULTIPLIER_IMMIGRANTS,
            "multiplier_steps": MULTIPLIER_STEPS,
            # Solutions already evaluated are not evaluated again when only
            # the multipliers change, so CCMGPSO and MGPSO spend the same
            # number of problem evaluations per iteration.
            "reuse_evaluations": True,
            "archive_strategy": strategy,
        },
    }


def legacy_ccpso_config(strategy, refresh_mode="clean", sentry_mode="fixed"):
    """The 22 Sept campaign's CCPSO: one shared penalty swarm, rho = 1.
    Not in any default campaign list."""
    return {
        "class": CoevolutionaryLagrangianSolver,
        "params": {
            "name": f"CCPSO_{strategy}",
            "objective_solver_class": MGPSO,
            "multiplier_solver_class": PSO,
            "objective_solver_params": {
                "swarm_size": SWARM_SIZE,
                "refresh_mode": refresh_mode,
                "sentry_mode": sentry_mode,
                **FIXED_PARAMS,
            },
            "multiplier_solver_params": dict(MULTIPLIER_PARAMS),
            "penalty_rho": 1.0,
            "max_multiplier": MAX_MULTIPLIER,
            "archive_strategy": strategy,
        },
    }


def constrained_configs(refresh_mode="clean", suffix=""):
    """The four algorithms run on every constrained problem."""
    return [
        mgpso_config("MGPSO" + suffix, refresh_mode=refresh_mode),
        mgpso_config("MGPSO_feasarch" + suffix, feasible_archive_only=True,
                     refresh_mode=refresh_mode),
        ccmgpso_config("filter", refresh_mode, name="CCMGPSO_filter" + suffix),
        ccmgpso_config("strict", refresh_mode, name="CCMGPSO_strict" + suffix),
    ]


def experiment_seed(problem_name, solver_name):
    """Deterministic per-experiment seed, stable across Python sessions."""
    key = f"{problem_name}:{solver_name}"
    return BASE_SEED + sum(ord(c) * (i + 1) for i, c in enumerate(key))
