# test/solver/chm/test_deb_feasibility.py
"""
Unit tests for the DebFeasibilityHandler.

The handler applies Deb's feasibility rules when comparing two solutions:
1.  Two feasible solutions are compared on fitness.
2.  A feasible solution beats an infeasible one, whatever their fitness.
3.  Two infeasible solutions are compared on total constraint violation.
"""
import pytest

from cilpy.problem import Evaluation
from cilpy.solver.chm.deb_feasibility import DebFeasibilityHandler


@pytest.fixture
def handler() -> DebFeasibilityHandler:
    """Provides a fresh handler for each test."""
    return DebFeasibilityHandler()


def _ev(fitness, ineq=None, eq=None) -> Evaluation:
    return Evaluation(
        fitness=fitness, constraints_inequality=ineq, constraints_equality=eq
    )


# --- Feasibility and violation helpers ---

@pytest.mark.parametrize("ineq, eq, expected", [
    (None, None, True),            # unconstrained counts as feasible
    ([-1.0, 0.0], None, True),     # g(x) <= 0 is satisfied on the boundary
    ([0.1], None, False),
    (None, [0.0005], True),        # |h(x)| within the 1e-3 tolerance
    (None, [0.01], False),
    ([-1.0], [0.01], False),       # one violated constraint is enough
])
def test_is_feasible(handler, ineq, eq, expected):
    """Tests feasibility for inequality and equality constraints."""
    assert handler._is_feasible(_ev(1.0, ineq, eq)) is expected


def test_total_violation_sums_only_violated_amounts(handler):
    """Satisfied inequalities contribute nothing; equalities use |h|."""
    evaluation = _ev(1.0, ineq=[2.0, -5.0, 0.5], eq=[-0.25])
    assert handler._total_violation(evaluation) == pytest.approx(2.75)


# --- Comparison logic (`is_better`) ---

def test_both_feasible_compares_fitness(handler):
    """Rule 1: the lower fitness wins when both are feasible."""
    better, worse = _ev(1.0, ineq=[-1.0]), _ev(2.0, ineq=[-1.0])
    assert handler.is_better(better, worse)
    assert not handler.is_better(worse, better)


def test_feasible_beats_infeasible_regardless_of_fitness(handler):
    """Rule 2: feasibility takes priority over fitness."""
    feasible, infeasible = _ev(100.0, ineq=[-1.0]), _ev(-100.0, ineq=[1.0])
    assert handler.is_better(feasible, infeasible)
    assert not handler.is_better(infeasible, feasible)


def test_both_infeasible_compares_violation(handler):
    """Rule 3: the smaller total violation wins, whatever the fitness."""
    less_violating = _ev(100.0, ineq=[0.5])
    more_violating = _ev(-100.0, ineq=[3.0])
    assert handler.is_better(less_violating, more_violating)
    assert not handler.is_better(more_violating, less_violating)


def test_equal_solutions_are_not_better(handler):
    """A solution is never strictly better than an identical one."""
    for evaluation in (_ev(1.0, ineq=[-1.0]), _ev(1.0, ineq=[1.0])):
        assert not handler.is_better(evaluation, evaluation)
