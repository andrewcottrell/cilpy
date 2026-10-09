# test/problem/test_constrained.py
"""
Unit tests for single-objective, constrained optimization problems.

This test suite is designed to be generic. To test a new constrained problem,
simply add its class to the `CONSTRAINED_PROBLEMS` list.
The `pytest.mark.parametrize` decorator will automatically run all defined
tests for the new problem, ensuring it adheres to the expected interface
and behavior.
"""
import pytest
from typing import List, Type

# Assuming the project is installed in editable mode or path is configured
from cilpy.problem import Problem, Evaluation
from cilpy.problem.constrained import G01, G02, G03, G04, G05, G06

# --- Test Configuration ---
# Add any new constrained problem class here to include it in the tests.
CONSTRAINED_PROBLEMS: List[Type[Problem]] = [
    G01, G02, G03, G04, G05, G06
]

# --- Test Cases ---

@pytest.mark.parametrize("problem_class", CONSTRAINED_PROBLEMS)
def test_constrained_problem_initialization(problem_class: Type[Problem]):
    """
    Tests if a constrained problem can be initialized correctly.

    Checks for:
    - Correct setting of dimension.
    - Correct structure and length of bounds.
    - Presence of a non-empty name attribute.
    """
    problem = problem_class() # type: ignore

    # Assert that basic properties are set correctly
    assert problem.dimension != None
    assert isinstance(problem.name, str) and problem.name

    # Assert that bounds are structured correctly
    lower_bounds, upper_bounds = problem.bounds
    assert isinstance(lower_bounds, list)
    assert isinstance(upper_bounds, list)
    assert len(lower_bounds) == problem.dimension
    assert len(upper_bounds) == problem.dimension


@pytest.mark.parametrize("problem_class", CONSTRAINED_PROBLEMS)
def test_constrained_problem_evaluate(problem_class: Type[Problem]):
    """
    Tests the `evaluate` method of an unconstrained problem.

    Ensures that for a valid input solution, the method returns an
    `Evaluation` object with the expected structure:
    - A float fitness value.
    - `None` for both inequality and equality constraints.
    """
    problem = problem_class() # type: ignore

    # Create a valid sample solution within the problem's bounds
    # (using the middle of the domain as a generic, safe point)
    lower_bound, upper_bound = problem.bounds[0][0], problem.bounds[1][0]
    mid_point = lower_bound + (upper_bound - lower_bound) / 2
    solution = [mid_point] * problem.dimension

    # Perform the evaluation
    evaluation_result = problem.evaluate(solution)

    # Assert that the return type is correct
    assert isinstance(evaluation_result, Evaluation)

    # Assert that the structure of the Evaluation object is correct for
    # a constrained problem
    assert isinstance(evaluation_result.fitness, float)
    assert (evaluation_result.constraints_inequality is not None) or (evaluation_result.constraints_equality is not None)


@pytest.mark.parametrize("problem_class", CONSTRAINED_PROBLEMS)
def test_constrained_problem_is_dynamic(problem_class: Type[Problem]):
    """
    Tests the `is_dynamic` method.

    For all benchmark functions currently implemented, the landscape is static.
    This test verifies that the method correctly returns (False, False).
    """
    problem = problem_class() # type: ignore
    dynamic_status = problem.is_dynamic()

    # Assert that the return value is the expected tuple for static problems
    assert dynamic_status == (False, False)
