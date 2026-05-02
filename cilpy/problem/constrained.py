# cilpy/problem/constrained.py
"""
Constrained Benchmark Optimization Problems.

This module provides implementations of common benchmark functions
for single-objective, constrained optimization, adhering to the `Problem`
interface.
"""
from typing import List, Tuple

import numpy as np

from numpy import inf

from cilpy.problem import Problem, Evaluation

"""

The following are implemented from CEC2006

"""
class G01(Problem[List[float], float]):
    """g01 from the CEC 2006 benchmark suite.

    This is a 13-dimensional minimization problem with nine linear inequality
    constraints.

    The objective function is:
    f(x) = sum_{j=1 to 4}(5*x_j - 5*x_j^2) - sum_{j=5 to 13}(x_j)

    Subject to 9 linear inequality constraints.

    The known global optimum is at
    x* = (1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1) with f(x*) = -15.
    """

    def __init__(self):
        """Initializes a G01 instance."""
        # Bounds: x_j in [0,1] for j=1..9
        #         x_j in [0,100] for j=10..12
        #         x_13 in [0,1]
        lower_bounds = [0.0] * 13
        upper_bounds = [1.0] * 9 + [100.0] * 3 + [1.0]
        super().__init__(dimension=13, bounds=(lower_bounds, upper_bounds), name="G01")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates the function for a given solution.

        Args:
            solution: A list of 13 floats.

        Returns:
            An Evaluation object containing the fitness and constraint
            violations.
        """
        x = [s for s in solution]  # Use a copy

        # Objective function
        sum1 = sum(5 * x[j] for j in range(4))
        sum2 = sum(5 * x[j] ** 2 for j in range(4))
        sum3 = sum(x[j] for j in range(4, 13))
        fitness = sum1 - sum2 - sum3

        # Inequality constraints (g(x) <= 0)
        constraints = [
            2 * x[0] + 2 * x[1] + x[9] + x[10] - 10,
            2 * x[0] + 2 * x[2] + x[9] + x[11] - 10,
            2 * x[1] + 2 * x[2] + x[10] + x[11] - 10,
            -8 * x[0] + x[9],
            -8 * x[1] + x[10],
            -8 * x[2] + x[11],
            -2 * x[3] - x[4] + x[9],
            -2 * x[5] - x[6] + x[10],
            -2 * x[7] - x[8] + x[11],
        ]

        return Evaluation(fitness=fitness, constraints_inequality=constraints)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that this function is not dynamic."""
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False
    
    def get_fitness_bounds(self):
        return (-15.0, 0.0)

class G02(Problem[List[float], float]):
    """
    g02 from CEC2006. 
    Global min at  x* = (3.16246061572185, 3.12833142812967, 3.09479212988791, 
    3.06145059523469, 3.02792915885555, 2.99382606701730, 2.95866871765285, 
    2.92184227312450, 0.49482511456933, 0.48835711005490, 0.48231642711865, 
    0.47664475092742, 0.47129550835493, 0.46623099264167, 0.46142004984199, 
    0.45683664767217, 0.45245876903267, 0.44826762241853, 0.44424700958760, 
    0.44038285956317)
    
    with f(x*) = −0.80361910412559 as optimal value (as far as we know)
    
    20 dimensions. 2 constraints.
    Bounds are (0,10].
    """
    def __init__(self):
        """Initializes a G02 instance."""
        # n = 20
        # Bounds: x_j (0,10] for j = 1,...,n
        lower_bounds = [1e-6] * 20
        upper_bounds = [10.0] * 20
        super().__init__(dimension=20, bounds=(lower_bounds, upper_bounds), name="G02")
        
        
    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        x = np.array(solution)
        # Objective function
        n = 20
        i = np.arange(1, n + 1)
        numerator = np.sum(np.cos(x)**4) - 2 * np.prod(np.cos(x)**2)
        denominator = np.sqrt(np.sum(i * x**2))
        fitness = - np.abs(numerator / denominator)
        
        # Constraints (g(x) <= 0)
        constraints = [
            0.75 - np.prod(x),
            np.sum(x) - 7.5 * n
        ]
        
        return Evaluation(fitness=fitness, constraints_inequality=constraints)

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)
        
    def is_multi_objective(self) -> bool:
        return False

    def get_fitness_bounds(self):
        return (-0.803619, 0.0)
    
class G03(Problem[List[float], float]):
    """g03 from the CEC 2006 benchmark suite.

    This is a 10-dimensional minimization problem with one equality constraint.

    The objective function is:
    f(x) = -(sqrt(n))^n * prod_{i=1}^{n}(x_i)

    Subject to one equality constraint:
    h1(x) = sum_{i=1}^{n}(x_i^2) - 1 = 0

    Bounds: x_i in [0, 1] for i = 1, ..., n.

    The known global optimum is at
    x* = (0.31624357647283069, ...) with f(x*) = -1.00050010001000.
    """

    def __init__(self):
        """Initializes a G03 instance."""
        # n = 10
        # Bounds: x_i in [0, 1] for i = 1,...,n
        lower_bounds = [0.0] * 10
        upper_bounds = [1.0] * 10
        super().__init__(dimension=10, bounds=(lower_bounds, upper_bounds), name="G03")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        x = np.array(solution)
        n = 10

        # Objective function: -(sqrt(n))^n * prod(x_i)
        fitness = -(np.sqrt(n) ** n) * np.prod(x)

        # Equality constraint: |h(x)| - epsilon <= 0
        # h1(x) = sum(x_i^2) - 1 = 0
        constraints = [
            np.abs(np.sum(x ** 2) - 1) - 1e-4
        ]

        return Evaluation(fitness=fitness, constraints_inequality=constraints)

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False
    
    def get_fitness_bounds(self):
        return (-1.0, 0.0)

class G04(Problem[List[float], float]):
    """g04 from the CEC 2006 benchmark suite.

    This is a 5-dimensional minimization problem with six inequality constraints.

    The objective function is:
    f(x) = 5.3578547*x3^2 + 0.8356891*x1*x5 + 37.293239*x1 - 40792.141

    Subject to 6 inequality constraints.

    Bounds: 78 <= x1 <= 102, 33 <= x2 <= 45, 27 <= x_i <= 45 (i = 3, 4, 5).

    The known global optimum is at
    x* = (78, 33, 29.9952560256815985, 45, 36.7758129057882073)
    with f(x*) = -30665.5386717834.
    """

    def __init__(self):
        """Initializes a G04 instance."""
        lower_bounds = [78.0, 33.0, 27.0, 27.0, 27.0]
        upper_bounds = [102.0, 45.0, 45.0, 45.0, 45.0]
        super().__init__(dimension=5, bounds=(lower_bounds, upper_bounds), name="G04")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        x = np.array(solution)

        # Objective function
        fitness = (5.3578547 * x[2] ** 2
                   + 0.8356891 * x[0] * x[4]
                   + 37.293239 * x[0]
                   - 40792.141)

        # Inequality constraints (g(x) <= 0)
        constraints = [
            85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4] - 92,
            -85.334407 - 0.0056858 * x[1] * x[4] - 0.0006262 * x[0] * x[3] + 0.0022053 * x[2] * x[4],
            80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2 - 110,
            -80.51249 - 0.0071317 * x[1] * x[4] - 0.0029955 * x[0] * x[1] - 0.0021813 * x[2] ** 2 + 90,
            9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3] - 25,
            -9.300961 - 0.0047026 * x[2] * x[4] - 0.0012547 * x[0] * x[2] - 0.0019085 * x[2] * x[3] + 20,
        ]

        return Evaluation(fitness=fitness, constraints_inequality=constraints)

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False
    
    def get_fitness_bounds(self):
        return (-30665.539, -20000.0)


class G05(Problem[List[float], float]):
    """g05 from the CEC 2006 benchmark suite.

    This is a 4-dimensional minimization problem with 2 inequality and
    3 equality constraints.

    The objective function is:
    f(x) = 3*x1 + 0.000001*x1^3 + 2*x2 + (0.000002/3)*x2^3

    Subject to 2 inequality and 3 equality constraints.

    Bounds: 0 <= x1 <= 1200, 0 <= x2 <= 1200,
            -0.55 <= x3 <= 0.55, -0.55 <= x4 <= 0.55.

    The known global optimum is at
    x* = (679.945148297028709, 1026.06697600004691,
          0.118876369094410433, -0.39623348521517826)
    with f(x*) = 5126.4967140071.
    """

    def __init__(self):
        """Initializes a G05 instance."""
        lower_bounds = [0.0,   0.0,   -0.55, -0.55]
        upper_bounds = [1200.0, 1200.0,  0.55,  0.55]
        super().__init__(dimension=4, bounds=(lower_bounds, upper_bounds), name="G05")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        x = np.array(solution)

        # Objective function
        fitness = (3 * x[0]
                   + 0.000001 * x[0] ** 3
                   + 2 * x[1]
                   + (0.000002 / 3) * x[1] ** 3)

        # Inequality constraints (g(x) <= 0)
        constraints_ineq = [
            -x[3] + x[2] - 0.55,
            -x[2] + x[3] - 0.55,
        ]

        # Equality constraints converted to inequalities: |h(x)| - epsilon <= 0
        h3 = 1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0]
        h4 = 1000 * np.sin(x[2] - 0.25) + 1000 * np.sin(x[2] - x[3] - 0.25) + 894.8 - x[1]
        h5 = 1000 * np.sin(x[3] - 0.25) + 1000 * np.sin(x[3] - x[2] - 0.25) + 1294.8

        constraints_eq = [
            np.abs(h3) - 1e-4,
            np.abs(h4) - 1e-4,
            np.abs(h5) - 1e-4,
        ]

        return Evaluation(
            fitness=fitness,
            constraints_inequality=constraints_ineq + constraints_eq
        )

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False
    
    def get_fitness_bounds(self):
        return (5126.498, 10000.0)
    
class G06(Problem[List[float], float]):
    """g06 from the CEC 2006 benchmark suite.

    This is a 2-dimensional minimization problem with two inequality constraints.

    The objective function is:
    f(x) = (x1 - 10)^3 + (x2 - 20)^3

    Subject to 2 inequality constraints.

    Bounds: 13 <= x1 <= 100, 0 <= x2 <= 100.

    The known global optimum is at
    x* = (14.095, 0.84296) with f(x*) = -6961.81387558015.
    Both constraints are active.
    """

    def __init__(self):
        """Initializes a G06 instance."""
        lower_bounds = [13.0, 0.0]
        upper_bounds = [100.0, 100.0]
        super().__init__(dimension=2, bounds=(lower_bounds, upper_bounds), name="G06")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        x = np.array(solution)

        # Objective function
        fitness = (x[0] - 10) ** 3 + (x[1] - 20) ** 3

        # Inequality constraints (g(x) <= 0)
        constraints = [
            -(x[0] - 5) ** 2 - (x[1] - 5) ** 2 + 100,
            (x[0] - 6) ** 2 + (x[1] - 5) ** 2 - 82.81,
        ]

        return Evaluation(fitness=fitness, constraints_inequality=constraints)

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False

    def get_fitness_bounds(self):
        return (-6961.814, 0.0)
# ==============================================================================
#  These problems are drawn from Appendix A.6 of Computational Intelligence: An
#  Introduction (second edition) by Andries P. Engelbrecht
# ==============================================================================


class C01(Problem[List[float], float]):
    """The global optimum is x = (0.5, 0.25), with f(x) = 0.25."""

    def __init__(self):
        """Initializes a Problem instance."""
        lower_bounds = [-0.5, -4]
        upper_bounds = [0.5, 1.0]
        super().__init__(dimension=2, bounds=(lower_bounds, upper_bounds), name="C01")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates a candidate solution."""
        x = [s for s in solution]  # Use a copy

        # Objective function
        fitness = 100 * (x[1] - x[0]) ** 2 + (1 - x[0]) ** 2

        # Inequality constraints (g(x) <= 0)
        constraints = [
            -x[0] - x[1] ** 2,
            -x[0] ** 2 - x[1],
        ]

        return Evaluation(fitness=fitness, constraints_inequality=constraints)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that this function is not dynamic."""
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False


class C02(Problem[List[float], float]):
    """The global optimum is x = (1, 1), with f(x) = 1."""

    def __init__(self):
        """Initializes a Problem instance."""
        lower_bounds = [-2.0, -2.0]
        upper_bounds = [2.0, 2.0]
        super().__init__(dimension=2, bounds=(lower_bounds, upper_bounds), name="C02")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Evaluates a candidate solution."""
        x = [s for s in solution]  # Use a copy

        # Objective function
        fitness = (x[0] - 2) ** 2 - (x[1] - 1) ** 2

        # Inequality constraints (g(x) <= 0)
        constraints = [
            -x[0] ** 2 - x[1],
            -x[0] - x[1] - 2,
        ]

        return Evaluation(fitness=fitness, constraints_inequality=constraints)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Indicates that this function is not dynamic."""
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False
