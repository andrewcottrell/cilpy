# Quickstart Guide

This guide shows you how to implement `cilpy`'s core interfaces and run a basic
experiment.

## The Core Interfaces

To use `cilpy`, you need to understand two key interfaces:

1. **`cilpy.problem.Problem`**: Defines an optimization problem. It requires the
   following methods:
    - `__init__(self, dimension, bounds, name)`: Initializes a Problem instance.
    - `evaluate(self, solution)`: Evaluates a candidate solution.
    - `is_dynamic(self)`: Indicates if the problem changes over time.
1. **`cilpy.solver.Solver`**: Defines an optimization algorithm. It requires the
   following methods:
    - `__init__(self, problem, name, constraint_handler, **kwargs)`:
      Initializes the solver.
    - `step(self)`: Performs one iteration of the optimization algorithm.
    - `get_result(self)`: Returns the best solution(s) found so far.

## Writing your own experiment
### Step 1: Implement a `Problem`

Create a file named `my_problem.py`. Here, we'll implement the 2D Sphere
function.

```python
# my_problem.py
from typing import List, Tuple
from cilpy.problem import Problem, Evaluation

class MySphere(Problem[List[float], float]):
    """A custom implementation of the Sphere function."""
    def __init__(self, dimension: int):
        super().__init__(
            dimension=dimension,
            bounds=([-5.12] * dimension, [5.12] * dimension),
            name="MySphere"
        )

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        """Calculates the sum of squares of the solution's elements."""
        fitness = sum(x**2 for x in solution)
        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        """This problem is not dynamic."""
        return (False, False)
```

### Step 2: Implement a `Solver`

Create a file named `my_solver.py`. Here, we'll implement a simple Random Search
algorithm.

```python
# my_solver.py
import random
from typing import List, Tuple
from cilpy.problem import Problem, Evaluation
from cilpy.solver import Solver

class RandomSearch(Solver[List[float], float]):
    """A simple solver that generates random solutions."""
    def __init__(self, problem: Problem[List[float], float], name: str):
        super().__init__(problem, name)
        self.best_solution = None
        self.best_evaluation = Evaluation(fitness=float('inf'))

    def step(self) -> None:
        """Generate one new random solution and update the best."""
        # Create a new random solution within the problem's bounds
        lower, upper = self.problem.bounds
        new_solution = [random.uniform(lower[i], upper[i])
                        for i in range(self.problem.dimension)]

        # Evaluate the new solution
        new_evaluation = self.problem.evaluate(new_solution)

        # If it's better than the current best, update
        if new_evaluation.fitness < self.best_evaluation.fitness:
            self.best_solution = new_solution
            self.best_evaluation = new_evaluation

    def get_result(self) -> List[Tuple[List[float], Evaluation[float]]]:
        """Return the best solution found so far."""
        return [(self.best_solution, self.best_evaluation)]
```

### Step 3: Run the Experiment

Now, use the `ExperimentRunner` to run your new solver on your new problem.
Create `run_my_experiment.py`.

```python
# run_my_experiment.py
from cilpy.runner import ExperimentRunner
from my_problem import MySphere
from my_solver import RandomSearch

# 1. Define the Problem
problems_to_run = [
    MySphere(dimension=2)
]

# 2. Configure the Solver
solver_configs = [
    {
        "class": RandomSearch,
        "params": {"name": "MyRandomSearch"}
    }
]

# 3. Set Experiment Parameters
number_of_runs = 5
max_iter = 100

# 4. Create and run the experiment
runner = ExperimentRunner(
    problems=problems_to_run,
    solver_configurations=solver_configs,
    num_runs=number_of_runs,
    max_iterations=max_iter
)
runner.run_experiments()
```

Run this script from your terminal:
```bash
python run_my_experiment.py
```

This will create a `MySphere_MyRandomSearch.out.csv` file with the experiment's
results.

## Using Included Components

You don't always need to create new components. `cilpy` includes standard
problems and solvers. Here is how you would run the included `GA` solver on the
included `Ackley` problem.

```python
# run_included_experiment.py
from cilpy.runner import ExperimentRunner
from cilpy.problem.functions import Ackley  # Included problem
from cilpy.solver.ga import GA              # Included solver

runner = ExperimentRunner(
    problems=[Ackley(dimension=10)],
    solver_configurations=[
        {
            "class": GA,
            "params": {
                "name": "GA_Standard",
                "population_size": 50,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1,
            }
        }
    ],
    num_runs=10,
    max_iterations=1000
)
runner.run_experiments()
```
