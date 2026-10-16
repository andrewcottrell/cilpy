# Developer Guide

Welcome to the `cilpy` developer guide! We're excited that you're interested in
contributing. This document provides everything you need to know to get your
development environment set up and contribute to the project effectively.

Our goal is to create a library that is powerful yet easy to maintain and extend.
Following these guidelines helps us achieve that.

## 1. Setting Up Your Development Environment

To contribute code, you first need to set up a local development environment.

**1. Fork and Clone the Repository**:  
First, [fork the repository](https://git.cs.sun.ac.za/help/user/project/repository/forking_workflow.md)
on the GitLab instance. Then, clone your fork to your local machine:  
```bash
git clone git@git.cs.sun.ac.za:Computer-Science/rw771/2025/24717274-AE4-src.git  
cd 24717274-AE4-src
```

**2. Install in Editable Mode**:  
Installing the package in "editable" mode allows you to test your changes
live without having to reinstall the package after every modification.
```bash
pip install -e .
```

## 2. Core Design Philosophy

`cilpy` is built on three core principles. When developing new features, please
keep these in mind:

*   **Genericity**: The library's interfaces should be abstract enough to
    support a wide variety of CI paradigms, from single-objective GAs to multi-
    population co-evolutionary systems. Avoid making assumptions that would
    limit the framework to a specific type of problem or solver.
*   **Extendability**: The primary goal is to make it easy for researchers to
    add their own components. This is achieved through the core `Problem` and
    `Solver` abstract base classes. A user should be able to add a new algorithm
    and immediately test it on all existing problems, and vice-versa.
*   **Maintainability**: Clean, readable, and well-documented code is essential.
    We enforce this through mandatory type hinting, adherence to style guides,
    and a requirement for unit tests.

## 3. The Contribution Workflow

We follow a standard fork-and-pull-request workflow.

1.  **Find an Issue**: Start by looking at the [To-Do List](todo.md) or the
    issue tracker for tasks. If you have a new idea, consider creating an issue
    first to discuss it with the maintainers.
2.  **Create a Branch**: From the `main` branch, create a new feature branch for
    your work. Please use the branch prefixes outlined below.
    ```bash
    # Example for a new feature
    git checkout -b ft/add-new-pso-variant main
    ```
3.  **Write Code**: Make your changes, following the coding style guidelines
    below.
4.  **Write Tests**: All new features must be accompanied by tests. If you add a
    new problem, add a test case for it in `test/test_problems.py`. If you add a
    new solver, create a test for it. Tests are crucial for ensuring the long-
    term stability of the library. Learn more about our testing strategy
    [here](testing.md)
5.  **Ensure All Tests Pass**: Run the test suite to make sure your changes
    haven't broken existing functionality.
6.  **Submit a Merge Request**: Push your branch to your fork and open a Merge
    Request to the main `cilpy` repository. Provide a clear description of the
    changes you have made.

### Branch Naming Prefixes
 * `ft/`: Feature development (e.g., `ft/quantum-pso`)
 * `fx/`: Bug fixes (e.g., `fx/fix-ga-selection-bug`)
 * `ch/`: Maintenance tasks (e.g., `ch/update-dependencies`)
 * `rf/`: Code refactoring (e.g., `rf/optimize-runner-loop`)
 * `dc/`: Documentation updates (e.g., `dc/clarify-solver-api`)
 * `ts/`: Testing-related work (e.g., `ts/add-tests-for-mpb`)

## 4. How to Add New Components

This is the most common way to contribute. The key is to correctly implement the
required interface.

### Adding a New Problem
1.  **Create the File**: Add a new Python file in the `cilpy/problem/`
    directory.
2.  **Implement the Interface**: Your new class must inherit from
    `cilpy.problem.Problem` and implement all of its abstract methods.
3.  **Write a Test**: Add a test case to `test/test_problems.py` that
    instantiates your problem and evaluates a known solution to ensure the
    fitness calculation is correct.

**Template for a new problem:**
```python
# In cilpy/problem/my_new_problem.py
from typing import List, Tuple
from cilpy.problem import Problem, Evaluation

class MyNewProblem(Problem[List[float], float]):
    def __init__(self, dimension: int):
        # Always call super().__init__()
        super().__init__(
            dimension=dimension,
            bounds=([-10.0] * dimension, [10.0] * dimension),
            name="MyNewProblem"
        )

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        # Your fitness logic here
        fitness = ...
        return Evaluation(fitness=fitness)

    def is_dynamic(self) -> Tuple[bool, bool]:
        # Return True for the first element if the objective is dynamic
        return (False, False)
```

### Adding a New Solver
1.  **Create the File**: Add a new Python file in the `cilpy/solver/solvers/`
    directory.
2.  **Implement the Interface**: Your new class must inherit from
    `cilpy.solver.Solver` and implement all of its abstract methods.
3.  **Write a Test**: Add a test that runs your solver on a simple, known
    problem (like Sphere) and asserts that it finds a reasonably good solution.

**Template for a new solver:**
```python
# In cilpy/solver/solvers/my_new_solver.py
from typing import List, Tuple
from cilpy.problem import Problem, Evaluation
from cilpy.solver import Solver

class MyNewSolver(Solver[List[float], float]):
    def __init__(self, problem: Problem, name: str, **kwargs):
        # Always call super().__init__()
        super().__init__(problem, name)
        # Your initialization logic here (e.g., population)
        self.best_solution = None
        self.best_evaluation = Evaluation(fitness=float('inf'))

    def step(self) -> None:
        # Your core algorithm logic for one iteration
        # ... update self.best_solution and self.best_evaluation
        pass

    def get_result(self) -> List[Tuple[List[float], Evaluation[float]]]:
        # Return the best result found so far
        return [(self.best_solution, self.best_evaluation)]
```

## 5. Coding Style and Conventions

Consistency is key. Please adhere to the following standards.

*   **Docstrings**: All public modules, classes, and functions must have
    [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
*   **Style Guide**: All code must adhere to [PEP 8](https://peps.python.org/pep-0008/).
    We recommend using an autoformatter like `black`.
*   **Line Length**: Maximum 80 characters for code and docstrings.
*   **Type Hinting**: All function and method signatures **must** include type
    hints.
*   **Imports**: Group imports in this order: (1) Python standard library, (2)
    third-party libraries, (3) `cilpy` imports.
*   **Dependencies**: The core library should have minimal dependencies. Please
    discuss with a maintainer before adding a new third-party dependency.

## Where to Start?

A great place to start contributing is by looking at our [To-Do List](todo.md)
or picking up an unassigned issue from the issue tracker. We look forward to
your contributions!
