### Testing Strategy

The `cilpy` library is developed with a strong emphasis on correctness and
reliability, which is maintained through a comprehensive suite of unit tests.
Our testing strategy ensures that every component of the library is
independently verifiable and that new contributions can be integrated with
confidence.

#### How Testing Works

Our testing is built on a few core principles and tools:

*   **Framework:** We use the `pytest` framework for writing and running tests.
    `pytest`'s test discovery mechanism automatically finds and runs all tests
    located in files named `test_*.py`.
*   **Isolation and Mocking:** To test a component's logic without interference
    from its dependencies, we heavily utilize mocking via Python's
    `unittest.mock` library. For example, when testing a `Solver`, the `Problem`
    it is trying to solve is replaced with a "mock" object. This allows us to
    control the exact fitness values returned for any given solution, enabling
    us to create predictable scenarios and verify that the solver's internal
    logic (e.g., selection, elitism) behaves as expected.
*   **Generic Design:** A key feature of the test suite is its generic and
    extensible nature. For components like `Problems` and `Constraint Handlers`,
    we use `pytest.mark.parametrize` to define a single set of tests that
    automatically runs against every new implementation. This means that when a
    developer adds a new benchmark function, they only need to add it to a list
    in the test file to have it fully validated against the library's interface
    contract.

#### Test Coverage

Our unit tests are organized to mirror the library's structure, providing
coverage for each of its core components:

1.  **Problems (`cilpy.problem`)**
    *   **Initialization:** Verifies that all problems are instantiated with the
        correct `dimension`, `bounds`, and `name`.
    *   **Evaluation:** Confirms that the `evaluate` method returns an
        `Evaluation` object with the correct structure and data types for
        fitness and constraints.
    *   **Dynamic Behavior:** For dynamic problems like the Moving Peaks
        Benchmark (MPB and CMPB), tests confirm that the landscape changes are
        triggered correctly based on the evaluation count.

2.  **Constraint Handling Mechanisms (`cilpy.solver.chm`)**
    *   **Initialization:** Ensures that handlers are created with valid
        parameters (e.g., `alpha` in `AlphaConstraintHandler` must be in the
        range).
    *   **Comparison Logic:** Rigorously tests the `is_better` method for all
        possible comparison scenarios. For example, the `AlphaConstraintHandler`
        tests cover cases where both solutions are feasible, only one is
        feasible, or their satisfaction levels are equal.
    *   **Internal Calculations:** Validates helper functions, such as the
        `_calculate_satisfaction` method in the `AlphaConstraintHandler`,
        against their formal definitions.

3.  **Solvers (`cilpy.solver`)**
    *   **Initialization:** Checks that solvers correctly set up their initial
        state and generate a valid initial population.
    *   **Algorithmic Operators:** Each core mechanism of a solver is tested in
        isolation using mocked problems. This includes:
        *   **GA:** Tournament selection, single-point crossover, Gaussian
            mutation, and elitism.
        *   **RIGA:** The replacement of the worst individuals with random
            immigrants.
        *   **HyperMGA:** The state-switching logic that toggles between normal
            and hyper-mutation modes based on environmental changes.
    *   **Result Retrieval:** Verifies that the `get_result` method correctly
        identifies and returns the best solution from the current population
        based on the active `comparator`.

To run the entire test suite, simply execute the following command from the root
directory of the project:
```bash
pytest
```
