# Welcome to `cilpy`

`cilpy` (which stands for "computational intelligence library for Python") is an
extensible Python library for computational intelligence, designed to streamline
research and experimentation with nature-inspired algorithms (NIAs).

The primary goal of `cilpy` is to provide a unified and robust framework for
tackling a wide range of optimization tasks, including:

* Single- and multi-objective optimization
* Constrained and unconstrained problems
* Static and dynamic objectives and constraints

`cilpy` strives to be a useful tool for researchers developing new algorithms,
and students learning about computational intelligence. `cilpy` offers tools to
design, execute, and analyze experiments with ease and rigor.

## The `cilpy` Philosophy: Modularity and Extensibility

The ecosystem for nature-inspired optimization algorithms is fragmented.
Researchers frequently have to build their experimental setups from scratch or
adapt to libraries that are difficult to extend, lack comprehensive features, or
are no longer maintained.

`cilpy` is designed to solve this problem by separating concerns. The library is
divided into four distinct, interoperable components:

1. **`cilpy.problem`**: This component is used to specify optimization problems,
   including objective function(s), search space, and constraints.
2. **`cilpy.solver`**: This component is used to specify solvers (NIAs), that
   search for solutions to a given problem.
3. **`cilpy.compare`**: This component will provide the tools for statistical
   analysis and visualization to compare the performance of different solvers on
   various problems.
4. **`cilpy.runner`**: This component is used to orchestrate the interaction
   between components, run experiments, and log results.

By enforcing a clean separation between these parts through well-defined
interfaces, `cilpy` allows users to seamlessly swap out components.

* **Want to test a new algorithm?** Implement the `Solver` interface, and you
    can immediately benchmark it against all existing problems.
* **Have a new benchmark problem?** Implement the `Problem` interface, and any
  existing solver can be used to tackle it.

This modularity drastically reduces boilerplate code and allows users to focus
on the novel aspects of their research.

## How `cilpy` Works: The Experiment Runner

At the core of `cilpy`'s workflow is the `ExperimentRunner` in `cilpy.runner`.
It orchestrates the entire experimental process. After being provided a list of
problems to solve, solvers and their configurations to test, and comparisons to
perform on the algorithm, the runner orchestrates all further action, and
performs the experiment.

The typical workflow looks like this:

1. **Define Problems**: Instantiate or create custom classes for the
   optimization problems you want to investigate.
2. **Configure Solvers**: Create a list of dictionaries, where each dictionary
   specifies a solver's class (e.g., `GA`, `PSO`) and its parameters (e.g.,
   `population_size`, `mutation_rate`).
3. **Specify Comparisons**: TODO
4. **Configure the Runner**: Initialize the `ExperimentRunner` with your
   problems, solver configurations, comparisons to perform, and experiment
   parameters (like the number of runs and iterations).
5. **Execute**: Call the `run_experiments()` method. The runner will
   systematically pair each solver with each problem, execute the specified
   number of independent runs, and save the results for later analysis.

This declarative approach makes experiments easy to define, reproduce, and
modify.

## How to Use This Documentation

This documentation is structured to guide you, whether you are using the library
for the first time or developing new components for it.

* **[Quickstart](quickstart.md)**: A quick guide on how to implement the core
  interfaces and run an experiment.
* **[Included](lib/index.md)**: Discusses the problems, solvers, and analysis
  tools that have been incorporated into `cilpy`.
* **[API Reference](api/index.md)**: A detailed, technical reference for all the
  core classes and functions in the library.
* **[Developer Guide](dev/index.md)**: For those who want to contribute to
  `cilpy`, this contains information on our coding standards, testing
  procedures, and how to get involved.
