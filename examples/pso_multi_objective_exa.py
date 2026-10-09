import random
import copy

import pandas as pd
from typing import List, Tuple
from matplotlib import pyplot as plt
import ast

from cilpy.problem import Problem, Evaluation
from cilpy.solver import Solver
from cilpy.runner import ExperimentRunner
from cilpy.problem.multi_objective import SCH1


class MOPSO(Solver[List[float], List[float]]):
    """
    A Multi-Objective Particle Swarm Optimization (MOPSO) solver.

    This implementation adapts the canonical PSO for multi-objective
    optimization.
    Key differences:
    1.  It maintains an 'archive' of non-dominated solutions instead of a single
        'gbest'.
    2.  Particles select their leader randomly from this archive to guide their
        search.
    3.  Dominance comparison is used to update personal bests and the archive.
    """

    def __init__(self,
                 problem: Problem[List[float], List[float]],
                 name: str,
                 swarm_size: int,
                 w: float,
                 c1: float,
                 c2: float,
                 **kwargs):
        super().__init__(problem, name)
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Initialize swarm positions and velocities
        self.positions = self._initialize_positions()
        self.velocities = self._initialize_velocities()

        # Evaluate initial swarm and set personal bests
        self.evaluations = [self.problem.evaluate(pos) for pos in self.positions]
        self.pbest_positions = copy.deepcopy(self.positions)
        self.pbest_evaluations = copy.deepcopy(self.evaluations)

        # Initialize the non-dominated archive
        self.archive: List[Tuple[List[float], Evaluation[List[float]]]] = []
        for i in range(self.swarm_size):
            self._update_archive(self.positions[i], self.evaluations[i])

    def _initialize_positions(self) -> List[List[float]]:
        """Creates the initial particle positions within bounds."""
        positions = []
        lower_bounds, upper_bounds = self.problem.bounds
        for _ in range(self.swarm_size):
            pos = [random.uniform(lower_bounds[d], upper_bounds[d])
                   for d in range(self.problem.dimension)]
            positions.append(pos)
        return positions

    def _initialize_velocities(self) -> List[List[float]]:
        """Creates the initial particle velocities."""
        lower_bounds, upper_bounds = self.problem.bounds
        return [
            [(random.uniform(-abs(upper_bounds[d] - lower_bounds[d]),
                             abs(upper_bounds[d] - lower_bounds[d])) * 0.1)
             for d in range(self.problem.dimension)]
            for _ in range(self.swarm_size)
        ]

    def _dominates(self, eval1: Evaluation, eval2: Evaluation) -> bool:
        """Checks if solution 1 dominates solution 2."""
        fit1 = eval1.fitness
        fit2 = eval2.fitness
        
        # A solution dominates if it is no worse in all objectives and
        # strictly better in at least one objective.
        not_worse = all(f1 <= f2 for f1, f2 in zip(fit1, fit2))
        is_better = any(f1 < f2 for f1, f2 in zip(fit1, fit2))
        return not_worse and is_better

    def _update_archive(self, solution: List[float], evaluation: Evaluation):
        """Updates the archive with a new solution."""
        # 1. Discard the new solution if it is dominated by any archive member
        if any(self._dominates(arc_eval, evaluation) for _, arc_eval in self.archive):
            return

        # 2. Remove any archive members that are now dominated by the new solution
        self.archive = [
            (arc_sol, arc_eval) for arc_sol, arc_eval in self.archive
            if not self._dominates(evaluation, arc_eval)
        ]

        # 3. Add the new solution to the archive
        self.archive.append((solution, evaluation))

    def step(self) -> None:
        """Performs one iteration of the MOPSO algorithm."""
        lower_bounds, upper_bounds = self.problem.bounds

        for i in range(self.swarm_size):
            # 1. Select a leader from the archive
            leader_solution, _ = random.choice(self.archive)

            # 2. Update Velocity
            for d in range(self.problem.dimension):
                r1, r2 = random.random(), random.random()
                cognitive = self.c1 * r1 * (self.pbest_positions[i][d] - self.positions[i][d])
                social = self.c2 * r2 * (leader_solution[d] - self.positions[i][d])
                inertia = self.w * self.velocities[i][d]
                self.velocities[i][d] = inertia + cognitive + social

            # 3. Update Position and clamp to bounds
            for d in range(self.problem.dimension):
                self.positions[i][d] += self.velocities[i][d]
                self.positions[i][d] = max(lower_bounds[d], min(self.positions[i][d], upper_bounds[d]))

            # 4. Evaluate new position
            self.evaluations[i] = self.problem.evaluate(self.positions[i])

            # 5. Update Personal Best (pbest)
            # A new position becomes pbest if it dominates the old one.
            if self._dominates(self.evaluations[i], self.pbest_evaluations[i]):
                self.pbest_positions[i] = copy.deepcopy(self.positions[i])
                self.pbest_evaluations[i] = copy.deepcopy(self.evaluations[i])

            # 6. Update the global archive with the new solution
            self._update_archive(self.positions[i], self.evaluations[i])

    def get_result(self) -> List[Tuple[List[float], Evaluation[List[float]]]]:
        """Returns the current archive of non-dominated solutions."""
        return self.archive


def run_example():
    # --- 1. Define the Experiment ---
    # The problem to solve
    problems_to_run = [
        SCH1(domain=(-100, 100))
    ]

    # The MOPSO solver configuration
    solver_configs = [
        {
            "class": MOPSO,
            "params": {
                "name": "MOPSO",
                "swarm_size": 20,
                "w": 0.4,
                "c1": 1.0,
                "c2": 1.0,
            }
        },
    ]

    # --- 2. Run the Experiment ---
    number_of_runs = 1
    max_iter = 100

    runner = ExperimentRunner(
        problems=problems_to_run,
        solver_configurations=solver_configs,
        num_runs=number_of_runs,
        max_iterations=max_iter
    )
    # The runner automatically saves results to a CSV file.
    runner.run_experiments()

def visualize_results():
    """
    Reads a multi-objective experiment output CSV, extracts the Pareto front 
    from the final iteration, and generates a 2D scatter plot.
    """
    csv_filepath = 'out/SCH1_MOPSO.out.csv'

    # 1. Read csv file
    print(f"Attempting to read data from: {csv_filepath}")
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    if df.empty:
        print("Error: The CSV file is empty.")
        return

    # 2. Get the result string from the very last row
    try:
        final_result_string = df.iloc[-1]['accuracy']
    except (IndexError, KeyError):
        print("Error: The CSV file seems to be missing rows or the 'accuracy' column.")
        return

    # 3. Safely parse the string into a list of lists.
    try:
        # This will convert "[[0.1, 0.2], [0.3, 0.4]]" into [[0.1, 0.2], [0.3, 0.4]]
        fitness_values = ast.literal_eval(final_result_string)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing the 'accuracy' column string: {e}")
        print("Please ensure the data is formatted as a list of lists.")
        return

    # 4. Extract the multi-objective fitness values.
    # The data is already in the correct format, so we can use it directly.
    if not isinstance(fitness_values, list) or not fitness_values:
        print("Error: Parsed data is not a valid list of results.")
        return

    try:
        # Unzip the fitness values into separate lists for plotting
        # f[0] is the first objective, f[1] is the second.
        f1_values = [f[0] for f in fitness_values]
        f2_values = [f[1] for f in fitness_values]
    except (IndexError, TypeError) as e:
        print(f"Error extracting fitness data from the parsed results: {e}")
        print("Each item in the list should be a sublist with at least two numbers, e.g., [f1, f2].")
        return

    print(f"Successfully extracted {len(f1_values)} points from the final Pareto front.")

    # 5. Create the scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(f1_values, f2_values, c='blue', marker='o', label='Discovered Pareto Front')
    plt.title('Final Pareto Front for SCH1 from MOPSO', fontsize=16)
    plt.xlabel('Objective 1', fontsize=12)
    plt.ylabel('Objective 2', fontsize=12)
    plt.grid(True)
    plt.legend()

    # Display the plot
    plt.show()

if __name__ == '__main__':
    run_example()
    visualize_results()
