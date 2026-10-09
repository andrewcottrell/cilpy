Extend the library so that the `compare` component has all the features of this
[stats tool](https://github.com/yesteryearer/Automated-Analysis-of-Metaheuristics).

Write unit tests.

Write integration tests.

Consider how the library could interact with libraries used for:

- [ ] empirical analysis
- [ ] fitness landscape analysis
- [ ] results repositories

Ensure the library enables the following constraint handling techniques:

- [ ] techniques ensuring feasibility of solutions throughout the search
  process.
- [ ] techniques allowing infeasible solutions during the search process, while
  applying repair mechanisms later.
- [X] techniques which formulate the constrained optimization problem as a
  box-constrained optimization problem through the use of penalty methods.
- [X] techniques which formulate the constrained optimization problem as a dual
  Lagrangian.
- [ ] techniques which formulate the constrained optimization problem as a
  box-constrained multi-/many-objective optimization problem, and then to use
  multi-/many-objective optimization problem to find feasible solutions.