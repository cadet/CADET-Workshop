---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Optimization - Exercise

+++

## Exercise 1: Setup simple optimization problem

Solve the following optimization problem.

$$
f(x) = \left(x - 2 \right)^3 - 3 \cdot \left( x - 2 \right) + 1 \\
s.t. \\
0.2 \le x \le 4.5
$$

Explore different optimizers, modify their options and start from different initial values.

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.optimization import OptimizationProblem

optimization_problem = OptimizationProblem('constrained_cubic')
optimization_problem.add_variable('x', lb=0.2, ub=4.5)

def constrained_cubic(x):
    """
    Compute a cubic objective function with a linear constraint.

    Parameters
    ----------
    x : float
        The variable for the objective function.

    Returns
    -------
    float
        The value of the objective function at.
    """
    obj_val = (x - 2)**3 - 3 * (x - 2) + 1

    return obj_val

optimization_problem.add_objective(constrained_cubic)
```

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.optimization import U_NSGA3, TrustConstr

optimizer = U_NSGA3()
optimizer.n_max_gen = 10

optimization_results = optimizer.optimize(optimization_problem)
```

```{code-cell} ipython3
:tags: [solution]

optimizer.results.plot_objectives()
```

```{code-cell} ipython3
optimizer = TrustConstr()
optimizer.n_max_gen = 10

optimization_results = optimizer.optimize(optimization_problem, x0=0.5)
```

```{code-cell} ipython3
optimizer.results.plot_objectives()
```
