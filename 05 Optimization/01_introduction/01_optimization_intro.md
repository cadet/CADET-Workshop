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

+++ {"slideshow": {"slide_type": "slide"}}

# Optimization

One of the main applications of **CADET-Process** is performing optimization studies.
Optimization refers to the selection of a solution with regard to some criterion.
In the simplest case, an optimization problem consists of minimizing some function $f(x)$ by systematically varying the input values $x$ and computing the value of that function.

$$
\min_x f(x)
$$

+++ {"slideshow": {"slide_type": "fragment"}}

In the context of physico-chemical processes, examples for the application of optimization studies include scenarios such as process optimization and parameter estimation.
Here, often many variables are subject to optimization, multiple criteria have to be balanced, and additional linear and nonlinear constraints need to be considered.

$$
\min_x f(x) \\
s.t. \\
    g(x) \le 0, \\
    h(x) = 0, \\
    x \in \mathbb{R}^n \\
$$


where $g$ summarizes all inequality constraint functions, and $h$ equality constraints.

+++ {"slideshow": {"slide_type": "slide"}}

In the following, the `optimization` module of **CADET-Process** is introduced. To decouple the problem formulation from the problem solution, two classes are provided:
- An `OptimizationProblem` class to specify optimization variables, objectives and constraints, and
- an abstract `Optimizer` interface which allows using different external optimizers to solve the problem.

+++ {"slideshow": {"slide_type": "slide"}}

## Example 1: Minimization of a quadratic function

Usually, the objective function is not known; it can only be evaluated at certain points.
For demonstration purpouses, consider a quadratic function to be minimized.

$$
f(x) = x^2
$$

+++

Since we already know a lot about this function, it can help to introduce some of the Optimization concepts of CADET-Process.
For example, the results should yield:
- $x_{opt} = 0$
- $f_{opt} = 0$.

```{code-cell} ipython3
:tags: [solution]

%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 101)
y = x**2

fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(x, y)
```

+++ {"slideshow": {"slide_type": "slide"}}

### OptimizationProblem

The `OptimizationProblem` class is is used to specify optimization variables, objectives and constraints.
After import, the `OptimizationProblem` initialized with a name.

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.optimization import OptimizationProblem

optimization_problem = OptimizationProblem('quadratic')
```

+++ {"slideshow": {"slide_type": "fragment"}}

#### Optimization Variables
Any number of variables can be added to the `OptimizationProblem`.
To add a variable, use the `add_variable` method.
In this case, there is only a single variable.
The first argument is the name of the variable.
Moreover, lower and upper bounds can be specified.

```{code-cell} ipython3
:tags: [solution]

optimization_problem.add_variable(name='x', lb=-10, ub=10)
```

+++ {"slideshow": {"slide_type": "fragment"}}

The total number of variables is stored in `n_variables` and the names in `variable_names`

```{code-cell} ipython3
:tags: [solution]

print(optimization_problem.n_variables)
print(optimization_problem.variable_names)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Objectives
Any `callable` (i.e. an object that can be called using the `( )` operator) can be added as an objective as long as it takes x as the first argument.
Multi-objective optimization is also possible with CADET-Python (more on that later).
For now, the objective must return a single, scalar value.

```{note}
Usually, there are multiple variables involved. Hence, the function is expected to accept a list.
```

```{code-cell} ipython3
:tags: [solution]

def objective_fun(x):
    return x[0] ** 2

optimization_problem.add_objective(objective_fun)
```

+++ {"slideshow": {"slide_type": "fragment"}}

To evaluate the this function, the `evaluate_objective` method can be used.
This is useful to test whether everything works as expected.

```{code-cell} ipython3
:tags: [solution]

print(optimization_problem.evaluate_objectives([2]))
```

+++ {"slideshow": {"slide_type": "notes"}}

If the value is out of bounds, an error will be thrown.

+++ {"slideshow": {"slide_type": "slide"}}

### Optimizer
The `OptimizerAdapter` provides a unified interface for using external optimization libraries.
It is responsible for converting the `OptimizationProblem` to the specific API of the external `Optimizer`.
Currently, adapters to **Pymoo** and **Scipy's** optimization suite are implemented, all of which are published under open source licenses that allow for academic as well as commercial use.

Before the optimization can be run, the `Optimizer` needs to be initialized and configured.
For this example, `U_NSGA3` is used, a genetic algorithm.

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.optimization import U_NSGA3

optimizer = U_NSGA3()
```

+++ {"slideshow": {"slide_type": "fragment"}}

All options can be displayed the following way:

```{code-cell} ipython3
:tags: [solution]

print(optimizer.options)
```

+++ {"slideshow": {"slide_type": "fragment"}}

To reduce the calculation time, let's limit the maximum number of generations that the `Optimizer` evaluates:

```{code-cell} ipython3
:tags: [solution]

optimizer.n_max_gen = 10
```

+++ {"slideshow": {"slide_type": "slide"}}

To optimize the `OptimizationProblem`, call the `optimize()` method.
By default, CADET-Process tries to autogenerate initial values.
However, it's also possible to pass them as an additional `x0` argument.
More on generating initial values later.

```{code-cell} ipython3
:tags: [solution]

optimization_results = optimizer.optimize(optimization_problem, x0=1, log_level="WARNING")
```

+++ {"slideshow": {"slide_type": "slide"}}

### Optimization Progress and Results

The `OptimizationResults` which are returned contain information about the progress of the optimization.
For example, the attributes `x` and `f` contain the final value(s) of parameters and the objective function.

```{code-cell} ipython3
:tags: [solution]

print(optimization_results.x)
print(optimization_results.f)
```

+++ {"slideshow": {"slide_type": "slide"}}

After optimization, several figures can be plotted to vizualize the results.
For example, the convergence plot shows how the function value changes with the number of evaluations.

```{code-cell} ipython3
:tags: [solution]

optimization_results.plot_convergence('objectives')
```

+++ {"slideshow": {"slide_type": "slide"}}

The `plot_objectives` method shows the objective function values of all evaluated individuals.
Here, lighter color represent later evaluations.
Note that by default the values are plotted on a log scale if they span many orders of magnitude.
To disable this, set `autoscale=False`.

```{code-cell} ipython3
optimization_results.plot_corner()
```

```{code-cell} ipython3
:tags: [solution]

optimization_results.plot_objectives(autoscale=False)
```

+++ {"slideshow": {"slide_type": "slide"}}

Note that more figures are created for constrained optimization, as well as multi-objective optimization.
All figures are also saved automatically in the `working_directory`.
Moreover, results are stored in a `.csv` file.
- The `results_all.csv` file contains information about all evaluated individuals.
- The `results_last.csv` file contains information about the last generation of evaluated individuals.
- The `results_pareto.csv` file contains only the best individual(s).

+++ {"slideshow": {"slide_type": "slide"}}

## Example 2: Constrained Optimization

Example taken from [SciPy Documentation](https://docs.scipy.org/doc/scipy/tutorial/optimize.html#id34)

As an example let us consider the constrained minimization of the Rosenbrock function:

$$
\min_{x_0, x_1} & ~~100\left(x_{1}-x_{0}^{2}\right)^{2}+\left(1-x_{0}\right)^{2} &\\
    \text{subject to: } & x_0 + 2 x_1 \leq 1 & \\
                        & x_0^2 + x_1 \leq 1  & \\
                        & x_0^2 - x_1 \leq 1  & \\
                        & 2 x_0 + x_1 = 1 & \\
                        & 0 \leq  x_0  \leq 1 & \\
                        & -0.5 \leq  x_1  \leq 2.0. &
$$

+++ {"slideshow": {"slide_type": "slide"}}

This optimization problem has the unique solution $[x_0, x_1] = [0.4149,~ 0.1701]$.

```{figure} ./figures/rosenbrock.png
:align: center
:width: 50%

```

+++ {"slideshow": {"slide_type": "slide"}}

To setup this problem, first a new `OptimizationProblem` is created and the variables are added, including bounds.
It is important to note, that `x0` cannot be used as variable name since it is reserved for the initial values.

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.optimization import OptimizationProblem

rosenbrock_problem = OptimizationProblem('rosenbrock')

rosenbrock_problem.add_variable('x_0', lb=0, ub=1)
rosenbrock_problem.add_variable('x_1', lb=-0.5, ub=2.0)
```

+++ {"slideshow": {"slide_type": "fragment"}}

Then, then objective function is defined and added.

```{code-cell} ipython3
def rosenbrock_objective(x):
    x_0 = x[0]
    x_1 = x[1]

    return 100 * (x_1 - x_0 ** 2) ** 2 + (1 - x_0) ** 2


rosenbrock_problem.add_objective(rosenbrock_objective)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Linear inequality constraints
Linear constraints are usually defined in the following way

$$
A \cdot x \leq b
$$

In **CADET-Process**, add each row $a$ of the constraint matrix needs to be added individually.
The `add_linear_constraint` function takes the variables subject to the constraint as first argument.
The left-hand side $a$ and the bound $b_a$ are passed as second and third argument.
It is important to note that the column order in $a$ is inferred from the order in which the optimization variables are passed.

To add the constraints of the Rosenbrock function to the optimization problem, add the following:

```{code-cell} ipython3
:tags: [solution]

rosenbrock_problem.add_linear_constraint(['x_0', 'x_1'], [1, 2], 1)
```

+++ {"slideshow": {"slide_type": "slide"}}

To wheck if a point fulfils the linear inequality constraints, use the `check_linear_constraints` method.
It returns `True` if the point is within bounds and `False` otherwise.

```{code-cell} ipython3
:tags: [solution]

rosenbrock_x0 = [0.5, 0]
rosenbrock_problem.check_linear_constraints(rosenbrock_x0)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Linear equality constraints
Linear equality constraints are usually defined in the following way

$$
A_{eq} \cdot x = b_{eq}
$$

In **CADET-Process**, add each row $a_{eq}$ of the constraint matrix needs to be added individually.
The `add_linear_equality_constraint` function takes the variables subject to the constraint as first argument.
The left-hand side $a_{eq}$ and the bound $b_{eq, a}$ are passed as second and third argument.
It is important to note that the column order in $a$ is inferred from the order in which the optimization variables are passed.

To add this constraint of the Rosenbrock function

$$
2 x_0 + x_1 = 1
$$

to the optimization problem, add the following:

```{code-cell} ipython3
:tags: [solution]

rosenbrock_problem.add_linear_equality_constraint(['x_0', 'x_1'], lhs=[2, 1], beq=1, eps=1e-8)
```

+++ {"slideshow": {"slide_type": "slide"}}

To wheck if a point fulfils the linear equality constraints, use the `check_linear_equality_constraints` method.
It returns `True` if the point is within bounds and `False` otherwise.

```{code-cell} ipython3
:tags: [solution]

rosenbrock_problem.check_linear_equality_constraints(rosenbrock_x0)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Nonlinear constraints
It is also possible to add nonlinear constraints to the `OptimizationProblem`.

$$
g(x) \le 0 \\
$$

In contrast to linear constraints, and analogous to objective functions, nonlinear constraints need to be added as a callable functions.
Note that multiple nonlinear constraints can be added.
In addition to the function, lower or upper bounds can be added.

+++ {"slideshow": {"slide_type": "slide"}}

To add the constraints of the Rosenbrock function to the optimization problem, add the following.

```{code-cell} ipython3
:tags: [solution]

def nonlincon_1(x):
    x_0 = x[0]
    x_1 = x[1]

    return x_0 ** 2 + x_1


rosenbrock_problem.add_nonlinear_constraint(nonlincon_1, bounds=[1])


def nonlincon_2(x):
    x_0 = x[0]
    x_1 = x[1]

    return x_0 ** 2 - x_1


rosenbrock_problem.add_nonlinear_constraint(nonlincon_2, bounds=[1])
```

+++ {"slideshow": {"slide_type": "slide"}}

Again, the function can be evaluated manually.

```{code-cell} ipython3
:tags: [solution]

rosenbrock_problem.check_nonlinear_constraints(rosenbrock_x0)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Optimizer
To solve this problem, a **trust region method** is used, here:

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.optimization import TrustConstr

trust_const_optimizer = TrustConstr()

rosenbrock_optimization_results = trust_const_optimizer.optimize(
    rosenbrock_problem,
    x0=rosenbrock_x0
)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Optimization Progress and Results

Since in this problem, nonlinear constraints are involved, their convergence can also be plotted

```{code-cell} ipython3
:tags: [solution]

rosenbrock_optimization_results.x
```

```{code-cell} ipython3
:tags: [solution]

rosenbrock_optimization_results.plot_convergence('objectives')
```

```{code-cell} ipython3
:tags: [solution]

rosenbrock_optimization_results.plot_convergence('nonlinear_constraints')
```

```{code-cell} ipython3
:tags: [solution]

rosenbrock_optimization_results.plot_objectives()
```

+++ {"slideshow": {"slide_type": "slide"}}

### Initial Values

To start solving any optimization problem, initial values are required.
In CADET-Process, this can be done automatically if
- all variables have upper and lower bounds
- no linear equality constraints exist (this will be improved in the future!)

+++ {"slideshow": {"slide_type": "fragment"}}

To create initial values, call `create_initial_values` and specify the number of individuals that should be returned.
By default, a random value is returned which fulfills all bound constraints and linear constraints.

```{code-cell} ipython3
#if rosenbrock_problem.n_linear_equality_constraints > 0:
#    rosenbrock_problem.remove_linear_equality_constraint(0) # just for demonstration purposes
```

```{code-cell} ipython3
x0 = rosenbrock_problem.create_initial_values(method='chebyshev')
print(x0)
```

+++ {"slideshow": {"slide_type": "fragment"}}

By specifying `method='chebyshev'`, the so-called Chebyshev center is returned; the center of a minimal-radius ball enclosed by the convex parameter space.
If no information about the location is known, this can be a good starting point.

```{code-cell} ipython3
x0 = rosenbrock_problem.create_initial_values(method='chebyshev')
print(x0)
```

+++ {"slideshow": {"slide_type": "slide"}}

For population based algorithms such as genetic algorithm, an entire population is required.
Because efficiently sampling a convex polytope can be difficult, we make use  of [hopsy](https://modsim.github.io/hopsy/index.html), a tool for Markov chain Monte Carlo sampling on convex polytopes.

+++

Let's first create a method to better visualize these points

```{code-cell} ipython3
def plot_initial_values(x0):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots()
    try:
        ax.scatter(x0[:, 0], x0[:, 1])
        ax.set_xlabel(r'$x_0$')
        ax.set_ylabel(r'$x_1$')
    except IndexError:
        ax.scatter(x0, np.ones_like((x0)))
        ax.set_xlabel(r'$x_0$')
    fig.tight_layout()

x0 = rosenbrock_problem.create_initial_values(500)
plot_initial_values(x0)
#for x in x0:
#    print(rosenbrock_problem.check_linear_equality_constraints(x))
```

```{code-cell} ipython3

```
