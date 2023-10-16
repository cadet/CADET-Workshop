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

# Advanced Optimization Techniques

In this tutorial, we will look at some advanced techniques that can
- improve convergence,
- facilitate setting up complex optimization problems
- provide usefull feedback during (long) optimization runs.

+++ {"slideshow": {"slide_type": "slide"}}

## Parameter Normalization
Most optimization algorithms struggle when optimization variables spread over multiple orders of magnitude.
**CADET-Process** provides several transformation methods which can help to soften these challenges.

+++ {"slideshow": {"slide_type": "fragment"}}

```{figure} ./figures/transform.png
```

+++ {"slideshow": {"slide_type": "slide"}}

### Linear Normalization
The linear normalization maps the variable space from the lower and upper bound to a range between $0$ and $1$ by applying the following transformation:

$$
x^\prime = \frac{x - x_{lb}}{x_{ub} - x_{lb}}
$$

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.optimization import OptimizationProblem

optimization_problem = OptimizationProblem('transform_demo')
optimization_problem.add_variable('var_lin', lb=-100, ub=100, transform='linear')
```

+++ {"slideshow": {"slide_type": "slide"}}

### Log Normalization
The log normalization maps the variable space from the lower and upper bound to a range between $0$ and $1$ by applying the following transformation:

$$
x^\prime = \frac{log \left( \frac{x}{x_{lb}} \right) }{log \left( \frac{x_{ub} }{x_{lb}} \right) }
$$

```{code-cell} ipython3
:tags: [solution]

optimization_problem.add_variable('var_log', lb=-100, ub=100, transform='log')
```

+++ {"slideshow": {"slide_type": "slide"}}

### Auto Transform
This transform will automatically switch between a linear and a log transform if the ratio of upper and lower bounds is larger than some value ($1000$ by default).

```{code-cell} ipython3
:tags: [solution]

optimization_problem.add_variable('var_auto', lb=-100, ub=100, transform='auto')
```

+++ {"slideshow": {"slide_type": "slide"}}

# Evaluation Toolchains

In many situations, some pre- and postprocessing steps are required before the objective function can be evaluated.


```{figure} ./figures/evaluation_steps.png
```

+++ {"slideshow": {"slide_type": "slide"}}

For example, to calculate performance indicators of a process, the following steps need to be performed:
- Simulate the process until stationarity is reached.
- Determine fractionation times under purity constraints.
- Calculate objective functions; Here, two performance indicators are considered as objectives:
    - Productivity,
    - Yield recovery.

```{figure} ./figures/evaluation_example.png
```

+++ {"slideshow": {"slide_type": "notes"}}

As can be seen in this particular evaluation chain, all objectives and constraints require the same simulation results.
To avoid having to recompute all required steps, **CADET-Process** provides a mechanism to add `Evaluators` to an `OptimizationProblem` which can be referenced by objective and constraint functions.
The intermediate results are then cached automatically.

+++ {"slideshow": {"slide_type": "slide"}}

## Evaluation Objects
In the context of **CADET-Process**, the `OptimizationVariables` usually refer to attributes of a `Process` model such as model parameters or event times.
But also fractionation times of the `Fractionator` or attributes of other custom evaluation objects can be used as `OptimizationVariables`.
In this context, the term `EvaluationObject` is introduced which manages that variable.

To associate an `OptimizationVariable` with an `EvaluationObject`, it first needs to be added to the `OptimizationProblem`.
For this purpose, consider a simple `Process` object.

```{code-cell} ipython3
:tags: [solution]

#ToDO: there appears to be a lot missing here.
optimization_problem.add_evaluation_object(evaluation_object)
```

When adding variables, it is now possible to specify with which `EvaluationObject` the variable is associated.
Moreover, the path to the variable in the evaluation object needs to be specified.

![evaluation_single_variable.png](attachment:94704eef-4c4b-4470-a009-227a17ef16a8.png)

```{code-cell} ipython3
:tags: [solution]

optimization_problem.add_variable('var_0', evaluation_objects=[evaluation_object], parameter_path='path.to.variable', lb=0, ub=100)
```

By default, the variable is associated with all evaluation objects.
If no path is provided, the name is also used as path.
Hence, the variable definition can be simplified to:

```{code-cell} ipython3
optimization_problem.add_variable('path.to.variable', lb=0, ub=100)
```

To demonstrate the flexibility of this approach, consider two `EvaluationObjects` and two `OptimizationVariables` where one variable is associated with a single `EvaluationObject`, and the other with both.

![evaluation_multiple_variables.png](attachment:70631541-b26e-4ff5-b7d5-c0eaabd3f1bf.png)

```{code-cell} ipython3
:tags: [solution]

optimization_problem.add_evaluation_object(eval_obj_1)
optimization_problem.add_evaluation_object(eval_obj_2)
optimization_problem.add_variable('var_1', parameter_path='path.to.variable')
optimization_problem.add_variable('var_2', evaluation_objects=[eval_obj_1])
```

+++ {"jp-MarkdownHeadingCollapsed": true, "slideshow": {"slide_type": "slide"}}

### Evaluators
Any callable function can be added as `Evaluator`, assuming the first argument is the result of the previous step and it returns a single result object which is then processed by the next step.

+++

## Callbacks
A `callback` function is a user function that is called periodically by the optimizer in order to allow the user to query the state of the optimization.
For example, a simple user callback function might be used to plot results.
The function is called after each iteration for all best individuals at that state.

```{figure} ./figures/callbacks.png
:width: 33%
```
```{figure} ./figures/callbacks_evaluation.png
```

+++ {"slideshow": {"slide_type": "slide"}}

The callback signature may include any of the following arguments:
- `results`: obj

    x or final result of evaluation toolchain.
- `individual`: {class}`Individual`, optional

    Information about current step of optimzer.
- `evaluation_object`: obj, optional

    Current evaluation object.
- `callbacks_dir`: Path, optional

    Path to store results.

```{code-cell} ipython3
# def callback(individual, evaluation_object, callbacks_dir):
#     print(individual.x, individual.f)
#     evaluation_object.plot(file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_evaluation.png')

# def callback_fractionation(fractionation, individual, evaluation_object, callbacks_dir):
#     fractionation.plot_fraction_signal(0, end=0,
#         file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_fractionation.png',
#         show=False
#     )
# ToDO: this cell was riddled with syntax errors and I don't know how it was supposed to be. I've tried to repair it best as I could
```

To add the function to the `OptimizationProblem`, use the `add_callback` method.
Analogous to objectives,

```{code-cell} ipython3
# optimization_problem.add_callback(
#     callback, requires=[process_simulator, frac_opt]
# )
```

```{figure} ./figures/callbacks_evaluation.png
```

+++

### Multi-Objective
Often, multiple objectives are used.

For example, in the case of a process optimization, where we might care about optimizing purity, performance, and cost. Or for a parameter estimation inverse-fitting approach, we might try to minimize the difference between the experimental data peak and the simulated peak with respect to both the peak position and the peak shape.

These multi-objective optimization problems then lead to the existence of a Pareto-Front:

![image.png](attachment:fc8d7fc4-ec77-46a0-adfd-0bf0c706bb83.png)

The definition of a Pareto-front is: "A Pareto-front is the set of all Pareto efficient solutions." As that isn't a very intuitive definition, consider this explanation:

If you have a trade-off between two (or more) objectives, the Pareto-front contains all the solutions to this trade-off where it is impossible to improve in one objective _without_ making another objective worse. Any solution to the trade-off with room for improvements are called "dominated solutions".

+++

## Multiobjective Optimization example
Let's consider a simple Multiobjective example:

1 Optimization Variable

2 Objectives: simple quadratic functions which we both want to minimize

```{code-cell} ipython3
def objective_1(x):
    return (x - 2) ** 2


def objective_2(x):
    return (x - 8) ** 2 + 10
```

Let's plot the objectives

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x = np.arange(0, 11, 1)


def plot_objectives():
    fig, ax = plt.subplots(1)
    ax.plot(x, objective_1(x), label="Objective 1")
    ax.scatter([2], [0], marker="x", s=300)
    ax.text(1, 6, "Min. obj. 1", size=15)
    ax.plot(x, objective_2(x), label="Objective 2")
    ax.text(6.5, 3, "Min. obj. 2", size=15)
    ax.scatter([8], [10], marker="x", s=300)
    ax.vlines(x=[2, 8], ymin=0, ymax=ax.get_ylim()[1], colors="green", linestyles=":", label="Tradeoff region")
    rect = mpl.patches.Rectangle([2, 0], 6, 80, color="green", alpha=0.2)
    ax.add_patch(rect)
    ax.set_xlabel("Independent variable [-]")
    ax.set_ylabel("Objective performance [-]")
    ax.legend()


plot_objectives()
```

And plot possible solutions. To start, let's explore evenly spaced solutions between x = -1 and x = 11

```{code-cell} ipython3
def plot_solutions(x):
    y1 = objective_1(x)
    y2 = objective_2(x)

    fig, ax = plt.subplots(1)
    ax.scatter(y1, y2)
    ax.set_xlabel("Objective 1 [-]")
    ax.set_ylabel("Objective 2 [-]")
    for point in x:
        ax.text(objective_1(point), objective_2(point), f"x = {point}", size=15)


plot_solutions(np.arange(-1, 12, 1))
```

We can see, that for the points between 2 and 8, no improvement in Objective 1 is possible without making Objective 2 worse. And vice versa. These points make up the pareto front. Points -1 to 1 and 9 to 11 are dominated by the pareto front.

```{code-cell} ipython3
def plot_solutions_with_pareto(x):
    y1 = objective_1(x)
    y2 = objective_2(x)

    fig, ax = plt.subplots(1)
    ax.scatter(y1, y2)
    ax.set_xlabel("Objective 1 [-]")
    ax.set_ylabel("Objective 2 [-]")
    x_pareto = np.linspace(2, 8, 100)
    ax.plot(objective_1(x_pareto), objective_2(x_pareto), color="green", label="Pareto front")
    for point in x:
        ax.text(objective_1(point), objective_2(point), f"x = {point}", size=15)
    ax.vlines(objective_1(0), objective_2(0), objective_2(4), linestyles="--", linewidth=1, label="Dominated",
              colors="red")
    ax.hlines(objective_2(10), objective_1(10), objective_1(6), linestyles="--", linewidth=1, colors="red")
    ax.legend()


plot_solutions_with_pareto(np.arange(-1, 12, 1))
```

We can now optimize this problem in `CADETProcess`

```{code-cell} ipython3
from CADETProcess.optimization import OptimizationProblem

optimization_problem = OptimizationProblem('multi_objective')
optimization_problem.add_variable('x', lb=0, ub=10)
```

Add the objectives we defined above:

```{code-cell} ipython3
optimization_problem.add_objective(objective_1)
optimization_problem.add_objective(objective_2)
```

Alternatively, a single function that returns both objectives can be added. In this case, the number of objectives the function returns needs to be specified by adding `n_objectives` as argument.

```{code-cell} ipython3
def multi_objective(x):
    f_1 = x[0] ** 2 + x[1] ** 2
    f_2 = (x[0] - 2) ** 2 + (x[1] - 2) ** 2
    return np.hstack((f_1, f_2))

# optimization_problem.add_objective(multi_objective, n_objectives=2)
```

In both cases, the total number of objectives is stored in `n_objectives`.

```{code-cell} ipython3
optimization_problem.n_objectives
```

The objective(s) can be evaluated with the `evaluate_objectives()` method.

```{code-cell} ipython3
optimization_problem.evaluate_objectives([1])
```

It is also possible to evaluate multiple sets of input variables at once by passing a 2D array to the `evaluate_objectives_population()` method.

```{code-cell} ipython3
optimization_problem.evaluate_objectives_population([[2], [6], [8]])
```

And run a quick genetic algorithm to find the solutions.

```{code-cell} ipython3
from CADETProcess.optimization import U_NSGA3

optimizer = U_NSGA3()
optimizer.n_max_gen = 5

optimization_results = optimizer.optimize(optimization_problem, log_level="WARNING")
```

```{code-cell} ipython3
print(optimization_results.f)
print(optimization_results.x)
optimization_results.plot_convergence('objectives')
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.scatter(optimization_results.f[:, 0], optimization_results.f[:, 1], label="Solutions on the Pareto-front")
ax.legend()
# ToDo: Is there a built-in pareto plot?
```
