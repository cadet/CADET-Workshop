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

+++

```{figure} ./figures/evaluation_example.png
```

+++ {"slideshow": {"slide_type": "slide"}}

## Evaluation Objects

```{figure} ./figures/evaluation_steps.png
```

- `OptimizationVariables` usually refers to attributes of a `Process` model (e.g. model parameters / event times.
- `EvaluationObject` objects manage the value of that optimization variable
- `Evaluators` execute (intermediate) steps required for calculating the objective (e.g. simulation)

+++ {"slideshow": {"slide_type": "slide"}}

```{figure} ./figures/evaluation_single_variable.png
:width: 30%
```


To associate an `OptimizationVariable` with an `EvaluationObject`, it first needs to be added to the `OptimizationProblem`.
For this purpose, consider a simple `Process` object from the [examples collection](https://cadet-process.readthedocs.io/en/stable/examples/batch_elution/process.html).

```{code-cell} ipython3
:tags: [solution]

from examples.batch_elution.process import process

optimization_problem = OptimizationProblem('evaluator')

optimization_problem.add_evaluation_object(process)
```

+++ {"slideshow": {"slide_type": "fragment"}}

Then add the variable. In addition, specify:

- `parameter_path`: Path to the variable in the evaluation object
- `evaluation_objects`: The evaluation object(s) for which the variable should be set.

```{code-cell} ipython3
:tags: [solution]

optimization_problem.add_variable('length', evaluation_objects=[process], parameter_path='flow_sheet.column.length', lb=0, ub=1)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Multiple Evaluation Objects

```{figure} ./figures/evaluation_multiple_variables.png
:width: 30%
```

```{code-cell} ipython3
:tags: [solution]

optimization_problem = OptimizationProblem('two_eval_obj')

import copy
process_2 = copy.deepcopy(process)
process_2.name = 'foo'

optimization_problem.add_evaluation_object(process)
optimization_problem.add_evaluation_object(process_2)

optimization_problem.add_variable('flow_sheet.column.length', lb=0, ub=1)
optimization_problem.add_variable('flow_sheet.column.diameter', lb=0, ub=1, evaluation_objects=process_2)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Evaluators
Any callable function can be added as `Evaluator`, assuming the first argument is the result of the previous step and it returns a single result object which is then processed by the next step.

```{figure} ./figures/evaluation_steps.png
```

- Any callable function can be added as `Evaluator`.
- Each `Evaluator` takes the previous result as input and returns a new (intermediate) result.
- Intermediate results are automatically cached.

+++ {"slideshow": {"slide_type": "slide"}}

## Evaluator Example

In this example, two steps are required:
- Process Simulation
- Fractionation

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: [solution]
---
from CADETProcess.simulator import Cadet
simulator = Cadet()

optimization_problem.add_evaluator(simulator)

from CADETProcess.fractionation import FractionationOptimizer
frac_opt = FractionationOptimizer()

optimization_problem.add_evaluator(
    frac_opt,
    kwargs={
        'purity_required': [0.95, 0.95],
        'ignore_failed': False,
        'allow_empty_fractions': False,
    }
)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Adding Objectives

Now, when adding objectives, specify which steps are required for each objective

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.performance import Productivity, Recovery, Purity

productivity = Productivity()
optimization_problem.add_objective(
    productivity,
    n_objectives=2,
    requires=[simulator, frac_opt]
)

recovery = Recovery()
optimization_problem.add_objective(
    recovery,
    n_objectives=2,
    requires=[simulator, frac_opt]
)

purity = Purity()
optimization_problem.add_nonlinear_constraint(
    purity,
    n_nonlinear_constraints=2,
    requires=[simulator, frac_opt],
    bounds=[0.95, 0.95]    
)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Evaluate Toolchain

To check the toolchain, simply call `evaluate_objectives`

```{code-cell} ipython3
:tags: [solution]

optimization_problem.evaluate_objectives([0.5, 0.01])
```

```{code-cell} ipython3
optimization_problem.objective_labels
```

+++ {"slideshow": {"slide_type": "fragment"}}

```{error} 

This should take into account the evaluation objects!
Let's learn how to report a bug!
```

+++ {"slideshow": {"slide_type": "slide"}}

## Callbacks
A `callback` function is a user function that is called periodically by the optimizer in order to allow the user to query the state of the optimization.
For example, a simple user callback function might be used to plot results.
The function is called after each iteration for all best individuals at that state.

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
:tags: [solution]

def callback(fractionation, individual, evaluation_object, callbacks_dir):
    fractionation.plot_fraction_signal(
        file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_fractionation.png',
        show=False
    )
```

+++ {"slideshow": {"slide_type": "fragment"}}

To add the function to the `OptimizationProblem`, use the `add_callback` method.
Analogous to objectives,

```{code-cell} ipython3
:tags: [solution]

optimization_problem.add_callback(
    callback, requires=[simulator, frac_opt]
)
```
