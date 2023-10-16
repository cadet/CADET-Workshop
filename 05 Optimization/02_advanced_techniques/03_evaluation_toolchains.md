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

# Evaluation Toolchains

In many situations, some pre- and postprocessing steps are required before the objective function can be evaluated.
For example, to calculate performance indicators of a process, the following steps need to be performed:
- Simulate the process until stationarity is reached.
- Determine fractionation times under purity constraints.
- Calculate objective functions; Here, two performance indicators are considered as objectives:
    - Productivity,
    - Yield recovery.

![evaluation_example.png](attachment:dd33a663-2da4-46e4-85ea-ac922a0f773a.png)

As can be seen in this particular evaluation chain, all objectives and constraints require the same simulation results.
To avoid having to recompute all required steps, **CADET-Process** provides a mechanism to add `Evaluators` to an `OptimizationProblem` which can be referenced by objective and constraint functions.
The intermediate results are then cached automatically.

+++

### Evaluation Objects
Usually, the `OptimizationVariables` refer to attributes of a `ProcessModel` such as model parameters or events.
But also fractionation times of the `Fractionator` or attributes of custom evaluation objects can be used as `OptimizationVariables`.
In this context, an `EvaluationObject` is introduced.

![evaluation_steps_moo.png](attachment:66cb9ec1-da54-4e84-bc2a-a16cd76ab66d.png)

To associate the `OptimizationVariable` with these `EvaluationObjects`, they first need to be added to the `OptimizationProblem`.

```{code-cell} ipython3
#ToDO: there appears to be a lot missing here.
optimization_problem.add_evaluation_object(evaluation_object)
```

When adding variables, it is now possible to specify with which `EvaluationObject` the variable is associated.
Moreover, the path to the variable in the evaluation object needs to be specified.

![evaluation_single_variable.png](attachment:94704eef-4c4b-4470-a009-227a17ef16a8.png)

```{code-cell} ipython3
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
optimization_problem.add_evaluation_object(eval_obj_1)
optimization_problem.add_evaluation_object(eval_obj_2)
optimization_problem.add_variable('var_1', parameter_path='path.to.variable')
optimization_problem.add_variable('var_2', evaluation_objects=[eval_obj_1])c
```

+++ {"jp-MarkdownHeadingCollapsed": true}

### Evaluators
Any callable function can be added as `Evaluator`, assuming the first argument is the result of the previous step and it returns a single result object which is then processed by the next step.
