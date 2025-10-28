---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"slideshow": {"slide_type": ""}, "editable": true}

# Process Optimization

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

First, an `OptimizationProblem` is instantiated.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from CADETProcess.optimization import OptimizationProblem
optimization_problem = OptimizationProblem('batch_elution')
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

The fully configured `Process` is imported from the examples and added to the `OptimizationProblem`.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from examples.batch_elution.process import process
optimization_problem.add_evaluation_object(process)
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

In this case, the cycle time of the process, as well as the feed duration are to be optimized.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
optimization_problem.add_variable('cycle_time', lb=10, ub=600)
optimization_problem.add_variable('feed_duration.time', lb=10, ub=300)
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

To ensure that the feed duration is always shorter than the cycle time, a linear constraint is added.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
optimization_problem.add_linear_constraint(
    ['feed_duration.time', 'cycle_time'], [1, -1]
)
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

Now, the a simulator is configured and registered as evaluator.
We want to ensure that the simulator repeats the simulatation until cyclic stationarity is reached.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from CADETProcess.simulator import Cadet
process_simulator = Cadet()
process_simulator.evaluate_stationarity = True

optimization_problem.add_evaluator(process_simulator)
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

Here, the fractionation optimizer is configured and registered as another evaluator.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
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

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

Now, the objectives are defined.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from CADETProcess.performance import PerformanceProduct
ranking = [1, 1]
performance = PerformanceProduct(ranking=ranking)

optimization_problem.add_objective(
    performance,
    requires=[process_simulator, frac_opt],
    minimize=False,
)
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

Add callback for post-processing

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def callback(fractionation, individual, evaluation_object, callbacks_dir):
    fractionation.plot_fraction_signal(
        file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_fractionation.png',
        show=False
    )

optimization_problem.add_callback(
    callback, requires=[process_simulator, frac_opt]
)
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

Finally, the an optimizer is configured.
Again, we use `U_NSGA3`.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from CADETProcess.optimization import U_NSGA3
optimizer = U_NSGA3()

optimizer.n_cores = 8
optimizer.pop_size = 32
optimizer.n_max_gen = 16
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
results = optimizer.optimize(
    optimization_problem,
    use_checkpoint=False,
)
```
