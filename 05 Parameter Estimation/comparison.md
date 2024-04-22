---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Parameter Estimation
One important aspect in modelling is parameter estimation.
For this purpose, model parameters are varied until the simulated output matches some reference (usually experimental data).
To quantify the difference between simulation and reference, **CADET-Process** provides a `comparison` module.

+++ {"slideshow": {"slide_type": "fragment"}}

Consider a simple tracer pulse injection onto a chromatographic column.
The following (experimental) concentration profile is measured at the column outlet.

```{code-cell} ipython3
import numpy as np
data = np.loadtxt('./experimental_data/non_pore_penetrating_tracer.csv', delimiter=',')
time_experiment = data[:, 0]
dextran_experiment = data[:, 1]

import matplotlib.pyplot as plt
_ = plt.plot(time_experiment, dextran_experiment)
```

+++ {"slideshow": {"slide_type": "fragment"}}

The goal is to determine the bed porosity of the column, as well the axial dispersion.
Other process parameters like the column geometry and particle sizes are assumed to be known.

+++ {"slideshow": {"slide_type": "slide"}}

## References
To properly work with **CADET-Process**, the experimental data needs to be converted to an internal standard.
The `reference` module provides different classes for different types of experiments.
For in- and outgoing streams of unit operations, the `ReferenceIO` class must be used.

```{code-cell} ipython3
from CADETProcess.reference import ReferenceIO

reference = ReferenceIO('dextran experiment', time_experiment, dextran_experiment)
```

+++ {"slideshow": {"slide_type": "fragment"}}

Similarly to the `SolutionIO` class, the `ReferenceIO` class also provides a plot method:

```{code-cell} ipython3
_ = reference.plot()
```

+++ {"slideshow": {"slide_type": "slide"}}

## Comparator

The `Comparator` class comparing the simulation output with experimental data. It provides several methods for visualizing and analyzing the differences between the data sets. Users can choose from a range of metrics to quantify the differences between the two data sets, such as sum squared errors or shape comparison.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
from CADETProcess.comparison import Comparator
comparator = Comparator()

comparator.add_reference(reference)
```

+++ {"slideshow": {"slide_type": "fragment"}}

```{note}
It's also possible to add multiple references, e.g. for triplicate experiments or for different sensors.
```

+++ {"slideshow": {"slide_type": "slide"}}

## Difference Metrics
There are many metrics which can be used to quantify the difference between the simulation and the reference.
Most commonly, the sum squared error (SSE) is used.

+++ {"slideshow": {"slide_type": "fragment"}}

However, SSE is often not an ideal measurement for chromatography.
Because of experimental non-idealities like pump delays and fluctuations in flow rate there is a tendency for the peaks to shift in time.
This causes the optimizer to favour peak position over peak shape and can lead for example to an overestimation of axial dispersion.

In contrast, the peak shape is dictated by the physics of the physico-chemical interactions while the position can shift slightly due to systematic errors like pump delays.
Hence, a metric which prioritizes the shape of the peaks being accurate over the peak eluting exactly at the correct time is preferable.
For this purpose, **CADET-Process** offers a `Shape` metric.

+++ {"jupyterlab-deck": {"layer": "fragment"}}

To add a difference metric, the following arguments need to be passed to the `add_difference_metric` method:
- `difference_metric`: The type of the metric.
- `reference`: The reference which should be used for the metric.
- `solution_path`: The path to the corresponding solution in the simulation results.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
comparator.add_difference_metric('SSE', reference, 'column.outlet')
```

+++ {"slideshow": {"slide_type": "fragment"}}

Optionally, a start and end time can be specified to only evaluate the difference metric at that slice.
This is particularly useful if system noise (e.g. injection peaks) should be ignored or if certain peaks correspond to certain components.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
comparator = Comparator()
comparator.add_reference(reference)
comparator.add_difference_metric(
    'SSE', reference, 'column.outlet', start=3*60, end=6*60
)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Reference Model

Next to the experimental data, a reference model needs to be configured.
It must include relevant details s.t. it is capable of accurately predicting the experimental system (e.g. tubing, valves etc.).
For this example, the full process configuration can be found {ref}`here <dextran_pulse_example>`.

As an initial guess, the bed porosity is set to $0.5$, and the axial dispersion to $1.0 \cdot 10^{-7}$.
After process simulation, the `evaluate` method is called with the simulation results.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
from CADETProcess.simulator import Cadet
simulator = Cadet()

from examples.characterize_chromatographic_system.column_transport_parameters import process
process.flow_sheet.column.bed_porosity = 0.5
process.flow_sheet.column.axial_dispersion = 1e-7

simulation_results = simulator.simulate(process)

metrics = comparator.evaluate(simulation_results)
print(metrics)
```

+++ {"slideshow": {"slide_type": "fragment"}}

The difference can also be visualized:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
_ = comparator.plot_comparison(simulation_results)
```

+++ {"slideshow": {"slide_type": "fragment"}}

The comparison shows that there is still a large discrepancy between simulation and experiment.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Optimization

Instead of manually adjusting these parameters, an `OptimizationProblem` can be set up which automatically determines the parameter values.
For this purpose, an `OptimimizationProblem` is defined and the process is added as an evaluation object.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
from CADETProcess.optimization import OptimizationProblem
optimization_problem = OptimizationProblem('bed_porosity_axial_dispersion')

optimization_problem.add_evaluation_object(process)
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

Then, the optimization variables are added.
Note, the parameter path associates the variable with the parameter of the corresponding column unit operation.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
optimization_problem.add_variable(
    name='bed_porosity', parameter_path='flow_sheet.column.bed_porosity',
    lb=0.1, ub=0.5,
    transform='auto'
)

optimization_problem.add_variable(
    name='axial_dispersion', parameter_path='flow_sheet.column.axial_dispersion',
    lb=1e-10, ub=1e-6,
    transform='auto'
)
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

Before the difference metrics, which we want to minimize, the `Process` needs to be simulated.
For this purpose, register the `Cadet` simulator instance as an evaluator.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
optimization_problem.add_evaluator(simulator)
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

Now, when adding the `Comparator` (which determines the difference metrics) as objective function, the simulator can be added to the `required` list.
Note that the number of metrics needs to be passed as `n_objectives`.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
optimization_problem.add_objective(
    comparator,
    n_objectives=comparator.n_metrics,
    requires=[simulator]
)
```

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

To get some feedback during optimization, a callback function is added which plots the comparison with the experimental data and stores it in a separate directory.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
def callback(simulation_results, individual, evaluation_object, callbacks_dir='./'):
    comparator.plot_comparison(
        simulation_results,
        file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_comparison.png',
        show=False
    )


optimization_problem.add_callback(callback, requires=[simulator])
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Optimizer

A couple of optimizers are available in **CADET-Process**.
Depending on the problem at hand, some optimizers might outperform others.
Generally, `U_NSGA3`, a genetic algorithm, is a robust choice.
While not necessarily the most efficient, it usually manages to handle complex problems with multiple dimensions, constraints, and objectives.
Here, we limit the number of cores, the population size, as well as the maximum number of generations.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
from CADETProcess.optimization import U_NSGA3
optimizer = U_NSGA3()
optimizer.n_cores = 8
optimizer.pop_size = 64
optimizer.n_max_gen = 16
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
optimization_results = optimizer.optimize(
    optimization_problem,
    use_checkpoint=False
)
```
