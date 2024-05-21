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

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Comparison - Exercise

For this exercise, consider the following flow sheet:

```{figure} ./figures/comparator_uv_cond.png
```

To characterize the system periphery, a pore-penetrating tracer (Acetone) pulse was injected into the system without a column and UV and conductivity were measured.

```{note}
To model the additional dispersion of the system, two `Cstr`s were introduced
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Exercise 1: Comparator

The (synthetic) experiment was repeated to account for system variability. The data was converted to concentrations in $mM$ and can be found in `./experimental_data`.

**Task:**
- Import and plot the experimental data using the `ReferenceIO` class.
- Add the references to the `Comparator`.
- Add the `SSE` difference metric and compare with simulation results.
- Compare with other metrics.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Inlet, Cstr, TubularReactor, Outlet
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process
from CADETProcess.simulator import Cadet

# Some Variables
Q_ml_min = 0.5  # ml/min
Q_m3_s = Q_ml_min/(60*1e6)
V_tracer = 50e-9  # m3

# Component System
component_system = ComponentSystem(['Acetone'])

# Unit Operations
acetone = Inlet(component_system, name='acetone')
acetone.c = [131.75]

water = Inlet(component_system, name='water')
water.c = [0]

inlet_valve = Cstr(component_system, name='inlet_valve')
inlet_valve.V = 0.3e-6
inlet_valve.c = [0]

tubing = TubularReactor(component_system, name='tubing')
tubing.length = 0.85
tubing.diameter = 0.75e-3
tubing.axial_dispersion = 1e-7
tubing.c = [0]

uv_detector = Cstr(component_system, name='uv_detector')
uv_detector.V = 0.1e-6
uv_detector.c = [0]

cond_detector = Cstr(component_system, name='cond_detector')
cond_detector.V = 0.2e-6
cond_detector.c = [0]

outlet = Outlet(component_system, name='outlet')

# Flow Sheet
fs = FlowSheet(component_system)
fs.add_unit(acetone)
fs.add_unit(water)
fs.add_unit(inlet_valve)
fs.add_unit(tubing)
fs.add_unit(uv_detector)
fs.add_unit(cond_detector)
fs.add_unit(outlet)

fs.add_connection(acetone, inlet_valve)
fs.add_connection(water, inlet_valve)
fs.add_connection(inlet_valve, tubing)
fs.add_connection(tubing, uv_detector)
fs.add_connection(uv_detector, cond_detector)
fs.add_connection(cond_detector, outlet)

# Process
process = Process(fs, 'Acetone_Pulse_no_column')
process.cycle_time = 500

process.add_event('pulse_acetone_on', 'flow_sheet.acetone.flow_rate', Q_m3_s, 0)
process.add_event('pulse_acetone_off', 'flow_sheet.acetone.flow_rate', 0, V_tracer/Q_m3_s)

process.add_event('feed_water_on', 'flow_sheet.water.flow_rate', Q_m3_s, V_tracer/Q_m3_s)
process.add_event('feed_water_off', 'flow_sheet.water.flow_rate', 0, process.cycle_time)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
simulator = Cadet()

simulation_results = simulator.simulate(process)

fig, ax = None, None
y_max = 30
fig, ax = simulation_results.solution.inlet_valve.outlet.plot(fig=fig, ax=ax, y_max=y_max)
fig, ax = simulation_results.solution.tubing.outlet.plot(fig=fig, ax=ax, y_max=y_max)
fig, ax = simulation_results.solution.uv_detector.outlet.plot(fig=fig, ax=ax, y_max=y_max)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
from CADETProcess.reference import ReferenceIO
import numpy as np

data_uv_1 = np.loadtxt('./experimental_data/uv_detector_1.csv', delimiter=',')
time_uv_1 = data_uv_1[:, 0]
uv_1 = data_uv_1[:, 1]
reference_uv_1 = ReferenceIO('uv_1', time_uv_1, uv_1)
_ = reference_uv_1.plot()

data_uv_2 = np.loadtxt('./experimental_data/uv_detector_2.csv', delimiter=',')
time_uv_2 = data_uv_2[:, 0]
uv_2 = data_uv_2[:, 1]
reference_uv_2 = ReferenceIO('uv_2', time_uv_2, uv_2)
_ = reference_uv_2.plot()

data_cond_1 = np.loadtxt('./experimental_data/cond_detector_1.csv', delimiter=',')
time_cond_1 = data_cond_1[:, 0]
cond_1 = data_cond_1[:, 1]
reference_cond_1 = ReferenceIO('cond_1', time_cond_1, cond_1)
_ = reference_cond_1.plot()

data_cond_2 = np.loadtxt('./experimental_data/cond_detector_2.csv', delimiter=',')
time_cond_2 = data_cond_2[:, 0]
cond_2 = data_cond_2[:, 1]
reference_cond_2 = ReferenceIO('cond_2', time_cond_2, cond_2)
_ = reference_cond_2.plot()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
from CADETProcess.comparison import Comparator

comparator = Comparator()

comparator.add_reference(reference_uv_1)
comparator.add_reference(reference_uv_2)
comparator.add_reference(reference_cond_1)
comparator.add_reference(reference_cond_2)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
comparator.add_difference_metric('Shape', reference_uv_1, 'uv_detector.outlet')
comparator.add_difference_metric('Shape', reference_uv_2, 'uv_detector.outlet')

comparator.add_difference_metric('Shape', reference_cond_1, 'cond_detector.outlet')
comparator.add_difference_metric('Shape', reference_cond_2, 'cond_detector.outlet')
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
comparator.evaluate(simulation_results)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
_ = comparator.plot_comparison(simulation_results)
```

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

```{note}
It's also possible to add multiple references, e.g. for triplicate experiments or for different sensors.
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Exercise 2: Optimization

**Task:**
- Setup optimization problem
- Add the process to the optimization problem and define the following optimization variables.
  - `inlet_valve.V`: [1e-7, 1e-6] m³
  - `tubing.axial_dispersion` = [1e-8, 1e-6] m²/s
  - `uv_detector.V`: [1e-7, 1e-6] m³
  - `cond_detector.V`: [1e-7, 1e-6] m³
- Add the `Comparator` as objective, using the simulator as evaluator
- Define a callback function that compares simulation output with experimental data.
- Setup an optimizer, e.g. `U_NSGA3` and run the optimization

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from CADETProcess.optimization import OptimizationProblem
optimization_problem = OptimizationProblem('System periphery')

optimization_problem.add_evaluation_object(process)

optimization_problem.add_variable(
    name='Inlet valve volume', parameter_path='flow_sheet.inlet_valve.V',
    lb=1e-7, ub=1e-6,
    transform='auto'
)

optimization_problem.add_variable(
    name='Tubing axial dispersion', parameter_path='flow_sheet.tubing.axial_dispersion',
    lb=1e-8, ub=1e-6,
    transform='auto'
)

optimization_problem.add_variable(
    name='UV Detector volume', parameter_path='flow_sheet.uv_detector.V',
    lb=1e-7, ub=1e-6,
    transform='auto'
)

optimization_problem.add_variable(
    name='Conductivity detector volume', parameter_path='flow_sheet.cond_detector.V',
    lb=1e-7, ub=1e-6,
    transform='auto'
)

optimization_problem.add_evaluator(simulator)

optimization_problem.add_objective(
    comparator,
    n_objectives=comparator.n_metrics,
    requires=[simulator]
)

def callback(simulation_results, individual, evaluation_object, callbacks_dir='./'):
    comparator.plot_comparison(
        simulation_results,
        file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_comparison.png',
        show=False
    )


optimization_problem.add_callback(callback, requires=[simulator])
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from CADETProcess.optimization import U_NSGA3
optimizer = U_NSGA3()
optimizer.n_cores = 8
optimizer.pop_size = 64
optimizer.n_max_gen = 16

optimization_results = optimizer.optimize(
    optimization_problem,
    use_checkpoint=False
)
```