---
jupytext:
  formats: ipynb,md:myst
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

# Residence time distributions

The residence time is a measure of the time material spends in a volume through which it flows. Molecules or small volume elements have a single residence time, but more complex systems have a characteristic residence time distribution (RTD).

Basic residence time theory treats a system with an input and an output. The residence time of a small particle is the time between entering and leaving the system.

```{figure} ./resources/system.png
:width: 50%
:align: center
```

+++ {"slideshow": {"slide_type": "slide"}}

The residence time distribution can be described with the function $E(t)$ which has the properties of a probability distribution:
$$E(t) \ge 0~\text{and}~\int_0^\infty E(t)~dt = 1$$

+++ {"slideshow": {"slide_type": "fragment"}}

Residence time distributions are measured by introducing a non-reactive tracer into the system at the inlet:
1. Change input concentration according to a known function (e.g. Dirac $\delta$-function or step function)
2. Measure output concentration
3. Transform output concentration in residence time distribution curve $E(t)$

+++ {"slideshow": {"slide_type": "fragment"}}

For some simple systems (e.g. CSTR and Plug flow reactor model) analytic solutions exist and we can compare them with the simulations of CADET.

+++ {"slideshow": {"slide_type": "slide"}}

**In this lesson, we will:**
- Learn about system responses.
- Setup our first 'real' unit operation models.
- Analyze the solution and take a look 'inside' a column.

+++ {"slideshow": {"slide_type": "slide"}}

## Example 1: Continuous stirred-tank reactor

In an ideal continuous stirred-tank reactor (CSTR), the flow at the inlet is completely and instantly mixed into the bulk of the reactor. The reactor and the outlet fluid have identical, homogeneous compositions at all times. The residence time distribution is exponential:

$$E(t) = \frac{1}{\tau} \cdot e^{-\frac{t}{\tau}}$$

with

$$\tau = \frac{V}{Q}$$

+++ {"slideshow": {"slide_type": "fragment"}}

```{note}
In reality, it is impossible to obtain such rapid mixing, especially on industrial scales where reactor vessels may range between 1 and thousands of cubic meters, and hence the RTD of a real reactor will deviate from the ideal exponential decay.
```

+++ {"slideshow": {"slide_type": "slide"}}

To model this system, the following flow sheet is assumed:

```{figure} ./resources/RTD_CSTR.png
:width: 50%
:align: center
```

+++ {"slideshow": {"slide_type": "slide"}}

### CSTR model

The `CSTR` [model](https://cadet.github.io/master/modelling/unit_operations/cstr.html#cstr-model) in CADET requires the following [parameters](https://cadet.github.io/master/interface/unit_operations/cstr.html#cstr-config):
- Initial volume
- Initial concentration

In later examples, we will also associate [adsorption models](https://cadet.github.io/master/modelling/binding/index.html) and [chemical reactions](https://cadet.github.io/master/modelling/reactions.html) with the `CSTR` unit operation.
For this example, however, we will only consider convective flow.
Also, we assume that the flow rate is constant over time.

+++ {"slideshow": {"slide_type": "slide"}}

Assume the following parameters:


| Parameter           | Value | Unit | Attribute |
| ------------------- | ----- | ---- | --------- |
| Volume              | 1     | L    | `V`       |
| Mean Residence Time | 1     | min  | -         |


**Note:** Since the `CSTR` has a variable volume, the flow rate also needs to be set.

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.processModel import ComponentSystem

component_system = ComponentSystem(1)

tau = 60
V = 1e-3
Q = V/tau

# Unit Operations
from CADETProcess.processModel import Inlet, Cstr, Outlet

## Inlet
# We assume constant flow. Concentrations will later be modified using events.
inlet = Inlet(component_system, 'inlet')
inlet.flow_rate = Q

cstr = Cstr(component_system, 'cstr')
cstr.c = [0]
cstr.V = V
cstr.flow_rate = Q

outlet = Outlet(component_system, 'outlet')

# Flow Sheet
from CADETProcess.processModel import FlowSheet

flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(inlet)
flow_sheet.add_unit(cstr)
flow_sheet.add_unit(outlet)

flow_sheet.add_connection(inlet, cstr)
flow_sheet.add_connection(cstr, outlet)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Pulse experiment CSTR

This method requires the introduction of a very small volume of concentrated tracer at the inlet of a CSTR, such that it approaches the Dirac $\delta$-function.
By definition, the integral of this function is equal to one.
Although an infinitely short injection cannot be produced, it can be made much smaller than the mean residence time of the vessel.

```{figure} ./resources/system_dirac.png
:width: 50%
:align: center
```

```{code-cell} ipython3
:tags: [solution]

step_size = 1e-3

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'rtd_cstr')
process.cycle_time = 10 * tau

process.add_event('start peak', 'flow_sheet.inlet.c', 1/step_size, 0)
process.add_event('end peak', 'flow_sheet.inlet.c', 0, step_size)
```

+++ {"user_expressions": [], "slideshow": {"slide_type": "slide"}}

### Simulate Process

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.simulator import Cadet

simulator = Cadet()

simulation_results = simulator.simulate(process)
simulation_results.solution.cstr.inlet.plot()
simulation_results.solution.cstr.outlet.plot()
```

+++ {"user_expressions": [], "slideshow": {"slide_type": "slide"}}

## Example 2: Plug flow reactor

In an ideal plug flow reactor the fluid elements that enter at the same time continue to move at the same rate and leave together.
Therefore, fluid entering at time $t$ will exit at time $t + \tau$, where $\tau$ is the residence time.
The fraction leaving is a step function, going from $0$ to $1$ at time $\tau$.
The distribution function is therefore a Dirac delta function at $\tau$.

```{figure} ./resources/RTD_PFR.png
:width: 50%
:align: center

```

$$E(t) = \delta (t - \tau)$$


The RTD of a real reactor deviates from that of an ideal reactor, depending on the hydrodynamics (e.g. the axial dispersion) within the vessel.

+++ {"slideshow": {"slide_type": "slide"}}

### PFR model

Although in CADET there is no explicit implementation of the PFR model, we can still model this reactor if we use any of the column models and set the porosity to 1 and the axial dispersion to 0.

In this example, we will use the `LumpedRateModelWithoutPores`. For the model equations see [here](https://cadet.github.io/master/modelling/unit_operations/lumped_rate_model_without_pores.html) and the parameters [here](https://cadet.github.io/master/interface/unit_operations/lumped_rate_model_without_pores.html).

+++ {"slideshow": {"slide_type": "fragment"}}

Assume the following parameters:


| Parameter                  | Value | Unit | Attribute            |
| -------------------------- | ----- | ---- | -------------------- |
| Reactor Length             | 1     | L    | `V`                  |
| Reactor Cross Section Area | 0.1   | m²   | `cross_section_area` |
| Mean Residence Time        | 1     | min  | -                    |

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.processModel import ComponentSystem

component_system = ComponentSystem(1)

tau = 60
length = 1
cross_section_area = 0.1
V = cross_section_area * length
Q = V/tau

# Unit Operations
from CADETProcess.processModel import Inlet, TubularReactor, Outlet

## Inlet
# We assume constant flow. Concentrations will later be modified using events.
inlet = Inlet(component_system, 'inlet')
inlet.flow_rate = V/tau

pfr = TubularReactor(component_system, 'pfr')
pfr.length = 1
pfr.cross_section_area = cross_section_area

pfr.axial_dispersion = 0

outlet = Outlet(component_system, 'outlet')

# Flow Sheet
from CADETProcess.processModel import FlowSheet

flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(inlet)
flow_sheet.add_unit(pfr)
flow_sheet.add_unit(outlet)

flow_sheet.add_connection(inlet, pfr)
flow_sheet.add_connection(pfr, outlet)
```

+++ {"slideshow": {"slide_type": "fragment"}}
```{code-cell} ipython3
tags: [solution]

step_size = 1e-3

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'rtd_pfr')
process.cycle_time = 2 * tau

process.add_event('start peak', 'flow_sheet.inlet.c', 1/step_size, 0)
process.add_event('end peak', 'flow_sheet.inlet.c', 0, step_size)
```

+++ {"user_expressions": [], "slideshow": {"slide_type": "slide"}}

### Simulate Process

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.simulator import Cadet

simulator = Cadet()

simulation_results = simulator.simulate(process)
simulation_results.solution.pfr.inlet.plot()
simulation_results.solution.pfr.outlet.plot()
```

+++ {"slideshow": {"slide_type": "slide"}}

```{warning}
Because of numerical dispersion, solvers like CADET are not suited to simulate stiff systems like the one presented.
To get a more acurate solution, the number of axial cells needs to be increased (a lot) which also increases simulation time (a lot).
Since usually there is some (physical) dispersion in real systems anyway, mostly this is not a problem because it will smoothen the profiles anyway.
The example just serves to show the limitations of CADET and that while it may not be very accurate in this case, the value of the mean residence time is still where we would expect it.
```

+++ {"slideshow": {"slide_type": "slide"}}

### Discretization

In case of the column, there are several options for adapting the spatial discretization of the PDE model.
However, the two most important ones are the number of grid cells in the column (axial direction) and the particles.
Since the lumped rate model without pores does not have particles, we only need to specify axial cells `n_col`.
The default is $100$ which should work for most scenarios.


```{note}
CADET currently uses a finite volume scheme for the spatial discretization.
However, we are in the process of implementing a new method, the *Discontinuous Galerkin* method which will increase speed substantially.
```

```{code-cell} ipython3
:tags: [solution]

pfr.discretization
```

+++ {"slideshow": {"slide_type": "fragment"}}

### High discretization

```{code-cell} ipython3
:tags: [solution]

pfr.discretization.ncol = 2000
simulation_results = simulator.simulate(process)
simulation_results.solution.pfr.outlet.plot()
```

+++ {"user_expressions": []}

### Low discretization

```{code-cell} ipython3
:tags: [solution]

pfr.discretization.ncol = 20
simulation_results = simulator.simulate(process)
simulation_results.solution.pfr.outlet.plot()
```

+++ {"slideshow": {"slide_type": "slide"}}

### Visualization

Additionally to the solution at the inlet and outlet of a unit operation, we can also take a look inside the column to see the peak move.

For this purpose, set the flag in the unit's `SolutionRecorder`.
Then, the `SimulationResults` will also contain an entry for the bulk.

**Note:** Since this solution is two-dimensinal (space and time), the solution can be plotted at a given position (`plot_at_location`) or a given time (`plot_at_time`).

```{code-cell} ipython3
:tags: [solution]

pfr.solution_recorder.write_solution_bulk = True

simulation_results = simulator.simulate(process)
```

```{code-cell} ipython3
:tags: [solution]

simulation_results.solution.pfr.bulk.plot_at_position(0.5)
simulation_results.solution.pfr.bulk.plot_at_time(0.01)
```

```{code-cell} ipython3
:tags: [solution]

from ipywidgets import interact, interactive
import ipywidgets as widgets

Visualization
def graph_column(time=0):
    fig, ax = simulation_results.solution.pfr.bulk.plot_at_time(time)
    ax.set_ylim(0,0.1)

style = {'description_width': 'initial'}
interact(
    graph_column,
    time=widgets.IntSlider(
        min=0, max=process.cycle_time, step=10, layout={'width': '800px'}, style=style, description='Time'
    )
)
```
