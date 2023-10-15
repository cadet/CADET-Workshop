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

+++
# Residence Time Distribution - Exercises


+++
## Exercise 1: Step function in CSTR

Analyze how the concentration profile of a `CSTR` reacts to a step function:
```{figure} ./resources/step.png
:width: 50%
:align: center
```

```{figure} ./resources/RTD_CSTR.png
:width: 50%
:align: center
```



***Hint:*** Always check the input arguments of our model template functions.


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
# We assume constant flow and constant inlet concentrations.
inlet = Inlet(component_system, 'inlet')
inlet.c = [1]
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

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'rtd_cstr')
process.cycle_time = 10 * tau
```

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.simulator import Cadet

simulator = Cadet()

simulation_results = simulator.simulate(process)
simulation_results.solution.cstr.inlet.plot()
simulation_results.solution.cstr.outlet.plot()
```

+++ {"slideshow": {"slide_type": "slide"}}
## Exercise 2: Step function in Tubular reactor

**Task:** Also analyze the system behaviour of a Tubular reactor for different input profiles (see Exercise 1).


```{code-cell} ipython3
:tags: [solution]

from CADETProcess.processModel import ComponentSystem

component_system = ComponentSystem(1)

tau = 60
V = 1e-3
Q = V/tau

# Unit Operations
from CADETProcess.processModel import Inlet, TubularReactor, Outlet

## Inlet
# We assume constant flow and constant inlet concentrations.
inlet = Inlet(component_system, 'inlet')
inlet.c = [1]
inlet.flow_rate = Q

pfr = TubularReactor(component_system, 'pfr')
pfr.length = 1
pfr.diameter = 0.1
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

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'rtd_pfr')
process.cycle_time = 10 * tau
```

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.simulator import Cadet

simulator = Cadet()

simulation_results = simulator.simulate(process)
simulation_results.solution.pfr.inlet.plot()
simulation_results.solution.pfr.outlet.plot()
```

+++ {"slideshow": {"slide_type": "slide"}}
## Bonus Exercise
Many systems can be modelled by a chain of unit operations.

```{figure} ./resources/system_chain.png
:width: 50%
:align: center
```
Try connecting combining both the CSTR with a Tubular reactor and analyze the behavior.


```{code-cell} ipython3
:tags: [solution]


```
