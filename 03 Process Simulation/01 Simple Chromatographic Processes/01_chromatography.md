---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region user_expressions=[] -->
# Simple Chromatographic Processes

Chromatography is a thermal separation technique for the separation of mixtures dissolved in a fluid called the mobile phase, which carries the gasous or liquid mixture through a structure holding another material, called the stationary phase.
The various constituents of the mixture travel at different speeds, causing them to separate.
The separation itself is based on different partitioning between the mobile and stationary phases.

Different mechanisms can be used for the separation, e.g. adsorption, ion exchange, size exclusion to achieve high purities separations of multicomponent mixtures.
For each mechanism, various stationary phases are available.

Generally, chromatographic models are used for optimization of preparative processes.
In contrast to analytical chromatography, which serves to identify or quantify analytes in a mixture, the purpose of preparative chromatography, is the isolation and purification of sufficient quantities of a specific substance for further use.

For modelling these processes, we have to combine all of the techniques we learnt in the previous lessons:
- Configure unit operations models.
- Associate adsorption models with unit operations.
- Generate dynamic inlet profiles.
- Chemical reactions (if required)
<!-- #endregion -->

<!-- #region user_expressions=[] -->
## Example 1: Dextran pulse

In this exercise, we will consider the following system:

![RTD_PFR.png](attachment:b4ae80b3-1a64-4a16-9473-4454ab605e9c.png)

Before considering 'real' chromatography, we will model a simple experiment meant to find the porosity and axial dispersion of a column by sending a dextran pulse through the column.
There will be no binding or pore penetration considered yet.


For the column, assume the following parameters which are usually provided by the manufacturer (or can be measured):
- length: $0.1~m$
- diameter: $0.01~m$
- particle radius: $4.5 \cdot 10^{-5}~m$
- particle porosity: $0.33$

Moreover, since Dextran does not penetrate pores, the film diffusion coefficient can be set to $0~m \cdot s^{-1}$.

Finally, bed porosity and axial dispersion need to be specified.
Usually, these parameters will be estimated using an inverse method (see later tutorials).
For now, assume the following values:
- bed porosity: $0.37$
- axial dispersion: $2.0 \cdot 10^{-7}~m^2 \cdot s^{-1}$
<!-- #endregion -->

```python
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Inlet, LumpedRateModelWithPores, Outlet
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process

component_system = ComponentSystem(['Dextran'])

inlet = Inlet(component_system, 'inlet')
inlet.flow_rate = 1e-6/60

column = LumpedRateModelWithPores(component_system, 'column')
column.length = 0.1
column.diameter = 0.01
column.bed_porosity = 0.37
column.particle_radius = 4.5e-5
column.particle_porosity = 0.33

column.axial_dispersion = 2.0e-7
column.film_diffusion = [0.0]

outlet = Outlet(component_system, 'outlet')

flow_sheet = FlowSheet(component_system)
flow_sheet.add_unit(inlet)
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet)

flow_sheet.add_connection(inlet, column)
flow_sheet.add_connection(column, outlet)
```

<!-- #region user_expressions=[] -->
For the injection, we need to Introduce two sections.
In the first section, which lasts $50~s$, the concentration of Dextran at the `INLET` is $1.0~mM$, afterwards it is $0.0~mM$.
The flow rate is a constant $1~mL \cdot min^{-1}$.

![dextran_inlet.png](attachment:02683823-b0ae-44f5-a2f6-16a9ed6a8be6.png)
<!-- #endregion -->

```python
process = Process(flow_sheet, 'dextran')
process.cycle_time = 600

process.add_event('inject_on', 'flow_sheet.inlet.c', [1.0], 0)
process.add_event('inject_off', 'flow_sheet.inlet.c', [0.0], 50.0)

from CADETProcess.simulator import Cadet
simulator = Cadet()

simulation_results = simulator.simulate(process)

_ = simulation_results.solution.column.outlet.plot()
```

<!-- #region user_expressions=[] -->
## Example 2: Multi Component Langmuir Separation

Now, we will use the same system, but add `Langmuir` model to the column with two components using batch elution chromatography.
This process is often used for the purification of small molecules like amino acids or sugars.

For the Langmuir isotherm, use the following parameters:
- adsorption rate: $[0.02, 0.03]~m^3 mol^{-1} s^{-1}$
- desorption rate: $[1, 1]~s^{-1}$
- binding capacity: $[100, 100]~mM$

<!-- #endregion -->

```python
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import Langmuir
from CADETProcess.processModel import (
    Inlet, LumpedRateModelWithPores, Outlet
)
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process

# Component System
component_system = ComponentSystem()
component_system.add_component('A')
component_system.add_component('B')

# Binding Model
binding_model = Langmuir(component_system, name='langmuir')
binding_model.is_kinetic = False
binding_model.adsorption_rate = [0.02, 0.03]
binding_model.desorption_rate = [1, 1]
binding_model.capacity = [100, 100]

# Unit Operations
feed = Inlet(component_system, name='feed')
feed.c = [10, 10]

eluent = Inlet(component_system, name='eluent')
eluent.c = [0, 0]

column = LumpedRateModelWithPores(component_system, 'column')
column.binding_model = binding_model
column.length = 0.1
column.diameter = 0.01
column.bed_porosity = 0.37
column.particle_radius = 4.5e-5
column.particle_porosity = 0.33
column.axial_dispersion = 2.0e-7
column.film_diffusion = [1e-4, 1e-4]

outlet = Outlet(component_system, name='outlet')

# flow sheet
flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(feed)
flow_sheet.add_unit(eluent)
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet)

flow_sheet.add_connection(feed, column)
flow_sheet.add_connection(eluent, column)
flow_sheet.add_connection(column, outlet)
```

<!-- #region user_expressions=[] -->
Again, we create two sections to model the injections.
<!-- #endregion -->

```python
# Process
process = Process(flow_sheet, 'batch elution')

## Create Events and Durations
Q = 1e-6/60
process.add_event('feed_on', 'flow_sheet.feed.flow_rate', Q, 0)
process.add_event('feed_off', 'flow_sheet.feed.flow_rate', 0.0, 60)

process.add_event('eluent_off', 'flow_sheet.eluent.flow_rate', 0.0, 0)
process.add_event('eluent_on', 'flow_sheet.eluent.flow_rate', Q, 60)

## Set process times
process.cycle_time = 1200
```

<!-- #region user_expressions=[] -->
### A Note on Event Dependencies

Often, multiple `Events` happen simulateneously.
Here, for example, when the feed is turned off, the eluent also needs to be switched on.
To eliminate the need to manually change all event times, dependencies can be specified.
<!-- #endregion -->

```python
# Process
process = Process(flow_sheet, 'batch elution')

## Create Events and Durations
Q = 1e-6/60
process.add_event('feed_on', 'flow_sheet.feed.flow_rate', Q, 0)
process.add_event('feed_off', 'flow_sheet.feed.flow_rate', 0.0, 60)

process.add_event('eluent_on', 'flow_sheet.eluent.flow_rate', Q)
process.add_event_dependency('eluent_on', ['feed_off'])
process.add_event('eluent_off', 'flow_sheet.eluent.flow_rate', 0.0)
process.add_event_dependency('eluent_off', ['feed_on'])

## Set process times
process.cycle_time = 1200
```

```python
process.dependent_events
```

```python
from CADETProcess.simulator import Cadet
process_simulator = Cadet()

simulation_results = process_simulator.simulate(process)
_ = simulation_results.solution.column.outlet.plot()
```

<!-- #region user_expressions=[] -->
## Example 3: Load wash elute with steric mass action law binding

The [steric mass action model](https://cadet.github.io/master/modelling/binding/steric_mass_action.html#steric-mass-action-model) takes charges of the molecules into account and is, thus, often used in ion-exchange chromatography.
Each component has a characteristic charge $\nu$ that determines the number of available binding sites $\Lambda$ (ionic capacity) used up by a molecule.
Due to the molecule’s shape, some additional binding sites (steric shielding factor $\sigma$) may be shielded from other molecules and are not available for binding.

$$\frac{\mathrm{d} q_i}{\mathrm{d} t} = k_{a,i} c_{p,i}\bar{q}_0^{\nu_i} - k_{d,i} q_i c_{p,0}^{\nu_i}$$

where $c_{p,0}$ denotes the mobile phase salt concentration, and

$$\bar{q}_0 = \Lambda - \sum_{j=1}^{N_{\text{comp}} - 1} \left( \nu_j + \sigma_j \right) q_j$$

is the number of available binding sites which is related to the number of bound salt ions. Using the parameter transformation

$$k_{a,i} = \tilde{k}_{a,i} q_{\text{ref}}^{-\nu_i}$$

$$k_{d,i} = \tilde{k}_{d,i} c_{\text{ref}}^{-\nu_i}$$

we obtain the modified model equation:

$$\frac{\mathrm{d} q_i}{\mathrm{d} t} = \tilde{k}_{a,i} c_{p,i} \left(\frac{\bar{q}_0}{q_{\text{ref}}}\right)^{\nu_i} - \tilde{k}_{d,i} q_i \left(\frac{c_{p,0}}{c_{\text{ref}}}\right)^{\nu_i}$$

This transformation serves as a (partial) nondimensionalization of the adsorption and desorption rates.

The basic goal is to have $\left(\frac{\bar{q}_0}{q_{\text{ref}}}\right) \leq 1$ and $\left(\frac{c_{p,0}}{c_{\text{ref}}}\right) \leq 1$

Recommended choices for $c_{\text{ref}}$ are the average or maximum inlet concentration of the mobile phase modifier $c_0$, and for $q_{\text{ref}}$ the ionic capacity $\Lambda$.
Note that setting the reference concentrations to ${1.0}$ each results in the original binding model.

<div class="alert alert-info">

**Note:**

From a practical perspective, modern resins have a very high capacity and large proteins can can have a very high charactistic charge.
If the concentration is not normalized, the system is often numerically unstable.
It may run slowly or not at all.

</div>

In this example, we will look at a typical process for protein purification.
First, protein which is in a mixture with salt is loaded on the column and binds to the resin.
Then, the column is washed with a lower concentrated salt solution.
Finally, the protein is eluted by adding a linear salt gradient.

![lwe_inlet.png](attachment:509429d7-700f-4c81-be24-b74abfe2ff84.png)
<!-- #endregion -->

<!-- #region user_expressions=[] -->
First, define the `ComponentSystem` and the parameters for the `StericMassAction` model.
As mentioned earlier, consider a reference concentration in the pore for numeric purposes.
From the [manual](https://cadet.github.io/master/modelling/binding/reference_concentrations.html?highlight=reference):
> Recommended choices for the reference concentration are the average or maximum inlet concentration of the mobile phase modifier for `c0`, and the ionic capacity Λ for `qref`.

***Note:*** In CADET-Process, `lambda` is a reserved keyword in Python.
In such occasions, it is common to suffix variable names with an underscore.
<!-- #endregion -->

```python
import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import StericMassAction
from CADETProcess.processModel import Inlet, GeneralRateModel, Outlet
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process

# Component System
component_system = ComponentSystem()
component_system.add_component('Salt')
component_system.add_component('Protein')

# Binding Model
binding_model = StericMassAction(component_system, name='SMA')
binding_model.is_kinetic = True
binding_model.adsorption_rate = [0.0, 0.3]
binding_model.desorption_rate = [0.0, 1.5]
binding_model.characteristic_charge = [0.0, 7.0]
binding_model.steric_factor = [0.0, 50.0]
binding_model.capacity = 225.0
```

<!-- #region user_expressions=[] -->
Then, we define the system of unit operations.
<!-- #endregion -->

```python
# Unit Operations
inlet = Inlet(component_system, name='inlet')
inlet.flow_rate = 2.88e-8

column = GeneralRateModel(component_system, name='column')
column.binding_model = binding_model

column.length = 0.25
column.diameter = 0.0115
column.bed_porosity = 0.37
column.particle_radius = 4.5e-5
column.particle_porosity = 0.33
column.axial_dispersion = 2.0e-7
column.film_diffusion = [2.0e-5, 2.0e-7]
column.pore_diffusion = [7e-5, 1e-9]
column.surface_diffusion = [0.0, 0.0]

outlet = Outlet(component_system, name='outlet')

# Flow Sheet
flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(inlet)
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet)

flow_sheet.add_connection(inlet, column)
flow_sheet.add_connection(column, outlet)
```

<!-- #region user_expressions=[] -->
The protein is loaded for $7500 s$, then there is a wash step, which takes $2000 s$, and the gradient takes another $5500 s$.
<!-- #endregion -->

```python
# Process
process = Process(flow_sheet, 'lwe')
process.cycle_time = 15000.0

## Create Events and Durations
wash_start = 7500.0
gradient_start = 9500.0
concentration_difference = np.array([500.0, 0.0]) - np.array([70.0, 0.0])
gradient_duration = process.cycle_time - gradient_start
gradient_slope = concentration_difference/gradient_duration

_ = process.add_event('load', 'flow_sheet.inlet.c', [180.0, 0.1])
_ = process.add_event('wash', 'flow_sheet.inlet.c', [70.0, 0.0], wash_start)
_ = process.add_event(
    'grad_start',
    'flow_sheet.inlet.c',
    [[70.0, gradient_slope[0]], [0, gradient_slope[1]]],
    gradient_start
)
```

<!-- #region user_expressions=[] -->
Finally, we set the initial conditions of the column.
We assume, that in the beginning of the process, the stationary phase is fully loaded with  salt.
<!-- #endregion -->

```python
column.c = [180, 0]
column.q = [binding_model.capacity, 0]
```

<!-- #region user_expressions=[] -->
Now, we run the simulation and plot the results. Because the concentration ranges are very different, we use different scales for both components.
<!-- #endregion -->

```python
from CADETProcess.simulator import Cadet
process_simulator = Cadet()

simulation_results = process_simulator.simulate(process)

from CADETProcess.plotting import SecondaryAxis
sec = SecondaryAxis()
sec.components = ["Salt"]
sec.y_label = '$c_{salt}$'

_ = simulation_results.solution.column.outlet.plot(secondary_axis=sec)
```
