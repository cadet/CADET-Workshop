---
jupytext:
  formats: ipynb,md:myst
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
inlet_valve.V = 0.5e-6
inlet_valve.c = [0]

tubing = TubularReactor(component_system, name='tubing')
tubing.length = 0.85
tubing.diameter = 0.75e-3
tubing.axial_dispersion = 2.5e-7
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
from CADETProcess.simulator import Cadet
simulator = Cadet()

simulation_results = simulator.simulate(process)
```

```{code-cell} ipython3
import numpy as np

def add_noise(solution, noise_percentage=2.5):
    solution = (solution + (np.random.random(solution.shape) - 0.5) * noise_percentage / 100 * solution.max(axis=0))
    return solution
```

```{code-cell} ipython3
target = 'uv_detector'
time = simulation_results.solution[target].outlet.time
solution = simulation_results.solution[target].outlet.solution

solution_noisy_1 = add_noise(solution)
together_1 = np.hstack((time.reshape(-1, 1), solution_noisy_1))
np.savetxt(f"{target}_1.csv", together_1, delimiter=",")

solution_noisy_2 = add_noise(solution)
together_2 = np.hstack((time.reshape(-1, 1), solution_noisy_2))
np.savetxt(f"{target}_2.csv", together_2, delimiter=",")

target = 'cond_detector'
time = simulation_results.solution[target].outlet.time
solution = simulation_results.solution[target].outlet.solution

solution_noisy_1 = add_noise(solution)
together_1 = np.hstack((time.reshape(-1, 1), solution_noisy_1))
np.savetxt(f"{target}_1.csv", together_1, delimiter=",")

solution_noisy_2 = add_noise(solution)
together_2 = np.hstack((time.reshape(-1, 1), solution_noisy_2))
np.savetxt(f"{target}_2.csv", together_2, delimiter=",")
```
