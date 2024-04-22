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

# Fractionation - Exercise

Take the following process from [this example](https://cadet-process.readthedocs.io/en/stable/examples/load_wash_elute/lwe_flow_rate.html).

- Simulate the process.
- Plot the outlet concentration. Use a secondary axis for the `Salt` component.
- Instantiate a `Fractionator` and manually set fractionation times to purify component `C`.
- Plot the results and analyse the `performance`, especially the purity.
- Exclude the `Salt` component from the fractionation and analyse the `performance` again.
- Use a `FractionationOptimizer` to automatically determine adequate cut times. Play around with different purity requirements and objecives.

```{code-cell} ipython3
from examples.load_wash_elute.lwe_flow_rate import process
```

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.simulator import Cadet

simulator = Cadet()

simulation_results = simulator.simulate(process)

from CADETProcess.plotting import SecondaryAxis
sec = SecondaryAxis()
sec.components = ['Salt']
sec.y_label = '$c_{salt}$'

simulation_results.solution.outlet.inlet.plot(secondary_axis=sec)
```

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.fractionation import Fractionator
fractionator = Fractionator(simulation_results)

fractionator.add_fractionation_event('start_C', 2, 5*60)
fractionator.add_fractionation_event('end_C', -1, 8*60)

fractionator.plot_fraction_signal(secondary_axis=sec)
fractionator.performance
```

```{code-cell} ipython3
:tags: [solution]

fractionator = Fractionator(simulation_results, components=['A', 'B', 'C'])

fractionator.add_fractionation_event('start_C', 0, 5*60)
fractionator.add_fractionation_event('end_C', -1, 8*60)

fractionator.plot_fraction_signal()
fractionator.performance
```

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.fractionation import FractionationOptimizer
fractionation_optimizer = FractionationOptimizer()

fractionator = fractionation_optimizer.optimize_fractionation(
    simulation_results,
    components=['A', 'B', 'C'],
    purity_required=[0, 0, 0.95]
)
print(fractionator.performance)
_ = fractionator.plot_fraction_signal()
```
