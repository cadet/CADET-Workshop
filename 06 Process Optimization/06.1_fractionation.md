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

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

# Product Fractionation

- Key information for evaluating the separation performance of a chromatographic process: Amounts of the target components in the collected product fractions.
- Evaluation of chromatograms $c_{i,k}\left(t\right)$ at the outlet(s) of the process must be evaluated.

$$
m_{i} = \sum_{k=1}^{n_{chrom}} \sum_{j=1}^{n_{frac, k}^{i}}\int_{t_{start, j}}^{t_{end, j}} Q_k(t) \cdot c_{i,k}(t) dt,\\
$$

where $n_{frac, k}^{i}$ is the number of fractions considered for component $i$ in chromatogram $k$, and $n_{chrom}$ is the number of chromatograms that is evaluated.

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

## Key Performance Indicators (KPI)

### Productivity
$$
PR_{i} = \frac{m_i}{V_{solid} \cdot \Delta t_{cycle}},\\
$$
with $V_{solid}$: volume of stationary phase.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

### Recovery Yield
$$
Y_{i} = \frac{m_i}{m_{feed, i}},\\
$$
with $m_{feed}$: injected amount of mixture.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

### Eluent Consumption
$$
EC_{i} = \frac{V_{solvent}}{m_i},\\
$$
with $V_{solvent}$: solvent used during a cycle.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

### Purity

$$
PU_{i} = \frac{m_{i}^{i}}{\sum_{l=1}^{n_{comp}} m_{l}^{i}},\\
$$
where $n_{comp}$ is the number of mixture components and $m_{l}^{i}$ is the mass of component $l$ in target fraction $i$.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Fractionator

In **CADET-Process**, the `fractionation` module provides methods to calculate these performance indicators.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

The `Fractionator` allows slicing the solution and pool fractions for the individual components.
It enables evaluating multiple chromatograms at once and multiple fractions per component per chromatogram.

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

The most basic strategy is to manually set all fractionation times manually.
To demonstrate the strategy, a process from the [examples collection](https://cadet-process.readthedocs.io/en/latest/examples/batch_elution/process.html) is used.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
from examples.batch_elution.process import process
```

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

To enable the calculation of the process parameters, it is necessary to specify which of the inlets should be considered for the feed and eluent consumption.
Moreover, the outlet(s) which are used for evaluation need to be defined.

```
process.flow_sheet.add_feed_inlet('feed')
process.flow_sheet.add_eluent_inlet('eluent')
process.flow_sheet.add_chromatogram_outlet('outlet')
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
from CADETProcess.simulator import Cadet
process_simulator = Cadet()
simulation_results = process_simulator.simulate(process)
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

For reference, this is the chromatogram at the outlet that needs to be fractionated:

```{code-cell} ipython3
---
editable: true
render:
  figure:
    caption: 'Concentration profile at column outlet.

      '
    name: column_outlet
slideshow:
  slide_type: ''
tags: [solution]
---
_ = simulation_results.solution.outlet.outlet.plot()
```

+++ {"slideshow": {"slide_type": "slide"}, "editable": true}

After import, the `Fractionator` is instantiated with the simulation results.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [solution]
---
from CADETProcess.fractionation import Fractionator
fractionator = Fractionator(simulation_results)
```

+++ {"slideshow": {"slide_type": "fragment"}, "editable": true}

To add a fractionation event, the following arguments need to be provided:
- `event_name`: Name of the event.
- `target`: Pool to which fraction is added. `-1` indicates waste.
- `time`: Time of the event
- `chromatogram`: Name of the chromatogram. Optional if only one outlet is set as `chromatogram_sink`.

Here, component $A$ seems to have sufficient purity between $5 \colon 00~min$ and $5 \colon 45~min$ and component $B$ between $6 \colon 30~min$ and $9 \colon 00~min$.

```{code-cell} ipython3
:tags: [solution]

fractionator.add_fractionation_event('start_A', 0, 5*60, 'outlet')
fractionator.add_fractionation_event('end_A', -1, 5.75*60)
fractionator.add_fractionation_event('start_B', 1, 6.5*60)
fractionator.add_fractionation_event('end_B', -1, 9*60)
```

+++ {"slideshow": {"slide_type": "slide"}}

The `performance` object of the `Fractionator` contains the parameters:

```{code-cell} ipython3
:tags: [solution]

print(fractionator.performance)
```

+++ {"slideshow": {"slide_type": "fragment"}}

With these fractionation times, the both component fractions reach a purity of $99.7~\%$, and $97.2~\%$  respectively.
The recovery yields are $65.2~\%$ and $63.4~\%$.

+++ {"slideshow": {"slide_type": "slide"}}

The chromatogram can be plotted with the fraction times overlaid:

```{code-cell} ipython3
:tags: [solution]

_ = fractionator.plot_fraction_signal()
```

+++ {"slideshow": {"slide_type": "slide"}}

## Optimization of Fractionation Times
- The `fractionation` module provides tools to automatically determines optimal cut times.
- By default, the mass of the components is maximized under purity constraints.
- Different purity requirements can be specified for each component

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.fractionation import FractionationOptimizer
fractionation_optimizer = FractionationOptimizer()
fractionation_optimizer.optimizer.rhobeg = 1e-3  # more on that later!
```

+++ {"slideshow": {"slide_type": "slide"}}

To automatically optimize the fractionation times, pass the simulation results to the `optimize_fractionation` function.

```{code-cell} ipython3
:tags: [solution]

fractionator = fractionation_optimizer.optimize_fractionation(simulation_results, purity_required=[0.95, 0])
```

+++ {"slideshow": {"slide_type": "fragment"}}

The results are stored in a `Performance` object.

```{code-cell} ipython3
:tags: [solution]

print(fractionator.performance)
```

+++ {"slideshow": {"slide_type": "slide"}}

The chromatogram can also be plotted with the fraction times overlaid:

```{code-cell} ipython3
:tags: [solution]

_ = fractionator.plot_fraction_signal()
```

+++ {"slideshow": {"slide_type": "slide"}}

For comparison, this is the results if only the second component is relevant:

```{code-cell} ipython3
:tags: [solution]

fractionator = fractionation_optimizer.optimize_fractionation(simulation_results, purity_required=[0, 0.95])

print(fractionator.performance)
_ = fractionator.plot_fraction_signal()
```

+++ {"slideshow": {"slide_type": "slide"}}

But of course, also both components can be valuable.
Here, the required purity is also reduced to demonstrate that overlapping fractions are automatically avoided by internally introducing linear constraints.

```{code-cell} ipython3
:tags: [solution]

fractionator = fractionation_optimizer.optimize_fractionation(simulation_results, purity_required=[0.8, 0.8])

print(fractionator.performance)
_ = fractionator.plot_fraction_signal()
```

+++ {"slideshow": {"slide_type": "slide"}}

## Alternative Objectives

- define function that that takes a `Performance` as an input.
- Here, also consider concentration of the fraction.

+++ {"slideshow": {"slide_type": "fragment"}}

```{note}
As previously mentioned, `COBYLA` only handles single objectives.
Hence, a `RankedPerformance` is used which transforms the `Performance` object by adding a weight $w_i$ to each component.
```

$$
p = \frac{\sum_i^{n_{comp}}w_i \cdot p_i}{\sum_i^{n_{comp}}(w_i)}
$$

+++ {"slideshow": {"slide_type": "fragment"}}

It is also important to remember that by convention, objectives are minimized.
Since in this example, the product of mass and concentration should be maximized, the value of the objective function is multiplied by $-1$.

```{code-cell} ipython3
:tags: [solution]

from CADETProcess.performance import RankedPerformance
ranking = [1, 1]

def alternative_objective(performance):
    performance = RankedPerformance(performance, ranking)
    return - performance.mass * performance.concentration

fractionator = fractionation_optimizer.optimize_fractionation(
    simulation_results, purity_required=[0.95, 0.95],
    obj_fun=alternative_objective,
)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: slide
tags: [solution]
---
print(fractionator.performance)
_ = fractionator.plot_fraction_signal()
```

+++ {"slideshow": {"slide_type": "notes"}}

The resulting fractionation times show that in this case, it is advantageous to discard some slices of the peak in order not to dilute the overall product fraction.
