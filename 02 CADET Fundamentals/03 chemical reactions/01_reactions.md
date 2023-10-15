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

# Chemical Reactions

$$
\require{mhchem}
$$

Since version 4, it is possible to model chemical reactions with CADET using mass action law type reactions (see [Reaction models](https://cadet.github.io/master/modelling/reactions.html#reaction-models)).
The mass action law states that the speed of a reaction is proportional to the product of the concentrations of their reactants.

In CADET-Process, a reaction module was implemented to facilitate the setup of these reactions.
There are two different classes: the `MassActionLaw` which is used for bulk phase reactions, as well as `MassActionLawParticle` which is specifically designed to model reactions in particle pore phase.

In this tutorial, we're going to learn how to setup:
- Forward Reactions
- Equilibrium Reactions

+++ {"slideshow": {"slide_type": "slide"}}
## Forward Reactions
As a simple example, consider the following system:

$$
\ce{1 A ->[k_{AB}] 1 B}
$$


+++
First, initiate a `ComponentSystem` with components `A` and `B`.


```{code-cell} ipython3
:tags: [solution]

from CADETProcess.processModel import ComponentSystem
component_system = ComponentSystem(['A', 'B'])
```

+++ {"slideshow": {"slide_type": "slide"}}
Then, configure the `MassActionLaw` reaction model.
To instantiate it, pass the `ComponentSystem`.
Then, add the reaction using the `add_reaction` method.
The following arguments are expected:
- indices: The indices of the components that take part in the reaction (useful for bigger systems)
- stoichiometric coefficients in the order of the indices
- forward reaction rate
- backward reaction rate


```{code-cell} ipython3
:tags: [solution]

from CADETProcess.processModel import MassActionLaw
reaction_system = MassActionLaw(component_system)
reaction_system.add_reaction(
    indices=[0,1],
    coefficients=[-1, 1],
    k_fwd=0.1,
    k_bwd=0
)
```

+++ {"slideshow": {"slide_type": "slide"}}
To demonstrate this reaction, a `Cstr` is instantiated and the reaction is added to the tank.
Moreover, the initial conditions are set.
In principle, the `Cstr` supports reactions in bulk and particle pore phase.
Since the porosity is $1$ by default, only the bulk phase is considered.



```{code-cell} ipython3
:tags: [solution]

from CADETProcess.processModel import Cstr

reactor = Cstr(component_system, 'reactor')
reactor.bulk_reaction_model = reaction_system
reactor.V = 1e-6
reactor.c = [1.0, 0.0]
```

+++ {"slideshow": {"slide_type": "slide"}}
Now, the reactor is added to a `FlowSheet` and a `Process` is set up.
Here, the `FlowSheet` only consists of a single `Cstr`, and there are no `Events` in the process.


```{code-cell} ipython3
:tags: [solution]

from CADETProcess.processModel import FlowSheet
flow_sheet = FlowSheet(component_system)
flow_sheet.add_unit(reactor)

from CADETProcess.processModel import Process
process = Process(flow_sheet, 'reaction_demo')
process.cycle_time = 100
```

+++ {"slideshow": {"slide_type": "slide"}}
After simulation, the results can be plotted:


```{code-cell} ipython3
:tags: [solution]

from CADETProcess.simulator import Cadet
simulator = Cadet()
sim_results = simulator.run(process)
_ = sim_results.solution.reactor.outlet.plot()
```

+++ {"slideshow": {"slide_type": "slide"}}
## Equilibrium Reactions
It is also possible to consider equilibrium reactions where the product can react back to the educts.

$$
\ce{ 2 A <=>[k_{AB}][k_{BA}] B}
$$


+++ {"slideshow": {"slide_type": "slide"}}
Here, the same units, flow sheet, and process are reused which were defined above.


```{code-cell} ipython3
:tags: [solution]

reaction_system = MassActionLaw(component_system)
reaction_system.add_reaction(
    indices=[0,1],
    coefficients=[-2, 1],
    k_fwd=0.2,
    k_bwd=0.1
)

reactor.bulk_reaction_model = reaction_system
```

+++ {"slideshow": {"slide_type": "slide"}}
After simulation, the results can be plotted:


```{code-cell} ipython3
:tags: [solution]

sim_results = simulator.run(process)
_ = sim_results.solution.reactor.outlet.plot()
```
