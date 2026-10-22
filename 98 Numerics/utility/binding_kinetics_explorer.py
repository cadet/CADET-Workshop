# -*- coding: utf-8 -*-
"""

Interactive exploration of the binding kinetics in the isolated three component
setting of utility.setting_Col1D_langmuir_realistic.

The notebook cell is a slider. Everything it needs lives here: running the
simulation, drawing the chromatogram against the rapid-equilibrium limit, and
keeping the record of what each setting cost.

The stiffness indicator reported is the one this chapter uses: the rate of the
binding divided by the rate of the slowest other process in the system, i.e.
how far apart the fastest and the slowest thing in the model are.

"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cadet import Cadet
from IPython.display import display

import utility.cadet_runtime as cadet_runtime
import utility.setting_Col1D_langmuir_realistic as case
import utility.stiffness_metrics as stiffness_metrics
import utility.time_scales as time_scales
from utility.stiffness_metrics import COLORS, style_axis

N_ELEMENTS = 25                       # meets 1 % of the feed peak, see the notebook
WORK_DIRECTORY = os.path.join('sweeps', 'binding_kinetics_explorer')
EQUILIBRIUM_LABEL = 'rapid equilibrium'

_COMPONENT_NAMES = ('weakly bound', 'intermediate', 'strongly bound')


def rate_table(n_elements=N_ELEMENTS, kinetics_factor=1.0):
    """Characteristic rates of the setting.

    Two rows of the table, convection and dispersion across one cell, are set
    by the discretization, so the grid has to be passed in rather than assumed.

    Parameters
    ----------
    n_elements : int
        Number of finite volume cells.
    kinetics_factor : float
        Common factor on ka and kd.

    Returns
    -------
    pd.DataFrame
    """
    model = case.get_model(n_elements=n_elements,
                           kinetics_factor=kinetics_factor)

    return time_scales.characteristic_rates(
        model.input.model.unit_001,
        velocity=case.interstitial_velocity(),
        feed_concentration=case.LOADING)


def slowest_other_process(n_elements=N_ELEMENTS):
    """Rate and name of the slowest process that is not the binding.

    Parameters
    ----------
    n_elements : int
        Number of finite volume cells.

    Returns
    -------
    (float, string)
    """
    table = rate_table(n_elements=n_elements)
    others = table[~table['process'].str.contains('binding')]
    slowest = others.loc[others['rate / s^-1'].idxmin()]

    return float(slowest['rate / s^-1']), str(slowest['process'])


def simulate(kinetics_factor, is_kinetic=True, n_elements=N_ELEMENTS):
    """Run one setting and return its integrator work, times and outlet."""
    os.makedirs(WORK_DIRECTORY, exist_ok=True)

    tag = (f'kin_{kinetics_factor:g}' if is_kinetic else 'equilibrium')
    filename = os.path.join(WORK_DIRECTORY, f'{tag}_n{n_elements}.h5')

    model = Cadet()
    model.root = case.get_model(n_elements=n_elements,
                                kinetics_factor=kinetics_factor,
                                is_kinetic=is_kinetic)
    model.filename = filename
    measured = stiffness_metrics.run_and_measure(model, label=tag,
                                                 every_step=False)

    with h5py.File(filename, 'r') as file:
        outlet = file['output/solution/unit_001/SOLUTION_OUTLET'][()]
        times = file['output/solution/SOLUTION_TIMES'][()]

    return measured, times, outlet


def plot_chromatogram(times, outlet, axis=None, reference=None, title=None):
    """Draw the three component chromatogram, optionally over a reference."""
    if axis is None:
        _, axis = plt.subplots(figsize=(8.5, 4.0))

    if reference is not None:
        for component in range(reference.shape[1]):
            axis.plot(times / 60.0, reference[:, component],
                      color='#404040', linewidth=3.0, alpha=0.25,
                      label='rapid equilibrium' if component == 0 else None)

    for component in range(outlet.shape[1]):
        axis.plot(times / 60.0, outlet[:, component],
                  color=COLORS[component % len(COLORS)], linewidth=2.0,
                  label=_COMPONENT_NAMES[component])

    axis.set_xlabel('time / min')
    axis.set_ylabel(r'concentration / mol m$^{-3}$')
    if title:
        axis.set_title(title, fontsize=11)
    axis.legend(frameon=False, fontsize=9)
    style_axis(axis)

    return axis


def show_setting(n_elements=N_ELEMENTS):
    """Plot the base case and return the rates it is made of.

    Parameters
    ----------
    n_elements : int
        Number of finite volume cells.

    Returns
    -------
    pd.DataFrame
    """
    cadet_runtime.configure()
    _, times, outlet = simulate(1.0, n_elements=n_elements)

    figure, axis = plt.subplots(figsize=(8.5, 4.0))
    plot_chromatogram(times, outlet, axis=axis,
                      title='three proteins, competitive Langmuir, '
                            f'{n_elements} finite volume cells')
    figure.tight_layout()

    return rate_table(n_elements=n_elements)


def make_explorer(n_elements=N_ELEMENTS, start=-6, end=14, step=1):
    """Build the slider callback, together with the record it accumulates.

    A plain function rather than a class, because ipywidgets.interact inspects
    the callable's signature and name.

    Parameters
    ----------
    n_elements : int
        Number of finite volume cells, held fixed across the sweep.
    start, end : int
        First and last decade of the kinetics factor, inclusive.
    step : int
        Decades between neighbouring slider positions.

    Returns
    -------
    callable
        Takes the slider value and returns nothing, drawing the chromatogram
        and displaying the record. It carries three attributes for the widget
        to read: `options` for the slider, `default` for its starting position,
        and `records` for everything run so far.
    """
    cadet_runtime.configure()

    records = []
    slowest_rate, _ = slowest_other_process(n_elements=n_elements)
    _, _, equilibrium = simulate(1.0, is_kinetic=False, n_elements=n_elements)

    def show(kinetics):
        is_kinetic = kinetics is not None
        factor = kinetics if is_kinetic else 1.0

        measured, times, outlet = simulate(factor, is_kinetic=is_kinetic,
                                           n_elements=n_elements)

        if is_kinetic:
            rate = case.binding_rate(factor)
            indicator = rate / slowest_rate
            label = f'binding rate {rate:.3g} / s'
        else:
            rate = np.inf
            indicator = np.inf
            label = EQUILIBRIUM_LABEL

        records.append({
            'setting': label,
            'binding rate / s^-1': f'{rate:.3g}',
            'stiffness indicator': f'{indicator:.3g}',
            'BDF steps': measured.n_steps,
            'mean BDF order': round(_mean_order(measured), 2),
            'compute time / s': round(measured.compute_time, 3),
        })

        figure, axis = plt.subplots(figsize=(8.5, 4.0))
        plot_chromatogram(times, outlet, axis=axis, reference=equilibrium,
                          title=label)
        figure.tight_layout()
        plt.show()

        display(pd.DataFrame(records))

    show.records = records
    show.options = slider_options(start=start, end=end, step=step)

    # Start on the position closest to a factor of one, so that an arbitrary
    # range cannot hand the widget a value it does not offer.
    factors = [value for _, value in show.options if value is not None]
    show.default = min(factors, key=lambda value: abs(np.log10(value)))

    return show


def _mean_order(measured):
    """BDF order averaged over the steps actually taken."""
    order = measured.bdf_order.astype(float)
    weight = measured.steps.astype(float)
    taken = weight > 0

    return float((order[taken] * weight[taken]).sum() / weight[taken].sum())


def slider_options(start=-6, end=14, step=1):
    """Slider positions over the kinetics factor, then rapid equilibrium.

    Parameters
    ----------
    start, end : int
        First and last decade of the kinetics factor, inclusive.
    step : int
        Decades between neighbouring positions.

    Returns
    -------
    list of (string, float or None)
        The last entry carries None, which selects rapid equilibrium.
    """
    exponents = list(range(start, end + 1, step))
    if exponents and exponents[-1] != end:
        exponents.append(end)

    options = [(f'1e{exponent:+03d}', 10.0 ** exponent)
               for exponent in exponents]
    options.append((EQUILIBRIUM_LABEL, None))

    return options
