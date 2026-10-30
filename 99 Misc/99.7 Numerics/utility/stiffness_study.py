# -*- coding: utf-8 -*-
"""

The shared machinery behind the stiffness case studies.

Every study in this chapter has the same shape: an isolated, physically
realistic setting, one or two knobs swept over decades, and a record of what
each setting cost the time integrator. Only two things differ between studies,
the model that gets built and the knobs that get turned, so only those live in
the study modules. Everything else is here.

A study module defines a Study and hands it to make_explorer(). The notebook
cell is then a slider and nothing more:

    show = study_tools.make_explorer(my_study)
    interact(show, **show.sliders)

A study's `build` and `label_of` receive the knobs by keyword, under the names
given in `options`, so adding a second knob changes nothing but those two
functions.

The stiffness indicator reported throughout is the one this chapter uses: the
rate of the process being swept divided by the rate of the slowest other
process in the system. It is read out of utility.time_scales, so the indicator
and the rate table can never disagree.

"""

import inspect
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cadet import Cadet
from IPython.display import display

import utility.cadet_runtime as cadet_runtime
import utility.stiffness_metrics as stiffness_metrics
import utility.time_scales as time_scales
from utility.stiffness_metrics import COLORS, style_axis

TIMES_PATH = 'output/solution/SOLUTION_TIMES'


def outlet_path(unit='unit_001'):
    """Where a unit operation's outlet signal is stored."""
    return f'output/solution/{unit}/SOLUTION_OUTLET'


OUTLET_PATH = outlet_path()


@dataclass
class Study:
    """Everything that distinguishes one case study from the next.

    Attributes
    ----------
    name : string
        Short identifier, used for the directory that holds the results.
    build : callable
        Takes the knobs by keyword and returns the model configuration.
    options : list or dict
        Slider positions as (label, value) pairs. A list is one knob, named by
        `knob_name`. A dict maps each knob's name to its own list.
    label_of : callable
        Takes the knobs by keyword and returns the label for the record row.
    velocity : float
        Interstitial velocity in m/s, needed for the rate table.
    feed_concentration : float
        Representative liquid concentration, needed for the binding rate.
    knob_process : string
        Name of the row in the rate table that the knobs move, if any. When
        given, the record carries that rate and a stiffness indicator.
    reference : dict
        Knob values whose solution is drawn faintly behind every other one, as
        the limit the sweep converges to. None draws no reference.
    component_names : sequence of string
        Legend entries for the chromatogram.
    describe : callable
        Takes (metrics, outlet, **knobs) and returns extra columns for the
        record. For studies whose knob is not a rate, or that want to report
        something about the solution itself.
    knob_name : string
        Name of the single knob, when `options` is a list.
    outlet_unit : string
        Which unit operation's outlet is recorded and plotted. Flowsheets with
        more than one unit need to say which signal is the chromatogram.
    notebook : string
        Folder of the notebook this study belongs to, relative to
        `data/`. Decides where its simulation files are kept.
    defaults : dict
        Starting slider position per knob. Defaults to the value closest to
        one on a logarithmic scale, or the first option.
    """

    name: str
    build: Callable[..., Any]
    options: Union[Sequence[Tuple[str, Any]], dict]
    label_of: Callable[..., str]
    velocity: float
    feed_concentration: float = 1.0
    knob_process: Optional[str] = None
    reference: Optional[dict] = None
    component_names: Sequence[str] = field(default_factory=tuple)
    describe: Optional[Callable[..., dict]] = None
    knob_name: str = 'value'
    notebook: str = ''
    outlet_unit: str = 'unit_001'
    defaults: Optional[dict] = None

    @property
    def work_directory(self):
        """Where this study keeps its simulation files.

        Under the folder of the notebook that shows the study, so
        that everything one notebook owns sits in one place.
        """
        return os.path.join('data', self.notebook, 'sweeps',
                            self.name)

    @property
    def knob_options(self):
        """The options as a dict, whether one knob was given or several."""
        if isinstance(self.options, dict):
            return dict(self.options)

        return {self.knob_name: list(self.options)}

    def default_knobs(self):
        """Starting position of every slider."""
        chosen = {}
        for name, options in self.knob_options.items():
            if self.defaults and name in self.defaults:
                chosen[name] = self.defaults[name]
                continue

            numeric = [value for _, value in options
                       if isinstance(value, (int, float))
                       and not isinstance(value, bool) and value > 0]
            chosen[name] = (min(numeric, key=lambda v: abs(np.log10(v)))
                            if numeric else options[0][1])

        return chosen


def decade_options(start, end, step=1, tail=None):
    """Slider positions over decades of a knob, optionally with a final entry.

    Parameters
    ----------
    start, end : int
        First and last decade, inclusive.
    step : int
        Decades between neighbouring positions.
    tail : (string, value)
        An extra position appended at the end, e.g. a model variant that the
        sweep converges to. None appends nothing.

    Returns
    -------
    list of (string, float)
    """
    exponents = list(range(start, end + 1, step))
    if exponents and exponents[-1] != end:
        exponents.append(end)

    options = [(f'1e{exponent:+03d}', 10.0 ** exponent) for exponent in exponents]
    if tail is not None:
        options.append(tail)

    return options


def value_options(values, formatter='{:g}', tail=None):
    """Slider positions over an explicit list of values.

    `formatter` is either a format string or a callable turning a value into
    its label.
    """
    label = formatter if callable(formatter) else formatter.format
    options = [(label(value), value) for value in values]
    if tail is not None:
        options.append(tail)

    return options


def _tag(label):
    """Filename-safe version of a row label."""
    return re.sub(r'[^0-9A-Za-z._+-]+', '_', str(label)).strip('_')


def _is_complete(filename, unit='unit_001'):
    """True if the file holds a finished simulation, not a half written one."""
    try:
        with h5py.File(filename, 'r') as file:
            return (outlet_path(unit) in file
                    and 'meta/idas/IDAS_INTEGRATION_STEP_SIZE' in file)
    except OSError:
        return False


def simulate(study, **knobs):
    """Run one setting of a study, or load it if it has been run before.

    Returns
    -------
    (IdasMetrics, np.array, np.array)
        Integrator work, solution times, outlet concentrations.
    """
    os.makedirs(study.work_directory, exist_ok=True)
    label = study.label_of(**knobs)
    filename = os.path.join(study.work_directory, _tag(label) + '.h5')

    if _is_complete(filename, study.outlet_unit):
        measured = stiffness_metrics.measure(filename, label=label)
    else:
        model = Cadet()
        model.root = study.build(**knobs)
        model.filename = filename
        measured = stiffness_metrics.run_and_measure(model, label=label,
                                                     every_step=False)

    with h5py.File(filename, 'r') as file:
        return (measured, file[TIMES_PATH][()],
                file[outlet_path(study.outlet_unit)][()])


def rate_table(study, **knobs):
    """Characteristic rates of the model at one knob setting."""
    model = study.build(**knobs)

    return time_scales.characteristic_rates(
        model.input.model.unit_001,
        velocity=study.velocity,
        feed_concentration=study.feed_concentration)


def knob_and_slowest(study, **knobs):
    """Rate of the swept process and of the slowest other one, in 1/s.

    Returns
    -------
    (float, float, string)
        Knob rate, slowest other rate, name of that slowest process. The knob
        rate is nan when the study does not name a process.
    """
    table = rate_table(study, **knobs)

    if study.knob_process is None:
        others = table
        knob_rate = np.nan
    else:
        matches = table['process'].str.contains(study.knob_process, regex=False)
        knob_rate = (float(table.loc[matches, 'rate / s^-1'].iloc[0])
                     if matches.any() else np.inf)
        others = table[~matches]

    slowest = others.loc[others['rate / s^-1'].idxmin()]

    return knob_rate, float(slowest['rate / s^-1']), str(slowest['process'])


def plot_chromatogram(times, outlet, axis=None, reference=None, title=None,
                      component_names=()):
    """Draw a chromatogram, optionally over a faint reference."""
    if axis is None:
        _, axis = plt.subplots(figsize=(8.5, 4.0))

    if reference is not None:
        for component in range(reference.shape[1]):
            axis.plot(times / 60.0, reference[:, component],
                      color='#404040', linewidth=3.0, alpha=0.25,
                      label='limit' if component == 0 else None)

    for component in range(outlet.shape[1]):
        name = (component_names[component]
                if component < len(component_names) else f'component {component}')
        axis.plot(times / 60.0, outlet[:, component],
                  color=COLORS[component % len(COLORS)], linewidth=2.0, label=name)

    axis.set_xlabel('time / min')
    axis.set_ylabel(r'concentration / mol m$^{-3}$')
    if title:
        axis.set_title(title, fontsize=11)
    axis.legend(frameon=False, fontsize=9)
    style_axis(axis)

    return axis


def mean_bdf_order(measured):
    """BDF order averaged over the steps actually taken."""
    order = measured.bdf_order.astype(float)
    weight = measured.steps.astype(float)
    taken = weight > 0

    return float((order[taken] * weight[taken]).sum() / weight[taken].sum())


def show_setting(study, title=None, **knobs):
    """Plot one setting of a study and return the rates it is made of."""
    cadet_runtime.configure()

    if not knobs:
        knobs = study.default_knobs()

    _, times, outlet = simulate(study, **knobs)

    figure, axis = plt.subplots(figsize=(8.5, 4.0))
    plot_chromatogram(times, outlet, axis=axis, title=title,
                      component_names=study.component_names)
    figure.tight_layout()

    return rate_table(study, **knobs)


def make_explorer(study):
    """Build the slider callback for a study, together with its record.

    Returns
    -------
    callable
        Takes the knobs by keyword, draws the chromatogram and displays the
        record. Carries `sliders` ready for ipywidgets.interact, and `records`
        for everything run so far.
    """
    cadet_runtime.configure()

    records = []
    knob_options = study.knob_options

    reference = None
    if study.reference is not None:
        try:
            _, _, reference = simulate(study, **study.reference)
        except RuntimeError:
            reference = None

    def show(**knobs):
        label = study.label_of(**knobs)

        # Some studies push a model until it stops working. A failure is a
        # result, so it is recorded and shown rather than raised.
        try:
            measured, times, outlet = simulate(study, **knobs)
        except RuntimeError as error:
            reason = str(error).split(':')[-1].strip()
            records.append({'setting': label, 'BDF steps': None,
                            'result': f'FAILED - {reason}'})
            print(f'{label}: the simulation failed, {reason}')
            display(pd.DataFrame(records))
            return

        row = {'setting': label}

        if study.knob_process is not None:
            knob_rate, slowest_rate, _ = knob_and_slowest(study, **knobs)
            row['rate / s^-1'] = f'{knob_rate:.3g}'
            row['stiffness indicator'] = f'{knob_rate / slowest_rate:.3g}'

        row.update({
            'BDF steps': measured.n_steps,
            'mean BDF order': round(mean_bdf_order(measured), 2),
            'error test failures': measured.n_error_test_failures,
            'compute time / s': round(measured.compute_time, 3),
        })

        if study.describe is not None:
            row.update(study.describe(measured, outlet, **knobs))

        row['result'] = 'ok'
        records.append(row)

        figure, axis = plt.subplots(figsize=(8.5, 4.0))
        plot_chromatogram(times, outlet, axis=axis, reference=reference,
                          title=label, component_names=study.component_names)
        figure.tight_layout()
        plt.show()

        display(pd.DataFrame(records))

    # ipywidgets matches widgets to arguments by name, so advertise the study's
    # own names rather than a generic one.
    show.__signature__ = inspect.Signature([
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in knob_options
    ])
    show.__name__ = study.name

    show.records = records
    show.sliders = _sliders(knob_options, study.default_knobs())

    # A single knob keeps the plain list and scalar it has always had, so that
    # notebooks building their own slider are unaffected by the move to several.
    if len(knob_options) == 1:
        (only_name, only_options), = knob_options.items()
        show.options = only_options
        show.default = study.default_knobs()[only_name]
    else:
        show.options = knob_options
        show.default = study.default_knobs()

    return show


def _sliders(knob_options, defaults):
    """Configured selection sliders, one per knob, ready for interact()."""
    from ipywidgets import widgets

    return {
        name: widgets.SelectionSlider(options=options,
                                      value=defaults[name],
                                      description=name.replace('_', ' '),
                                      continuous_update=False)
        for name, options in knob_options.items()
    }
