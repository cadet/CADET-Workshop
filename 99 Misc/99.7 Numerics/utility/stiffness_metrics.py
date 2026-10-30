# -*- coding: utf-8 -*-
"""

This script implements evaluation functionalities related to the work performed
by the time integrator (IDAS) in CADET simulations.

Stiffness is paid for in integrator work: many small time steps, repeated order
restarts and failed error tests. CADET reports this work when the simulation is
run with WRITE_IDAS_META = 1, see enable_idas_meta().

Two resolutions are available:

    - With USER_SOLUTION_TIMES set, the integrator data is sampled at the
      requested output times. Cheap, and enough to see where the time goes.
    - Without USER_SOLUTION_TIMES, every step taken by the integrator is
      reported. Required to resolve the order restarts at section transitions,
      which happen on a time scale far below any sensible output grid.

Note that CADET re-initializes the integrator at every section transition, so
the cumulative counters restart from zero in each section. The scalar
IDAS_NTIMESTEPS therefore only holds the step count of the LAST section; use
IdasMetrics.n_steps for the total over all sections.

IDAS_N_LINSOLVER_SETUP_CALLS is not read here: CADET hands IDAS its own linear
solver, so IDAS never counts a setup and the field is identically zero.

"""

from dataclasses import dataclass, field

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito, a colorblind safe qualitative palette. Assigned in this fixed
# order, never cycled: a sixth curve means the figure shows too much. Shared
# with the other plots of this chapter, so that a colour always means the same.
COLORS = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7']
GRID_COLOR = '#C8C8C8'
SECTION_COLOR = '#808080'


def enable_idas_meta(model, every_step=False):
    """Configure a model to report the work done by the time integrator.

    Parameters
    ----------
    model : CADET object
        Model to be configured, modified in place.
    every_step : bool
        If True, USER_SOLUTION_TIMES is removed so that the integrator reports
        every step it takes, at the price of an output grid that is dictated by
        the solver. Defaults to False, i.e. reporting at the requested output
        times.

    Returns
    -------
    CADET object
        The configured model.
    """
    model.root.input['return'].WRITE_IDAS_META = 1
    model.root.input['return'].write_solution_times = 1

    if every_step:
        for key in ['user_solution_times', 'USER_SOLUTION_TIMES']:
            model.root.input.solver.pop(key, None)

    return model


@dataclass
class IdasMetrics:
    """Work done by the time integrator over the course of one simulation.

    All arrays are sampled at `time`, i.e. either at the requested output times
    or at every integrator step, see `every_step`.
    """

    time: np.ndarray
    step_size: np.ndarray
    bdf_order: np.ndarray
    steps: np.ndarray
    error_test_failures: np.ndarray
    residual_calls: np.ndarray
    section_times: np.ndarray
    restart_indices: np.ndarray
    n_steps_per_section: np.ndarray
    compute_time: float
    every_step: bool
    label: str = ''
    extras: dict = field(default_factory=dict)

    @property
    def n_steps(self):
        """Total number of time steps taken over all sections."""
        return int(self.n_steps_per_section.sum())

    @property
    def n_error_test_failures(self):
        """Total number of failed local error tests over all sections."""
        return int(self.error_test_failures.sum())

    @property
    def n_residual_calls(self):
        """Total number of residual evaluations over all sections."""
        return int(self.residual_calls.sum())

    @property
    def steps_per_simulated_second(self):
        """Time steps per second of simulated time.

        The rule-of-thumb stiffness indicator: if this is far above what the
        shape of the chromatogram requires, something fast is being resolved
        that nobody asked for.
        """
        span = self.time[-1] - self.time[0]
        return self.n_steps / span if span > 0 else np.nan

    @property
    def smallest_step(self):
        """Smallest step size actually taken."""
        taken = self.step_size[self.step_size > 0.0]
        return taken.min() if taken.size else np.nan

    @property
    def largest_step(self):
        """Largest step size actually taken."""
        return self.step_size.max() if self.step_size.size else np.nan


def _undo_section_restart(cumulative):
    """Convert a per-section cumulative IDAS counter to increments.

    CADET re-initializes IDAS at every section transition, so the counters
    restart from zero. At a restart the raw value already is the increment
    since the start of the section.

    Parameters
    ----------
    cumulative : np.array
        Cumulative counter as written by CADET.

    Returns
    -------
    (np.array, np.array, np.array)
        Increments per sample point, indices at which a restart happened, and
        the total of the counter per section.
    """
    cumulative = np.asarray(cumulative, dtype=float)

    restarts = np.flatnonzero(np.diff(cumulative) < 0.0) + 1

    increments = np.diff(cumulative, prepend=0.0)
    increments[restarts] = cumulative[restarts]

    bounds = np.concatenate(([0], restarts, [cumulative.size]))
    per_section = np.array([cumulative[end - 1] for end in bounds[1:]])

    return increments, restarts, per_section


def measure(simulation, label=''):
    """Get the integrator work from a simulation that ran with IDAS meta data.

    Parameters
    ----------
    simulation : string or CADET object
        Specifies the simulation of interest. Read via h5py, so that result
        files can be evaluated without a CADET installation.
    label : string
        Name of this run, used in plot legends and summary tables.

    Returns
    -------
    IdasMetrics
    """
    file_name = simulation if isinstance(simulation, str) else simulation.filename

    with h5py.File(file_name, 'r') as file:

        if 'meta/idas/IDAS_INTEGRATION_STEP_SIZE' not in file:
            raise ValueError(
                "No IDAS meta data in " + str(file_name) +
                ". Run the simulation with WRITE_IDAS_META = 1, "
                "see enable_idas_meta()."
            )

        idas = file['meta/idas']
        step_size = idas['IDAS_INTEGRATION_STEP_SIZE'][()]
        bdf_order = idas['IDAS_BDF_ORDER'][()]
        cumulative_steps = idas['IDAS_KUMULATIVE_NTIMESTEPS'][()]
        cumulative_etf = idas['IDAS_N_LOCAL_ERROR_TEST_FAILURES'][()]
        cumulative_res = idas['IDAS_N_RESIDUAL_CALLS'][()]

        time = file['output/solution/SOLUTION_TIMES'][()]
        section_times = file['input/solver/sections/SECTION_TIMES'][()]
        compute_time = float(np.squeeze(file['meta/TIME_SIM'][()]))

        every_step = 'USER_SOLUTION_TIMES' not in file['input/solver']

    steps, restarts, per_section = _undo_section_restart(cumulative_steps)
    error_test_failures, _, _ = _undo_section_restart(cumulative_etf)
    residual_calls, _, _ = _undo_section_restart(cumulative_res)

    return IdasMetrics(
        time=time,
        step_size=step_size,
        bdf_order=bdf_order,
        steps=steps,
        error_test_failures=error_test_failures,
        residual_calls=residual_calls,
        section_times=section_times,
        restart_indices=restarts,
        n_steps_per_section=per_section,
        compute_time=compute_time,
        every_step=every_step,
        label=label,
    )


def run_and_measure(model, label='', every_step=False, **kwargs):
    """Run a simulation and return the work done by its time integrator.

    Parameters
    ----------
    model : CADET object
        Model to be simulated. `model.filename` must be set.
    label : string
        Name of this run, used in plot legends and summary tables.
    every_step : bool
        Report at every integrator step instead of at the output times, see
        enable_idas_meta().

    Returns
    -------
    IdasMetrics
    """
    enable_idas_meta(model, every_step=every_step)
    model.save()

    data = model.run_simulation(**kwargs)

    if data.return_code != 0:
        raise RuntimeError(
            "Simulation '" + str(label) + "' failed: " + str(data.error_message)
        )

    return measure(model.filename, label=label)


def _ramp_length(step_size, tolerance=1e-9):
    """Length of the geometric start-up ramp at the beginning of a section.

    Coming out of a restart, IDAS doubles the step size every step until the
    error control takes over. The end of that doubling is where the start-up
    ends, which makes the length of the ramp parameter free. The initial step
    size is reported twice, so the doubling starts at the third entry.

    Parameters
    ----------
    step_size : np.array
        Step sizes from the restart onwards.
    tolerance : float
        Relative tolerance on the doubling.

    Returns
    -------
    int
        Number of steps in the ramp.
    """
    if step_size.size < 3:
        return int(step_size.size)

    for index in range(2, step_size.size):
        if step_size[index] < 2.0 * step_size[index - 1] * (1.0 - tolerance):
            return index

    return int(step_size.size)


def restart_cost(metrics):
    """Report what the integrator restart at each section transition costs.

    At a section transition, IDAS restarts at order one with INIT_STEP_SIZE and
    doubles its way back up. Two numbers describe that:

        - the start-up ramp, i.e. the steps spent doubling before the error
          control takes over. These are pure overhead and are the steps that a
          larger INIT_STEP_SIZE would save.
        - the steps taken before the step size reaches what it already was
          before the restart. That includes the ramp, but also the genuine
          transient that the new section introduces, which no solver setting
          can avoid.

    The start-up at the beginning of the simulation is not counted: an initial
    value problem has to start somewhere, so that one is unavoidable.

    Parameters
    ----------
    metrics : IdasMetrics
        Measured with every_step=True, otherwise the restart is not resolved:
        the ramp happens far below any sensible output grid.

    Returns
    -------
    pd.DataFrame
        One row per restart, empty if the simulation has only one section.
    """
    if not metrics.every_step:
        raise ValueError(
            "restart_cost needs every integrator step, i.e. metrics measured "
            "with every_step=True."
        )

    restarts = metrics.restart_indices
    ends = np.concatenate((restarts[1:], [metrics.time.size]))

    rows = []
    for restart, end in zip(restarts, ends):
        step_size_before = metrics.step_size[restart - 1]
        section = metrics.step_size[restart:end]

        n_ramp = _ramp_length(section)

        recovered = np.flatnonzero(section > step_size_before)
        n_recover = int(recovered[0]) if recovered.size else int(section.size)

        rows.append({
            'Restart at': metrics.time[restart],
            'Step size before': step_size_before,
            'Ramp steps': n_ramp,
            'Step size after ramp': section[min(n_ramp, section.size - 1)],
            'Steps below previous': n_recover,
            'Steps in section': int(section.size),
        })

    return pd.DataFrame(rows)


def summary(metrics):
    """One line description of the integrator work, e.g. for a figure title.

    Parameters
    ----------
    metrics : IdasMetrics

    Returns
    -------
    string
    """
    return (
        f"{metrics.n_steps} steps, "
        f"{metrics.n_error_test_failures} error test failures, "
        f"h in [{metrics.smallest_step:.1e}, {metrics.largest_step:.1e}], "
        f"{metrics.compute_time:.2f} s compute"
    )


def summary_table(metrics):
    """Compare the integrator work of several runs.

    Parameters
    ----------
    metrics : IdasMetrics or list of IdasMetrics

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(metrics, IdasMetrics):
        metrics = [metrics]

    return pd.DataFrame([{
        'Run': m.label,
        'Time steps': m.n_steps,
        'Steps per sim. second': m.steps_per_simulated_second,
        'Error test failures': m.n_error_test_failures,
        'Residual calls': m.n_residual_calls,
        'Smallest step': m.smallest_step,
        'Largest step': m.largest_step,
        'Compute time': m.compute_time,
    } for m in metrics])


def style_axis(axis, section_times=None, label_sections=False):
    """Apply the plot style used throughout this chapter."""
    axis.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.6)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    if section_times is None:
        return

    interior = np.asarray(section_times)[1:-1]

    for section_time in interior:
        axis.axvline(
            section_time, color=SECTION_COLOR,
            linewidth=1.0, linestyle='--', zorder=0
        )

    # Label the first transition only, rotated along the line. Section
    # boundaries can sit arbitrarily close together, so one label for all of
    # them is the only placement that cannot collide.
    if label_sections and interior.size:
        axis.annotate(
            'section transition' + ('s' if interior.size > 1 else ''),
            xy=(interior[0], 1.0), xycoords=('data', 'axes fraction'),
            xytext=(-11, -4), textcoords='offset points',
            color=SECTION_COLOR, fontsize=8,
            rotation=90, verticalalignment='top'
        )


def plot_step_size(metrics, axis=None, show_sections=True):
    """Plot the time step size over the simulated time.

    The telling picture: where the curve dips, the integrator is working hard.

    Parameters
    ----------
    metrics : IdasMetrics or list of IdasMetrics
        One curve per entry, labelled by `IdasMetrics.label`.
    axis : matplotlib axis
        Axis to draw on, created if not given.
    show_sections : bool
        Mark the section transitions.

    Returns
    -------
    matplotlib axis
    """
    if isinstance(metrics, IdasMetrics):
        metrics = [metrics]

    if axis is None:
        _, axis = plt.subplots(figsize=(9, 3.5))

    for index, m in enumerate(metrics):
        taken = m.step_size > 0.0
        axis.plot(
            m.time[taken], m.step_size[taken],
            color=COLORS[index % len(COLORS)], linewidth=2.0,
            label=m.label or None
        )

    axis.set_yscale('log')
    axis.set_xlabel(r'simulated time $/~s$')
    axis.set_ylabel(r'time step size $h~/~s$')
    style_axis(
        axis,
        metrics[0].section_times if show_sections else None,
        label_sections=True
    )

    if len(metrics) > 1:
        axis.legend(frameon=False)

    return axis


def plot_step_density(metrics, axis=None, show_sections=True, n_bins=150):
    """Plot how many time steps the integrator spends per simulated second.

    The steps are collected into equally spaced time bins rather than plotted
    per sample point, so that the curve means the same thing whether the data
    was reported at the output times or at every single step. Reporting every
    step puts hundreds of samples at one instant, where a per-sample density is
    the reciprocal step size and says nothing new.

    Parameters
    ----------
    metrics : IdasMetrics or list of IdasMetrics
        One curve per entry, labelled by `IdasMetrics.label`.
    axis : matplotlib axis
        Axis to draw on, created if not given.
    show_sections : bool
        Mark the section transitions.
    n_bins : int
        Number of time bins the steps are collected in.

    Returns
    -------
    matplotlib axis
    """
    if isinstance(metrics, IdasMetrics):
        metrics = [metrics]

    if axis is None:
        _, axis = plt.subplots(figsize=(9, 3.5))

    for index, m in enumerate(metrics):
        edges = np.linspace(m.time[0], m.time[-1], n_bins + 1)
        counts, _ = np.histogram(m.time, bins=edges, weights=m.steps)
        density = counts / np.diff(edges)

        axis.stairs(
            density, edges,
            color=COLORS[index % len(COLORS)], linewidth=1.8,
            label=m.label or None
        )

    axis.set_yscale('log')
    axis.set_xlabel(r'simulated time $/~s$')
    axis.set_ylabel(r'time steps $/~s$')
    style_axis(axis, metrics[0].section_times if show_sections else None)

    if len(metrics) > 1:
        axis.legend(frameon=False)

    return axis


def plot_bdf_order(metrics, axis=None, show_sections=True):
    """Plot the BDF order over the simulated time.

    Drops to one are integrator restarts: every section transition costs the
    order that the integrator had built up.

    Parameters
    ----------
    metrics : IdasMetrics or list of IdasMetrics
        One curve per entry, labelled by `IdasMetrics.label`.
    axis : matplotlib axis
        Axis to draw on, created if not given.
    show_sections : bool
        Mark the section transitions.

    Returns
    -------
    matplotlib axis
    """
    if isinstance(metrics, IdasMetrics):
        metrics = [metrics]

    if axis is None:
        _, axis = plt.subplots(figsize=(9, 2.5))

    for index, m in enumerate(metrics):
        axis.step(
            m.time, m.bdf_order, where='post',
            color=COLORS[index % len(COLORS)], linewidth=1.5,
            label=m.label or None
        )

    axis.set_ylim(0, 6)
    axis.set_yticks(range(0, 6))
    axis.set_xlabel(r'simulated time $/~s$')
    axis.set_ylabel('BDF order')
    style_axis(axis, metrics[0].section_times if show_sections else None)

    if len(metrics) > 1:
        axis.legend(frameon=False)

    return axis


def plot_integrator_work(metrics, panels=('step_size', 'density', 'order'),
                         show_sections=True, title=None):
    """Plot the integrator work of one or several runs, one panel per quantity.

    Parameters
    ----------
    metrics : IdasMetrics or list of IdasMetrics
        One curve per entry, labelled by `IdasMetrics.label`.
    panels : tuple of strings
        Any of 'step_size', 'density' and 'order'.
    show_sections : bool
        Mark the section transitions.
    title : string
        Figure title, defaults to summary() for a single run.

    Returns
    -------
    (matplotlib figure, np.array of matplotlib axes)
    """
    if isinstance(metrics, IdasMetrics):
        metrics = [metrics]

    plotters = {
        'step_size': plot_step_size,
        'density': plot_step_density,
        'order': plot_bdf_order,
    }
    unknown = set(panels) - set(plotters)
    if unknown:
        raise ValueError(
            "Unknown panel(s) " + str(sorted(unknown)) +
            ", available are " + str(sorted(plotters))
        )

    heights = [2.0 if panel == 'order' else 3.0 for panel in panels]
    figure, axes = plt.subplots(
        len(panels), 1, sharex=True,
        figsize=(9, sum(heights)),
        gridspec_kw={'height_ratios': heights}
    )
    axes = np.atleast_1d(axes)

    for axis, panel in zip(axes, panels):
        plotters[panel](metrics, axis=axis, show_sections=show_sections)

    for axis in axes[:-1]:
        axis.set_xlabel('')

    if title is None and len(metrics) == 1:
        title = (metrics[0].label + ': ' if metrics[0].label else '') \
            + summary(metrics[0])
    if title is not None:
        figure.suptitle(title, fontsize=12)

    figure.tight_layout()

    return figure, axes
