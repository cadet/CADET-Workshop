# -*- coding: utf-8 -*-
"""

Where does a discretization produce negative concentrations, and when is that
fatal?

Two studies share this explorer, because they are the same failure seen from
two sides:

  * utility/setting_Col1D_empm_forum1283.py puts a steep front in the bulk and
    a binding model that evaluates c ** beta. The undershoot is small, but a
    non-integer beta turns it into a NaN and the run dies.

  * utility/setting_Col1D_langmuir_shrinking_core.py puts a steep front inside
    the particle. Nothing crashes; the undershoot simply sits there, in a
    field nobody plots.

Both are explored the same way: pick a method and a resolution, and read three
panels.

  top left    the error against a converged reference, over the number of
              spatial points. The usual convergence picture.
  top right   the most negative concentration anywhere in the field, over the
              same axis. This is the panel the error norm hides: a run can be
              converging nicely on the left and still be carrying negative
              concentrations on the right.
  bottom      the chromatogram of the setting currently on the sliders.

A run that crashes has no error and no minimum, so it is drawn as a cross on
the resolution where it died.

The reference for the error is the finest finite volume run of the same
setting. Finite volumes with WENO is monotone, so it is the one member of the
family guaranteed not to undershoot, which is what makes it usable as the
truth here; a DG reference would have to be trusted not to have the very
defect being measured.

"""

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence, Tuple

import h5py
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from cadet import Cadet

METHOD_COLOR = {
    0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green',
    3: 'tab:red', 4: 'tab:purple', 5: 'tab:brown',
}
METHOD_MARKER = {0: 's', 1: 'o', 2: '^', 3: 'D', 4: 'v', 5: 'P'}

RUN_TIMEOUT = 900.0


def method_label(spatial_method):
    """Name of a spatial method: FV, or DG with its polynomial degree."""
    return 'FV' if spatial_method == 0 else f'DG{spatial_method}'


def method_style(spatial_method):
    """Colour and marker for one method, consistent across both panels."""
    return (METHOD_COLOR.get(spatial_method, 'tab:gray'),
            METHOD_MARKER.get(spatial_method, 'o'))


@dataclass
class Result:
    """One simulation: what it cost, how wrong it was, how negative it went."""

    spatial_method: int
    n_points: int
    dofs: int
    times: np.ndarray = None
    outlet: np.ndarray = None
    minimum: float = float('nan')
    simulation_time: float = float('nan')
    error: float = float('nan')
    message: str = ''

    @property
    def crashed(self):
        return self.outlet is None


@dataclass
class Case:
    """Everything the explorer needs to know about one of the two studies."""

    title: str
    build: Callable
    dofs: Callable
    resolutions: Dict[int, Sequence[int]]
    # The field searched for negative values, and what to call it.
    negative_dataset: Tuple[str, ...]
    negative_label: str
    component_names: Sequence[str]
    knob_description: str = ''
    knob_options: Sequence = field(default_factory=tuple)
    knob_default: float = 0.0
    knob_unit: str = ''
    time_unit: str = 's'
    chromatogram_label: str = 'Outlet concentration / mol m$^{-3}$'
    # Below this outlet scale there is nothing at the outlet to measure an
    # error against, so the accuracy panel stays empty rather than reporting
    # the ratio of two numerical zeros. Called with the knob value.
    signal_floor: Callable = None


def _dataset(group, candidates):
    """First dataset in the group that one of the candidate names matches.

    CADET writes a single array when the components are not split and one
    array per component when they are, so both layouts have to be accepted.
    """
    for name in candidates:
        if name in group:
            return np.asarray(group[name])

    stacked = [np.asarray(group[key]) for key in sorted(group)
               if any(key.startswith(name + '_COMP') for name in candidates)]
    if stacked:
        return np.stack(stacked, axis=-1)

    return None


def simulate(case, spatial_method, n_points, knob, output_directory):
    """Run one setting and reduce it to a Result."""
    os.makedirs(output_directory, exist_ok=True)
    result = Result(spatial_method=spatial_method, n_points=n_points,
                    dofs=case.dofs(spatial_method, n_points))

    model = Cadet()
    model.root = case.build(spatial_method, n_points, knob)
    model.filename = os.path.join(
        output_directory,
        f'{method_label(spatial_method)}_n{n_points}_k{knob:g}.h5')
    model.save()

    run_info = model.run_simulation(timeout=RUN_TIMEOUT)
    if run_info.return_code != 0:
        result.message = str(run_info.error_message).strip()[:160]
        return result

    with h5py.File(model.filename, 'r') as handle:
        group = handle['output/solution/unit_001']
        outlet = _dataset(group, ('SOLUTION_OUTLET',))
        watched = _dataset(group, case.negative_dataset)
        times = np.asarray(handle['output/solution/SOLUTION_TIMES'])
        timing = handle.get('meta/TIME_SIM')
        result.simulation_time = (float(np.asarray(timing))
                                  if timing is not None else float('nan'))

    result.times = times
    result.outlet = outlet if outlet.ndim > 1 else outlet.reshape(-1, 1)
    result.minimum = (float(watched.min()) if watched is not None
                      else float('nan'))

    return result


class Ladder:
    """The runs done so far, for one value of the case's knob."""

    def __init__(self, case, knob, output_directory):
        self.case = case
        self.knob = knob
        self.output_directory = output_directory
        self.results = {}

    def get(self, spatial_method, n_points, status=None):
        key = (spatial_method, n_points)
        if key not in self.results:
            if status is not None:
                status.value = (
                    f'Simulating {method_label(spatial_method)} on'
                    f' {n_points} &hellip;')
            self.results[key] = simulate(self.case, spatial_method, n_points,
                                         self.knob, self.output_directory)
            if status is not None:
                status.value = ''

        return self.results[key]

    def fill(self, spatial_method, status=None):
        """Run the whole ladder of one method, so the panels have a curve."""
        for n_points in self.case.resolutions[spatial_method]:
            self.get(spatial_method, n_points, status=status)

    def reference(self, status=None):
        """The finest finite volume run, which is the monotone one."""
        finest = self.case.resolutions[0][-1]
        result = self.get(0, finest, status=status)

        return None if result.crashed else result

    def scored(self, status=None):
        """Every result, with the error filled in against the reference."""
        reference = self.reference(status=status)
        if reference is not None:
            scale = float(np.max(abs(reference.outlet)))
            floor = (self.case.signal_floor(self.knob)
                     if self.case.signal_floor is not None else 0.0)
            if scale <= floor:
                return self.results
            for result in self.results.values():
                if result.crashed or result is reference:
                    continue
                if result.outlet.shape != reference.outlet.shape:
                    continue
                result.error = float(
                    np.max(abs(result.outlet - reference.outlet)) / scale)

        return self.results


def _plot_ladder(axis, results, panel, current):
    """One panel: error or minimum over the number of spatial points."""
    methods = sorted({key[0] for key in results})
    for spatial_method in methods:
        colour, marker = method_style(spatial_method)
        points = sorted((result for (method, _), result in results.items()
                         if method == spatial_method),
                        key=lambda result: result.dofs)
        good = [result for result in points if not result.crashed]
        if panel == 'error':
            good = [result for result in good if np.isfinite(result.error)]
            values = [result.error for result in good]
        else:
            values = [result.minimum for result in good]
        if good:
            axis.plot([result.dofs for result in good], values,
                      marker=marker, color=colour, linewidth=1.4,
                      markersize=5, label=method_label(spatial_method))
        for result in points:
            if result.crashed:
                axis.plot(result.dofs, axis.get_ylim()[0], marker='x',
                          color=colour, markersize=11, markeredgewidth=2.4,
                          clip_on=False, zorder=5)

    if current is not None and not current.crashed:
        value = current.error if panel == 'error' else current.minimum
        if np.isfinite(value):
            axis.plot(current.dofs, value, marker='o', markersize=13,
                      markerfacecolor='none', markeredgecolor='black',
                      markeredgewidth=1.6, zorder=6)

    axis.set_xscale('log')
    axis.set_xlabel('Spatial points per component')
    axis.grid(True, which='both', alpha=0.25)


def plot_case(case, results, current, knob):
    """The three panels: two ladders side by side, the chromatogram below."""
    figure = plt.figure(figsize=(12, 8.5))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.9), hspace=0.33,
                               wspace=0.22)
    error_axis = figure.add_subplot(grid[0, 0])
    minimum_axis = figure.add_subplot(grid[0, 1])
    chromatogram_axis = figure.add_subplot(grid[1, :])

    _plot_ladder(error_axis, results, 'error', current)
    error_axis.set_yscale('log')
    error_axis.set_ylabel('Relative error against the fine FV run')
    error_axis.set_title('Accuracy')
    if any(np.isfinite(result.error) for result in results.values()):
        error_axis.legend(fontsize=9, ncol=2)
    else:
        error_axis.text(
            0.5, 0.5,
            'Nothing leaves the column on this timescale,\n'
            'so the outlet cannot measure an error here.\n'
            'The panel on the right still can.',
            transform=error_axis.transAxes, ha='center', va='center',
            fontsize=10, color='tab:gray')

    _plot_ladder(minimum_axis, results, 'minimum', current)
    finite = [result.minimum for result in results.values()
              if np.isfinite(result.minimum)]
    if finite:
        span = max(abs(value) for value in finite)
        minimum_axis.set_yscale('symlog',
                                linthresh=max(span * 1e-6, 1e-16))
    minimum_axis.axhline(0.0, color='black', linewidth=1.0)
    minimum_axis.set_ylabel(f'Smallest {case.negative_label}'
                            ' / mol m$^{-3}$')
    minimum_axis.set_title('How far below zero the solution goes')
    minimum_axis.legend(fontsize=9, ncol=2)

    if current is None or current.crashed:
        chromatogram_axis.text(
            0.5, 0.5,
            'The selected setting did not run.\n'
            + (current.message if current is not None else ''),
            transform=chromatogram_axis.transAxes, ha='center', va='center',
            fontsize=11, color='tab:red', wrap=True)
        chromatogram_axis.set_xticks([])
        chromatogram_axis.set_yticks([])
    else:
        for index, name in enumerate(case.component_names):
            if index >= current.outlet.shape[1]:
                break
            chromatogram_axis.plot(current.times, current.outlet[:, index],
                                   linewidth=1.6, label=name)
        chromatogram_axis.set_xlabel(f'Time / {case.time_unit}')
        chromatogram_axis.set_ylabel(case.chromatogram_label)
        chromatogram_axis.grid(True, alpha=0.25)
        chromatogram_axis.legend(fontsize=9)

    if current is not None:
        state = (f'{method_label(current.spatial_method)},'
                 f' {current.n_points} '
                 + ('cells' if current.spatial_method == 0 else 'elements')
                 + f', {current.dofs} points per component')
        if current.crashed:
            state += '  -  CRASHED'
        else:
            state += f',  smallest value {current.minimum:.3e}'
        chromatogram_axis.set_title(state, fontsize=11)

    knob_text = (f'{case.knob_description} = {knob:g}{case.knob_unit}'
                 if case.knob_description else '')
    figure.suptitle(f'{case.title}' + (f'   ({knob_text})' if knob_text else ''),
                    fontsize=13)

    return figure


def negative_concentration_explorer(case, output_directory):
    """Sliders over method, resolution and the case's own knob."""
    output_directory = str(output_directory)
    ladders = {}

    plot_output = widgets.Output()
    status = widgets.HTML()

    methods = sorted(case.resolutions)
    method_slider = widgets.SelectionSlider(
        options=[(method_label(method), method) for method in methods],
        value=methods[0], description='Method', continuous_update=False,
    )
    points_slider = widgets.SelectionSlider(
        options=list(case.resolutions[methods[0]]),
        value=case.resolutions[methods[0]][0],
        description='Resolution', continuous_update=False,
    )
    knob_slider = widgets.SelectionSlider(
        options=list(case.knob_options), value=case.knob_default,
        description=case.knob_description, continuous_update=False,
        layout=widgets.Layout(
            display='none' if not case.knob_options else None,
            width='460px'),
    )
    fill_button = widgets.Button(
        description='Fill this ladder', icon='play',
        tooltip='Run every resolution of the selected method',
        layout=widgets.Layout(width='180px'),
    )

    def ladder():
        knob = knob_slider.value
        if knob not in ladders:
            ladders[knob] = Ladder(case, knob, output_directory)

        return ladders[knob]

    def show(change=None):
        current_ladder = ladder()
        current = current_ladder.get(method_slider.value,
                                     points_slider.value, status=status)
        results = current_ladder.scored(status=status)
        with plot_output:
            plot_output.clear_output(wait=True)
            figure = plot_case(case, results, current, knob_slider.value)
            plt.show()
            plt.close(figure)
        if current.crashed:
            status.value = (f'<b style="color:#b00">The run failed:</b> '
                            f'{current.message}')
        else:
            status.value = (
                f'{method_label(current.spatial_method)} on'
                f' {current.n_points}: {current.dofs} points per component,'
                f' smallest value {current.minimum:.3e}')

    def on_method(change):
        """Offer only the resolutions this method can actually be built on."""
        options = list(case.resolutions[method_slider.value])
        # Stay as close as possible to the resolution that was selected, so
        # that switching the method is a comparison and not a jump.
        wanted = points_slider.value
        points_slider.unobserve(show, names='value')
        points_slider.options = options
        points_slider.value = min(options,
                                  key=lambda count: abs(count - wanted))
        points_slider.observe(show, names='value')
        show()

    def on_fill(button=None):
        fill_button.disabled = True
        try:
            ladder().fill(method_slider.value, status=status)
        finally:
            fill_button.disabled = False
        show()

    method_slider.observe(on_method, names='value')
    points_slider.observe(show, names='value')
    knob_slider.observe(show, names='value')
    fill_button.on_click(on_fill)

    show()

    return widgets.VBox([
        widgets.HBox([
            widgets.VBox([method_slider, points_slider, knob_slider]),
            widgets.VBox([fill_button]),
        ]),
        status,
        plot_output,
    ])


# -- The two studies ------------------------------------------------------

FORUM1283_RESOLUTIONS = {
    0: (8, 16, 32, 64, 128, 256),
    1: (4, 8, 16, 32, 64, 128),
    2: (4, 8, 16, 32, 64),
    3: (4, 8, 16, 32, 64),
    4: (2, 4, 8, 16, 32),
    5: (2, 4, 8, 16, 32),
}

# A discontinuous Galerkin particle carries degree + 1 points per element, so
# each method is offered only the counts it can actually be built on. The
# ladders stop around twenty points because refining past that does not remove
# the undershoot, which is the finding, and every run here is live.
SHRINKING_CORE_RESOLUTIONS = {
    0: (2, 4, 6, 8, 12, 16, 20),
    1: (2, 4, 6, 8, 12, 16, 20),
    2: (3, 6, 9, 12, 18),
    3: (4, 8, 12, 16, 20),
    4: (5, 10, 15, 20),
    5: (6, 12, 18),
}


def forum1283_case():
    """The thread's setting: the undershoot is in the bulk, and it is fatal."""
    import utility.setting_Col1D_empm_forum1283 as setting

    return Case(
        title='EMPM binding on a steep bulk front (forum thread 1283)',
        build=lambda method, points, knob: setting.get_model(
            spatial_method=method, n_cells=points, beta=knob),
        dofs=setting.spatial_dofs,
        resolutions=FORUM1283_RESOLUTIONS,
        negative_dataset=('SOLUTION_BULK',),
        negative_label='bulk concentration',
        component_names=setting.COMPONENT_NAMES,
        knob_description='beta',
        knob_options=(('0 (integer: harmless)', 0.0),
                      ('0.1 (the thread)', 0.1),
                      ('0.25', 0.25), ('0.5', 0.5),
                      ('1 (integer: harmless)', 1.0)),
        knob_default=0.1,
    )


def shrinking_core_case():
    """A sharp front inside the bead, where nothing crashes and nobody looks."""
    import utility.setting_Col1D_langmuir_shrinking_core as setting

    def label(c_feed):
        return f'{c_feed:g}  (lambda = {setting.lambda_ratio(c_feed):.0f})'

    # Every feed down to 1.0 breaks through inside the simulated time; the
    # two below it do not, which is the regime itself showing up.
    feeds = (10.0, 3.0, 1.0, 0.3, 0.1)

    return Case(
        title='Shrinking-core uptake: a steep front inside the particle',
        build=lambda method, points, knob: setting.get_model(
            spatial_method=method, n_points=points, c_feed=knob),
        dofs=setting.spatial_dofs,
        resolutions=SHRINKING_CORE_RESOLUTIONS,
        negative_dataset=('SOLUTION_PARTICLE_PARTYPE_000',
                          'SOLUTION_PARTICLE'),
        negative_label='pore concentration',
        component_names=setting.COMPONENT_NAMES,
        knob_description='Feed',
        knob_options=tuple((label(c_feed), c_feed) for c_feed in feeds),
        knob_default=setting.C_FEED,
        knob_unit=' mol m^-3',
        signal_floor=lambda c_feed: 0.01 * c_feed,
    )
