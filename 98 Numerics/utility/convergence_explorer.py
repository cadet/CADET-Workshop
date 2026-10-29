"""Interactive convergence studies from CADET-Verification result files.

A convergence study refines one setting over a sequence of axial resolutions,
for the WENO finite volume scheme and for DG of several polynomial degrees, and
measures the outlet error of each against a shared reference. Plotting that
error against the compute time puts the methods on a common footing: the useful
method is the one that reaches a given accuracy soonest.

The measurements are the ones published with CADET-Verification, read from the
json files under ``data/spatial_discretization/convergence``. Those hold
errors and compute times but no chromatograms, so the outlets of the sweep are
copied out of the result files CADET-Verification wrote, by
``utility/convergence_settings.py --collect``, and stored beside the json
together with the reference. A study
whose outlets are at hand offers exactly those levels on its sliders and draws
each one over the reference, which is what makes the numerical dispersion of
the finite volumes and the oscillations of DG visible. A study without them
shows the reference alone.

The accuracy against cost plot fills up as levels are visited, rather than
showing the whole sweep at once, so that a comparison is built deliberately,
one discretization at a time.

Errors are absolute errors in mol/m^3, as published, in whichever of the three
norms is selected, rather than the relative errors of the load-wash-elute
study.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from utility import convergence_settings as recipes

DEFAULT_DIRECTORY = Path('data') / 'spatial_discretization' / 'convergence'

# Colour and marker per method, following the published figures: squares for
# the finite volume scheme, circles for the DG degrees. Orange is left out
# because it marks whichever level is selected.
METHOD_STYLE = {
    'FVWENO3': ('#9467bd', 's'),
    'DG_P3': ('#1f77b4', 'o'),
    'DG_P4': ('#2ca02c', 'o'),
    'DG_P5': ('#d62728', 'o'),
}
CURRENT_COLOR = '#ff7f0e'
# The salt of the load-wash-elute example, on its own axis to the right.
SALT_COLOR = '#7f7f7f'

METHOD_LABELS = {
    'FVWENO3': 'FV (WENO3)',
    'DG_P3': 'DG P3',
    'DG_P4': 'DG P4',
    'DG_P5': 'DG P5',
}

# The three error norms of the published tables, with the columns they are
# stored under and the axis label they get. All three are absolute errors of
# the outlet against the reference, over every component and output time:
# the largest deviation, the sum of them, and the root of the sum of squares.
NORMS = {
    'L-infinity': ('Max. error', 'Max. EOC', r'$L^\infty$ error in mol / m$^3$'),
    'L1': (r'$L^1$ error', r'$L^1$ EOC', r'$L^1$ error in mol / m$^3$'),
    'L2': (r'$L^2$ error', r'$L^2$ EOC', r'$L^2$ error in mol / m$^3$'),
}
DEFAULT_NORM = 'L-infinity'

# How each study's chromatogram is laid out beside the plot. The load-wash-elute
# salt runs two orders of magnitude above the proteins, so it gets a panel of
# its own, stacked over the proteins.
STUDY_DISPLAY = {
    'lrmp_sma': {
        'components': ('Salt', 'Component 1', 'Component 2', 'Component 3'),
        'separate_first': True,
    },
    'lrm_langmuir': {
        'components': ('Component 1', 'Component 2'),
        'separate_first': False,
    },
}

STORED, RERUN = 'Published result', 'Rerun simulation'


def method_label(method):
    return METHOD_LABELS.get(method, method)


def outlet_errors(outlet, reference):
    """The three published norms of one outlet against the reference.

    These are the definitions of ``convergence.convergency_table`` with unit
    weights, which is how the committed tables were produced: a rerun of a
    published level reproduces its numbers in all three norms.
    """
    deviation = abs(np.asarray(outlet) - np.asarray(reference))

    return {
        'L-infinity': float(np.max(deviation)),
        'L1': float(np.sum(deviation)),
        'L2': float(np.sqrt(np.sum(np.square(deviation)))),
    }


@dataclass
class ConvergenceLevel:
    """One refinement level of one method: what was published, and any rerun."""

    method: str
    polynomial_degree: int
    n_elements: int
    bulk_dofs: int
    simulation_time: float
    errors: dict
    eocs: dict
    outlet: np.ndarray = None
    measured_errors: dict = None
    measured_time: float = None
    measured_outlet: np.ndarray = None

    @property
    def label(self):
        return method_label(self.method)

    @property
    def is_measured(self):
        return self.measured_time is not None

    def error(self, norm=DEFAULT_NORM):
        """The error to plot: the one just measured, else the published one."""
        if self.measured_errors is not None:
            return self.measured_errors[norm]

        return self.errors[norm]

    def eoc(self, norm=DEFAULT_NORM):
        return self.eocs[norm]

    @property
    def time(self):
        """The compute time to plot: just measured, else as published."""
        return (self.simulation_time if self.measured_time is None
                else self.measured_time)

    @property
    def curve(self):
        """The outlet of this level: just rerun, else the one collected."""
        return self.outlet if self.measured_outlet is None else self.measured_outlet


@dataclass
class ConvergenceStudy:
    """The published levels of one setting, with the outlets that were stored."""

    key: str
    name: str
    levels: dict = field(default_factory=dict)
    times: np.ndarray = None
    reference: np.ndarray = None

    @property
    def display(self):
        return STUDY_DISPLAY.get(self.key, {})

    def methods(self):
        """Finite volumes first, then the DG degrees in order."""
        return sorted(self.levels,
                      key=lambda method: (method != 'FVWENO3', method))

    def resolutions(self, method):
        return tuple(level.n_elements for level in self.levels[method])

    def get(self, method, n_elements):
        for level in self.levels.get(method, ()):
            if level.n_elements == n_elements:
                return level

        return None


def load_convergence(study='lrmp_sma', directory=DEFAULT_DIRECTORY):
    """Read one published study, its reference, and any outlets stored for it."""
    if study not in recipes.STUDIES:
        raise ValueError(f'Unknown study {study!r}; expected one of '
                         f'{sorted(recipes.STUDIES)}.')

    recipe = recipes.STUDIES[study]
    directory = Path(directory)
    json_file = directory / recipe['json']
    if not json_file.exists():
        raise FileNotFoundError(f'No convergence results at {json_file}.')

    with json_file.open() as handle:
        content = json.load(handle)

    loaded = ConvergenceStudy(key=study, name=content['meta']['name'])
    for method, record in content['convergence'].items():
        table = record['outlet']
        loaded.levels[method] = [
            ConvergenceLevel(
                method=method,
                polynomial_degree=int(table['$N_d$'][index]),
                n_elements=int(table['$N_e^z$'][index]),
                bulk_dofs=int(table['Bulk DoF'][index]),
                simulation_time=float(table['Sim. time'][index]),
                errors={norm: float(table[columns[0]][index])
                        for norm, columns in NORMS.items()},
                eocs={norm: float(table[columns[1]][index])
                      for norm, columns in NORMS.items()},
            )
            for index in range(len(table['$N_e^z$']))
        ]

    loaded.times, loaded.reference = recipes.read_reference(recipe)

    # The sweep's own chromatograms, where CADET-Verification's result files
    # were at hand. A study that has them offers exactly those levels, so that
    # every position on the slider can show what that discretization does to
    # the solution; a study without them shows the reference for every level.
    curves_file = Path(recipes.outlets_file(recipe))
    if curves_file.exists():
        with np.load(curves_file) as stored:
            loaded.times = stored['times']
            for method, levels in loaded.levels.items():
                for level in levels:
                    key = recipes.level_key(method, level.n_elements)
                    if key in stored.files:
                        level.outlet = stored[key]
        loaded.levels = {
            method: [level for level in levels if level.outlet is not None]
            for method, levels in loaded.levels.items()
        }
        loaded.levels = {method: levels for method, levels in loaded.levels.items()
                         if levels}

    return loaded


def run_level(study, level, output_directory='data'):
    """Simulate one level as the published sweep did, and measure it.

    Fills in the level's measured compute time, errors and outlet. The errors
    can only be measured if the study's reference file is present.
    """
    recipe = recipes.STUDIES[study.key]
    times, outlet, seconds = recipes.simulate(
        recipe, level.polynomial_degree, level.n_elements,
        recipe['idas_abstol'],
        f'rerun_{study.key}_{recipes.level_key(level.method, level.n_elements)}',
        directory=output_directory)

    level.measured_time = seconds
    level.measured_outlet = outlet
    if study.reference is not None and outlet.shape == study.reference.shape:
        level.measured_errors = outlet_errors(outlet, study.reference)
    if study.times is None:
        study.times = times

    return level


def plot_performance(history, current=None, norm=DEFAULT_NORM, ax=None):
    """Error against compute time for every level visited so far.

    ``history`` maps ``(method, axial cells)`` to the level shown for it. One
    dashed line per method, ordered by compute time. A level that has been
    rerun sits where it was just measured, with a dotted arrow from where it
    was published.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    by_method = {}
    for level in history.values():
        by_method.setdefault(level.method, []).append(level)

    for method in sorted(by_method, key=lambda name: (name != 'FVWENO3', name)):
        color, marker = METHOD_STYLE.get(method, ('#7f7f7f', 'o'))
        levels = sorted(by_method[method], key=lambda level: level.time)
        ax.plot([level.time for level in levels],
                [level.error(norm) for level in levels],
                linestyle='--', linewidth=1.8, color=color,
                marker=marker, markersize=8, label=method_label(method))
        for level in levels:
            if level.is_measured:
                ax.annotate(
                    '', xy=(level.time, level.error(norm)),
                    xytext=(level.simulation_time, level.errors[norm]),
                    arrowprops=dict(arrowstyle='->', color='gray',
                                    linewidth=1, linestyle=':'),
                )

    if current is not None:
        _, marker = METHOD_STYLE.get(current.method, ('#7f7f7f', 'o'))
        ax.plot(current.time, current.error(norm), linestyle='none',
                marker=marker, markersize=12, color=CURRENT_COLOR, zorder=5,
                label='Current')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute time in seconds')
    ax.set_ylabel(NORMS[norm][2])
    ax.grid(True, which='major', alpha=0.4)
    if by_method:
        ax.legend(fontsize=9, loc='best')

    return ax


def summarize_level(level, norm=DEFAULT_NORM):
    """The numbers of one level, as the one line above the plot."""
    if level.is_measured:
        return (f'{norm} error {level.error(norm):.4g} mol/m&sup3; in'
                f' {level.measured_time:.3g} s, measured on this machine'
                f' (published: {level.errors[norm]:.4g} in'
                f' {level.simulation_time:.3g} s)')

    return (f'{norm} error {level.errors[norm]:.4g} mol/m&sup3; in'
            f' {level.simulation_time:.3g} s, as published'
            f' (EOC {level.eoc(norm):.2f})')


def draw_outlet(ax, times, curve, reference, components, first=0, title=None):
    """Draw one outlet over its reference, in a single panel.

    A setting whose first component runs on another scale, the salt of the
    load-wash-elute example, keeps it in the same panel on a second axis to
    the right, so that the shape of both is read against one time axis.
    """
    salt_ax = ax.twinx() if first else None
    if salt_ax is not None:
        if reference is not None:
            salt_ax.plot(times, reference[:, 0], color=SALT_COLOR,
                         linestyle='--', linewidth=1)
        if curve is not None:
            salt_ax.plot(times, curve[:, 0], color=SALT_COLOR, linewidth=1.4,
                         label=components[0])
        else:
            salt_ax.plot([], [], color=SALT_COLOR, linewidth=1.4,
                         label=components[0])
        salt_ax.set_ylabel(f'{components[0]}' + r' $[mol \cdot m^{-3}]$',
                           color=SALT_COLOR)
        salt_ax.tick_params(axis='y', labelcolor=SALT_COLOR)

    if reference is not None:
        ax.plot(times, reference[:, first:], color='black',
                linestyle='--', linewidth=1)
        ax.plot([], [], color='black', linestyle='--', linewidth=1,
                label='Reference')
    if curve is not None:
        for index in range(first, curve.shape[1]):
            ax.plot(times, curve[:, index], color=f'C{index - first}',
                    linewidth=1.4, label=components[index])

    ax.set_xlabel(r'Time $[s]$')
    ax.set_ylabel(r'Concentration $[mol \cdot m^{-3}]$')
    ax.grid(alpha=0.3)
    if title is not None:
        ax.set_title(title, fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    if salt_ax is not None:
        extra = salt_ax.get_legend_handles_labels()
        handles, labels = handles + extra[0], labels + extra[1]
    ax.legend(handles, labels, fontsize=8)

    return ax


def plot_case(study, level, history=None, norm=DEFAULT_NORM, axes=None):
    """Accuracy against cost next to, or above, the chromatogram."""
    display = study.display
    components = display.get('components')
    separate = display.get('separate_first', False)
    key = (level.method, level.n_elements)
    history = dict(history or {key: level})

    if axes is None:
        figure = plt.figure(figsize=(13, 5.4))
        grid = figure.add_gridspec(1, 2, width_ratios=(1, 1.15))
        performance_ax = figure.add_subplot(grid[0, 0])
        axes = [figure.add_subplot(grid[0, 1])]
    else:
        figure = axes[0].figure
        performance_ax = None

    if performance_ax is not None:
        plot_performance(history, current=level, norm=norm, ax=performance_ax)
        performance_ax.set_title('Accuracy against cost', fontsize=12)

    # The chromatogram is the selected level over the reference, so that what
    # the discretization does to the solution is visible: the finite volumes
    # smear a front, DG overshoots it. Where a study has no outlet per level,
    # the reference alone says what the setting looks like.
    times = study.times
    curve = level.curve
    reference = study.reference
    if curve is None and reference is None:
        axes[0].text(
            0.5, 0.5,
            'Nothing stored for this level.\n'
            'Switch to "Rerun simulation" to compute\n'
            'its chromatogram.',
            transform=axes[0].transAxes, ha='center', va='center',
            fontsize=11, color='dimgray')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        figure.tight_layout()

        return figure

    if components is None:
        shape = curve if curve is not None else reference
        components = tuple(f'Component {index}'
                           for index in range(shape.shape[1]))

    outlet_ax = axes[0]
    first = 1 if separate else 0
    draw_outlet(outlet_ax, times, curve, reference, components, first,
                title=(f'{level.label}, {level.n_elements} axial cells'
                       if curve is not None else 'Outlet (reference)'))

    figure.tight_layout()

    return figure


def convergence_explorer(study='lrmp_sma', directory=DEFAULT_DIRECTORY,
                         output_directory='data'):
    """Slider explorer over one published convergence study.

    The sliders pick a level, marked in orange. Every level that is visited
    stays on the accuracy against cost plot, so a comparison builds up one
    discretization at a time; the clear button starts it over. The error norm
    switch redraws the plot in any of the three published norms.

    Switching the mode to ``Rerun simulation`` runs the selected level through
    CADET, exactly as the published sweep did, and moves its point to the time
    and error just measured.
    """
    loaded = (study if isinstance(study, ConvergenceStudy)
              else load_convergence(study, directory))
    methods = loaded.methods()
    history = {}

    plot_output = widgets.Output()
    status = widgets.HTML()

    mode_buttons = widgets.ToggleButtons(
        options=[STORED, RERUN], value=STORED, description='Mode',
        tooltips=['Show the published result of this level',
                  'Run this level through CADET now'],
    )
    norm_buttons = widgets.ToggleButtons(
        options=list(NORMS), value=DEFAULT_NORM, description='Error',
        tooltips=['Largest deviation from the reference',
                  'Sum of the deviations',
                  'Root of the sum of the squared deviations'],
    )
    method_slider = widgets.SelectionSlider(
        options=[(method_label(method), method) for method in methods],
        value=methods[0], description='Method', continuous_update=False,
    )
    elements_slider = widgets.SelectionSlider(
        options=list(loaded.resolutions(methods[0])),
        value=loaded.resolutions(methods[0])[0],
        description='Axial cells', continuous_update=False,
    )
    run_button = widgets.Button(
        description='Run simulation', button_style='primary',
        layout=widgets.Layout(width='160px', display='none'),
    )
    clear_button = widgets.Button(
        description='Clear history', icon='eraser',
        tooltip='Start the accuracy against cost plot over',
        layout=widgets.Layout(width='160px'),
    )

    def current_level():
        return loaded.get(method_slider.value, elements_slider.value)

    def draw(level):
        history[(level.method, level.n_elements)] = level
        with plot_output:
            plot_output.clear_output(wait=True)
            figure = plot_case(loaded, level, history, norm_buttons.value)
            plt.show()
            plt.close(figure)

    def show(change=None):
        level = current_level()
        caption = (f'{level.label}, {level.n_elements} axial cells:'
                   f' {level.bulk_dofs} bulk DoF')
        if mode_buttons.value == RERUN and not level.is_measured:
            status.value = (
                f'{caption} &mdash; press <b>Run simulation</b> to run it now.'
                f' It took {level.simulation_time:.3g} s when it was published.')
        else:
            status.value = (f'{caption} &mdash; '
                            + summarize_level(level, norm_buttons.value))
            draw(level)

    def on_mode(change):
        run_button.layout.display = 'none' if change['new'] == STORED else None
        show()

    def on_method(change):
        """Offer that method's levels, starting at its coarsest."""
        resolutions = loaded.resolutions(change['new'])
        elements_slider.unobserve(show, names='value')
        elements_slider.options = list(resolutions)
        elements_slider.value = resolutions[0]
        elements_slider.observe(show, names='value')
        show()

    def on_run(button=None):
        level = current_level()
        if not level.is_measured:
            run_button.disabled = True
            status.value = (f'Simulating {level.label} on {level.n_elements}'
                            ' axial cells &hellip;')
            try:
                run_level(loaded, level, output_directory)
            finally:
                run_button.disabled = False
        show()

    def on_clear(button=None):
        """Forget the visited levels, keeping the one on the sliders."""
        history.clear()
        show()

    mode_buttons.observe(on_mode, names='value')
    norm_buttons.observe(show, names='value')
    method_slider.observe(on_method, names='value')
    elements_slider.observe(show, names='value')
    run_button.on_click(on_run)
    clear_button.on_click(on_clear)

    show()

    return widgets.VBox([
        widgets.HBox([mode_buttons, norm_buttons]),
        widgets.HBox([
            widgets.VBox([method_slider, elements_slider]),
            widgets.VBox([run_button, clear_button]),
        ]),
        status,
        plot_output,
    ])
