"""Interactive discretization study for the SMA load-wash-elute benchmark.

The explorer compares the outlet of the four component LWE example of
``setting_Col1D_SMA_4comp_LWE_benchmark1`` against the stored reference, for a
chosen spatial method, axial resolution and particle resolution.

Simulating a setting costs seconds to minutes, which is far too slow to explore
with a slider, so every setting is simulated once by
``utility/make_lwe_discretization_results.py`` and stored under
``data/01_spatial_discretization/lwe_discretization``. The sliders load those
results, and the mode
switch reruns the selected setting live when you want to see it happen.

The stored result file also defines the slider ranges: the explorer offers
exactly the resolutions it holds.

Above the two chromatograms sits an accuracy against cost plot that fills up as
settings are visited: one dashed line per spatial method, the setting currently
on the sliders marked in orange, and the numbers of that setting beside it.
"""

from dataclasses import dataclass
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cadet import Cadet

from utility import convergence
from utility import setting_Col1D_SMA_4comp_LWE_benchmark1 as lwe
from utility.convergence_explorer import draw_outlet


DEFAULT_STORE = (Path('data') / '01_spatial_discretization'
                 / 'lwe_discretization')
DEFAULT_REFERENCE = (Path('data') / '02_time_discretization'
                     / 'ref_LWE.h5')

# Reruns triggered from a slider land beside the published results
# of the notebook that offers them.
RERUN_DIRECTORY = (Path('data') / '01_spatial_discretization'
                   / 'rerun')
COMPONENT_NAMES = ('Salt', 'Component 1', 'Component 2', 'Component 3')

STORED, RERUN = 'Load stored result', 'Rerun simulation'

# One colour and marker per spatial method for the accuracy against cost plot:
# squares for the finite volumes, circles for the DG degrees. Orange is left
# out of the methods because it marks whichever setting is selected.
METHOD_STYLE = {
    0: ('#9467bd', 's'),
    1: ('#1f77b4', 'o'),
    2: ('#2ca02c', 'o'),
    3: ('#d62728', 'o'),
    4: ('#8c564b', 'o'),
    5: ('#17becf', 'o'),
}
CURRENT_COLOR = '#ff7f0e'


def spatial_dofs(spatial_method, n_axial, n_particle=1, lumped_particle=False):
    """Number of spatial discretization points per component.

    ``spatial_method`` is 0 for finite volumes and the polynomial degree for DG.
    A lumped particle (LRMP) holds one concentration per axial point instead of
    a resolved profile, so it is not discretized and ``n_particle`` is ignored.
    """
    nodes = 1 if spatial_method == 0 else spatial_method + 1
    if lumped_particle:
        return 2 * nodes * n_axial

    return nodes * n_axial * (1 + nodes * n_particle)


def method_label(spatial_method):
    return 'FV' if spatial_method == 0 else f'DG{spatial_method}'


@dataclass
class DiscretizationResult:
    """Outlet of one discretization, with its errors and simulation time."""

    spatial_method: int
    n_axial: int
    n_particle: int
    times: np.ndarray
    outlet: np.ndarray
    error_salt: float
    error_components: float
    simulation_time: float
    source: str = 'stored'
    lumped_particle: bool = False

    @property
    def dofs(self):
        return spatial_dofs(self.spatial_method, self.n_axial, self.n_particle,
                            self.lumped_particle)

    @property
    def label(self):
        return method_label(self.spatial_method)

    @property
    def resolution(self):
        """How the setting reads: with or without a particle resolution."""
        if self.lumped_particle:
            return f'{self.n_axial} axial cells'

        return f'{self.n_axial} axial and {self.n_particle} particle cells'


class StoredResults:
    """The precomputed study: one outlet and one timing per setting."""

    def __init__(self, directory=DEFAULT_STORE):
        directory = Path(directory)
        runs_csv = directory / 'runs.csv'
        solutions_npz = directory / 'solutions.npz'
        if not (runs_csv.exists() and solutions_npz.exists()):
            raise FileNotFoundError(
                f'No stored results in {directory}. Generate them with '
                '"python utility/make_lwe_discretization_results.py".')

        self.runs = pd.read_csv(runs_csv)
        with np.load(solutions_npz) as stored:
            self.times = stored['times']
            self.reference = (np.asarray(stored['reference'])
                              if 'reference' in stored.files else None)
            self.solutions = {name: stored[name] for name in stored.files
                              if name not in ('times', 'reference')}
        if self.runs.empty:
            raise ValueError(f'The stored results in {directory} are empty.')

        # A lumped particle (LRMP) is not discretized, so these studies vary
        # the axial resolution alone.
        self.particle_type = str(self.runs.get(
            'particle_type', pd.Series(['GENERAL_RATE_PARTICLE'])).iloc[0])
        self.lumped_particle = self.particle_type == 'HOMOGENEOUS_PARTICLE'

    @property
    def tolerances(self):
        """The time integration tolerances the stored runs were computed with."""
        return float(self.runs.abstol.iloc[0]), float(self.runs.reltol.iloc[0])

    def resolutions(self):
        """``{spatial method: (axial counts, particle counts)}`` that are stored."""
        offered = {}
        for method, group in self.runs.groupby('spatial_method', sort=True):
            offered[int(method)] = (
                tuple(sorted(group.n_axial.unique())),
                tuple(sorted(group.n_particle.unique())),
            )

        return offered

    def get(self, spatial_method, n_axial, n_particle):
        """The stored result of one setting, or None if it was not computed."""
        matching = self.runs[(self.runs.spatial_method == spatial_method)
                             & (self.runs.n_axial == n_axial)
                             & (self.runs.n_particle == n_particle)]
        if matching.empty:
            return None

        run = matching.iloc[0]

        return DiscretizationResult(
            spatial_method=spatial_method,
            n_axial=n_axial,
            n_particle=n_particle,
            times=self.times,
            outlet=self.solutions[run.key],
            error_salt=float(run.error_salt),
            error_components=float(run.error_components),
            simulation_time=float(run.simulation_time),
            source='stored',
            lumped_particle=self.lumped_particle,
        )


def read_reference(reference_file=DEFAULT_REFERENCE):
    """The reference outlet, read straight from its result file.

    Read with h5py rather than through ``convergence``, so that the stored
    results can be explored without a CADET installation.
    """
    import h5py

    with h5py.File(str(Path(reference_file)), 'r') as file:
        return np.asarray(file['output/solution/unit_000/SOLUTION_OUTLET'][()])


def run_case(spatial_method, n_axial, n_particle=1, reference=None,
             output_directory=RERUN_DIRECTORY, idas_abstol=1e-12, idas_reltol=1e-10,
             lumped_particle=False):
    """Simulate one discretization and measure its outlet error.

    ``reference`` is the reference outlet array; the errors are relative to its
    maximum, separately for the salt and for the three protein components.
    With ``lumped_particle`` the column is an LRMP, whose particle holds a
    single concentration and therefore takes no resolution.
    """
    if spatial_method < 0 or n_axial < 1 or n_particle < 1:
        raise ValueError('Resolutions must be positive and the method non-negative.')
    if reference is None:
        raise ValueError('A reference outlet is needed to measure the error.')

    particle_type = ('HOMOGENEOUS_PARTICLE' if lumped_particle
                     else 'GENERAL_RATE_PARTICLE')
    model = Cadet()
    model.root = lwe.get_model(
        spatial_method_bulk=spatial_method,
        spatial_method_particle=spatial_method,
        particle_type=particle_type,
        axNElem=n_axial,
        parZ=n_particle,
        idas_abstol=idas_abstol,
        idas_reltol=idas_reltol,
    )
    tag = ('' if lumped_particle else f'_par{n_particle}')
    model.filename = str(
        Path(output_directory)
        / f'lwe_{particle_type[:4].lower()}_{method_label(spatial_method)}'
          f'_ax{n_axial}{tag}.h5'
    )
    model.save()
    run_info = model.run_simulation()
    if run_info.return_code != 0:
        raise RuntimeError(f'LWE simulation failed: {run_info}')

    times = np.asarray(convergence.get_solution_times(model.filename))
    outlet = np.asarray(convergence.get_outlet(model.filename, unit='000'))
    if outlet.shape != reference.shape:
        raise ValueError('Solution and reference are sampled differently.')

    return DiscretizationResult(
        spatial_method=spatial_method,
        n_axial=n_axial,
        n_particle=n_particle,
        times=times,
        outlet=outlet,
        error_salt=float(
            np.max(abs(reference[:, 0] - outlet[:, 0])) / np.max(abs(reference[:, 0]))),
        error_components=float(
            np.max(abs(reference[:, 1:] - outlet[:, 1:])) / np.max(abs(reference[:, 1:]))),
        simulation_time=float(convergence.get_compute_time(model.filename)),
        source='measured',
        lumped_particle=lumped_particle,
    )


def plot_performance(history, current_key=None, ax=None):
    """Error against compute time for every setting visited so far.

    ``history`` maps ``(spatial method, axial cells, particle cells)`` to the
    result shown for it. One dashed line per spatial method, ordered by compute
    time, so that each method traces out what refining it buys. The setting on
    the sliders right now is marked on top of its line.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    by_method = {}
    for (method, _, _), result in history.items():
        by_method.setdefault(method, []).append(result)

    for method in sorted(by_method):
        color, marker = METHOD_STYLE.get(method, ('#7f7f7f', 'o'))
        points = sorted(by_method[method], key=lambda result: result.simulation_time)
        ax.plot([point.simulation_time for point in points],
                [point.error_components * 100 for point in points],
                linestyle='--', linewidth=1.8, color=color,
                marker=marker, markersize=8, label=method_label(method))

    if current_key in history:
        current = history[current_key]
        _, marker = METHOD_STYLE.get(current.spatial_method, ('#7f7f7f', 'o'))
        ax.plot(current.simulation_time, current.error_components * 100,
                linestyle='none', marker=marker, markersize=12,
                color=CURRENT_COLOR, zorder=5, label='Current')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute time in seconds')
    ax.set_ylabel(r'Rel. $L^\infty$ error in %')
    ax.grid(True, which='major', alpha=0.4)
    if by_method:
        ax.legend(fontsize=9, loc='best')

    return ax


def summarize_case(result, stored_time=None):
    """The numbers of one result, as the one line above the plot."""
    if result.source == 'measured':
        timing = f'{result.simulation_time:.3g} s, measured on this machine'
        if stored_time is not None:
            timing += f' (stored: {stored_time:.3g} s)'
    else:
        timing = f'{result.simulation_time:.3g} s, as stored'

    return (f'rel. max. error {result.error_components * 100:.3g} % on the'
            f' components and {result.error_salt * 100:.3g} % on the salt,'
            f' in {timing}')


def plot_case(result, reference=None, stored_time=None, history=None,
              axes=None):
    """Plot one result: accuracy against cost beside the outlet.

    ``history`` is passed on to :func:`plot_performance`; without it only the
    current setting is marked.
    """
    if axes is None:
        figure = plt.figure(figsize=(13, 5.4))
        grid = figure.add_gridspec(1, 2, width_ratios=(1, 1.15))
        performance_ax = figure.add_subplot(grid[0, 0])
        axes = [figure.add_subplot(grid[0, 1])]
    else:
        figure = axes[0].figure
        performance_ax = None

    if performance_ax is not None:
        key = (result.spatial_method, result.n_axial, result.n_particle)
        plot_performance(dict(history or {key: result}), key, ax=performance_ax)
        performance_ax.set_title('Accuracy against cost (component error)',
                                 fontsize=12)

    # The salt runs two orders of magnitude above the proteins, so it shares
    # the panel on a second axis to the right.
    draw_outlet(axes[0], result.times, result.outlet, reference,
                COMPONENT_NAMES, first=1,
                title=f'{result.label}, {result.resolution}')

    figure.tight_layout()

    return figure


def lwe_discretization_explorer(
        store=DEFAULT_STORE, reference_file=DEFAULT_REFERENCE,
        output_directory=RERUN_DIRECTORY, default_method=0, show_reference=True):
    """Slider explorer over the stored discretization study.

    It starts in ``Load stored result`` mode, where moving a slider shows the
    precomputed outlet immediately. Switching to ``Rerun simulation`` adds a
    button that runs the selected setting through CADET, at the tolerances the
    stored runs used, and reports the time it took on this machine.

    It opens on the coarsest resolution of ``default_method``, and drops back
    to the coarsest one whenever the method changes, so that every method is
    met at the cheap end of its range and refined deliberately from there.

    The working directory must be the one holding the stored results and the
    reference. CADET only has to be configured for the rerun mode.
    """
    results = store if isinstance(store, StoredResults) else StoredResults(store)
    resolutions = results.resolutions()
    abstol, reltol = results.tolerances
    lumped = results.lumped_particle
    # A study of a lumped particle carries its own reference, because an LRMP
    # is a different model from the GRM that data/ref_LWE.h5 holds.
    reference = (results.reference if results.reference is not None
                 else read_reference(reference_file))
    measured = {}
    # Every setting that has been shown, in the order it was first visited, so
    # that the accuracy against cost plot fills up as the sliders are explored.
    history = {}

    if default_method not in resolutions:
        raise ValueError('The default method is not among the stored methods.')
    coarsest_axial, coarsest_particle = (counts[0] for counts
                                         in resolutions[default_method])

    plot_output = widgets.Output()
    status = widgets.HTML()

    mode_buttons = widgets.ToggleButtons(
        options=[STORED, RERUN], value=STORED, description='Mode',
        tooltips=['Show the result that was computed beforehand',
                  'Run the selected setting through CADET now'],
    )
    method_slider = widgets.SelectionSlider(
        options=[(method_label(method), method) for method in resolutions],
        value=default_method, description='Method', continuous_update=False,
    )
    axial_slider = widgets.SelectionSlider(
        options=list(resolutions[default_method][0]), value=coarsest_axial,
        description='Axial cells', continuous_update=False,
    )
    particle_slider = widgets.SelectionSlider(
        options=list(resolutions[default_method][1]), value=coarsest_particle,
        description='Particle cells', continuous_update=False,
        layout=widgets.Layout(display='none' if lumped else None),
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

    def current_key():
        return (method_slider.value, axial_slider.value, particle_slider.value)

    def draw(result, stored_time=None):
        key = (result.spatial_method, result.n_axial, result.n_particle)
        history[key] = result
        with plot_output:
            plot_output.clear_output(wait=True)
            figure = plot_case(result, reference if show_reference else None,
                               stored_time=stored_time, history=history)
            plt.show()
            plt.close(figure)

    def show(change=None):
        """Show what the current mode has for the current setting."""
        key = current_key()
        resolution = (f'{key[1]} axial cells' if lumped else
                      f'{key[1]} axial and {key[2]} particle cells')
        caption = (f'{method_label(key[0])}, {resolution}:'
                   f' {spatial_dofs(*key, lumped_particle=lumped)}'
                   ' points per component')
        stored = results.get(*key)

        if mode_buttons.value == RERUN and key in measured:
            stored_time = stored.simulation_time if stored else None
            status.value = (f'{caption} &mdash; '
                            + summarize_case(measured[key], stored_time))
            draw(measured[key], stored_time)
        elif mode_buttons.value == RERUN:
            # Say what it cost when it was stored, so that nobody starts a ten
            # minute simulation without meaning to.
            expected = ('' if stored is None else
                        f' It took {stored.simulation_time:.3g} s when stored.')
            status.value = (f'{caption} &mdash; press <b>Run simulation</b> to'
                            f' run it now at abstol {abstol:.0e},'
                            f' reltol {reltol:.0e}.{expected}')
        elif stored is not None:
            status.value = f'{caption} &mdash; ' + summarize_case(stored)
            draw(stored)
        else:
            status.value = (f'{caption} &mdash; not in the stored results.'
                            ' Switch to <b>Rerun simulation</b> to compute it.')

    def on_mode(change):
        run_button.layout.display = 'none' if change['new'] == STORED else None
        show()

    def on_method(change):
        """Offer the resolutions stored for the method, starting at its coarsest.

        Methods differ by more than an order of magnitude in how much they need,
        so carrying a resolution over would land on a setting that is needlessly
        fine for one method and far too coarse for another.
        """
        axial_counts, particle_counts = resolutions[change['new']]
        for slider, counts in ((axial_slider, axial_counts),
                               (particle_slider, particle_counts)):
            slider.unobserve(show, names='value')
            slider.options = list(counts)
            slider.value = counts[0]
            slider.observe(show, names='value')
        show()

    def on_run(button=None):
        key = current_key()
        if key not in measured:
            run_button.disabled = True
            status.value = (
                f'Simulating {method_label(key[0])} with {key[1]} axial cells'
                + ('' if lumped else f' and {key[2]} particle cells')
                + ' &hellip;')
            try:
                measured[key] = run_case(
                    *key, reference=reference, output_directory=output_directory,
                    idas_abstol=abstol, idas_reltol=reltol,
                    lumped_particle=lumped)
            finally:
                run_button.disabled = False
        show()

    def on_clear(button=None):
        """Forget the visited settings, keeping the one on the sliders."""
        history.clear()
        show()

    mode_buttons.observe(on_mode, names='value')
    method_slider.observe(on_method, names='value')
    for slider in (axial_slider, particle_slider):
        slider.observe(show, names='value')
    run_button.on_click(on_run)
    clear_button.on_click(on_clear)

    show()

    return widgets.VBox([
        mode_buttons,
        widgets.HBox([
            widgets.VBox([method_slider, axial_slider, particle_slider]),
            widgets.VBox([run_button, clear_button]),
        ]),
        status,
        plot_output,
    ])
