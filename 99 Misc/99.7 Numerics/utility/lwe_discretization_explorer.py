"""Interactive discretization study for the SMA load-wash-elute benchmark.

The explorer compares the outlet of the four component LWE example of
``setting_Col1D_SMA_4comp_LWE_benchmark1`` against the stored reference. The
bulk and the particle are separate spatial domains, so there are four knobs: a
spatial method and a resolution for each. Choosing finite volumes in one and
discontinuous Galerkin in the other is allowed, and is the point of the
study.

Simulating a setting costs seconds to minutes, which is far too slow to
explore with a slider, so every setting is simulated once by
``utility/make_lwe_discretization_results.py``, which records what each one
cost and how wrong it was in
``data/01_spatial_discretization/lwe_discretization/convergence.json``. That
table is a convergence table in the shape cadet-verification writes, and it
is all the explorer reads: a few hundred kilobytes in place of the outlets
themselves.

What it does not contain is a chromatogram per setting. Keeping 756 of them
costs tens of megabytes to say almost the same thing 756 times, so the
chromatogram panel shows the **reference** solution, and the figure says so.
Switching to ``Rerun simulation`` computes the selected setting here and
draws its own outlet over that reference.

The table also defines the slider ranges: the explorer offers exactly the
resolutions it holds, and says in red when a combination was never computed.

Beside the chromatogram sits an accuracy against cost plot that fills up as
settings are visited: one dashed line per spatial method, the setting
currently on the sliders marked in orange. ``Add all points`` puts the whole
stored study on it at once and draws the **Pareto front** through the lower
left edge of the cloud: the settings nothing cheaper is more accurate than.
That front is the table the lesson prints beside it, row for row -- both come
from :func:`pareto_front`, so they are the same settings by construction.
"""

import io
import json
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
                     / 'reference' / 'ref_LWE.h5')

# Reruns triggered from a slider land beside the published results
# of the notebook that offers them.
RERUN_DIRECTORY = (Path('generated') / '01_spatial_discretization'
                   / 'rerun')
COMPONENT_NAMES = ('Salt', 'Component 1', 'Component 2', 'Component 3')

STORED, RERUN = 'Load stored result', 'Rerun simulation'

# The accuracy against cost plot carries one line per (bulk, particle)
# combination, so the two domains are told apart by different channels: the
# bulk method picks the colour, the particle method picks the marker. Orange
# is left out of the colours because it marks whichever setting is selected.
METHOD_COLOR = {
    0: '#9467bd',
    1: '#1f77b4',
    2: '#2ca02c',
    3: '#d62728',
    4: '#8c564b',
    5: '#17becf',
}
METHOD_MARKER = {0: 's', 1: 'o', 2: '^', 3: 'D', 4: 'v', 5: 'P'}


def method_style(bulk_method, particle_method):
    """Colour from the bulk method, marker from the particle method."""
    return (METHOD_COLOR.get(bulk_method, '#7f7f7f'),
            METHOD_MARKER.get(particle_method, 'o'))


CURRENT_COLOR = '#ff7f0e'

FRONT_COLOR = '#222222'


def pareto_front(results):
    """The settings nothing cheaper is more accurate than.

    Walking the settings from cheapest to dearest and keeping every one that
    beats the best error so far. What comes out is non-dominated by
    construction: anything cheaper than a point on the front is less accurate
    than it, and anything more accurate costs more. It is the lower left edge
    of the cloud in the accuracy against cost plot, and the only part of that
    cloud anybody should ever choose a setting from.

    ``results`` is any iterable of :class:`DiscretizationResult`, and the
    front comes back ordered by compute time.
    """
    front, best = [], float('inf')
    for result in sorted(results, key=lambda point: point.simulation_time):
        if not np.isfinite(result.simulation_time * result.error_components):
            continue
        if result.error_components < best:
            best = result.error_components
            front.append(result)

    return front


def spatial_dofs(bulk_method, particle_method, n_axial, n_particle=1,
                 lumped_particle=False):
    """Number of spatial discretization points per component.

    A method is 0 for finite volumes and the polynomial degree for DG. The
    bulk contributes its nodes per axial element, and every one of those
    carries a particle discretized with the particle method's nodes.

    A lumped particle (LRMP) holds one concentration per axial point instead
    of a resolved profile, so it is not discretized and the particle
    arguments are ignored.
    """
    bulk_nodes = 1 if bulk_method == 0 else bulk_method + 1
    particle_nodes = 1 if particle_method == 0 else particle_method + 1
    if lumped_particle:
        return 2 * bulk_nodes * n_axial

    return bulk_nodes * n_axial * (1 + particle_nodes * n_particle)


def method_label(spatial_method):
    return 'FV' if spatial_method == 0 else f'DG{spatial_method}'


def pair_label(bulk_method, particle_method, lumped_particle=False):
    """How a combination reads: bulk method over particle method."""
    if lumped_particle:
        return method_label(bulk_method)

    return (f'{method_label(bulk_method)} / '
            f'{method_label(particle_method)}')


@dataclass
class DiscretizationResult:
    """Outlet of one discretization, with its errors and simulation time."""

    bulk_method: int
    particle_method: int
    n_axial: int
    n_particle: int
    # Only a rerun carries an outlet; a stored setting keeps its numbers
    # alone, and the chromatogram panel falls back to the reference.
    times: np.ndarray = None
    outlet: np.ndarray = None
    error_salt: float = float('nan')
    error_components: float = float('nan')
    simulation_time: float = float('nan')
    source: str = 'stored'
    lumped_particle: bool = False

    @property
    def dofs(self):
        return spatial_dofs(self.bulk_method, self.particle_method,
                            self.n_axial, self.n_particle,
                            self.lumped_particle)

    @property
    def label(self):
        return pair_label(self.bulk_method, self.particle_method,
                          self.lumped_particle)

    @property
    def resolution(self):
        """How the setting reads: with or without a particle resolution."""
        if self.lumped_particle:
            return f'{self.n_axial} axial cells'

        return f'{self.n_axial} axial and {self.n_particle} particle cells'


def _degree(label):
    """A method column of the convergence table, as a spatial method number."""
    return 0 if str(label) == 'FV' else int(label)


class StoredResults:
    """The precomputed study: what each setting cost and how wrong it was.

    Read from ``convergence.json``, the table
    ``make_lwe_discretization_results.py`` writes in the shape
    cadet-verification uses. It holds no outlets, so the chromatogram panel
    is drawn from the reference.
    """

    def __init__(self, directory=DEFAULT_STORE):
        directory = Path(directory)
        convergence_json = directory / 'convergence.json'
        if not convergence_json.exists():
            raise FileNotFoundError(
                f'No convergence table in {directory}. Generate it with '
                '"python utility/make_lwe_discretization_results.py".')

        with io.open(convergence_json, encoding='utf-8') as handle:
            document = json.load(handle)
        self.meta = document.get('meta', {})

        rows = []
        for entry in document['convergence'].values():
            table = entry['outlet']
            for position in range(len(table['DoF'])):
                rows.append({
                    'spatial_method_bulk': _degree(table['$N_d$'][position]),
                    'spatial_method_particle':
                        _degree(table['$N_d^r$'][position]),
                    'n_axial': int(table['$N_e^z$'][position]),
                    'n_particle': int(table['$N_e^r$'][position]),
                    'dofs': int(table['DoF'][position]),
                    'error_components': float(table['Max. error'][position]),
                    'error_salt': float(
                        table.get('Max. error salt',
                                  table['Max. error'])[position]),
                    'simulation_time': float(table['Sim. time'][position]),
                })
        self.runs = pd.DataFrame(rows)
        if self.runs.empty:
            raise ValueError(f'The convergence table in {directory} is empty.')

        # The outlets are not stored any more; the reference stands in.
        self.times = None
        self.reference = None

        # A lumped particle (LRMP) is not discretized, so these studies vary
        # the axial resolution alone.
        self.particle_type = str(self.meta.get('particle_type',
                                               'GENERAL_RATE_PARTICLE'))
        self.lumped_particle = self.particle_type == 'HOMOGENEOUS_PARTICLE'

    @property
    def tolerances(self):
        """The time integration tolerances the stored runs were computed with."""
        return float(self.meta.get('abstol', 1e-12)), \
            float(self.meta.get('reltol', 1e-10))

    def bulk_methods(self):
        """Bulk spatial methods the store holds, in order."""
        return tuple(sorted(int(value) for value
                            in self.runs.spatial_method_bulk.unique()))

    def particle_methods(self):
        """Particle spatial methods the store holds, in order."""
        return tuple(sorted(int(value) for value
                            in self.runs.spatial_method_particle.unique()))

    def axial_counts(self, bulk_method):
        """Axial resolutions stored for one bulk method."""
        group = self.runs[self.runs.spatial_method_bulk == bulk_method]

        return tuple(sorted(int(value) for value in group.n_axial.unique()))

    def particle_counts(self, particle_method):
        """Particle resolutions stored for one particle method."""
        group = self.runs[self.runs.spatial_method_particle == particle_method]

        return tuple(sorted(int(value) for value in group.n_particle.unique()))

    def get(self, bulk_method, particle_method, n_axial, n_particle):
        """The stored result of one setting, or None if it was not computed."""
        matching = self.runs[
            (self.runs.spatial_method_bulk == bulk_method)
            & (self.runs.spatial_method_particle == particle_method)
            & (self.runs.n_axial == n_axial)
            & (self.runs.n_particle == n_particle)]
        if matching.empty:
            return None

        run = matching.iloc[0]

        return DiscretizationResult(
            bulk_method=bulk_method,
            particle_method=particle_method,
            n_axial=n_axial,
            n_particle=n_particle,
            error_salt=float(run.error_salt),
            error_components=float(run.error_components),
            simulation_time=float(run.simulation_time),
            source='stored',
            lumped_particle=self.lumped_particle,
        )

    def all_results(self):
        """Every stored setting, keyed as the explorer's history is.

        Built in one pass over the table rather than by asking for the
        settings one at a time, because the whole study is several hundred
        of them and the button that uses this has to feel instant.
        """
        return {
            (int(run.spatial_method_bulk), int(run.spatial_method_particle),
             int(run.n_axial), int(run.n_particle)): DiscretizationResult(
                bulk_method=int(run.spatial_method_bulk),
                particle_method=int(run.spatial_method_particle),
                n_axial=int(run.n_axial),
                n_particle=int(run.n_particle),
                error_salt=float(run.error_salt),
                error_components=float(run.error_components),
                simulation_time=float(run.simulation_time),
                source='stored',
                lumped_particle=self.lumped_particle,
            )
            for run in self.runs.itertuples()
        }


def read_reference(reference_file=DEFAULT_REFERENCE):
    """The reference outlet and its time axis, from its result file.

    Read with h5py rather than through ``convergence``, so that the stored
    results can be explored without a CADET installation.
    """
    import h5py

    with h5py.File(str(Path(reference_file)), 'r') as file:
        outlet = np.asarray(
            file['output/solution/unit_000/SOLUTION_OUTLET'][()])
        times = np.asarray(file['output/solution/SOLUTION_TIMES'][()])

    return times, outlet


def run_case(bulk_method, particle_method, n_axial, n_particle=1,
             reference=None, output_directory=RERUN_DIRECTORY,
             idas_abstol=1e-12, idas_reltol=1e-10, lumped_particle=False):
    """Simulate one discretization and measure its outlet error.

    ``reference`` is the reference outlet array; the errors are relative to its
    maximum, separately for the salt and for the three protein components.
    With ``lumped_particle`` the column is an LRMP, whose particle holds a
    single concentration and therefore takes no resolution.
    """
    if min(bulk_method, particle_method) < 0 or n_axial < 1 or n_particle < 1:
        raise ValueError('Resolutions must be positive and the methods '
                         'non-negative.')
    if reference is None:
        raise ValueError('A reference outlet is needed to measure the error.')

    particle_type = ('HOMOGENEOUS_PARTICLE' if lumped_particle
                     else 'GENERAL_RATE_PARTICLE')
    model = Cadet()
    model.root = lwe.get_model(
        spatial_method_bulk=bulk_method,
        spatial_method_particle=particle_method,
        particle_type=particle_type,
        axNElem=n_axial,
        parZ=n_particle,
        idas_abstol=idas_abstol,
        idas_reltol=idas_reltol,
    )
    tag = ('' if lumped_particle else f'_par{n_particle}')
    # Git carries no empty folders, so a fresh clone has none.
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    model.filename = str(
        Path(output_directory)
        / f'lwe_{particle_type[:4].lower()}_{method_label(bulk_method)}'
          f'-{method_label(particle_method)}_ax{n_axial}{tag}.h5'
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
        bulk_method=bulk_method,
        particle_method=particle_method,
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

    ``history`` maps ``(bulk method, particle method, axial cells, particle
    cells)`` to the result shown for it. One dashed line per combination of
    methods, ordered by compute time, so that each traces out what refining it
    buys. The bulk method sets the colour and the particle method the marker,
    so a line that changes only one domain is easy to find. The setting on the
    sliders right now is marked on top of its line.

    Once the plot is full enough for the ladders to blur together, the black
    line through its lower left edge is :func:`pareto_front` of whatever is
    on it. With the whole study on show that line is the lesson's table, row
    for row.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    by_pair = {}
    shown = []
    for (bulk, particle, _, _), result in history.items():
        # A setting that was never computed has no error and no runtime to
        # put here; it is still what the sliders say, and is marked by the
        # chromatogram panel alone.
        if not np.isfinite(result.simulation_time * result.error_components):
            continue
        by_pair.setdefault((bulk, particle), []).append(result)
        shown.append(result)

    # A few visited settings deserve fat markers; the whole study on one panel
    # would be a solid block of them, so the ink thins out as it fills up.
    crowded = len(shown) > 60

    for pair in sorted(by_pair):
        color, marker = method_style(*pair)
        points = sorted(by_pair[pair],
                        key=lambda result: result.simulation_time)
        ax.plot([point.simulation_time for point in points],
                [point.error_components * 100 for point in points],
                linestyle='--', linewidth=0.9 if crowded else 1.8,
                color=color, alpha=0.65 if crowded else 1.0,
                marker=marker, markersize=4 if crowded else 8,
                label=pair_label(*pair))

    if crowded:
        # With everything on show the individual ladders blur together, and
        # what is actually being asked is the lower left edge of the cloud.
        # These are the settings the lesson's table lists, drawn from the
        # same function, so the figure and the table cannot drift apart.
        front = pareto_front(shown)
        ax.plot([point.simulation_time for point in front],
                [point.error_components * 100 for point in front],
                linestyle='-', linewidth=1.8, color=FRONT_COLOR, zorder=4,
                label='Pareto front')

    if current_key in history:
        current = history[current_key]
        _, marker = method_style(current.bulk_method, current.particle_method)
        ax.plot(current.simulation_time, current.error_components * 100,
                linestyle='none', marker=marker, markersize=12,
                color=CURRENT_COLOR, zorder=5, label='Current')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute time in seconds')
    ax.set_ylabel(r'Rel. $L^\infty$ error in %')
    ax.grid(True, which='major', alpha=0.4)
    if by_pair:
        # Showing the whole study at once puts every pair of methods in the
        # legend, so it is given more columns and smaller type as it grows
        # rather than covering the data.
        columns = 1 if len(by_pair) <= 6 else (2 if len(by_pair) <= 16 else 3)
        size = 8 if columns < 3 else 6
        ax.legend(fontsize=size, loc='best', ncol=columns,
                  title='bulk / particle', title_fontsize=size,
                  labelspacing=0.3, columnspacing=1.0, handlelength=1.6)

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


# Said under the figure whenever the chromatogram is the reference rather
# than the selected setting, so that nobody reads it as the latter.
REFERENCE_DISCLAIMER = (
    'The chromatogram is the reference solution, not the selected setting.'
    ' The stored study keeps each setting\'s error and runtime but not its'
    ' outlet, which is what the panel on the left is drawn from. Switch to'
    ' "Rerun simulation" to compute the selected setting here and see its'
    ' own chromatogram over this reference.')


def plot_case(result, reference=None, stored_time=None, history=None,
              axes=None, reference_times=None):
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
        key = (result.bulk_method, result.particle_method,
               result.n_axial, result.n_particle)
        plot_performance(
            dict(history if history is not None else {key: result}), key,
            ax=performance_ax)
        performance_ax.set_title('Accuracy against cost (component error)',
                                 fontsize=12)

    # The salt runs two orders of magnitude above the proteins, so it shares
    # the panel on a second axis to the right.
    if result.outlet is not None:
        draw_outlet(axes[0], result.times, result.outlet, reference,
                    COMPONENT_NAMES, first=1,
                    title=f'{result.label}, {result.resolution}')
    else:
        # Nothing was stored for this setting beyond its numbers, so the
        # reference itself is the curve on show.
        draw_outlet(axes[0], reference_times, reference, None,
                    COMPONENT_NAMES, first=1,
                    title='Reference solution')

    figure.tight_layout()
    if result.outlet is None and performance_ax is not None:
        figure.subplots_adjust(bottom=0.21)
        figure.text(0.5, 0.035, REFERENCE_DISCLAIMER, ha='center',
                    va='bottom', fontsize=8.5, color='#555555', wrap=True)

    return figure


def lwe_discretization_explorer(
        store=DEFAULT_STORE, reference_file=DEFAULT_REFERENCE,
        output_directory=RERUN_DIRECTORY, default_method=0,
        show_reference=True):
    """Slider explorer over the stored discretization study.

    The bulk and the particle are discretized separately, so each gets its own
    method slider and its own resolution slider. Mixing the two families is
    allowed: finite volumes in the bulk with DG in the particle, or the other
    way round.

    It starts in ``Load stored result`` mode, where moving a slider shows the
    precomputed outlet immediately. Switching to ``Rerun simulation`` adds a
    button that runs the selected setting through CADET, at the tolerances the
    stored runs used, and reports the time it took on this machine.

    It opens on the coarsest resolution of ``default_method`` in both domains,
    and a resolution drops back to the coarsest whenever its own method
    changes, so that every method is met at the cheap end of its range and
    refined deliberately from there.

    The working directory must be the one holding the stored results and the
    reference. CADET only has to be configured for the rerun mode.
    """
    results = store if isinstance(store, StoredResults) else StoredResults(store)
    bulk_methods = results.bulk_methods()
    particle_methods = results.particle_methods()
    abstol, reltol = results.tolerances
    lumped = results.lumped_particle
    # A study of a lumped particle carries its own reference, because an LRMP
    # is a different model from the GRM that the LWE reference holds.
    reference_times, reference = read_reference(reference_file)
    measured = {}
    # Every setting that has been shown, in the order it was first visited, so
    # that the accuracy against cost plot fills up as the sliders are explored.
    history = {}

    if default_method not in bulk_methods:
        raise ValueError('The default method is not among the stored bulk '
                         'methods.')
    default_particle_method = (default_method if default_method
                               in particle_methods else particle_methods[0])
    coarsest_axial = results.axial_counts(default_method)[0]
    coarsest_particle = results.particle_counts(default_particle_method)[0]

    plot_output = widgets.Output()
    status = widgets.HTML()

    mode_buttons = widgets.ToggleButtons(
        options=[STORED, RERUN], value=STORED, description='Mode',
        tooltips=['Show the result that was computed beforehand',
                  'Run the selected setting through CADET now'],
    )
    bulk_method_slider = widgets.SelectionSlider(
        options=[(method_label(method), method) for method in bulk_methods],
        value=default_method, description='Bulk method',
        continuous_update=False,
    )
    axial_slider = widgets.SelectionSlider(
        options=list(results.axial_counts(default_method)),
        value=coarsest_axial, description='Axial cells',
        continuous_update=False,
    )
    particle_method_slider = widgets.SelectionSlider(
        options=[(method_label(method), method)
                 for method in particle_methods],
        value=default_particle_method, description='Particle method',
        continuous_update=False,
        layout=widgets.Layout(display='none' if lumped else None),
    )
    particle_slider = widgets.SelectionSlider(
        options=list(results.particle_counts(default_particle_method)),
        value=coarsest_particle,
        description='Particle cells', continuous_update=False,
        layout=widgets.Layout(display='none' if lumped else None),
    )
    run_button = widgets.Button(
        description='Run simulation', button_style='primary',
        layout=widgets.Layout(width='160px', display='none'),
    )
    all_button = widgets.Button(
        description='Add all points', icon='th',
        tooltip='Put every stored setting on the accuracy against cost plot',
        layout=widgets.Layout(width='160px'),
    )
    clear_button = widgets.Button(
        description='Clear history', icon='eraser',
        tooltip='Start the accuracy against cost plot over',
        layout=widgets.Layout(width='160px'),
    )

    def current_key():
        return (bulk_method_slider.value, particle_method_slider.value,
                axial_slider.value, particle_slider.value)

    def draw(result, stored_time=None, record=True):
        """Show one setting, and keep it on the performance plot.

        ``record`` is off for a setting that was never computed: it has no
        numbers to put on that plot, but the plot is still redrawn, so that
        the picture always belongs to the sliders as they stand.
        """
        key = (result.bulk_method, result.particle_method,
               result.n_axial, result.n_particle)
        if record:
            history[key] = result
        with plot_output:
            plot_output.clear_output(wait=True)
            figure = plot_case(result, reference if show_reference else None,
                               stored_time=stored_time, history=history,
                               reference_times=reference_times)
            plt.show()
            plt.close(figure)

    def show(change=None):
        """Show what the current mode has for the current setting."""
        key = current_key()
        resolution = (f'{key[2]} axial cells' if lumped else
                      f'{key[2]} axial and {key[3]} particle cells')
        caption = (f'{pair_label(key[0], key[1], lumped)}, {resolution}:'
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
            status.value = (
                f'<span style="color:#b00020">{caption} &mdash; <b>not in'
                ' the stored results.</b> Switch to <b>Rerun simulation</b>'
                ' to compute it.</span>')
            draw(DiscretizationResult(*key, lumped_particle=lumped),
                 record=False)

    def on_mode(change):
        run_button.layout.display = 'none' if change['new'] == STORED else None
        show()

    def _reset(slider, counts):
        """Offer `counts` on `slider`, starting at the coarsest, once."""
        slider.unobserve(show, names='value')
        slider.options = list(counts)
        slider.value = counts[0]
        slider.observe(show, names='value')

    def on_bulk_method(change):
        """Offer the axial resolutions stored for the new bulk method.

        Methods differ by more than an order of magnitude in how much they
        need, so carrying a resolution over would land on a setting that is
        needlessly fine for one method and far too coarse for another. Only
        the domain whose method changed is reset; the other one stays put, so
        that one domain can be varied against a fixed other.
        """
        _reset(axial_slider, results.axial_counts(change['new']))
        show()

    def on_particle_method(change):
        """The same for the particle domain."""
        _reset(particle_slider, results.particle_counts(change['new']))
        show()

    def on_run(button=None):
        key = current_key()
        if key not in measured:
            run_button.disabled = True
            status.value = (
                f'Simulating {pair_label(key[0], key[1], lumped)} with'
                f' {key[2]} axial cells'
                + ('' if lumped else f' and {key[3]} particle cells')
                + ' &hellip;')
            try:
                measured[key] = run_case(
                    *key, reference=reference, output_directory=output_directory,
                    idas_abstol=abstol, idas_reltol=reltol,
                    lumped_particle=lumped)
            finally:
                run_button.disabled = False
        show()

    def on_all(button=None):
        """Put the whole stored study on the plot, not just what was visited.

        A setting that was rerun here keeps the result measured on this
        machine, so pressing this does not quietly replace a measured point
        with the stored one beside it.
        """
        for key, result in results.all_results().items():
            history.setdefault(key, result)
        show()

    def on_clear(button=None):
        """Forget the visited settings, keeping the one on the sliders."""
        history.clear()
        show()

    mode_buttons.observe(on_mode, names='value')
    bulk_method_slider.observe(on_bulk_method, names='value')
    particle_method_slider.observe(on_particle_method, names='value')
    for slider in (axial_slider, particle_slider):
        slider.observe(show, names='value')
    run_button.on_click(on_run)
    all_button.on_click(on_all)
    clear_button.on_click(on_clear)

    show()

    return widgets.VBox([
        mode_buttons,
        widgets.HBox([
            widgets.VBox([bulk_method_slider, axial_slider,
                          particle_method_slider, particle_slider]),
            widgets.VBox([run_button, all_button, clear_button]),
        ]),
        status,
        plot_output,
    ])
