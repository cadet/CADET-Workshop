"""FV/DG refinement study for the kinetic linear LRM and CASEMA reference.

Errors are discrete absolute L-infinity norms on the reference output times.
EOC uses consecutive grids; the corresponding log-log slope is its negative.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
import numpy as np
from cadet import Cadet

from utility import convergence
from utility import setting_Col1D_linLRM_1comp_benchmark1 as LRM_linear_benchmark


@dataclass
class ReferenceSolution:
    """Analytical outlet concentrations and their sampling times."""

    times: np.ndarray
    outlet: np.ndarray


@dataclass
class MethodResults:
    """One error and runtime per grid; EOC starts at the second grid."""

    errors: np.ndarray
    eoc: np.ndarray
    runtimes: np.ndarray
    finest_outlet: np.ndarray


def load_reference(filename):
    """Load the CASEMA column outlet (unit 001)."""
    times = convergence.get_solution_times(str(filename))
    outlet = np.asarray(convergence.get_outlet(str(filename), unit='001')).reshape(-1)
    if times.shape != outlet.shape or not np.all(np.isfinite(outlet)):
        raise ValueError('The reference has invalid outlet concentrations or times.')
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('Reference times must be finite and strictly increasing.')
    return ReferenceSolution(times, outlet)


def calculate_eoc(coarse_error, fine_error, refinement_ratio):
    """Return log(coarse error / fine error) / log(refinement ratio)."""
    return np.log(coarse_error / fine_error) / np.log(refinement_ratio)


def run_refinement_study(reference, axial_cell_counts, methods, output_directory):
    """Run each FV/DG grid, print errors and EOC, and return results by method.

    ``methods`` maps display names to polynomial degrees (0 selects FV).
    CADET must be configured before calling this function. Simulation files
    are saved in the existing output directory, one per method and grid.
    """
    counts = np.asarray(axial_cell_counts)
    if (counts.ndim != 1 or counts.size < 2 or not np.all(np.isfinite(counts))
            or np.any(counts <= 0) or np.any(counts != np.floor(counts))
            or np.any(np.diff(counts) <= 0)):
        raise ValueError('Provide at least two strictly increasing positive integer grids.')
    if not methods or any(
        not isinstance(degree, (int, np.integer)) or degree < 0
        for degree in methods.values()
    ):
        raise ValueError('Provide nonnegative integer degrees: 0 selects FV, positive values select DG.')

    results = {}
    header = (
        f"{'Method':>6} {'Cells/elements':>14} {'L-infinity error':>18}"
        f" {'EOC':>10} {'Runtime [s]':>14}"
    )
    print(header)
    for method_name, degree in methods.items():
        print('-' * len(header))
        errors, eoc_values, runtimes = [], [], []
        for grid_index, n_cells in enumerate(axial_cell_counts):
            model = Cadet()
            model.root = LRM_linear_benchmark.get_model(
                spatial_method_bulk=degree,
                axRefinement=n_cells / 8,  # The benchmark starts with 8 cells/elements.
                idas_reftol=1e-12,
            )
            model.root.input.solver.user_solution_times = reference.times
            model.filename = str(Path(output_directory) / f'lin1comp_{method_name}_{n_cells}.h5')
            model.save()
            run_info = model.run_simulation()
            if run_info.return_code != 0:
                raise RuntimeError(f'{method_name} with {n_cells} cells/elements failed: {run_info}')

            times = convergence.get_solution_times(model)
            outlet = np.asarray(convergence.get_outlet(model, unit='001')).reshape(-1)
            np.testing.assert_array_equal(times, reference.times)
            if outlet.shape != reference.outlet.shape or not np.all(np.isfinite(outlet)):
                raise ValueError(f'Invalid outlet for {method_name}, {n_cells} cells/elements.')

            error = np.linalg.norm(outlet - reference.outlet, ord=np.inf)
            if not np.isfinite(error) or error <= 0:
                raise ValueError('Logarithmic convergence plots require finite positive errors.')
            runtime = convergence.get_compute_time(model)
            if not np.isfinite(runtime) or runtime <= 0:
                raise ValueError(f'Invalid runtime for {method_name}, {n_cells} cells/elements.')

            eoc_text = '-'
            if grid_index > 0:
                ratio = n_cells / axial_cell_counts[grid_index - 1]
                eoc = calculate_eoc(errors[-1], error, ratio)
                eoc_values.append(eoc)
                eoc_text = f'{eoc:.3f}'
            errors.append(error)
            runtimes.append(runtime)
            print(f'{method_name:>6} {n_cells:14d} {error:18.6e} {eoc_text:>10} {runtime:14.3e}')

        results[method_name] = MethodResults(
            np.array(errors), np.array(eoc_values), np.array(runtimes), outlet
        )
    return results


def plot_refinement_study(reference, axial_cell_counts, results):
    """Show solutions, L-infinity convergence, and grouped runtime columns."""
    fig, (solution_ax, convergence_ax, runtime_ax) = plt.subplots(1, 3, figsize=(18, 4.5))
    solution_ax.plot(reference.times, reference.outlet, 'k-', linewidth=2, label='CASEMA reference')
    for method_name, result in results.items():
        solution_ax.plot(
            reference.times, result.finest_outlet, '--',
            label=f'{method_name}, N={axial_cell_counts[-1]}',
        )
        convergence_ax.loglog(axial_cell_counts, result.errors, 'o-', label=method_name)
    solution_ax.set_title('Outlet solution')
    solution_ax.set_xlabel('Time [s]')
    solution_ax.set_ylabel(r'Outlet concentration $[mol \cdot M^{-3}]$')
    solution_ax.legend()

    convergence_ax.set_title('Grid convergence')
    convergence_ax.set_xlabel('Number of axial cells (FV) / elements (DG)')
    convergence_ax.set_ylabel(r'$L^\infty$ outlet error')
    convergence_ax.xaxis.set_major_locator(FixedLocator(axial_cell_counts))
    convergence_ax.xaxis.set_major_formatter(ScalarFormatter())
    convergence_ax.xaxis.set_minor_locator(NullLocator())
    convergence_ax.grid(True, which='both', alpha=0.3)
    convergence_ax.legend()

    group_positions = np.arange(len(axial_cell_counts))
    bar_width = 0.8 / len(results)
    for method_index, (method_name, result) in enumerate(results.items()):
        offset = (method_index - (len(results) - 1) / 2) * bar_width
        runtime_ax.bar(group_positions + offset, result.runtimes, width=bar_width, label=method_name)
    runtime_ax.set_title('Simulation runtime')
    runtime_ax.set_xlabel('Number of axial cells (FV) / elements (DG)')
    runtime_ax.set_ylabel('Simulation time [s]')
    runtime_ax.set_xticks(group_positions, labels=axial_cell_counts)
    runtime_ax.set_axisbelow(True)
    runtime_ax.grid(axis='y', alpha=0.3)
    runtime_ax.legend()
    fig.tight_layout()
    plt.show()
    return fig


def print_max_slopes(axial_cell_counts, results):
    """Report each method's steepest decay slope and refinement interval."""
    print('\nMaximum decay slopes (most negative log-log slopes):')
    for method_name, result in results.items():
        best_interval = np.argmax(result.eoc)
        coarse = axial_cell_counts[best_interval]
        fine = axial_cell_counts[best_interval + 1]
        print(f'{method_name}: {-result.eoc[best_interval]:.3f} ({coarse}->{fine})')
