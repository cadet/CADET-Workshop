"""Interactive time-tolerance study for the kinetic Langmuir LRM pulse.

The spatial discretization is fixed, so the reference is the same model solved
with very tight IDAS tolerances. The L-infinity outlet error then isolates the
time integration error caused by RELTOL and ABSTOL.
"""

from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from cadet import Cadet

from utility import convergence
from utility import setting_Col1D_langLRM_1comp_pulse as case

REFERENCE_RELTOL = 1e-14
REFERENCE_ABSTOL = 1e-14
TOLERANCE_EXPONENTS = range(-14, 0)


def _discretization_label(polynomial_degree, n_elements):
    if polynomial_degree > 0:
        return f'DG, p={polynomial_degree}, {n_elements} elements'
    return f'FV (WENO3), {n_elements} cells'


def run_simulation(filename, polynomial_degree, n_elements, reltol, abstol):
    """Run one simulation and return (outlet, runtime) or raise on failure."""
    model = Cadet()
    model.root = case.get_model(
        polynomial_degree=polynomial_degree, n_elements=n_elements,
        reltol=reltol, abstol=abstol,
    )
    model.filename = str(filename)
    model.save()
    run_info = model.run_simulation()
    if run_info.return_code != 0:
        raise RuntimeError(run_info.error_message)

    outlet = np.asarray(convergence.get_outlet(model, unit='001')).reshape(-1)
    runtime = convergence.get_compute_time(model)
    return outlet, runtime


def make_explorer(output_directory=Path('data'), polynomial_degree=3, n_elements=32,
                  reltol=1e-2, abstol=1e-4):
    """Build the RELTOL/ABSTOL slider widget on a fixed spatial discretization.

    CADET must be configured before calling this function.
    """
    output_directory = Path(output_directory)
    times = case.solution_times()
    discretization = _discretization_label(polynomial_degree, n_elements)
    prefix = f'langLRM_1comp_pulse_P{polynomial_degree}_N{n_elements}'

    reference, reference_runtime = run_simulation(
        output_directory / f'{prefix}_reference.h5',
        polynomial_degree, n_elements, REFERENCE_RELTOL, REFERENCE_ABSTOL,
    )
    history = {}

    def show(reltol, abstol):
        try:
            outlet, runtime = run_simulation(
                output_directory / f'{prefix}_tolerance_study.h5',
                polynomial_degree, n_elements, reltol, abstol,
            )
        except RuntimeError as error:
            print(f'Simulation failed for RELTOL={reltol:.0e}, ABSTOL={abstol:.0e}:\n{error}')
            return

        pointwise_error = np.abs(outlet - reference)
        error = np.max(pointwise_error)
        history[(reltol, abstol)] = (error, runtime)

        fig, (solution_ax, error_ax, runtime_ax) = plt.subplots(1, 3, figsize=(18, 4.5))

        solution_ax.plot(times, reference, 'k-', linewidth=2, label='Reference')
        solution_ax.plot(times, outlet, '--', color='tab:orange', label='Simulation')
        solution_ax.set_title('Outlet solution')
        solution_ax.set_xlabel('Time [s]')
        solution_ax.set_ylabel(r'Outlet concentration $[mol \cdot m^{-3}]$')
        solution_ax.legend()

        error_ax.semilogy(times, np.maximum(pointwise_error, 1e-16), color='tab:orange')
        error_ax.axhline(error, color='tab:red', linestyle=':',
                         label=rf'$L^\infty$ = {error:.2e}')
        error_ax.set_title('Pointwise outlet error')
        error_ax.set_xlabel('Time [s]')
        error_ax.set_ylabel(r'$|c - c_{\mathrm{ref}}|$')
        error_ax.grid(True, which='both', alpha=0.3)
        error_ax.legend()

        errors, runtimes = np.array(list(history.values())).T
        runtime_ax.loglog(runtimes, errors, 'o', color='lightgray', label='Previous runs')
        runtime_ax.loglog(runtime, error, 'o', color='tab:orange', markersize=10,
                          label='Current run')
        runtime_ax.axvline(reference_runtime, color='k', linestyle=':',
                           label=f'Reference ({reference_runtime:.2e} s)')
        runtime_ax.set_title('Work-precision')
        runtime_ax.set_xlabel('Simulation time [s]')
        runtime_ax.set_ylabel(r'$L^\infty$ outlet error')
        runtime_ax.grid(True, which='both', alpha=0.3)
        runtime_ax.legend()

        fig.tight_layout()
        plt.show()

        print(
            f'Discretization:   {discretization}\n'
            f'Tolerances:       RELTOL = {reltol:.0e}, ABSTOL = {abstol:.0e}\n'
            f'Simulation time:  {runtime:.3e} s\n'
            f'L-infinity error: {error:.3e} '
            f'({error / np.max(np.abs(reference)) * 100:.3f}% of peak)\n'
            f'{"-" * 60}\n'
            f'Reference tolerances: RELTOL = {REFERENCE_RELTOL:.0e}, '
            f'ABSTOL = {REFERENCE_ABSTOL:.0e}'
        )

    options = [(f'1e{i}', float(f'1e{i}')) for i in TOLERANCE_EXPONENTS]
    slider_style = {'description_width': 'initial'}
    slider_layout = widgets.Layout(width='500px')
    return widgets.interactive(
        show,
        reltol=widgets.SelectionSlider(
            options=options, value=reltol, description='RELTOL',
            continuous_update=False, style=slider_style, layout=slider_layout,
        ),
        abstol=widgets.SelectionSlider(
            options=options, value=abstol, description='ABSTOL',
            continuous_update=False, style=slider_style, layout=slider_layout,
        ),
    )
