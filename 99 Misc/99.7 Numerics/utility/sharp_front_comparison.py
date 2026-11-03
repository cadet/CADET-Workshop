"""Interactive pulse and sustained-step comparisons of FV/WENO and DG."""

from dataclasses import dataclass
from pathlib import Path
from math import lcm

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from cadet import Cadet

from utility import convergence
from utility import setting_Col1D_transport_realistic as transport


DEFAULT_METHODS = (('FV', 3), ('DG', 1), ('DG', 2), ('DG', 3))

# The reference run resolves the dispersive front with this many axial cells
# across it, up to a cap that keeps a click interactive. Ten per front width
# is converged wherever the cap allows it; at the sharpest dispersions the cap
# binds, and `reference_outlet` says so rather than pretending otherwise.
REFERENCE_NODES_PER_WIDTH = 10
REFERENCE_NODES_CAP = 1024

# Finite volumes with WENO3 carry the reference. The scheme is monotone, so
# the curve the methods are judged against can never itself overshoot; where
# the cap leaves it under-resolved it smears instead of ringing, which is the
# failure mode that misleads least in a study about oscillations.
REFERENCE_WENO_ORDER = 3

# Reruns triggered from a slider land beside the published results
# of the notebook that offers them.
RERUN_DIRECTORY = (Path('generated') / '01_spatial_discretization'
                   / 'rerun')


def method_settings(methods):
    """Map (family, order) tuples to display labels and solver settings."""
    settings = {}
    for method in methods:
        if not isinstance(method, tuple) or len(method) != 2:
            raise ValueError("Methods must be tuples such as ('FV', 1) or ('DG', 3).")
        family, order = method
        if not isinstance(order, int) or isinstance(order, bool):
            raise ValueError('Method orders must be integers.')
        if family == 'FV' and order in (1, 2, 3):
            settings[f'FV, WENO{order}'] = (0, order)
        elif family == 'DG' and order > 0:
            settings[f'DG{order}'] = (order, None)
        else:
            raise ValueError(
                f'Unsupported method {method!r}. Use FV orders 1-3 or positive DG degrees. '
                'CADET does not support WENO0; FV order 1 is first-order upwind.'
            )
    if len(settings) != len(methods):
        raise ValueError('Provide a list of unique method names.')
    return settings


@dataclass
class FrontResult:
    """Outlet samples and excursions outside the physical concentration range."""

    times: np.ndarray
    outlet: np.ndarray
    undershoot: float
    overshoot: float
    mode: str = 'pulse'


def run_sharp_front_comparison(
        total_nodes=96, dispersion=1e-10, output_directory=RERUN_DIRECTORY,
        mode='pulse',
        methods=DEFAULT_METHODS, on_result=None):
    """Run selected methods at equal axial DOFs, notifying after each result.

    ``mode='pulse'`` injects for 60 s and then washes; ``mode='step'`` keeps
    feeding throughout the run. Both use a 0.1 m column without adsorption or
    particles. CADET must be configured first.

    The printed runtime is CADET's simulation time, excluding plotting and
    Python-side setup.
    """
    settings = method_settings(methods)
    multiple = lcm(*(degree + 1 for degree, _ in settings.values()))
    if not isinstance(total_nodes, int) or total_nodes <= 0 or total_nodes % multiple:
        raise ValueError(f'total_nodes must be a positive multiple of {multiple} for these methods.')
    if not np.isfinite(dispersion) or dispersion < 0:
        raise ValueError('Axial dispersion must be finite and non-negative.')
    if mode not in ('pulse', 'step'):
        raise ValueError("mode must be 'pulse' or 'step'.")
    if not settings:
        print('No methods selected; showing the ground truth only.')
        return {}

    end_time = 180.0
    sample_times = np.linspace(0.0, end_time, 3601)
    results = {}
    if dispersion > 0:
        print(f'Axial dispersion: {dispersion:.1e} m^2/s; '
              f'column Peclet: {transport.peclet(dispersion):.3g}')
    else:
        print('Axial dispersion: none; pure advection, infinite column Peclet')
    print('Excursions are measured over the sampled outlet signal, relative to the feed.')
    print(
        f"{'Method':>12} {'Cells/elements':>14} {'Axial DOFs':>12}"
        f" {'Undershoot [%]':>16} {'Overshoot [%]':>16} {'Runtime [s]':>14}"
    )
    for name, (degree, weno_order) in settings.items():
        n_elements = total_nodes if degree == 0 else transport.elements_for(degree, total_nodes)
        model = Cadet()
        model.root = transport.get_model(
            polynomial_degree=degree,
            total_nodes=total_nodes,
            col_dispersion=dispersion,
            mode=mode,
            end_time=end_time,
            idas_abstol=1e-12,
            idas_algtol=1e-12,
            idas_reltol=1e-10,
        )
        if degree == 0:
            model.root.input.model.unit_001.discretization.weno.WENO_ORDER = weno_order
        model.root.input.solver.user_solution_times = sample_times
        method_tag = name.replace(', ', '_')
        # Git carries no empty folders, so a fresh clone has none.
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        model.filename = str(
            Path(output_directory) / f'sharp_front_{mode}_{method_tag}_N{total_nodes}_D{dispersion:.6e}.h5'
        )
        model.save()
        run_info = model.run_simulation()
        if run_info.return_code != 0:
            raise RuntimeError(f'{name} sharp-front simulation failed: {run_info}')
        times = convergence.get_solution_times(model)
        outlet = np.asarray(convergence.get_outlet(model, unit='001')).reshape(-1)
        np.testing.assert_array_equal(times, sample_times)
        if outlet.shape != sample_times.shape or not np.all(np.isfinite(outlet)):
            raise ValueError(f'Invalid outlet solution for {name}.')
        undershoot = max(0.0, -float(outlet.min()))
        overshoot = max(0.0, float(outlet.max()) - transport.FEED)
        runtime = float(convergence.get_compute_time(model))
        if not np.isfinite(runtime) or runtime < 0:
            raise ValueError(f'Invalid simulation runtime for {name}: {runtime}.')
        results[name] = FrontResult(times, outlet, undershoot, overshoot, mode)
        print(
            f'{name:>12} {n_elements:14d} {total_nodes:12d}'
            f' {100 * undershoot / transport.FEED:16.6f}'
            f' {100 * overshoot / transport.FEED:16.6f}'
            f' {runtime:14.3e}'
        )
        if on_result is not None:
            on_result(name, results[name])
    return results


def reference_nodes(dispersion, total_nodes):
    """Axial cells the reference run uses at this dispersion.

    Enough to put `REFERENCE_NODES_PER_WIDTH` cells across the dispersive
    front, never coarser than twice the grid under study, never finer than
    the cap. Zero means no run is needed because the answer is exact.
    """
    if dispersion <= 0.0:
        return 0
    width = transport.front_width(dispersion)
    wanted = int(np.ceil(REFERENCE_NODES_PER_WIDTH * transport.BED_LENGTH
                         / width))
    wanted = max(wanted, 2 * int(total_nodes))
    wanted = 64 * int(np.ceil(wanted / 64))

    return min(wanted, REFERENCE_NODES_CAP)


def reference_outlet(times, dispersion, mode='pulse', total_nodes=96,
                     output_directory=RERUN_DIRECTORY, verbose=True):
    """The curve the methods are judged against, at the dispersion in use.

    With no dispersion the column only delays the inlet signal, so the outlet
    is known in closed form and is used as it stands.

    With dispersion there is no closed form for what CADET actually solves:
    the textbook advection-dispersion solution assumes a semi-infinite column
    and differs from a converged CADET run by about 0.8 % of the feed here,
    which is the size of the very excursions this study is about. So the
    reference is the same model on a grid fine enough to have converged,
    integrated with a monotone scheme.

    Returns
    -------
    (ndarray, str)
        The reference outlet sampled at `times`, and a label for the legend.
    """
    times = np.asarray(times, dtype=float)
    if mode not in ('pulse', 'step'):
        raise ValueError("mode must be 'pulse' or 'step'.")

    nodes = reference_nodes(dispersion, total_nodes)
    if nodes == 0:
        if verbose:
            print('Reference: exact pure-advection outlet, '
                  'the inlet signal delayed by L / u.')
        return zero_dispersion_outlet(times, mode), 'Reference (exact, D = 0)'

    width = transport.front_width(dispersion)
    per_width = width / (transport.BED_LENGTH / nodes)

    Path(output_directory).mkdir(parents=True, exist_ok=True)
    filename = Path(output_directory) / (
        f'reference_{mode}_N{nodes}_D{dispersion:.6e}.h5')

    if not filename.exists():
        model = Cadet()
        model.root = transport.get_model(
            polynomial_degree=0,
            total_nodes=nodes,
            col_dispersion=dispersion,
            mode=mode,
            end_time=float(times[-1]),
            idas_abstol=1e-12,
            idas_algtol=1e-12,
            idas_reltol=1e-10,
        )
        model.root.input.model.unit_001.discretization.weno.WENO_ORDER = (
            REFERENCE_WENO_ORDER)
        model.root.input.solver.user_solution_times = times
        model.filename = str(filename)
        model.save()
        run_info = model.run_simulation()
        if run_info.return_code != 0:
            raise RuntimeError(f'Reference simulation failed: {run_info}')

    outlet = np.asarray(
        convergence.get_outlet(str(filename), unit='001')).reshape(-1)
    if outlet.shape != times.shape or not np.all(np.isfinite(outlet)):
        raise ValueError('Invalid reference outlet.')

    if verbose:
        print(f'Reference: FV WENO{REFERENCE_WENO_ORDER} on {nodes} cells, '
              f'{per_width:.1f} cells across the front.')
        if per_width < REFERENCE_NODES_PER_WIDTH:
            print('  The cap on the reference grid binds at this dispersion, '
                  'so the reference is not fully converged either; read it '
                  'as the best available curve rather than as truth.')

    return outlet, f'Reference (FV WENO{REFERENCE_WENO_ORDER}, {nodes} cells)'


def zero_dispersion_outlet(times, mode='pulse'):
    """Exact pure-advection outlet; not exact for positive axial dispersion."""
    if mode not in ('pulse', 'step'):
        raise ValueError("mode must be 'pulse' or 'step'.")
    travel_time = transport.BED_LENGTH / transport.interstitial_velocity()
    arrived = np.asarray(times) >= travel_time
    if mode == 'pulse':
        arrived &= np.asarray(times) < travel_time + transport.INJECTION_TIME
    return transport.FEED * arrived.astype(float)


def plot_sharp_front_comparison(
        results, zoom_half_width=20.0, visible_methods=None, mode='pulse',
        method_order=None, ax=None, reference=None):
    """Plot a pulse or step, optionally on an existing subplot.

    `reference` is the (outlet, label) pair from `reference_outlet`. Without
    one the exact zero-dispersion curve is drawn, which is the reference only
    when there is no dispersion.
    """
    modes = {result.mode for result in results.values()}
    if len(modes) > 1:
        raise ValueError('Plot pulse and step results separately.')
    if not np.isfinite(zoom_half_width) or zoom_half_width <= 0:
        raise ValueError('The zoom half-width must be finite and positive.')
    mode = next(iter(modes), mode)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure
    if visible_methods is None:
        visible_methods = results.keys()
    if method_order is None:
        method_order = list(results)
    for name, result in results.items():
        if name in visible_methods:
            ax.plot(result.times, result.outlet, color=f'C{method_order.index(name) % 10}', label=name)
    times = next(iter(results.values())).times if results else np.linspace(0, 180, 3601)
    if reference is None:
        reference_curve, reference_label = (zero_dispersion_outlet(times, mode),
                                            'Reference (exact, D = 0)')
    else:
        reference_curve, reference_label = reference
    if len(reference_curve) == len(times):
        # A pure-advection reference is a discontinuity and is drawn as one;
        # a dispersive one is a smooth curve and is drawn as one.
        if np.array_equal(reference_curve,
                          zero_dispersion_outlet(times, mode)):
            ax.step(times, reference_curve, where='post', color='black',
                    linestyle='--', linewidth=1.5, label=reference_label)
        else:
            ax.plot(times, reference_curve, color='black', linestyle='--',
                    linewidth=1.5, label=reference_label)
    travel_time = transport.BED_LENGTH / transport.interstitial_velocity()
    ax.axhline(0.0, color='gray', linestyle=':', linewidth=1)
    ax.axhline(transport.FEED, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'Outlet concentration $[mol \cdot M^{-3}]$')
    ax.grid(alpha=0.3)
    ax.legend()
    if mode == 'pulse':
        ax.set_title('60 s pulse followed by wash: leading and trailing fronts')
    else:
        ax.set_title('Step: single leading front')
        ax.set_xlim(travel_time - zoom_half_width, travel_time + zoom_half_width)
    if standalone:
        fig.tight_layout()
        plt.show()
    return fig


def sharp_gradient_explorer(mode='pulse', output_directory=RERUN_DIRECTORY,
                            methods=DEFAULT_METHODS):
    """Run on click; ``mode='both'`` pairs a step and pulse with shared controls.

    Curve buttons redraw cached results only and apply to both panels.
    """
    if mode not in ('pulse', 'step', 'both'):
        raise ValueError("mode must be 'pulse', 'step', or 'both'.")
    modes = ('step', 'pulse') if mode == 'both' else (mode,)
    methods = tuple(methods)
    settings = method_settings(methods)
    multiple = lcm(*(degree + 1 for degree, _ in settings.values()))
    base_nodes = multiple * max(1, 48 // multiple)

    results = {case: {} for case in modes}
    references = {case: None for case in modes}
    plot_output = widgets.Output(layout=widgets.Layout(min_width='0', flex='1 1 auto'))
    curve_toggles = {
        name: widgets.ToggleButton(
            value=True, description=name, icon='check', button_style='info',
            tooltip=f'Show or hide {name}',
            layout=widgets.Layout(width='112px', height='34px', margin='0 0 8px 0'),
        )
        for name in settings
    }

    def redraw(change=None):
        visible_methods = [name for name, toggle in curve_toggles.items() if toggle.value]
        with plot_output:
            plot_output.clear_output(wait=True)
            if len(modes) == 2:
                figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                for case, ax in zip(modes, axes):
                    plot_sharp_front_comparison(
                        results[case], visible_methods=visible_methods,
                        mode=case, method_order=list(settings), ax=ax,
                        reference=references[case],
                    )
                figure.tight_layout()
                plt.show()
            else:
                figure = plot_sharp_front_comparison(
                    results[mode], visible_methods=visible_methods,
                    mode=mode, method_order=list(settings),
                    reference=references[mode],
                )
            plt.close(figure)

    def run(total_nodes, dispersion):
        for case_results in results.values():
            case_results.clear()
        for case in modes:
            references[case] = None
        redraw()

        def add_result(name, result):
            results[result.mode][name] = result
            redraw()

        if not settings:
            print('No methods selected; showing the ground truth only.')
            return
        for case in modes:
            print(f'\n{"Sustained front" if case == "step" else "60 s pulse"}')
            references[case] = reference_outlet(
                np.linspace(0.0, 180.0, 3601), dispersion, mode=case,
                total_nodes=total_nodes, output_directory=output_directory,
            )
            redraw()
            run_sharp_front_comparison(
                total_nodes=total_nodes, dispersion=dispersion,
                output_directory=output_directory, mode=case, methods=methods,
                on_result=add_result,
            )

    controls = widgets.interactive(
        run,
        {'manual': True, 'manual_name': 'Run simulations'},
        total_nodes=widgets.SelectionSlider(
            options=[base_nodes * factor for factor in (1, 2, 4, 8)],
            value=base_nodes * 2, description='Axial DOFs',
            continuous_update=False,
        ),
        dispersion=widgets.SelectionSlider(
            options=[('0 (pure advection)', 0.0)]
                    + [(f'{value:.0e}', value)
                       for value in (1e-11, 1e-10, 1e-9, 1e-8, 1e-7)],
            value=0.0, description='D [m^2/s]', continuous_update=False,
        ),
    )
    def update_visibility(change):
        toggle = change['owner']
        toggle.button_style = 'info' if toggle.value else ''
        toggle.icon = 'check' if toggle.value else ''
        redraw()

    for toggle in curve_toggles.values():
        toggle.observe(update_visibility, names='value')
    curve_panel = widgets.VBox(
        list(curve_toggles.values()),
        layout=widgets.Layout(
            width='128px', min_width='128px', flex='0 0 128px',
            padding='16px 8px', margin='24px 0 0 12px',
        ),
    )
    redraw()
    return widgets.VBox([
        controls,
        widgets.HBox(
            [plot_output, curve_panel],
            layout=widgets.Layout(width='100%', align_items='flex-start'),
        ),
    ])
