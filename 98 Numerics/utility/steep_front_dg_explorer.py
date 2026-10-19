# -*- coding: utf-8 -*-
"""

Study: a steep concentration front on a discontinuous Galerkin grid.

The companion of the finite volume steep front study, with the same column,
the same step input and the same dispersion knob, but a third knob: how the
degrees of freedom are arranged. At a fixed number of nodes, P1 on many small
elements and P15 on few large ones are the same system size and the same
physics, and only the shape of the spatial operator differs.

That third knob matters here in a way it did not for finite volumes. The FV
study ran WENO, which is monotone, so a front narrower than a cell was simply
smeared and nothing else happened. A DG element has no such guarantee: a front
narrower than the element it sits in cannot be represented by the polynomial,
and what comes out instead is an oscillation. The solution itself becomes
harder to integrate, so the cost is no longer governed by resolution alone.

Three knobs, and the interesting comparisons hold two of them fixed:

    col_dispersion      how sharp the front is
    total_nodes         how many degrees of freedom pay for it
    polynomial_degree   how those degrees of freedom are arranged

Degree zero selects finite volumes with WENO, so the baseline of the FV study
sits on the same slider as the DG variants and can be read against them.

"""

import numpy as np

import utility.setting_Col1D_transport_realistic as case
import utility.stiffness_study as study_tools
from utility.stiffness_study import Study

DISPERSIONS = (1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10, 1e-11)
# Every node count is divisible by every (degree + 1) below, so any
# combination of the three sliders is a model that can actually be built.
NODES = (64, 128, 256, 512)
DEGREES = (0, 1, 3, 7, 15)

END_TIME = 300.0
COMPONENT_NAMES = ('tracer',)


def degree_label(polynomial_degree):
    """Slider label for a polynomial degree, with FV as degree zero."""
    if polynomial_degree == 0:
        return 'FV (WENO)'

    return f'P{polynomial_degree}'


def elements(polynomial_degree, total_nodes):
    """Number of elements, or of cells when the scheme is finite volumes."""
    if polynomial_degree == 0:
        return total_nodes

    return case.elements_for(polynomial_degree, total_nodes)


def element_size(polynomial_degree, total_nodes):
    """Length of one element in m."""
    return case.BED_LENGTH / elements(polynomial_degree, total_nodes)


def node_spacing(polynomial_degree, total_nodes):
    """Average distance between degrees of freedom in m.

    The fair comparison at a fixed system size: whatever the degree, this many
    numbers describe the profile, spread over this much column.
    """
    return case.BED_LENGTH / total_nodes


def smallest_spacing(polynomial_degree, total_nodes):
    """Smallest distance between neighbouring nodes in m.

    Nodes inside a DG element cluster towards its edges, so the tightest
    spacing is far below the average and it is the one the convective
    eigenvalues see. This is the same estimate utility.time_scales uses, so
    the rate table and the stiffness ratio reported here cannot disagree.
    """
    count = elements(polynomial_degree, total_nodes)
    if polynomial_degree > 1:
        return case.BED_LENGTH / (count * polynomial_degree ** 2)

    return case.BED_LENGTH / count


def stiffness_ratio(col_dispersion, polynomial_degree, total_nodes):
    """Classical stiffness ratio: fastest process over slowest."""
    fastest = case.interstitial_velocity() / smallest_spacing(
        polynomial_degree, total_nodes)
    slowest = col_dispersion / case.BED_LENGTH ** 2

    return fastest / slowest


def oscillation(outlet, feed=case.FEED):
    """How far the outlet signal leaves the range the solution should stay in.

    A step response rises from zero to the feed concentration and should never
    leave that interval. Anything outside it is the polynomial of an element
    failing to represent a front narrower than itself. Reported as a percentage
    of the feed, taking whichever side is worse.
    """
    above = float(outlet.max()) - feed
    below = -float(outlet.min())

    return 100.0 * max(above, below, 0.0) / feed


def _build(col_dispersion, total_nodes, polynomial_degree):
    """Pure transport with a step input, on the requested grid."""
    try:
        case.elements_for(polynomial_degree, total_nodes)
    except ValueError as error:
        # make_explorer records a RuntimeError as a failed row rather than
        # stopping the sweep, which is the right outcome for a grid that
        # cannot be built.
        raise RuntimeError(f'Invalid grid: {error}') from error

    return case.get_model(polynomial_degree=polynomial_degree,
                          total_nodes=total_nodes,
                          col_dispersion=col_dispersion, mode='step',
                          end_time=END_TIME)


def _label(col_dispersion, total_nodes, polynomial_degree):
    """Row label for one slider position."""
    return (f'{degree_label(polynomial_degree)}, {total_nodes} nodes, '
            f'dispersion {col_dispersion:.0e}')


def _describe(measured, outlet, col_dispersion, total_nodes,
              polynomial_degree):
    """Resolution from two sides, and whether the solution stayed monotone."""
    width = case.front_width(col_dispersion)

    return {
        'elements': elements(polynomial_degree, total_nodes),
        'front / node': round(width / node_spacing(polynomial_degree,
                                                   total_nodes), 3),
        'front / element': round(width / element_size(polynomial_degree,
                                                      total_nodes), 3),
        'oscillation / %': round(oscillation(outlet), 2),
        'stiffness ratio': f'{stiffness_ratio(col_dispersion, polynomial_degree, total_nodes):.3g}',
    }


def make_study():
    """Assemble the Study for this case."""
    return Study(
        name='steep_front_dg',
        build=_build,
        options={
            'col_dispersion': study_tools.value_options(DISPERSIONS, '{:.0e}'),
            'total_nodes': study_tools.value_options(NODES, '{:d}'),
            'polynomial_degree': study_tools.value_options(DEGREES,
                                                           degree_label),
        },
        label_of=_label,
        velocity=case.interstitial_velocity(),
        feed_concentration=case.FEED,
        component_names=COMPONENT_NAMES,
        describe=_describe,
        defaults={'col_dispersion': 1e-07, 'total_nodes': 256,
                  'polynomial_degree': 3},
    )


def show_setting():
    """Plot the base case and return the rates it is made of."""
    return study_tools.show_setting(
        make_study(), col_dispersion=1e-07, total_nodes=256,
        polynomial_degree=3,
        title='a step into a column of pure transport, P3 on 64 elements, '
              f'column Peclet {case.peclet(1e-07):.0f}')


TOLERANCES = (1e-08, 1e-07, 1e-06, 1e-05, 1e-04)


def _build_at_tolerance(col_dispersion, total_nodes, polynomial_degree,
                        reltol):
    """The same model, integrated to a looser or tighter local error."""
    _build(col_dispersion, total_nodes, polynomial_degree)   # validates the grid

    return case.get_model(polynomial_degree=polynomial_degree,
                          total_nodes=total_nodes,
                          col_dispersion=col_dispersion, mode='step',
                          end_time=END_TIME,
                          idas_reltol=reltol, idas_abstol=reltol * 1e-2)


def _tolerance_study():
    """A second Study over the same settings plus the error tolerance.

    Separate from the main one so that the three sliders of the notebook stay
    about the model and the grid, while this stays about the integrator.
    """
    return Study(
        name='steep_front_dg_tolerance',
        build=_build_at_tolerance,
        options={
            'col_dispersion': study_tools.value_options(DISPERSIONS, '{:.0e}'),
            'total_nodes': study_tools.value_options(NODES, '{:d}'),
            'polynomial_degree': study_tools.value_options(DEGREES,
                                                           degree_label),
            'reltol': study_tools.value_options(TOLERANCES, '{:.0e}'),
        },
        label_of=lambda col_dispersion, total_nodes, polynomial_degree, reltol:
            (f'{degree_label(polynomial_degree)}, {total_nodes} nodes, '
             f'dispersion {col_dispersion:.0e}, reltol {reltol:.0e}'),
        velocity=case.interstitial_velocity(),
        feed_concentration=case.FEED,
        component_names=COMPONENT_NAMES,
    )


def tolerance_sweep(col_dispersion=1e-10, total_nodes=256, degrees=(0, 15),
                    tolerances=TOLERANCES):
    """Loosen the error tolerance and see whether the steps come back.

    The test that separates the two reasons a step size can be small. If the
    integrator is held back by **accuracy**, asking for less of it returns the
    steps in proportion. If it is held back by **stability**, there is a floor
    it cannot rise above however much error is allowed, because a larger step
    would not be wrong, it would be divergent.

    Returns
    -------
    pandas.DataFrame
        One row per tolerance, one pair of columns per grid: the step count
        and what it is relative to the tightest tolerance.
    """
    import pandas as pd

    import utility.cadet_runtime as cadet_runtime
    cadet_runtime.configure()

    study = _tolerance_study()
    table = {'RELTOL': [f'{value:.0e}' for value in tolerances]}

    for degree in degrees:
        counts = []
        for reltol in tolerances:
            measured, _, _ = study_tools.simulate(
                study, col_dispersion=col_dispersion, total_nodes=total_nodes,
                polynomial_degree=degree, reltol=reltol)
            counts.append(measured.n_steps)

        name = degree_label(degree)
        table[f'{name} steps'] = counts
        table[f'{name} vs tightest'] = [round(count / counts[0], 2)
                                        for count in counts]

    return pd.DataFrame(table)


def compare_fronts(col_dispersion=1e-10, total_nodes=256, degrees=(0, 15)):
    """Draw the same front as each scheme resolves it, zoomed in.

    The two failure modes side by side. Both grids carry the same number of
    unknowns and solve the same equation; what differs is what they do when
    the front is sharper than they can represent.

    Returns
    -------
    (matplotlib figure, np.array of matplotlib axes)
    """
    import matplotlib.pyplot as plt

    import utility.cadet_runtime as cadet_runtime
    from utility.stiffness_metrics import COLORS, style_axis

    cadet_runtime.configure()
    study = make_study()

    figure, axes = plt.subplots(1, len(degrees), figsize=(11.0, 3.8),
                                sharey=True)
    axes = np.atleast_1d(axes)

    for axis, degree in zip(axes, degrees):
        measured, times, outlet = study_tools.simulate(
            study, col_dispersion=col_dispersion, total_nodes=total_nodes,
            polynomial_degree=degree)
        signal = outlet[:, 0]

        axis.axhline(0.0, color='#999999', linewidth=0.8)
        axis.axhline(case.FEED, color='#999999', linewidth=0.8)
        axis.plot(times / 60.0, signal, color=COLORS[0], linewidth=1.8)

        # Centre the view on the front, wherever the breakthrough lands.
        arrival = times[int(np.argmax(signal > 0.5 * case.FEED))] / 60.0
        axis.set_xlim(arrival - 0.3, arrival + 0.3)

        axis.set_title(f'{degree_label(degree)}, {elements(degree, total_nodes)}'
                       f' elements, {measured.n_steps} steps', fontsize=10)
        axis.set_xlabel('time / min')
        axis.annotate(f'{oscillation(signal):.2f} % outside [0, feed]',
                      xy=(0.04, 0.62), xycoords='axes fraction', fontsize=9,
                      color=COLORS[1])
        style_axis(axis)

    axes[0].set_ylabel(r'concentration / mol m$^{-3}$')
    figure.suptitle(f'the same front on {total_nodes} nodes, '
                    f'dispersion {col_dispersion:.0e} '
                    r'm$^2$/s', fontsize=11)
    figure.tight_layout()

    return figure, axes


def make_explorer():
    """Build the slider callback together with the record it accumulates."""
    return study_tools.make_explorer(make_study())
