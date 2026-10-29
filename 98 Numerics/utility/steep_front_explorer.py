# -*- coding: utf-8 -*-
"""

Study: a steep concentration front, isolated from everything else.

A step into a column of pure transport produces exactly one front and nothing
else: no binding, no particle, no section transition. The axial dispersion sets
how sharp that front is, and the grid decides whether it can be seen. Those are
the two knobs.

This study reports two numbers side by side on purpose. The classical stiffness
ratio, fastest process over slowest, rises steeply as the dispersion is cut,
because the slowest process in the model *is* the dispersion. And the width of
the front measured in grid cells falls. Only the second one predicts the cost,
which is the point: a slow process does not make a system expensive, an
unresolved feature does.

"""

import utility.setting_Col1D_transport_realistic as case
import utility.stiffness_study as study_tools
from utility.stiffness_study import Study

DISPERSIONS = (1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10, 1e-11)
GRIDS = (40, 80, 160, 320, 640)

END_TIME = 300.0
COMPONENT_NAMES = ('tracer',)


def cell_size(cells):
    """Axial cell size in m."""
    return case.BED_LENGTH / cells


def front_cells(col_dispersion, cells):
    """Width of the front measured in grid cells.

    Below about one, the grid cannot represent the front, and refining the
    dispersion further changes nothing the solver can see.
    """
    return case.front_width(col_dispersion) / cell_size(cells)


def stiffness_ratio(col_dispersion, cells):
    """Classical stiffness ratio: fastest process over slowest.

    The fastest is convection across one cell, which the dispersion does not
    touch. The slowest is dispersion over the whole column, which it does. So
    this ratio rises as the front sharpens, for a reason that has nothing to do
    with the cost.
    """
    fastest = case.interstitial_velocity() / cell_size(cells)
    slowest = col_dispersion / case.BED_LENGTH ** 2

    return fastest / slowest


def _build(col_dispersion, cells):
    """Pure transport with a step input, finite volumes."""
    return case.get_model(polynomial_degree=0, total_nodes=cells,
                          col_dispersion=col_dispersion, mode='step',
                          end_time=END_TIME)


def _label(col_dispersion, cells):
    """Row label for one slider position."""
    return f'dispersion {col_dispersion:.0e}, {cells} cells'


def _describe(measured, outlet, col_dispersion, cells):
    """The two competing indicators, reported together."""
    return {
        'Peclet': f'{case.peclet(col_dispersion):.3g}',
        'stiffness ratio': f'{stiffness_ratio(col_dispersion, cells):.3g}',
        'front width / cell': round(front_cells(col_dispersion, cells), 3),
    }


def make_study():
    """Assemble the Study for this case."""
    return Study(
        name='steep_front',
        notebook='stiffness/Stiffness 3a - Steep fronts FV',
        build=_build,
        options={
            'col_dispersion': study_tools.value_options(DISPERSIONS, '{:.0e}'),
            'cells': study_tools.value_options(GRIDS, '{:d}'),
        },
        label_of=_label,
        velocity=case.interstitial_velocity(),
        feed_concentration=case.FEED,
        component_names=COMPONENT_NAMES,
        describe=_describe,
        defaults={'col_dispersion': 1e-07, 'cells': 160},
    )


def show_setting():
    """Plot the base case and return the rates it is made of."""
    return study_tools.show_setting(
        make_study(), col_dispersion=1e-07, cells=160,
        title='a step into a column of pure transport, '
              f'column Peclet {case.peclet(1e-07):.0f}')


def make_explorer():
    """Build the slider callback together with the record it accumulates."""
    return study_tools.make_explorer(make_study())
