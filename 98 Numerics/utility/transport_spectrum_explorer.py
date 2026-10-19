# -*- coding: utf-8 -*-
"""

Study: the spatial discretization and the time integrator, on pure transport.

Two knobs, because one cannot answer the question. The polynomial degree is
swept at a fixed number of degrees of freedom, which changes the spectrum of
the discrete operator while holding the system size and the physics constant.
The relative tolerance is swept alongside it, because that is what separates
the two reasons a step size can be small:

    accuracy limited  -> loosening the tolerance reduces the step count
    stability limited -> it cannot

Everything generic lives in utility.stiffness_study.

"""

import utility.setting_Col1D_transport_realistic as case
import utility.stiffness_study as study_tools
from utility.stiffness_study import Study

METHODS = (0,) + case.DEGREES          # 0 is finite volumes with WENO
TOLERANCES = (1e-4, 1e-6, 1e-8, 1e-10)

GRID_PROCESS = 'convection, one grid node'
COMPONENT_NAMES = ('tracer',)


def method_label(polynomial_degree):
    """Human readable name of a spatial discretization."""
    if polynomial_degree == 0:
        return 'FV/WENO'

    return f'DG P{polynomial_degree}'


def _build(col_dispersion):
    """Return a builder for a given axial dispersion."""
    def build(polynomial_degree, reltol):
        return case.get_model(polynomial_degree=polynomial_degree,
                              col_dispersion=col_dispersion,
                              idas_reltol=reltol,
                              idas_abstol=reltol * 1e-2)

    return build


def _label(polynomial_degree, reltol):
    """Row label for one slider position."""
    return f'{method_label(polynomial_degree)}, reltol {reltol:.0e}'


def _describe(measured, outlet, polynomial_degree, reltol):
    """Columns that are specific to this study."""
    elements = (case.TOTAL_NODES if polynomial_degree == 0
                else case.elements_for(polynomial_degree))

    return {'elements': elements, 'nodes': case.TOTAL_NODES}


def make_study(col_dispersion=case.AXIAL_DISPERSION):
    """Assemble the Study for this case.

    Parameters
    ----------
    col_dispersion : float
        Axial dispersion in m^2/s. The default is the realistic value, giving a
        column Peclet number of about 1100.

    Returns
    -------
    Study
    """
    return Study(
        name='transport_spectrum',
        build=_build(col_dispersion),
        options={
            'polynomial_degree': study_tools.value_options(
                METHODS, formatter=lambda degree: method_label(degree)),
            'reltol': study_tools.value_options(TOLERANCES, '{:.0e}'),
        },
        label_of=_label,
        velocity=case.interstitial_velocity(),
        feed_concentration=case.FEED,
        knob_process=GRID_PROCESS,
        component_names=COMPONENT_NAMES,
        describe=_describe,
        defaults={'polynomial_degree': 3, 'reltol': 1e-8},
    )


def show_setting(col_dispersion=case.AXIAL_DISPERSION):
    """Plot the base case and return the rates it is made of."""
    study = make_study(col_dispersion)

    return study_tools.show_setting(
        study, polynomial_degree=3, reltol=1e-8,
        title=f'pure transport, {case.TOTAL_NODES} nodes, '
              f'column Peclet {case.peclet(col_dispersion):.0f}')


def make_explorer(col_dispersion=case.AXIAL_DISPERSION):
    """Build the slider callback together with the record it accumulates."""
    return study_tools.make_explorer(make_study(col_dispersion))
