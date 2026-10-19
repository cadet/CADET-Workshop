# -*- coding: utf-8 -*-
"""

Study 1: the binding kinetics, in the isolated three component setting of
utility.setting_Col1D_langmuir_realistic.

The binding rate constants are scaled jointly, so the equilibrium constant and
therefore the isotherm never change. Only the speed of the binding moves. The
sweep ends one step beyond the fast limit, at IS_KINETIC = 0.

Everything generic lives in utility.stiffness_study; this module only says what
the model is and which knob is turned.

"""

import utility.setting_Col1D_langmuir_realistic as case
import utility.stiffness_study as study_tools
from utility.stiffness_study import Study

N_ELEMENTS = 25                       # meets 1 % of the feed peak, and no more
EQUILIBRIUM_LABEL = 'rapid equilibrium'
BINDING_PROCESS = 'binding (Langmuir)'

COMPONENT_NAMES = ('weakly bound', 'intermediate', 'strongly bound')


def _build(n_elements):
    """Return a builder that turns a slider value into a model."""
    def build(kinetics):
        is_kinetic = kinetics is not None

        return case.get_model(n_elements=n_elements,
                              kinetics_factor=kinetics if is_kinetic else 1.0,
                              is_kinetic=is_kinetic)

    return build


def _label(kinetics):
    """Row label for one slider position."""
    if kinetics is None:
        return EQUILIBRIUM_LABEL

    return f'binding rate {case.binding_rate(kinetics):.3g} / s'


def make_study(n_elements=N_ELEMENTS, start=-6, end=14, step=1):
    """Assemble the Study for this case.

    Parameters
    ----------
    n_elements : int
        Number of finite volume cells, held fixed across the sweep.
    start, end : int
        First and last decade of the kinetics factor, inclusive.
    step : int
        Decades between neighbouring slider positions.

    Returns
    -------
    Study
    """
    return Study(
        name='binding_kinetics_explorer',
        build=_build(n_elements),
        options=study_tools.decade_options(
            start, end, step, tail=(EQUILIBRIUM_LABEL, None)),
        label_of=_label,
        velocity=case.interstitial_velocity(),
        feed_concentration=case.LOADING,
        knob_process=BINDING_PROCESS,
        component_names=COMPONENT_NAMES,
        knob_name='kinetics',
    )


def rate_table(n_elements=N_ELEMENTS, kinetics_factor=1.0):
    """Characteristic rates of the setting on a given grid."""
    return study_tools.rate_table(make_study(n_elements=n_elements),
                                  kinetics=kinetics_factor)


def slowest_other_process(n_elements=N_ELEMENTS):
    """Rate and name of the slowest process that is not the binding."""
    _, rate, name = study_tools.knob_and_slowest(
        make_study(n_elements=n_elements), kinetics=1.0)

    return rate, name


def simulate(kinetics_factor, is_kinetic=True, n_elements=N_ELEMENTS):
    """Run one setting and return its integrator work, times and outlet."""
    return study_tools.simulate(make_study(n_elements=n_elements),
                                kinetics=kinetics_factor if is_kinetic else None)


def plot_chromatogram(times, outlet, axis=None, reference=None, title=None):
    """Draw the three component chromatogram, optionally over a reference."""
    return study_tools.plot_chromatogram(times, outlet, axis=axis,
                                         reference=reference, title=title,
                                         component_names=COMPONENT_NAMES)


def show_setting(n_elements=N_ELEMENTS):
    """Plot the base case and return the rates it is made of."""
    return study_tools.show_setting(
        make_study(n_elements=n_elements), kinetics=1.0,
        title='three proteins, competitive Langmuir, '
              f'{n_elements} finite volume cells')


def slider_options(start=-6, end=14, step=1):
    """Slider positions over the kinetics factor, then rapid equilibrium."""
    return study_tools.decade_options(start, end, step,
                                      tail=(EQUILIBRIUM_LABEL, None))


def make_explorer(n_elements=N_ELEMENTS, start=-6, end=14, step=1):
    """Build the slider callback together with the record it accumulates.

    Parameters
    ----------
    n_elements : int
        Number of finite volume cells, held fixed across the sweep.
    start, end : int
        First and last decade of the kinetics factor, inclusive.
    step : int
        Decades between neighbouring slider positions.

    Returns
    -------
    callable
        Carries `options`, `default` and `records`.
    """
    return study_tools.make_explorer(
        make_study(n_elements=n_elements, start=start, end=end, step=step))
