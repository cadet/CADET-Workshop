# -*- coding: utf-8 -*-
"""

Study: what a section boundary costs when it changes nothing.

The previous study cut a gradient into a staircase, so every boundary carried a
jump and the cost mixed two things: re-initializing the integrator, and
resolving the discontinuity. Here the elution is split into sections that are
identical on both sides. The inlet does not change, the physics does not
change, and the solution does not change. Only the number of boundaries does.

Whatever that costs is the transition itself.

Three knobs, because there are three questions.

    n_sections       how many boundaries, each of which changes nothing
    binding          an ODE (kinetic) or a DAE (rapid equilibrium)
    init_step_size   the step size the integrator restarts from

The second one is the point of the study. Rapid equilibrium turns the bound
states into algebraic variables, so CADET has to solve for them at every
boundary rather than carry them across. That ought to make a DAE more expensive
to restart than an ODE.

"""

import utility.setting_Col1D_langmuir_realistic as case
import utility.stiffness_study as study_tools
from utility.stiffness_study import Study

SECTION_COUNTS = (1, 2, 5, 10, 25, 50)
INIT_STEP_SIZES = (1e-10, 1e-8, 1e-6, 1e-4)
BINDINGS = ('kinetic', 'equilibrium')

N_ELEMENTS = 25
FAST_KINETICS = 1e3            # far past the point where the answer stops moving
COMPONENT_NAMES = ('weakly bound', 'intermediate', 'strongly bound')


def binding_label(binding):
    """Name of a binding treatment."""
    if binding == 'kinetic':
        rate = case.binding_rate(FAST_KINETICS)
        return f'kinetic, k = {rate:.3g} /s  (ODE)'

    return 'rapid equilibrium  (DAE)'


def _build(n_sections, binding, init_step_size):
    is_kinetic = binding == 'kinetic'

    return case.get_model(n_elements=N_ELEMENTS,
                          n_sections=n_sections,
                          kinetics_factor=FAST_KINETICS if is_kinetic else 1.0,
                          is_kinetic=is_kinetic,
                          init_step_size=init_step_size)


def _label(n_sections, binding, init_step_size):
    boundaries = n_sections - 1
    which = 'ODE' if binding == 'kinetic' else 'DAE'

    return (f'{boundaries} boundaries, {which}, init h {init_step_size:.0e}')


def _describe_with(study):
    def describe(measured, outlet, n_sections, binding, init_step_size):
        baseline = study_tools.simulate(
            study, n_sections=1, binding=binding,
            init_step_size=init_step_size)[0].n_steps

        boundaries = n_sections - 1
        extra = measured.n_steps - baseline

        return {
            'boundaries': boundaries,
            'extra steps': extra,
            'per boundary': round(extra / boundaries, 1) if boundaries else None,
        }

    return describe


def make_study():
    """Assemble the Study for this case."""
    study = Study(
        name='initialization',
        build=_build,
        options={
            'n_sections': study_tools.value_options(SECTION_COUNTS, '{:d}'),
            'binding': study_tools.value_options(BINDINGS, formatter=binding_label),
            'init_step_size': study_tools.value_options(INIT_STEP_SIZES, '{:.0e}'),
        },
        label_of=_label,
        velocity=case.interstitial_velocity(),
        feed_concentration=case.LOADING,
        component_names=COMPONENT_NAMES,
        defaults={'n_sections': 25, 'binding': 'kinetic',
                  'init_step_size': 1e-10},
    )
    study.describe = _describe_with(study)

    return study


def show_setting():
    """Plot the undivided run and return the rates it is made of."""
    return study_tools.show_setting(
        make_study(), n_sections=1, binding='kinetic', init_step_size=1e-10,
        title='the three protein separation, run as one uninterrupted elution')


def make_explorer():
    """Build the slider callback together with the record it accumulates."""
    return study_tools.make_explorer(make_study())
