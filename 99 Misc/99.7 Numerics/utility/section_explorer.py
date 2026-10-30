# -*- coding: utf-8 -*-
"""

Study: section transitions, isolated from everything else.

The same inlet gradient is written two ways. One section carrying a linear
coefficient gives an exact ramp with no interior transition at all. The same
ramp cut into n constant steps gives a staircase that converges to it, and
n - 1 section boundaries along the way.

Past about fifty steps the staircase and the ramp are the same chromatogram, so
whatever the staircase costs is the price of having written the gradient that
way, and nothing else.

The second knob is INIT_STEP_SIZE, because that is what the integrator restarts
from after every transition. It separates the two halves of the cost: the climb
back out of the initial step size, which is pure overhead and recoverable, and
the resolution of the jump itself, which is not.

"""

import numpy as np

import utility.setting_Col1D_transport_realistic as case
import utility.stiffness_study as study_tools
from utility.stiffness_study import Study

SECTION_COUNTS = (1, 2, 5, 10, 25, 50, 100, 200)
INIT_STEP_SIZES = (1e-10, 1e-8, 1e-6, 1e-4, 1e-3)

GRID_CELLS = 100
END_TIME = 300.0
COMPONENT_NAMES = ('tracer',)

_reference = {}


def _build(n_sections, init_step_size):
    return case.get_model(polynomial_degree=0, total_nodes=GRID_CELLS,
                          mode='gradient', n_sections=n_sections,
                          init_step_size=init_step_size, end_time=END_TIME)


def _label(n_sections, init_step_size):
    if n_sections == 1:
        shape = 'one ramp (lin_coeff)'
    else:
        shape = f'{n_sections} step staircase'

    return f'{shape}, init h {init_step_size:.0e}'


def _reference_outlet(study, init_step_size):
    """The exact ramp, which every staircase is approximating."""
    if init_step_size not in _reference:
        _, _, outlet = study_tools.simulate(study, n_sections=1,
                                            init_step_size=init_step_size)
        _reference[init_step_size] = outlet

    return _reference[init_step_size]


def _describe_with(study):
    def describe(measured, outlet, n_sections, init_step_size):
        reference = _reference_outlet(study, init_step_size)
        deviation = 100.0 * np.sqrt(np.mean((outlet - reference) ** 2)) / case.FEED

        transitions = n_sections - 1
        baseline = study_tools.simulate(
            study, n_sections=1, init_step_size=init_step_size)[0].n_steps
        extra = measured.n_steps - baseline

        return {
            'transitions': transitions,
            'extra steps': extra,
            'per transition': round(extra / transitions, 1) if transitions else np.nan,
            'deviation from ramp %': round(deviation, 3),
        }

    return describe


def make_study():
    """Assemble the Study for this case."""
    study = Study(
        name='sections',
        notebook='stiffness/Stiffness 6 - Section transitions',
        build=_build,
        options={
            'n_sections': study_tools.value_options(SECTION_COUNTS, '{:d}'),
            'init_step_size': study_tools.value_options(INIT_STEP_SIZES, '{:.0e}'),
        },
        label_of=_label,
        velocity=case.interstitial_velocity(),
        feed_concentration=case.FEED,
        component_names=COMPONENT_NAMES,
        defaults={'n_sections': 25, 'init_step_size': 1e-10},
    )
    study.describe = _describe_with(study)

    return study


def show_setting():
    """Plot the exact ramp and return the rates it is made of."""
    return study_tools.show_setting(
        make_study(), n_sections=1, init_step_size=1e-10,
        title='an inlet gradient, written as one section with a slope')


def make_explorer():
    """Build the slider callback together with the record it accumulates."""
    return study_tools.make_explorer(make_study())
