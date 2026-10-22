# -*- coding: utf-8 -*-
"""

Study: dead volume in the flowsheet.

The column of the earlier notebooks, with a detector cell after it:

    inlet -> column -> detector -> outlet

A stirred tank relaxes at Q / V, so a small tank is a fast tank. The knob is
the detector volume, and it reaches well past anything a real instrument has in
both directions, from a mixing chamber down to a volume no plumbing could hold.

What is recorded is the detector outlet, which is the signal an instrument
would show rather than what leaves the column.

"""

import utility.setting_flowsheet_realistic as case
import utility.stiffness_study as study_tools
from utility.stiffness_study import Study

VOLUMES = (1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10, 1e-11, 1e-12, 1e-13)
REFERENCE_VOLUME = 1e-13       # a tank so small it cannot smear anything

N_ELEMENTS = 25
COMPONENT_NAMES = ('weakly bound', 'intermediate', 'strongly bound')


def _build(detector_volume):
    return case.get_model(detector_volume=detector_volume,
                          n_elements=N_ELEMENTS)


def _label(detector_volume):
    microlitres = detector_volume * 1e9

    if microlitres >= 1000:
        size = f'{microlitres / 1000:g} mL'
    elif microlitres >= 1:
        size = f'{microlitres:g} uL'
    else:
        size = f'{microlitres * 1000:g} nL'

    return f'detector {size}'


def _describe(measured, outlet, detector_volume):
    grid_rate = case.column_case.interstitial_velocity() / (
        case.column_case.BED_LENGTH / N_ELEMENTS)

    return {
        'volume / uL': f'{detector_volume * 1e9:.3g}',
        'tau / s': f'{case.residence_time(detector_volume):.3g}',
        'mixing rate / s^-1': f'{case.mixing_rate(detector_volume):.3g}',
        'rate / grid rate': f'{case.mixing_rate(detector_volume) / grid_rate:.3g}',
        'peak height': round(float(outlet.max()), 5),
    }


def make_study():
    """Assemble the Study for this case."""
    return Study(
        name='flowsheet',
        build=_build,
        options=study_tools.value_options(VOLUMES, formatter=_label),
        label_of=_label,
        velocity=case.column_case.interstitial_velocity(),
        feed_concentration=case.FEED,
        reference={'detector_volume': REFERENCE_VOLUME},
        component_names=COMPONENT_NAMES,
        describe=_describe,
        knob_name='detector_volume',
        outlet_unit='unit_002',
        defaults={'detector_volume': 1e-08},
    )


def show_setting():
    """Plot the base case and return the rates of the column it contains."""
    return study_tools.show_setting(
        make_study(), detector_volume=1e-08,
        title='inlet, column, 10 uL detector, outlet')


def make_explorer():
    """Build the slider callback together with the record it accumulates."""
    return study_tools.make_explorer(make_study())
