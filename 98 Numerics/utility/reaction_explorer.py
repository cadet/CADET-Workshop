# -*- coding: utf-8 -*-
"""

Study: a fast reaction, as the direct counterpart of the binding study.

A reversible reaction A <-> B in the bulk liquid of a column that is otherwise
pure transport. The two rate constants are scaled jointly, so the equilibrium
composition never changes and only the speed moves, exactly as the binding
study does with its isotherm.

The point of repeating the first study with a different model class is that the
answer should not depend on which physical process the fast mode belongs to. A
reaction relaxes towards equilibrium at k_fwd + k_bwd, which is a real negative
eigenvalue, the same kind the binding contributed.

Note what is *not* here: the quasi-stationary reaction, MAL_IS_KINETIC = 0, is
not offered as a slider position because it does not work in this CADET version
for a bulk liquid reaction. See the notebook.

"""

import utility.setting_Col1D_reaction_realistic as case
import utility.stiffness_study as study_tools
from utility.stiffness_study import Study

FAST_LIMIT = 1e12              # far past the point where the answer stops moving
COMPONENT_NAMES = ('A', 'B')


def _build(reaction):
    return case.get_model(reaction_factor=reaction, is_kinetic=True)


def _label(reaction):
    return f'reaction rate {case.reaction_rate(reaction):.3g} / s'


def _describe(measured, outlet, reaction):
    """How much of the material has reacted by the time it leaves."""
    total = outlet[:, 0].sum() + outlet[:, 1].sum()
    converted = outlet[:, 1].sum() / total if total else float('nan')

    grid_rate = case.interstitial_velocity() / (case.BED_LENGTH / 100)

    return {
        'rate / grid rate': f'{case.reaction_rate(reaction) / grid_rate:.3g}',
        'fraction converted': round(converted, 4),
    }


def make_study(start=-2, end=12, step=1):
    """Assemble the Study for this case."""
    return Study(
        name='reaction',
        build=_build,
        options=study_tools.decade_options(start, end, step),
        label_of=_label,
        velocity=case.interstitial_velocity(),
        feed_concentration=case.FEED,
        reference={'reaction': FAST_LIMIT},
        component_names=COMPONENT_NAMES,
        describe=_describe,
        knob_name='reaction',
        defaults={'reaction': 1.0},
    )


def show_setting():
    """Plot the base case and return the rates it is made of."""
    return study_tools.show_setting(
        make_study(), reaction=1.0,
        title='A to B in the bulk liquid, pure transport otherwise')


def make_explorer(start=-2, end=12, step=1):
    """Build the slider callback together with the record it accumulates."""
    return study_tools.make_explorer(make_study(start, end, step))
