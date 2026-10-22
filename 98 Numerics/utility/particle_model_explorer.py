# -*- coding: utf-8 -*-
"""

Study: lumping the particle, or resolving it.

The lumped rate model with pores treats a particle as one well mixed volume
behind a film resistance. The general rate model resolves the radial direction.
LRMP is the limit of the GRM for infinitely fast pore diffusion, so the two
describe the same column whenever the particle really is uniform, and different
columns whenever it is not.

Two knobs, because the question has two halves. The particle model and its
radial resolution say how much is being resolved. The pore diffusivity decides
whether there is anything there to resolve: the penetration depth

    delta ~ sqrt(D_p t)

is the distance the solute reaches into the particle while the bulk changes.
Much larger than the radius and the particle is uniform, much smaller and it
carries a thin shell over a dead core.

Every row is scored against its own converged general rate model at the same
pore diffusivity, so the error column is the price of lumping rather than the
difference between two unrelated models.

"""

import numpy as np

import utility.setting_Col1D_grm_realistic as case
import utility.stiffness_study as study_tools
from utility.stiffness_study import Study

LUMPED = 0                      # slider value standing for "no radial grid"
PARTICLE_MODELS = (LUMPED, 1, 2, 4, 8, 16, 32)
PORE_DIFFUSIONS = (1e-09, 1e-10, 1e-11, 1e-12, 1e-13)

REFERENCE_CELLS = 128
COMPONENT_NAMES = ('protein',)

_reference = {}


def model_label(particle):
    """Name of a particle treatment."""
    if particle == LUMPED:
        return 'LRMP (lumped)'

    return f'GRM, {particle} radial cells'


def unknowns_per_axial_cell(particle):
    """Bulk, particle liquid and solid unknowns carried at each axial position."""
    if particle == LUMPED:
        return 3

    return 1 + 2 * particle


def _build(particle, pore_diffusion):
    if particle == LUMPED:
        return case.get_model(particle_model='LRMP')

    return case.get_model(particle_model='GRM', particle_cells=particle,
                          pore_diffusion=pore_diffusion)


def _label(particle, pore_diffusion):
    if particle == LUMPED:
        # The lumped model has no pore diffusivity, so it is one simulation and
        # not one per setting of a knob it does not have.
        return model_label(particle)

    return f'{model_label(particle)}, D_p {pore_diffusion:.0e}'


def _reference_outlet(study, pore_diffusion):
    """Converged general rate model at one pore diffusivity, computed once."""
    if pore_diffusion not in _reference:
        _, _, outlet = study_tools.simulate(study, particle=REFERENCE_CELLS,
                                            pore_diffusion=pore_diffusion)
        _reference[pore_diffusion] = outlet

    return _reference[pore_diffusion]


def _describe_with(study):
    def describe(measured, outlet, particle, pore_diffusion):
        reference = _reference_outlet(study, pore_diffusion)
        error = 100.0 * np.sqrt(np.mean((outlet - reference) ** 2)) / case.FEED
        depth = case.penetration_depth(pore_diffusion) / case.PARTICLE_RADIUS

        return {
            'unknowns per cell': unknowns_per_axial_cell(particle),
            'depth / particle': round(depth, 3),
            'error % of feed': round(error, 3),
        }

    return describe


def make_study():
    """Assemble the Study for this case."""
    study = Study(
        name='particle_model',
        build=_build,
        options={
            'particle': study_tools.value_options(PARTICLE_MODELS,
                                                  formatter=model_label),
            'pore_diffusion': study_tools.value_options(PORE_DIFFUSIONS, '{:.0e}'),
        },
        label_of=_label,
        velocity=case.interstitial_velocity(),
        feed_concentration=case.FEED,
        component_names=COMPONENT_NAMES,
        defaults={'particle': 8, 'pore_diffusion': case.PORE_DIFFUSION},
    )
    study.describe = _describe_with(study)

    return study


def show_setting():
    """Plot the base case and return the rates it is made of."""
    return study_tools.show_setting(
        make_study(), particle=8, pore_diffusion=case.PORE_DIFFUSION,
        title='general rate model, 8 radial cells, '
              f'D_p = {case.PORE_DIFFUSION:.0e}')


def make_explorer():
    """Build the slider callback together with the record it accumulates."""
    return study_tools.make_explorer(make_study())
