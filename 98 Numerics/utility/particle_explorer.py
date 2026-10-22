# -*- coding: utf-8 -*-
"""

Study: the particle, isolated from everything that was shown to confound it.

A general rate model with a linear isotherm at rapid equilibrium, so no
isotherm curvature can sharpen a front and no binding rate can add a time
scale. What is left is a particle: a film resistance, a pore diffusion time
scale, and a radial grid of its own.

Two knobs. The pore diffusivity sets how far into the particle the solute
reaches while the bulk changes,

    delta ~ sqrt(D_p t),

and the radial grid decides whether that can be represented. The pairing is the
same one that governed the axial direction in the steep front study, which is
the comparison the notebook is built on.

The record also carries the error against a converged reference, because the
particle turns out to be a case where refining the grid changes both the cost
and the answer, and in the same direction.

"""

import numpy as np

import utility.setting_Col1D_grm_realistic as case
import utility.stiffness_study as study_tools
from utility.stiffness_study import Study

PORE_DIFFUSIONS = (1e-09, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14)
PARTICLE_GRIDS = (1, 2, 4, 8, 16, 32)

REFERENCE_CELLS = 128
PORE_CELL_PROCESS = 'pore diffusion, one particle cell'
COMPONENT_NAMES = ('protein',)

_reference = {}


def depth_in_cells(pore_diffusion, particle_cells):
    """Penetration depth measured in radial cells.

    Below about one the grid cannot represent the shell that forms near the
    particle surface.
    """
    return (case.penetration_depth(pore_diffusion)
            / case.particle_cell_size(particle_cells))


def depth_in_particles(pore_diffusion):
    """Penetration depth as a fraction of the particle radius.

    Much greater than one means the particle is nearly uniform, much less than
    one means a thin shell over an untouched core.
    """
    return case.penetration_depth(pore_diffusion) / case.PARTICLE_RADIUS


def _build(pore_diffusion, particle_cells):
    return case.get_model(particle_cells=particle_cells,
                          pore_diffusion=pore_diffusion)


def _label(pore_diffusion, particle_cells):
    return f'D_p {pore_diffusion:.0e}, {particle_cells} particle cells'


def _reference_outlet(study, pore_diffusion):
    """Converged solution for one pore diffusivity, computed once."""
    if pore_diffusion not in _reference:
        _, _, outlet = study_tools.simulate(study, pore_diffusion=pore_diffusion,
                                            particle_cells=REFERENCE_CELLS)
        _reference[pore_diffusion] = outlet

    return _reference[pore_diffusion]


def _describe_with(study):
    """Build the describe callback, which needs the study to find its reference."""
    def describe(measured, outlet, pore_diffusion, particle_cells):
        reference = _reference_outlet(study, pore_diffusion)
        error = 100.0 * np.sqrt(np.mean((outlet - reference) ** 2)) / case.FEED

        return {
            'depth / particle': round(depth_in_particles(pore_diffusion), 3),
            'depth / cell': round(depth_in_cells(pore_diffusion, particle_cells), 3),
            'error % of feed': round(error, 3),
        }

    return describe


def make_study():
    """Assemble the Study for this case."""
    study = Study(
        name='particle',
        build=_build,
        options={
            'pore_diffusion': study_tools.value_options(PORE_DIFFUSIONS, '{:.0e}'),
            'particle_cells': study_tools.value_options(PARTICLE_GRIDS, '{:d}'),
        },
        label_of=_label,
        velocity=case.interstitial_velocity(),
        feed_concentration=case.FEED,
        knob_process=PORE_CELL_PROCESS,
        component_names=COMPONENT_NAMES,
        defaults={'pore_diffusion': case.PORE_DIFFUSION, 'particle_cells': 8},
    )
    study.describe = _describe_with(study)

    return study


def show_setting():
    """Plot the base case and return the rates it is made of."""
    return study_tools.show_setting(
        make_study(), pore_diffusion=case.PORE_DIFFUSION, particle_cells=8,
        title='general rate model, linear isotherm at rapid equilibrium')


def make_explorer():
    """Build the slider callback together with the record it accumulates."""
    return study_tools.make_explorer(make_study())
