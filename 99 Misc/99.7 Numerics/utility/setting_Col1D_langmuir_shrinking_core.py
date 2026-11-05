# -*- coding: utf-8 -*-
"""

A shrinking-core particle: the sharp front that lives inside the bead.

The forum setting shows an undershoot produced by a front in the *bulk*. This
one moves the same problem into the *particle*, where it is easier to provoke
and harder to notice, because nothing downstream has to show it.

A strongly favourable Langmuir isotherm on a slowly diffusing pore makes the
uptake shell-progressive: everything that reaches the solid is adsorbed where
it lands, so the bead fills from the outside in, with a saturated outer shell,
a sharp front and an untouched core. How sharp depends on

    lambda = (1 - eps_p) q_max / (eps_p c_feed),

how much more the solid holds than the pore liquid it sits in. Lowering the
feed at fixed capacity raises lambda, and the front inside the bead steepens
until the radial discretization can no longer represent it.

That is the same situation as a steep front in the bulk, so the same thing
happens: a discontinuous Galerkin particle oscillates around the front and
produces negative pore concentrations, while finite volumes stays monotone.
The study swaps the particle method at an **identical number of radial
points**, so the comparison is at equal cost per bead.

Built on utility/setting_Col1D_grm_realistic.py, overriding the isotherm,
the feed and the particle discretization.

"""

import numpy as np

import utility.setting_Col1D_grm_realistic as grm

Q_MAX = 100.0                  # mol/m^3 of solid
K_EQ = 1.0e+04                 # k_a / k_d, strongly favourable
PORE_DIFFUSION = 1.0e-11       # m^2/s, slow: the bead is the bottleneck
FILM_DIFFUSION = 1.0e-04       # m/s, fast: the film is not the bottleneck
# The undershoot is radial: it is the same to four digits on 10, 16 and
# 40 axial cells, so the bulk grid is kept cheap enough to explore live.
AXIAL_ELEMENTS = 16
END_TIME = 6000.0              # s
# The largest lambda at which the column still breaks through inside
# END_TIME. Below this the bed simply swallows everything, and the
# outlet carries no signal to measure an error against.
C_FEED = 1.0                   # mol/m^3

COMPONENT_NAMES = ('protein',)


def method_label(spatial_method):
    """Name of a spatial method: FV, or DG with its polynomial degree."""
    return 'FV' if spatial_method == 0 else f'DG{spatial_method}'


def fits(spatial_method, n_points):
    """Can this method be built with exactly this many radial points?

    A discontinuous Galerkin particle carries degree + 1 points per element,
    so only multiples of degree + 1 are available to it. Finite volumes takes
    any count.
    """
    if spatial_method == 0:
        return n_points >= 1

    elements, remainder = divmod(n_points, spatial_method + 1)

    return elements >= 1 and remainder == 0


def lambda_ratio(c_feed=C_FEED, q_max=Q_MAX):
    """How much more the solid holds than the pore liquid carries."""
    porosity = grm.PARTICLE_POROSITY

    return (1.0 - porosity) * q_max / (porosity * c_feed)


def spatial_dofs(spatial_method, n_points, n_elements=AXIAL_ELEMENTS):
    """Spatial discretization points per component.

    The bulk is finite volumes here and is held fixed, so what varies is the
    radial resolution carried by every axial cell.
    """
    return n_elements * (1 + n_points)


def get_model(spatial_method=0, n_points=8, c_feed=C_FEED, q_max=Q_MAX,
              k_eq=K_EQ, n_elements=AXIAL_ELEMENTS, end_time=END_TIME,
              n_time_points=601, idas_abstol=1e-10, idas_reltol=1e-8):
    """Build the shrinking-core column.

    Parameters
    ----------
    spatial_method : int
        Particle discretization: 0 selects finite volumes, anything else is
        the discontinuous Galerkin polynomial degree.
    n_points : int
        Radial points per particle, the same for both methods.
    c_feed : float
        Feed concentration in mol/m^3. Lowering it sharpens the front inside
        the bead; see lambda_ratio.

    Returns
    -------
    cadet.Cadet-ready addict.Dict
    """
    if not fits(spatial_method, n_points):
        raise ValueError(f'{n_points} radial points do not divide into '
                         f'{method_label(spatial_method)} elements.')

    model = grm.get_model(
        particle_cells=n_points,
        pore_diffusion=PORE_DIFFUSION,
        particle_model='GRM',
        n_elements=n_elements,
        film_diffusion=FILM_DIFFUSION,
        end_time=end_time,
    )
    column = model.input.model.unit_001
    particle = column.particle_type_000

    # A strongly favourable isotherm, so that the front inside the particle
    # is sharp rather than spread out.
    particle.adsorption_model = 'MULTI_COMPONENT_LANGMUIR'
    particle.adsorption = {
        'is_kinetic': 0,
        'mcl_ka': [k_eq],
        'mcl_kd': [1.0],
        'mcl_qmax': [q_max],
    }

    if spatial_method > 0:
        particle.discretization.SPATIAL_METHOD = 'DG'
        particle.discretization.PAR_POLYDEG = spatial_method
        particle.discretization.PAR_NELEM = n_points // (spatial_method + 1)
        particle.discretization.pop('NCELLS', None)
        particle.discretization.pop('FV_BOUNDARY_ORDER', None)
    else:
        particle.discretization.SPATIAL_METHOD = 'FV'
        particle.discretization.NCELLS = n_points
        particle.discretization.FV_BOUNDARY_ORDER = 2

    # A sustained feed, so the particles fill and the front sits inside them.
    solver = model.input.solver
    solver.sections.nsec = 1
    solver.sections.section_times = [0.0, end_time]
    solver.sections.section_continuity = []
    solver.user_solution_times = np.linspace(0.0, end_time, n_time_points)
    solver.time_integrator.ABSTOL = idas_abstol
    solver.time_integrator.RELTOL = idas_reltol
    solver.time_integrator.ALGTOL = idas_abstol * 100

    inlet = model.input.model.unit_000
    inlet.sec_000.const_coeff = [c_feed]
    inlet.pop('sec_001', None)

    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_001.write_solution_particle = 1
    model.input['return'].unit_001.write_solution_solid = 1
    model.input['return'].write_solution_times = 1

    return model
