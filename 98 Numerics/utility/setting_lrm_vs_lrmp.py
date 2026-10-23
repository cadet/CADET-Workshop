# -*- coding: utf-8 -*-
"""

A batch elution run twice: once as an LRM, once as an LRMP with the film
transfer turned up until it should not matter.

Rebuilt on the current COLUMN_MODEL_1D interface from a pair of configurations
reported on the forum, where the LRM finished in seconds and the LRMP took
about twenty minutes on the same machine, same physics and same grid.

The two models are set up to be the same problem:

    LRM     total porosity 0.72
    LRMP    bed porosity 0.3 and particle porosity 0.6, which is the same 0.72

with linear binding in rapid equilibrium, no axial dispersion, and a film
diffusion coefficient of 1 m/s. That last number is the whole point. It is not
a physical film coefficient, it is a deliberately enormous one, chosen so that
the pore liquid is pinned to the bulk liquid and the LRMP reduces to the LRM.
At a particle radius of 5 um it makes the film transfer about

    3 k_f / r_p * (1 - eps_c) / eps_c  =  1.4e6 / s

against convection across one of 500 cells at about 6 / s.

Everything that distinguishes the two models is exposed as a keyword, so the
notebook can take the gap apart one parameter at a time.

"""

import numpy as np
from addict import Dict

BED_LENGTH = 0.6
CROSS_SECTION_AREA = 4.52e-04
FLOW_RATE = 1.0e-06
BED_POROSITY = 0.3
PARTICLE_POROSITY = 0.6
TOTAL_POROSITY = BED_POROSITY + (1.0 - BED_POROSITY) * PARTICLE_POROSITY
PARTICLE_RADIUS = 5.0e-06
FILM_DIFFUSION = 1.0

FEED = 10.0
LIN_KA = (2.0, 3.0)
LIN_KD = (1.0, 1.0)

INJECTION_TIME = 60.0
END_TIME = 600.0
N_COMP = 2


def interstitial_velocity():
    """Interstitial velocity in m/s, as the flow rate and geometry give it."""
    return FLOW_RATE / (CROSS_SECTION_AREA * BED_POROSITY)


def film_rate(film_diffusion=FILM_DIFFUSION, par_radius=PARTICLE_RADIUS):
    """Rate of film mass transfer in 1/s, as the rate table defines it."""
    return (3.0 * film_diffusion / par_radius
            * (1.0 - BED_POROSITY) / BED_POROSITY)


def convection_rate(n_col=500):
    """Rate of convection across one axial cell in 1/s."""
    return interstitial_velocity() / (BED_LENGTH / n_col)


def get_model(transport_model='LRMP', **kwargs):
    """Build the batch elution column.

    Parameters
    ----------
    transport_model : string
        'LRM' or 'LRMP'.
    film_diffusion : float
        Film coefficient in m/s. Only the LRMP has one.
    par_radius : float
        Particle radius in m. Only the LRMP has one.
    n_col : int
        Number of axial finite volume cells.
    is_kinetic : bool
        False is the reported case: linear binding in rapid equilibrium, which
        makes the bound states algebraic.
    idas_abstol, idas_reltol, idas_algtol : float
        Integrator tolerances. The reported case runs ALGTOL six decades
        tighter than ABSTOL, which is one of the things worth varying.

    Returns
    -------
    addict.Dict
    """
    n_col = int(kwargs.get('n_col', 500))
    is_kinetic = bool(kwargs.get('is_kinetic', False))

    model = Dict()
    model.input.model.nunits = 4
    model.input.model.connections.connections_include_ports = 0
    model.input.model.connections.nswitches = 2

    column = Dict()
    column.unit_type = 'COLUMN_MODEL_1D'
    column.geometry = 'AXIAL_FLOW_CYLINDER'
    column.inlet_cross_section_area = CROSS_SECTION_AREA
    column.cross_section_area = CROSS_SECTION_AREA
    column.bed_length = BED_LENGTH
    column.forward_flow = 1

    column.ncomp = N_COMP
    column.npartype = 1
    column.col_dispersion = kwargs.get('col_dispersion', 0.0)
    column.init_c = [0.0] * N_COMP

    # Same retention either way: the LRM carries the total porosity, the LRMP
    # splits it into a bed and a particle porosity that multiply back to it.
    column.total_porosity = TOTAL_POROSITY
    if transport_model != 'LRM':
        column.col_porosity = BED_POROSITY

    column.discretization.SPATIAL_METHOD = 'FV'
    column.discretization.NCOL = n_col
    column.discretization.RECONSTRUCTION = 'WENO'
    column.discretization.weno.BOUNDARY_MODEL = 0
    column.discretization.weno.WENO_EPS = 1e-10
    column.discretization.weno.WENO_ORDER = int(kwargs.get('weno_order', 2))
    column.discretization.GS_TYPE = 1
    column.discretization.MAX_KRYLOV = int(kwargs.get('max_krylov', 0))
    column.discretization.MAX_RESTARTS = int(kwargs.get('max_restarts', 10))
    column.discretization.SCHUR_SAFETY = kwargs.get('schur_safety', 1.0e-8)
    column.discretization.USE_ANALYTIC_JACOBIAN = 1

    # With an FV bulk and an FV particle, CADET defaults to the legacy
    # block-structured ("arrow head") solver, which eliminates the particle
    # unknowns into a Schur complement and solves that iteratively. Setting
    # this to False routes the same model through ColumnModel1D's global
    # direct sparse solver instead. Left unset, CADET picks the default.
    if 'arrow_head' in kwargs:
        column.discretization.FV_ARROW_HEAD_OPTIMIZATION = bool(
            kwargs['arrow_head'])

    particle = column.particle_type_000
    if transport_model == 'LRM':
        particle.has_film_diffusion = False
    else:
        particle.has_film_diffusion = True
        particle.has_pore_diffusion = False
        particle.par_geom = 'SPHERE'
        particle.par_radius = kwargs.get('par_radius', PARTICLE_RADIUS)
        particle.par_coreradius = 0.0
        particle.par_porosity = PARTICLE_POROSITY
        particle.film_diffusion = [
            kwargs.get('film_diffusion', FILM_DIFFUSION)] * N_COMP
        particle.init_cp = [0.0] * N_COMP

    particle.nbound = [1] * N_COMP
    particle.init_cs = [0.0] * N_COMP
    particle.adsorption_model = 'LINEAR'
    particle.adsorption.is_kinetic = 1 if is_kinetic else 0
    particle.adsorption.LIN_KA = list(LIN_KA)
    particle.adsorption.LIN_KD = list(LIN_KD)

    model.input.model.unit_002 = column

    # Two inlets and a connection switch, as in the reported configuration:
    # the feed during the injection, a zero inlet for the wash.
    for index, concentration in ((0, FEED), (1, 0.0)):
        inlet = model.input.model[f'unit_{index:03d}']
        inlet.unit_type = 'INLET'
        inlet.inlet_type = 'PIECEWISE_CUBIC_POLY'
        inlet.ncomp = N_COMP
        inlet.sec_000.const_coeff = [concentration] * N_COMP
        inlet.sec_000.lin_coeff = [0.0] * N_COMP
        inlet.sec_000.quad_coeff = [0.0] * N_COMP
        inlet.sec_000.cube_coeff = [0.0] * N_COMP

    for switch, source in ((0, 0), (1, 1)):
        entry = model.input.model.connections[f'switch_{switch:03d}']
        entry.section = switch
        entry.connections = [
            float(source), 2.0, -1.0, -1.0, FLOW_RATE,
            2.0, 3.0, -1.0, -1.0, FLOW_RATE,
        ]

    model.input.model.unit_003.unit_type = 'OUTLET'
    model.input.model.unit_003.ncomp = N_COMP

    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08

    model.input.solver.consistent_init_mode = int(
        kwargs.get('consistent_init_mode', 1))
    model.input.solver.consistent_init_mode_sens = 1
    model.input.solver.nthreads = 1
    model.input.solver.sections.nsec = 2
    model.input.solver.sections.section_continuity = [0]
    model.input.solver.sections.section_times = [0.0, INJECTION_TIME, END_TIME]

    integrator = model.input.solver.time_integrator
    integrator.ABSTOL = kwargs.get('idas_abstol', 1e-08)
    integrator.RELTOL = kwargs.get('idas_reltol', 1e-06)
    integrator.ALGTOL = kwargs.get('idas_algtol', 1e-12)
    integrator.INIT_STEP_SIZE = kwargs.get('init_step_size', 1e-06)
    integrator.MAX_STEPS = 1000000

    model.input.solver.user_solution_times = np.linspace(
        0.0, END_TIME, 6001)

    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].write_solution_times = 1
    model.input['return'].unit_002.write_solution_outlet = 1
    model.input['return'].write_idas_meta = 1

    return model
