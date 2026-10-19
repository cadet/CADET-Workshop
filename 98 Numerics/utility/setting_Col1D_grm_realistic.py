# -*- coding: utf-8 -*-
"""

A general rate model on the same realistic lab column, with linear binding at
rapid equilibrium, so that the particle is the only thing left to study.

The binding is linear and algebraic on purpose. A nonlinear isotherm would
sharpen fronts, and kinetic binding would add a relaxation rate; both were
shown to confound earlier studies. What remains is a particle with a pore
diffusion time scale, a film resistance, and a radial grid.

The interesting length inside a particle is the penetration depth. While the
bulk concentration changes over a time t, diffusion reaches

    delta ~ sqrt(D_p t)

into the particle. When that is comparable to the particle radius, the particle
is nearly uniform. When it is much smaller, the particle carries a sharp shell
and an untouched core, which is a steep gradient in the radial direction.

Column parameters match setting_Col1D_langmuir_realistic.

"""

import numpy as np
from addict import Dict

BED_LENGTH = 0.1
COLUMN_RADIUS = 0.005
BED_POROSITY = 0.37
PARTICLE_POROSITY = 0.75
PARTICLE_RADIUS = 4.5e-05
SUPERFICIAL_VELOCITY = 4.2e-04
AXIAL_DISPERSION = 1.0e-07
FILM_DIFFUSION = 1.0e-05
PORE_DIFFUSION = 1.0e-11          # protein in a resin pore

EQUILIBRIUM_CONSTANT = 10.0       # linear isotherm, q = K c
FEED = 1.0                        # mol/m^3
INJECTION_TIME = 60.0
END_TIME = 1200.0

TOTAL_POROSITY = BED_POROSITY + (1.0 - BED_POROSITY) * PARTICLE_POROSITY


def interstitial_velocity():
    """Interstitial velocity in m/s."""
    return SUPERFICIAL_VELOCITY / BED_POROSITY


def dead_time():
    """Time for an unretained tracer to cross the column, in s."""
    return BED_LENGTH * TOTAL_POROSITY / SUPERFICIAL_VELOCITY


def penetration_depth(pore_diffusion=PORE_DIFFUSION, time=None):
    """How far diffusion reaches into a particle, in m.

    Parameters
    ----------
    pore_diffusion : float
        Pore diffusivity in m^2/s.
    time : float
        Time available, defaulting to the column dead time, which is the scale
        on which the bulk concentration changes here.

    Returns
    -------
    float
    """
    if time is None:
        time = dead_time()

    return np.sqrt(pore_diffusion * time)


def particle_cell_size(particle_cells):
    """Radial cell size in m."""
    return PARTICLE_RADIUS / particle_cells


def get_model(particle_cells=4, pore_diffusion=PORE_DIFFUSION,
              particle_model='GRM', **kwargs):
    """Build the column with a resolved or a lumped particle.

    Parameters
    ----------
    particle_cells : int
        Number of radial finite volume cells inside the particle. Ignored for
        the lumped particle, which has no radial coordinate.
    pore_diffusion : float
        Pore diffusivity in m^2/s. Ignored for the lumped particle.
    particle_model : string
        'GRM' resolves the radial direction. 'LRMP' lumps the particle into one
        well mixed volume behind a film resistance, which is the limit of the
        general rate model for infinitely fast pore diffusion.

    Returns
    -------
    addict.Dict
    """
    if particle_model not in ('GRM', 'LRMP'):
        raise ValueError(f"Unknown particle model: {particle_model}")
    end_time = kwargs.get('end_time', END_TIME)
    cross_section = np.pi * COLUMN_RADIUS ** 2

    model = Dict()
    model.input.model.nunits = 3
    model.input.model.connections.connections_include_ports = 0
    model.input.model.connections.nswitches = 1

    column = Dict()
    column.unit_type = 'COLUMN_MODEL_1D'
    column.geometry = 'AXIAL_FLOW_CYLINDER'
    column.inlet_cross_section_area = cross_section
    column.cross_section_area = cross_section
    column.bed_length = BED_LENGTH
    column.forward_flow = 1

    column.ncomp = 1
    column.npartype = 1
    column.col_porosity = BED_POROSITY
    column.total_porosity = TOTAL_POROSITY
    column.col_dispersion = kwargs.get('col_dispersion', AXIAL_DISPERSION)
    column.init_c = [0.0]

    column.discretization.SPATIAL_METHOD = 'FV'
    column.discretization.NCOL = kwargs.get('n_elements', 100)
    column.discretization.RECONSTRUCTION = 'WENO'
    column.discretization.weno.BOUNDARY_MODEL = 0
    column.discretization.weno.WENO_EPS = 1e-10
    column.discretization.weno.WENO_ORDER = 3
    column.discretization.GS_TYPE = 1
    column.discretization.MAX_KRYLOV = 0
    column.discretization.MAX_RESTARTS = 10
    column.discretization.SCHUR_SAFETY = 1.0e-8
    column.discretization.USE_ANALYTIC_JACOBIAN = 1

    particle = column.particle_type_000
    particle.has_film_diffusion = True
    particle.has_pore_diffusion = (particle_model == 'GRM')
    particle.has_surface_diffusion = False
    particle.par_geom = 'SPHERE'
    particle.par_radius = kwargs.get('par_radius', PARTICLE_RADIUS)
    particle.par_coreradius = 0.0
    particle.par_porosity = PARTICLE_POROSITY
    particle.film_diffusion = [kwargs.get('film_diffusion', FILM_DIFFUSION)]
    particle.init_cp = [0.0]
    particle.nbound = [1]
    particle.init_cs = [0.0]

    if particle_model == 'GRM':
        particle.pore_diffusion = [pore_diffusion]
        particle.surface_diffusion = [0.0]
        particle.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
        particle.discretization.SPATIAL_METHOD = 'FV'
        particle.discretization.NCELLS = particle_cells
        particle.discretization.FV_BOUNDARY_ORDER = 2

    # Linear isotherm at rapid equilibrium: no nonlinearity to sharpen fronts,
    # no binding rate to add a time scale.
    particle.adsorption_model = 'LINEAR'
    particle.adsorption = {
        'is_kinetic': 0,
        'lin_ka': [kwargs.get('equilibrium_constant', EQUILIBRIUM_CONSTANT)],
        'lin_kd': [1.0],
    }

    model.input.model.unit_001 = column

    flow_rate = SUPERFICIAL_VELOCITY * cross_section
    model.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, flow_rate,
        1.0, 2.0, -1.0, -1.0, flow_rate,
    ]
    model.input.model.connections.switch_000.section = 0

    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08

    model.input.solver.consistent_init_mode = 1
    model.input.solver.consistent_init_mode_sens = 3
    model.input.solver.nthreads = 1
    model.input.solver.sections.nsec = 2
    model.input.solver.sections.section_continuity = [0]
    model.input.solver.sections.section_times = [0.0, INJECTION_TIME, end_time]
    model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_abstol', 1e-10)
    model.input.solver.time_integrator.ALGTOL = kwargs.get('idas_algtol', 1e-8)
    model.input.solver.time_integrator.RELTOL = kwargs.get('idas_reltol', 1e-8)
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 1000000
    model.input.solver.user_solution_times = np.linspace(
        0.0, end_time, int(end_time) + 1)

    model.input.model.unit_000.UNIT_TYPE = 'INLET'
    model.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_000.ncomp = 1
    model.input.model.unit_000.sec_000.const_coeff = [FEED]
    model.input.model.unit_000.sec_000.lin_coeff = [0.0]
    model.input.model.unit_000.sec_000.quad_coeff = [0.0]
    model.input.model.unit_000.sec_000.cube_coeff = [0.0]
    model.input.model.unit_000.sec_001.const_coeff = [0.0]
    model.input.model.unit_000.sec_001.lin_coeff = [0.0]
    model.input.model.unit_000.sec_001.quad_coeff = [0.0]
    model.input.model.unit_000.sec_001.cube_coeff = [0.0]

    model.input.model.unit_002.UNIT_TYPE = 'OUTLET'
    model.input.model.unit_002.ncomp = 1

    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].write_solution_times = 1
    model.input['return'].unit_000.write_solution_outlet = 0
    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_002.write_solution_outlet = 0

    return model
