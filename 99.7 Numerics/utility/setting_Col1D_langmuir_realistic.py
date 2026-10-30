# -*- coding: utf-8 -*-
"""

A physically realistic three component protein separation, built so that the
binding kinetics can be isolated as the only source of stiffness.

The published two component Langmuir benchmark is deliberately *not* used for
that study: it is an overloaded case designed to challenge spatial schemes with
a self sharpening front, and that front dominates everything else. Here the
column is an ordinary lab scale one and the loading is a knob, so that the
isotherm can be nonlinear without producing a shock.

Column parameters and their justification:

    bed length          0.1 m            lab column
    diameter            0.01 m           lab column
    superficial vel.    4.2e-4 m/s       150 cm/h, typical range 100-600
    bed porosity        0.37             packed sphere bed
    particle porosity   0.75             standard resin
    particle radius     4.5e-5 m         standard resin
    axial dispersion    1e-7 m^2/s       particle Peclet ~ 1, column Peclet ~ 1100
    film diffusion      1e-5 m/s         protein in a stirred film
    binding capacity    2 mol/m^3        about 100 g/L for a 50 kDa protein

The transport model is the lumped rate model with pores. A general rate model
would be more detailed, but its particle discretization introduces a second
grid and a second fast process, which is exactly the contamination this setting
avoids. Film mass transfer sits at a fixed rate of about 1.1 1/s and does not
move while the binding kinetics are swept across it.

Knobs
    loading           feed concentration in mol/m^3, sets the isotherm curvature
    kinetics_factor   common factor on ka and kd, leaves the isotherm untouched
    is_kinetic        False selects rapid equilibrium
    n_elements        number of finite volume cells
    col_dispersion    axial dispersion

"""

import numpy as np
from addict import Dict

# Geometry and flow
BED_LENGTH = 0.1
COLUMN_RADIUS = 0.005
BED_POROSITY = 0.37
PARTICLE_POROSITY = 0.75
PARTICLE_RADIUS = 4.5e-05
SUPERFICIAL_VELOCITY = 4.2e-04
AXIAL_DISPERSION = 1.0e-07
FILM_DIFFUSION = 1.0e-05

# Binding: three proteins of increasing affinity, competing for one site type
CAPACITY = 2.0                       # mol/m^3
EQUILIBRIUM_CONSTANTS = (5.0, 10.0, 20.0)   # m^3/mol, ka/kd
DESORPTION_RATE = 1.0                # 1/s at kinetics_factor = 1

# Operation
LOADING = 0.02                       # mol/m^3 in the feed
INJECTION_TIME = 60.0                # s
END_TIME = 2500.0                    # s

TOTAL_POROSITY = BED_POROSITY + (1.0 - BED_POROSITY) * PARTICLE_POROSITY


def interstitial_velocity():
    """Interstitial velocity in m/s."""
    return SUPERFICIAL_VELOCITY / BED_POROSITY


def binding_rate(kinetics_factor=1.0, loading=LOADING, component=-1):
    """Relaxation rate of the binding, ka * c + kd, in 1/s.

    Parameters
    ----------
    kinetics_factor : float
        Common factor on ka and kd.
    loading : float
        Feed concentration in mol/m^3.
    component : int
        Which component, -1 for the most strongly bound one.

    Returns
    -------
    float
    """
    equilibrium = EQUILIBRIUM_CONSTANTS[component]
    desorption = DESORPTION_RATE * kinetics_factor
    adsorption = equilibrium * desorption

    return adsorption * loading + desorption


def retention_factor(component=-1):
    """Linear retention factor k' of one component, for orientation."""
    phase_ratio = (1.0 - TOTAL_POROSITY) / TOTAL_POROSITY

    return phase_ratio * CAPACITY * EQUILIBRIUM_CONSTANTS[component]


def elution_time(component=-1):
    """Expected retention time in s, in the linear limit."""
    dead_time = BED_LENGTH * TOTAL_POROSITY / SUPERFICIAL_VELOCITY

    return dead_time * (1.0 + retention_factor(component))


def get_binding_configuration(kinetics_factor=1.0, is_kinetic=True):
    """Competitive Langmuir binding with a tunable relaxation time.

    Scaling ka and kd together leaves ka/kd, and therefore the isotherm and the
    equilibrium chromatogram, exactly where they are. Only the speed changes.
    """
    desorption = DESORPTION_RATE * kinetics_factor

    return {
        'is_kinetic': 1 if is_kinetic else 0,
        'mcl_ka': [equilibrium * desorption for equilibrium in EQUILIBRIUM_CONSTANTS],
        'mcl_kd': [desorption] * len(EQUILIBRIUM_CONSTANTS),
        'mcl_qmax': [CAPACITY] * len(EQUILIBRIUM_CONSTANTS),
    }


def get_model(n_elements=100, kinetics_factor=1.0, is_kinetic=True, **kwargs):
    """Build the three component protein separation.

    Parameters
    ----------
    n_elements : int
        Number of finite volume cells.
    kinetics_factor : float
        Common factor on ka and kd.
    is_kinetic : bool
        False selects rapid equilibrium.

    Returns
    -------
    addict.Dict
    """
    n_comp = len(EQUILIBRIUM_CONSTANTS)
    loading = kwargs.get('loading', LOADING)
    end_time = kwargs.get('end_time', END_TIME)

    # The elution can be split into sections that change nothing at all. The
    # inlet is identical on either side of such a boundary, so what they cost
    # is the re-initialization and the order restart, with no discontinuity to
    # resolve on top.
    n_sections = int(kwargs.get('n_sections', 1))

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

    column.ncomp = n_comp
    column.npartype = 1
    column.col_porosity = BED_POROSITY
    column.total_porosity = TOTAL_POROSITY
    column.col_dispersion = kwargs.get('col_dispersion', AXIAL_DISPERSION)
    column.init_c = [0.0] * n_comp

    # Finite volumes with WENO: no spatial oscillations to confound the study.
    column.discretization.SPATIAL_METHOD = 'FV'
    column.discretization.NCOL = n_elements
    column.discretization.RECONSTRUCTION = 'WENO'
    column.discretization.weno.BOUNDARY_MODEL = 0
    column.discretization.weno.WENO_EPS = 1e-10
    column.discretization.weno.WENO_ORDER = 3
    column.discretization.GS_TYPE = 1
    column.discretization.MAX_KRYLOV = 0
    column.discretization.MAX_RESTARTS = 10
    column.discretization.SCHUR_SAFETY = 1.0e-8
    column.discretization.USE_ANALYTIC_JACOBIAN = 1

    column.particle_type_000.has_film_diffusion = True
    column.particle_type_000.par_geom = 'SPHERE'
    column.particle_type_000.par_radius = kwargs.get('par_radius', PARTICLE_RADIUS)
    column.particle_type_000.par_coreradius = 0.0
    column.particle_type_000.par_porosity = PARTICLE_POROSITY
    column.particle_type_000.film_diffusion = \
        [kwargs.get('film_diffusion', FILM_DIFFUSION)] * n_comp
    column.particle_type_000.init_cp = [0.0] * n_comp
    column.particle_type_000.nbound = [1] * n_comp
    column.particle_type_000.init_cs = [0.0] * n_comp
    column.particle_type_000.adsorption_model = 'MULTI_COMPONENT_LANGMUIR'
    column.particle_type_000.adsorption = get_binding_configuration(
        kinetics_factor=kinetics_factor, is_kinetic=is_kinetic)

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
    elution = np.linspace(INJECTION_TIME, end_time, n_sections + 1)
    model.input.solver.sections.nsec = 1 + n_sections
    model.input.solver.sections.section_continuity = [0] * n_sections
    model.input.solver.sections.section_times = [0.0] + elution.tolist()
    model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_abstol', 1e-10)
    model.input.solver.time_integrator.ALGTOL = kwargs.get('idas_algtol', 1e-8)
    model.input.solver.time_integrator.RELTOL = kwargs.get('idas_reltol', 1e-8)
    model.input.solver.time_integrator.INIT_STEP_SIZE = kwargs.get(
        'init_step_size', 1e-10)
    model.input.solver.time_integrator.MAX_STEPS = 1000000
    model.input.solver.user_solution_times = np.linspace(
        0.0, end_time, int(end_time) + 1)

    model.input.model.unit_000.UNIT_TYPE = 'INLET'
    model.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_000.ncomp = n_comp
    model.input.model.unit_000.sec_000.const_coeff = [loading] * n_comp
    model.input.model.unit_000.sec_000.lin_coeff = [0.0] * n_comp
    model.input.model.unit_000.sec_000.quad_coeff = [0.0] * n_comp
    model.input.model.unit_000.sec_000.cube_coeff = [0.0] * n_comp
    for index in range(1, n_sections + 1):
        section = model.input.model.unit_000[f'sec_{index:03d}']
        section.const_coeff = [0.0] * n_comp
        section.lin_coeff = [0.0] * n_comp
        section.quad_coeff = [0.0] * n_comp
        section.cube_coeff = [0.0] * n_comp

    model.input.model.unit_002.UNIT_TYPE = 'OUTLET'
    model.input.model.unit_002.ncomp = n_comp

    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].write_solution_times = 1
    model.input['return'].unit_000.write_solution_outlet = 0
    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_002.write_solution_outlet = 0

    return model
