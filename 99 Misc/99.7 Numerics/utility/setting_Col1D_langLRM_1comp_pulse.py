"""

Single-component Lumped Rate Model (LRM) with kinetic Langmuir adsorption and
a short rectangular pulse injection.

The column is discretized by FV (polynomial_degree = 0) or DG
(polynomial_degree > 0). The time integrator tolerances are exposed so that
the effect of RELTOL and ABSTOL can be studied on a fixed spatial grid.

"""

from addict import Dict
import numpy as np

BED_LENGTH = 0.1            # m
TOTAL_POROSITY = 0.4        # -
CROSS_SECTION_AREA = 1e-4   # m^2
INTERSTITIAL_VELOCITY = 1e-3  # m/s
COL_DISPERSION = 1e-6       # m^2/s

LANGMUIR_KA = 1.0           # m^3/(mol s)
LANGMUIR_KD = 1.0           # 1/s
LANGMUIR_QMAX = 2.0         # mol/m^3

FEED_CONCENTRATION = 5.0    # mol/m^3
PULSE_DURATION = 10.0       # s
END_TIME = 800.0            # s
OUTPUTS_PER_SECOND = 2


def flow_rate():
    """Volumetric flow rate in m^3/s."""
    return INTERSTITIAL_VELOCITY * CROSS_SECTION_AREA * TOTAL_POROSITY


def solution_times():
    """Output times in s."""
    return np.linspace(0.0, END_TIME, int(END_TIME * OUTPUTS_PER_SECOND) + 1)


def get_model(
        polynomial_degree=3,
        n_elements=32,
        reltol=1e-6,
        abstol=1e-8,
        algtol=1e-10,
        **kwargs):
    """Build the CADET input for the pulse injection.

    Parameters
    ----------
    polynomial_degree : int
        0 selects FV (WENO3), values > 0 select DG with this polynomial degree.
    n_elements : int
        Number of axial cells (FV) or elements (DG).
    reltol, abstol, algtol : float
        IDAS time integrator tolerances.
    """
    model = Dict()

    model.input.model.nunits = 3

    #%% Flow sheet
    Q = flow_rate()
    model.input.model.connections.connections_include_ports = 0
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]
    model.input.model.connections.switch_000.section = 0

    #%% Column unit
    column = Dict()

    column.UNIT_TYPE = 'COLUMN_MODEL_1D'
    column.ncomp = 1
    column.geometry = 'AXIAL_FLOW_CYLINDER'
    column.bed_length = BED_LENGTH
    column.total_porosity = TOTAL_POROSITY
    column.cross_section_area = CROSS_SECTION_AREA
    column.col_dispersion = kwargs.get('col_dispersion', COL_DISPERSION)
    column.forward_flow = 1
    column.npartype = 1
    column.init_c = [0.0]

    if polynomial_degree > 0:
        column.discretization.SPATIAL_METHOD = 'DG'
        column.discretization.POLYDEG = polynomial_degree
        column.discretization.NELEM = n_elements
    else:
        column.discretization.SPATIAL_METHOD = 'FV'
        column.discretization.NCOL = n_elements
        column.discretization.RECONSTRUCTION = 'WENO'
        column.discretization.weno.BOUNDARY_MODEL = 0
        column.discretization.weno.WENO_EPS = 1e-10
        column.discretization.weno.WENO_ORDER = 3
    column.discretization.USE_ANALYTIC_JACOBIAN = 1

    # Lumped rate model: no film, pore or surface diffusion
    particle = column.particle_type_000
    particle.has_film_diffusion = 0
    particle.has_pore_diffusion = 0
    particle.has_surface_diffusion = 0
    particle.nbound = [1]
    particle.init_cs = [0.0]

    particle.adsorption_model = 'MULTI_COMPONENT_LANGMUIR'
    particle.adsorption.is_kinetic = 1
    particle.adsorption.mcl_ka = [kwargs.get('ka', LANGMUIR_KA)]
    particle.adsorption.mcl_kd = [kwargs.get('kd', LANGMUIR_KD)]
    particle.adsorption.mcl_qmax = [kwargs.get('qmax', LANGMUIR_QMAX)]

    model.input.model.unit_001 = column

    #%% Inlet: short rectangular pulse, then pure buffer
    inlet = model.input.model.unit_000
    inlet.UNIT_TYPE = 'INLET'
    inlet.inlet_type = 'PIECEWISE_CUBIC_POLY'
    inlet.ncomp = 1
    inlet.sec_000.const_coeff = [kwargs.get('feed', FEED_CONCENTRATION)]
    inlet.sec_001.const_coeff = [0.0]
    for section in ('sec_000', 'sec_001'):
        inlet[section].lin_coeff = [0.0]
        inlet[section].quad_coeff = [0.0]
        inlet[section].cube_coeff = [0.0]

    model.input.model.unit_002.UNIT_TYPE = 'OUTLET'
    model.input.model.unit_002.ncomp = 1

    #%% Solver
    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-8

    model.input.solver.nthreads = 1
    model.input.solver.consistent_init_mode = 1
    model.input.solver.sections.nsec = 2
    model.input.solver.sections.section_times = [0.0, PULSE_DURATION, END_TIME]
    model.input.solver.sections.section_continuity = [0]
    model.input.solver.user_solution_times = solution_times()

    model.input.solver.time_integrator.RELTOL = reltol
    model.input.solver.time_integrator.ABSTOL = abstol
    model.input.solver.time_integrator.ALGTOL = algtol
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-6
    model.input.solver.time_integrator.MAX_STEPS = 1000000

    #%% Return data
    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].unit_001.write_solution_outlet = 1

    return model
