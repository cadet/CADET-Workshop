# -*- coding: utf-8 -*-
"""

This script defines the load-wash-elute case study posted in CADET forum
thread #908, "Help with k_kin in SMA isotherm":

    https://forum.cadet-web.de/t/help-with-k-kin-in-sma-isotherm/908

A user tried to carry a parameter set over from GoSilico, which writes the
kinetic SMA as

    k_kin dq_i/dt = k_eq,i qbar^nu_i c_p,i - c_salt^nu_i q_i,

into CADET's convention

    dq_i/dt = k_a,i qbar^nu_i c_p,i - k_d,i c_salt^nu_i q_i,

by setting k_a = k_eq / k_kin and k_d = 1 / k_kin. With the reported
k_kin = 1.8e-8 this gives k_d = 5.6e7 and the simulation fails at the first
section transition. The same model runs as soon as IS_KINETIC is switched off.

The module builds the model for CADET-Core v6 (COLUMN_MODEL_1D with a
homogeneous particle, i.e. the lumped rate model with pores) and exposes the
knobs that the thread discusses, so that the failure and the proposed
workarounds can be reproduced from a notebook.

"""

import numpy as np
from addict import Dict

# --- the numbers exactly as posted in the thread -----------------------------
C_SALT_LOW = 10.0        # mol / m^3, load and wash
C_SALT_HIGH = 335.735    # mol / m^3, end of the elution gradient
C_PROTEIN = 0.00667      # mol / m^3, feed

FLOW_RATE = 2.312e-8     # m^3 / s
WASH_TIME = 713.22       # s, end of load
ELUTION_TIME = 2448.12   # s, start of the salt gradient
CYCLE_TIME = 12293.77    # s

K_KIN = 1.8e-8           # the GoSilico "Kinetic (SMA)" value
K_EQ = 7e-7              # the GoSilico "Equilibrium (SMA)" value
CAPACITY = 1400.0        # mol / m^3, SMA_LAMBDA
NU = 10.11               # characteristic charge


def get_binding_configuration(k_kin=K_KIN, is_kinetic=True, nu=NU, **kwargs):
    """SMA parameters in CADET's convention, derived from k_eq and k_kin.

    Parameters
    ----------
    k_kin : float
        GoSilico kinetic constant. Enters as k_a = k_eq / k_kin and
        k_d = 1 / k_kin, so a larger k_kin means slower binding.
    is_kinetic : bool
        False selects rapid equilibrium, where only the ratio k_a / k_d
        matters.
    nu : float
        Characteristic charge.

    Other Parameters
    ----------------
    sma_ka, sma_kd, sma_refc0, sma_refq : float
        Override the derived values, e.g. to study reference concentrations.
    """
    return {
        'is_kinetic': 1 if is_kinetic else 0,
        'sma_ka': [0.0, kwargs.get('sma_ka', K_EQ / k_kin)],
        'sma_kd': [0.0, kwargs.get('sma_kd', 1.0 / k_kin)],
        'sma_lambda': CAPACITY,
        'sma_nu': [0.0, nu],
        'sma_sigma': [0.0, 0.0],
        'sma_refc0': kwargs.get('sma_refc0', 1.0),
        'sma_refq': kwargs.get('sma_refq', 1.0),
    }


def get_model(**kwargs):
    """Build the thread's model.

    Other Parameters
    ----------------
    k_kin, is_kinetic, nu, sma_ka, sma_kd, sma_refc0, sma_refq
        Passed on to get_binding_configuration().
    c_protein : float
        Feed concentration, 0.0 switches the protein off entirely.
    ncol : int
        Number of axial finite volume cells.
    abstol, reltol, algtol, init_step_size, max_steps, max_errtest_fail,
    max_newton_iter, max_step_size, consistent_init_mode
        Time integrator settings.
    n_solution_times : int
        Size of the output grid.

    Returns
    -------
    addict.Dict
        Model ready to be handed to a Cadet object.
    """
    model = Dict()
    model.input.model.nunits = 3

    # --- Column: lumped rate model with pores, i.e. a homogeneous particle
    column = Dict()
    column.UNIT_TYPE = 'COLUMN_MODEL_1D'
    column.ncomp = 2
    column.geometry = 'AXIAL_FLOW_CYLINDER'
    column.bed_length = 0.2
    column.cross_section_area = np.pi * (0.0113 / 2.0) ** 2
    column.inlet_cross_section_area = np.pi * (0.0113 / 2.0) ** 2
    column.forward_flow = 1
    column.col_porosity = 0.89
    column.total_porosity = 0.89 + (1.0 - 0.89) * 0.83
    column.col_dispersion = 4e-07
    column.init_c = [C_SALT_LOW, 0.0]
    column.npartype = 1

    column.discretization.SPATIAL_METHOD = 'FV'
    column.discretization.NCOL = kwargs.get('ncol', 100)
    column.discretization.RECONSTRUCTION = 'WENO'
    column.discretization.weno.BOUNDARY_MODEL = 0
    column.discretization.weno.WENO_EPS = 1e-10
    column.discretization.weno.WENO_ORDER = 3
    column.discretization.GS_TYPE = 1
    column.discretization.MAX_KRYLOV = 0
    column.discretization.MAX_RESTARTS = 10
    column.discretization.SCHUR_SAFETY = 1e-8
    column.discretization.USE_ANALYTIC_JACOBIAN = kwargs.get(
        'use_analytic_jacobian', 1)

    particle = column.particle_type_000
    particle.has_film_diffusion = 1
    particle.has_pore_diffusion = 0
    particle.par_geom = 'SPHERE'
    particle.par_radius = 43e-5
    particle.par_coreradius = 0.0
    particle.par_porosity = 0.83
    particle.film_diffusion = [5e-06, 5e-06]
    particle.nbound = [1, 1]
    particle.init_cp = [C_SALT_LOW, 0.0]
    particle.init_cs = [CAPACITY, 0.0]
    particle.adsorption_model = 'STERIC_MASS_ACTION'
    particle.adsorption = get_binding_configuration(**kwargs)

    model.input.model.unit_000 = column

    # --- Inlet: load, wash, salt gradient
    inlet = model.input.model.unit_001
    inlet.UNIT_TYPE = 'INLET'
    inlet.ncomp = 2
    inlet.inlet_type = 'PIECEWISE_CUBIC_POLY'

    c_protein = kwargs.get('c_protein', C_PROTEIN)
    slope = (C_SALT_HIGH - C_SALT_LOW) / (CYCLE_TIME - ELUTION_TIME)

    profile = [
        ([C_SALT_LOW, c_protein], [0.0, 0.0]),   # load
        ([C_SALT_LOW, 0.0], [0.0, 0.0]),         # wash
        ([C_SALT_LOW, 0.0], [slope, 0.0]),       # elution gradient
    ]
    for i, (const_coeff, lin_coeff) in enumerate(profile):
        section = inlet['sec_{0:03d}'.format(i)]
        section.const_coeff = const_coeff
        section.lin_coeff = lin_coeff
        section.quad_coeff = [0.0, 0.0]
        section.cube_coeff = [0.0, 0.0]

    # --- Outlet
    model.input.model.unit_002.UNIT_TYPE = 'OUTLET'
    model.input.model.unit_002.ncomp = 2

    # --- Flow sheet
    model.input.model.connections.connections_include_ports = 0
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.section = 0
    model.input.model.connections.switch_000.connections = [
        1, 0, -1, -1, FLOW_RATE,
        0, 2, -1, -1, FLOW_RATE,
    ]

    # --- Global system solver
    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-8

    # --- Time integration
    model.input.solver.nthreads = 1
    model.input.solver.sections.nsec = 3
    model.input.solver.sections.section_continuity = [0, 0]
    model.input.solver.sections.section_times = [
        0.0, WASH_TIME, ELUTION_TIME, CYCLE_TIME]

    if 'consistent_init_mode' in kwargs:
        model.input.solver.consistent_init_mode = kwargs['consistent_init_mode']

    time_integrator = model.input.solver.time_integrator
    time_integrator.ABSTOL = kwargs.get('abstol', 1e-8)
    time_integrator.ALGTOL = kwargs.get('algtol', 1e-12)
    time_integrator.RELTOL = kwargs.get('reltol', 1e-6)
    time_integrator.INIT_STEP_SIZE = kwargs.get('init_step_size', 1e-6)
    time_integrator.MAX_STEPS = kwargs.get('max_steps', 1000000)
    for key, field in [('max_errtest_fail', 'MAX_ERRTEST_FAIL'),
                       ('max_newton_iter', 'MAX_NEWTON_ITER'),
                       ('max_step_size', 'MAX_STEP_SIZE')]:
        if key in kwargs:
            time_integrator[field] = kwargs[key]

    # --- Return data
    model.input.solver.user_solution_times = np.linspace(
        0, CYCLE_TIME, kwargs.get('n_solution_times', 2001))
    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].unit_000.write_solution_outlet = 1
    model.input['return'].unit_000.write_solution_bulk = kwargs.get(
        'write_solution_bulk', 0)
    model.input['return'].unit_000.write_solution_solid = kwargs.get(
        'write_solution_solid', 0)
    model.input['return'].unit_000.write_solution_particle = kwargs.get(
        'write_solution_particle', 0)
    model.input['return'].unit_000.write_coordinates = kwargs.get(
        'write_solution_bulk', 0)
    model.input['return'].write_solution_times = 1

    return model


def effective_rate_constants(k_kin=K_KIN, nu=NU, c_salt=C_SALT_LOW):
    """Linear rate constants of the binding ODE, dq/dt = A c_p - B q.

    SMA multiplies k_a and k_d by qbar^nu and c_salt^nu. With nu around 10
    those factors are what actually sets the speed of the binding, not k_a
    and k_d themselves.

    Returns
    -------
    (float, float)
        A = k_a qbar^nu evaluated at qbar = SMA_LAMBDA, and B = k_d c_salt^nu.
    """
    k_a = K_EQ / k_kin
    k_d = 1.0 / k_kin
    return k_a * CAPACITY ** nu, k_d * c_salt ** nu
