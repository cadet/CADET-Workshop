# -*- coding: utf-8 -*-
"""

This script defines a multi component Langmuir case study with a configurable
transport model, number of components, discretization and binding kinetics.

It exists so that two settings can be built that differ in everything the eye
notices, i.e. transport model, number of components and number of degrees of
freedom, while the binding kinetics, which is what actually decides the compute
time, is just one more row in the parameter table.

Derived from setting_Col1D_langLRM_2comp_benchmark1.py.

"""

import numpy as np
from addict import Dict

# Base Langmuir parameters, repeated and spread over the requested components.
_KA = [0.1, 0.05, 0.08, 0.03, 0.12, 0.06, 0.09, 0.04]
_QMAX = [10.0, 10.0, 8.0, 12.0, 9.0, 11.0, 10.0, 10.0]
_FEED = [10.0, 10.0, 8.0, 6.0, 9.0, 7.0, 8.0, 6.0]


def get_binding_configuration(n_comp, kinetics_factor=1.0, is_kinetic=True,
                              qmax_scale=1.0):
    """Multi component Langmuir binding with a tunable relaxation time.

    `kinetics_factor` scales adsorption and desorption rate constants jointly,
    so the equilibrium constant ka/kd, and therefore the chromatogram, stays
    where it is while the binding gets faster or slower. That is the knob the
    first dedicated stiffness notebook turns.

    Parameters
    ----------
    n_comp : int
        Number of components.
    kinetics_factor : float
        Common factor on ka and kd. Larger means faster binding.
    is_kinetic : bool
        False replaces the rate equations by the equilibrium condition.

    Returns
    -------
    dict
        Adsorption configuration.
    """
    if n_comp > len(_KA):
        raise ValueError(f"At most {len(_KA)} components are parameterised.")

    return {
        'is_kinetic': 1 if is_kinetic else 0,
        'mcl_ka': [ka * kinetics_factor for ka in _KA[:n_comp]],
        'mcl_kd': [1.0 * kinetics_factor] * n_comp,
        'mcl_qmax': [q * qmax_scale for q in _QMAX[:n_comp]],
    }


def get_model(
        spatial_method_bulk,
        n_comp=2,
        transport_model='LRM',
        n_elements=12,
        kinetics_factor=1.0,
        is_kinetic=True,
        **kwargs):
    """Build a multi component Langmuir column model.

    Parameters
    ----------
    spatial_method_bulk : int
        Polynomial degree for DG, 0 for FV.
    n_comp : int
        Number of components.
    transport_model : string
        'LRM' for the lumped rate model, 'LRMP' for the lumped rate model with
        pores, i.e. with film diffusion and a particle liquid phase.
    n_elements : int
        Number of axial elements (DG) or cells (FV).
    kinetics_factor : float
        Common factor on the binding rate constants, see
        get_binding_configuration().
    is_kinetic : bool
        False uses rapid equilibrium binding.

    Returns
    -------
    addict.Dict
        Model configuration.
    """
    if transport_model not in ('LRM', 'LRMP'):
        raise ValueError(f"Unknown transport model: {transport_model}")

    model = Dict()
    model.input.model.nunits = 3
    model.input.model.connections.connections_include_ports = 0
    model.input.model.connections.nswitches = 1

    bed_length = 0.1
    inlet_radius = 0.02
    cross_section_area = np.pi * inlet_radius**2

    column = Dict()
    column.unit_type = 'COLUMN_MODEL_1D'
    column.geometry = 'AXIAL_FLOW_CYLINDER'
    column.inlet_cross_section_area = cross_section_area
    column.cross_section_area = cross_section_area
    column.bed_length = bed_length
    column.forward_flow = 1

    column.ncomp = n_comp
    column.npartype = 1
    column.col_dispersion = kwargs.get('col_dispersion', 1e-07)
    column.init_c = [0.0] * n_comp

    # The two transport models are compared at the same total porosity, so that
    # the retention of the components is the same and the chromatograms are
    # comparable. The LRMP splits that porosity into bed and particle.
    total_porosity = 0.4
    if transport_model == 'LRM':
        column.total_porosity = total_porosity
    else:
        column.col_porosity = kwargs.get('col_porosity', 0.37)
        column.total_porosity = total_porosity

    velocity = 0.01
    flow_rate = velocity * cross_section_area * total_porosity
    model.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, flow_rate,
        1.0, 2.0, -1.0, -1.0, flow_rate,
    ]
    model.input.model.connections.switch_000.section = 0

    if spatial_method_bulk > 0:
        column.discretization.SPATIAL_METHOD = 'DG'
        column.discretization.USE_COLLOCATION_DG = kwargs.get('use_collocation_dg', 0)
        column.discretization.POLYDEG = spatial_method_bulk
        column.discretization.NELEM = n_elements
    else:
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

    # Particle phase
    if transport_model == 'LRM':
        column.particle_type_000.has_film_diffusion = False
    else:
        column.particle_type_000.has_film_diffusion = True
        column.particle_type_000.par_geom = 'SPHERE'
        column.particle_type_000.par_radius = kwargs.get('par_radius', 4.5e-05)
        column.particle_type_000.par_coreradius = 0.0
        column.particle_type_000.par_porosity = kwargs.get('par_porosity', 0.75)
        column.particle_type_000.film_diffusion = \
            [kwargs.get('film_diffusion', 1.0e-05)] * n_comp
        column.particle_type_000.init_cp = [0.0] * n_comp

    column.particle_type_000.nbound = [1] * n_comp
    column.particle_type_000.init_cs = [0.0] * n_comp
    column.particle_type_000.adsorption_model = 'MULTI_COMPONENT_LANGMUIR'
    column.particle_type_000.adsorption = get_binding_configuration(
        n_comp, kinetics_factor=kinetics_factor, is_kinetic=is_kinetic,
        qmax_scale=kwargs.get('qmax_scale', 1.0)
    )

    model.input.model.unit_001 = column

    # Nonlinear solver
    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08

    # Time integration
    t_end = kwargs.get('t_end', 40.0)
    model.input.solver.consistent_init_mode = 1
    model.input.solver.consistent_init_mode_sens = 3
    model.input.solver.nthreads = 1
    model.input.solver.sections.nsec = 2
    model.input.solver.sections.section_continuity = [0]
    model.input.solver.sections.section_times = [0.0, 12.0, t_end]

    abstol = kwargs.get('idas_abstol', 1e-10)
    reltol = kwargs.get('idas_reltol', 1e-8)
    model.input.solver.time_integrator.ABSTOL = abstol
    model.input.solver.time_integrator.ALGTOL = kwargs.get('idas_algtol', 1e-8)
    model.input.solver.time_integrator.RELTOL = reltol
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 1000000
    model.input.solver.user_solution_times = np.linspace(0.0, t_end, int(t_end) * 10 + 1)

    # Inlet and outlet
    model.input.model.unit_000.UNIT_TYPE = 'INLET'
    model.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_000.ncomp = n_comp
    feed_scale = kwargs.get('feed_scale', 1.0)
    model.input.model.unit_000.sec_000.const_coeff =         [c * feed_scale for c in _FEED[:n_comp]]
    model.input.model.unit_000.sec_000.lin_coeff = [0.0] * n_comp
    model.input.model.unit_000.sec_000.quad_coeff = [0.0] * n_comp
    model.input.model.unit_000.sec_000.cube_coeff = [0.0] * n_comp
    model.input.model.unit_000.sec_001.const_coeff = [0.0] * n_comp
    model.input.model.unit_000.sec_001.lin_coeff = [0.0] * n_comp
    model.input.model.unit_000.sec_001.quad_coeff = [0.0] * n_comp
    model.input.model.unit_000.sec_001.cube_coeff = [0.0] * n_comp

    model.input.model.unit_002.UNIT_TYPE = 'OUTLET'
    model.input.model.unit_002.ncomp = n_comp

    # Return data
    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].write_solution_times = 1
    model.input['return'].unit_000.write_solution_outlet = 0
    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_001.write_solution_bulk = kwargs.get('write_solution_bulk', 0)
    model.input['return'].unit_001.write_coordinates = kwargs.get('write_solution_bulk', 0)
    model.input['return'].unit_002.write_solution_outlet = 0

    return model


def degrees_of_freedom(n_comp, transport_model, polynomial_degree, n_elements):
    """Number of unknowns of the spatially discretized system.

    Parameters
    ----------
    n_comp : int
        Number of components.
    transport_model : string
        'LRM' or 'LRMP'.
    polynomial_degree : int
        DG polynomial degree, 0 for FV.
    n_elements : int
        Number of axial elements or cells.

    Returns
    -------
    int
        Bulk plus particle plus solid unknowns.
    """
    nodes = n_elements * (polynomial_degree + 1 if polynomial_degree > 0 else 1)

    # bulk liquid + bound states, plus the particle liquid phase for the LRMP
    phases = 3 if transport_model == 'LRMP' else 2

    return nodes * n_comp * phases
