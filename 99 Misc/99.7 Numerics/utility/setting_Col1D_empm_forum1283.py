# -*- coding: utf-8 -*-
"""

The setting from CADET forum thread 1283: DG, EMPM binding, and a crash.

    https://forum.cadet-web.de/t/error-with-dg-discretization-and-empm-binding-model/1283

A lumped rate model without pores is loaded with two components and then
washed. Binding is the extended mobile phase modulator, whose isotherm
contains the factor

    c_salt ** beta

with beta the ion exchange characteristic. For beta = 0 that factor is 1 for
every value of c_salt, including a negative one. For a non-integer beta it is
undefined as soon as c_salt goes below zero, and the residual returns a NaN,
which ends the integration.

Nothing in the model produces a negative concentration; the discretization
does. A discontinuous Galerkin element carrying a front steeper than its
polynomial can represent undershoots next to that front, and one undershoot
anywhere in the column is enough. Finite volumes with WENO is monotone and
does not produce one, which is why the same setting runs under FV.

The module is written directly against CADET-Python rather than through
CADET-Process, because CADET-Process 0.12.0 aliases `beta` onto
`ion_exchange_characteristic` and its `parameters` mapping returns None for
both, so the exponent never reaches CADET and the setting silently loses the
very parameter the thread is about.

"""

import numpy as np
from addict import Dict

LENGTH = 0.40                  # m
DIAMETER = 0.015               # m
POROSITY = 0.4
DISPERSION = 5e-5              # m^2/s
FLOW_RATE = 4e-7               # m^3/s
LOAD_TIME = 200.0              # s
END_TIME = 600.0               # s

FEED = (1.0, 5.0)              # mol/m^3, the loaded sample
ELUENT = (0.001, 0.001)        # mol/m^3, the wash
INITIAL_SOLID = (0.001, 0.001)

COMPONENT_NAMES = ('salt', 'protein')


def method_label(spatial_method):
    """Name of a spatial method: FV, or DG with its polynomial degree."""
    return 'FV' if spatial_method == 0 else f'DG{spatial_method}'


def spatial_dofs(spatial_method, n_cells):
    """Spatial discretization points per component.

    The lumped rate model without pores has only the bulk domain, so this is
    the number of axial nodes: one per finite volume cell, or degree + 1 per
    discontinuous Galerkin element.
    """
    nodes_per_element = 1 if spatial_method == 0 else spatial_method + 1

    return nodes_per_element * n_cells


def get_model(spatial_method=0, n_cells=20, beta=0.1, feed=FEED,
              eluent=ELUENT, idas_abstol=1e-10, idas_reltol=1e-8,
              end_time=END_TIME, n_time_points=601):
    """Build the forum setting.

    Parameters
    ----------
    spatial_method : int
        0 selects finite volumes with WENO, anything else is the polynomial
        degree of a discontinuous Galerkin discretization.
    n_cells : int
        Axial finite volume cells, or discontinuous Galerkin elements.
    beta : float
        The ion exchange characteristic of the binding model, the exponent of
        the salt concentration. The thread reports beta = 0.1; beta = 0 is the
        control, where the same undershoots are harmless because c ** 0 is 1
        for every c.

    Returns
    -------
    addict.Dict
    """
    area = np.pi * (DIAMETER / 2.0) ** 2

    model = Dict()
    model.input.model.nunits = 3
    model.input.model.connections.connections_include_ports = 0
    model.input.model.connections.nswitches = 1

    column = Dict()
    column.UNIT_TYPE = 'COLUMN_MODEL_1D'
    column.ncomp = 2
    column.geometry = 'AXIAL_FLOW_CYLINDER'
    column.bed_length = LENGTH
    column.cross_section_area = area
    column.inlet_cross_section_area = area
    column.total_porosity = POROSITY
    column.col_dispersion = DISPERSION
    column.forward_flow = 1
    column.npartype = 1
    column.init_c = list(eluent)

    if spatial_method > 0:
        column.discretization.SPATIAL_METHOD = 'DG'
        column.discretization.POLYDEG = spatial_method
        column.discretization.NELEM = n_cells
    else:
        column.discretization.SPATIAL_METHOD = 'FV'
        column.discretization.NCOL = n_cells
        column.discretization.RECONSTRUCTION = 'WENO'
        column.discretization.weno.BOUNDARY_MODEL = 0
        column.discretization.weno.WENO_EPS = 1e-10
        column.discretization.weno.WENO_ORDER = 3
    column.discretization.USE_ANALYTIC_JACOBIAN = 1

    # Lumped rate model without pores: no film, no pore, no surface
    # diffusion, so the binding sits directly on the bulk concentration.
    particle = column.particle_type_000
    particle.has_film_diffusion = 0
    particle.has_pore_diffusion = 0
    particle.has_surface_diffusion = 0
    particle.nbound = [1, 1]
    particle.init_cs = list(INITIAL_SOLID)
    particle.adsorption_model = 'EXTENDED_MOBILE_PHASE_MODULATOR'
    particle.adsorption.is_kinetic = 1
    particle.adsorption.empm_ka = [10.0, 1.0]
    particle.adsorption.empm_kd = [3.0, 1.0]
    particle.adsorption.empm_qmax = [10.0, 1.0]
    # The exponent this study is about.
    particle.adsorption.empm_beta = [beta, 0.0]
    particle.adsorption.empm_gamma = [0.0, 0.0]
    particle.adsorption.empm_comp_mode = [2, 0]

    model.input.model.unit_001 = column

    inlet = model.input.model.unit_000
    inlet.unit_type = 'INLET'
    inlet.ncomp = 2
    inlet.inlet_type = 'PIECEWISE_CUBIC_POLY'
    inlet.sec_000.const_coeff = list(feed)
    inlet.sec_001.const_coeff = list(eluent)
    for section in ('sec_000', 'sec_001'):
        for name in ('lin_coeff', 'quad_coeff', 'cube_coeff'):
            inlet[section][name] = [0.0, 0.0]

    model.input.model.unit_002.unit_type = 'OUTLET'
    model.input.model.unit_002.ncomp = 2

    model.input.model.connections.switch_000.section = 0
    model.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, FLOW_RATE,
        1.0, 2.0, -1.0, -1.0, FLOW_RATE,
    ]

    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08

    model.input.solver.sections.nsec = 2
    model.input.solver.sections.section_times = [0.0, LOAD_TIME, end_time]
    model.input.solver.sections.section_continuity = [0]
    model.input.solver.user_solution_times = np.linspace(0.0, end_time,
                                                         n_time_points)
    model.input.solver.nthreads = 1
    model.input.solver.consistent_init_mode = 1
    model.input.solver.time_integrator.ABSTOL = idas_abstol
    model.input.solver.time_integrator.RELTOL = idas_reltol
    model.input.solver.time_integrator.ALGTOL = idas_abstol * 100
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 1000000

    model.input['return'].split_components_data = 0
    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_001.write_solution_bulk = 1
    model.input['return'].unit_001.write_solution_solid = 1
    model.input['return'].write_solution_times = 1

    return model
