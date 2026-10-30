# -*- coding: utf-8 -*-
"""

Pure transport on the same realistic lab column: convection and axial
dispersion, nothing else. No particles, no binding, no reactions.

This is the most isolated setting of the chapter, and it exists for one
question. Every dissipative process in a chromatography model contributes a
real, negative eigenvalue, and BDF is unconditionally stable for those. The one
operator that does not is convection: discretized, its eigenvalues reach
towards the imaginary axis, where BDF of order 5 is only stable within 52
degrees of the negative real axis. How far they reach depends on how much
numerical dissipation the spatial scheme adds, which is what the polynomial
degree controls.

So the knob is the DG polynomial degree, swept at a **fixed number of degrees
of freedom**: P1 on 80 elements, P3 on 40, P7 on 20 and so on all carry 160
nodes. The system size is constant, the physics is constant, and only the
shape of the spectrum changes.

Column parameters are those of setting_Col1D_langmuir_realistic, minus the
particle:

    bed length          0.1 m
    diameter            0.01 m
    superficial vel.    4.2e-4 m/s      150 cm/h
    bed porosity        0.37
    axial dispersion    1e-7 m^2/s      column Peclet about 1100

"""

import numpy as np
from addict import Dict

BED_LENGTH = 0.1
COLUMN_RADIUS = 0.005
BED_POROSITY = 0.37
SUPERFICIAL_VELOCITY = 4.2e-04
AXIAL_DISPERSION = 1.0e-07

FEED = 1.0                 # mol/m^3
INJECTION_TIME = 60.0      # s
END_TIME = 300.0           # s

# Combinations of polynomial degree and element count that all carry the same
# number of axial nodes, so that the system size is held fixed.
TOTAL_NODES = 160
DEGREES = (1, 3, 4, 7, 9, 15)


def interstitial_velocity():
    """Interstitial velocity in m/s."""
    return SUPERFICIAL_VELOCITY / BED_POROSITY


def elements_for(polynomial_degree, total_nodes=TOTAL_NODES):
    """Element count that gives the requested number of nodes."""
    nodes_per_element = polynomial_degree + 1
    if total_nodes % nodes_per_element:
        raise ValueError(
            f"polynomial degree {polynomial_degree} does not divide "
            f"{total_nodes} nodes evenly")

    return total_nodes // nodes_per_element


def peclet(col_dispersion=AXIAL_DISPERSION):
    """Column Peclet number, interstitial velocity times length over dispersion."""
    return interstitial_velocity() * BED_LENGTH / col_dispersion


def spectral_reach(polynomial_degree, total_nodes=TOTAL_NODES):
    """Order of magnitude of the convective eigenvalues, in 1/s.

    For DG the node spacing inside an element is finer than the element itself,
    and the spectral radius of the discrete convection operator grows roughly
    with the polynomial degree at a fixed number of nodes.
    """
    elements = elements_for(polynomial_degree, total_nodes)

    return (interstitial_velocity() * (polynomial_degree + 1) ** 2
            * elements / BED_LENGTH)


def front_width(col_dispersion=AXIAL_DISPERSION):
    """Spread of a step front after travelling the column, in m.

    Dispersion broadens a front as sqrt(2 D t), and the front reaches the
    outlet after t = L / u. This is the physical length the grid has to
    resolve, and the only thing the dispersion knob changes.
    """
    travel_time = BED_LENGTH / interstitial_velocity()

    return np.sqrt(2.0 * col_dispersion * travel_time)


def get_model(polynomial_degree=3, total_nodes=TOTAL_NODES, **kwargs):
    """Build the pure transport column.

    Parameters
    ----------
    polynomial_degree : int
        DG polynomial degree. Zero selects finite volumes with WENO, which is
        the most dissipative option and serves as the baseline.
    total_nodes : int
        Number of axial degrees of freedom.
    mode : string
        'pulse' injects for INJECTION_TIME and then washes, which needs two
        sections. 'step' feeds for the whole run, so there is exactly one
        front and no section transition to confound it. 'gradient' ramps the
        inlet from zero to FEED over the whole run, implemented either as one
        section with a slope or as `n_sections` constant steps.
    n_sections : int
        For the gradient: how many sections the ramp is cut into. One uses a
        linear coefficient and has no interior transition at all.

    Returns
    -------
    addict.Dict
    """
    end_time = kwargs.get('end_time', END_TIME)
    mode = kwargs.get('mode', 'pulse')
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
    column.npartype = 0                       # no particle phase at all
    # Without a particle phase CADET asks for the bed porosity by name.
    column.col_porosity = BED_POROSITY
    column.col_dispersion = kwargs.get('col_dispersion', AXIAL_DISPERSION)
    column.init_c = [0.0]

    if polynomial_degree > 0:
        column.discretization.SPATIAL_METHOD = 'DG'
        column.discretization.USE_COLLOCATION_DG = kwargs.get('use_collocation_dg', 0)
        column.discretization.POLYDEG = polynomial_degree
        column.discretization.NELEM = elements_for(polynomial_degree, total_nodes)
    else:
        column.discretization.SPATIAL_METHOD = 'FV'
        column.discretization.NCOL = total_nodes
        column.discretization.RECONSTRUCTION = 'WENO'
        column.discretization.weno.BOUNDARY_MODEL = 0
        column.discretization.weno.WENO_EPS = 1e-10
        column.discretization.weno.WENO_ORDER = 3
        column.discretization.GS_TYPE = 1
        column.discretization.MAX_KRYLOV = 0
        column.discretization.MAX_RESTARTS = 10
        column.discretization.SCHUR_SAFETY = 1.0e-8
    column.discretization.USE_ANALYTIC_JACOBIAN = 1

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
    n_sections = int(kwargs.get('n_sections', 1))

    if mode == 'gradient':
        boundaries = np.linspace(0.0, end_time, n_sections + 1)
        model.input.solver.sections.nsec = n_sections
        model.input.solver.sections.section_times = boundaries.tolist()
        model.input.solver.sections.section_continuity = [0] * (n_sections - 1)
    elif mode == 'step':
        model.input.solver.sections.nsec = 1
        model.input.solver.sections.section_times = [0.0, end_time]
    else:
        model.input.solver.sections.nsec = 2
        model.input.solver.sections.section_continuity = [0]
        model.input.solver.sections.section_times = [0.0, INJECTION_TIME, end_time]
    model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_abstol', 1e-10)
    model.input.solver.time_integrator.ALGTOL = kwargs.get('idas_algtol', 1e-8)
    model.input.solver.time_integrator.RELTOL = kwargs.get('idas_reltol', 1e-8)
    model.input.solver.time_integrator.INIT_STEP_SIZE = kwargs.get(
        'init_step_size', 1e-10)
    model.input.solver.time_integrator.MAX_STEPS = 1000000
    model.input.solver.user_solution_times = np.linspace(
        0.0, end_time, int(end_time) * 2 + 1)

    model.input.model.unit_000.UNIT_TYPE = 'INLET'
    model.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_000.ncomp = 1
    feed = kwargs.get('feed', FEED)

    if mode == 'gradient':
        # The same ramp either way: one section carrying a slope, or n sections
        # each holding the value the ramp has at its midpoint.
        for index in range(n_sections):
            section = model.input.model.unit_000[f'sec_{index:03d}']
            if n_sections == 1:
                section.const_coeff = [0.0]
                section.lin_coeff = [feed / end_time]
            else:
                midpoint = (index + 0.5) / n_sections
                section.const_coeff = [feed * midpoint]
                section.lin_coeff = [0.0]
            section.quad_coeff = [0.0]
            section.cube_coeff = [0.0]
    else:
        model.input.model.unit_000.sec_000.const_coeff = [feed]
        model.input.model.unit_000.sec_000.lin_coeff = [0.0]
        model.input.model.unit_000.sec_000.quad_coeff = [0.0]
        model.input.model.unit_000.sec_000.cube_coeff = [0.0]
    if mode == 'pulse':
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
