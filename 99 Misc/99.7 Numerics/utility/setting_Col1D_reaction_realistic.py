# -*- coding: utf-8 -*-
"""

A reversible reaction in the bulk liquid, on the same realistic lab column.

Pure transport apart from the reaction: no particle, no binding. Two components
A and B with

    A <-> B,   k_fwd / k_bwd = 4,

so four fifths of the material ends up as B once the reaction has had time to
reach equilibrium. The two rate constants are scaled **jointly**, which leaves
the equilibrium composition untouched and moves only the speed, exactly as the
binding study does with its isotherm.

The sweep can go one step further than any rate: `MAL_IS_KINETIC = 0` replaces
the rate equations by the equilibrium condition, which is the reaction
counterpart of rapid-equilibrium binding.

Column parameters match setting_Col1D_transport_realistic.

"""

import numpy as np
from addict import Dict

BED_LENGTH = 0.1
COLUMN_RADIUS = 0.005
BED_POROSITY = 0.37
SUPERFICIAL_VELOCITY = 4.2e-04
AXIAL_DISPERSION = 1.0e-07

FORWARD_RATE = 0.04            # 1/s at reaction_factor = 1
EQUILIBRIUM_CONSTANT = 4.0     # k_fwd / k_bwd

FEED = 1.0                     # mol/m^3 of A
INJECTION_TIME = 60.0
END_TIME = 400.0


def interstitial_velocity():
    """Interstitial velocity in m/s."""
    return SUPERFICIAL_VELOCITY / BED_POROSITY


def reaction_rate(reaction_factor=1.0):
    """Relaxation rate of the reversible reaction, k_fwd + k_bwd, in 1/s.

    A first order reversible reaction relaxes towards equilibrium at the sum of
    its two rate constants, whatever the equilibrium constant is.
    """
    forward = FORWARD_RATE * reaction_factor

    return forward + forward / EQUILIBRIUM_CONSTANT


def get_model(reaction_factor=1.0, is_kinetic=True, n_elements=100, **kwargs):
    """Build the column with a bulk reaction.

    Parameters
    ----------
    reaction_factor : float
        Common factor on both rate constants. The equilibrium is unchanged.
    is_kinetic : bool
        False selects the quasi-stationary reaction, i.e. the equilibrium
        condition instead of the rate equations.
    n_elements : int
        Number of axial finite volume cells.

    Returns
    -------
    addict.Dict
    """
    end_time = kwargs.get('end_time', END_TIME)
    cross_section = np.pi * COLUMN_RADIUS ** 2
    forward = FORWARD_RATE * reaction_factor

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

    column.ncomp = 2
    column.npartype = 0
    column.col_porosity = BED_POROSITY
    column.col_dispersion = kwargs.get('col_dispersion', AXIAL_DISPERSION)
    column.init_c = [0.0, 0.0]

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

    # A <-> B in the bulk liquid
    column.nreac_liquid = 1
    column.liquid_reaction_000.TYPE = 'MASS_ACTION_LAW'
    column.liquid_reaction_000.MAL_KFWD = [forward]
    column.liquid_reaction_000.MAL_KBWD = [forward / EQUILIBRIUM_CONSTANT]
    column.liquid_reaction_000.MAL_IS_KINETIC = [1 if is_kinetic else 0]
    column.liquid_reaction_000.MAL_STOICHIOMETRY = [-1.0, 1.0]

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
    model.input.model.unit_000.ncomp = 2
    model.input.model.unit_000.sec_000.const_coeff = [FEED, 0.0]
    model.input.model.unit_000.sec_000.lin_coeff = [0.0, 0.0]
    model.input.model.unit_000.sec_000.quad_coeff = [0.0, 0.0]
    model.input.model.unit_000.sec_000.cube_coeff = [0.0, 0.0]
    model.input.model.unit_000.sec_001.const_coeff = [0.0, 0.0]
    model.input.model.unit_000.sec_001.lin_coeff = [0.0, 0.0]
    model.input.model.unit_000.sec_001.quad_coeff = [0.0, 0.0]
    model.input.model.unit_000.sec_001.cube_coeff = [0.0, 0.0]

    model.input.model.unit_002.UNIT_TYPE = 'OUTLET'
    model.input.model.unit_002.ncomp = 2

    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].write_solution_times = 1
    model.input['return'].unit_000.write_solution_outlet = 0
    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_002.write_solution_outlet = 0

    return model
