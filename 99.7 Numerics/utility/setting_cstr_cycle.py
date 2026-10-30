# -*- coding: utf-8 -*-
"""

A cyclic reaction in a stirred tank: the boundary case of this chapter.

Every other study here found fast processes to be free, because every
dissipative term in a chromatography model contributes a real negative
eigenvalue and BDF is stable along that axis at any magnitude. That argument
rests on the terms being symmetrizable, which needs positive rates and
something conserved.

A closed cycle with no reverse reactions,

    X_0 -> X_1 -> ... -> X_{N-1} -> X_0,

satisfies neither. Its Jacobian is a circulant matrix, and its eigenvalues are

    lambda_j = k (exp(2 pi i j / N) - 1) - Q/V,

which sit at an angle of 90 - 180/N degrees from the negative real axis, shifted
left by the dilution rate. Short cycles stay inside the sector BDF5 can handle;
from six species on they do not.

A stirred tank rather than a column on purpose: there is no spatial
discretization, so the Jacobian *is* the reaction network and nothing else can
be blamed. It is also the harder test, because the tank reaches a steady state
and the error control then stretches the step size while the fast mode is
invisible, which is exactly the situation stiffness is supposed to punish.

This setting is deliberately not chromatography. It is here to find where the
argument stops.

"""

import numpy as np
from addict import Dict

VOLUME = 1.0e-06               # m^3, a 1 mL tank
FLOW_RATE = 3.3e-08            # m^3/s, about 2 mL/min
FEED = 1.0                     # mol/m^3 of the first species
END_TIME = 400.0


def dilution_rate():
    """Q / V, the rate at which the tank exchanges its contents, in 1/s."""
    return FLOW_RATE / VOLUME


def eigenvalues(n_species, rate):
    """Exact eigenvalues of the tank with the cycle running, in 1/s."""
    roots = np.exp(2j * np.pi * np.arange(n_species) / n_species)

    return rate * (roots - 1.0) - dilution_rate()


def worst_angle(n_species, rate):
    """Largest angle away from the negative real axis, in degrees.

    BDF of order five is unconditionally stable within about 52 degrees. Beyond
    that, a large enough step size leaves the stable region.
    """
    values = eigenvalues(n_species, rate)
    values = values[np.abs(values.imag) > 1e-12]

    if not values.size:
        return 0.0

    return float(np.degrees(np.abs(np.angle(-values))).max())


def stoichiometry(n_species):
    """Row major NVAR x NREACT matrix for X_i -> X_(i+1), wrapping around."""
    matrix = np.zeros((n_species, n_species))
    for reaction in range(n_species):
        matrix[reaction, reaction] = -1.0
        matrix[(reaction + 1) % n_species, reaction] = 1.0

    return matrix.flatten().tolist()


def get_model(n_species=6, rate=1.0, **kwargs):
    """Build the stirred tank with a cyclic reaction.

    Parameters
    ----------
    n_species : int
        Length of the cycle.
    rate : float
        Forward rate constant of every step, in 1/s.

    Returns
    -------
    addict.Dict
    """
    end_time = kwargs.get('end_time', END_TIME)

    model = Dict()
    model.input.model.nunits = 3
    model.input.model.connections.connections_include_ports = 0
    model.input.model.connections.nswitches = 1

    tank = Dict()
    tank.UNIT_TYPE = 'CSTR'
    tank.ncomp = n_species
    tank.npartype = 0
    tank.init_c = [0.0] * n_species
    tank.init_liquid_volume = VOLUME
    tank.const_solid_volume = 0.0
    tank.flowrate_filter = [0.0]
    tank.use_analytic_jacobian = 1

    tank.nreac_liquid = 1
    tank.liquid_reaction_000.TYPE = 'MASS_ACTION_LAW'
    tank.liquid_reaction_000.MAL_KFWD = [rate] * n_species
    tank.liquid_reaction_000.MAL_KBWD = [0.0] * n_species
    tank.liquid_reaction_000.MAL_IS_KINETIC = [1] * n_species
    tank.liquid_reaction_000.MAL_STOICHIOMETRY = stoichiometry(n_species)

    model.input.model.unit_001 = tank

    model.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, FLOW_RATE,
        1.0, 2.0, -1.0, -1.0, FLOW_RATE,
    ]
    model.input.model.connections.switch_000.section = 0

    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08

    model.input.solver.consistent_init_mode = 1
    model.input.solver.consistent_init_mode_sens = 3
    model.input.solver.nthreads = 1
    model.input.solver.sections.nsec = 1
    model.input.solver.sections.section_times = [0.0, end_time]
    model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_abstol', 1e-10)
    model.input.solver.time_integrator.ALGTOL = kwargs.get('idas_algtol', 1e-8)
    model.input.solver.time_integrator.RELTOL = kwargs.get('idas_reltol', 1e-8)
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 1000000
    model.input.solver.user_solution_times = np.linspace(
        0.0, end_time, int(end_time) * 2 + 1)

    inlet = model.input.model.unit_000
    inlet.UNIT_TYPE = 'INLET'
    inlet.inlet_type = 'PIECEWISE_CUBIC_POLY'
    inlet.ncomp = n_species
    inlet.sec_000.const_coeff = [FEED] + [0.0] * (n_species - 1)
    inlet.sec_000.lin_coeff = [0.0] * n_species
    inlet.sec_000.quad_coeff = [0.0] * n_species
    inlet.sec_000.cube_coeff = [0.0] * n_species

    model.input.model.unit_002.UNIT_TYPE = 'OUTLET'
    model.input.model.unit_002.ncomp = n_species

    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].write_solution_times = 1
    model.input['return'].unit_000.write_solution_outlet = 0
    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_002.write_solution_outlet = 0

    return model
