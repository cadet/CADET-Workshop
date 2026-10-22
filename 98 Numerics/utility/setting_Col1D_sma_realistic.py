# -*- coding: utf-8 -*-
"""

Steric mass action on the realistic lab column: salt plus one protein, loaded
at low salt, washed, then eluted with a salt gradient.

This setting exists to study conditioning rather than stiffness. The SMA flux
is a difference of two terms,

    k_a c_p (Lambda - (nu + sigma) q)^nu   -   k_d q c_salt^nu,

and the exponent nu sits on quantities of order hundreds to a thousand. At
nu = 12 and a salt of 500 mol/m^3 the second factor alone is about 10^32. The
two terms have to cancel to whatever precision the solution needs, out of
numbers that large.

Two knobs change that without changing the chemistry:

    nu               the exponent, which sets how far the magnitudes are blown up
    kinetics_factor  a common factor on k_a and k_d, which leaves the
                     equilibrium untouched and scales both terms together

The equilibrium constant is retuned whenever nu changes, so that the protein
keeps eluting at roughly the same salt concentration and the chromatogram stays
comparable across the sweep.

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

IONIC_CAPACITY = 1200.0        # mol/m^3, Lambda
SHIELDING = 20.0               # sigma
TARGET_RETENTION = 2.0         # k' at the salt where we want it to elute

SALT_LOW = 50.0                # mol/m^3, load and wash
SALT_HIGH = 500.0              # mol/m^3, end of the gradient
ELUTION_SALT = 300.0           # where we want the protein to come off
PROTEIN_FEED = 0.05            # mol/m^3

LOAD_TIME = 120.0
WASH_TIME = 240.0
END_TIME = 1800.0

TOTAL_POROSITY = BED_POROSITY + (1.0 - BED_POROSITY) * PARTICLE_POROSITY


def interstitial_velocity():
    """Interstitial velocity in m/s."""
    return SUPERFICIAL_VELOCITY / BED_POROSITY


def equilibrium_constant(nu):
    """k_a / k_d chosen so that the protein elutes at a similar salt for any nu.

    The linear retention factor of an SMA protein is proportional to
    K (Lambda / c_salt)^nu, which moves by many decades when nu changes. Fixing
    the retention instead of the constant keeps the chromatogram comparable.
    """
    phase_ratio = (1.0 - TOTAL_POROSITY) / TOTAL_POROSITY

    return TARGET_RETENTION / (phase_ratio * (IONIC_CAPACITY / ELUTION_SALT) ** nu)


def flux_magnitude(nu, kinetics_factor=1.0, salt=ELUTION_SALT):
    """Size of the desorption term, k_d q c_salt^nu, as an order of magnitude.

    This is the number that has to cancel against the adsorption term. It is
    what the conditioning of the residual depends on.
    """
    desorption = kinetics_factor

    return desorption * salt ** nu


def get_binding_configuration(nu, kinetics_factor=1.0, is_kinetic=True):
    """SMA parameters for salt plus one protein."""
    desorption = kinetics_factor
    adsorption = equilibrium_constant(nu) * desorption

    return {
        'is_kinetic': 1 if is_kinetic else 0,
        'sma_ka': [0.0, adsorption],
        'sma_kd': [0.0, desorption],
        'sma_nu': [0.0, float(nu)],
        'sma_sigma': [0.0, SHIELDING],
        'sma_lambda': IONIC_CAPACITY,
    }


def get_model(nu=4.0, kinetics_factor=1.0, n_elements=50, is_kinetic=True,
              **kwargs):
    """Build the load, wash and gradient elution run.

    Parameters
    ----------
    nu : float
        Characteristic charge of the protein.
    kinetics_factor : float
        Common factor on k_a and k_d. The equilibrium is unchanged.
    n_elements : int
        Number of axial finite volume cells.
    is_kinetic : bool
        False selects rapid equilibrium binding.

    Returns
    -------
    addict.Dict
    """
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

    column.ncomp = 2
    column.npartype = 1
    column.col_porosity = BED_POROSITY
    column.total_porosity = TOTAL_POROSITY
    column.col_dispersion = kwargs.get('col_dispersion', AXIAL_DISPERSION)
    column.init_c = [SALT_LOW, 0.0]

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

    particle = column.particle_type_000
    particle.has_film_diffusion = True
    particle.has_pore_diffusion = False
    particle.par_geom = 'SPHERE'
    particle.par_radius = PARTICLE_RADIUS
    particle.par_coreradius = 0.0
    particle.par_porosity = PARTICLE_POROSITY
    particle.film_diffusion = [FILM_DIFFUSION, FILM_DIFFUSION]
    particle.init_cp = [SALT_LOW, 0.0]
    particle.nbound = [1, 1]
    particle.init_cs = [IONIC_CAPACITY, 0.0]
    particle.adsorption_model = 'STERIC_MASS_ACTION'
    particle.adsorption = get_binding_configuration(nu, kinetics_factor,
                                                    is_kinetic)

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
    model.input.solver.sections.nsec = 3
    model.input.solver.sections.section_continuity = [0, 0]
    model.input.solver.sections.section_times = [0.0, LOAD_TIME, WASH_TIME,
                                                 end_time]
    model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_abstol', 1e-08)
    model.input.solver.time_integrator.ALGTOL = kwargs.get('idas_algtol', 1e-06)
    model.input.solver.time_integrator.RELTOL = kwargs.get('idas_reltol', 1e-06)
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 1000000
    model.input.solver.user_solution_times = np.linspace(
        0.0, end_time, int(end_time) + 1)

    # load, wash, then a linear salt gradient
    inlet = model.input.model.unit_000
    inlet.UNIT_TYPE = 'INLET'
    inlet.inlet_type = 'PIECEWISE_CUBIC_POLY'
    inlet.ncomp = 2

    inlet.sec_000.const_coeff = [SALT_LOW, PROTEIN_FEED]
    inlet.sec_001.const_coeff = [SALT_LOW, 0.0]
    inlet.sec_002.const_coeff = [SALT_LOW, 0.0]
    for index in range(3):
        section = inlet[f'sec_{index:03d}']
        section.lin_coeff = [0.0, 0.0]
        section.quad_coeff = [0.0, 0.0]
        section.cube_coeff = [0.0, 0.0]
    inlet.sec_002.lin_coeff = [(SALT_HIGH - SALT_LOW) / (end_time - WASH_TIME),
                               0.0]

    model.input.model.unit_002.UNIT_TYPE = 'OUTLET'
    model.input.model.unit_002.ncomp = 2

    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].write_solution_times = 1
    model.input['return'].unit_000.write_solution_outlet = 0
    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_002.write_solution_outlet = 0

    return model
