# -*- coding: utf-8 -*-
"""

This script defines steric mass action load-wash-elute case studies used for
numerical benchmarks in https://doi.org/10.1016/j.compchemeng.2023.108340

"""

import numpy as np
from addict import Dict


def get_binding_configuration(is_kinetic:bool=True):
    return {
        'is_kinetic': 1 if is_kinetic else 0,
        'sma_ka': [ 0.0, 35.5, 1.59, 7.7 ],
        'sma_kd': [ 0.0, 1000.0, 1000.0, 1000.0],
        'sma_lambda': 1200.0,
        'sma_nu': [ 0.0, 4.7, 5.29, 3.7 ],
        'sma_sigma': [ 0.0, 11.83, 10.6, 10.0 ]
        }


def get_column_geometry_configuration(geometry: str):

    # for all geometries:
    # velocity = 0.000575 m/s
    # col_porosity = 0.37
    # bed length is 0.014 m
    # col radius at inlet is 0.35cm = 0.0035m
    # flow rate Q = velocity / (cross_section * col_porosity)
    axial_flow_cross_section_area = np.pi * 0.0035 * 0.0035

    if geometry == 'AXIAL_FLOW_CYLINDER':
        return {
            'geometry': geometry,
            # A = v * \pi * r^2 * \varepsilon
            'cross_section_area': axial_flow_cross_section_area,
            'inlet_cross_section_area': axial_flow_cross_section_area,
            'bed_length': 0.014,
        }
    elif geometry == 'RADIAL_FLOW_CYLINDER_SHELL':
        cylinder_height = 0.00025
        col_radius_outer = axial_flow_cross_section_area / 2.0 / np.pi / cylinder_height
        col_radius_inner = col_radius_outer - 0.014
        return {
            'geometry': geometry,
            # A = 2 * pi * \rho * L^b -> \rho = A / 2.0 / pi / L^b
            'inlet_cross_section_area': axial_flow_cross_section_area,
            'cross_section_area_outer': axial_flow_cross_section_area,
            'cylinder_height': cylinder_height,
            'cross_section_area_outer': axial_flow_cross_section_area,
            'cross_section_area_inner': 2.0 * np.pi * col_radius_inner * cylinder_height,
            'bed_length': 0.014,
        }
    elif geometry == 'AXIAL_FLOW_FRUSTUM':
        return {
            'geometry': geometry,
            'inlet_cross_section_area': axial_flow_cross_section_area,
            'cross_section_area_large_end': axial_flow_cross_section_area,
            'cross_section_area_small_end': axial_flow_cross_section_area * 0.75,
            'bed_length': 0.014,
        }
    else:
        raise ValueError(f"Unknown geometry: {geometry}")


def get_model(
        spatial_method_bulk,
        particle_type='GENERAL_RATE_PARTICLE',
        refinement=1,
        column_geometry='AXIAL_FLOW_CYLINDER',
        **kwargs):
    
    # The axial resolution is either given directly as a number of cells /
    # elements, or as a refinement factor on the base resolution of 8.
    axNElem = int(kwargs.get('axNElem', 8 * kwargs.get('axRefinement', refinement)))
    parNElem = int(kwargs.get('parZ', 1))
    
    model = Dict()
    
    model.input.model.nunits = 2
    
    # Column unit
    column = Dict()
    column.UNIT_TYPE = 'COLUMN_MODEL_1D'
    column.update(get_column_geometry_configuration(column_geometry))
    column.forward_flow = 1
    column.col_porosity = 0.37
    column.total_porosity = 0.8425
    column.npartype = 0 if particle_type is None else 1
    
    column.ncomp = 4
    column.init_c = [50.0, 0.0, 0.0, 0.0]
    column.col_dispersion = 5.75e-08

    # Spatial discretization of interstitial / bulk volume
    if spatial_method_bulk > 0:
        column.discretization.SPATIAL_METHOD = 'DG'
        column.discretization.USE_COLLOCATION_DG = kwargs.get('use_collocation_dg', 1 if column_geometry == 'AXIAL_FLOW_CYLINDER' else 0)
        column.discretization.POLYDEG = spatial_method_bulk
        column.discretization.NELEM = axNElem
    else:
        column.discretization.SPATIAL_METHOD = 'FV'
        column.discretization.NCOL = axNElem
        column.discretization.RECONSTRUCTION = 'WENO'
        column.discretization.weno.BOUNDARY_MODEL = 0
        column.discretization.weno.WENO_EPS = 1e-10
        column.discretization.weno.WENO_ORDER = 3
        column.discretization.GS_TYPE = 1
        column.discretization.MAX_KRYLOV = 0
        column.discretization.MAX_RESTARTS = 10
        column.discretization.SCHUR_SAFETY = 1.0e-8
    column.discretization.USE_ANALYTIC_JACOBIAN = 1
    
    if particle_type is not None:
        
        if particle_type in ['HOMOGENEOUS_PARTICLE', 'GENERAL_RATE_PARTICLE']:
            
            column.particle_type_000.has_film_diffusion = 1
            column.particle_type_000.par_geom = 'SPHERE'
            column.particle_type_000.par_radius = kwargs.get('par_radius', 4.5e-05)
            column.particle_type_000.par_coreradius = 0.0
            column.particle_type_000.par_porosity = 0.75
            column.particle_type_000.film_diffusion = kwargs.get(
                'film_diffusion', [6.9e-06, 6.9e-06, 6.9e-06, 6.9e-06])
            
            if particle_type == "GENERAL_RATE_PARTICLE":
                column.particle_type_000.has_pore_diffusion = 1
                column.particle_type_000.pore_diffusion = kwargs.get(
                    'pore_diffusion', [7.00e-10, 6.07e-11, 6.07e-11, 6.07e-11])
                column.particle_type_000.has_surface_diffusion = 0
                column.particle_type_000.surface_diffusion = [0.,0.,0.,0.]
                column.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
                if kwargs['spatial_method_particle'] > 0:
                    column.particle_type_000.discretization.SPATIAL_METHOD = 'DG'
                    column.particle_type_000.discretization.PAR_POLYDEG = kwargs['spatial_method_particle']
                    column.particle_type_000.discretization.PAR_NELEM = parNElem
                else:
                    column.particle_type_000.discretization.SPATIAL_METHOD = 'FV'
                    column.particle_type_000.discretization.NCELLS = parNElem
                    column.particle_type_000.discretization.FV_BOUNDARY_ORDER = 2
                    
        else:
            column.particle_type_000.has_film_diffusion = 0
        
        column.particle_type_000.nbound = [1, 1, 1, 1]
        column.particle_type_000.init_cp = [50.0, 0.0, 0.0, 0.0]
        column.particle_type_000.init_cs = [1200.0, 0.0, 0.0, 0.0]
        
        column.particle_type_000.adsorption_model = 'STERIC_MASS_ACTION'
        column.particle_type_000.adsorption = get_binding_configuration(kwargs.get('is_kinetic', True))
    
    model.input.model.unit_000 = column

    # Flow sheet
    flowRate = 0.000575 * column.inlet_cross_section_area * 0.37
    model.input.model.connections.connections_include_ports = 1
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = [
        1.0, 0.0, -1.0, -1.0, -1.0, -1.0, flowRate
    ]
    model.input.model.connections.switch_000.section = 0
    
    # Inlet / Feed unit
    model.input.model.unit_001.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_001.ncomp = 4
    model.input.model.unit_001.sec_000.const_coeff = [50.0, 1.0, 1.0, 1.0]
    model.input.model.unit_001.sec_000.cube_coeff = [0.0, 0.0, 0.0, 0.0]
    model.input.model.unit_001.sec_000.lin_coeff = [0.0, 0.0, 0.0, 0.0]
    model.input.model.unit_001.sec_000.quad_coeff = [0.0, 0.0, 0.0, 0.0]
    model.input.model.unit_001.sec_001.const_coeff = [50.0, 0.0, 0.0, 0.0]
    model.input.model.unit_001.sec_001.cube_coeff = [0.0, 0.0, 0.0, 0.0]
    model.input.model.unit_001.sec_001.lin_coeff = [0.0, 0.0, 0.0, 0.0]
    model.input.model.unit_001.sec_001.quad_coeff = [0.0, 0.0, 0.0, 0.0]
    model.input.model.unit_001.sec_002.const_coeff = [100.0, 0.0, 0.0, 0.0]
    model.input.model.unit_001.sec_002.cube_coeff = [0.0, 0.0, 0.0, 0.0]
    model.input.model.unit_001.sec_002.lin_coeff = [0.2, 0.0, 0.0, 0.0]
    model.input.model.unit_001.sec_002.quad_coeff = [0.0, 0.0, 0.0, 0.0]
    model.input.model.unit_001.UNIT_TYPE = 'INLET'

    # Global system solver
    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08

    # Time integration / solver
    model.input.solver.consistent_init_mode_sens = 3
    model.input.solver.nthreads = 1
    model.input.solver.sections.nsec = 3
    model.input.solver.sections.section_continuity = [ 0, 0 ]
    model.input.solver.sections.section_times = [ 0.0, 10.0, 90.0, 1500.0]
    # Tolerances: idas_abstol / idas_reltol / idas_algtol are the preferred
    # names. idas_reftol is kept as an alias for the absolute tolerance so that
    # older calls keep working. Unless given explicitly, the relative and
    # algebraic tolerances follow the absolute one by two orders of magnitude.
    abstol = kwargs.get('idas_abstol', kwargs.get('idas_reftol', 1e-12))
    model.input.solver.time_integrator.ABSTOL = abstol
    model.input.solver.time_integrator.ALGTOL = kwargs.get('idas_algtol', abstol * 100)
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 1000000
    model.input.solver.time_integrator.RELTOL = kwargs.get('idas_reltol', abstol * 100)
    
    # Return data
    model.input.solver.user_solution_times = np.linspace(0, 1500, 1501)
    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].unit_000.write_coordinates = kwargs.get('write_solution_bulk', False) or kwargs.get('write_solution_particle', False)
    model.input['return'].unit_000.write_sens_bulk = 0
    model.input['return'].unit_000.write_sens_last = 0
    model.input['return'].unit_000.write_sens_outlet = 1
    model.input['return'].unit_000.write_solution_bulk =  kwargs.get('write_solution_bulk', False)
    model.input['return'].unit_000.write_solution_flux = 0
    model.input['return'].unit_000.write_solution_inlet = 0
    model.input['return'].unit_000.write_solution_outlet = 1
    model.input['return'].unit_000.write_solution_particle = kwargs.get('write_solution_particle', False)
    model.input['return'].unit_000.write_solution_solid = kwargs.get('write_solution_particle', False)
    model.input['return'].write_solution_times = 1
    
    return model
