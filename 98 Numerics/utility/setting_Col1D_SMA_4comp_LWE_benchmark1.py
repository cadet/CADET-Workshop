# -*- coding: utf-8 -*-
"""

This script defines steric mass action load-wash-elute case studies used for
numerical benchmarks in https://doi.org/10.1016/j.compchemeng.2023.108340

"""

import numpy as np
from addict import Dict

def get_model(
        spatial_method_bulk,
        particle_type='GENERAL_RATE_PARTICLE',
        refinement=1,
        **kwargs):
    
    axNElem = 8 * kwargs.get('axRefinement', refinement)
    parNElem = kwargs.get('parZ', 1)
    
    model = Dict()
    
    model.input.model.nunits = 2
    
    # Flow sheet
    model.input.model.connections.connections_include_ports = 1
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = [
        1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 1.0
    ]
    model.input.model.connections.switch_000.section = 0
    
    # Column unit
    column = Dict()
    column.UNIT_TYPE = 'COLUMN_MODEL_1D'
    column.col_length = 0.014
    column.col_radius = 0.01
    column.col_porosity = 0.37
    column.total_porosity = 0.8425
    column.npartype = 0 if particle_type is None else 1
    
    column.ncomp = 4
    column.init_c = [50.0, 0.0, 0.0, 0.0]
    column.col_dispersion = 5.75e-08
    column.velocity = 0.000575

    # Spatial discretization of interstitial / bulk volume
    if spatial_method_bulk > 0:
        column.discretization.SPATIAL_METHOD = 'DG'
        column.discretization.EXACT_INTEGRATION = kwargs.get('exact_integration', 0)
        column.discretization.POLYDEG = spatial_method_bulk
        column.discretization.NELEM = axNElem
    else:
        column.discretization.SPATIAL_METHOD = 'FV'
        column.discretization.NCOL = axNElem
        column.discretization.RECONSTRUCTION = 'WENO'
        column.discretization.weno.BOUNDARY_MODEL = 0
        column.discretization.weno.WENO_EPS = 1e-10
        column.discretization.weno.WENO_ORDER = 2
        column.discretization.GS_TYPE = 1
        column.discretization.MAX_KRYLOV = 0
        column.discretization.MAX_RESTARTS = 10
        column.discretization.SCHUR_SAFETY = 1.0e-8
    column.discretization.USE_ANALYTIC_JACOBIAN = 1
    
    if particle_type is not None:
        
        if particle_type in ['HOMOGENEOUS_PARTICLE', 'GENERAL_RATE_PARTICLE']:
            
            column.particle_type_000.has_film_diffusion = 1
            column.particle_type_000.par_geom = 'SPHERE'
            column.particle_type_000.par_radius = 4.5e-05
            column.particle_type_000.par_coreradius = 0.0
            column.particle_type_000.par_porosity = 0.75
            column.particle_type_000.film_diffusion = [6.9e-06, 6.9e-06, 6.9e-06, 6.9e-06]
            
            if particle_type == "GENERAL_RATE_PARTICLE":
                column.particle_type_000.pore_diffusion = [7.00e-10, 6.07e-11, 6.07e-11, 6.07e-11]
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
        column.particle_type_000.adsorption = {
                'is_kinetic': 1,
                'sma_ka': [ 0.0, 35.5, 1.59, 7.7 ],
                'sma_kd': [ 0.0, 1000.0, 1000.0, 1000.0],
                'sma_lambda': 1200.0,
                'sma_nu': [ 0.0, 4.7, 5.29, 3.7 ],
                'sma_sigma': [ 0.0, 11.83, 10.6, 10.0 ]
                }
    
    model.input.model.unit_000 = column
    
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
    model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_abstol', 1e-12)
    model.input.solver.time_integrator.ALGTOL = kwargs.get('idas_algtol', 1e-10)
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 10000
    model.input.solver.time_integrator.RELTOL = kwargs.get('idas_reltol', 1e-10)
    
    # Return data
    model.input.solver.user_solution_times = np.linspace(0, 1500, 1501)
    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].unit_000.write_coordinates = kwargs.get('return_bulk', False) or kwargs.get('return_particle', False)
    model.input['return'].unit_000.write_sens_bulk = 0
    model.input['return'].unit_000.write_sens_last = 0
    model.input['return'].unit_000.write_sens_outlet = 1
    model.input['return'].unit_000.write_solution_bulk =  kwargs.get('return_bulk', False)
    model.input['return'].unit_000.write_solution_flux = 0
    model.input['return'].unit_000.write_solution_inlet = 0
    model.input['return'].unit_000.write_solution_outlet = 1
    model.input['return'].unit_000.write_solution_particle = kwargs.get('return_particle', False)
    model.input['return'].unit_000.write_solution_solid = kwargs.get('return_particle', False)
    model.input['return'].write_solution_times = 1
    
    return model


#%% sensitivities


def add_sensitivity_LRM_SMA_4comp_benchmark1(model, sensName):

    sensDepIdx = {
        'COL_DISPERSION': {'sens_comp': np.int64(-1)},
        'TOTAL_POROSITY': { },
        'SMA_KA': {'sens_comp': np.int64(1), 'sens_boundphase': np.int64(0)}
    }    

    if sensName not in sensDepIdx:
        raise Exception(f'Sensitivity dependencies for {sensName} unknown, please implement!')

    if 'sensitivity' in model['input']:
        model['input']['sensitivity']['NSENS'] += 1
    else:
        model['input']['sensitivity'] = {'NSENS': np.int64(1)}
        model['input']['sensitivity']['sens_method'] = np.bytes_(b'ad1')

    sensIdx = str(model['input']['sensitivity']['NSENS'] - 1).zfill(3)
    
    model['input']['sensitivity'][f'param_{sensIdx}'] = {}
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_name'] = str(sensName)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_unit'] = np.int64(0)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_partype'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_reaction'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_section'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_boundphase'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_comp'] = np.int64(-1)
    
    if sensName in sensDepIdx:
        param = model['input']['sensitivity'][f'param_{sensIdx}']
        for key, value in {**sensDepIdx[sensName]}.items():
            model['input']['sensitivity'][f'param_{sensIdx}'][key] = value

    return model


def get_LRM_sensbenchmark1(spatial_method_bulk):
    
    model = get_model(spatial_method_bulk, particle_type=	'EQUILIBRIUM_PARTICLE')
    model['input'].pop('sensitivity', None)
    model = add_sensitivity_LRM_SMA_4comp_benchmark1(model, 'COL_DISPERSION')
    model = add_sensitivity_LRM_SMA_4comp_benchmark1(model, 'TOTAL_POROSITY')
    model = add_sensitivity_LRM_SMA_4comp_benchmark1(model, 'SMA_KA')
    
    return model

def add_sensitivity_LRMP_SMA_4comp_benchmark1(model, sensName):

    sensDepIdx = {
        'COL_DISPERSION': {'sens_comp': np.int64(-1)},
        'FILM_DIFFUSION': {'sens_comp': np.int64(0)},
        'SMA_KA': {'sens_comp': np.int64(1), 'sens_boundphase': np.int64(0)}
    }    

    if sensName not in sensDepIdx:
        raise Exception(f'Sensitivity dependencies for {sensName} unknown, please implement!')

    if 'sensitivity' in model['input']:
        model['input']['sensitivity']['NSENS'] += 1
    else:
        model['input']['sensitivity'] = {'NSENS': np.int64(1)}
        model['input']['sensitivity']['sens_method'] = np.bytes_(b'ad1')

    sensIdx = str(model['input']['sensitivity']['NSENS'] - 1).zfill(3)
    
    model['input']['sensitivity'][f'param_{sensIdx}'] = {}
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_name'] = str(sensName)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_unit'] = np.int64(0)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_partype'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_reaction'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_section'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_boundphase'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_comp'] = np.int64(-1)
    
    if sensName in sensDepIdx:
        param = model['input']['sensitivity'][f'param_{sensIdx}']
        for key, value in {**sensDepIdx[sensName]}.items():
            model['input']['sensitivity'][f'param_{sensIdx}'][key] = value

    return model


def get_LRMP_sensbenchmark1(spatial_method_bulk):
    
    model = get_model(spatial_method_bulk, particle_type=	'HOMOGENEOUS_PARTICLE')
    model['input'].pop('sensitivity', None)
    model = add_sensitivity_LRMP_SMA_4comp_benchmark1(model, 'COL_DISPERSION')
    model = add_sensitivity_LRMP_SMA_4comp_benchmark1(model, 'FILM_DIFFUSION')
    model = add_sensitivity_LRMP_SMA_4comp_benchmark1(model, 'SMA_KA')
    
    return model

def add_sensitivity_GRM_SMA_4comp_benchmark1(model, sensName):

    sensDepIdx = {
        'COL_DISPERSION': {'sens_comp': np.int64(-1)},
        'FILM_DIFFUSION': {'sens_comp': np.int64(1)},
        'PAR_DIFFUSION': {'sens_comp': np.int64(1)},
        'PAR_SURFDIFFUSION': {'sens_comp': np.int64(1), 'sens_boundphase': np.int64(0)},
        'PAR_RADIUS': {},
        'SMA_KA': {'sens_comp': np.int64(1), 'sens_boundphase': np.int64(0)}
    }    

    if sensName not in sensDepIdx:
        raise Exception(f'Sensitivity dependencies for {sensName} unknown, please implement!')

    if 'sensitivity' in model['input']:
        model['input']['sensitivity']['NSENS'] += 1
    else:
        model['input']['sensitivity'] = {'NSENS': np.int64(1)}
        model['input']['sensitivity']['sens_method'] = np.bytes_(b'ad1')

    sensIdx = str(model['input']['sensitivity']['NSENS'] - 1).zfill(3)
    
    model['input']['sensitivity'][f'param_{sensIdx}'] = {}
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_name'] = str(sensName)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_unit'] = np.int64(0)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_partype'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_reaction'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_section'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_boundphase'] = np.int64(-1)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_comp'] = np.int64(-1)
    
    if sensName in sensDepIdx:
        param = model['input']['sensitivity'][f'param_{sensIdx}']
        for key, value in {**sensDepIdx[sensName]}.items():
            model['input']['sensitivity'][f'param_{sensIdx}'][key] = value

    return model


def get_GRM_sensbenchmark1(spatial_method_bulk, spatial_method_particle):
    
    model = get_model(spatial_method_bulk,
                      particle_type='GENERAL_RATE_PARTICLE',
                      spatial_method_particle=spatial_method_particle)
    model['input'].pop('sensitivity', None)
    model = add_sensitivity_GRM_SMA_4comp_benchmark1(model, 'COL_DISPERSION')
    model = add_sensitivity_GRM_SMA_4comp_benchmark1(model, 'PAR_DIFFUSION')
    model = add_sensitivity_GRM_SMA_4comp_benchmark1(model, 'SMA_KA')
    
    return model

def get_GRM_sensbenchmark2(spatial_method_bulk, spatial_method_particle):
    
    model = get_model(spatial_method_bulk,
                      particle_type='GENERAL_RATE_PARTICLE',
                      spatial_method_particle=spatial_method_particle)
    model['input'].pop('sensitivity', None)
    model = add_sensitivity_GRM_SMA_4comp_benchmark1(model, 'FILM_DIFFUSION')
    model = add_sensitivity_GRM_SMA_4comp_benchmark1(model, 'PAR_SURFDIFFUSION')
    model = add_sensitivity_GRM_SMA_4comp_benchmark1(model, 'PAR_RADIUS')
    
    return model