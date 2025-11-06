"""

This script defines linear adsorption case studies used for numerical
benchmarks in https://doi.org/10.1016/j.compchemeng.2023.108340

"""

from addict import Dict
import numpy as np

def get_model(
        spatial_method_bulk,
        particle_type='GENERAL_RATE_PARTICLE',
        refinement=1,
        **kwargs):
    
    axNElem = 8 * kwargs.get('axRefinement', refinement)
    parNElem = kwargs.get('parZ', 1)
    
    model = Dict()
    
    model.input.model.nunits = 3
    
    # Flow sheet
    model.input.model.connections.connections_include_ports = 0
    model.input.model.connections.nswitches = 1
    model.input.model.connections.switch_000.connections = [
        0.e+00, 1.e+00,-1.e+00,-1.e+00, 6.e-05, 1.e+00, 2.e+00,-1.e+00,-1.e+00, 6.e-05
        ]
    model.input.model.connections.switch_000.section = 0
    
    #%% Column unit
    column = Dict()
    
    column.UNIT_TYPE = 'COLUMN_MODEL_1D'
    column.ncomp = 1
    column.col_dispersion = 5.75e-08
    column.col_length = 0.014
    column.col_porosity = 0.37
    if particle_type == 'EQUILIBRIUM_PARTICLE':
        column.total_porosity = 0.37 + (1.0 - 0.37) * 0.75
    column.velocity = 0.000575
    column.npartype = 1
    column.par_type_volfrac = 1
    
    if spatial_method_bulk > 0:
        column.discretization.SPATIAL_METHOD = "DG"
        column.discretization.EXACT_INTEGRATION = kwargs.get('exact_integration', 0)
        column.discretization.POLYDEG = spatial_method_bulk
        column.discretization.NELEM = axNElem
    else:
        column.discretization.SPATIAL_METHOD = "FV"
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
    column.init_c = [ 0.0 ]
    
    if particle_type in ["HOMOGENEOUS_PARTICLE", "GENERAL_RATE_PARTICLE"]:
        
        column.particle_type_000.film_diffusion = 6.9e-06
        column.particle_type_000.has_film_diffusion = 1
        column.particle_type_000.par_geom = 'SPHERE'
        column.particle_type_000.par_radius = 4.5e-05
        column.particle_type_000.par_coreradius = 0.0
        column.particle_type_000.par_porosity = 0.75
        
        if particle_type == "GENERAL_RATE_PARTICLE":
            
            column.particle_type_000.has_pore_diffusion = 1
            column.particle_type_000.pore_diffusion = 6.07e-11
            
            if 'spatial_method_particle' in kwargs:
                column.particle_type_000.discretization.SPATIAL_METHOD = kwargs['spatial_method_particle']
            else:
                raise Exception(f"keyword argument spatial_method_particle needs to be specified for general rate particles")
            
            if kwargs.get('surface_diffusion', 0.0) > 0.0:
                column.particle_type_000.has_surface_diffusion = 1
                column.particle_type_000.surface_diffusion = kwargs.get('surface_diffusion', 0.0)
            else:
                column.particle_type_000.has_surface_diffusion = 0
                
            if kwargs['spatial_method_particle'] > 0:
                column.discretization.SPATIAL_METHOD = "DG"
                column.particle_type_000.discretization.PAR_POLYDEG = kwargs['spatial_method_particle']
                column.particle_type_000.discretization.PAR_NELEM = parNElem
            else:
                column.discretization.SPATIAL_METHOD = "FV"
                column.particle_type_000.discretization.NCELLS = parNElem
                column.particle_type_000.discretization.FV_BOUNDARY_ORDER = 2
                
            column.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
            
        else:
            column.particle_type_000.has_pore_diffusion = 0
    else:
        column.particle_type_000.has_film_diffusion = 0

    column.particle_type_000.nbound = [ 1 ]
    column.particle_type_000.adsorption.is_kinetic = kwargs.get('is_kinetic', 1)
    column.particle_type_000.adsorption.lin_ka = [ 3.55 ]
    column.particle_type_000.adsorption.lin_kd = [ 0.1 ]
    column.particle_type_000.adsorption_model = 'LINEAR'
    column.particle_type_000.init_cp = [ 0.0 ]
    column.particle_type_000.init_cs = [ 0.0 ]
    
    model.input.model.unit_001 = column
    
    #%% time integration parameters
    # non-linear solver
    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08
    # time integration / solver specifics
    model.input.solver.consistent_init_mode = 1
    model.input.solver.consistent_init_mode_sens = 3
    model.input.solver.nthreads = 1
    model.input.solver.sections.nsec = 2
    model.input.solver.sections.section_continuity = [ 0 ]
    model.input.solver.sections.section_times = [ 0.0, 10.0, 1500.0 ]
    model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_reftol', 1e-12)
    model.input.solver.time_integrator.ALGTOL = kwargs.get('idas_reftol', 1e-10)
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 10000
    model.input.solver.time_integrator.RELTOL = kwargs.get('idas_reftol', 1e-10)
    model.input.solver.user_solution_times = np.linspace(0.0, 1500.0, 1500*4 + 1)
    
    #%% auxiliary units: inlet and outlet
    model.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.input.model.unit_000.ncomp = 1
    model.input.model.unit_000.sec_000.const_coeff = [1.]
    model.input.model.unit_000.sec_000.cube_coeff = [0.]
    model.input.model.unit_000.sec_000.lin_coeff = [0.]
    model.input.model.unit_000.sec_000.quad_coeff = [0.]
    model.input.model.unit_000.sec_001.const_coeff = [0.]
    model.input.model.unit_000.sec_001.cube_coeff = [0.]
    model.input.model.unit_000.sec_001.lin_coeff = [0.]
    model.input.model.unit_000.sec_001.quad_coeff = [0.]
    model.input.model.unit_000.UNIT_TYPE = 'INLET'
    
    model.input.model.unit_002.ncomp = 1
    model.input.model.unit_002.UNIT_TYPE = 'OUTLET'
    
    
    #%% return data
    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].unit_000.write_solution_bulk = 0
    model.input['return'].unit_000.write_solution_inlet = 0
    model.input['return'].unit_000.write_solution_outlet = 0
    model.input['return'].unit_001.write_coordinates = 0
    model.input['return'].unit_001.write_sens_bulk = 0
    model.input['return'].unit_001.write_sens_last = 0
    model.input['return'].unit_001.write_sens_outlet = 1
    model.input['return'].unit_001.write_solution_bulk = 0
    model.input['return'].unit_001.write_solution_inlet = 0
    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_001.write_solution_particle = 0
    model.input['return'].unit_002.write_solution_bulk = 0
    model.input['return'].unit_002.write_solution_inlet = 0
    model.input['return'].unit_002.write_solution_outlet = 0
    
    return model


#%% sensitivities

def add_sensitivity_LRMP_dynLin_1comp_benchmark1(model, sensName):

    sensDepIdx = {
        'COL_DISPERSION': {'sens_comp': np.int64(0)},
        'FILM_DIFFUSION': {'sens_comp': np.int64(0)},
        'LIN_KA': {'sens_comp': np.int64(0), 'sens_boundphase': np.int64(0)}
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
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_unit'] = np.int64(1)
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
    
    model = get_model(spatial_method_bulk, particle_type='HOMOGENEOUS_PARTICLE')
    model['input'].pop('sensitivity', None)
    model = add_sensitivity_LRMP_dynLin_1comp_benchmark1(model, 'COL_DISPERSION')
    model = add_sensitivity_LRMP_dynLin_1comp_benchmark1(model, 'FILM_DIFFUSION')
    model = add_sensitivity_LRMP_dynLin_1comp_benchmark1(model, 'LIN_KA')
    
    return model

def add_sensitivity_GRM_dynLin_1comp_benchmark1(model, sensName):

    sensDepIdx = {
        'COL_DISPERSION': {'sens_comp': np.int64(0)},
        'FILM_DIFFUSION': {'sens_comp': np.int64(0)},
        'PAR_DIFFUSION': {'sens_comp': np.int64(0)},
        'PAR_SURFDIFFUSION': {'sens_comp': np.int64(0), 'sens_boundphase': np.int64(0)},
        'PAR_RADIUS': {},
        'LIN_KA': {'sens_comp': np.int64(0), 'sens_boundphase': np.int64(0)}
    }    

    if sensName not in sensDepIdx:
        raise Exception(f'Sensitivity dependencies for {sensName} unknown, please implement!')

    if 'sensitivity' in model['input']:
        model['input']['sensitivity']['NSENS'] += 1
    else:
        model['input']['sensitivity'] = {'NSENS': np.int64(1)}
        model['input']['sensitivity']['sens_method'] = 'ad1'

    sensIdx = str(model['input']['sensitivity']['NSENS'] - 1).zfill(3)
    
    model['input']['sensitivity'][f'param_{sensIdx}'] = {}
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_name'] = str(sensName)
    model['input']['sensitivity'][f'param_{sensIdx}']['sens_unit'] = np.int64(1)
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
                      spatial_method_particle=spatial_method_particle,
                      particle_type='GENERAL_RATE_PARTICLE')
    model['input'].pop('sensitivity', None)
    model = add_sensitivity_GRM_dynLin_1comp_benchmark1(model, 'COL_DISPERSION')
    model = add_sensitivity_GRM_dynLin_1comp_benchmark1(model, 'PAR_DIFFUSION')
    model = add_sensitivity_GRM_dynLin_1comp_benchmark1(model, 'LIN_KA')
    
    return model

def get_GRM_sensbenchmark2(spatial_method_bulk, spatial_method_particle):
    
    model = get_model(spatial_method_bulk,
                      spatial_method_particle=spatial_method_particle,
                      particle_type='GENERAL_RATE_PARTICLE')
    model['input']['model']['unit_001']['PAR_SURFDIFFUSION'] = 5.0e-11
    model['input'].pop('sensitivity', None)
    model = add_sensitivity_GRM_dynLin_1comp_benchmark1(model, 'FILM_DIFFUSION')
    model = add_sensitivity_GRM_dynLin_1comp_benchmark1(model, 'PAR_SURFDIFFUSION')
    if spatial_method_particle == 0:
        model = add_sensitivity_GRM_dynLin_1comp_benchmark1(model, 'PAR_RADIUS')
    
    return model

from cadet import Cadet

Cadet.cadet_path = r"C:\Users\jmbr\Cadet_testBuild\CADET-Core\out\install\aRELEASE\bin\cadet-cli.exe"

model = Cadet()

model.root = get_model(0, particle_type="GENERAL_RATE_PARTICLE", spatial_method_particle=0)

model.filename = "axialGRM.h5"

model.save()

return_data = model.run_simulation()

if not return_data.return_code == 0:
    print(return_data.error_message)
    raise Exception(f"simulation failed")
model.load_from_file()