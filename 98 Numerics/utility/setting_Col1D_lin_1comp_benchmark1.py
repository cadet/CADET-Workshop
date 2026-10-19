"""

This script defines linear adsorption case studies used for numerical
benchmarks in https://doi.org/10.1016/j.compchemeng.2023.108340

"""

from addict import Dict
import numpy as np


def get_column_geometry_configuration(geometry: str):

    # for all geometries:
    # velocity = 0.000575 m/s
    # col_porosity = 0.37
    # bed length is 0.014 m
    # flow rate Q = 6.e-05

    axial_flow_cross_section_area = 6.e-05 / 0.000575 / 0.37

    if geometry == 'AXIAL_FLOW_CYLINDER':
        return {
            'geometry': geometry,
            # A = Q / (velocity * col_porosity)
            'cross_section_area': axial_flow_cross_section_area,
            'bed_length': 0.014,
        }
    elif geometry == 'RADIAL_FLOW_CYLINDER_SHELL':
        cylinder_height = 0.25
        # The inlet (outer) cross section area is the cylinder surface
        # A_outer = 2 pi r_outer h, i.e. r_outer = A_outer / (2 pi h).
        radius_outer = axial_flow_cross_section_area / (2.0 * np.pi * cylinder_height)
        radius_inner = radius_outer - 0.014
        return {
            'geometry': geometry,
            'cylinder_height': cylinder_height,
            'cross_section_area_outer': axial_flow_cross_section_area,
            # optional field, CADET-Core cross-checks it against the inner
            # radius derived from cross_section_area_outer, cylinder_height
            # and bed_length
            'cross_section_area_inner': 2.0 * np.pi * radius_inner * cylinder_height,
            'bed_length': 0.014,
        }
    elif geometry == 'AXIAL_FLOW_FRUSTUM':
        return {
            'geometry': geometry,
            'cross_section_area_large_end': axial_flow_cross_section_area,
            'cross_section_area_small_end': axial_flow_cross_section_area * 0.75,
            'bed_length': 0.014,
        }
    elif geometry == 'SMOOTHLY_VARYING':
        # The cross section area is prescribed at every DG node and thus depends on
        # the spatial discretization, so CROSS_SECTIONAL_AREA_AT_NODES has to be added
        # by the caller, see setting_Col1D_pureTransport_1comp_benchmark1.get_model.
        return {
            'geometry': geometry,
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
    if column_geometry not in ['AXIAL_FLOW_CYLINDER', 'RADIAL_FLOW_CYLINDER_SHELL', 'AXIAL_FLOW_FRUSTUM']:
        raise ValueError(f"Unknown column geometry: {column_geometry}")
    column.unit_type = "COLUMN_MODEL_1D"
    column.update(get_column_geometry_configuration(column_geometry))
    column.forward_flow = 1
    
    column.ncomp = 1
    column.col_dispersion = 5.75e-08
    column.col_porosity = 0.37
    if particle_type == 'EQUILIBRIUM_PARTICLE':
        column.total_porosity = 0.37 + (1.0 - 0.37) * 0.75
    column.npartype = 1 if particle_type is not None else 0
    column.par_type_volfrac = 1
    
    if spatial_method_bulk > 0:
        column.discretization.SPATIAL_METHOD = "DG"
        # collocation DG is only implemented for the axial cylinder; the other
        # geometries use the (exact integration) variable cross section DG
        column.discretization.USE_COLLOCATION_DG = kwargs.get(
            'USE_COLLOCATION_DG', 1 if column_geometry == 'AXIAL_FLOW_CYLINDER' else 0)
        column.discretization.POLYDEG = spatial_method_bulk
        column.discretization.NELEM = axNElem
    elif spatial_method_bulk == 0:
        column.discretization.SPATIAL_METHOD = "FV"
        column.discretization.NCOL = axNElem
    column.discretization.RECONSTRUCTION = 'WENO'
    column.discretization.weno.BOUNDARY_MODEL = 0
    column.discretization.weno.WENO_EPS = 1e-10
    column.discretization.weno.WENO_ORDER = kwargs.get('weno_order', 3)
    column.discretization.GS_TYPE = 1
    column.discretization.MAX_KRYLOV = 0
    column.discretization.MAX_RESTARTS = 10
    column.discretization.SCHUR_SAFETY = 1.0e-8
    if spatial_method_bulk >= 0:
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
            
            if 'spatial_method_particle' not in kwargs:
                raise Exception("keyword argument spatial_method_particle needs to be specified for general rate particles")
            
            if kwargs.get('surface_diffusion', 0.0) > 0.0:
                column.particle_type_000.has_surface_diffusion = 1
                column.particle_type_000.surface_diffusion = kwargs.get('surface_diffusion', 0.0)
            else:
                column.particle_type_000.has_surface_diffusion = 0
                
            # The particle discretization is independent of the bulk discretization,
            # so it must only set fields of the particle type, never of the column.
            if kwargs['spatial_method_particle'] > 0:
                column.particle_type_000.discretization.SPATIAL_METHOD = "DG"
                column.particle_type_000.discretization.PAR_POLYDEG = kwargs['spatial_method_particle']
                column.particle_type_000.discretization.PAR_NELEM = parNElem
            elif kwargs['spatial_method_particle'] == 0:
                column.particle_type_000.discretization.SPATIAL_METHOD = "FV"
                column.particle_type_000.discretization.NCELLS = parNElem
                column.particle_type_000.discretization.FV_BOUNDARY_ORDER = 2
            if kwargs['spatial_method_particle'] >= 0:  
                column.particle_type_000.discretization.PAR_DISC_TYPE = 'EQUIDISTANT_PAR'
            
        else:
            column.particle_type_000.has_pore_diffusion = 0
    elif particle_type == "EQUILIBRIUM_PARTICLE":
        column.particle_type_000.has_film_diffusion = 0
    if particle_type is not None:
        column.particle_type_000.nbound = [ 1 ]
        column.particle_type_000.adsorption.is_kinetic = kwargs.get('is_kinetic', 1)
        column.particle_type_000.adsorption.lin_ka = [ 3.55 ]
        column.particle_type_000.adsorption.lin_kd = [ 0.1 ]
        column.particle_type_000.adsorption_model = 'LINEAR'
        if not particle_type == "EQUILIBRIUM_PARTICLE":
            column.particle_type_000.init_cp = [ 0.0 ]
        column.particle_type_000.init_cs = [ 0.0 ]
    
    model.input.model.unit_001 = column
    
    #%% time integration parameters
    if spatial_method_bulk >= 0:
        # non-linear solver
        model.input.model.solver.gs_type = 1
        model.input.model.solver.max_krylov = 0
        model.input.model.solver.max_restarts = 10
        model.input.model.solver.schur_safety = 1e-08
        # time integration / solver specifics
        model.input.solver.consistent_init_mode = 1
        model.input.solver.consistent_init_mode_sens = 3
        model.input.solver.nthreads = 1
        model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_reftol', 1e-12)
        model.input.solver.time_integrator.ALGTOL = kwargs['idas_reftol'] * 100 if 'idas_reftol' in kwargs else 1e-10
        model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
        model.input.solver.time_integrator.MAX_STEPS = 10000
        model.input.solver.time_integrator.RELTOL = kwargs['idas_reftol'] * 100 if 'idas_reftol' in kwargs else 1e-10
    
    model.input.solver.sections.nsec = 2
    model.input.solver.sections.section_continuity = [ 0 ]
    model.input.solver.sections.section_times = [ 0.0, 10.0, 1500.0 ]
    model.input.solver.user_solution_times = np.linspace(0.0, 1500.0, 1500*4 + 1)
    if 'user_solution_times_unit_state' in kwargs:
        model.input.solver.user_solution_times_unit_state = kwargs['user_solution_times_unit_state']

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
    model.input['return'].unit_000.write_solution_inlet = 0
    model.input['return'].unit_000.write_solution_outlet = 0
    model.input['return'].unit_001.write_coordinates = kwargs.get('write_solution_bulk', 0) or kwargs.get('write_solution_solid', 0) or kwargs.get('write_solution_particle', 0)
    model.input['return'].unit_001.write_sens_bulk = 0
    model.input['return'].unit_001.write_sens_last = 0
    model.input['return'].unit_001.write_sens_outlet = 1
    model.input['return'].unit_001.write_solution_inlet = 0
    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_001.write_solution_bulk = kwargs.get('write_solution_bulk', 0)
    model.input['return'].unit_001.write_solution_solid = kwargs.get('write_solution_solid', 0)
    model.input['return'].unit_001.write_solution_particle = kwargs.get('write_solution_particle', 0)
    model.input['return'].unit_002.write_solution_inlet = 0
    model.input['return'].unit_002.write_solution_outlet = 0
    
    return model
