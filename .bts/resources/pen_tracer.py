from pathlib import Path
from datetime import datetime

import numpy
from cadet import Cadet
from matplotlib import pyplot as plt


def create_sim_pen():
    sim = Cadet()
    root = sim.root
    root.input.model.connections.connections_include_dynamic_flow = 1
    root.input.model.connections.nswitches = 2
    root.input.model.connections.switch_000.connections = numpy.array(
        [0.00000000e+00, 1.00000000e+00, -1.00000000e+00, -1.00000000e+00,
         8.3333e-9, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         1.00000000e+00, 2.00000000e+00, -1.00000000e+00, -1.00000000e+00,
         8.3333e-9, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
    root.input.model.connections.switch_000.section = 0
    root.input.model.connections.switch_001.connections = numpy.array(
        [0.00000000e+00, 1.00000000e+00, -1.00000000e+00, -1.00000000e+00,
         8.3333e-9, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         1.00000000e+00, 2.00000000e+00, -1.00000000e+00, -1.00000000e+00,
         8.3333e-9, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
    root.input.model.connections.switch_001.section = 1
    root.input.model.nunits = 3
    root.input.model.solver.gs_type = 1
    root.input.model.solver.linear_solution_mode = 0
    root.input.model.solver.max_krylov = 0
    root.input.model.solver.max_restarts = 10
    root.input.model.solver.schur_safety = 1e-08
    root.input.model.unit_000.discretization.nbound = numpy.array([0])
    root.input.model.unit_000.inlet_type = b'PIECEWISE_CUBIC_POLY'
    root.input.model.unit_000.ncomp = 1
    root.input.model.unit_000.sec_000.const_coeff = numpy.array([0.,])
    root.input.model.unit_000.sec_000.cube_coeff = numpy.array([0.,])
    root.input.model.unit_000.sec_000.lin_coeff = numpy.array([0.,])
    root.input.model.unit_000.sec_000.quad_coeff = numpy.array([0.,])
    root.input.model.unit_000.sec_001.const_coeff = numpy.array([1,])
    root.input.model.unit_000.sec_001.cube_coeff = numpy.array([0.,])
    root.input.model.unit_000.sec_001.lin_coeff = numpy.array([0,])
    root.input.model.unit_000.sec_001.quad_coeff = numpy.array([0.,])
    root.input.model.unit_000.sec_002.const_coeff = numpy.array([0,])
    root.input.model.unit_000.sec_002.cube_coeff = numpy.array([0.,])
    root.input.model.unit_000.sec_002.lin_coeff = numpy.array([0,])
    root.input.model.unit_000.sec_002.quad_coeff = numpy.array([0.,])
    root.input.model.unit_000.unit_type = b'INLET'

    root.input.model.unit_001.adsorption.adsorption_model = b'LINEAR'
    root.input.model.unit_001.adsorption.is_kinetic = False
    root.input.model.unit_001.adsorption.lin_ka = numpy.array([0.])
    root.input.model.unit_001.adsorption.lin_kd = numpy.array([1.])
    root.input.model.unit_001.adsorption_model = b'LINEAR'
    root.input.model.unit_001.col_dispersion = 5.75e-8

    root.input.model.unit_001.col_length = 0.02
    root.input.model.unit_001.col_porosity = 0.37
    root.input.model.unit_001.cross_section_area = numpy.pi * (0.004 ** 2)
    root.input.model.unit_001.film_diffusion = numpy.array([1e-03])
    root.input.model.unit_001.init_c = numpy.array([0, ])
    root.input.model.unit_001.init_cp = numpy.array([0, ])
    root.input.model.unit_001.init_q = numpy.array([0., ])
    root.input.model.unit_001.ncomp = 1
    root.input.model.unit_001.par_diffusion = numpy.array([4.00e-11, ])
    root.input.model.unit_001.par_porosity = 0.5
    root.input.model.unit_001.par_radius = 4.5e-05
    root.input.model.unit_001.par_surfdiffusion = numpy.array([0.])
    root.input.model.unit_001.par_surfdiffusion_multiplex = 0
    root.input.model.unit_001.unit_type = b'GENERAL_RATE_MODEL'
    root.input.model.unit_001.velocity = 1

    root.input.model.unit_001.discretization.consistency_solver.init_damping = 0.01
    root.input.model.unit_001.discretization.consistency_solver.max_iterations = 50
    root.input.model.unit_001.discretization.consistency_solver.min_damping = 0.0001
    root.input.model.unit_001.discretization.consistency_solver.solver_name = b'LEVMAR'
    root.input.model.unit_001.discretization.consistency_solver.subsolvers = b'LEVMAR'
    root.input.model.unit_001.discretization.fix_zero_surface_diffusion = False
    root.input.model.unit_001.discretization.gs_type = True
    root.input.model.unit_001.discretization.max_krylov = 0
    root.input.model.unit_001.discretization.max_restarts = 10
    root.input.model.unit_001.discretization.nbound = numpy.array([1])
    root.input.model.unit_001.discretization.ncol = 100
    root.input.model.unit_001.discretization.npar = 4
    root.input.model.unit_001.discretization.par_boundary_order = 2
    root.input.model.unit_001.discretization.par_disc_type = b'EQUIDISTANT_PAR'
    root.input.model.unit_001.discretization.par_geom = b'SPHERE'
    root.input.model.unit_001.discretization.reconstruction = b'WENO'
    root.input.model.unit_001.discretization.schur_safety = 1e-08
    root.input.model.unit_001.discretization.use_analytic_jacobian = True
    root.input.model.unit_001.discretization.weno.boundary_model = 0
    root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    root.input.model.unit_001.discretization.weno.weno_order = 3

    root.input.model.unit_002.discretization.nbound = numpy.array([0])
    root.input.model.unit_002.ncomp = 1
    root.input.model.unit_002.unit_type = b'OUTLET'

    root.input['return'].split_components_data = False
    root.input['return'].split_ports_data = False
    root.input['return'].unit_000.write_coordinates = False
    root.input['return'].unit_000.write_sens_inlet = False
    root.input['return'].unit_000.write_sens_outlet = False
    root.input['return'].unit_000.write_sensdot_inlet = False
    root.input['return'].unit_000.write_sensdot_outlet = False
    root.input['return'].unit_000.write_soldot_inlet = False
    root.input['return'].unit_000.write_soldot_outlet = False
    root.input['return'].unit_000.write_solution_inlet = False
    root.input['return'].unit_000.write_solution_last_unit = False
    root.input['return'].unit_000.write_solution_outlet = False
    root.input['return'].unit_001.write_coordinates = False
    root.input['return'].unit_001.write_sens_bulk = False
    root.input['return'].unit_001.write_sens_flux = False
    root.input['return'].unit_001.write_sens_inlet = False
    root.input['return'].unit_001.write_sens_outlet = False
    root.input['return'].unit_001.write_sens_particle = False
    root.input['return'].unit_001.write_sens_solid = False
    root.input['return'].unit_001.write_sensdot_bulk = False
    root.input['return'].unit_001.write_sensdot_flux = False
    root.input['return'].unit_001.write_sensdot_inlet = False
    root.input['return'].unit_001.write_sensdot_outlet = False
    root.input['return'].unit_001.write_sensdot_particle = False
    root.input['return'].unit_001.write_sensdot_solid = False
    root.input['return'].unit_001.write_soldot_bulk = False
    root.input['return'].unit_001.write_soldot_flux = False
    root.input['return'].unit_001.write_soldot_inlet = False
    root.input['return'].unit_001.write_soldot_outlet = False
    root.input['return'].unit_001.write_soldot_particle = False
    root.input['return'].unit_001.write_soldot_solid = False
    root.input['return'].unit_001.write_solution_bulk = True
    root.input['return'].unit_001.write_solution_flux = False
    root.input['return'].unit_001.write_solution_inlet = True
    root.input['return'].unit_001.write_solution_last_unit = False
    root.input['return'].unit_001.write_solution_outlet = True
    root.input['return'].unit_001.write_solution_particle = True
    root.input['return'].unit_001.write_solution_solid = True
    root.input['return'].unit_002.write_coordinates = False
    root.input['return'].unit_002.write_sens_inlet = False
    root.input['return'].unit_002.write_sens_outlet = False
    root.input['return'].unit_002.write_sensdot_inlet = False
    root.input['return'].unit_002.write_sensdot_outlet = False
    root.input['return'].unit_002.write_soldot_inlet = False
    root.input['return'].unit_002.write_soldot_outlet = False
    root.input['return'].unit_002.write_solution_inlet = False
    root.input['return'].unit_002.write_solution_last_unit = False
    root.input['return'].unit_002.write_solution_outlet = False
    root.input['return'].write_sens_last = True
    root.input['return'].write_solution_last = True
    root.input['return'].write_solution_times = True

    root.input.sensitivity.nsens = 0
    root.input.sensitivity.sens_method = b'ad1'
    root.input.solver.consistent_init_mode = 1
    root.input.solver.consistent_init_mode_sens = 1
    root.input.solver.nthreads = 1
    root.input.solver.sections.nsec = 3
    root.input.solver.sections.section_continuity = numpy.array([0])
    root.input.solver.sections.section_times = numpy.array([0., 20, 30., 400.])
    root.input.solver.time_integrator.errortest_sens = False
    root.input.solver.time_integrator.init_step_size = 1e-06
    root.input.solver.time_integrator.max_convtest_fail = 1000000
    root.input.solver.time_integrator.max_errtest_fail = 1000000
    root.input.solver.time_integrator.max_newton_iter = 1000000
    root.input.solver.time_integrator.max_newton_iter_sens = 1000000
    root.input.solver.time_integrator.max_step_size = 0.0
    root.input.solver.time_integrator.max_steps = 1000000
    root.input.solver.time_integrator.abstol = 0.001
    root.input.solver.time_integrator.algtol = 0.1
    root.input.solver.time_integrator.reltol = 0.01
    # root.input.solver.time_integrator.abstol = 0.000001
    # root.input.solver.time_integrator.algtol = 0.0001
    # root.input.solver.time_integrator.reltol = 0.00001
    root.input.solver.time_integrator.reltol_sens = 1e-12
    root.input.solver.user_solution_times = numpy.linspace(0, 300, 201)

    return sim


if __name__ == '__main__':

    start_time = datetime.now()
    sim = create_sim_pen()
    sim.filename = 'sim.h5'
    sim.cadet_path = r"C:/Users/ronal/mambaforge/envs/interactive/bin/cadet-cli.exe"

    sim.root.input.solver.user_solution_times = numpy.linspace(0, sim.root.input.solver.user_solution_times[-1], 1000)
    for i in range(1):
        sim.filename = Path(sim.filename.replace("sim.h5", "sim2.h5"))
        sim.save()
        return_info = sim.run()
        sim.load()
    end_time = datetime.now()
    print(end_time - start_time)

    output = sim.root.output.solution.unit_001.solution_outlet
    output_times = sim.root.output.solution.solution_times

    fig, ax = plt.subplots(1)
    ax_twin = ax.twinx()
    ax.plot(output_times, output[:, 0])
    ax_twin.plot(output_times, output[:, 0])
    plt.show()

    output_protein = output[:, 0]
    output_protein = output_protein + (numpy.random.random(output_protein.shape) - 0.5) * 0.03 * output_protein.max()
    output_combined = numpy.stack([output_times, output_protein])

    numpy.save("pen_tracer.npy", output_combined)
