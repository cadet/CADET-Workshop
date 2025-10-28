import matplotlib.pyplot as plt
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Inlet, TubularReactor, LumpedRateModelWithoutPores, Outlet
from CADETProcess.processModel import Process
from CADETProcess.processModel import StericMassAction
from CADETProcess.simulator import Cadet

component_system = ComponentSystem(1)

volumetric_flow_rate = 1.67e-8

column = LumpedRateModelWithoutPores(component_system, name='column')
column.length = 0.014
column.total_porosity = 0.37
column.diameter = 0.01
column.axial_dispersion = 5.75e-8

pipe1 = TubularReactor(component_system, name="pipe1")
pipe1.length = 0.05
pipe1.diameter = 0.002
pipe1.axial_dispersion = 6e-6

pipe2 = TubularReactor(component_system, name="pipe2")
pipe2.length = 0.05
pipe2.diameter = 0.002
pipe2.axial_dispersion = 6e-6

inlet = Inlet(component_system, "inlet")
inlet.flow_rate = volumetric_flow_rate

outlet = Outlet(component_system, "outlet")

flow_sheet = FlowSheet(component_system, "flow_sheet")

flow_sheet.add_unit(column)
flow_sheet.add_unit(pipe1)
flow_sheet.add_unit(pipe2)
flow_sheet.add_unit(inlet)
flow_sheet.add_unit(outlet)
flow_sheet.add_connection(inlet, pipe1)
flow_sheet.add_connection(pipe1, column)
flow_sheet.add_connection(column, pipe2)
flow_sheet.add_connection(pipe2, outlet)

process = Process(flow_sheet, "process")
process.cycle_time = 100

process.add_event("load", "flow_sheet.inlet.c", [1], time=0)
process.add_event("wash", "flow_sheet.inlet.c", [0], time=1)

simulator = Cadet()
simulator.timeout = 100
simulator.time_resolution = 1
sim_results_complete = simulator.simulate(process)

fig, ax = plt.subplots(1)
sim_results_complete.solution.outlet.outlet.plot(ax=ax)

flow_sheet = FlowSheet(component_system, "flow_sheet")

flow_sheet.add_unit(column)
flow_sheet.add_unit(inlet)
flow_sheet.add_unit(outlet)
flow_sheet.add_connection(inlet, column)
flow_sheet.add_connection(column, outlet)

process = Process(flow_sheet, "process")
process.cycle_time = 100

process.add_event("load", "flow_sheet.inlet.c", [1], time=0)
process.add_event("wash", "flow_sheet.inlet.c", [0], time=1)

simulator = Cadet()
simulator.timeout = 100
simulator.time_resolution = 1
sim_results = simulator.simulate(process)

sim_results.solution.column.outlet.plot(ax=ax)

from CADETProcess.reference import ReferenceIO

reference = ReferenceIO(
    "uv_1",
    sim_results_complete.time_complete,
    sim_results_complete.solution.outlet.outlet.solution_original
)
_ = reference.plot()

from CADETProcess.comparison import Comparator

comparator = Comparator()
comparator.add_reference(reference)

comparator.add_difference_metric('Shape', reference, 'outlet.outlet')
comparator.evaluate(sim_results)

_ = comparator.plot_comparison(sim_results)

from CADETProcess.optimization import OptimizationProblem
optimization_problem = OptimizationProblem('System periphery')

optimization_problem.add_evaluation_object(process)


optimization_problem.add_variable(
    name='Column porosity', parameter_path='flow_sheet.column.total_porosity',
    lb=0.1, ub=0.8,
    transform='auto'
)

optimization_problem.add_variable(
    name='Column axial dispersion', parameter_path='flow_sheet.column.axial_dispersion',
    lb=1e-8, ub=1e-6,
    transform='auto'
)


optimization_problem.add_evaluator(simulator)

optimization_problem.add_objective(
    comparator,
    n_objectives=comparator.n_metrics,
    requires=[simulator]
)

def callback(simulation_results, individual, evaluation_object, callbacks_dir='./'):
    comparator.plot_comparison(
        simulation_results,
        file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_comparison.png',
        show=False
    )


optimization_problem.add_callback(callback, requires=[simulator])

from CADETProcess.optimization import U_NSGA3
optimizer = U_NSGA3()
optimizer.n_cores = 8
optimizer.pop_size = 64
optimizer.n_max_gen = 18

optimization_results = optimizer.optimize(
    optimization_problem,
    use_checkpoint=False
)
