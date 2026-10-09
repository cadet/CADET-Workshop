"""Generate the synthetic tracer breakthrough used in the first-steps exercises.

The column is the one from the lesson, except for the total porosity (0.35), which the
exercise asks to determine. Gaussian noise is added to mimic a detector signal.
"""
from pathlib import Path

import numpy as np
from CADETProcess.processModel import (
    ComponentSystem,
    FlowSheet,
    Inlet,
    LumpedRateModelWithoutPores,
    Outlet,
    Process,
)
from CADETProcess.simulator import Cadet

component_system = ComponentSystem(["Tracer"])

inlet = Inlet(component_system, "inlet")
inlet.c = 1
inlet.flow_rate = 1e-7

column = LumpedRateModelWithoutPores(component_system, "column")
column.length = 0.1
column.diameter = 0.02
column.total_porosity = 0.35
column.axial_dispersion = 1e-7

outlet = Outlet(component_system, "outlet")

flow_sheet = FlowSheet(component_system, "flow_sheet")
for unit in (inlet, column, outlet):
    flow_sheet.add_unit(unit)
flow_sheet.add_connection(inlet, column)
flow_sheet.add_connection(column, outlet)

process = Process(flow_sheet, "process")
process.cycle_time = 600

simulator = Cadet()
simulator.time_resolution = 2
solution = simulator.simulate(process).solution.outlet.outlet

rng = np.random.default_rng(42)
signal = solution.solution[:, 0] + rng.normal(0, 0.01, solution.time.size)

path = Path(__file__).parent / "measured_breakthrough.csv"
np.savetxt(
    path,
    np.column_stack([solution.time, signal]),
    delimiter=",",
    header="time / s,concentration / mol m^-3",
    comments="",
    fmt="%.6g",
)
