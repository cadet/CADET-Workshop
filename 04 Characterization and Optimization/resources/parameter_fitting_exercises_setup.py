"""Reference processes for the Parameter Fitting exercises.

The exercises practise connecting measurements to these models; the models themselves
use only components, unit operations, flow sheets and events from earlier lessons.
"""

from CADETProcess.processModel import (
    ComponentSystem, FlowSheet, Inlet, LumpedRateModelWithoutPores, Outlet, Process,
    StericMassAction, TubularReactor,
)

FLOW_RATE = 1.67e-8  # m³/s, about 1 mL/min


def _column(component_system):
    """Create the 1.4 cm × 1 cm column with its initial transport estimates."""
    column = LumpedRateModelWithoutPores(component_system, name="column")
    column.length = 0.014
    column.diameter = 0.01
    column.total_porosity = 0.5
    column.axial_dispersion = 5.75e-7
    return column


def _pipes(component_system):
    """Create the tubing before the column (pipe5), to UV (pipe6) and to conductivity (pipe7)."""
    pipes = []
    for name, length in (("pipe5", 0.1), ("pipe6", 0.02), ("pipe7", 0.1)):
        pipe = TubularReactor(component_system, name=name)
        pipe.length = length
        pipe.diameter = 0.001
        pipe.axial_dispersion = 6e-6
        pipe.discretization.ncol = 50
        pipes.append(pipe)
    return pipes


def _flow_sheet(component_system, column):
    """Connect Inlet → pipe5 → column → pipe6 (UV) → pipe7 (conductivity) → Outlet."""
    pipe5, pipe6, pipe7 = _pipes(component_system)
    inlet = Inlet(component_system, name="inlet")
    inlet.flow_rate = FLOW_RATE
    outlet = Outlet(component_system, name="outlet")

    flow_sheet = FlowSheet(component_system)
    for unit in (inlet, pipe5, column, pipe6, pipe7, outlet):
        flow_sheet.add_unit(unit)
    flow_sheet.add_connection(inlet, pipe5)
    flow_sheet.add_connection(pipe5, column)
    flow_sheet.add_connection(column, pipe6)
    flow_sheet.add_connection(pipe6, pipe7)
    flow_sheet.add_connection(pipe7, outlet)
    return flow_sheet


def tracer_process():
    """Return the Acetone pulse process: 1 mM for 1 s, observed for 180 s."""
    component_system = ComponentSystem(["Acetone"])
    flow_sheet = _flow_sheet(component_system, _column(component_system))

    process = Process(flow_sheet, "tracer")
    process.cycle_time = 180
    process.add_event("load", "flow_sheet.inlet.c", 1, 0)
    process.add_event("wash", "flow_sheet.inlet.c", 0, 1)
    return process


def _binding_flow_sheet(component_system):
    """Return the flow sheet with the initial SMA estimates, equilibrated at 50 mM salt."""
    binding_model = StericMassAction(component_system)
    binding_model.is_kinetic = True
    binding_model.adsorption_rate = [0, 2.3e-5, 5.59e-1, 3.5e-3]
    binding_model.desorption_rate = [0, 1, 1, 1]
    binding_model.characteristic_charge = [1, 6.9, 2.3, 5.8]  # Salt valence, then proteins
    binding_model.steric_factor = [0, 10, 10.6, 11.83]
    binding_model.capacity = 1.2e3

    column = _column(component_system)
    column.binding_model = binding_model
    column.c = [50, 0, 0, 0]
    column.q = [1.2e3, 0, 0, 0]

    flow_sheet = _flow_sheet(component_system, column)
    for name in ("pipe5", "pipe6", "pipe7"):
        getattr(flow_sheet, name).c = [50, 0, 0, 0]
    return flow_sheet


def breakthrough_process(component_system):
    """Return the breakthrough: 50 mM salt and 1 mM of each protein for 60 min."""
    process = Process(_binding_flow_sheet(component_system), "breakthrough")
    process.cycle_time = 60 * 60
    process.add_event("load", "flow_sheet.inlet.c", [50, 1, 1, 1], 0)
    return process


def gradient_process(gradient_length_in_cv, component_system):
    """Return a gradient elution: 10 s load, wash to 90 s, 80 → 500 mM salt, 60 s per CV."""
    gradient_duration = gradient_length_in_cv * 60
    process = Process(_binding_flow_sheet(component_system), f"gradient {gradient_length_in_cv}")
    process.cycle_time = gradient_duration + 600

    slope = (500 - 80) / gradient_duration
    process.add_event("load", "flow_sheet.inlet.c", [50, 1, 1, 1], 0)
    process.add_event("wash", "flow_sheet.inlet.c", [50, 0, 0, 0], 10)
    process.add_event(
        "grad1_start", "flow_sheet.inlet.c",
        [[80, slope, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 90,
    )
    process.add_event("grad1_end", "flow_sheet.inlet.c", [500, 0, 0, 0], 90 + gradient_duration)
    return process
