"""Fresh version of the two-protein batch-elution case from Fractionation."""

from CADETProcess.processModel import (
    ComponentSystem, Langmuir, Inlet, LumpedRateModelWithoutPores,
    Outlet, FlowSheet, Process,
)


def batch_elution_process():
    """Supply the model; leave optimization variables and metrics to the lesson."""
    components = ComponentSystem(["A", "B"])
    binding = Langmuir(components, "langmuir")
    binding.is_kinetic = False
    binding.adsorption_rate = [0.02, 0.03]
    binding.desorption_rate = [1, 1]
    binding.capacity = [100, 100]
    feed = Inlet(components, "feed")
    feed.c = [10, 10]
    eluent = Inlet(components, "eluent")
    eluent.c = [0, 0]
    column = LumpedRateModelWithoutPores(components, "column")
    column.binding_model = binding
    column.length = 0.6
    column.diameter = 0.024
    column.total_porosity = 0.7
    column.axial_dispersion = 4.7e-7
    column.discretization.ncol = 30
    outlet = Outlet(components, "outlet")
    sheet = FlowSheet(components, "flow_sheet")
    sheet.add_unit(feed, feed_inlet=True)
    sheet.add_unit(eluent, eluent_inlet=True)
    sheet.add_unit(column)
    sheet.add_unit(outlet, product_outlet=True)
    sheet.add_connection(feed, column)
    sheet.add_connection(eluent, column)
    sheet.add_connection(column, outlet)
    process = Process(sheet, "batch_elution_design")
    process.cycle_time = 600
    process.add_event("feed_on", "flow_sheet.feed.flow_rate", 1e-6, 0)
    process.add_event("feed_off", "flow_sheet.feed.flow_rate", 0, 60)
    process.add_event("eluent_off", "flow_sheet.eluent.flow_rate", 0, 0)
    process.add_event("eluent_on", "flow_sheet.eluent.flow_rate", 1e-6, 60)
    process.add_duration("feed_duration", 60)
    process.add_event_dependency("feed_off", ["feed_on", "feed_duration"], [1, 1])
    process.add_event_dependency("eluent_on", ["feed_off"])
    return process
