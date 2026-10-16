"""Supplied tracer setup and delay calculation for the optimization lesson."""

import numpy as np
from CADETProcess.processModel import (
    ComponentSystem, Inlet, Cstr, TubularReactor, Outlet, FlowSheet, Process,
)


def tracer_delay_process(name, pulse_length=5):
    """Build an independent copy of the mixer-and-tubing pulse experiment."""
    components = ComponentSystem(["Tracer"])
    inlet = Inlet(components, "inlet")
    inlet.flow_rate = 1e-7  # 6 mL/min
    inlet.c = 0
    mixer = Cstr(components, "mixer")
    mixer.init_liquid_volume = 1e-6  # 1 mL
    mixer.const_solid_volume = 0
    mixer.flow_rate = inlet.flow_rate
    tubing = TubularReactor(components, "tubing")
    tubing.length = 1
    tubing.diameter = np.sqrt(4 * 2e-6 / np.pi)  # 2 mL at 1 m length
    tubing.axial_dispersion = 1e-8
    tubing.discretization.ncol = 20
    outlet = Outlet(components, "outlet")
    sheet = FlowSheet(components, "flow_sheet")
    sheet.add_unit(inlet)
    sheet.add_unit(mixer)
    sheet.add_unit(tubing)
    sheet.add_unit(outlet, product_outlet=True)
    sheet.add_connection(inlet, mixer)
    sheet.add_connection(mixer, tubing)
    sheet.add_connection(tubing, outlet)
    process = Process(sheet, name)
    process.cycle_time = 300
    process.add_event("pulse_on", "flow_sheet.inlet.c", 1, 0)
    process.add_event("pulse_off", "flow_sheet.inlet.c", 0, pulse_length)
    return process


def transport_delay(results):
    """Return outlet minus inlet pulse mean time, in seconds.

    For the rectangular inlet pulse, its exact centroid is half its duration.
    Using the event time avoids a sampling error at the concentration step.
    """
    signal = results.solution.outlet.outlet
    t = signal.time
    c = signal.solution[:, 0]
    outlet_mean = np.trapezoid(t * c, t) / np.trapezoid(c, t)
    inlet_mean = results.process.pulse_off.time / 2
    return float(outlet_mean - inlet_mean)
