"""Generate reproducible tracer data for the workshop characterization story.

One instrument: a 1 mL mixer and 2 mL tubing, optionally followed by the
10 x 2 cm column. The packing porosity (0.35) is unknown to participants.
All signals are synthetic and noise-free so numerical moments are transparent.
Run from the workshop environment; output locations do not depend on cwd.
"""
from pathlib import Path

import numpy as np
from CADETProcess.processModel import (
    ComponentSystem, FlowSheet, Inlet, Cstr, TubularReactor,
    LumpedRateModelWithoutPores, Outlet, Process,
)
from CADETProcess.simulator import Cadet

FLOW_RATE = 1e-7  # 6 mL/min
MIXER_VOLUME = 1e-6  # 1 mL
TUBING_VOLUME = 2e-6  # 2 mL
PULSE_DURATION = 5.0


def build_instrument(with_column):
    component_system = ComponentSystem(["Tracer"])
    inlet = Inlet(component_system, "inlet")
    inlet.flow_rate = FLOW_RATE
    mixer = Cstr(component_system, "mixer")
    mixer.init_liquid_volume = MIXER_VOLUME
    mixer.const_solid_volume = 0
    tubing = TubularReactor(component_system, "tubing")
    tubing.length = 1
    tubing.diameter = np.sqrt(4 * TUBING_VOLUME / np.pi)
    tubing.axial_dispersion = 1e-8
    units = [inlet, mixer, tubing]
    if with_column:
        column = LumpedRateModelWithoutPores(component_system, "column")
        column.length = 0.1
        column.diameter = 0.02
        column.total_porosity = 0.35
        column.axial_dispersion = 1e-7
        units.append(column)
    units.append(Outlet(component_system, "outlet"))
    sheet = FlowSheet(component_system, "flow_sheet")
    for unit in units:
        sheet.add_unit(unit)
    for origin, destination in zip(units, units[1:]):
        sheet.add_connection(origin, destination)
    return Process(sheet, "column_breakthrough" if with_column else "system_pulse")


def save_signal(path, solution):
    np.savetxt(
        path, np.column_stack([solution.time, solution.solution[:, 0]]),
        delimiter=",", header="time / s,concentration / mol m^-3",
        comments="", fmt="%.10g",
    )


def main():
    resources = Path(__file__).resolve().parent
    system_resources = resources.parents[1] / "02.4 Residence Time Distributions" / "resources"
    simulator = Cadet()
    simulator.time_resolution = 0.5

    breakthrough = build_instrument(with_column=True)
    breakthrough.flow_sheet.inlet.c = 1
    breakthrough.cycle_time = 600
    save_signal(resources / "measured_breakthrough.csv",
                simulator.simulate(breakthrough).solution.outlet.outlet)

    pulse = build_instrument(with_column=False)
    pulse.cycle_time = 300
    pulse.add_event("pulse_on", "flow_sheet.inlet.c", 1, 0)
    pulse.add_event("pulse_off", "flow_sheet.inlet.c", 0, PULSE_DURATION)
    save_signal(system_resources / "measured_system_pulse.csv",
                simulator.simulate(pulse).solution.outlet.outlet)


if __name__ == "__main__":
    main()
