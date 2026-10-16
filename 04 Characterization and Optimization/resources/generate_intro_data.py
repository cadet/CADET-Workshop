"""Regenerate the synthetic column-only tracer measurement for Essentials."""

from pathlib import Path

import numpy as np
from CADETProcess.simulator import Cadet

from optimization_intro import porosity_demo_configuration


if __name__ == "__main__":
    process = porosity_demo_configuration().process
    process.flow_sheet.column.total_porosity = 0.35
    simulator = Cadet()
    simulator.time_resolution = 0.5
    result = simulator.simulate(process)
    signal = result.solution.column.outlet
    np.savetxt(
        Path(__file__).with_name("column_porosity_pulse.csv"),
        np.column_stack([signal.time, signal.solution[:, 0]]),
        delimiter=",",
        header="time / s,concentration / mol m^-3",
        comments="",
        fmt="%.10g",
    )
