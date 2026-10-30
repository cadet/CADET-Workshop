# -*- coding: utf-8 -*-
"""

The same realistic column, with the plumbing around it.

A real system is not just a column. There is tubing between the injector and
the inlet frit, and there is a detector cell after the outlet, and both hold
liquid. Each is well mixed to a good approximation, so each is a stirred tank
with a time constant

    tau = V / Q,

and a small volume means a fast tank. A 10 microlitre detector cell at
2 mL/min has tau = 0.3 s, which is faster than anything in the column except
the grid.

The flowsheet is

    inlet -> column -> detector -> outlet

and the knob is the detector volume. The signal recorded is the detector
outlet, which is what an instrument would actually show.

Realistic volumes for orientation, at the flow rate of this column:

    detector flow cell    5 to 20 uL
    short connecting tube 50 to 150 uL
    mixing chamber        0.5 to 2 mL

"""

import numpy as np
from addict import Dict

import utility.setting_Col1D_langmuir_realistic as column_case

FEED = column_case.LOADING
INJECTION_TIME = column_case.INJECTION_TIME
END_TIME = column_case.END_TIME

DETECTOR_VOLUME = 1.0e-08          # 10 uL


def flow_rate():
    """Volumetric flow rate through the system, in m^3/s."""
    cross_section = np.pi * column_case.COLUMN_RADIUS ** 2

    return column_case.SUPERFICIAL_VELOCITY * cross_section


def residence_time(volume):
    """Time constant of a stirred tank of this volume, in s."""
    return volume / flow_rate()


def mixing_rate(volume):
    """Relaxation rate of a stirred tank of this volume, in 1/s."""
    return flow_rate() / volume


def get_model(detector_volume=DETECTOR_VOLUME, n_elements=25, **kwargs):
    """Build inlet, column, detector and outlet.

    Parameters
    ----------
    detector_volume : float
        Liquid volume of the stirred tank after the column, in m^3.
    n_elements : int
        Axial cells in the column.

    Returns
    -------
    addict.Dict
    """
    n_comp = len(column_case.EQUILIBRIUM_CONSTANTS)
    end_time = kwargs.get('end_time', END_TIME)

    # Start from the column-only setting and rearrange it into a flowsheet, so
    # that the column itself is identical to the one used everywhere else.
    model = column_case.get_model(n_elements=n_elements, end_time=end_time,
                                  **kwargs)

    model.input.model.nunits = 4

    detector = Dict()
    detector.UNIT_TYPE = 'CSTR'
    detector.ncomp = n_comp
    detector.npartype = 0
    detector.init_c = [0.0] * n_comp
    detector.init_liquid_volume = detector_volume
    detector.const_solid_volume = 0.0
    detector.flowrate_filter = [0.0]
    detector.use_analytic_jacobian = 1
    model.input.model.unit_002 = detector

    outlet = Dict()
    outlet.UNIT_TYPE = 'OUTLET'
    outlet.ncomp = n_comp
    model.input.model.unit_003 = outlet

    rate = flow_rate()
    model.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, rate,     # inlet  -> column
        1.0, 2.0, -1.0, -1.0, rate,     # column -> detector
        2.0, 3.0, -1.0, -1.0, rate,     # detector -> outlet
    ]

    model.input['return'].unit_002.write_solution_outlet = 1
    model.input['return'].unit_003.write_solution_outlet = 0

    return model
