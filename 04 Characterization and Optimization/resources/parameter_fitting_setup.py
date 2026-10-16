"""Prepared non-binding Acetone pulse process for the Parameter Fitting lesson."""

from CADETProcess.processModel import (
    ComponentSystem,
    FlowSheet,
    Inlet,
    LumpedRateModelWithoutPores,
    Outlet,
    Process,
)


def acetone_process():
    """Build the 1 s pulse experiment at 1 mL/min with initial transport estimates."""
    components = ComponentSystem(["Acetone"])
    inlet = Inlet(components, "inlet")
    inlet.flow_rate = 1e-6 / 60
    inlet.c = [0]

    column = LumpedRateModelWithoutPores(components, "column")
    column.length = 0.014
    column.diameter = 0.01
    column.total_porosity = 0.5
    column.axial_dispersion = 1e-6

    outlet = Outlet(components, "outlet")
    flow_sheet = FlowSheet(components)
    flow_sheet.add_unit(inlet)
    flow_sheet.add_unit(column)
    flow_sheet.add_unit(outlet)
    flow_sheet.add_connection(inlet, column)
    flow_sheet.add_connection(column, outlet)

    process = Process(flow_sheet, "acetone")
    process.cycle_time = 180
    process.add_event("inject_on", "flow_sheet.inlet.c", [1], 0)
    process.add_event("inject_off", "flow_sheet.inlet.c", [0], 1)
    return process


def acetone_estimation_configuration():
    """Configure the estimation GUI for the same experiment as the Python model."""
    from cadetgui.widgets.instrument.configuration import ConfigurationWidget

    process = acetone_process()
    config = ConfigurationWidget(workspace_header=False, starting_values=False)
    config.components = ["Acetone"]
    config.select_models(
        template="Pulse Feed (Single Component)",
        column="Lumped Rate Model Without Pores (LRM)",
        binding="None",
    )
    column = process.flow_sheet.column
    config.column_form.set_values(
        {
            "length": column.length,
            "diameter": column.diameter,
            "total_porosity": column.total_porosity,
            "axial_dispersion": float(column.axial_dispersion[0]),
        }
    )
    config._model_form.set_values(
        {
            "component": "Acetone",
            "concentration": 1,
            "flow_rate": 1e-6 / 60,
            "pulse_duration": 1,
            "cycle_time": 180,
        }
    )
    return config
