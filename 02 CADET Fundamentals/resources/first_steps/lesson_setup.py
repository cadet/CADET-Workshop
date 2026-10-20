"""Shared GUI lesson setup and solution checkpoints.

Notebook calls that populate completed examples retain their solution tags.
The Python teaching examples stay in the notebooks.
"""

from CADETProcess.processModel import ComponentSystem, FlowSheet, Process


def empty_model(name):
    """Create empty objects for participants to configure with widgets."""
    component_system = ComponentSystem()
    units = {}
    flow_sheet = FlowSheet(component_system, "flow_sheet")
    process = Process(flow_sheet, name)
    return component_system, units, flow_sheet, process


def create_tracer_units(component_system, units):
    """Create the three tracer units without setting experiment parameters."""
    from CADETProcess.processModel import Inlet, LumpedRateModelWithoutPores, Outlet

    units.update(
        inlet=Inlet(component_system, "inlet"),
        column=LumpedRateModelWithoutPores(component_system, "column"),
        outlet=Outlet(component_system, "outlet"),
    )


def set_tracer_parameters(units):
    """Apply the feed and column settings at the parameter-editing step."""
    units["inlet"].c = 1
    units["inlet"].flow_rate = 1e-7
    column = units["column"]
    column.length = 0.1
    column.diameter = 0.02
    column.total_porosity = 0.4
    column.axial_dispersion = 1e-7


def connect_tracer_units(units, flow_sheet):
    """Add and connect the units at the flow-sheet step."""
    for unit in units.values():
        flow_sheet.add_unit(unit)
    flow_sheet.add_connection(units["inlet"], units["column"])
    flow_sheet.add_connection(units["column"], units["outlet"])


def tracer_exercise_model(name="tracer"):
    """Create a fresh supplied tracer column for the First Model exercises."""
    component_system, units, flow_sheet, process = empty_model(name)
    component_system.add_component("Tracer")
    create_tracer_units(component_system, units)
    set_tracer_parameters(units)
    connect_tracer_units(units, flow_sheet)
    process.cycle_time = 1200
    return component_system, units, flow_sheet, process


def unit_creation_widget(component_system, units):
    """Keep creation-only Show code separate from later parameter assignments."""
    from cadetgui import UnitOperationsWidget

    class UnitCreationWidget(UnitOperationsWidget):
        def code(self):
            classes = sorted({type(unit).__name__ for unit in self.units.values()})
            if not classes:
                return "# No unit operations yet."
            lines = [f"from CADETProcess.processModel import {', '.join(classes)}", ""]
            lines.extend(
                f'{name} = {type(unit).__name__}(component_system, "{name}")'
                for name, unit in self.units.items()
            )
            return "\n".join(lines)

    return UnitCreationWidget(component_system, units, edit=False)


def populate_inlet_model(component_system, units, flow_sheet):
    """Restore the Salt/Protein inlet-to-outlet GUI checkpoint in place."""
    # The finished model, as built in the widgets (empty in the teaching version).
    from CADETProcess.processModel import Inlet, Outlet

    component_system.add_component("Salt")
    component_system.add_component("Protein")

    inlet = Inlet(component_system, "inlet")
    inlet.c = [0, 0]
    inlet.flow_rate = 1e-6

    outlet = Outlet(component_system, "outlet")

    units.update(inlet=inlet, outlet=outlet)

    flow_sheet.add_unit(inlet)
    flow_sheet.add_unit(outlet)
    flow_sheet.add_connection(inlet, outlet)


def instrument_widgets():
    """Create the linked instrument, configuration and solution panels."""
    from cadetgui.widgets.instrument import InstrumentWidget, ConfigurationWidget, SolutionWidget

    # Only the column is active by default.
    instrument = InstrumentWidget()
    configuration = ConfigurationWidget(instrument=instrument, workspace_header=False)
    solution = SolutionWidget()
    solution.bind_to_config(configuration)
    return instrument, configuration, solution


def populate_instrument(configuration):
    """Restore the tracer Breakthrough recipe in the instrument configuration."""
    # Supplied on both branches: prefilled lesson widget and exercise starter.
    from dataclasses import replace
    state = replace(
        configuration.snapshot(), components=["Tracer"], template_key="Breakthrough",
        column_values={"length": 0.1, "diameter": 0.02, "total_porosity": 0.4,
                       "axial_dispersion": 1e-7},
        binding_values={"is_kinetic": False, "adsorption_rate": [0], "desorption_rate": [1]},
        model_values={"c_sample": [1], "flow_rate": 1e-7, "cycle_time": 600},
    )
    # Restore the saved-recipe state through the widget's configuration loader.
    configuration._apply_state("Tracer checkpoint", state)


def lwe_gui_model(flow_sheet):
    """Create the empty 180 s process for the inlet-profile editor."""
    process = Process(flow_sheet, "lwe_gui")
    process.cycle_time = 180
    return process


def populate_lwe_events(process):
    """Restore the four inlet events demonstrated in the GUI."""
    # Completed GUI checkpoint; teaching builds these events in the editor.
    process.add_event("load", "flow_sheet.inlet.c", [50, 1], 0)
    process.add_event("wash", "flow_sheet.inlet.c", [50, 0], 60)
    process.add_event("elute", "flow_sheet.inlet.c", [[50, 7.5], [0]], 90)
    process.add_event("final_wash", "flow_sheet.inlet.c", [500, 0], 150)


def inlet_exercise_model():
    """Create the supplied Salt/Protein model with no concentration events."""
    from CADETProcess.processModel import Inlet, Outlet

    component_system = ComponentSystem(["Salt", "Protein"])
    inlet = Inlet(component_system, "inlet")
    inlet.c = [0, 0]
    inlet.flow_rate = 1e-7
    outlet = Outlet(component_system, "outlet")
    flow_sheet = FlowSheet(component_system, "flow_sheet")
    flow_sheet.add_unit(inlet)
    flow_sheet.add_unit(outlet)
    flow_sheet.add_connection(inlet, outlet)
    process = Process(flow_sheet, "inlet_profile")
    process.cycle_time = 300
    return flow_sheet, process


def populate_inlet_exercise_events(process):
    """Restore the completed 300 s method for the inlet-profile exercises."""
    process.add_event("load", "flow_sheet.inlet.c", [20, 1], 0)
    process.add_event("wash", "flow_sheet.inlet.c", [20, 0], 30)
    process.add_event("elute", "flow_sheet.inlet.c", [[20, 1.5], [0, 0]], 90)
    process.add_event("final_wash", "flow_sheet.inlet.c", [200, 0], 210)


def _compact_unit_widget_class():
    """Build the shared unit editor with nonconstant coefficients collapsed."""
    import ipywidgets as W
    from cadetgui import UnitOperationsWidget
    from cadetgui.widgets.ui.layout import collapsible

    class CompactUnitOperationsWidget(UnitOperationsWidget):
        def _show_editor(self):
            previous = self.editor
            super()._show_editor()
            if self.editor is None or self.editor is previous:
                return
            for profile in self.editor.editors:
                children = profile.root.children
                profile.root.children = (
                    children[0],
                    profile.fields[0],
                    collapsible("Polynomial coefficients", W.VBox(profile.fields[1:])),
                    *children[5:],
                )

    return CompactUnitOperationsWidget


def compact_unit_widget(component_system, units, *, create=False):
    """Show inlet start values with other polynomial coefficients collapsed."""
    return _compact_unit_widget_class()(component_system, units, create=create)


def rtd_bypass_widget():
    """Show only the instrument diagram and its column-bypass control."""
    from dataclasses import replace
    from types import SimpleNamespace
    import numpy as np
    import ipywidgets as W
    from cadetgui.widgets.base import Widget
    from cadetgui.widgets.diagrams import SystemDiagram
    from cadetgui.widgets.instrument import InstrumentWidget
    from cadetgui.widgets.ui.theme import style_tag
    from cadetgui.instrument.system import BYPASSABLE_UNITS
    from CADETProcess.processModel import LumpedRateModelWithoutPores

    instrument = InstrumentWidget(hardware_expanded=True)
    instrument.components = ["Tracer"]
    instrument.set_column_and_binding(LumpedRateModelWithoutPores, None)
    instrument.apply_state(replace(
        instrument.snapshot(),
        bypass_units=[
            name for name in BYPASSABLE_UNITS
            if name not in {"mixer", "tubing_pre_column", "column"}
        ],
        unit_values={
            "mixer": {"init_liquid_volume": 1e-6},
            "tubing_pre_column": {
                "length": 1,
                "diameter": np.sqrt(4 * 2e-6 / np.pi),
                "axial_dispersion": 1e-8,
            },
        },
    ))
    diagram = SystemDiagram()
    column_toggle = W.Checkbox(description="Column", value=True, indent=False)

    def redraw():
        # Omit the unused waste branch from this drawing, preserving the model.
        diagram_sheet = SimpleNamespace(units_dict={
            name: unit for name, unit in instrument.flow_sheet.units_dict.items()
            if name != "waste"
        })
        diagram.update(
            diagram_sheet,
            bypassed=instrument.bypass_units(),
            inlets=["buffer_a"],
            carries={"buffer_a": ["Tracer"]},
        )

    def toggle_column(change):
        state = instrument.snapshot()
        bypassed = [name for name in state.bypass_units if name != "column"]
        if not change["new"]:
            bypassed.append("column")
        instrument.apply_state(replace(state, bypass_units=bypassed))
        redraw()

    column_toggle.observe(toggle_column, names="value")
    redraw()
    # Display only the public read-only diagram and the bypass demonstration control.
    view = Widget()
    view.root = W.VBox([W.HTML(style_tag()), diagram.root, column_toggle])
    view.root.add_class("cadetgui-panel")
    return view


def rtd_system_model(name="system", include_mixer=True):
    """Create the lab-scale column-bypass tracer experiment."""
    import numpy as np
    from CADETProcess.processModel import Inlet, Cstr, TubularReactor, Outlet

    components = ComponentSystem(["Tracer"])
    inlet = Inlet(components, "inlet")
    inlet.c = 0
    inlet.flow_rate = 1e-7
    units = {"inlet": inlet}
    if include_mixer:
        mixer = Cstr(components, "mixer")
        mixer.init_liquid_volume = 1e-6
        mixer.const_solid_volume = 0
        mixer.flow_rate = 1e-7
        units["mixer"] = mixer
    tubing = TubularReactor(components, "tubing")
    tubing.length = 1
    tubing.diameter = np.sqrt(4 * 2e-6 / np.pi)
    tubing.axial_dispersion = 1e-8
    units.update(tubing=tubing, outlet=Outlet(components, "outlet"))
    sheet = FlowSheet(components, "flow_sheet")
    for unit in units.values():
        sheet.add_unit(unit)
    ordered = list(units.values())
    for origin, destination in zip(ordered, ordered[1:]):
        sheet.add_connection(origin, destination)
    process = Process(sheet, name)
    process.cycle_time = 300
    process.add_event("pulse_on", "flow_sheet.inlet.c", 1, 0)
    process.add_event("pulse_off", "flow_sheet.inlet.c", 0, 5)
    return components, units, sheet, process


def pulse_centroid(signal):
    """Return the concentration-weighted mean time of the first component, in seconds."""
    import numpy as np

    t = signal.time
    c = signal.solution[:, 0]
    return np.trapezoid(t * c, t) / np.trapezoid(c, t)


def plot_rtd_response(results, *, mark_centroids=False, inlet_centroid=2.5):
    """Show mixer and detector pulses, optionally marking their mean arrival times."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    signals = [results.solution.mixer.outlet, results.solution.outlet.outlet]
    labels = ["Mixer outlet", "Detector"]
    for index, (signal, label) in enumerate(zip(signals, labels)):
        color = f"C{index}"
        ax.plot(signal.time, signal.solution[:, 0], color=color, label=label)
        if mark_centroids:
            centroid = pulse_centroid(signal)
            ax.axvline(
                centroid, color=color, linestyle=":",
                label=f"{label} centroid: {centroid:.1f} s",
            )
    if mark_centroids:
        delay = pulse_centroid(signals[1]) - inlet_centroid
        ax.set_title(f"Extra-column delay: {delay:.1f} s")
    ax.set_xlim(0, 90)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Concentration / mol/m³")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig, ax


def tracer_configuration_widget(process, units):
    """Edit tracer units and export their complete constant-feed process setup."""
    from cadetgui.widgets.diagrams.flow_sheet import flow_sheet_edges

    class TracerConfigurationWidget(_compact_unit_widget_class()):
        def code(self):
            lines = [
                "from CADETProcess.processModel import ComponentSystem, FlowSheet, Process",
                "",
                f"component_system = ComponentSystem({list(process.component_system.names)!r})",
                "",
                super().code(),
                "",
                f'flow_sheet = FlowSheet(component_system, "{process.flow_sheet.name}")',
            ]
            lines.extend(f"flow_sheet.add_unit({unit.name})" for unit in process.flow_sheet.units)
            lines.extend(
                f"flow_sheet.add_connection({origin.name}, {destination.name})"
                for origin, destination in flow_sheet_edges(process.flow_sheet)
            )
            lines.extend([
                "",
                f'process = Process(flow_sheet, "{process.name}")',
                f"process.cycle_time = {process.cycle_time!r}",
            ])
            return "\n".join(lines)

    return TracerConfigurationWidget(process.component_system, units, create=False)


