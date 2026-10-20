"""Model and visual checkpoints for the batch-elution optimization lesson."""

import matplotlib.pyplot as plt
import numpy as np

from CADETProcess.processModel import (
    ComponentSystem, Langmuir, Inlet, LumpedRateModelWithoutPores,
    Outlet, FlowSheet, Process,
)


def batch_elution_process():
    """Supply the model; leave optimization variables and metrics to the lesson."""
    component_system = ComponentSystem(["A", "B"])
    binding = Langmuir(component_system, "langmuir")
    binding.is_kinetic = False
    binding.adsorption_rate = [0.02, 0.03]
    binding.desorption_rate = [1, 1]
    binding.capacity = [100, 100]
    feed = Inlet(component_system, "feed")
    feed.c = [10, 10]
    eluent = Inlet(component_system, "eluent")
    eluent.c = [0, 0]
    column = LumpedRateModelWithoutPores(component_system, "column")
    column.binding_model = binding
    column.length = 0.6
    column.diameter = 0.024
    column.total_porosity = 0.7
    column.axial_dispersion = 4.7e-7
    column.discretization.ncol = 30
    outlet = Outlet(component_system, "outlet")
    sheet = FlowSheet(component_system, "flow_sheet")
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


def process_setup_view(process):
    """Render the model with the same unit symbols as the earlier GUI lessons."""
    import ipywidgets as widgets
    from cadetgui.widgets.diagrams import render_flow_sheet_svg
    from cadetgui.widgets.ui.theme import style_tag

    view = widgets.VBox([
        widgets.HTML(style_tag()),
        widgets.HTML(render_flow_sheet_svg(process.flow_sheet)),
        widgets.HTML(
            "<b>Feed:</b> A and B, each at 10 mol/m³. "
            "<b>Eluent:</b> protein-free liquid. "
            "Both pass through the same <b>60 cm × 2.4 cm column</b> "
            "at <b>60 mL/min</b>; the outlet supplies the product pools."
        ),
    ])
    view.add_class("cadetgui-panel")
    return view


def _draw_cycle(ax, settings, labels):
    """Draw feed followed by elution, with settings supplied in seconds."""
    colors = ["#2864a0", "#c6dce9"]
    longest_cycle = max(cycle for cycle, feed in settings) / 60
    for row, (cycle, feed) in enumerate(settings):
        ax.barh(row, feed / 60, color=colors[0], height=0.45)
        ax.barh(row, (cycle - feed) / 60, left=feed / 60,
                color=colors[1], height=0.45)
        ax.text(cycle / 60 + 0.1, row,
                f"Feed {feed:.0f} s; cycle {cycle / 60:.2f} min", va="center")
    ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel="Cycle time / min",
           xlim=(0, longest_cycle * 1.5), ylim=(-0.55, len(labels) - 0.45))
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.plot([], [], color=colors[0], linewidth=8, label="Feed")
    ax.plot([], [], color=colors[1], linewidth=8, label="Elution")
    ax.legend(loc="upper right", frameon=False, ncols=2)


def plot_feed_cycle(process):
    """Make the two decision variables visible before building the optimization."""
    fig, ax = plt.subplots(figsize=(9, 2.2))
    _draw_cycle(ax, [(process.cycle_time, process.feed_duration.time)], ["Baseline"])
    ax.set_title("Feed first, then elute until the next cycle")
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_process_comparison(baseline_settings, selected_settings,
                            baseline_fractionation, selected_fractionation,
                            required_purity=0.95):
    """Compare actual operating settings and collected-product performance."""
    fig = plt.figure(figsize=(10, 6), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.8])
    schedule = fig.add_subplot(grid[0, :])
    _draw_cycle(schedule, [baseline_settings, selected_settings],
                ["Baseline", "Selected"])
    schedule.set_title("Operating settings")

    before = baseline_fractionation.performance
    after = selected_fractionation.performance
    names = selected_fractionation.component_system.names
    positions = np.arange(len(names))
    panels = [
        ("Productivity", "mol / (L stationary phase · h)",
         np.asarray(before.productivity) * 3.6,
         np.asarray(after.productivity) * 3.6),
        ("Pool purity", "%", np.asarray(before.purity) * 100,
         np.asarray(after.purity) * 100),
        ("Recovery", "% of fed component", np.asarray(before.recovery) * 100,
         np.asarray(after.recovery) * 100),
    ]
    axes = []
    for index, (title, unit, initial, selected) in enumerate(panels):
        ax = fig.add_subplot(grid[1, index])
        axes.append(ax)
        ax.bar(positions - 0.18, initial, width=0.36, color="#9baabb",
               label="Baseline")
        ax.bar(positions + 0.18, selected, width=0.36, color="#16805d",
               label="Selected")
        ax.set(title=title, ylabel=unit, xticks=positions, xticklabels=names)
        ax.spines[["top", "right"]].set_visible(False)
        if title == "Pool purity":
            ax.axhline(100 * required_purity, color="#b04a43", linestyle="--",
                       label=f"{100 * required_purity:.0f}% required")
            ax.set_ylim(0, 115)
        elif title == "Recovery":
            ax.set_ylim(0, 108)
        else:
            ax.set_ylim(0, max(np.max(initial), np.max(selected)) * 1.25)
        for offset, values in [(-0.18, initial), (0.18, selected)]:
            for position, value in zip(positions, values):
                ax.annotate(f"{value:.4f}" if title == "Productivity" else f"{value:.1f}",
                            (position + offset, value), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=9)
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].legend(handles=[axes[1].lines[0]], loc="upper left", frameon=False)
    plt.show()
    return fig, (schedule, *axes)
