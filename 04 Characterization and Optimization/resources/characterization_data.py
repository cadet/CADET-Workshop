"""Load the synthetic RTD experiments used throughout the workshop."""

from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

import matplotlib.pyplot as plt
import numpy as np
from CADETProcess.simulator import Cadet
from IPython.display import Markdown, display


def load_rtd_measurements():
    """Return the bypass pulse and column step experiments, with time in seconds."""
    fundamentals = Path(__file__).resolve().parents[2] / "02 CADET Fundamentals"
    pulse = np.loadtxt(
        fundamentals
        / "02.4 Residence Time Distributions/resources/measured_system_pulse.csv",
        delimiter=",",
        skiprows=1,
    )
    breakthrough = np.loadtxt(
        fundamentals / "resources/first_steps/measured_breakthrough.csv",
        delimiter=",",
        skiprows=1,
    )
    return pulse, breakthrough


def pulse_delay(pulse):
    """Subtract the 5 s injection's centroid from the detector pulse centroid."""
    t, c = pulse.T
    return np.trapezoid(t * c, t) / np.trapezoid(c, t) - 2.5


def breakthrough_delay(breakthrough):
    """Integrate the normalized step response; inlet concentration is 1 mol/m³."""
    t, c = breakthrough.T
    return np.trapezoid(1 - c, t)


def plot_rtd_measurements(pulse, breakthrough):
    """Show the different experiments without conflating pulse and step delays."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(pulse[:, 0] / 60, pulse[:, 1])
    axes[0].set(title="Column bypass: 5 s pulse", ylabel="Tracer / mol/m³")
    axes[1].plot(breakthrough[:, 0] / 60, breakthrough[:, 1])
    axes[1].set(title="Column connected: continuous feed")
    axes[0].set_xlabel("Time / min")
    axes[1].set_xlabel("Time / min")
    fig.tight_layout()
    plt.show()
    return fig, axes


def _reference_instrument(with_column=True):
    """Reuse the synthetic-data model builder without regenerating measurement files."""
    generator = (
        Path(__file__).resolve().parents[2]
        / "02 CADET Fundamentals/resources/first_steps/generate_breakthrough_data.py"
    )
    spec = spec_from_file_location("workshop_reference_instrument", generator)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_instrument(with_column=with_column)


def characterization_setup_widget():
    """Show the actual experiment flow sheets with a single experiment selector."""
    import ipywidgets as widgets
    from cadetgui.widgets.diagrams import render_flow_sheet_svg
    from cadetgui.widgets.ui.theme import style_tag

    connected = _reference_instrument()
    connected.flow_sheet.column.total_porosity = 0.4  # Initial model, before fitting.
    connected.flow_sheet.inlet.c = 1
    connected.cycle_time = 600
    bypass = _reference_instrument(with_column=False)
    bypass.cycle_time = 300
    bypass.add_event("pulse_on", "flow_sheet.inlet.c", 1, 0)
    bypass.add_event("pulse_off", "flow_sheet.inlet.c", 0, 5)
    selector = widgets.ToggleButtons(
        options=[("Column connected", True), ("Column bypassed", False)],
        value=True,
        description="Experiment:",
    )
    diagram = widgets.HTML()
    caption = widgets.HTML()

    def redraw(_change=None):
        process = connected if selector.value else bypass
        diagram.value = render_flow_sheet_svg(process.flow_sheet)
        caption.value = (
            "<b>Continuous tracer feed:</b> the detector signal includes mixer, tubing "
            "and column transport. Column: <b>10 cm × 2 cm</b>; porosity is to be fitted."
            if selector.value
            else "<b>5 s tracer pulse:</b> the detector signal includes mixer and tubing "
            "transport. The column is bypassed."
        )

    selector.observe(redraw, names="value")
    redraw()
    view = widgets.VBox([widgets.HTML(style_tag()), selector, diagram, caption])
    view.add_class("cadetgui-panel")
    return view


def plot_delay_contributions(system_delay, total_delay):
    """Show the instrument contribution held fixed while fitting column porosity."""
    column_delay = total_delay - system_delay
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.barh(0, system_delay, color="#2864a0", height=0.5)
    ax.barh(0, column_delay, left=system_delay, color="#16805d", height=0.5)
    ax.text(
        system_delay / 2,
        0,
        f"Instrument\n{system_delay:.0f} s",
        ha="center",
        va="center",
        color="white",
    )
    ax.text(
        system_delay + column_delay / 2,
        0,
        f"Column\n{column_delay:.0f} s",
        ha="center",
        va="center",
        color="white",
    )
    ax.text(total_delay + 3, 0, f"Total\n{total_delay:.0f} s", va="center")
    ax.set(
        xlabel="Contribution to mean delay / s",
        yticks=[],
        xlim=(0, total_delay * 1.23),
        ylim=(-0.6, 0.6),
        title="Hold the instrument contribution fixed; fit the column contribution",
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_porosity_transfer(apparent_porosity, fitted_porosity):
    """Make the porosity bias from assigning instrument delay to the column visible."""
    fig, ax = plt.subplots(figsize=(8, 2.8))
    positions = np.array([0, 1])
    values = [apparent_porosity, fitted_porosity]
    ax.barh(positions, values, color=["#c17e38", "#16805d"], height=0.5)
    for position, value in zip(positions, values):
        ax.text(value + 0.01, position, f"{value:.3f}", va="center")
    ax.axvline(fitted_porosity, color="#16805d", linestyle=":", alpha=0.7)
    ax.set(
        yticks=positions,
        yticklabels=["Assign all delay to column", "Include instrument contribution"],
        xlabel="Column porosity / −",
        xlim=(0, 0.6),
        title="The bypass result prevents an inflated column porosity",
    )
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.show()
    return fig, ax


def check_characterized_column(accepted, breakthrough):
    """Compare two dispersion choices at the accepted porosity and system volume."""

    simulator = Cadet()
    simulator.time_resolution = 0.5
    checks = {}
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(breakthrough[:, 0] / 60, breakthrough[:, 1], "k", label="Reference")
    for dispersion, style in [(1e-7, "--"), (1e-6, "-.")]:
        process = _reference_instrument()
        process.flow_sheet.inlet.c = 1
        process.cycle_time = float(breakthrough[-1, 0])
        # Retain the known 1:2 split between mixer and tubing volumes.
        process.flow_sheet.mixer.init_liquid_volume = accepted["system_volume"] / 3
        tubing_volume = accepted["system_volume"] * 2 / 3
        process.flow_sheet.tubing.diameter = np.sqrt(4 * tubing_volume / np.pi)
        process.flow_sheet.column.total_porosity = accepted["porosity"]
        process.flow_sheet.column.axial_dispersion = dispersion
        result = simulator.simulate(process)
        signal = result.solution.outlet.outlet
        concentration = np.interp(
            breakthrough[:, 0], signal.time, signal.solution[:, 0]
        )
        checks[dispersion] = {
            "mean_delay": float(np.trapezoid(1 - concentration, breakthrough[:, 0])),
            "rmse": float(np.sqrt(np.mean((concentration - breakthrough[:, 1]) ** 2))),
        }
        ax.plot(
            signal.time / 60,
            signal.solution[:, 0],
            style,
            label=f"D = {dispersion:.0e} m²/s",
        )
    ax.set(xlabel="Time / min", ylabel="Tracer / mol/m³", xlim=(0, 5))
    ax.legend()
    fig.tight_layout()
    plt.show()

    reference_delay = breakthrough_delay(breakthrough)
    display(
        Markdown(
            "| Curve | Mean delay / s | Curve RMSE / mol/m³ |\n"
            "|---|---:|---:|\n"
            f"| Reference | {reference_delay:.2f} | 0 |\n"
            f"| D = 1e-7 m²/s | {checks[1e-7]['mean_delay']:.2f} | {checks[1e-7]['rmse']:.4f} |\n"
            f"| D = 1e-6 m²/s | {checks[1e-6]['mean_delay']:.2f} | {checks[1e-6]['rmse']:.4f} |"
        )
    )
    print(
        f"Parameter check: porosity {accepted['porosity']:.3f} is between 0 and 1: "
        f"{0 < accepted['porosity'] < 1}. "
        f"System volume: {accepted['system_volume'] * 1e6:.1f} mL (bypass result)."
    )
    return checks
