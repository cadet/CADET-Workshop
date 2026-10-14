"""Shared, explicit model setup and supplied plots for the LWE case study."""

import matplotlib.pyplot as plt

from CADETProcess.processModel import (
    ComponentSystem, Inlet, Outlet, LumpedRateModelWithoutPores,
    StericMassAction, FlowSheet, Process,
)


def separation_model(name="lwe", with_binding=True):
    """Build a fresh familiar column; leave the process events for the notebook."""
    components = ComponentSystem(["Salt", "Protein A", "Protein B"])
    inlet = Inlet(components, "inlet")
    inlet.flow_rate = 1e-7  # 6 mL/min
    inlet.c = [50, 0, 0]

    column = LumpedRateModelWithoutPores(components, "column")
    column.length = 0.1
    column.diameter = 0.02
    column.total_porosity = 0.4
    column.axial_dispersion = 1e-7
    column.c = [50, 0, 0]

    outlet = Outlet(components, "outlet")
    sheet = FlowSheet(components, "flow_sheet")
    sheet.add_unit(inlet)
    sheet.add_unit(column)
    sheet.add_unit(outlet, product_outlet=True)
    sheet.add_connection(inlet, column)
    sheet.add_connection(column, outlet)
    process = Process(sheet, name)

    if with_binding:
        binding = StericMassAction(components, "SMA")
        binding.is_kinetic = True
        binding.adsorption_rate = [0.0, 0.3, 0.2]  # Salt placeholder, Protein A, Protein B
        binding.desorption_rate = [0.0, 1.5, 1.5]
        binding.characteristic_charge = [1, 7, 5]  # Explicit monovalent salt valence
        binding.steric_factor = [0, 50, 50]
        binding.capacity = 225
        binding.reference_liquid_phase_conc = 300
        binding.reference_solid_phase_conc = 225
        column.binding_model = binding
        column.q = [225, 0, 0]

    return column, process


def _draw_signal(ax, signal, title):
    """Use a separate scale for salt so both protein curves remain readable."""
    ax.plot(signal.time / 60, signal.solution[:, 1], color="C0", label="Protein A")
    ax.plot(signal.time / 60, signal.solution[:, 2], color="C1", linestyle="--", label="Protein B")
    ax.set(title=title, ylabel="Protein / mol/m³")
    ax.grid(alpha=0.2)
    salt_ax = ax.twinx()
    salt_ax.plot(
        signal.time / 60, signal.solution[:, 0], color="gray", linestyle="--", label="Salt"
    )
    salt_ax.set(ylabel="Salt / mol/m³", ylim=(0, 550))
    protein_lines, protein_labels = ax.get_legend_handles_labels()
    salt_lines, salt_labels = salt_ax.get_legend_handles_labels()
    salt_ax.legend(
        protein_lines + salt_lines, protein_labels + salt_labels,
        loc="upper right", framealpha=1,
    )


def plot_lwe(results):
    """Compare the programmed inlet and the resulting column outlet in minutes."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    _draw_signal(axes[0], results.solution.column.inlet, "Inlet: load, wash and elution")
    _draw_signal(axes[1], results.solution.column.outlet, "Outlet: the protein separation")
    axes[1].set_xlabel("Time / min")
    fig.tight_layout()
    plt.show()
    return fig, axes


def plot_gradient_comparison(original, changed, original_label, changed_label):
    """Keep identical axes when comparing two outlet chromatograms."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, sharey=True)
    _draw_signal(axes[0], original.solution.column.outlet, original_label)
    _draw_signal(axes[1], changed.solution.column.outlet, changed_label)
    axes[1].set_xlabel("Time / min")
    fig.tight_layout()
    plt.show()
    return fig, axes
