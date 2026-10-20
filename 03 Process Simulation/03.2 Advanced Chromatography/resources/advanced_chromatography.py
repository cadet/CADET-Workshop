"""Supplied models and collection plots for the valve, SMB and carousel examples."""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display


def valve_process():
    """Return the lesson's Example 1: feed and eluent → valve (Cstr) → Langmuir column → outlet.

    Feed (10 mM A and B) flows for the first 60 s, then eluent, at 1 mL/min for 20 min.
    """
    from CADETProcess.processModel import (
        ComponentSystem, Cstr, FlowSheet, Inlet, Langmuir, LumpedRateModelWithPores, Outlet,
        Process,
    )

    Q = 1e-6 / 60  # 1 mL/min
    component_system = ComponentSystem(["A", "B"])

    binding_model = Langmuir(component_system, name="langmuir")
    binding_model.is_kinetic = False
    binding_model.adsorption_rate = [0.02, 0.03]
    binding_model.desorption_rate = [1, 1]
    binding_model.capacity = [100, 100]

    feed = Inlet(component_system, name="feed")
    feed.c = [10, 10]
    eluent = Inlet(component_system, name="eluent")
    eluent.c = [0, 0]

    valve = Cstr(component_system, "valve")
    valve.init_liquid_volume = 1e-6
    valve.const_solid_volume = 0

    column = LumpedRateModelWithPores(component_system, "column")
    column.binding_model = binding_model
    column.length = 0.1
    column.diameter = 0.01
    column.bed_porosity = 0.37
    column.particle_radius = 4.5e-5
    column.particle_porosity = 0.33
    column.axial_dispersion = 2.0e-7
    column.film_diffusion = [1e-4, 1e-4]

    outlet = Outlet(component_system, name="outlet")

    flow_sheet = FlowSheet(component_system)
    for unit in (feed, eluent, valve, column, outlet):
        flow_sheet.add_unit(unit)
    flow_sheet.add_connection(feed, valve)
    flow_sheet.add_connection(eluent, valve)
    flow_sheet.add_connection(valve, column)
    flow_sheet.add_connection(column, outlet)

    process = Process(flow_sheet, "batch elution")
    process.cycle_time = 1200
    process.add_event("feed_on", "flow_sheet.feed.flow_rate", Q, 0)
    process.add_event("feed_off", "flow_sheet.feed.flow_rate", 0.0, 60)
    process.add_event("eluent_off", "flow_sheet.eluent.flow_rate", 0.0, 0)
    process.add_event("eluent_on", "flow_sheet.eluent.flow_rate", Q, 60)
    return process


def show_smb_collection(raffinate, extract, feed):
    """Show full-cycle outlet pools; the model uses molar concentration units."""
    raffinate_amount = np.asarray(raffinate.mass) * 1e6  # mol to µmol
    extract_amount = np.asarray(extract.mass) * 1e6
    raffinate_share = 100 * raffinate.purity
    extract_share = 100 * extract.purity
    display(Markdown(
        "| Collected stream | Volume / mL | Protein A / µmol | Protein B / µmol | Target component / molar % |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| Raffinate | {raffinate.volume * 1e6:.1f} | {raffinate_amount[0]:.1f} | {raffinate_amount[1]:.1f} | A: {raffinate_share[0]:.2f} |\n"
        f"| Extract | {extract.volume * 1e6:.1f} | {extract_amount[0]:.1f} | {extract_amount[1]:.1f} | B: {extract_share[1]:.2f} |"
    ))

    fig, ax = plt.subplots(figsize=(7, 3))
    positions = np.array([0, 1])
    ax.bar(positions - 0.18, [raffinate_amount[0], extract_amount[0]],
           width=0.36, color="C0", label="Protein A")
    ax.bar(positions + 0.18, [raffinate_amount[1], extract_amount[1]],
           width=0.36, color="C1", label="Protein B")
    ax.set_xticks(positions, ["Raffinate", "Extract"])
    ax.set_ylabel("Amount collected / µmol")
    ax.set_title("Collected over the last simulated cycle")
    ax.legend()
    fig.tight_layout()
    plt.show()

    collected = np.asarray(raffinate.mass) + np.asarray(extract.mass)
    relative_to_feed = np.divide(
        collected, feed.mass, out=np.zeros_like(collected), where=np.asarray(feed.mass) > 0
    ) * 100
    display(Markdown(
        f"**Outlet totals relative to feed input in this cycle:** "
        f"Protein A {relative_to_feed[0]:.1f}%; Protein B {relative_to_feed[1]:.1f}%. "
        "The system is still filling during startup; these are not steady-state recoveries."
    ))
    return fig, ax


def carousel_column_template():
    """Create a fresh SMA column for capturing Target while Impurity flows through."""
    from CADETProcess.processModel import (
        ComponentSystem, StericMassAction, LumpedRateModelWithoutPores,
    )

    component_system = ComponentSystem(["Salt", "Impurity", "Target"])
    binding = StericMassAction(component_system, "SMA")
    binding.is_kinetic = True
    # Lists follow Salt, Impurity, Target; salt rate/steric zeros are placeholders.
    binding.adsorption_rate = [0, 0.001, 0.3]
    binding.desorption_rate = [0, 1.5, 1.5]
    binding.characteristic_charge = [1, 1, 7]
    binding.steric_factor = [0, 0, 50]
    binding.capacity = 225
    binding.reference_liquid_phase_conc = 300
    binding.reference_solid_phase_conc = 225

    column = LumpedRateModelWithoutPores(component_system, "master_column")
    column.length = 0.1
    column.diameter = 0.01
    column.total_porosity = 0.4
    column.axial_dispersion = 1e-7
    column.binding_model = binding
    column.c = [50, 0, 0]
    column.q = [225, 0, 0]
    column.discretization.ncol = 20  # A supplied grid for the qualitative demonstration.
    return component_system, column


def show_carousel_collection(product, waste, feed):
    """Compare protein pools; exclude buffer salt from protein composition."""
    product_protein = np.asarray(product.mass)[1:]
    waste_protein = np.asarray(waste.mass)[1:]
    feed_protein = np.asarray(feed.mass)[1:]
    product_amount = product_protein * 1e6
    waste_amount = waste_protein * 1e6
    product_share = np.divide(
        product_protein, product_protein.sum(), out=np.zeros_like(product_protein),
        where=product_protein.sum() > 0,
    ) * 100
    waste_share = np.divide(
        waste_protein, waste_protein.sum(), out=np.zeros_like(waste_protein),
        where=waste_protein.sum() > 0,
    ) * 100
    display(Markdown(
        "| Collected stream | Volume / mL | Impurity / µmol | Target / µmol | Target / protein molar % |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| Product | {product.volume * 1e6:.1f} | {product_amount[0]:.3f} | {product_amount[1]:.3f} | {product_share[1]:.2f} |\n"
        f"| Waste | {waste.volume * 1e6:.1f} | {waste_amount[0]:.3f} | {waste_amount[1]:.3f} | {waste_share[1]:.2f} |"
    ))
    fig, ax = plt.subplots(figsize=(7, 3))
    positions = np.array([0, 1])
    ax.bar(positions - 0.18, [product_amount[0], waste_amount[0]],
           width=0.36, color=(0, 0, 0.6), label="Impurity")
    ax.bar(positions + 0.18, [product_amount[1], waste_amount[1]],
           width=0.36, color=(0.6, 0, 0), label="Target")
    ax.set_xticks(positions, ["Product", "Waste"])
    ax.set_ylabel("Protein amount collected / µmol")
    ax.set_title("Collected over the last simulated cycle")
    ax.legend()
    fig.tight_layout()
    plt.show()

    collected = product_protein + waste_protein
    relative_to_feed = np.divide(
        collected, feed_protein, out=np.zeros_like(collected), where=feed_protein > 0
    ) * 100
    display(Markdown(
        f"**Outlet totals relative to feed input in this cycle:** "
        f"Impurity {relative_to_feed[0]:.1f}%; Target {relative_to_feed[1]:.1f}%. "
        "Protein composition excludes salt. Startup inventory can change, so a cycle's outlet/input ratio "
        "can exceed 100% as stored material leaves. Two cycles alone do not establish cyclic stationarity."
    ))
    return fig, ax
