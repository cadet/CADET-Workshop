"""Link illustrative process settings to productivity and product purity."""

import matplotlib.pyplot as plt
import numpy as np


class ProductivityComparison:
    """Inspect three operating points with a fixed 95% purity requirement."""

    def __init__(self):
        self.feed_durations = np.array([30, 45, 30])
        self.cycle_times = np.array([10, 8, 8])
        self.target_amounts = np.array([5.0, 5.0, 4.6])
        self.purities = np.array([97, 93, 96])
        self.productivities = (
            self.target_amounts / self.cycle_times
        )  # 1 mL stationary phase.
        self.feasible = self.purities >= 95
        eligible = np.flatnonzero(self.feasible)
        self.best_index = int(eligible[np.argmax(self.productivities[eligible])])
        self.best_candidate = self.best_index + 1

    def plot(self):
        """Plot all operating settings and their performance without selection controls."""
        self.figure, (settings_ax, performance_ax) = plt.subplots(
            1, 2, figsize=(10, 3.6)
        )
        performance_ax.axvspan(95, 100, color="#16805d", alpha=0.08)
        performance_ax.axvline(95, color="#16805d", linestyle="--", linewidth=1.5)
        for index in range(3):
            if index == self.best_index:
                color, marker, size = "#16805d", "*", 250
                label = f"{index + 1}: best acceptable"
            elif self.feasible[index]:
                color, marker, size = "#2864a0", "o", 70
                label = f"{index + 1}: acceptable"
            else:
                color, marker, size = "#c34f4f", "X", 90
                label = f"{index + 1}: purity too low"
            for ax, x, y in [
                (settings_ax, self.feed_durations[index], self.cycle_times[index]),
                (performance_ax, self.purities[index], self.productivities[index]),
            ]:
                ax.scatter(x, y, c=color, marker=marker, s=size, zorder=3)
                ax.annotate(
                    str(index + 1) if ax is settings_ax else label,
                    (x, y),
                    xytext=(10, 8),
                    textcoords="offset points",
                    color=color,
                    fontsize=9,
                )
        settings_ax.set(
            xlabel="Feed duration / s",
            ylabel="Cycle time / min",
            title="Parameter space: operating settings",
            xlim=(25, 52),
            ylim=(7.3, 10.7),
            yticks=[8, 9, 10],
        )
        performance_ax.set(
            xlabel="Product purity / %",
            ylabel="Productivity / (mg/(mL·min))",
            title="Performance\nProductivity, with purity ≥ 95%",
            xlim=(91, 101),
            ylim=(0.475, 0.675),
            yticks=[0.5, 0.55, 0.6, 0.65],
        )
        for ax in (settings_ax, performance_ax):
            ax.grid(alpha=0.15)
            ax.spines[["top", "right"]].set_visible(False)
        self.figure.tight_layout()
        return self.figure, (settings_ax, performance_ax)


def _pareto_mask(values, eligible):
    """Identify nondominated eligible points when both columns are maximized."""
    front = np.zeros(len(values), dtype=bool)
    candidates = values[eligible]
    for index in np.flatnonzero(eligible):
        dominates = np.all(candidates >= values[index], axis=1) & np.any(
            candidates > values[index], axis=1
        )
        front[index] = not np.any(dominates)
    return front


def plot_productivity_pareto(comparison):
    """Illustrate a smooth trade-off envelope and the effect of a purity constraint."""
    from scipy.interpolate import PchipInterpolator

    # A monotone illustrative envelope through the original three candidate results.
    order = np.argsort(comparison.purities)
    envelope = PchipInterpolator(
        comparison.purities[order], comparison.productivities[order]
    )
    purity_grid = np.linspace(93, 97, 501)
    productivity_grid = envelope(purity_grid)
    rng = np.random.default_rng(42)
    cloud_purity = rng.uniform(93, 97, 160)
    cloud_productivity = envelope(cloud_purity) - rng.uniform(0.015, 0.12, 160)
    # Fourth illustrative setting: 45 s feed, 10 min cycle, 5.4 mg target, 95% purity.
    purity = np.append(comparison.purities, 95)
    productivity = np.append(comparison.productivities, 5.4 / 10)
    values = np.column_stack([purity, productivity])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True)
    for ax, constrained in zip(axes, [False, True]):
        eligible = purity >= 95 if constrained else np.ones(len(purity), dtype=bool)
        front = _pareto_mask(values, eligible)
        color = "#16805d" if constrained else "#7652a0"
        cloud_eligible = cloud_purity >= 95 if constrained else np.ones(160, dtype=bool)
        ax.scatter(
            cloud_purity[~cloud_eligible],
            cloud_productivity[~cloud_eligible],
            c="#b5bcc4",
            s=14,
            alpha=0.15,
            edgecolors="none",
        )
        ax.scatter(
            cloud_purity[cloud_eligible],
            cloud_productivity[cloud_eligible],
            c="#7892a5",
            s=17,
            alpha=0.35,
            edgecolors="none",
            label="Other illustrative candidates",
        )
        if constrained:
            ax.axvspan(95, 97.7, color="#16805d", alpha=0.07)
            ax.axvline(95, color="#16805d", linestyle="--", linewidth=1.3)
            excluded = purity_grid < 95
            ax.plot(
                purity_grid[excluded],
                productivity_grid[excluded],
                color="#b5bcc4",
                linestyle=":",
                linewidth=2,
            )
        shown = (
            purity_grid >= 95 if constrained else np.ones(len(purity_grid), dtype=bool)
        )
        ax.plot(
            purity_grid[shown],
            productivity_grid[shown],
            color=color,
            label="Illustrative Pareto front",
            linewidth=2.5,
        )
        for index in range(len(purity)):
            point_color = color if front[index] else "gray"
            marker = "o" if front[index] else "x"
            if not eligible[index]:
                point_color, marker = "#c34f4f", "X"
            ax.scatter(
                purity[index],
                productivity[index],
                c=point_color,
                marker=marker,
                s=65,
                zorder=4,
            )
            ax.annotate(
                str(index + 1),
                (purity[index], productivity[index]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
            )
        ax.annotate(
            "",
            xy=(96, 0.575),
            xytext=(95, 0.54),
            arrowprops={
                "arrowstyle": "->",
                "color": "gray",
                "shrinkA": 10,
                "shrinkB": 10,
            },
        )
        if constrained:
            best_productivity = float(envelope(95))
            ax.scatter(95, best_productivity, marker="*", c=color, s=180, zorder=5)
            ax.annotate(
                "Highest acceptable\nproductivity on this curve",
                xy=(95, best_productivity),
                xytext=(95.5, 0.64),
                fontsize=9,
                color=color,
                arrowprops={"arrowstyle": "->", "color": color},
            )
        ax.set(
            xlabel="Product purity / %",
            title="Objectives with purity ≥ 95%"
            if constrained
            else "Objective space: productivity and purity",
            xlim=(92.5, 97.7),
            ylim=(0.36, 0.675),
            yticks=[0.4, 0.5, 0.6],
        )
        ax.grid(alpha=0.12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("Productivity / (mg/(mL·min))")
    fig.tight_layout()
    return fig, axes
