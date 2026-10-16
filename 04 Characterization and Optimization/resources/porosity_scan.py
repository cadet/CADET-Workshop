"""Compare tracer curves and their fitting errors while changing column porosity."""

import copy

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from CADETProcess.simulator import Cadet
from IPython.display import clear_output, display


class PorosityScanWidget(widgets.VBox):
    """Precompute a small scan so the slider responds without new simulations."""

    def __init__(self, process, data):
        super().__init__()
        self.data = np.asarray(data)
        self.porosities = np.round(np.linspace(0.2, 0.6, 17), 3)
        self.signals = []
        self.scores = []
        working = copy.deepcopy(process)
        simulator = Cadet()
        simulator.time_resolution = 0.5
        for porosity in self.porosities:
            working.flow_sheet.column.total_porosity = float(porosity)
            result = simulator.simulate(working)
            signal = result.solution.column.outlet
            concentration = np.interp(
                self.data[:, 0], signal.time, signal.solution[:, 0]
            )
            self.signals.append(concentration)
            self.scores.append(float(np.sum((concentration - self.data[:, 1]) ** 2)))

        self.slider = widgets.SelectionSlider(
            options=[
                (f"{value:.3f}", index) for index, value in enumerate(self.porosities)
            ],
            value=int(np.argmin(abs(self.porosities - 0.4))),
            description="Porosity",
            continuous_update=False,
            layout=widgets.Layout(width="95%"),
        )
        self.output = widgets.Output()
        self.children = (self.slider, self.output)
        self.slider.observe(self._update, names="value")
        self._update()

    def _update(self, _change=None):
        index = self.slider.value
        with self.output:
            clear_output(wait=True)
            self.figure, (ax, ax_score) = plt.subplots(1, 2, figsize=(9, 3.2))
            time_min = self.data[:, 0] / 60
            ax.plot(time_min, self.data[:, 1], "k:", label="Reference")
            ax.plot(time_min, self.signals[index], label="Candidate")
            ax.fill_between(
                time_min, self.data[:, 1], self.signals[index], color="red", alpha=0.2
            )
            ax.set(
                xlabel="Time / min",
                ylabel="Tracer concentration / mol/m³",
                title=f"Column porosity: {self.porosities[index]:.3f}",
                ylim=(-0.02, 1.1 * max(self.data[:, 1].max(), np.max(self.signals))),
            )
            ax.legend()
            ax_score.plot(self.porosities, self.scores, ".-", color="gray")
            ax_score.plot(self.porosities[index], self.scores[index], "ro")
            ax_score.axvline(self.porosities[index], color="red", linestyle=":")
            ax_score.set(
                xlabel="Porosity / −",
                ylabel="Sum of squared errors / (mol/m³)²",
                title=f"Candidate error: {self.scores[index]:.3g}",
            )
            self.figure.tight_layout()
            display(self.figure)
            plt.close(self.figure)
