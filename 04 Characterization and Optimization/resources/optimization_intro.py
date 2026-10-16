"""Prepared tracer models and estimation panel for the optimization lessons."""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
from CADETProcess.comparison import Comparator, SSE
from CADETProcess.optimization import OptimizationProblem
from CADETProcess.simulator import Cadet

from cadetgui.estimation.optimizer_runner import RunSpec
from cadetgui.estimation.parameter_estimation import (
    EstimationResult,
    list_fittable_parameters,
)
from cadetgui.io.experimental_data import ExperimentalDataset
from cadetgui.widgets.estimation import ParameterEstimationWidget
from cadetgui.widgets.instrument.configuration import ConfigurationWidget


def load_porosity_measurement():
    """Load the supplied column-only, 20 s pulse experiment; time is in seconds."""
    return np.loadtxt(
        Path(__file__).with_name("column_porosity_pulse.csv"), delimiter=",", skiprows=1
    )


def measured_column_delay(data):
    """Remove the inlet centroid (10 s) from the detector pulse centroid."""
    t, c = data.T
    return float(np.trapezoid(t * c, t) / np.trapezoid(c, t) - 10)


def porosity_demo_configuration():
    """Prepare the familiar column and pulse without displaying configuration controls."""
    config = ConfigurationWidget(workspace_header=False, starting_values=False)
    config.components = ["Tracer"]
    config.select_models(
        template="Pulse Feed (Single Component)",
        column="Lumped Rate Model Without Pores (LRM)",
        binding="None",
    )
    config.column_form.set_values(
        {
            "length": 0.1,
            "diameter": 0.02,
            "total_porosity": 0.4,
            "axial_dispersion": 1e-7,
        }
    )
    config._model_form.set_values(
        {
            "component": "Tracer",
            "concentration": 1,
            "flow_rate": 1e-7,
            "pulse_duration": 20,
            "cycle_time": 300,
        }
    )
    return config


class PorosityIntroWidget(ParameterEstimationWidget):
    """Use the estimation panel with the non-binding column's available parameters.

    NoBinding needs no binding form; the fitted problem uses the metric-instance API.
    """

    porosity_bounds = (0.2, 0.6)

    def _on_config_changed(self, _process):
        config = self._config_widget
        params = list_fittable_parameters(
            "column",
            config.column_form.spec.fields,
            config.column_form.collect_values(),
        )
        # Keep useful demonstration bounds without restricting the parameter picker.
        params = [
            replace(param, lb=self.porosity_bounds[0], ub=self.porosity_bounds[1])
            if param.name == "total_porosity"
            else param
            for param in params
        ]
        self.param_space.set_params(params)
        self._refresh_component_options()
        self._refresh_signal_options()
        self._refresh_active_label()
        if self.auto_preview:
            self._refresh_preview()
        else:
            self._preview_stale = True

    def _build_run_spec(self):
        error = self._validation_error()
        if error:
            return error
        self._last_result = None
        rows = self.param_space.rows()
        params = list(self.param_space.params)
        baseline = copy.deepcopy(self._config_widget.process)
        working = copy.deepcopy(baseline)
        reference, _ = self._current_reference()
        unit, port = self._signal_picker.value
        simulator = Cadet()
        comparator = Comparator()
        comparator.add_difference_metric(SSE(reference), f"{unit}.{port}")
        problem = OptimizationProblem("column_tracer_fit")
        problem.add_evaluation_object(working)
        for index, _start, lower, upper in rows:
            param = params[index]
            kwargs = (
                {}
                if param.component_index is None
                else {"indices": [param.component_index]}
            )
            problem.add_variable(
                f"var_{index}",
                targets=[working],
                parameter_path=f"flow_sheet.column.{param.name}",
                lb=lower,
                ub=upper,
                normalization="auto",
                **kwargs,
            )
        problem.add_evaluator(simulator)
        problem.add_objective(comparator, name="tracer_error", requires=[simulator])

        def candidate_result(values):
            candidate = copy.deepcopy(baseline)
            for (index, *_), value in zip(rows, values):
                param = params[index]
                if param.component_index is None:
                    setattr(candidate.flow_sheet.column, param.name, float(value))
                else:
                    entries = list(getattr(candidate.flow_sheet.column, param.name))
                    entries[param.component_index] = float(value)
                    setattr(candidate.flow_sheet.column, param.name, entries)
            return Cadet().simulate(candidate)

        def render_preview(values, ax):
            candidate_result(values).solution[unit][port].plot(ax=ax)
            ax.plot(reference.time / 60, reference.solution, "--", label="Measured")
            ax.legend()

        def fitted(values):
            return {index: float(values[f"var_{index}"]) for index, *_ in rows}

        def on_finished(result):
            if result.cancelled or not result.success:
                return
            self._display_result = candidate_result(
                [result.x_best[f"var_{index}"] for index, *_ in rows]
            )
            self._last_result = EstimationResult(
                fitted(result.x_best),
                result.objective,
                True,
                result.message,
                self._display_result.process,
            )
            self._redraw_overlay()
            self._notify(self._last_result)

        return RunSpec(
            problem,
            [start for _index, start, _lower, _upper in rows],
            render_preview,
            lambda values: self._fit_table_html(fitted(values)),
            lambda values: self._apply_fitted(fitted(values)),
            on_finished,
        )


def porosity_estimation_widget(
    data, *, configuration=None, label="Column tracer pulse", porosity_bounds=(0.2, 0.6)
):
    """Preload the measurement and baseline; leave parameter selection to the presenter."""
    widget = PorosityIntroWidget()
    widget.porosity_bounds = porosity_bounds
    widget.auto_preview = False
    widget.data._time_unit.value = "s"
    widget.data.datasets.append(
        ExperimentalDataset(label=label, time_min=data[:, 0] / 60, signal=data[:, 1])
    )
    widget.data._finish_import()
    # Keep data import and signal/calibration controls available for demonstration.
    widget._base_process_picker.layout.display = "none"
    widget._btn_refresh_store.layout.display = "none"
    config = (
        configuration if configuration is not None else porosity_demo_configuration()
    )
    widget.bind_to_config(config)
    # Match the signal used for the Python interpretation.
    options = [label for label in widget._signal_picker.option_labels]
    widget._signal_picker.selected_index = options.index("Column outlet")
    widget._runner._optimizer_picker.value = "Nelder-Mead"
    widget._runner._knob_fields["Nelder-Mead"][0].value = 100
    widget.auto_preview = True
    widget.refresh_preview()
    return widget
