
import asyncio
from time import time

import cv2
import imageio.v3 as iio
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.style as mplstyle
import numpy as np
from IPython.core.display_functions import display
from ipywidgets import HBox, VBox
from ipywidgets import Layout
from matplotlib.gridspec import GridSpec
from scipy.interpolate import PchipInterpolator, interp1d

from .langmuir import create_sim_langmuir
from .lwe import create_sim_lwe
from .non_pen_tracer import create_sim_non_pen
from .pen_tracer import create_sim_pen

mpl.rcParams['image.interpolation'] = "none"
mpl.rcParams["path.simplify"] = "True"
mpl.rcParams['path.simplify_threshold'] = 1
mplstyle.use('fast')

particle_large = iio.imread("resources/particle_large.png")
particle_shells = iio.imread("resources/particle_shells.png")

class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def start(self):
        self._task = asyncio.ensure_future(self._job())

    def cancel(self):
        self._task.cancel()


def throttle(wait):
    """ Decorator that prevents a function from being called
        more than once every wait period. """

    def decorator(fn):
        time_of_last_call = 0
        scheduled, timer = False, None
        new_args, new_kwargs = None, None

        def throttled(*args, **kwargs):
            nonlocal new_args, new_kwargs, time_of_last_call, scheduled, timer

            def call_it():
                nonlocal new_args, new_kwargs, time_of_last_call, scheduled, timer
                time_of_last_call = time()
                fn(*new_args, **new_kwargs)
                scheduled = False

            time_since_last_call = time() - time_of_last_call
            new_args, new_kwargs = args, kwargs
            if not scheduled:
                scheduled = True
                new_wait = max(0, wait - time_since_last_call)
                timer = Timer(new_wait, call_it)
                timer.start()

        return throttled

    return decorator


def debounce(wait):
    """ Decorator that will postpone a function's
        execution until after `wait` seconds
        have elapsed since the last time it was invoked. """
    def decorator(fn):
        timer = None
        def debounced(*args, **kwargs):
            nonlocal timer
            def call_it():
                fn(*args, **kwargs)
            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)
            timer.start()
        return debounced
    return decorator



def scale(arr, maximum=None, minimum=None):
    if maximum is None:
        maximum = arr.max()
    if minimum is None:
        minimum = arr.min()
    arr[arr < 0] = 0
    return (arr - minimum) / (maximum - minimum)


def interpolate(input_array, fineness=800, kind="nearest"):
    interpolated_list = []
    x_in = np.arange(0, input_array.shape[1])
    x_out = np.linspace(0, input_array.shape[1] - 1, fineness)
    if kind == "nearest":
        for i in range(input_array.shape[0]):  # iterate over timepoints
            if len(input_array[i, :].shape) > 1:
                shell_interpolations = []
                for j in range(input_array[i, :].shape[1]):  # iterate over particle shells
                    sell_interpolated = interp1d(x_in, input_array[i, :, j], kind="previous")(x_out)
                    shell_interpolations.append(sell_interpolated)
                interpolated = np.stack(shell_interpolations).T
            else:
                interpolated = interp1d(x_in, input_array[i, :], kind="previous")(x_out)

            interpolated_list.append(interpolated)
    else:
        for i in range(input_array.shape[0]):  # iterate over timepoints
            interpolated = PchipInterpolator(x_in, input_array[i, :])(x_out)
            interpolated_list.append(interpolated)

    interpolated_array = np.stack(interpolated_list)
    return interpolated_array


def set_ax_ylims(axis, data, auto=False):
    min = data.flatten().min()
    max = data.flatten().max()
    range = max - min
    min_point = 0
    max_point = max + range * 0.05
    axis.set_ylim(min_point, max_point, auto=auto)


class TkLikeDropdown(widgets.Dropdown):
    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class OrderedTkLikeDropdown(TkLikeDropdown):
    def get(self):
        return self.options.index(self.value)


class TkLikeTextBox(widgets.widgets.BoundedIntText):
    def get(self):
        return self.value


class TkLikeSliderInterface:
    def get(self):
        return self.value

    def set(self, value):
        self.value = value

    def configure(self, state=None, options=None, from_=None, to=None, **kwargs):
        if state is not None:
            if state == "disabled":
                self.disabled = True
            elif state == "normal":
                self.disabled = False
            else:
                raise ValueError(f"Unknown option {options}.")
        if from_ is not None:
            self.min = from_
        if to is not None:
            self.max = to


class TkLikeSlider(widgets.FloatSlider, TkLikeSliderInterface):
    pass


class TkLikeLogSlider(widgets.FloatLogSlider, TkLikeSliderInterface):
    pass


class TkLikeCheckbox(widgets.Checkbox):
    def get(self):
        return self.value

    def select(self):
        self.value = True

    def deselect(self):
        self.value = False


class Gui:
    def __init__(self):
        self.fig = None
        self.ax_inlet = None
        self.ax_inlet_twin = None
        self.ax_column = None
        self.ax_bulk = None
        self.ax_bulk_twin = None
        self.ax_outlet = None
        self.ax_outlet_twin = None
        self.axes = None
        self.reference_line = None
        self.previous_porosity = (0, 0)

        self.allow_simulations = False

        self.creat_figures()

        self.experiment_id = OrderedTkLikeDropdown(options=('Large tracer', 'Small tracer', "Langmuir", "SMA"),
                                                   value='Large tracer')

        self.slider_inlet = TkLikeSlider(value=0, min=0, max=1, step=0.01, readout=False, description="Time")

        self.checkbox_precision = TkLikeCheckbox(value=False, description='High precision')
        self.checkbox_timepoint = TkLikeCheckbox(value=False, description='Keep timepoint')
        self.checkbox_reference = TkLikeCheckbox(value=False, description='Show reference')

        self.textbox_n_cols = TkLikeTextBox(
            value=100,
            min=1,
            max=400,
            step=1,
            description='n_cols:',
            disabled=False
        )

        left_box = VBox([self.experiment_id,
                         self.checkbox_precision,
                         self.checkbox_timepoint,
                         self.checkbox_reference,
                         self.textbox_n_cols,
                         self.slider_inlet])

        self.right_box = []
        self.slider_col_dispersion = self.add_slider(value=1e-6, min=-9, max=-2, step=0.01, readout=True,
                                                     description="Colum dispersion", log=True)
        self.slider_col_porosity = self.add_slider(value=0.4, min=0.1, max=0.9, step=0.01, readout=True,
                                                   description="Colum porosity", log=False)
        self.slider_par_porosity = self.add_slider(description="Particle porosity: ", value=0.5, min=0.1, max=0.8,
                                                   step=0.01, readout=True)
        self.slider_film_diffusion = self.add_slider(description="Film diffusion: ", value=1e-5, min=-7, max=-3,
                                                     log=True, step=0.01, readout=True)
        self.slider_par_diffusion = self.add_slider(description="Particle diffusion: ", value=1e-8, min=-13, max=-6,
                                                    log=True, step=0.1, readout=True)
        self.slider_keq = self.add_slider(description="Equilibrium constant: ", value=0.01, min=-3, max=0.69897000433,
                                          log=True, step=0.01, readout=True)
        self.slider_nu = self.add_slider(description="Characteristic charge: ", min=1, max=28, step=1, readout=True)
        # self.slider_qmax = self.add_slider(description="Capacity: ", min=1, max=2.5, log=True, step=0.01, readout=True)

        right_box = VBox(self.right_box)
        display(HBox([left_box, right_box], layout=Layout(width='100%', )))

        self.set_experiment()
        self.has_prepared_image = False
        self.allow_simulations = True
        self.checkbox_timepoint.deselect()

        widgets.interactive_output(self.plot, {"offset": self.slider_inlet})
        widgets.interactive_output(self.set_experiment, {"id": self.experiment_id})
        widgets.interactive_output(self.simulate, {"_": self.checkbox_precision})
        widgets.interactive_output(self.toggle_reference, {"_": self.checkbox_reference})

    def plot(self, offset):
        idx_limit = max(1, int(offset * len(self.output_times)))

        self.plot_io_update(self.outlet_line, self.outlet_twin_line, self.outlet_vline,
                            idx_limit, self.output, self.output_times)
        self.inlet_vline.set_xdata([self.output_times[idx_limit - 1], self.output_times[idx_limit - 1]])

        self.plot_bulk_update(idx_limit)

        if offset != 1 and self.has_prepared_image is False:
            self.prepare_image()
        self.plot_column_update(idx_limit)

    def draw_all(self):
        return

    def plot_column_initial(self):
        bulk_image_slice = self.bulk_image[0, :, :]
        image = self.ax_column.imshow(bulk_image_slice, aspect=1)
        return image

    def plot_column_update(self, idx_limit):
        bulk_image_slice = self.bulk_image[idx_limit - 1, :, :]
        if self.column_image is None and self.has_prepared_image:
            image = self.ax_column.imshow(bulk_image_slice, aspect=1)
            self.column_image = image
        else:
            self.column_image.set_data(bulk_image_slice)

    def plot_bulk_initial(self):
        if self.experiment_id.get() < 2:
            protein_index = 0
        else:
            protein_index = 1

        bulk_slice = self.bulk[0, :, :]
        solid_slice = self.solid[0, :, :] / self.solid_max * self.bulk_protein_max
        line1 = self.ax_bulk.plot(bulk_slice[::-1, protein_index], range(bulk_slice.shape[0])[::-1])[0]
        if self.experiment_id.get() == 5:
            line_twin = self.ax_bulk_twin.plot(bulk_slice[::-1, 0], range(bulk_slice.shape[0])[::-1], color="orange")[0]
            line2 = self.ax_bulk.plot(solid_slice[::-1, 0], range(bulk_slice.shape[0])[::-1], color="green")[0]
        elif self.experiment_id.get() == 3 or self.experiment_id.get() == 4:
            line_twin = self.ax_bulk_twin.plot(bulk_slice[::-1, 0], range(bulk_slice.shape[0])[::-1], color="orange")[0]
            line2 = self.ax_bulk.plot(solid_slice[::-1, 1], range(bulk_slice.shape[0])[::-1], color="green")[0]
        elif self.experiment_id.get() == 2:
            line2 = self.ax_bulk.plot(solid_slice[::-1, 0], range(bulk_slice.shape[0])[::-1], color="green")[0]
            line_twin = self.ax_bulk_twin.plot([0, 0], [0, 0], color="orange")[0]
        else:
            line2 = self.ax_bulk.plot([0, 0], [0, 0], color="green")[0]
            line_twin = self.ax_bulk_twin.plot([0, 0], [0, 0], color="orange")[0]
        return line1, line2, line_twin

    def plot_bulk_update(self, idx_limit):
        if self.experiment_id.get() < 2:
            protein_index = 0
        else:
            protein_index = 1

        bulk_slice = self.bulk[idx_limit - 1, :, :]

        self.bulk_line1.set_data(bulk_slice[::-1, protein_index], np.arange(bulk_slice.shape[0])[::-1])
        if self.experiment_id.get() == 5:
            self.bulk_line_twin.set_data(bulk_slice[::-1, 0], np.arange(bulk_slice.shape[0])[::-1])
            solid_slice = self.solid[idx_limit - 1, :, 0, :]  # / self.solid_max * self.bulk_protein_max
            self.bulk_line2.set_data(solid_slice[::-1, 0], np.arange(bulk_slice.shape[0])[::-1])
        elif self.experiment_id.get() == 3 or self.experiment_id.get() == 4:
            self.bulk_line_twin.set_data(bulk_slice[::-1, 0], np.arange(bulk_slice.shape[0])[::-1])
            solid_slice = self.solid[idx_limit - 1, :, 0, :]  # / self.solid_max * self.bulk_protein_max
            self.bulk_line2.set_data(solid_slice[::-1, 1], np.arange(bulk_slice.shape[0])[::-1])
        elif self.experiment_id.get() >= 2:
            solid_slice = self.solid[idx_limit - 1, :, 0, :]  # / self.solid_max * self.bulk_protein_max
            self.bulk_line2.set_data(solid_slice[::-1, 0], np.arange(bulk_slice.shape[0])[::-1])
        else:
            self.bulk_line_twin.set_data([0, 0], [0, 0])
            self.bulk_line2.set_data([0, 0], [0, 0])

    def plot_io_initial(self, axis, axis_twin, idx_limit, data, times):
        # axis_twin.clear()
        if self.experiment_id.get() >= 3:
            line_twin = axis_twin.plot(times[:idx_limit], data[:idx_limit, 0], color="orange")[0]
        else:
            line_twin = axis_twin.plot([0, 0], [0, 0], color="orange")[0]

        # axis.clear()
        if self.experiment_id.get() < 2:
            protein_index = 0
        else:
            protein_index = 1
        line = axis.plot(times[:idx_limit], data[:idx_limit, protein_index])[0]
        vline = axis.plot([times[idx_limit - 1], times[idx_limit - 1]],
                          [min(data[:, protein_index]) - max(data[:, protein_index]) * 0.05,
                           max(data[:, protein_index]) + max(data[:, protein_index]) * 0.05],
                          ":", color="black")[0]
        return line, line_twin, vline

    def plot_io_update(self, line, line_twin, vline, idx_limit, data, times):
        # axis_twin.clear()
        if self.experiment_id.get() >= 3:
            line_twin.set_data(times[:idx_limit], data[:idx_limit, 0])
        else:
            line_twin.set_data([0, 0], [0, 0])
        # axis.clear()

        if self.experiment_id.get() < 2:
            protein_index = 0
        else:
            protein_index = 1

        line.set_data(times[:idx_limit], data[:idx_limit, protein_index])
        vline.set_xdata([times[idx_limit - 1], times[idx_limit - 1]])

    def creat_figures(self):
        # configure window
        self.fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(3, 3, figure=self.fig, width_ratios=[1, 0.5, 3], height_ratios=[1, 3, 3])

        ## CREATE INLET FIGURE
        self.ax_inlet = self.fig.add_subplot(gs[0, 0])
        self.ax_inlet_twin = self.ax_inlet.twinx()

        ## CREATE Column FIGURE
        self.ax_column = self.fig.add_subplot(gs[1:, 1])
        self.ax_column.set_xticks([])
        self.ax_column.set_yticks([])

        ## CREATE bulk FIGURE
        self.ax_bulk = self.fig.add_subplot(gs[1:, 0])
        self.ax_bulk_twin = self.ax_bulk.twiny()

        ## CREATE Outlet FIGURE
        self.ax_outlet = self.fig.add_subplot(gs[1:, 2])
        self.ax_outlet_twin = self.ax_outlet.twinx()

        self.axes = [self.ax_outlet, self.ax_outlet_twin, self.ax_inlet_twin, self.ax_inlet,
                     self.ax_column, self.ax_bulk, self.ax_bulk_twin]
        self.fig.tight_layout()
        # plt.show()

    def toggle_reference(self, _=None):
        if self.experiment_id.get() == 0:
            data_name = "non_pen_tracer.npy"
        elif self.experiment_id.get() == 1:
            data_name = "pen_tracer.npy"
        elif self.experiment_id.get() == 2:
            data_name = "langmuir.npy"
        elif self.experiment_id.get() == 3:
            data_name = "lwe.npy"
        elif self.experiment_id.get() == 4:
            data_name = "breakthrough.npy"
        else:
            data_name = "HIC.npy"

        data = np.load(data_name)

        if self.checkbox_reference.get():
            self.reference_line.set_data(data[0, :], data[1, :])
        else:
            self.reference_line.set_data([0, 1], [0, 0])

    #@debounce(0.1)
    def simulate(self, _=None):
        if not self.allow_simulations:
            return
        if self.checkbox_precision.get():
            times = self.sim.root.input.solver.user_solution_times
            self.sim.root.input.solver.user_solution_times = np.linspace(0, times.max(), 300)
            self.sim.root.input.solver.time_integrator.abstol = 0.000001
            self.sim.root.input.solver.time_integrator.algtol = 0.0001
            self.sim.root.input.solver.time_integrator.reltol = 0.00001
        else:
            times = self.sim.root.input.solver.user_solution_times
            self.sim.root.input.solver.user_solution_times = np.linspace(0, times.max(), 100)
            self.sim.root.input.solver.time_integrator.abstol = 0.0001
            self.sim.root.input.solver.time_integrator.algtol = 0.01
            self.sim.root.input.solver.time_integrator.reltol = 0.001

        if hasattr(self, "textbox_n_cols"):
            self.sim.root.input.model.unit_001.discretization.ncol = int(self.textbox_n_cols.get())

        # SLIDERS Processing
        self.sim.root.input.model.unit_001.col_porosity = self.slider_col_porosity.get()
        self.sim.root.input.model.unit_001.col_dispersion = self.slider_col_dispersion.get()
        if self.experiment_id.get() >= 1:
            self.sim.root.input.model.unit_001.par_porosity = self.slider_par_porosity.get()
            self.sim.root.input.model.unit_001.film_diffusion[:] = self.slider_film_diffusion.get()
            self.sim.root.input.model.unit_001.par_diffusion[:] = self.slider_par_diffusion.get()
        if self.experiment_id.get() == 2:
            self.sim.root.input.model.unit_001.adsorption.mcl_ka = [self.slider_keq.get()] * 2
            self.sim.root.input.model.unit_001.adsorption.mcl_kd = [1, 1]
        if self.experiment_id.get() == 3 or self.experiment_id.get() == 4:
            self.sim.root.input.model.unit_001.adsorption.sma_ka[1] = self.slider_keq.get()
            self.sim.root.input.model.unit_001.adsorption.sma_kd[1] = 1
            self.sim.root.input.model.unit_001.adsorption.sma_nu[1] = self.slider_nu.get()
            if hasattr(self, "slider_qmax"):
                self.sim.root.input.model.unit_001.adsorption.sma_lambda = self.slider_qmax.get()
        self.sim.save()
        return_code = self.sim.run_load()
        if len(return_code.stderr) != 0:
            print(return_code.returncode, return_code.stderr)

        self.load_sim_values()
        if self.previous_porosity != (
                self.sim.root.input.model.unit_001.col_porosity, self.sim.root.input.model.unit_001.par_porosity
        ):
            self.load_particle_mask()
            self.previous_porosity = (
                self.sim.root.input.model.unit_001.col_porosity, self.sim.root.input.model.unit_001.par_porosity
            )
        self.set_ax_limits()
        self.has_prepared_image = False
        if self.checkbox_timepoint.get():
            self.plot(offset=self.slider_inlet.get())
        else:
            self.plot(offset=1)
            self.slider_inlet.set(1)

    def add_slider(self, log=False, *args, **kwargs):
        style = {'description_width': '200px', "handle_color": "blue"}
        layout = Layout(width="500px")
        if log:
            slider = TkLikeLogSlider(*args, **kwargs, style=style, layout=layout)
        else:
            slider = TkLikeSlider(*args, **kwargs, style=style, layout=layout)
        widgets.interactive_output(self.simulate, {"_": slider})
        self.right_box.append(slider)
        return slider

    def set_experiment(self, id=None):
        self.checkbox_reference.deselect()

        # To prevent running a simulation each time parameters are set, we freeze the model,
        # which is caught in the .simulate() method
        self.allow_simulations = False

        if self.experiment_id.get() == 0:
            self.sim = create_sim_non_pen()
        elif self.experiment_id.get() == 1:
            self.sim = create_sim_pen()
        elif self.experiment_id.get() == 2:
            self.sim = create_sim_langmuir()
        elif self.experiment_id.get() == 3:
            self.sim = create_sim_lwe()
            self.checkbox_precision.select()

        if hasattr(self, "reference_line"):
            del self.reference_line

        if self.experiment_id.get() != 3:
            self.checkbox_precision.deselect()

        if self.experiment_id.get() == 2:
            self.slider_keq.configure(from_=-3, to=0)
        if self.experiment_id.get() == 3 or self.experiment_id.get() == 4:
            self.slider_keq.configure(from_=-5, to=1)
        if self.experiment_id.get() == 5:
            self.slider_keq.configure(from_=-5, to=0)

        self.sim.filename = r'tmp\sim.h5'
        self.sim.cadet_path = r"C:/Users/ronal/mambaforge/envs/interactive/bin/cadet-cli.exe"
        self.sim.save()
        return_code = self.sim.run()
        if len(return_code.stderr) != 0:
            print(return_code.stderr, return_code.stdout)
        self.sim.load()

        self.slider_col_dispersion.set(self.sim.root.input.model.unit_001.col_dispersion)
        self.slider_col_porosity.set(self.sim.root.input.model.unit_001.col_porosity)
        self.slider_par_porosity.set(self.sim.root.input.model.unit_001.par_porosity)
        self.slider_film_diffusion.set(self.sim.root.input.model.unit_001.film_diffusion[-1])
        self.slider_par_diffusion.set(self.sim.root.input.model.unit_001.par_diffusion[-1])

        if self.experiment_id.get() == 2:
            self.slider_keq.set(self.sim.root.input.model.unit_001.adsorption.mcl_ka[0])

        if self.experiment_id.get() == 3 or self.experiment_id.get() == 4:
            self.slider_keq.set(self.sim.root.input.model.unit_001.adsorption.sma_ka[1] /
                                self.sim.root.input.model.unit_001.adsorption.sma_kd[1])
            self.slider_nu.set(self.sim.root.input.model.unit_001.adsorption.sma_nu[1])
            if hasattr(self, "slider_qmax"):
                self.slider_qmax.set(self.sim.root.input.model.unit_001.adsorption.sma_lambda)

        if self.experiment_id.get() <= 0:
            self.slider_par_porosity.configure(state="disabled", button_color="grey", )
            self.slider_film_diffusion.configure(state="disabled", button_color="grey", )
            self.slider_par_diffusion.configure(state="disabled", button_color="grey", )
        else:
            self.slider_par_porosity.configure(state="normal", button_color="#3B8ED0", )
            self.slider_film_diffusion.configure(state="normal", button_color="#3B8ED0", )
            self.slider_par_diffusion.configure(state="normal", button_color="#3B8ED0", )
        if self.experiment_id.get() <= 1:
            self.slider_keq.configure(state="disabled", button_color="grey", )
        else:
            self.slider_keq.configure(state="normal", button_color="#3B8ED0", )
        if self.experiment_id.get() <= 2:
            self.slider_nu.configure(state="disabled", button_color="grey", )
        else:
            self.slider_nu.configure(state="normal", button_color="#3B8ED0", )
        if hasattr(self, "slider_qmax"):
            if self.experiment_id.get() <= 3:
                self.slider_qmax.configure(state="disabled", button_color="grey", )
            else:
                self.slider_qmax.configure(state="normal", button_color="#3B8ED0", )

        self.allow_simulations = True

        self.load_sim_values()
        self.prepare_image()
        self.plot_all_initial()

    def load_sim_values(self):
        self.inlet = self.sim.root.output.solution.unit_001.solution_inlet
        self.output = self.sim.root.output.solution.unit_001.solution_outlet
        self.bulk = self.sim.root.output.solution.unit_001.solution_bulk
        self.solid = self.sim.root.output.solution.unit_001.solution_solid
        self.particle = self.sim.root.output.solution.unit_001.solution_particle
        self.output_times = self.sim.root.output.solution.solution_times

        self.bulk_scaled = ((self.bulk - self.bulk.min(axis=1).min(axis=0))
                            / (self.bulk.max(axis=1).max(axis=0) - self.bulk.min(axis=1).min(axis=0)))

        if self.experiment_id.get() < 2:
            self.bulk_protein_min = min(self.bulk[:, :, 0].flatten())
            self.bulk_protein_max = max(self.bulk[:, :, 0].flatten())
            self.bulk_salt_min = 0
            self.bulk_salt_max = 0.1
        else:
            self.bulk_protein_min = min(self.bulk[:, :, 1].flatten())
            self.bulk_protein_max = max(self.bulk[:, :, 1].flatten())
            self.bulk_salt_min = min(self.bulk[:, :, 0].flatten())
            self.bulk_salt_max = max(self.bulk[:, :, 0].flatten())

        if self.experiment_id.get() == 0 or self.experiment_id.get() == 1:
            self.solid_max = 1e-8
        elif self.experiment_id.get() == 2:
            self.solid_max = max(max(self.solid[:, :, :, 0].flatten()), 1e-8)
        elif self.experiment_id.get() == 3 or self.experiment_id.get() == 4:
            self.solid_max = max(max(self.solid[:, :, :, 1].flatten()), 1e-8)
        elif self.experiment_id.get() == 5:
            self.solid_max = max(max(self.solid[:, :, :, 0].flatten()), 1e-8)

    def plot_all_initial(self):
        for ax in self.axes:
            ax.clear()

        if self.experiment_id.get() <= 2:
            self.ax_inlet_twin.set_yticks([])
            self.ax_outlet_twin.set_yticks([])
            self.ax_bulk_twin.set_xticks([])
        else:
            self.ax_inlet_twin.set_ylabel("Salt conc. [mM]", color="orange")
            self.ax_outlet_twin.set_ylabel("Salt conc. [mM]", color="orange")
            # self.ax_bulk_twin.set_xlabel("Salt conc. [mM]", color="orange")
            self.ax_bulk_twin.set_xticks([])
            self.ax_inlet_twin.yaxis.set_label_coords(1.3, 0.5)
            self.ax_outlet_twin.yaxis.set_label_coords(1.2, 0.5)
            # self.ax_bulk_twin.xaxis.set_label_coords(0.5, 1.1)
            self.ax_inlet_twin.tick_params(axis='y', colors='orange')
            self.ax_outlet_twin.tick_params(axis='y', colors='orange')
            # self.ax_bulk_twin.tick_params(axis='x', colors='orange')

        lines = self.plot_io_initial(self.ax_outlet, self.ax_outlet_twin, 1, self.output, self.output_times)

        self.reference_line = self.ax_outlet.plot([0, 1], [0, 0], ":", color="grey")[0]

        self.outlet_line, self.outlet_twin_line, self.outlet_vline = lines
        lines = self.plot_io_initial(self.ax_inlet, self.ax_inlet_twin, len(self.output_times), self.inlet,
                                     self.output_times)
        self.inlet_line, self.inlet_twin_line, self.inlet_vline = lines
        lines = self.plot_bulk_initial()
        self.bulk_line1, self.bulk_line2, self.bulk_line_twin = lines

        if self.has_prepared_image:
            self.column_image = self.plot_column_initial()
        else:
            self.column_image = None
        self.set_ax_limits()
        # self.fig.tight_layout()

    def set_ax_limits(self):

        if self.experiment_id.get() < 2:
            protein_index = 0
        else:
            protein_index = 1

        set_ax_ylims(self.ax_outlet_twin, self.output[:, 0])
        set_ax_ylims(self.ax_outlet, self.output[:, protein_index])
        self.ax_outlet.set_xlim(self.output_times.min(), self.output_times.max(), auto=False)
        self.ax_inlet.set_xlim(self.output_times.min(), self.output_times.max(), auto=False)
        set_ax_ylims(self.ax_inlet_twin, self.inlet[:, 0])
        set_ax_ylims(self.ax_inlet, self.inlet[:, protein_index])

        self.ax_bulk_twin.set_xlim(self.bulk_salt_max, 0, auto=False)
        self.ax_bulk.set_ylim(self.bulk.shape[1] - 1, 0, auto=False)
        self.ax_bulk.set_xlim(max(self.bulk_protein_max, self.solid_max), 0, auto=False)

        self.ax_inlet.set_title("Inlet")
        self.ax_bulk.set_title("Column")
        self.ax_outlet.set_title("Outlet")

        self.ax_inlet.set_ylabel("Conc. [mM]")
        self.ax_inlet.set_xlabel("Time [s]")
        self.ax_outlet.set_ylabel("Conc. [mM]")
        self.ax_outlet.set_xlabel("Time [s]")
        self.ax_bulk.set_xlabel("Conc. [mM]")
        self.ax_bulk.set_ylabel("Position [mm]")

        self.ax_column.set_xticks([])
        self.ax_column.set_yticks([])
        plt.tight_layout()

        self.draw_all()

    def prepare_image(self):
        if self.experiment_id.get() < 2:
            protein_index = 0
        else:
            protein_index = 1

        # minus_blue = (0.7686274509803921, 0.44313725490196076, 0.1843137254901961)
        minus_blue = (1, 0.7, 0.2)
        if self.experiment_id.get() >= 3:
            bulk_salt = interpolate(self.bulk[:, :, 0], fineness=400)
            bulk_protein = interpolate(self.bulk[:, :, 1], fineness=400)
            bulk_salt = scale(bulk_salt)
            bulk_protein = scale(bulk_protein)

            self.bulk_image_bulk = (1
                                    - np.stack([bulk_salt.T * x for x in [0, 0.4, 1]], axis=-1)
                                    - np.stack([bulk_protein.T * x for x in [1, 1, 0]], axis=-1)
                                    )
        else:
            bulk_protein = interpolate(self.bulk[:, :, 0], fineness=400)
            bulk_protein = scale(bulk_protein)

            self.bulk_image_bulk = (1
                                    - np.stack([bulk_protein.T * x for x in minus_blue], axis=-1)
                                    )

        # Bulk
        if self.experiment_id.get() == 0 or self.experiment_id.get() == 1:
            self.bulk_image_solid = np.zeros_like(self.bulk_image_bulk)
            npar = self.sim.root.input.model.unit_001.discretization.npar
            self.bulk_image_solid = np.stack([self.bulk_image_solid, ] * npar, axis=0)
        else:
            if self.experiment_id.get() == 3 or self.experiment_id.get() == 4:
                solid_protein = interpolate(self.solid[:, :, :, 1], fineness=400)
            else:
                solid_protein = interpolate(self.solid[:, :, :, 0], fineness=400)
            solid_protein = scale(solid_protein, minimum=0)
            self.bulk_image_solid = (0
                                     + np.stack([np.power(solid_protein.T, 1 / 6) * x for x in [0, 1, 0]], axis=-1)
                                     )
        # Particles
        if self.experiment_id.get() == 0:
            self.image_particle = np.ones_like(self.bulk_image_bulk)
            npar = self.sim.root.input.model.unit_001.discretization.npar
            self.image_particle = np.stack([self.image_particle, ] * npar, axis=0)
        elif self.experiment_id.get() == 1:
            particle_protein = interpolate(self.particle[:, :, :, 0], fineness=400)
            particle_protein = scale(particle_protein, maximum=max(self.bulk[:, :, protein_index].flatten()))
            self.image_particle = (1
                                   - np.stack([particle_protein.T * x for x in minus_blue], axis=-1)
                                   )
        elif self.experiment_id.get() == 2:
            particle_protein = interpolate(self.particle[:, :, :, 1], fineness=400)
            particle_protein = scale(particle_protein, maximum=max(self.bulk[:, :, 1].flatten()))
            self.image_particle = (1
                                   - np.stack([particle_protein.T * x for x in minus_blue], axis=-1)
                                   )

        elif self.experiment_id.get() >= 3:
            particle_salt = interpolate(self.particle[:, :, :, 0], fineness=400)
            particle_salt = scale(particle_salt)
            particle_protein = interpolate(self.particle[:, :, :, 1], fineness=400)
            particle_protein = scale(particle_protein, maximum=max(self.bulk[:, :, 1].flatten()))
            self.image_particle = (1
                                   - np.stack([particle_salt.T * x for x in [0, 0.4, 1]], axis=-1)
                                   - np.stack([particle_protein.T * x for x in [1, 1, 0]], axis=-1)
                                   )

        self.bulk_image_solid = np.multiply.outer(self.bulk_image_solid, np.ones(80), ).transpose((2, 1, 4, 0, 3))

        self.image_particle = np.multiply.outer(self.image_particle, np.ones(80), ).transpose((2, 1, 4, 0, 3))

        self.bulk_image_bulk = np.multiply.outer(self.bulk_image_bulk, np.ones(80), ).transpose((1, 0, 3, 2))

        self.load_particle_stocks()
        self.load_particle_mask()
        self.has_prepared_image = True

    def load_particle_stocks(self):
        self.particle_stock = particle_large
        self.particle_shells_stock = particle_shells

    def scale_particle(self, input_array):
        col_porosity = self.sim.root.input.model.unit_001.col_porosity
        size = int((col_porosity * -60 + 60) // 2 * 2)

        resized_array = np.ones((40, 40)) * 255
        shell_resized = cv2.resize(input_array, (size, size))
        offset_left = (40 - size) // 2
        offset_right = 40 - size - offset_left
        if offset_right == 0:
            resized_array = shell_resized[-offset_left:, -offset_left:]
        elif offset_left > 0:
            resized_array[offset_left:-offset_right, offset_left:-offset_right] = shell_resized[:, :]
        else:
            resized_array = shell_resized[-offset_left:offset_right, -offset_left:offset_right]
        return resized_array

    def load_particle_mask(self):
        shells = self.scale_particle(self.particle_shells_stock[:, :, 0])
        solid_color = self.scale_particle(self.particle_stock[:, :, 2])
        liquid_color = self.scale_particle(self.particle_stock[:, :, 0])
        shells = np.concatenate([np.concatenate([shells, ] * 2, axis=1), ] * 10, axis=0)
        solid_color = np.concatenate([np.concatenate([solid_color, ] * 2, axis=1), ] * 10, axis=0)
        liquid_color = np.concatenate([np.concatenate([liquid_color, ] * 2, axis=1), ] * 10, axis=0)

        par_porosity = self.sim.root.input.model.unit_001.par_porosity
        threshold_liquid = (par_porosity - 0.1) / (0.8 - 0.1) * 233 + 20
        threshold_solid = 255 - threshold_liquid + 18

        shell_ranges = {0: (254, 159), 1: (159, 126), 2: (126, 100), 3: (100, 50)}
        self.bulk_image = self.bulk_image_bulk.copy()
        for i, (upper, lower) in shell_ranges.items():
            liquid_mask = (lower <= shells) & (shells < upper) & (0 <= liquid_color) & (liquid_color < threshold_liquid)
            solid_mask = (lower <= shells) & (shells < upper) & (0 <= solid_color) & (solid_color < threshold_solid)
            self.bulk_image[:, liquid_mask, :] = self.image_particle[:, liquid_mask, i, :]
            self.bulk_image[:, solid_mask, :] = self.bulk_image_solid[:, solid_mask, i, :]

        self.bulk_image[self.bulk_image < 0] = 0
        self.bulk_image[self.bulk_image > 1] = 1
