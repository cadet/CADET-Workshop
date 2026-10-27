# -*- coding: utf-8 -*-
"""

A pulse travelling down a column, twice: with and without binding kinetics.

The picture this chapter needs before it starts talking about rates. Two
identical columns, same isotherm, same equilibrium, same everything except how
fast the binding gets there:

    rapid equilibrium   IS_KINETIC = 0, the adsorbed phase is algebraic
    kinetically limited IS_KINETIC = 1 with a deliberately slow rate constant

Each representative particle is drawn as a circle split down the middle:

    left half   pore liquid, filled relative to the feed concentration
    right half  adsorbed phase, hatched, filled relative to its loading in
                equilibrium with the feed

In the upper column both halves move together, because the adsorbed phase is
slaved to the liquid and has no dynamics of its own. In the lower one the left
half still fills almost instantly - the film transfer is fast in both cases -
while the right half visibly lags behind it and is still emptying after the
band has gone.

That contrast is the chapter in one picture: a fast process that is slaved
costs nothing to carry, and only a process with its own time scale is a process
the integrator has to resolve.

Lumped rate model with pores, one component, linear isotherm. The film
coefficient is fast enough that the pore liquid tracks the bulk, but
deliberately not as fast as it could be: see the LRMP cost case study for what
happens to the linear solver past a film rate of about 1e6 / s. Here it is
about 1e3 / s, which is non-limiting without being pathological.

"""

import os

import h5py
import numpy as np
from addict import Dict

BED_LENGTH = 0.1                  # m
COLUMN_RADIUS = 0.005             # m
BED_POROSITY = 0.37
PARTICLE_POROSITY = 0.75
PARTICLE_RADIUS = 4.5e-05         # m
SUPERFICIAL_VELOCITY = 3.7e-04    # m/s, so the band crosses in about 300 s

FILM_DIFFUSION = 1.0e-02          # m/s, film rate about 1.1e3 / s

# Low enough that the band is not visibly diluted on its way down the bed. At
# 2e-7 the peak fell from 0.85 to 0.55 between a fifth and four fifths of the
# column, so the particles filled less and less and the binding looked as if it
# were slowing down. It is not: the band was simply being spread thin. Here the
# peak holds above 0.95 throughout, and the band is still 9 mm wide, about
# eleven cells, so nothing is under-resolved.
AXIAL_DISPERSION = 2.0e-08        # m^2/s

EQUILIBRIUM_CONSTANT = 1.5        # linear isotherm, q = K c
KINETICS_FACTOR = 0.1             # scales k_a and k_d together, so K is kept

FEED = 1.0                        # mol/m^3
INJECTION_TIME = 25.0             # s
END_TIME = 340.0                  # s, the band leaves the bed at about 300 s
N_COL = 120
N_FRAMES = 110
N_PARTICLES = 9                   # representative particles drawn along the bed

CASES = (
    ('rapid', 'rapid equilibrium', dict(is_kinetic=False)),
    ('kinetic', f'kinetically limited, $k_d$ = {KINETICS_FACTOR:g} s$^{{-1}}$',
     dict(is_kinetic=True, kinetics_factor=KINETICS_FACTOR)),
)

WORK = os.path.join('sweeps', 'animation')
VIDEO = os.path.join('data', 'video', 'column_pulse.webm')


def interstitial_velocity():
    """Interstitial velocity in m/s."""
    return SUPERFICIAL_VELOCITY / BED_POROSITY


def film_rate(film_diffusion=FILM_DIFFUSION):
    """Rate of film mass transfer in 1/s."""
    return (3.0 * film_diffusion / PARTICLE_RADIUS
            * (1.0 - BED_POROSITY) / BED_POROSITY)


def convection_rate(n_col=N_COL):
    """Rate of convection across one axial cell in 1/s."""
    return interstitial_velocity() / (BED_LENGTH / n_col)


def binding_rate(kinetics_factor=KINETICS_FACTOR):
    """Rate at which the adsorbed phase relaxes to equilibrium, in 1/s."""
    return kinetics_factor * (EQUILIBRIUM_CONSTANT + 1.0)


def get_model(is_kinetic=False, **kwargs):
    """Build the column, asking for the bulk and particle fields as well.

    The outlet alone is not enough here: the animation needs the concentration
    everywhere inside the bed, the pore liquid and the loading of every
    particle.

    Parameters
    ----------
    is_kinetic : bool
        False puts the linear isotherm in rapid equilibrium, so the adsorbed
        phase is algebraic. True gives it a rate.
    kinetics_factor : float
        Scales k_a and k_d together, so the equilibrium constant, and therefore
        the retention, is the same either way. Only the speed changes.
    """
    cross_section = np.pi * COLUMN_RADIUS ** 2
    end_time = kwargs.get('end_time', END_TIME)
    n_col = int(kwargs.get('n_col', N_COL))
    factor = kwargs.get('kinetics_factor', KINETICS_FACTOR)

    model = Dict()
    model.input.model.nunits = 3
    model.input.model.connections.connections_include_ports = 0
    model.input.model.connections.nswitches = 1

    column = Dict()
    column.unit_type = 'COLUMN_MODEL_1D'
    column.geometry = 'AXIAL_FLOW_CYLINDER'
    column.inlet_cross_section_area = cross_section
    column.cross_section_area = cross_section
    column.bed_length = BED_LENGTH
    column.forward_flow = 1

    column.ncomp = 1
    column.npartype = 1
    column.col_porosity = BED_POROSITY
    column.total_porosity = (BED_POROSITY
                             + (1.0 - BED_POROSITY) * PARTICLE_POROSITY)
    column.col_dispersion = kwargs.get('col_dispersion', AXIAL_DISPERSION)
    column.init_c = [0.0]

    column.discretization.SPATIAL_METHOD = 'FV'
    column.discretization.NCOL = n_col
    column.discretization.RECONSTRUCTION = 'WENO'
    column.discretization.weno.BOUNDARY_MODEL = 0
    column.discretization.weno.WENO_EPS = 1e-10
    column.discretization.weno.WENO_ORDER = 3
    column.discretization.GS_TYPE = 1
    column.discretization.MAX_KRYLOV = 0
    column.discretization.MAX_RESTARTS = 10
    column.discretization.SCHUR_SAFETY = 1.0e-8
    column.discretization.USE_ANALYTIC_JACOBIAN = 1

    particle = column.particle_type_000
    particle.has_film_diffusion = True
    particle.has_pore_diffusion = False
    particle.par_geom = 'SPHERE'
    particle.par_radius = PARTICLE_RADIUS
    particle.par_coreradius = 0.0
    particle.par_porosity = PARTICLE_POROSITY
    particle.film_diffusion = [kwargs.get('film_diffusion', FILM_DIFFUSION)]
    particle.init_cp = [0.0]
    particle.nbound = [1]
    particle.init_cs = [0.0]
    particle.adsorption_model = 'LINEAR'
    particle.adsorption.is_kinetic = 1 if is_kinetic else 0
    # Both constants carry the same factor, so k_a / k_d stays at K and the two
    # columns retain the band equally. Only the approach to equilibrium differs.
    particle.adsorption.LIN_KA = [EQUILIBRIUM_CONSTANT * factor]
    particle.adsorption.LIN_KD = [1.0 * factor]

    model.input.model.unit_001 = column

    flow_rate = SUPERFICIAL_VELOCITY * cross_section
    model.input.model.connections.switch_000.connections = [
        0.0, 1.0, -1.0, -1.0, flow_rate,
        1.0, 2.0, -1.0, -1.0, flow_rate,
    ]
    model.input.model.connections.switch_000.section = 0

    model.input.model.solver.gs_type = 1
    model.input.model.solver.max_krylov = 0
    model.input.model.solver.max_restarts = 10
    model.input.model.solver.schur_safety = 1e-08

    model.input.solver.consistent_init_mode = 1
    model.input.solver.consistent_init_mode_sens = 3
    model.input.solver.nthreads = 1
    model.input.solver.sections.nsec = 2
    model.input.solver.sections.section_continuity = [0]
    model.input.solver.sections.section_times = [0.0, INJECTION_TIME, end_time]
    model.input.solver.time_integrator.ABSTOL = kwargs.get('idas_abstol', 1e-10)
    model.input.solver.time_integrator.ALGTOL = kwargs.get('idas_algtol', 1e-8)
    model.input.solver.time_integrator.RELTOL = kwargs.get('idas_reltol', 1e-8)
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 1000000

    # Omitting USER_SOLUTION_TIMES makes CADET report its own internal steps
    # instead of interpolating onto a grid we chose. That is the only way to
    # see what the integrator actually computed rather than what it can
    # reconstruct afterwards.
    if kwargs.get('solution_times') is not None:
        model.input.solver.user_solution_times = np.asarray(
            kwargs['solution_times'], dtype=float)
    elif kwargs.get('user_times', True):
        model.input.solver.user_solution_times = np.linspace(
            0.0, end_time, int(kwargs.get('n_frames', N_FRAMES)))

    inlet = model.input.model.unit_000
    inlet.UNIT_TYPE = 'INLET'
    inlet.inlet_type = 'PIECEWISE_CUBIC_POLY'
    inlet.ncomp = 1
    inlet.sec_000.const_coeff = [FEED]
    inlet.sec_000.lin_coeff = [0.0]
    inlet.sec_000.quad_coeff = [0.0]
    inlet.sec_000.cube_coeff = [0.0]
    inlet.sec_001.const_coeff = [0.0]
    inlet.sec_001.lin_coeff = [0.0]
    inlet.sec_001.quad_coeff = [0.0]
    inlet.sec_001.cube_coeff = [0.0]

    model.input.model.unit_002.UNIT_TYPE = 'OUTLET'
    model.input.model.unit_002.ncomp = 1

    model.input['return'].split_components_data = 0
    model.input['return'].split_ports_data = 0
    model.input['return'].write_solution_times = 1
    model.input['return'].unit_001.write_solution_outlet = 1
    model.input['return'].unit_001.write_solution_bulk = 1
    model.input['return'].unit_001.write_solution_particle = 1
    model.input['return'].unit_001.write_solution_solid = 1
    model.input['return'].write_idas_meta = 1

    return model


def simulate(name='rapid', force=False, **settings):
    """Run one of the two columns, or load it from disk.

    Returns
    -------
    dict
        times, bulk, particle (pore liquid), solid (adsorbed) and outlet.
    """
    from cadet import Cadet

    import utility.cadet_runtime as cadet_runtime

    filename = os.path.join(WORK, f'{name}.h5')

    if force or not os.path.exists(filename):
        cadet_runtime.configure()
        os.makedirs(WORK, exist_ok=True)

        model = Cadet()
        model.root = get_model(**settings)
        model.filename = filename
        model.save()

        info = model.run_simulation()
        if info.return_code:
            raise RuntimeError(f'simulation failed: {info.error_message}')

    with h5py.File(filename, 'r') as handle:
        group = handle['output/solution/unit_001']
        names = list(group)

        def field(prefix):
            matches = [name for name in names if name.startswith(prefix)]
            if not matches:
                raise KeyError(f'{prefix} not in {names}')

            return np.squeeze(group[matches[0]][()])

        return {
            'time': handle['output/solution/SOLUTION_TIMES'][()],
            'bulk': field('SOLUTION_BULK'),
            'particle': field('SOLUTION_PARTICLE'),
            'solid': field('SOLUTION_SOLID'),
            'outlet': field('SOLUTION_OUTLET'),
        }


def _colormap():
    """White to the chapter's blue, for the bulk liquid."""
    from matplotlib.colors import LinearSegmentedColormap

    from utility.stiffness_metrics import COLORS

    return LinearSegmentedColormap.from_list('bulk', ['#FFFFFF', COLORS[0]])


def metrics():
    """What the time integrator did in each of the two columns.

    Both runs solve the same transport on the same grid to the same tolerances.
    The only difference is whether the adsorbed phase is an algebraic equation
    or an ordinary differential one, so anything the integrator does
    differently is down to that.

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    import utility.stiffness_metrics as stiffness_metrics
    import utility.stiffness_study as study_tools

    rows = []
    for name, label, settings in CASES:
        filename = os.path.join(WORK, f'{name}.h5')
        if not os.path.exists(filename):
            simulate(name, **settings)

        measured = stiffness_metrics.measure(filename, label=name)
        rows.append({
            'column': label,
            'binding': ('algebraic (DAE)' if not settings.get('is_kinetic')
                        else f"rate {binding_rate(settings['kinetics_factor']):.3g} / s"),
            'BDF steps': measured.n_steps,
            'mean BDF order': round(study_tools.mean_bdf_order(measured), 2),
            'error test failures': measured.n_error_test_failures,
            'compute time / s': round(measured.compute_time, 3),
        })

    return pd.DataFrame(rows)


def _build_figure(runs, scale):
    """Lay the five panels out and return the artists the frames update.

    The axes are positioned first and the particle radius is derived from the
    shape of the box afterwards. Doing it the other way round means either
    ellipses, if the box is wide, or circles clipped top and bottom, if it is
    short.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Circle, Wedge

    from utility.stiffness_metrics import COLORS, style_axis

    time = runs[CASES[0][0]]['time']
    n_col = runs[CASES[0][0]]['bulk'].shape[1]
    positions = (np.arange(n_col) + 0.5) * BED_LENGTH / n_col
    sample = np.linspace(0, n_col - 1, N_PARTICLES).astype(int)

    figure = Figure(figsize=(16.0, 9.0))
    FigureCanvasAgg(figure)
    axes = figure.subplots(
        5, 1,
        gridspec_kw={'height_ratios': [0.55, 1.25, 0.55, 1.25, 1.15],
                     'hspace': 0.75})

    # ---- first pass: the panels themselves ----
    strips, particle_rows = [], []

    for case, (name, label, _) in enumerate(CASES):
        strip_axis = axes[2 * case]
        particle_axis = axes[2 * case + 1]

        strips.append(strip_axis.imshow(
            runs[name]['bulk'][0][None, :], aspect='auto', cmap=_colormap(),
            vmin=0.0, vmax=FEED, origin='lower',
            extent=(0.0, BED_LENGTH, 0.0, 1.0), interpolation='bilinear'))
        strip_axis.set_yticks([])
        strip_axis.set_xticks([])
        strip_axis.set_ylabel('bulk', rotation=0, ha='right', va='center',
                              fontsize=9 * scale)
        strip_axis.set_title(label, loc='left', fontsize=11 * scale)
        for spine in strip_axis.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(1.2 * scale)

        particle_axis.set_xlim(0.0, BED_LENGTH)
        particle_axis.set_yticks([])
        particle_axis.set_ylabel('particles', rotation=0, ha='right',
                                 va='center', fontsize=9 * scale)
        particle_axis.tick_params(labelsize=9 * scale)
        if case == len(CASES) - 1:
            particle_axis.set_xlabel('axial position / m', fontsize=10 * scale)
        else:
            particle_axis.set_xticklabels([])
        for spine in ('top', 'right', 'left'):
            particle_axis.spines[spine].set_visible(False)
        particle_rows.append(particle_axis)

    for case, (name, label, _) in enumerate(CASES):
        axes[4].plot(time, runs[name]['outlet'],
                     color=COLORS[0] if case == 0 else COLORS[1],
                     linewidth=1.8 * scale,
                     linestyle='-' if case == 0 else '--', label=label)
    cursor = axes[4].axvline(time[0], color='#555555', linewidth=1.3 * scale)
    axes[4].set_xlim(time[0], time[-1])
    axes[4].set_xlabel('time / s', fontsize=10 * scale)
    axes[4].set_ylabel('outlet', rotation=0, ha='right', va='center',
                       fontsize=9 * scale)
    axes[4].tick_params(labelsize=9 * scale)
    axes[4].legend(frameon=False, fontsize=9 * scale, loc='upper left')
    style_axis(axes[4])

    figure.text(0.5, 0.972,
                'left half: pore liquid          '
                'right half: adsorbed (hatched)',
                ha='center', va='center', fontsize=10 * scale,
                color='#444444')
    clock = figure.text(0.015, 0.972, '', ha='left', va='center',
                        fontsize=11 * scale)
    figure.subplots_adjust(top=0.90, left=0.085, right=0.985, bottom=0.07)

    # ---- the box is now fixed, so a circle can be sized to fit it ----
    width_inches, height_inches = figure.get_size_inches()
    box = particle_rows[0].get_position()
    half = 0.5 * BED_LENGTH * ((box.height * height_inches)
                               / (box.width * width_inches))
    radius = min(0.85 * BED_LENGTH / (2 * N_PARTICLES), 0.94 * half)

    # ---- second pass: the particles ----
    liquids, solids = [], []

    for particle_axis in particle_rows:
        particle_axis.set_ylim(-half, half)

        case_liquid, case_solid = [], []
        for index in sample:
            centre = (positions[index], 0.0)
            particle_axis.add_patch(Circle(
                centre, radius, facecolor='white', edgecolor='#555555',
                linewidth=1.1 * scale, zorder=2))

            liquid = Wedge(centre, 1e-9, 90, 270, facecolor=COLORS[0],
                           edgecolor='none', zorder=3)
            solid = Wedge(centre, 1e-9, -90, 90, facecolor=COLORS[1],
                          alpha=0.9, edgecolor='#7A3B06', hatch='////',
                          linewidth=0.0, zorder=3)
            particle_axis.add_patch(liquid)
            particle_axis.add_patch(solid)
            case_liquid.append(liquid)
            case_solid.append(solid)

            particle_axis.plot([positions[index]] * 2, [-radius, radius],
                               color='#555555', linewidth=0.9 * scale,
                               zorder=4)

        liquids.append(case_liquid)
        solids.append(case_solid)

    return figure, dict(strips=strips, liquids=liquids, solids=solids,
                        cursor=cursor, clock=clock, sample=sample,
                        radius=radius, time=time)


def make_animation(path=VIDEO, fps=14, width=3840, height=2160, dpi=240,
                   force=False):
    """Render both columns as a 4K video.

    Written as VP9 in a WebM container, through the ffmpeg that OpenCV already
    ships. H.264 is not available here - OpenCV's avc1 encoder needs an
    openh264 library that is not installed - and a GIF cannot carry this
    resolution at any sensible size. WebM plays natively in every current
    browser, so the notebook can show it inline.

    Returns
    -------
    str
        The path the video was written to.
    """
    import cv2

    runs = {name: simulate(name, force=force, **settings)
            for name, _, settings in CASES}

    figure, art = _build_figure(runs, scale=width / dpi / 9.5)
    figure.set_dpi(dpi)
    canvas = figure.canvas

    time = art['time']
    sample, radius = art['sample'], art['radius']
    liquid_reference = FEED
    solid_reference = EQUILIBRIUM_CONSTANT * FEED

    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'VP90'), fps,
                             (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'could not open a VP9 encoder for {path}')

    try:
        for frame in range(len(time)):
            for case, (name, _, _) in enumerate(CASES):
                run = runs[name]
                art['strips'][case].set_data(run['bulk'][frame][None, :])

                for position, index in enumerate(sample):
                    pore = min(run['particle'][frame, index]
                               / liquid_reference, 1.0)
                    bound = min(run['solid'][frame, index]
                                / solid_reference, 1.0)
                    # Area proportional to content, so a full half means "in
                    # equilibrium with the feed".
                    art['liquids'][case][position].set_radius(
                        radius * np.sqrt(max(pore, 0.0)))
                    art['solids'][case][position].set_radius(
                        radius * np.sqrt(max(bound, 0.0)))

            art['cursor'].set_xdata([time[frame], time[frame]])
            art['clock'].set_text(f't = {time[frame]:.0f} s')

            canvas.draw()
            image = np.asarray(canvas.buffer_rgba())[:, :, :3]
            if image.shape[:2] != (height, width):
                raise RuntimeError(
                    f'canvas is {image.shape[1]}x{image.shape[0]}, '
                    f'expected {width}x{height}')
            writer.write(image[:, :, ::-1])            # matplotlib RGB -> BGR
    finally:
        writer.release()

    return path


FILM_SWEEP = (1e-04, 1e-03, 1e-02, 1e-01, 1e00)


def film_sweep(coefficients=FILM_SWEEP):
    """Make the fast process faster and see what it costs.

    The honest way to ask whether stiffness is expensive. Slowing a process
    down changes the answer, so a slow-against-fast comparison measures the
    solution as much as the spectrum. Speeding it up does not: past the point
    where the film is no longer limiting, the chromatogram stops moving while
    the eigenvalue keeps growing, and whatever the step count does from there
    is attributable to the spectrum alone.

    All runs are in rapid equilibrium, so the binding contributes nothing that
    changes between them.

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    import utility.stiffness_metrics as stiffness_metrics
    import utility.stiffness_study as study_tools

    runs = {}
    for coefficient in coefficients:
        name = f'film_{coefficient:.0e}'
        runs[coefficient] = (
            simulate(name, is_kinetic=False, film_diffusion=coefficient),
            os.path.join(WORK, f'{name}.h5'))

    limit = runs[coefficients[-1]][0]['outlet']

    rows = []
    for coefficient in coefficients:
        data, filename = runs[coefficient]
        measured = stiffness_metrics.measure(filename, label=str(coefficient))
        rows.append({
            'k_f / m s^-1': f'{coefficient:.0e}',
            'film rate / s^-1': f'{film_rate(coefficient):.3g}',
            'vs the fastest run': f'{np.abs(data["outlet"] - limit).max():.1e}',
            'BDF steps': measured.n_steps,
            'mean BDF order': round(study_tools.mean_bdf_order(measured), 2),
            'compute time / s': round(measured.compute_time, 3),
        })

    return pd.DataFrame(rows)

TOLERANCE_CASES = (
    ('coarse', 'coarse tolerance, RELTOL $10^{-3}$',
     dict(idas_reltol=1e-03, idas_abstol=1e-05)),
    ('fine', 'fine tolerance, RELTOL $10^{-8}$',
     dict(idas_reltol=1e-08, idas_abstol=1e-10)),
)

TOLERANCE_VIDEO = os.path.join('data', 'video', 'column_tolerance.webm')

# Bottom of the particle colour scale, as a fraction of the loading in
# equilibrium with the feed. Everything fainter than this is drawn as
# empty. The scale is logarithmic from here up: on a linear one roughly
# half of a particle's history is indistinguishable from white, and that
# half is exactly the approach and the release that the coarse run steps
# over.
COLOUR_FLOOR = 1.0e-05


def explicit_step_limit():
    """Largest step an explicit Euler could take here, in s.

    Set by the fastest eigenvalue in the model, which is the film transfer:
    stability needs |1 + h lambda| <= 1, so h <= 2 / |lambda|. Nothing about
    the answer being computed requires steps anywhere near that small.
    """
    return 2.0 / film_rate()


def _hold(step_times, values, frame_times):
    """Value last *computed* at or before each frame time.

    Zero order hold on purpose. Between its steps the integrator has not
    evaluated anything, and the point of this video is to show what was
    computed rather than what can be reconstructed from it afterwards.

    Returns
    -------
    (np.array, np.array)
        The held values, and the index of the step they came from.
    """
    index = np.clip(np.searchsorted(step_times, frame_times, side='right') - 1,
                    0, len(step_times) - 1)

    return values[index], index


def _window(run, low=0.30, high=0.78):
    """The stretch of time in which the band crosses the middle of the bed."""
    n_col = run['bulk'].shape[1]
    centres = np.array([np.argmax(run['bulk'][i])
                        for i in range(len(run['time']))])
    inside = run['bulk'].max(axis=1) > 0.05 * FEED

    start = run['time'][inside & (centres > low * n_col)][0]
    stop = run['time'][inside & (centres < high * n_col)][-1]

    return start, stop


STEP_RULER_WINDOW = 5.0           # s either side of the current time


def _tick_data(times, low, high, centre, height=0.34):
    """Vertical tick marks at `times`, as one polyline with gaps."""
    selected = times[(times >= low) & (times <= high)]

    x = np.repeat(selected, 3).astype(float)
    x[2::3] = np.nan
    y = np.tile([centre - height, centre + height, np.nan], len(selected))

    return x, y, len(selected)


def make_tolerance_animation(path=TOLERANCE_VIDEO, fps=10, n_frames=270,
                             width=3840, height=2160, dpi=240, force=False):
    """The same column twice, integrated to coarse and to fine tolerances.

    The two rows look the same, and that is the result rather than a failure of
    the drawing: both runs are accurate, so their particles carry the same
    concentrations to within 0.3 % of the colour scale. An error controller
    exists precisely to stop consecutive steps from jumping, so no accurate
    integration can be made to look chunky.

    What does differ is *when each integrator had to stop and work*, and that is
    what the step ruler shows: one tick per actual time step inside a sliding
    ten second window. The coarse run places about eleven, the fine one about
    seventy, and an explicit method would have no say at all - stability alone
    caps it at `explicit_step_limit()`, some 1.8 ms, which is thousands of
    ticks in the same window and renders as a solid bar.

    Returns
    -------
    str
    """
    import cv2
    import matplotlib.cm
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.patches import Circle, Rectangle

    from utility.stiffness_metrics import COLORS, style_axis

    runs = {name: simulate(name, force=force, is_kinetic=False,
                           user_times=False, **settings)
            for name, _, settings in TOLERANCE_CASES}

    solid_reference = EQUILIBRIUM_CONSTANT * FEED
    colormap = LinearSegmentedColormap.from_list(
        'loading', ['#FFFFFF', COLORS[1]])
    norm = LogNorm(vmin=COLOUR_FLOOR, vmax=1.0)

    start, stop = _window(runs['fine'])
    frame_times = np.linspace(start, stop, n_frames)

    n_col = runs['fine']['bulk'].shape[1]
    positions = (np.arange(n_col) + 0.5) * BED_LENGTH / n_col
    sample = np.linspace(0, n_col - 1, N_PARTICLES).astype(int)
    scale = width / dpi / 9.5

    figure = Figure(figsize=(16.0, 9.0))
    FigureCanvasAgg(figure)
    figure.set_dpi(dpi)
    axes = figure.subplots(
        6, 1,
        gridspec_kw={'height_ratios': [0.45, 1.2, 0.45, 1.2, 1.1, 0.85],
                     'hspace': 0.72})

    strips, titles, particle_rows = [], [], []

    for case, (name, label, _) in enumerate(TOLERANCE_CASES):
        strip_axis, particle_axis = axes[2 * case], axes[2 * case + 1]

        strips.append(strip_axis.imshow(
            runs[name]['bulk'][0][None, :], aspect='auto', cmap=_colormap(),
            vmin=0.0, vmax=FEED, origin='lower',
            extent=(0.0, BED_LENGTH, 0.0, 1.0), interpolation='bilinear'))
        strip_axis.set_yticks([])
        strip_axis.set_xticks([])
        strip_axis.set_ylabel('bulk', rotation=0, ha='right', va='center',
                              fontsize=9 * scale)
        titles.append(strip_axis.set_title(label, loc='left',
                                           fontsize=11 * scale))
        for spine in strip_axis.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(1.2 * scale)

        particle_axis.set_xlim(0.0, BED_LENGTH)
        particle_axis.set_yticks([])
        particle_axis.set_ylabel('particles', rotation=0, ha='right',
                                 va='center', fontsize=9 * scale)
        particle_axis.tick_params(labelsize=9 * scale)
        particle_axis.set_xticklabels([])
        for spine in ('top', 'right', 'left'):
            particle_axis.spines[spine].set_visible(False)
        particle_rows.append(particle_axis)

    # ---- the step ruler: one tick per time step actually taken ----
    ruler = axes[4]
    lanes = {}
    for lane, (name, _, _) in enumerate(TOLERANCE_CASES):
        centre = 2.0 - lane
        line, = ruler.plot([], [], color=COLORS[lane], linewidth=1.1 * scale,
                           solid_capstyle='butt')
        lanes[name] = (line, centre)

    explicit = explicit_step_limit()
    explicit_ticks = int(2 * STEP_RULER_WINDOW / explicit)
    ruler.add_patch(Rectangle(
        (0.0, -0.34), 1.0, 0.68, transform=ruler.get_yaxis_transform(),
        facecolor='#333333', edgecolor='none'))
    ruler.annotate(
        f'one every {1e3 * explicit:.1f} ms - stability, not accuracy',
        xy=(0.99, 0.4 / 3.2), xycoords='axes fraction', ha='right',
        va='center', fontsize=9 * scale, color='white')

    ruler.set_ylim(-0.6, 2.6)
    ruler.set_yticks([2.0, 1.0, 0.0])
    ruler.set_yticklabels(['coarse', 'fine', 'explicit'],
                          fontsize=9 * scale)
    ruler.set_xlabel('simulated time / s', fontsize=10 * scale)
    ruler.tick_params(labelsize=9 * scale)
    ruler_title = ruler.set_title('', loc='left', fontsize=11 * scale)
    for spine in ('top', 'right', 'left'):
        ruler.spines[spine].set_visible(False)

    # ---- the chromatograms ----
    for case, (name, label, _) in enumerate(TOLERANCE_CASES):
        axes[5].plot(runs[name]['time'], runs[name]['outlet'],
                     color=COLORS[0] if case == 0 else COLORS[1],
                     linewidth=(3.0 if case == 0 else 1.4) * scale,
                     linestyle='-' if case == 0 else '--',
                     alpha=0.45 if case == 0 else 1.0, label=label)
    cursor = axes[5].axvline(frame_times[0], color='#555555',
                             linewidth=1.3 * scale)
    axes[5].axvspan(start, stop, color='#999999', alpha=0.13)
    axes[5].set_xlim(runs['fine']['time'][0], runs['fine']['time'][-1])
    axes[5].set_xlabel('time / s', fontsize=10 * scale)
    axes[5].set_ylabel('outlet', rotation=0, ha='right', va='center',
                       fontsize=9 * scale)
    axes[5].tick_params(labelsize=9 * scale)
    axes[5].legend(frameon=False, fontsize=9 * scale, loc='upper left')
    style_axis(axes[5])

    figure.text(0.38, 0.975,
                'particle colour: adsorbed concentration, log scale',
                ha='center', va='center', fontsize=10 * scale, color='#444444')
    clock = figure.text(0.015, 0.975, '', ha='left', va='center',
                        fontsize=11 * scale)
    figure.subplots_adjust(top=0.905, left=0.105, right=0.985, bottom=0.06)

    bar_axis = figure.add_axes([0.655, 0.966, 0.215, 0.014])
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap)
    bar = figure.colorbar(mappable, cax=bar_axis, orientation='horizontal')
    bar.set_ticks([1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1.0])
    bar.set_ticklabels(['$10^{-5}$', '', '$10^{-3}$', '', '$10^{-1}$',
                        'feed eq.'])
    bar_axis.minorticks_off()
    bar_axis.tick_params(labelsize=8 * scale, length=2 * scale)

    width_inches, height_inches = figure.get_size_inches()
    box = particle_rows[0].get_position()
    half = 0.5 * BED_LENGTH * ((box.height * height_inches)
                               / (box.width * width_inches))
    radius = min(0.85 * BED_LENGTH / (2 * N_PARTICLES), 0.94 * half)

    circles = []
    for particle_axis in particle_rows:
        particle_axis.set_ylim(-half, half)
        row = []
        for index in sample:
            patch = Circle((positions[index], 0.0), radius,
                           facecolor='white', edgecolor='#555555',
                           linewidth=1.1 * scale, zorder=2)
            particle_axis.add_patch(patch)
            row.append(patch)
        circles.append(row)

    held = {}
    for name, _, _ in TOLERANCE_CASES:
        run = runs[name]
        bulk, taken = _hold(run['time'], run['bulk'], frame_times)
        solid, _ = _hold(run['time'], run['solid'], frame_times)
        held[name] = {'bulk': bulk, 'solid': solid, 'steps': taken}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'VP90'), fps,
                             (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'could not open a VP9 encoder for {path}')

    try:
        for frame in range(n_frames):
            now = frame_times[frame]
            low, high = now - STEP_RULER_WINDOW, now + STEP_RULER_WINDOW
            visible = {}

            for case, (name, label, _) in enumerate(TOLERANCE_CASES):
                data = held[name]
                strips[case].set_data(data['bulk'][frame][None, :])
                for position, index in enumerate(sample):
                    loading = data['solid'][frame, index] / solid_reference
                    circles[case][position].set_facecolor(colormap(norm(
                        float(np.clip(loading, COLOUR_FLOOR, 1.0)))))

                taken = int(data['steps'][frame] - data['steps'][0])
                titles[case].set_text(f'{label}     -     {taken} steps so far')

                line, centre = lanes[name]
                x, y, n_ticks = _tick_data(runs[name]['time'], low, high,
                                           centre)
                line.set_data(x, y)
                visible[name] = n_ticks

            # The counts go in the title, where they sit on one line and
            # can be read against each other.
            ruler_title.set_text(
                'time steps taken in this 10 s window:        '
                f'coarse {visible["coarse"]},        '
                f'fine {visible["fine"]},        '
                f'explicit Euler {explicit_ticks:,}'.replace(',', ' '))
            ruler.set_xlim(low, high)
            cursor.set_xdata([now] * 2)
            clock.set_text(f't = {now:.1f} s')

            figure.canvas.draw()
            image = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
            if image.shape[:2] != (height, width):
                raise RuntimeError(
                    f'canvas is {image.shape[1]}x{image.shape[0]}, '
                    f'expected {width}x{height}')
            writer.write(image[:, :, ::-1])
    finally:
        writer.release()

    return path

SCALES_VIDEO = os.path.join('data', 'video', 'column_scales.webm')

# The three clocks, in seconds. Each panel plays its own window over the same
# fraction of the video, so the top runs the whole separation while the bottom
# runs ten milliseconds of it.
CLOCKS = (0.010, 2.0, END_TIME)


def _detail_grid():
    """Output times dense enough to resolve all three clocks at once."""
    return np.unique(np.concatenate([
        np.linspace(0.0, CLOCKS[0], 320),
        np.linspace(CLOCKS[0], CLOCKS[1], 320),
        np.linspace(CLOCKS[1], CLOCKS[2], 420)]))


def make_scales_animation(path=SCALES_VIDEO, fps=12, n_frames=240,
                          width=3840, height=2160, dpi=240, force=False):
    """One ordinary run, played on three clocks at once.

    Nothing here is set up to make a point: an empty column, the feed switched
    on at t = 0. The inlet step is a real discontinuity, so the film in the
    first cells is genuinely excited, and the three panels show the same run at

        10 ms   the pore liquid catching up with the bulk
        2 s     the entrance of the bed filling by convection
        340 s   the band crossing the column

    all advancing over the same fraction of the video. The bottom panel is the
    integrator's own step size against the largest step an explicit method
    could take, which is fixed by stability at about 1.8 ms whatever the
    accuracy asked for.

    Returns
    -------
    str
    """
    import cv2
    import matplotlib.cm
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.patches import Circle

    from utility.stiffness_metrics import COLORS, style_axis

    detail = simulate('scales_detail', force=force, is_kinetic=False,
                      solution_times=_detail_grid())
    steps = simulate('scales_steps', force=force, is_kinetic=False,
                     user_times=False)

    time = detail['time']
    bulk, pore, solid = detail['bulk'], detail['particle'], detail['solid']
    n_col = bulk.shape[1]
    positions = (np.arange(n_col) + 0.5) * BED_LENGTH / n_col
    sample = np.linspace(0, n_col - 1, N_PARTICLES).astype(int)

    step_times = steps['time'][1:]
    step_sizes = np.diff(steps['time'])
    cap = explicit_step_limit()

    solid_reference = EQUILIBRIUM_CONSTANT * FEED
    colormap = LinearSegmentedColormap.from_list(
        'loading', ['#FFFFFF', COLORS[1]])
    norm = LogNorm(vmin=COLOUR_FLOOR, vmax=1.0)
    scale = width / dpi / 9.5

    figure = Figure(figsize=(16.0, 9.0))
    FigureCanvasAgg(figure)
    figure.set_dpi(dpi)
    axes = figure.subplots(
        5, 1,
        gridspec_kw={'height_ratios': [0.42, 1.0, 1.05, 1.05, 1.05],
                     'hspace': 1.05})

    # ---- 1 and 2: the column on the slowest clock ----
    strip = axes[0].imshow(bulk[0][None, :], aspect='auto', cmap=_colormap(),
                           vmin=0.0, vmax=FEED, origin='lower',
                           extent=(0.0, BED_LENGTH, 0.0, 1.0),
                           interpolation='bilinear')
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[0].set_ylabel('bulk', rotation=0, ha='right', va='center',
                       fontsize=9 * scale)
    column_title = axes[0].set_title('', loc='left', fontsize=11.5 * scale)
    for spine in axes[0].spines.values():
        spine.set_edgecolor('#555555')
        spine.set_linewidth(1.2 * scale)

    axes[1].set_xlim(0.0, BED_LENGTH)
    axes[1].set_yticks([])
    axes[1].set_ylabel('particles', rotation=0, ha='right', va='center',
                       fontsize=9 * scale)
    axes[1].tick_params(labelsize=9 * scale)
    for spine in ('top', 'right', 'left'):
        axes[1].spines[spine].set_visible(False)

    # ---- 3 and 4: the inlet cell on the two fast clocks ----
    fast_axes, fast_lines = [], []
    for panel, (axis, span) in enumerate(zip(axes[2:4], CLOCKS[1::-1])):
        liquid, = axis.plot([], [], color=COLORS[0], linewidth=2.4 * scale,
                            label='bulk liquid, first cell')
        inside, = axis.plot([], [], color=COLORS[1], linewidth=2.4 * scale,
                            linestyle='--', label='pore liquid, same cell')
        axis.set_xlim(0.0, span)
        axis.set_ylabel(r'c / mol m$^{-3}$', fontsize=9 * scale)
        axis.tick_params(labelsize=9 * scale)
        axis.legend(frameon=False, fontsize=9 * scale, loc='upper left')
        style_axis(axis)
        fast_axes.append(axis)
        fast_lines.append((liquid, inside))

    fast_axes[0].set_xlabel('time / s', fontsize=10 * scale)
    fast_axes[1].set_xlabel('time / ms', fontsize=10 * scale)
    fast_titles = [axis.set_title('', loc='left', fontsize=11.5 * scale)
                   for axis in fast_axes]

    # The faster window, marked inside the slower one.
    fast_axes[0].axvspan(0.0, CLOCKS[0], color=COLORS[1], alpha=0.16)

    # ---- 5: the integrator's step against the explicit cap ----
    seismo = axes[4]
    taken, = seismo.plot([], [], color=COLORS[0], linewidth=1.8 * scale,
                         label='step the implicit method took')
    seismo.axhline(cap, color='#333333', linewidth=2.0 * scale,
                   linestyle='--',
                   label=f'largest step an explicit method could take, '
                         f'{1e3 * cap:.1f} ms')
    seismo.set_yscale('log')
    seismo.set_xlim(0.0, CLOCKS[2])
    seismo.set_ylim(1e-04, 1.0)
    seismo.set_xlabel('time / s', fontsize=10 * scale)
    seismo.set_ylabel('step / s', fontsize=9 * scale)
    seismo.tick_params(labelsize=9 * scale)
    seismo.legend(fontsize=9 * scale, loc='upper right', frameon=True,
                  facecolor='white', framealpha=0.92, edgecolor='none')
    seismo_title = seismo.set_title('', loc='left', fontsize=11.5 * scale)
    style_axis(seismo)

    clock = figure.text(0.015, 0.975, '', ha='left', va='center',
                        fontsize=11 * scale)
    figure.text(0.42, 0.975,
                'one ordinary run, three clocks, all advancing together',
                ha='center', va='center', fontsize=10.5 * scale,
                color='#444444')
    figure.subplots_adjust(top=0.895, left=0.085, right=0.985, bottom=0.075)

    bar_axis = figure.add_axes([0.80, 0.966, 0.17, 0.013])
    bar = figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap),
        cax=bar_axis, orientation='horizontal')
    bar.set_ticks([1e-05, 1e-03, 1e-01, 1.0])
    bar.set_ticklabels(['$10^{-5}$', '$10^{-3}$', '$10^{-1}$', 'feed eq.'])
    bar_axis.minorticks_off()
    bar_axis.tick_params(labelsize=8 * scale, length=2 * scale)

    width_inches, height_inches = figure.get_size_inches()
    box = axes[1].get_position()
    half = 0.5 * BED_LENGTH * ((box.height * height_inches)
                               / (box.width * width_inches))
    radius = min(0.85 * BED_LENGTH / (2 * N_PARTICLES), 0.94 * half)
    axes[1].set_ylim(-half, half)

    circles = []
    for index in sample:
        patch = Circle((positions[index], 0.0), radius, facecolor='white',
                       edgecolor='#555555', linewidth=1.1 * scale, zorder=2)
        axes[1].add_patch(patch)
        circles.append(patch)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'VP90'), fps,
                             (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'could not open a VP9 encoder for {path}')

    try:
        for frame in range(n_frames):
            fraction = (frame + 1) / n_frames
            slow, middle, fast = (fraction * span for span in CLOCKS[::-1])

            index = int(np.searchsorted(time, slow, side='right') - 1)
            strip.set_data(bulk[index][None, :])
            for patch, node in zip(circles, sample):
                loading = solid[index, node] / solid_reference
                patch.set_facecolor(colormap(norm(
                    float(np.clip(loading, COLOUR_FLOOR, 1.0)))))
            column_title.set_text(
                f'the separation            t = {slow:5.1f} s '
                f'of {CLOCKS[2]:.0f}')

            for panel, (span, upto) in enumerate(zip(CLOCKS[1::-1],
                                                     (middle, fast))):
                shown = time <= upto
                factor = 1.0 if panel == 0 else 1e3
                liquid, inside = fast_lines[panel]
                liquid.set_data(time[shown] * factor, bulk[shown, 0])
                inside.set_data(time[shown] * factor, pore[shown, 0])
                fast_axes[panel].set_xlim(0.0, span * factor)
                axis_top = max(bulk[shown, 0].max(), 1e-09) * 1.25
                fast_axes[panel].set_ylim(0.0, axis_top)

            fast_titles[0].set_text(
                f'the bed entrance filling            t = {middle:4.2f} s '
                f'of {CLOCKS[1]:.0f}')
            fast_titles[1].set_text(
                f'the film equilibrating            t = {1e3 * fast:5.2f} ms '
                f'of {1e3 * CLOCKS[0]:.0f}')

            visible = step_times <= slow
            taken.set_data(step_times[visible], step_sizes[visible])
            seismo_title.set_text(
                'what the integrator could do about it            '
                f'{int(visible.sum())} steps so far, '
                f'an explicit method would need '
                f'{int(slow / cap):,}'.replace(',', ' '))

            clock.set_text(f'{100 * fraction:3.0f} %')

            figure.canvas.draw()
            image = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
            if image.shape[:2] != (height, width):
                raise RuntimeError(
                    f'canvas is {image.shape[1]}x{image.shape[0]}, '
                    f'expected {width}x{height}')
            writer.write(image[:, :, ::-1])
    finally:
        writer.release()

    return path
