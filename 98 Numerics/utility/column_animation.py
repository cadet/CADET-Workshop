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
    model.input.solver.time_integrator.ABSTOL = 1e-10
    model.input.solver.time_integrator.ALGTOL = 1e-8
    model.input.solver.time_integrator.RELTOL = 1e-8
    model.input.solver.time_integrator.INIT_STEP_SIZE = 1e-10
    model.input.solver.time_integrator.MAX_STEPS = 1000000
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
