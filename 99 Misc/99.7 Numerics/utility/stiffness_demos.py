# -*- coding: utf-8 -*-
"""

This script implements the small textbook demonstrations used to introduce
stiffness, i.e. the ones that do not involve a chromatography model.

The system is a 2x2 linear ODE with two very different time scales. It is small
enough to have an analytical solution and to be integrated by two hand written
time integrators, which is the point: everything that happens here also happens
in a chromatography model, only there it is hidden behind the size of the
system.

"""

import matplotlib.pyplot as plt
import numpy as np

from utility.column_animation import convection_rate, film_rate
from utility.stiffness_metrics import COLORS, style_axis

# The two time scales used throughout this script are not invented. They
# are what one axial cell of the column in `utility/column_animation.py`
# actually does: material is convected across a cell at about 1.2 / s,
# while the film transfer into the particles sitting in that cell runs at
# about 1135 / s. So the 2x2 system below is not a textbook analogue of a
# chromatography model, it is one node of one.
COLUMN_SLOW = -convection_rate()
COLUMN_FAST = -film_rate()

# Darker than the grid, so that the real and imaginary axis read as axes.
_AXIS_COLOR = '#707070'
_MARKER_COLOR = '#1A1A1A'
_EXACT_COLOR = '#404040'


def _latex_float(value, digits=1):
    """Format a number as LaTeX scientific notation, e.g. 4.5 \\cdot 10^{19}.

    Python's own ``{:.0e}`` produces "5e+19", which mathtext then sets as
    "5e + 19". Worth the four lines.

    Parameters
    ----------
    value : float
        The number to format.
    digits : int
        Digits after the decimal point of the mantissa.

    Returns
    -------
    string
        LaTeX, without the surrounding dollar signs.
    """
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0 ** exponent

    return rf"{mantissa:.{digits}f} \cdot 10^{{{exponent}}}"


def two_scale_system(slow=COLUMN_SLOW, fast=COLUMN_FAST):
    """Build a 2x2 linear system with two prescribed time scales.

    The system is symmetric with eigenvectors (1, 1) and (1, -1), so that both
    processes are genuinely coupled: neither component of the solution belongs
    to one time scale alone.

    Parameters
    ----------
    slow : float
        Slow eigenvalue, i.e. the process the result depends on.
    fast : float
        Fast eigenvalue, i.e. the process that dictates the step size.

    Returns
    -------
    np.array
        System matrix of shape (2, 2).
    """
    return 0.5 * np.array([[slow + fast, slow - fast],
                           [slow - fast, slow + fast]])


def two_scale_solution(time, slow=COLUMN_SLOW, fast=COLUMN_FAST):
    """Analytical solution of two_scale_system() for y(0) = (1, 0).

    Parameters
    ----------
    time : np.array
        Times at which the solution is evaluated.
    slow, fast : float
        Eigenvalues, see two_scale_system().

    Returns
    -------
    np.array
        Solution of shape (len(time), 2).
    """
    time = np.asarray(time, dtype=float)

    slow_part = 0.5 * np.exp(slow * time)
    fast_part = 0.5 * np.exp(fast * time)

    return np.column_stack((slow_part + fast_part, slow_part - fast_part))


def explicit_euler(system_matrix, y0, h, t_end, bound=None):
    """Integrate y' = A y with the explicit Euler method.

    One step is y_(n+1) = y_n + h A y_n, so nothing has to be solved: the new
    value is written down directly from the old one.

    Parameters
    ----------
    system_matrix : np.array
        The matrix A.
    y0 : np.array
        Initial value.
    h : float
        Time step size.
    t_end : float
        End of the integration interval.
    bound : float
        Stop once the solution exceeds this magnitude, so that an unstable run
        can be plotted on a sensible axis. Defaults to no bound.

    Returns
    -------
    (np.array, np.array)
        Times and solution of shape (n_steps + 1, len(y0)).
    """
    n_steps = int(round(t_end / h))

    y = np.empty((n_steps + 1, y0.size))
    y[0] = y0

    for n in range(n_steps):
        y[n + 1] = y[n] + h * (system_matrix @ y[n])

        if bound is not None and np.abs(y[n + 1]).max() > bound:
            return h * np.arange(n + 2), y[:n + 2]

    return h * np.arange(n_steps + 1), y


def implicit_euler(system_matrix, y0, h, t_end):
    """Integrate y' = A y with the implicit Euler method.

    One step is (I - h A) y_(n+1) = y_n, i.e. a linear system has to be solved
    for every single step. The model equations of a chromatography simulation
    are nonlinear, so there it is a Newton solve instead.

    Parameters
    ----------
    system_matrix : np.array
        The matrix A.
    y0 : np.array
        Initial value.
    h : float
        Time step size.
    t_end : float
        End of the integration interval.

    Returns
    -------
    (np.array, np.array)
        Times and solution of shape (n_steps + 1, len(y0)).
    """
    n_steps = int(round(t_end / h))
    step_matrix = np.eye(y0.size) - h * system_matrix

    y = np.empty((n_steps + 1, y0.size))
    y[0] = y0

    for n in range(n_steps):
        y[n + 1] = np.linalg.solve(step_matrix, y[n])

    return h * np.arange(n_steps + 1), y


def plot_two_scale_solution(t_end=5.0, zoom_end=0.01, figsize=(10, 3.8),
                            **eigenvalues):
    """Plot the analytical solution of the two time scale system.

    Two panels, because the solution has two time scales and no single axis
    shows both: the fast process is over before the slow one has moved.

    Parameters
    ----------
    t_end : float
        End of the full time span.
    zoom_end : float
        End of the zoomed time span.
    figsize : tuple
        Figure size.
    **eigenvalues
        Passed on to two_scale_solution(), i.e. `slow` and `fast`.

    Returns
    -------
    (matplotlib figure, np.array of matplotlib axes)
    """
    figure, axes = plt.subplots(1, 2, figsize=figsize)

    spans = [
        (axes[0], t_end, 'what you are after: a slow decay'),
        (axes[1], zoom_end, 'the first few milliseconds'),
    ]

    for axis, end, title in spans:
        time = np.linspace(0.0, end, 2001)
        solution = two_scale_solution(time, **eigenvalues)

        axis.plot(time, solution[:, 0], color=COLORS[0], linewidth=2.0,
                  label=r'$y_1$')
        axis.plot(time, solution[:, 1], color=COLORS[1], linewidth=2.0,
                  label=r'$y_2$')

        axis.set_title(title, fontsize=11)
        axis.set_xlabel(r'time $t$')
        axis.legend(frameon=False, fontsize=9)
        style_axis(axis)

    axes[0].set_ylabel(r'$y$')

    figure.tight_layout()

    return figure, axes


def plot_two_scale_runs(runs, t_end=5.0, ylim=None, annotation=None,
                        title=None, figsize=(8.5, 4.2), **eigenvalues):
    """Plot numerical solutions of the two time scale system against the exact one.

    Parameters
    ----------
    runs : list of (label, time, solution)
        Numerical solutions, drawn in the order given.
    t_end : float
        End of the plotted time span.
    ylim : tuple
        Limits of the vertical axis. An unstable run leaves the frame, which is
        the point, so this is set rather than left to the data.
    annotation : string
        Note placed in the upper left, e.g. where an unstable run ended up.
    title : string
        Axis title.
    figsize : tuple
        Figure size.
    **eigenvalues
        Passed on to two_scale_solution(), i.e. `slow` and `fast`.

    Returns
    -------
    (matplotlib figure, matplotlib axis)
    """
    figure, axis = plt.subplots(figsize=figsize)

    time_exact = np.linspace(0.0, t_end, 4001)
    axis.plot(
        time_exact, two_scale_solution(time_exact, **eigenvalues)[:, 0],
        color=_EXACT_COLOR, linewidth=3.5, alpha=0.3, label='exact'
    )

    for index, (label, time, solution) in enumerate(runs):
        inside = time <= t_end
        axis.plot(
            time[inside], solution[inside, 0],
            color=COLORS[index % len(COLORS)], linewidth=2.0,
            marker='o', markersize=4,
            markevery=max(1, int(inside.sum()) // 40),
            label=label
        )

    if ylim is not None:
        axis.set_ylim(*ylim)

    if annotation is not None:
        axis.annotate(
            annotation, xy=(0.03, 0.93), xycoords='axes fraction',
            ha='left', va='top', fontsize=10, color=_MARKER_COLOR
        )

    if title is not None:
        axis.set_title(title, fontsize=11)

    axis.set_xlabel(r'time $t$')
    axis.set_ylabel(r'$y_1$')
    axis.legend(frameon=False, fontsize=9, loc='upper right')
    style_axis(axis)

    figure.tight_layout()

    return figure, axis


def plot_accuracy_and_blowup(implicit_run, explicit_run, t_end=2.0,
                             figsize=(11, 4.2), **eigenvalues):
    """Plot the two ways a time step can be too large, side by side.

    They are not two degrees of the same problem, so they do not share an axis.
    The implicit run is drawn against the exact solution, because the question
    there is how far off it is. The explicit run is drawn as a magnitude on a
    logarithmic axis, because the question there is not how far off it is but
    how fast it leaves.

    Parameters
    ----------
    implicit_run : (label, time, solution)
        The stable but inaccurate run.
    explicit_run : (label, time, solution)
        The unstable run, integrated far enough to show the growth.
    t_end : float
        End of the time span for the accuracy panel.
    figsize : tuple
        Figure size.
    **eigenvalues
        Passed on to two_scale_solution(), i.e. `slow` and `fast`.

    Returns
    -------
    (matplotlib figure, np.array of matplotlib axes)
    """
    figure, axes = plt.subplots(1, 2, figsize=figsize)

    # ---- left: stable, and a little wrong ----
    label, time, solution = implicit_run
    inside = time <= t_end

    time_exact = np.linspace(0.0, t_end, 4001)
    axes[0].plot(
        time_exact, two_scale_solution(time_exact, **eigenvalues)[:, 0],
        color=_EXACT_COLOR, linewidth=3.5, alpha=0.3, label='exact'
    )
    axes[0].plot(
        time[inside], solution[inside, 0], color=COLORS[0], linewidth=2.0,
        marker='o', markersize=4, label=label
    )

    reference = two_scale_solution(time[inside], **eigenvalues)[:, 0]
    error = np.abs(solution[inside, 0] - reference)
    relative_end = error[-1] / reference[-1] if reference[-1] else np.nan

    axes[0].annotate(
        f"largest error ${_latex_float(error.max())}$ at $t = {time[inside][error.argmax()]:.1f}$\n"
        f"off by {100 * relative_end:.0f} % at $t = {t_end:g}$",
        xy=(0.96, 0.92), xycoords='axes fraction', ha='right', va='top',
        fontsize=9, color=_MARKER_COLOR,
        bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                  edgecolor=_AXIS_COLOR, linewidth=0.8, alpha=0.9)
    )
    axes[0].set_title('stable, and a little wrong', fontsize=11)
    axes[0].set_ylabel(r'$y_1$')
    axes[0].legend(frameon=False, fontsize=9, loc='center right')

    # ---- right: not a little wrong ----
    label, time, solution = explicit_run
    magnitude = np.abs(solution[:, 0])
    step_size = time[1] - time[0] if time.size > 1 else np.nan
    growth = abs(1.0 + step_size * eigenvalues.get('fast', COLUMN_FAST))

    axes[1].plot(
        time, np.abs(two_scale_solution(time, **eigenvalues)[:, 0]),
        color=_EXACT_COLOR, linewidth=3.5, alpha=0.3, label='exact'
    )
    axes[1].plot(
        time, magnitude, color=COLORS[1], linewidth=2.0,
        marker='o', markersize=4, label=label
    )
    axes[1].set_yscale('log')

    axes[1].annotate(
        f"multiplied by {growth:.0f} every step\n"
        f"$|y_1| = {_latex_float(magnitude[-1])}$ after {time.size - 1} steps",
        xy=(0.04, 0.92), xycoords='axes fraction', ha='left', va='top',
        fontsize=9, color=_MARKER_COLOR,
        bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                  edgecolor=_AXIS_COLOR, linewidth=0.8, alpha=0.9)
    )
    axes[1].set_title('not a little wrong', fontsize=11)
    axes[1].set_ylabel(r'$|y_1|$')
    axes[1].legend(frameon=False, fontsize=9, loc='lower right')

    for axis in axes:
        axis.set_xlabel(r'time $t$')
        style_axis(axis)

    figure.tight_layout()

    return figure, axes


def plot_stability_regions(scaled_eigenvalues=None, figsize=(10, 4.2)):
    """Plot where explicit and implicit methods are stable.

    A method is stable for a step size h and an eigenvalue lambda if the scaled
    eigenvalue z = h * lambda lies in the method's stability region. For an
    explicit method that region is bounded, so a fast process, i.e. a large
    negative lambda, forces a small h. The implicit methods used in CADET cover
    the entire left half plane, so no step size is forbidden.

    Parameters
    ----------
    scaled_eigenvalues : dict
        Points z = h * lambda to mark, as {label: complex}.
    figsize : tuple
        Figure size.

    Returns
    -------
    (matplotlib figure, np.array of matplotlib axes)
    """
    real = np.linspace(-4.0, 2.0, 601)
    imaginary = np.linspace(-3.5, 3.5, 601)
    z = real[np.newaxis, :] + 1j * imaginary[:, np.newaxis]

    explicit_region = 1.0 + z
    runge_kutta_4 = 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
    with np.errstate(divide='ignore', invalid='ignore'):
        implicit_region = 1.0 / (1.0 - z)

    figure, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    regions = [
        (axes[0], [(runge_kutta_4, 'explicit Runge-Kutta 4', COLORS[1]),
                   (explicit_region, 'explicit Euler', COLORS[0])],
         'explicit: stability limits the step size'),
        (axes[1], [(implicit_region, 'implicit Euler (BDF of order 1)', COLORS[2])],
         'implicit: no step size is forbidden'),
    ]

    for axis, entries, title in regions:
        for amplification, label, color in entries:
            magnitude = np.abs(amplification)
            axis.contourf(
                real, imaginary, magnitude, levels=[0.0, 1.0],
                colors=[color], alpha=0.25
            )
            axis.contour(
                real, imaginary, magnitude, levels=[1.0],
                colors=[color], linewidths=2.0
            )
            # Proxy handle, a filled contour carries no legend entry.
            axis.plot([], [], color=color, linewidth=2.0, label=label)

        axis.axhline(0.0, color=_AXIS_COLOR, linewidth=0.8, zorder=1)
        axis.axvline(0.0, color=_AXIS_COLOR, linewidth=0.8, zorder=1)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel(r'$\mathrm{Re}(z)$,   $z = h \lambda$')
        axis.legend(frameon=False, loc='upper left', fontsize=9)
        style_axis(axis)

    axes[0].set_ylabel(r'$\mathrm{Im}(z)$')

    # The marked step sizes are annotations, not series, so they wear neutral
    # ink: a colour here would read as a fourth method. Labels alternate above
    # and below, because two marked step sizes can sit arbitrarily close.
    if scaled_eigenvalues:
        for index, (label, value) in enumerate(scaled_eigenvalues.items()):
            value = complex(value)

            # A step size far above the stability limit lands far outside the
            # plotted range. Dropping it silently would hide exactly the case
            # worth showing, so it becomes an arrow at the edge instead.
            off_scale = not (real[0] <= value.real <= real[-1])

            if off_scale:
                for axis in axes:
                    axis.annotate(
                        '', xy=(real[0], 0.0), xytext=(real[0] + 0.9, 0.0),
                        arrowprops=dict(arrowstyle='-|>', color=_MARKER_COLOR,
                                        linewidth=1.8)
                    )
                axes[0].annotate(
                    label + ' is off to the left',
                    xy=(real[0] + 0.9, 0.0), xytext=(6, 10),
                    textcoords='offset points',
                    ha='left', va='bottom', fontsize=9, color=_MARKER_COLOR
                )
                continue

            for axis in axes:
                axis.plot(
                    value.real, value.imag, marker='o', markersize=9,
                    color=_MARKER_COLOR, markeredgecolor='white',
                    markeredgewidth=1.5, zorder=5
                )
            axes[0].annotate(
                label, xy=(value.real, value.imag),
                xytext=(0, 14 if index % 2 else -22),
                textcoords='offset points',
                ha='center', fontsize=9, color=_MARKER_COLOR
            )

    figure.tight_layout()

    return figure, axes


# Rates measured across the ten case studies, in 1/s, as (lowest, highest).
MEASURED_RATES = (
    ('binding', 1.4e-06, 1.4e+14),
    ('reaction', 5.0e-04, 5.0e+10),
    ('dead volume mixing', 3.3e-03, 3.3e+05),
    ('film mass transfer', 1.1e-01, 1.14e+02),
    ('pore diffusion, one particle cell', 5.0e-03, 5.1e+00),
    ('convection, one grid cell', 2.8e-01, 2.5e+01),
)

# Courant number sustained by the most convection dominated run of the
# chapter: polynomial degree 15 on 256 nodes, column Peclet 1.1e7, median step
# size 7.1e-3 s against a nodal convection rate of 40.9 1/s. Reproducible from
# the sliders of "Stiffness 3b - Steep fronts DG".
#
# The median rather than the largest step: single steps reach C ~ 1e3 in every
# run of that study, but only after the column has reached steady state, where
# a convection operator acting on a flat profile excites nothing.
MEASURED_COURANT = 0.29

# Smallest Courant number at which an upwind convection locus first touches the
# unstable region of BDF5, found by sampling.
CRITICAL_COURANT = 4.0


def bdf_boundary(order, points=2001):
    """Boundary of the stability region of the BDF method of this order.

    The region is the *outside* of this curve: a BDF method is stable for every
    scaled eigenvalue the curve does not enclose.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, points)
    zeta = np.exp(1j * theta)

    return sum((1.0 / j) * (1.0 - 1.0 / zeta) ** j for j in range(1, order + 1))


def upwind_locus(courant, points=600):
    """Eigenvalues of an upwind convection operator, scaled by the step size.

    They lie on a circle through the origin, of radius equal to the Courant
    number, entirely in the left half plane.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, points)

    return -courant * (1.0 - np.exp(-1j * angle))


def plot_where_the_eigenvalues_go(order=5, step_size=1.0, figsize=(11.5, 4.8)):
    """What BDF cannot integrate, and where this chapter actually went.

    Left: the stability region near the origin, which is the only place there
    is any trouble at all. Right: how far out along the negative real axis the
    processes of the case studies reach, on a logarithmic ruler, because they
    span twenty decades.

    Parameters
    ----------
    order : int
        BDF order whose stability region is drawn.
    step_size : float
        Step size in seconds used to turn rates into z = h * lambda.
    figsize : tuple
        Figure size.

    Returns
    -------
    (matplotlib figure, np.array of matplotlib axes)
    """
    figure, axes = plt.subplots(1, 2, figsize=figsize,
                                gridspec_kw={'width_ratios': [1.0, 1.3]})

    # ---------------- left: the only region that is not stable -------------
    left = axes[0]
    boundary = bdf_boundary(order)

    left.fill(boundary.real, boundary.imag, color=COLORS[1], alpha=0.18,
              zorder=1)
    left.plot(boundary.real, boundary.imag, color=COLORS[1], linewidth=2.0,
              zorder=2, label=f'BDF{order} is unstable in here')

    for courant, style, label in (
            (MEASURED_COURANT, '-', f'convection, as measured (C = {MEASURED_COURANT:g})'),
            (CRITICAL_COURANT, '--', f'convection, where it would touch (C = {CRITICAL_COURANT:g})')):
        locus = upwind_locus(courant)
        left.plot(locus.real, locus.imag, color=COLORS[2], linewidth=1.8,
                  linestyle=style, zorder=3, label=label)

    left.annotate('', xy=(-6.5, 0.0), xytext=(-0.3, 0.0),
                  arrowprops=dict(arrowstyle='-|>', linewidth=3.5,
                                  color=COLORS[0], alpha=0.75))
    left.text(-7.2, -5.2, 'every dissipative\nprocess sits on\nthis axis, and it\n'
              'is stable however\nfar out',
              ha='left', va='top', fontsize=8.5, color=COLORS[0])

    left.axhline(0.0, color=_AXIS_COLOR, linewidth=0.8, zorder=0)
    left.axvline(0.0, color=_AXIS_COLOR, linewidth=0.8, zorder=0)
    left.set_xlim(-7.5, 19.5)
    left.set_ylim(-11.5, 11.5)
    left.set_xlabel(r'$\mathrm{Re}(z)$,  $z = h\lambda$')
    left.set_ylabel(r'$\mathrm{Im}(z)$')
    left.set_title('the only trouble there is, to scale', fontsize=11)
    left.legend(fontsize=7.5, loc='upper left', frameon=True,
                facecolor='white', framealpha=0.92, edgecolor='none')
    style_axis(left)

    # ---------------- right: how far the chapter actually went -------------
    right = axes[1]
    island = np.abs(bdf_boundary(order)).max()

    right.axvline(island, color=COLORS[1], linewidth=1.5, linestyle='--')
    right.annotate('the whole unstable island fits\ninside here, and even so it\n'
                   'never touches this axis',
                   xy=(island, 0.3), xytext=(8, 0),
                   textcoords='offset points', fontsize=8.5, color=COLORS[1],
                   ha='left', va='bottom')

    for index, (name, low, high) in enumerate(MEASURED_RATES):
        height = len(MEASURED_RATES) - 1 - index
        right.plot([low * step_size, high * step_size], [height, height],
                   color=COLORS[0], linewidth=4.0, solid_capstyle='round')
        right.plot([high * step_size], [height], marker='o', markersize=7,
                   color=COLORS[0])
        right.annotate(f'{name}  ', xy=(low * step_size, height),
                       xytext=(-6, 0), textcoords='offset points',
                       ha='right', va='center', fontsize=9, color='#333333')

    right.set_xscale('log')
    right.set_xlim(1e-7, 1e16)
    right.set_ylim(-0.8, len(MEASURED_RATES) - 0.2)
    right.set_yticks([])
    right.set_xlabel(r'$|z| = h\,|\lambda|$ on the negative real axis, '
                     'swept across the studies')
    right.set_title('and how far the knobs of this chapter reached', fontsize=11)
    style_axis(right)

    figure.tight_layout()

    return figure, axes


def bdf_is_stable(order, z):
    """Is the BDF method of this order stable at the scaled eigenvalue z?

    Tested the honest way, from the roots of the characteristic polynomial
    rather than from the picture: the method is stable at z when every root of

        rho(zeta) - z * zeta^order,    rho(zeta) = sum_j (1/j) zeta^(k-j) (zeta-1)^j

    lies in the closed unit disk. This matters because the boundary locus of
    BDF4 and BDF5 crosses itself, so asking whether a point is inside the drawn
    curve is ambiguous, while asking where the roots went is not.

    Parameters
    ----------
    order : int
        BDF order.
    z : complex or np.array
        Scaled eigenvalue(s) h * lambda.

    Returns
    -------
    bool or np.array of bool
    """
    scalar = np.isscalar(z) or np.ndim(z) == 0
    points = np.atleast_1d(np.asarray(z, dtype=complex))

    rho = np.zeros(order + 1, dtype=complex)
    for j in range(1, order + 1):
        term = np.polynomial.polynomial.polypow([-1.0, 1.0], j) / j
        rho[order - j:order + 1] += term

    stable = np.empty(points.shape, dtype=bool)
    for index, value in np.ndenumerate(points):
        coefficients = rho.copy()
        coefficients[order] -= value
        roots = np.polynomial.polynomial.polyroots(coefficients)
        stable[index] = bool(np.all(np.abs(roots) <= 1.0 + 1e-9))

    return bool(stable[0]) if scalar else stable


def wedge_angle(order, radii=None, samples=361):
    """Largest angle from the negative real axis that is stable at every radius.

    This is the alpha of A(alpha) stability: the widest sector around the
    negative real axis the method handles no matter how large the eigenvalue
    gets. Computed here rather than quoted, by walking rays outward from the
    origin and asking where the first unstable point appears.

    Parameters
    ----------
    order : int
        BDF order.
    radii : np.array
        Magnitudes tested along each ray.
    samples : int
        Number of angles tested between 0 and 90 degrees.

    Returns
    -------
    float
        The angle in degrees.
    """
    if radii is None:
        radii = np.logspace(-4, 4, 600)

    best = 0.0
    for degrees in np.linspace(0.0, 90.0, samples):
        ray = radii * np.exp(1j * (np.pi - np.radians(degrees)))
        if not bdf_is_stable(order, ray).all():
            break
        best = degrees

    return best


def plot_bdf_stability_regions(orders=(1, 2, 3, 4, 5), figsize=(13.5, 3.3)):
    """The stability regions of the five BDF methods IDAS switches between.

    IDAS runs backward differentiation formulas of order one to five and picks
    the order as it goes. Each panel shows one of them: the shaded island is
    where that order is *unstable*, and everything outside it is stable.

    Two things are worth reading off. The island grows with the order, so the
    accuracy of a high order is bought with a smaller stable set -- which is
    why IDAS drops the order when a step is rejected. And no island touches the
    negative real axis at any order, which is where every dissipative process
    in a chromatography model puts its eigenvalues.

    Parameters
    ----------
    orders : sequence of int
        BDF orders to draw.
    figsize : tuple
        Figure size.

    Returns
    -------
    (matplotlib figure, np.array of matplotlib axes)
    """
    figure, axes = plt.subplots(1, len(orders), figsize=figsize,
                                sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for axis, order in zip(axes, orders):
        boundary = bdf_boundary(order)
        axis.fill(boundary.real, boundary.imag, color=COLORS[1], alpha=0.20)
        axis.plot(boundary.real, boundary.imag, color=COLORS[1], linewidth=1.6)

        axis.axhline(0.0, color=_AXIS_COLOR, linewidth=0.7, zorder=0)
        axis.axvline(0.0, color=_AXIS_COLOR, linewidth=0.7, zorder=0)
        axis.annotate('', xy=(-5.0, 0.0), xytext=(0.0, 0.0),
                      arrowprops=dict(arrowstyle='-|>', linewidth=2.2,
                                      color=COLORS[0], alpha=0.85))

        alpha = wedge_angle(order)
        reach = np.abs(boundary).max()

        if alpha < 89.5:
            for sign in (1.0, -1.0):
                ray = 40.0 * np.exp(1j * sign * (np.pi - np.radians(alpha)))
                axis.plot([0.0, ray.real], [0.0, ray.imag], color=_AXIS_COLOR,
                          linewidth=1.0, linestyle=':', zorder=1)

        subtitle = ('A-stable' if alpha >= 89.5
                    else f'A({alpha:.0f}$^\\circ$)')
        axis.set_title(f'BDF{order}  --  {subtitle}', fontsize=10)
        axis.set_xlabel(r'$\mathrm{Re}(z)$')
        axis.text(0.96, 0.04, f'$|z| \\leq {reach:.1f}$', fontsize=8,
                  color=COLORS[1], ha='right', va='bottom',
                  transform=axis.transAxes)
        style_axis(axis)

    axes[0].set_ylabel(r'$\mathrm{Im}(z)$')
    axes[0].set_xlim(-6.0, 19.5)
    axes[0].set_ylim(-11.5, 11.5)
    axes[0].text(3.0, 7.2, 'unstable\ninside', fontsize=8, color=COLORS[1])
    axes[0].text(-5.6, -9.6, 'the dissipative\ndirection', fontsize=8,
                 color=COLORS[0])

    figure.tight_layout()

    return figure, axes


TIME_STEPS = (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)


def make_time_step_explorer(slow=COLUMN_SLOW, fast=COLUMN_FAST, t_end=5.0,
                            time_steps=TIME_STEPS, bound=1.0e8,
                            figsize=(11.0, 4.2)):
    """One knob, the time step, applied to both methods at once.

    The same two scale system integrated by explicit and implicit Euler, with
    the step size as a slider. Each method gets its own panel and its own
    table: the plot is the run at the current setting, the table collects every
    setting tried, sorted from the coarsest step to the finest. Running the
    same step size again replaces its row rather than adding a second one.

    The column worth watching is the amplification of the fast mode, the factor
    by which one step multiplies the component that decays in milliseconds:

        explicit    |1 + h * lambda_fast|        grows without bound
        implicit    1 / |1 - h * lambda_fast|    always below one

    The explicit method is stable exactly while its factor stays below one,
    i.e. for h < 2 / |lambda_fast|, which is 1.76 ms for the column's film
    rate. The implicit method has no such limit, which is the whole point.

    Parameters
    ----------
    slow, fast : float
        Eigenvalues of the system, see two_scale_system().
    t_end : float
        End of the integration interval.
    time_steps : sequence of float
        Slider positions.
    bound : float
        Magnitude at which a diverging run is abandoned, so that it can still
        be plotted.
    figsize : tuple
        Figure size.

    Returns
    -------
    callable
        Takes `time_step` and draws. Carries `sliders` for ipywidgets.interact
        and `records`, keyed by (method, time step) so that a rerun overwrites.
    """
    import pandas as pd
    from IPython.display import HTML, display

    IMPLICIT = 'implicit Euler'
    EXPLICIT = 'explicit Euler'

    system = two_scale_system(slow=slow, fast=fast)
    y0 = np.array([1.0, 0.0])
    records = {}
    error_column = f'error at t = {t_end:g}'

    def _error(time, solution):
        """Relative error of the slow answer at the end of the run.

        Deliberately not the largest error over the whole run. That number is
        misleading here, because a large step annihilates the fast transient in
        a single step and then never samples it, while a small step resolves
        the transient imperfectly and is charged for it - so the maximum error
        comes out *smaller* for the larger step. What the answer is actually
        made of is the slow scale, and there the relative error of both methods
        compounds step by step, so the end of the run is also its worst point.
        """
        exact = two_scale_solution(time[-1:], slow=slow, fast=fast)[0, 0]

        return float(solution[-1, 0] - exact) / exact

    def _panel(axis, time, solution, color, label, diverged):
        """One method's run, drawn against the exact solution."""
        fine = np.linspace(0.0, t_end, 4001)
        exact = two_scale_solution(fine, slow=slow, fast=fast)[:, 0]

        # Thousands of markers merge into a band that hides the exact curve
        # underneath, which is the one thing the panel is there to compare to.
        marker = 'o' if time.size <= 200 else None

        if diverged:
            # Linear axes cannot show a solution that reached 1e9, so switch
            # to magnitude on a logarithmic axis and say so in the title.
            axis.plot(fine, np.abs(exact), color=_EXACT_COLOR, linewidth=3.5,
                      alpha=0.3, label='exact')
            axis.plot(time, np.abs(solution[:, 0]), color=color, linewidth=2.0,
                      marker=marker, markersize=3.5, label=label)
            axis.set_yscale('log')
            axis.set_ylabel(r'$|y_1|$')
        else:
            axis.plot(fine, exact, color=_EXACT_COLOR, linewidth=3.5,
                      alpha=0.3, label='exact')
            axis.plot(time, solution[:, 0], color=color, linewidth=2.0,
                      marker=marker, markersize=3.5, label=label)
            axis.set_ylim(-0.15, 1.15)
            axis.set_ylabel(r'$y_1$')

        axis.set_xlabel(r'time $t$')
        axis.legend(fontsize=9, loc='upper right', frameon=True,
                    facecolor='white', framealpha=0.9, edgecolor='none')
        style_axis(axis)

    def _note(axis, text):
        axis.annotate(text, xy=(0.5, 0.06), xycoords='axes fraction',
                      ha='center', va='bottom', fontsize=9,
                      color=_MARKER_COLOR,
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor=_AXIS_COLOR, linewidth=0.8,
                                alpha=0.9))

    def _table(method):
        """Everything tried with this method, coarsest step first."""
        rows = [row for (name, _), row in records.items() if name == method]
        rows.sort(key=lambda row: -row['_step'])
        frame = pd.DataFrame(rows).drop(columns='_step')

        return frame.to_html(index=False, escape=False)

    def show(time_step):
        time_implicit, y_implicit = implicit_euler(system, y0, time_step, t_end)
        time_explicit, y_explicit = explicit_euler(system, y0, time_step,
                                                   t_end, bound=bound)

        diverged = time_explicit[-1] < t_end - 0.5 * time_step
        growth = {IMPLICIT: 1.0 / abs(1.0 - time_step * fast),
                  EXPLICIT: abs(1.0 + time_step * fast)}

        figure, axes = plt.subplots(1, 2, figsize=figsize)

        _panel(axes[0], time_implicit, y_implicit, COLORS[0], IMPLICIT, False)
        axes[0].set_title(f'implicit Euler,  h = {time_step:g},  '
                          f'{len(time_implicit) - 1} steps', fontsize=11)
        error_implicit = _error(time_implicit, y_implicit)
        _note(axes[0], f'off by {100 * error_implicit:+.2f} % '
                       f'at $t = {t_end:g}$')

        _panel(axes[1], time_explicit, y_explicit, COLORS[1], EXPLICIT,
               diverged)
        axes[1].set_title(
            f'explicit Euler,  h = {time_step:g},  '
            + (f'left after {len(time_explicit) - 1} steps' if diverged
               else f'{len(time_explicit) - 1} steps'), fontsize=11)
        if diverged:
            _note(axes[1], f'fast mode $\\times${growth[EXPLICIT]:.3g} '
                           'every step')
        else:
            error_explicit = _error(time_explicit, y_explicit)
            _note(axes[1], f'off by {100 * error_explicit:+.2f} % '
                           f'at $t = {t_end:g}$')

        figure.tight_layout()
        plt.show()

        for method, time, solution, ok in (
                (IMPLICIT, time_implicit, y_implicit, True),
                (EXPLICIT, time_explicit, y_explicit, not diverged)):
            records[(method, time_step)] = {
                '_step': time_step,
                'time step': f'{time_step:g}',
                'steps': len(time) - 1,
                'fast mode per step': f'{growth[method]:.4g}',
                error_column: (f'{100 * _error(time, solution):+.2f} %'
                               if ok else '-'),
                'result': 'ok' if ok else 'diverged',
            }

        display(HTML(
            '<div style="display:flex; gap:2.5rem; align-items:flex-start">'
            f'<div><b>{IMPLICIT}</b>{_table(IMPLICIT)}</div>'
            f'<div><b>{EXPLICIT}</b>{_table(EXPLICIT)}</div>'
            '</div>'))

    def sliders():
        from ipywidgets import widgets

        return {'time_step': widgets.SelectionSlider(
            options=[(f'{value:g}', value) for value in time_steps],
            value=time_steps[0],
            description='time step', continuous_update=False)}

    show.records = records
    show.sliders = sliders()
    show.stability_limit = 2.0 / abs(fast)

    return show
