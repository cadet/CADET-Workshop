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

from utility.stiffness_metrics import COLORS, style_axis

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


def two_scale_system(slow=-1.0, fast=-1000.0):
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


def two_scale_solution(time, slow=-1.0, fast=-1000.0):
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
    growth = abs(1.0 + step_size * eigenvalues.get('fast', -1000.0))

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
