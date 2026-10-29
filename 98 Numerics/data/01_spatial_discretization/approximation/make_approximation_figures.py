"""Approximation figures for the Spatial Discretization notebook: FV against DG.

Run it from this directory, no arguments, no CADET, a few seconds:

    python make_approximation_figures.py

It writes three PNGs next to itself and prints the errors it drew.

--------------------------------------------------------------------------
WHAT THE FIGURES SAY
--------------------------------------------------------------------------
``approximation_FV_vs_DG.png``
    First-order FV against DG at equal degrees of freedom. FV spends its
    unknowns on many small cells and holds the solution constant in each, so
    it draws a staircase; DG groups the same unknowns into a few large
    elements and carries a polynomial in each, so it tracks the curve. An
    inset magnifies one element interface, where the DG solution jumps: that
    is the word "discontinuous" made visible.
``approximation_FV_DG_orders.png``
    The same budget as constants per cell, linears per element (p=1) and
    degree-5 per element. Read left to right the degree rises while the
    number of cells or elements falls. The p=1 panel is where the interface
    jumps are plain without magnification, and it makes FV read as the
    lowest member of one family rather than a different animal.
``approximation_WENO_vs_DG.png``
    FV *with a WENO reconstruction* against DG. Both reach high order, and
    the figure is about how they pay for it: FV still stores one unknown per
    cell and reaches across five neighbouring averages (shaded band), DG
    carries six unknowns inside one element and never leaves it. With WENO
    the accuracy gap shrinks from a factor ~20 to a factor ~2, which is why
    the first figure deliberately uses first-order FV instead.

--------------------------------------------------------------------------
NOTES FOR PICKING THIS UP AGAIN
--------------------------------------------------------------------------
Invariants worth not breaking:

* **Equal degrees of freedom per panel.** Every comparison here is only
  honest because ``n_cells == n_elements * (degree + 1)``. The builders
  raise if that is violated. If you change the budget, change it everywhere.
* **Nothing is simulated.** Each panel shows the best representation the
  method's function space admits: the exact cell average for FV, the
  elementwise L2 projection for DG. That isolates approximation behaviour
  from transport and time stepping. For the *dynamic* failure modes
  (WENO smearing, DG oscillating at a steep front) see the explorers in
  ``98 Numerics/utility/sharp_front_comparison.py`` instead; do not try to
  make these figures do that job.
* **``check_weno_against_textbook_formulas`` must keep passing.** It runs
  before anything is drawn. It asserts the candidate stencils reproduce the
  standard Jiang-Shu expressions at both interfaces, and that the linear
  weights recover the five-cell degree-4 reconstruction.

Choices that are *visualization* decisions, not scheme definitions. Keep
them labelled honestly in the figures if you touch them:

* The DG curve is an L2 projection, not a DG solution. It is the yardstick a
  DG solution is measured against, and it is the fair counterpart to an
  exact cell average on the FV side.
* The WENO curve drawn *inside* each cell blends the candidate polynomials
  with the mean of the two one-sided weight sets. Only ``traces`` (the dots
  at the cell interfaces) is genuine scheme output, i.e. what a finite volume
  scheme actually hands to the flux. In smooth regions the nonlinear weights
  approach the linear ones and all of these coincide anyway.
* The WENO stencil is closed at the domain boundaries by constant extension.
  That is exact to plotting accuracy because the peak has decayed there, so
  the real order drop at the inlet (2nd order, see the notebook's WENO table)
  is *not* shown.

Layout constants that need re-tuning if the profile, the budget or the
figure size changes: the bracket height (1.12) and the per-figure ``ylim``,
the inset rectangle in ``draw_interface_inset``, the legend anchors in
``make_weno_figure``, and ``stencil_cell`` in ``make_weno_figure`` (it is
placed on the falling flank, where a reconstruction has real work to do).

Colour roles: black is always the exact profile, ``DISCRETE_COLOR`` the
discrete solution being judged, ``STATE_COLOR`` the stored unknowns when
they differ from it (only in the WENO panel), ``HIGHLIGHT_COLOR`` the
stencil/element call-out, red the error.

Obvious next steps if the topic comes up again:

* A second row with a steep front, where the trade-off flips: WENO smears
  and DG overshoots. Needs an actual solve, or at least a reconstruction of
  a discontinuous profile, so the nonlinear weights stop being decorative.
* A WENO1 / WENO2 / WENO3 panel row, matching ``WENO_ORDER`` in CADET.
* FV drawn explicitly as DG p=0, to close the family argument.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import legendre, polynomial

EXACT_COLOR = 'black'
DISCRETE_COLOR = '#1f4ea8'
STATE_COLOR = '#8aa7cf'
HIGHLIGHT_COLOR = '#d9770b'
ERROR_COLOR = '#d62728'
MESH_COLOR = '#9a9a9a'
BAND_COLOR = '#f1f4f8'

# The column and the peak on it. The centre sits off the element boundaries of
# every panel on purpose, so that no interface lands on the symmetry axis and
# accidentally shows a zero jump.
DOMAIN = (0.0, 1.0)
PEAK_CENTER = 0.42
PEAK_WIDTH = 0.07


def exact_profile(z):
    """A smooth concentration peak on the unit column."""
    return np.exp(-0.5 * ((np.asarray(z) - PEAK_CENTER) / PEAK_WIDTH) ** 2)


def lobatto_nodes(degree):
    """The degree+1 Legendre-Gauss-Lobatto nodes on [-1, 1].

    These are the interior extrema of the degree-p Legendre polynomial plus the
    two endpoints; CADET-Core places the DG unknowns on them, which is why the
    markers in the DG panels sit where they do.
    """
    if degree < 1:
        raise ValueError('The polynomial degree must be at least one.')
    if degree == 1:
        return np.array([-1.0, 1.0])
    coefficients = np.zeros(degree + 1)
    coefficients[degree] = 1.0
    interior = legendre.legroots(legendre.legder(coefficients))
    return np.concatenate(([-1.0], interior, [1.0]))


def _element_samples(boundaries, n_quadrature):
    """Gauss-Legendre weights and profile samples on every element."""
    nodes, weights = legendre.leggauss(n_quadrature)
    lower, upper = boundaries[:-1], boundaries[1:]
    half_width = 0.5 * (upper - lower)
    centers = 0.5 * (lower + upper)
    samples = exact_profile(centers[:, None] + half_width[:, None] * nodes[None, :])
    return nodes, weights, samples


def cell_averages(boundaries, n_quadrature=40):
    """Exact average of the profile over every cell: the FV unknowns."""
    _, weights, samples = _element_samples(boundaries, n_quadrature)
    return 0.5 * (samples @ weights)


def legendre_projection(boundaries, degree, n_quadrature=40):
    """Elementwise L2 projection of the profile onto the degree-p polynomials.

    Returns the Legendre coefficients per element: the best approximation the DG
    space admits, and the yardstick a DG solution is measured against.
    """
    nodes, weights, samples = _element_samples(boundaries, n_quadrature)
    coefficients = np.zeros((len(boundaries) - 1, degree + 1))
    for order in range(degree + 1):
        basis = np.zeros(order + 1)
        basis[order] = 1.0
        # <u, P_k> / <P_k, P_k>; the reference mass matrix is diagonal.
        coefficients[:, order] = (
            (samples * legendre.legval(nodes, basis)[None, :]) @ weights
        ) * (2 * order + 1) / 2
    return coefficients


def evaluate_projection(coefficients, boundaries, z):
    """Evaluate the piecewise polynomial at points of the domain.

    At a shared interface the later element wins, so do not use this to draw
    the two one-sided limits; see ``draw_interface_inset``, which evaluates
    each side against its own element.
    """
    values = np.empty_like(np.asarray(z, dtype=float))
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        inside = (z >= lower) & (z <= upper)
        reference = 2 * (z[inside] - lower) / (upper - lower) - 1
        values[inside] = legendre.legval(reference, coefficients[index])
    return values


# Fifth-order WENO-JS: three candidate stencils of three cells each, and the
# linear weights that make their blend fifth-order at the right and left cell
# interface. This is CADET's WENO_ORDER = 3.
WENO_STENCILS = ((-2, -1, 0), (-1, 0, 1), (0, 1, 2))
WENO_WEIGHTS_RIGHT = np.array([1.0, 6.0, 3.0]) / 10.0
WENO_WEIGHTS_LEFT = np.array([3.0, 6.0, 1.0]) / 10.0
WENO_EPSILON = 1e-6


def stencil_reconstruction_matrix(offsets):
    """Cell averages of the monomials over the cells of one stencil.

    Row k, column m holds the average of xi**m over the cell at offset
    ``offsets[k]``, with xi = (z - z_i) / h measured from the target cell.
    Solving against the stencil's averages gives the unique polynomial whose
    cell averages reproduce them. Deriving the candidates this way, rather than
    hard-coding the textbook coefficients, keeps the stencil order a parameter.
    """
    order = len(offsets)
    matrix = np.empty((order, order))
    for row, offset in enumerate(offsets):
        for power in range(order):
            matrix[row, power] = (
                (offset + 0.5) ** (power + 1) - (offset - 0.5) ** (power + 1)
            ) / (power + 1)
    return matrix


def smoothness_indicators(padded):
    """The Jiang-Shu smoothness indicators of the three candidate stencils."""
    minus2, minus1, center, plus1, plus2 = (padded[shift:shift + len(padded) - 4]
                                            for shift in range(5))
    return np.stack((
        13 / 12 * (minus2 - 2 * minus1 + center) ** 2
        + 1 / 4 * (minus2 - 4 * minus1 + 3 * center) ** 2,
        13 / 12 * (minus1 - 2 * center + plus1) ** 2
        + 1 / 4 * (minus1 - plus1) ** 2,
        13 / 12 * (center - 2 * plus1 + plus2) ** 2
        + 1 / 4 * (3 * center - 4 * plus1 + plus2) ** 2,
    ))


def weno_reconstruction(averages):
    """Fifth-order WENO-JS reconstruction from cell averages on a uniform grid.

    Returns ``(blended, traces, weights)``:

    ``traces['left'] / traces['right']``
        The reconstructed one-sided values at the cell interfaces. This is the
        genuine output of the scheme, what a finite volume method hands to the
        flux, and the only thing in this function that exists outside the plot.
    ``blended``
        Per-cell polynomial coefficients in ascending powers of xi, used only
        for drawing a curve that shows where the traces come from. It blends
        the candidates with the mean of the two one-sided weight sets, which is
        a visualization choice, not a formula from the literature. Where the
        solution is smooth the nonlinear weights approach the linear ones and
        this coincides with the central degree-4 reconstruction.

    The stencil is closed at the boundaries by constant extension, which is
    exact to plotting accuracy for a profile that has decayed there, and which
    therefore hides the real order drop at the column inlet.
    """
    averages = np.asarray(averages, dtype=float)
    padded = np.concatenate((
        np.repeat(averages[:1], 2), averages, np.repeat(averages[-1:], 2),
    ))

    # Candidate polynomial per stencil, in ascending powers of xi.
    candidates = np.empty((len(WENO_STENCILS), len(averages), len(WENO_STENCILS[0])))
    for index, offsets in enumerate(WENO_STENCILS):
        stencil_averages = np.stack([padded[2 + offset:2 + offset + len(averages)]
                                     for offset in offsets])
        candidates[index] = np.linalg.solve(
            stencil_reconstruction_matrix(offsets), stencil_averages
        ).T

    beta = smoothness_indicators(padded)
    weights = {}
    for side, linear in (('right', WENO_WEIGHTS_RIGHT), ('left', WENO_WEIGHTS_LEFT)):
        alpha = linear[:, None] / (WENO_EPSILON + beta) ** 2
        weights[side] = alpha / alpha.sum(axis=0, keepdims=True)

    traces = {
        side: np.einsum(
            'sc,sc->c', weights[side],
            polynomial.polyval(offset, candidates.transpose(2, 0, 1)),
        )
        for side, offset in (('right', 0.5), ('left', -0.5))
    }
    blended_weights = 0.5 * (weights['right'] + weights['left'])
    blended = np.einsum('sc,scm->cm', blended_weights, candidates)
    return blended, traces, weights


def evaluate_weno(blended, centers, width, z):
    """Evaluate the per-cell reconstruction polynomials at points of the domain."""
    values = np.empty_like(np.asarray(z, dtype=float))
    for index, center in enumerate(centers):
        inside = (z >= center - 0.5 * width) & (z <= center + 0.5 * width)
        values[inside] = polynomial.polyval(
            (z[inside] - center) / width, blended[index]
        )
    return values


def draw_mesh(axis, boundaries, line_width):
    """Shade alternating cells and draw the boundaries between them.

    The shading is what makes "many narrow cells" versus "four broad elements"
    readable at a glance, so keep it even though it looks like decoration.
    """
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if index % 2 == 0:
            axis.axvspan(lower, upper, color=BAND_COLOR, zorder=0)
    for boundary in boundaries:
        axis.axvline(boundary, color=MESH_COLOR, linewidth=line_width, zorder=1)


def draw_finite_volumes(axis, n_cells, z_fine):
    """A staircase of cell averages, one unknown per cell. Returns the error."""
    boundaries = np.linspace(DOMAIN[0], DOMAIN[1], n_cells + 1)
    averages = cell_averages(boundaries)
    centers = 0.5 * (boundaries[:-1] + boundaries[1:])

    draw_mesh(axis, boundaries, line_width=0.6)

    staircase = averages[np.clip(
        np.searchsorted(boundaries, z_fine, side='right') - 1, 0, n_cells - 1
    )]
    exact = exact_profile(z_fine)
    axis.fill_between(
        z_fine, exact, staircase, color=ERROR_COLOR, alpha=0.3, linewidth=0,
        zorder=2, label='approximation error',
    )
    axis.plot(z_fine, exact, color=EXACT_COLOR, linewidth=2.4, zorder=3,
              label='exact profile')
    axis.hlines(
        averages, boundaries[:-1], boundaries[1:],
        color=DISCRETE_COLOR, linewidth=2.0, linestyles=(0, (5, 2)), zorder=4,
        label='discrete solution',
    )
    # Open circles mark the unknowns themselves, so that the equal-budget claim
    # in the title can be counted off the picture rather than taken on trust.
    axis.plot(
        centers, averages, linestyle='none', marker='o', markersize=4.5,
        markerfacecolor='white', markeredgecolor=DISCRETE_COLOR,
        markeredgewidth=1.4, zorder=5, label='degree of freedom',
    )
    return np.max(np.abs(exact - staircase))


def draw_discontinuous_galerkin(axis, n_elements, degree, z_fine):
    """A polynomial per element, p+1 unknowns each.

    Returns ``(error, boundaries, coefficients)``; the callers need the latter
    two to place the interface inset and the element highlight.
    """
    boundaries = np.linspace(DOMAIN[0], DOMAIN[1], n_elements + 1)
    coefficients = legendre_projection(boundaries, degree)

    draw_mesh(axis, boundaries, line_width=1.4)

    approximation = evaluate_projection(coefficients, boundaries, z_fine)
    exact = exact_profile(z_fine)
    axis.fill_between(
        z_fine, exact, approximation, color=ERROR_COLOR, alpha=0.3, linewidth=0,
        zorder=2, label='approximation error',
    )
    axis.plot(z_fine, exact, color=EXACT_COLOR, linewidth=2.4, zorder=3,
              label='exact profile')
    # One call per element, so that the pieces are visibly separate curves and
    # matplotlib never draws a connecting segment across an interface.
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        inside = (z_fine >= lower) & (z_fine <= upper)
        axis.plot(
            z_fine[inside], approximation[inside], color=DISCRETE_COLOR,
            linewidth=2.0, dashes=(5, 2), zorder=4,
            label='discrete solution' if index == 0 else None,
        )
    reference_nodes = lobatto_nodes(degree)
    node_positions = (
        0.5 * (boundaries[:-1] + boundaries[1:])[:, None]
        + 0.5 * np.diff(boundaries)[:, None] * reference_nodes[None, :]
    ).ravel()
    axis.plot(
        node_positions,
        evaluate_projection(coefficients, boundaries, node_positions),
        linestyle='none', marker='o', markersize=4.5, markerfacecolor='white',
        markeredgecolor=DISCRETE_COLOR, markeredgewidth=1.4, zorder=5,
        label='degree of freedom',
    )
    return np.max(np.abs(exact - approximation)), boundaries, coefficients


def interface_jumps(coefficients):
    """Size of the discontinuity at every interior element interface."""
    return np.array([
        legendre.legval(-1.0, coefficients[index + 1])
        - legendre.legval(1.0, coefficients[index])
        for index in range(len(coefficients) - 1)
    ])


def draw_interface_inset(axis, boundaries, coefficients, interface_index):
    """Magnify one element interface, where the DG solution is discontinuous.

    At degree 5 the jump is ~4e-3 and simply invisible on the panel, so the
    window is scaled to the jump rather than to the figure: the x half-window
    is set from the local slope so that the profile climbs about as much as the
    jump across it, and the y span from the jump itself. The magnification ends
    up in the three-digit range and is printed in the inset title, because a
    zoom this extreme has to announce itself.
    """
    interface = boundaries[interface_index]
    jump = interface_jumps(coefficients)[interface_index - 1]
    left_limit = legendre.legval(1.0, coefficients[interface_index - 1])
    right_limit = legendre.legval(-1.0, coefficients[interface_index])

    slope = abs(
        (exact_profile(interface + 1e-6) - exact_profile(interface - 1e-6)) / 2e-6
    )
    half_window = 1.5 * abs(jump) / max(slope, 1e-12)
    center = 0.5 * (left_limit + right_limit)
    span = 2.2 * abs(jump)

    inset = axis.inset_axes((0.62, 0.54, 0.35, 0.28))
    # Each side is evaluated with its own element, so the two one-sided limits
    # at the shared interface stay apart instead of collapsing onto each other.
    for element, (lower, upper) in (
        (interface_index - 1, (interface - half_window, interface)),
        (interface_index, (interface, interface + half_window)),
    ):
        piece = np.linspace(lower, upper, 200)
        element_lower, element_upper = boundaries[element], boundaries[element + 1]
        reference = 2 * (piece - element_lower) / (element_upper - element_lower) - 1
        inset.plot(
            piece, legendre.legval(reference, coefficients[element]),
            color=DISCRETE_COLOR, linewidth=2.0, dashes=(4, 2), zorder=4,
        )
    zoom = np.linspace(interface - half_window, interface + half_window, 400)
    inset.plot(zoom, exact_profile(zoom), color=EXACT_COLOR, linewidth=1.8, zorder=3)
    inset.axvline(interface, color=MESH_COLOR, linewidth=1.4, zorder=1)
    inset.annotate(
        '', xy=(interface, left_limit), xytext=(interface, right_limit),
        arrowprops=dict(arrowstyle='<->', color=ERROR_COLOR, linewidth=1.3,
                        shrinkA=0, shrinkB=0),
        zorder=6,
    )

    inset.set_xlim(interface - half_window, interface + half_window)
    inset.set_ylim(center - span, center + span)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color(MESH_COLOR)
    magnification = (DOMAIN[1] - DOMAIN[0]) / (2 * half_window)
    inset.set_title(
        f'element interface, magnified {magnification:.0f}x\n'
        f'the polynomials do not meet: a jump of {abs(jump):.0e}',
        fontsize=8.5, pad=4, color='#444444',
    )
    axis.annotate(
        '', xy=(interface, exact_profile(interface)),
        xytext=(0.64, 0.54), textcoords=axis.transAxes,
        arrowprops=dict(arrowstyle='-', color=MESH_COLOR, linewidth=1.0,
                        connectionstyle='arc3,rad=-0.2'),
        zorder=6,
    )
    return inset


def draw_error_box(axis, lines):
    """The error read-out in the upper left corner of a panel."""
    axis.text(
        0.02, 0.97, '\n'.join(lines), transform=axis.transAxes, va='top',
        ha='left', fontsize=10, color=ERROR_COLOR, linespacing=1.4,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor=ERROR_COLOR, alpha=0.9),
    )


def draw_bracket(axis, left, right, height, color=HIGHLIGHT_COLOR):
    """A bracket spanning a stretch of the column, to be labelled by a caller."""
    axis.plot(
        [left, left, right, right],
        [height - 0.04, height, height, height - 0.04],
        color=color, linewidth=1.3, solid_joinstyle='miter', zorder=6,
    )


def annotate_bracket(axis, left, right, height, text, color=HIGHLIGHT_COLOR):
    """Draw the bracket and point at it from the free corner of the panel.

    The text goes to the top right rather than above the bracket because the
    error box already owns the top left and the two collide otherwise.
    """
    draw_bracket(axis, left, right, height, color)
    axis.annotate(
        text, xy=(0.5 * (left + right), height), xycoords='data',
        xytext=(0.985, 0.95), textcoords='axes fraction',
        ha='right', va='top', fontsize=9.5, color=color, linespacing=1.35,
        arrowprops=dict(arrowstyle='->', color=color, linewidth=1.1,
                        connectionstyle='arc3,rad=0.2'),
        zorder=6,
    )


def draw_weno_reconstruction(axis, n_cells, z_fine, stencil_cell):
    """The stored cell averages and the wide-stencil reconstruction over them.

    Returns ``(error of the averages alone, error after reconstruction)``. Both
    are reported in the figure: the gap between them is the whole point, since
    it is where a high-order finite volume method keeps its accuracy.
    """
    boundaries = np.linspace(DOMAIN[0], DOMAIN[1], n_cells + 1)
    width = boundaries[1] - boundaries[0]
    centers = 0.5 * (boundaries[:-1] + boundaries[1:])
    averages = cell_averages(boundaries)
    blended, traces, _ = weno_reconstruction(averages)

    draw_mesh(axis, boundaries, line_width=0.6)
    # Light band: the five cells the reconstruction reads. Dark band: the one
    # cell it reconstructs. This pair is the figure's argument in one image.
    stencil = [stencil_cell + offset for offset in range(-2, 3)]
    for cell in stencil:
        axis.axvspan(boundaries[cell], boundaries[cell + 1],
                     color=HIGHLIGHT_COLOR, alpha=0.12, zorder=0)
    axis.axvspan(boundaries[stencil_cell], boundaries[stencil_cell + 1],
                 color=HIGHLIGHT_COLOR, alpha=0.26, zorder=0)

    reconstruction = evaluate_weno(blended, centers, width, z_fine)
    staircase = averages[np.clip(
        np.searchsorted(boundaries, z_fine, side='right') - 1, 0, n_cells - 1
    )]
    exact = exact_profile(z_fine)
    axis.fill_between(
        z_fine, exact, reconstruction, color=ERROR_COLOR, alpha=0.3, linewidth=0,
        zorder=2, label='approximation error',
    )
    axis.plot(z_fine, exact, color=EXACT_COLOR, linewidth=2.4, zorder=3,
              label='exact profile')
    # The staircase is drawn in the muted STATE_COLOR, not the strong discrete
    # colour: here it is the stored state, not the thing being judged.
    axis.hlines(
        averages, boundaries[:-1], boundaries[1:], color=STATE_COLOR,
        linewidth=1.6, zorder=3, label='stored unknowns (cell averages)',
    )
    axis.plot(centers, averages, linestyle='none', marker='o', markersize=4.0,
              markerfacecolor='white', markeredgecolor=STATE_COLOR,
              markeredgewidth=1.3, zorder=4)
    for index, center in enumerate(centers):
        inside = (z_fine >= center - 0.5 * width) & (z_fine <= center + 0.5 * width)
        axis.plot(
            z_fine[inside], reconstruction[inside], color=DISCRETE_COLOR,
            linewidth=2.0, dashes=(5, 2), zorder=5,
            label='WENO reconstruction' if index == 0 else None,
        )
    # The genuine scheme output: two one-sided values per interface.
    axis.plot(
        np.concatenate((boundaries[1:], boundaries[:-1])),
        np.concatenate((traces['right'], traces['left'])),
        linestyle='none', marker='.', markersize=5.5, color=DISCRETE_COLOR,
        zorder=6, label='reconstructed interface values',
    )

    annotate_bracket(
        axis, boundaries[stencil[0]], boundaries[stencil[-1] + 1], 1.12,
        'the reconstruction inside one cell (dark)\n'
        'reaches across 5 cell averages (shaded)',
    )
    return (np.max(np.abs(exact - staircase)),
            np.max(np.abs(exact - reconstruction)))


def highlight_element(axis, boundaries, element, degree):
    """Shade one DG element and bracket it: its unknowns are all its own.

    Deliberately mirrors ``draw_weno_reconstruction``'s shading so the two
    panels can be compared band to band.
    """
    axis.axvspan(boundaries[element], boundaries[element + 1],
                 color=HIGHLIGHT_COLOR, alpha=0.18, zorder=0)
    annotate_bracket(
        axis, boundaries[element], boundaries[element + 1], 1.12,
        f'all {degree + 1} unknowns of this polynomial\n'
        'live inside the one element (shaded)',
    )


def make_contrast_figure(n_cells=24, n_elements=4, degree=5, filename=None):
    """Figure 1: first-order FV against DG at equal degrees of freedom."""
    if n_cells != n_elements * (degree + 1):
        raise ValueError(
            'The panels must spend the same number of degrees of freedom: '
            f'{n_cells} cells against {n_elements * (degree + 1)} DG unknowns.'
        )

    z_fine = np.linspace(DOMAIN[0], DOMAIN[1], 4001)
    figure, (fv_axis, dg_axis) = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)

    fv_error = draw_finite_volumes(fv_axis, n_cells, z_fine)
    dg_error, boundaries, coefficients = draw_discontinuous_galerkin(
        dg_axis, n_elements, degree, z_fine,
    )

    fv_axis.set_title(
        f'Finite Volumes: {n_cells} cells, 1 unknown each\n'
        'the solution is a cell average, constant within a cell',
        fontsize=11.5, pad=10,
    )
    dg_axis.set_title(
        f'Discontinuous Galerkin: {n_elements} elements, {degree + 1} unknowns each\n'
        f'the solution is a degree-{degree} polynomial within an element',
        fontsize=11.5, pad=10,
    )
    for axis, error in ((fv_axis, fv_error), (dg_axis, dg_error)):
        axis.set_xlim(*DOMAIN)
        axis.set_xlabel('axial position $z$')
        axis.spines[['top', 'right']].set_visible(False)
        draw_error_box(axis, [f'largest error  {error:.1e}'])
    fv_axis.set_ylim(-0.12, 1.18)
    fv_axis.set_ylabel('concentration $c$')
    draw_interface_inset(dg_axis, boundaries, coefficients, interface_index=2)

    handles, labels = fv_axis.get_legend_handles_labels()
    order = ['exact profile', 'discrete solution', 'degree of freedom',
             'approximation error']
    figure.legend(
        [handles[labels.index(name)] for name in order], order,
        loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0),
    )
    figure.suptitle(
        f'The same {n_cells} degrees of freedom, spent two ways',
        fontsize=14, y=0.99,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))

    if filename is not None:
        figure.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    return figure, fv_error, dg_error, boundaries, coefficients


def make_order_figure(n_dofs=24, degrees=(1, 5), filename=None):
    """Figure 2: constants per cell, then DG at a low and a high degree.

    Read left to right, the polynomial degree rises while the number of cells
    or elements falls, and the same unknowns buy a very different picture. The
    low-degree panel is where the jumps at the element interfaces are plainly
    visible, so the word discontinuous earns itself before the high degree
    makes them invisible again.

    ``degrees`` can take more entries (24 unknowns also split as 8x3 and 6x4);
    the figure widens itself. Keep the lowest degree small or the figure loses
    its middle step.
    """
    for degree in degrees:
        if n_dofs % (degree + 1):
            raise ValueError(
                f'{n_dofs} unknowns do not split into elements of degree {degree}.'
            )

    z_fine = np.linspace(DOMAIN[0], DOMAIN[1], 4001)
    figure, axes = plt.subplots(
        1, 1 + len(degrees), figsize=(5.4 * (1 + len(degrees)), 4.8), sharey=True,
    )

    errors = [draw_finite_volumes(axes[0], n_dofs, z_fine)]
    axes[0].set_title(
        f'Finite Volumes\n{n_dofs} cells x 1 unknown, constant per cell',
        fontsize=11.5, pad=10,
    )
    for axis, degree in zip(axes[1:], degrees):
        n_elements = n_dofs // (degree + 1)
        error, _, _ = draw_discontinuous_galerkin(axis, n_elements, degree, z_fine)
        errors.append(error)
        shape = {1: 'linear', 2: 'quadratic'}.get(degree, f'degree {degree}')
        axis.set_title(
            f'Discontinuous Galerkin, $p = {degree}$\n'
            f'{n_elements} elements x {degree + 1} unknowns, {shape} per element',
            fontsize=11.5, pad=10,
        )

    for axis, error in zip(axes, errors):
        axis.set_xlim(*DOMAIN)
        axis.set_xlabel('axial position $z$')
        axis.spines[['top', 'right']].set_visible(False)
        draw_error_box(axis, [f'largest error  {error:.1e}'])
    axes[0].set_ylim(-0.18, 1.2)
    axes[0].set_ylabel('concentration $c$')

    # Point at the jumps in whichever panel has the lowest degree: that is the
    # only one where they can be seen without an inset.
    lowest_degree_axis = axes[1 + int(np.argmin(degrees))]
    jump_position = PEAK_CENTER + 1.1 * PEAK_WIDTH
    lowest_degree_axis.annotate(
        'the pieces do not meet:\nhere the jumps are plain to see',
        xy=(jump_position, exact_profile(jump_position)),
        xytext=(0.62, 0.72), textcoords='axes fraction', fontsize=9.5,
        color='#444444', ha='left', linespacing=1.35,
        arrowprops=dict(arrowstyle='->', color=MESH_COLOR, linewidth=1.1,
                        connectionstyle='arc3,rad=0.25'),
    )

    handles, labels = axes[0].get_legend_handles_labels()
    order = ['exact profile', 'discrete solution', 'degree of freedom',
             'approximation error']
    figure.legend(
        [handles[labels.index(name)] for name in order], order,
        loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0),
    )
    figure.suptitle(
        f'The same {n_dofs} degrees of freedom, arranged three ways',
        fontsize=14, y=0.99,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))

    if filename is not None:
        figure.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    return figure, errors


def make_weno_figure(n_cells=24, n_elements=4, degree=5, filename=None):
    """Figure 3: two routes to high order, a wide stencil or a rich element.

    The FV panel keeps one unknown per cell, exactly as in figure 1, and
    reaches fifth order by reconstructing from five neighbouring averages. The
    DG panel reaches a comparable accuracy without leaving its element. The
    shaded bands and the brackets are the point of the figure: one method pays
    in stencil width, the other in unknowns per element.
    """
    if n_cells != n_elements * (degree + 1):
        raise ValueError(
            'The panels must spend the same number of degrees of freedom: '
            f'{n_cells} cells against {n_elements * (degree + 1)} DG unknowns.'
        )

    z_fine = np.linspace(DOMAIN[0], DOMAIN[1], 4001)
    figure, (weno_axis, dg_axis) = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True)

    # A cell on the falling flank, where a reconstruction has real work to do,
    # and far enough from the ends for the full five-cell stencil to fit.
    stencil_cell = int(np.clip(
        round((PEAK_CENTER + 1.2 * PEAK_WIDTH) / (1.0 / n_cells) - 0.5),
        2, n_cells - 3,
    ))
    average_error, reconstruction_error = draw_weno_reconstruction(
        weno_axis, n_cells, z_fine, stencil_cell,
    )
    dg_error, boundaries, _ = draw_discontinuous_galerkin(
        dg_axis, n_elements, degree, z_fine,
    )
    highlight_element(
        dg_axis, boundaries,
        element=int(np.searchsorted(boundaries, PEAK_CENTER, side='right') - 1),
        degree=degree,
    )

    weno_axis.set_title(
        f'Finite Volumes with WENO: {n_cells} cells, still 1 unknown each\n'
        'fifth order by reaching across neighbouring cells',
        fontsize=11.5, pad=10,
    )
    dg_axis.set_title(
        f'Discontinuous Galerkin: {n_elements} elements, {degree + 1} unknowns each\n'
        f'degree {degree} without ever leaving the element',
        fontsize=11.5, pad=10,
    )
    draw_error_box(weno_axis, [
        f'cell averages alone    {average_error:.1e}',
        f'after reconstruction   {reconstruction_error:.1e}',
    ])
    draw_error_box(dg_axis, [f'largest error  {dg_error:.1e}'])

    # The two panels carry different curves, so each gets its own legend in a
    # fixed order rather than one shared figure legend.
    for axis in (weno_axis, dg_axis):
        axis.set_xlim(*DOMAIN)
        axis.set_xlabel('axial position $z$')
        axis.spines[['top', 'right']].set_visible(False)
        handles, labels = axis.get_legend_handles_labels()
        order = [name for name in (
            'exact profile', 'stored unknowns (cell averages)',
            'WENO reconstruction', 'reconstructed interface values',
            'discrete solution', 'degree of freedom', 'approximation error',
        ) if name in labels]
        axis.legend(
            [handles[labels.index(name)] for name in order], order,
            loc='lower right', fontsize=8.5, frameon=False,
            bbox_to_anchor=(1.0, 0.14),
        )
    weno_axis.set_ylim(-0.18, 1.36)
    weno_axis.set_ylabel('concentration $c$')

    figure.suptitle(
        'Two routes to high order: reach outward, or carry more inside',
        fontsize=14, y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    if filename is not None:
        figure.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    return figure, average_error, reconstruction_error, dg_error


def check_weno_against_textbook_formulas():
    """The candidate stencils must reproduce the standard WENO-JS expressions.

    Run before drawing anything. The reconstruction above derives its stencils
    from a linear solve rather than from tabulated coefficients, so this is the
    only thing standing between a typo and a plausible-looking wrong figure.
    """
    rng = np.random.default_rng(0)
    averages = rng.normal(size=9)
    padded = np.concatenate((
        np.repeat(averages[:1], 2), averages, np.repeat(averages[-1:], 2),
    ))
    minus2, minus1, center, plus1, plus2 = (padded[shift:shift + len(averages)]
                                            for shift in range(5))
    expected_right = np.stack((
        (2 * minus2 - 7 * minus1 + 11 * center) / 6,
        (-minus1 + 5 * center + 2 * plus1) / 6,
        (2 * center + 5 * plus1 - plus2) / 6,
    ))
    expected_left = np.stack((
        (-minus2 + 5 * minus1 + 2 * center) / 6,
        (2 * minus1 + 5 * center - plus1) / 6,
        (11 * center - 7 * plus1 + 2 * plus2) / 6,
    ))
    for index, offsets in enumerate(WENO_STENCILS):
        stencil_averages = np.stack([padded[2 + offset:2 + offset + len(averages)]
                                     for offset in offsets])
        candidate = np.linalg.solve(
            stencil_reconstruction_matrix(offsets), stencil_averages
        ).T
        for offset, expected in ((0.5, expected_right), (-0.5, expected_left)):
            values = np.array([polynomial.polyval(offset, row) for row in candidate])
            if not np.allclose(values, expected[index]):
                raise AssertionError(
                    f'Stencil {offsets} disagrees with the textbook reconstruction.'
                )
    # With the linear weights the blend of the three candidates must be the
    # degree-4 reconstruction of the full five-cell stencil.
    full = np.linalg.solve(
        stencil_reconstruction_matrix((-2, -1, 0, 1, 2)),
        np.stack([padded[offset:offset + len(averages)] for offset in range(5)]),
    ).T
    blended_right = sum(
        WENO_WEIGHTS_RIGHT[index] * expected_right[index] for index in range(3)
    )
    full_right = np.array([polynomial.polyval(0.5, row) for row in full])
    if not np.allclose(blended_right, full_right):
        raise AssertionError('The linear weights do not give the five-cell stencil.')


if __name__ == '__main__':
    check_weno_against_textbook_formulas()
    directory = Path(__file__).resolve().parent

    target = directory / 'approximation_FV_vs_DG.png'
    _, fv_error, dg_error, _, coefficients = make_contrast_figure(filename=target)
    print(f'FV largest error: {fv_error:.3e}')
    print(f'DG largest error: {dg_error:.3e}')
    print('interface jumps:',
          ' '.join(f'{jump:.2e}' for jump in interface_jumps(coefficients)))
    print(f'written to {target}')

    target = directory / 'approximation_FV_DG_orders.png'
    _, errors = make_order_figure(filename=target)
    print('orders, largest errors:', ' '.join(f'{error:.3e}' for error in errors))
    print(f'written to {target}')

    target = directory / 'approximation_WENO_vs_DG.png'
    _, average_error, reconstruction_error, dg_error = make_weno_figure(filename=target)
    print(f'FV cell averages: {average_error:.3e}')
    print(f'WENO reconstruction: {reconstruction_error:.3e}')
    print(f'DG: {dg_error:.3e}')
    print(f'written to {target}')
