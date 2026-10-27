# -*- coding: utf-8 -*-
"""

GMRES on the unit operation level: the Schur complement of the arrow head FV
units, and the four settings that control the iterative solve of it.

The module has two halves.

The first half is a paper model of the linear system. It assembles the Schur
complement of the lumped rate model with pores from the same blocks CADET
assembles, and runs a GMRES that mirrors the one in SUNDIALS, so that the
notebook can show residual histories, iteration counts and orthogonalization
work without a CADET installation and without waiting. Everything in this half
runs in a second.

The second half measures the real thing: the batch elution case reported on the
forum, run through CADET with GS_TYPE, MAX_KRYLOV, MAX_RESTARTS and
SCHUR_SAFETY varied one at a time. Those runs are slow, so measure() writes CSV
files and the plots read them. Pass force=True to measure again.

What the two halves are for: the paper model explains the mechanism, the CADET
runs confirm that the mechanism is what the wall clock is doing.

"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utility.setting_lrm_vs_lrmp as case
from utility.stiffness_metrics import COLORS, style_axis

DATA = os.path.join('data', 'gmres')
RUNS_CSV = os.path.join(DATA, 'runs.csv')
CURVES_CSV = os.path.join(DATA, 'chromatograms.csv')
WORK = os.path.join('sweeps', 'gmres')

# Machine epsilon in double precision. A residual norm cannot be reduced much
# below this fraction of the norm of the terms that cancel in it, no matter how
# many iterations are spent, so it is the floor every tolerance has to respect.
EPS = np.finfo(float).eps

# The paper model gives the *slope* of the accuracy requirement, not its
# absolute position: the size of the Newton residual at the moment the linear
# solve is called is not derivable from the source, and it multiplies the
# initial residual directly. This constant anchors the curve so that the usual
# SCHUR_SAFETY of 1e-8 lands exactly on machine epsilon, which is where the
# measurement puts the cliff (1e-7 runs in 11.4 s, 1e-8 in 340 s).
#
# Only the position is fitted, and only once. How fast the requirement moves
# with SCHUR_SAFETY, how it moves with the film rate, and how far apart two
# film rates sit are all the model's own.
CLIFF_ANCHOR = 470.0

# Tolerance IDAS hands down to the linear solver on a Newton step. CADET
# registers its own solver and then calls IDASetEpsLin(1) and
# IDASetLSNormFactor(1) (SimulatorImpl.cpp), so IDAS does not scale anything
# down and the value is the Newton convergence coefficient itself, 0.33.
# CADET then multiplies by sqrt(numDofs) and by SCHUR_SAFETY, see
# LumpedRateModelWithPores-LinearSolver.cpp.
OUTER_TOL = 0.33


# ---------------------------------------------------------------------------
# Half one: the Schur complement on paper
# ---------------------------------------------------------------------------

def bdf_alpha(n_steps=4222, end_time=case.END_TIME, order=5):
    """Representative value of the BDF coefficient alpha for this case.

    The linear system solved inside a time step is J = dF/dy + alpha dF/dydot
    with alpha = c / h, where h is the step size and c the leading coefficient
    of the BDF formula of the current order. Averaging the measured step count
    over the simulated time gives the value that the case actually ran at.

    Parameters
    ----------
    n_steps : int
        Time steps taken, as measured.
    end_time : float
        Simulated time in s.
    order : int
        BDF order, 1 to 5.

    Returns
    -------
    float
    """
    leading = {1: 1.0, 2: 3 / 2, 3: 11 / 6, 4: 25 / 12, 5: 137 / 60}[order]

    return leading * n_steps / end_time


def schur_complement(n_col=500, film_diffusion=1.0, par_radius=None,
                     alpha=None, component=0, col_dispersion=0.0):
    """Schur complement of the arrow head Jacobian of the LRMP, as a matrix.

    CADET writes the Jacobian of a column with particles as an arrow head

        J = [ J_0            J_{0,f} ]
            [       J_p      J_{p,f} ]
            [ J_{f,0} J_{f,p}   J_f  ]

    with one diagonal block per phase and a flux block that couples them, and
    eliminates the phase blocks into the Schur complement

        S = J_f - J_{f,0} J_0^-1 J_{0,f} - J_{f,p} J_p^-1 J_{p,f}.

    For the LRMP the flux unknown is the concentration difference
    j_f = c - c_p, so J_f = I, J_{f,0} = -I and J_{f,p} = +I, and the film
    coefficient sits in the two off-diagonal blocks (residualFlux in
    LumpedRateModelWithPores.cpp):

        J_{0,f} =  3 k_f / r_p * (1 - eps_c) / eps_c  = the film rate a_c
        J_{p,f} = -3 k_f / (eps_p r_p)                = a_p

    which leaves

        S = I + a_c J_0^-1 + a_p [J_p^-1]_{cp,cp}.

    The particle block is a two by two per component, alpha on dc_p/dt, alpha
    (1-eps_p)/eps_p on dq/dt and the binding Jacobian in the algebraic row, so
    its relevant entry is available in closed form. The bulk block is taken as
    alpha I plus a first order upwind convection operator, which is the WENO
    stencil in the smooth limit.

    The one thing to read off the formula: **the film coefficient multiplies
    everything that makes S differ from the identity.** At k_f = 0 the Schur
    complement is the identity and GMRES is done in one iteration; every decade
    of k_f pushes S one decade further away from it.

    Parameters
    ----------
    n_col : int
        Number of axial FV cells, i.e. the size of the Schur complement per
        component.
    film_diffusion : float
        Film coefficient k_f in m/s.
    par_radius : float
        Particle radius in m, defaults to the case value.
    alpha : float
        BDF coefficient, defaults to bdf_alpha().
    component : int
        Which component, since the binding constants differ.
    col_dispersion : float
        Axial dispersion in m^2/s. The reported case runs at 0.

    Returns
    -------
    (np.ndarray, dict)
        The Schur complement and the rates that went into it.
    """
    par_radius = case.PARTICLE_RADIUS if par_radius is None else par_radius
    alpha = bdf_alpha() if alpha is None else alpha

    velocity = case.interstitial_velocity()
    dz = case.BED_LENGTH / n_col
    convection = velocity / dz

    # Bulk block: alpha I + first order upwind convection (+ dispersion).
    bulk = np.zeros((n_col, n_col))
    lower = (np.arange(1, n_col), np.arange(n_col - 1))
    np.fill_diagonal(bulk, alpha + convection)
    bulk[lower] = -convection

    if col_dispersion:
        diffusion = col_dispersion / dz ** 2
        np.fill_diagonal(bulk, bulk.diagonal() + 2.0 * diffusion)
        bulk[lower] -= diffusion
        bulk[(np.arange(n_col - 1), np.arange(1, n_col))] = -diffusion

    # Particle block, (c_p, q) with linear binding in rapid equilibrium:
    # [[alpha, alpha beta_p], [k_a, -k_d]], of which only the (c_p, c_p) entry
    # of the inverse enters the Schur complement.
    beta_p = (1.0 - case.PARTICLE_POROSITY) / case.PARTICLE_POROSITY
    retention = case.LIN_KA[component] / case.LIN_KD[component]
    inv_particle = 1.0 / (alpha * (1.0 + beta_p * retention))

    rate_bulk = case.film_rate(film_diffusion, par_radius)
    rate_particle = (3.0 * film_diffusion
                     / (case.PARTICLE_POROSITY * par_radius))

    schur = rate_bulk * np.linalg.inv(bulk)
    schur[np.diag_indices(n_col)] += 1.0 + rate_particle * inv_particle

    return schur, {
        'alpha': alpha,
        'convection_rate': convection,
        'film_rate_bulk': rate_bulk,
        'film_rate_particle': rate_particle,
        'interstitial_velocity': velocity,
    }


def flux_weight(film_diffusion=1.0, par_radius=None, abstol=1e-08,
                reltol=1e-06, feed=None):
    """IDAS error weight of a flux unknown, 1 / (reltol |j_f| + abstol).

    GMRES measures its residual in this weight, and the weight is where the
    film coefficient enters a second time. The flux unknown is j_f = c - c_p,
    which the film transfer drives towards zero: at a film rate far above the
    convection rate the pore liquid follows the bulk liquid so closely that
    reltol |j_f| drops below abstol and the weight saturates at 1 / abstol.

    Parameters
    ----------
    film_diffusion : float
        Film coefficient k_f in m/s.
    par_radius : float
        Particle radius in m, defaults to the case value.
    abstol, reltol : float
        Integrator tolerances.
    feed : float
        Concentration scale, defaults to the feed of the case.

    Returns
    -------
    float
    """
    feed = case.FEED if feed is None else feed
    par_radius = case.PARTICLE_RADIUS if par_radius is None else par_radius

    # Balance of the two rates acting on c - c_p: convection fills the
    # difference in, film transfer removes it.
    convection = case.convection_rate()
    film = case.film_rate(film_diffusion, par_radius)
    scale = feed * convection / (film + convection)

    return 1.0 / (reltol * scale + abstol)


def schur_tolerance(schur_safety=1e-08, n_dof=4002, outer_tol=OUTER_TOL):
    """The tolerance CADET passes to GMRES.

    From LumpedRateModelWithPores-LinearSolver.cpp:

        tolerance = sqrt(numDofs()) * outerTol * SCHUR_SAFETY

    where outerTol is what IDAS asks of the linear solve on this Newton step.
    The tolerance is *absolute*, on the weight-scaled residual norm, which is
    why the weight and the size of the Schur complement both matter.

    Parameters
    ----------
    schur_safety : float
        SCHUR_SAFETY.
    n_dof : int
        Degrees of freedom of the unit operation. The reported case has
        2 + 2*500*(1+1) + 2000 = 4002.
    outer_tol : float
        Tolerance handed down by IDAS.

    Returns
    -------
    float
    """
    return np.sqrt(n_dof) * outer_tol * schur_safety


def gmres(matrix_vector, rhs, guess, tolerance, max_krylov, max_restarts,
          weight=None, orthogonalization='modified', reorthogonalize=True):
    """GMRES as SUNDIALS SPGMR runs it, with the counters the notebook needs.

    The algorithm: build an orthonormal basis of the Krylov space

        span{r_0, S r_0, S^2 r_0, ...}

    one vector at a time by Arnoldi, and take at each step the correction out
    of that space which minimizes the residual. Two consequences drive every
    one of the four settings:

      - iteration k needs k inner products to orthogonalize against the basis
        so far, so the work of m iterations grows like m^2 while the number of
        matrix-vector products grows like m. Long runs are quadratically
        expensive, which is what MAX_KRYLOV bounds.
      - the residual norm is not computed from the solution but carried along
        in the Hessenberg factorization. It is an estimate, and in finite
        precision it drifts below the true residual. Restarting recomputes the
        true residual from scratch, which is the other thing MAX_RESTARTS buys.

    The stopping test is the SPGMR one: the 2-norm of the *weight-scaled*
    residual against an absolute tolerance.

    Parameters
    ----------
    matrix_vector : callable
        Applies the matrix. In CADET one call is one bulk solve plus one
        particle solve, i.e. as expensive as the direct part of the step.
    rhs, guess : np.ndarray
        Right hand side and initial guess. CADET starts from the right hand
        side itself, not from zero.
    tolerance : float
        Absolute tolerance on the scaled residual, see schur_tolerance().
    max_krylov : int
        Krylov vectors per cycle, i.e. MAX_KRYLOV (0 means the full size).
    max_restarts : int
        Restart cycles after the first, i.e. MAX_RESTARTS.
    weight : np.ndarray
        Error weights, see flux_weight(). Defaults to ones.
    orthogonalization : string
        'modified' (GS_TYPE 1) or 'classical' (GS_TYPE 0).
    reorthogonalize : bool
        Whether the classical variant reorthogonalizes when the new vector lost
        most of its norm, as SUNClassicalGS does. Set False to see what the
        safeguard is protecting against.

    Returns
    -------
    dict
        solution, residuals (the estimate after every iteration),
        true_residual, matrix_vector_products, inner_products,
        vector_passes, reorthogonalizations, restarts, converged,
        orthogonality (||V^T V - I||_2 of the last basis built).
    """
    size = rhs.size
    max_krylov = size if not max_krylov else min(int(max_krylov), size)
    scale = np.ones(size) if weight is None else np.asarray(weight, float)
    if scale.size == 1:
        scale = np.full(size, float(scale))

    solution = np.array(guess, dtype=float)
    residuals = []
    products = 0
    inner_products = 0
    vector_passes = 0
    reorthogonalizations = 0
    orthogonality = np.nan
    cycles = 0

    for cycle in range(int(max_restarts) + 1):
        cycles = cycle + 1
        residual = scale * (rhs - matrix_vector(solution))
        beta = np.linalg.norm(residual)
        residuals.append(beta)

        if beta <= tolerance or beta == 0.0:
            break

        basis = np.empty((max_krylov + 1, size))
        basis[0] = residual / beta
        hessenberg = np.zeros((max_krylov + 1, max_krylov))
        cos = np.zeros(max_krylov)
        sin = np.zeros(max_krylov)
        target = np.zeros(max_krylov + 1)
        target[0] = beta
        last = -1

        for k in range(max_krylov):
            last = k
            candidate = scale * matrix_vector(basis[k] / scale)
            products += 1

            if orthogonalization == 'classical':
                before = np.linalg.norm(candidate)
                hessenberg[:k + 1, k] = basis[:k + 1] @ candidate
                candidate = candidate - hessenberg[:k + 1, k] @ basis[:k + 1]
                inner_products += k + 1
                # One fused pass of dot products and one linear combination,
                # whatever k is: two traversals of the basis.
                vector_passes += 2

                # SUNClassicalGS reorthogonalizes once if the vector lost
                # more than three decades of norm, which is where a single
                # classical pass stops being trustworthy.
                after = np.linalg.norm(candidate)
                if reorthogonalize and 1000.0 * after < before:
                    correction = basis[:k + 1] @ candidate
                    hessenberg[:k + 1, k] += correction
                    candidate = candidate - correction @ basis[:k + 1]
                    inner_products += k + 1
                    vector_passes += 2
                    reorthogonalizations += 1
            else:
                for i in range(k + 1):
                    hessenberg[i, k] = basis[i] @ candidate
                    candidate = candidate - hessenberg[i, k] * basis[i]
                inner_products += k + 1
                # One dot product and one axpy per basis vector, strictly in
                # sequence: 2(k+1) traversals.
                vector_passes += 2 * (k + 1)

            growth = np.linalg.norm(candidate)
            hessenberg[k + 1, k] = growth

            # Apply the previous Givens rotations, then one more to eliminate
            # the subdiagonal entry. The last entry of the rotated right hand
            # side is the residual norm of the least squares problem.
            for i in range(k):
                rotated = (cos[i] * hessenberg[i, k]
                           + sin[i] * hessenberg[i + 1, k])
                hessenberg[i + 1, k] = (-sin[i] * hessenberg[i, k]
                                        + cos[i] * hessenberg[i + 1, k])
                hessenberg[i, k] = rotated

            norm = np.hypot(hessenberg[k, k], hessenberg[k + 1, k])
            if norm == 0.0:
                break

            cos[k] = hessenberg[k, k] / norm
            sin[k] = hessenberg[k + 1, k] / norm
            hessenberg[k, k] = norm
            hessenberg[k + 1, k] = 0.0
            target[k + 1] = -sin[k] * target[k]
            target[k] = cos[k] * target[k]

            residuals.append(abs(target[k + 1]))

            if abs(target[k + 1]) <= tolerance or growth <= 1e-300:
                break

            if k + 1 < max_krylov:
                basis[k + 1] = candidate / growth

        if last >= 0:
            step = np.linalg.solve(
                np.triu(hessenberg[:last + 1, :last + 1]), target[:last + 1])
            solution = solution + (step @ basis[:last + 1]) / scale

            used = basis[:last + 1]
            gram = used @ used.T
            orthogonality = np.linalg.norm(gram - np.eye(last + 1), 2)

        if residuals[-1] <= tolerance:
            break

    true_residual = np.linalg.norm(scale * (rhs - matrix_vector(solution)))

    return {
        'solution': solution,
        'residuals': np.array(residuals),
        'true_residual': true_residual,
        'matrix_vector_products': products,
        'inner_products': inner_products,
        'vector_passes': vector_passes,
        'reorthogonalizations': reorthogonalizations,
        'restarts': cycles - 1,
        'converged': bool(residuals[-1] <= tolerance),
        'orthogonality': orthogonality,
    }


def _model_problem(n_col=500, film_diffusion=1.0, schur_safety=1e-08, seed=1):
    """Everything one GMRES call on the model Schur complement needs."""
    schur, rates = schur_complement(n_col=n_col, film_diffusion=film_diffusion)

    weight = flux_weight(film_diffusion=film_diffusion)

    # The right hand side is a Newton correction. The integrator's own unit of
    # size is the weighted RMS norm, in which 1 means "at the error tolerance",
    # so that is what a correction it still cares about looks like.
    rng = np.random.default_rng(seed)
    rhs = rng.standard_normal(n_col)
    rhs *= np.sqrt(n_col) / np.linalg.norm(weight * rhs)

    return {
        'schur': schur,
        'matrix_vector': lambda v: schur @ v,
        'rhs': rhs,
        'guess': rhs.copy(),
        'weight': np.full(n_col, weight),
        'tolerance': schur_tolerance(schur_safety),
        'rates': rates,
    }


def residual_budget(film_diffusion=1.0, schur_safety=1e-08, n_col=500,
                    residual_scale=None):
    """The factors that set how many digits GMRES is asked for.

    The stopping test is absolute, so what GMRES actually has to achieve is a
    *reduction*

        tolerance / ||w (b - S b)||

    taken at CADET's initial guess x_0 = b. Everything in the denominator
    grows with the film coefficient: S grows because the film rate multiplies
    every entry that makes S differ from the identity, and the weight w grows
    because c - c_p shrinks until w saturates at 1/ABSTOL. The numerator is
    linear in SCHUR_SAFETY. A reduction below about machine epsilon cannot be
    achieved at all, however many iterations are spent.

    One factor this function cannot derive is how large the Newton residual b
    is when the linear solve is called; `residual_scale` multiplies it, and the
    reduction scales inversely with it. It defaults to CLIFF_ANCHOR, which
    places the curve where the measurement says the cliff is. Pass 1.0 to see
    the uncalibrated model instead.

    Returns
    -------
    dict
    """
    if residual_scale is None:
        residual_scale = CLIFF_ANCHOR

    problem = _model_problem(n_col=n_col, film_diffusion=film_diffusion,
                             schur_safety=schur_safety)

    initial = residual_scale * np.linalg.norm(
        problem['weight'] * (problem['rhs']
                             - problem['matrix_vector'](problem['guess'])))

    return {
        'film_rate': problem['rates']['film_rate_bulk'],
        'schur_norm': np.linalg.norm(problem['schur'], 2),
        'schur_condition': np.linalg.cond(problem['schur']),
        'weight': problem['weight'][0],
        'initial_residual': initial,
        'tolerance': problem['tolerance'],
        'required_reduction': problem['tolerance'] / initial,
    }


def cost_against_reduction(film_diffusion=1.0, n_col=500,
                           reductions=None, max_krylov=0, max_restarts=10,
                           **options):
    """Work in one linear solve against the reduction it is asked for.

    This is the parameter free statement of the whole notebook: whatever sets
    the requirement - SCHUR_SAFETY, the film rate, the error weight, the size
    of the Newton residual - the *cost* depends on it through this one curve,
    and the curve has a cliff where the requirement meets double precision.

    Returns
    -------
    pd.DataFrame
    """
    if reductions is None:
        reductions = np.logspace(-6, -18, 13)

    problem = _model_problem(n_col=n_col, film_diffusion=film_diffusion)
    initial = np.linalg.norm(
        problem['weight'] * (problem['rhs']
                             - problem['matrix_vector'](problem['guess'])))

    rows = []
    for reduction in reductions:
        result = gmres(problem['matrix_vector'], problem['rhs'],
                       problem['guess'], reduction * initial, max_krylov,
                       max_restarts, weight=problem['weight'], **options)
        rows.append({
            'required reduction': reduction,
            'matrix-vector products': result['matrix_vector_products'],
            'inner products': result['inner_products'],
            'converged': result['converged'],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Half one, figures
# ---------------------------------------------------------------------------

def plot_arrow_head(n_col=12, figsize=(10.5, 3.3)):
    """The arrow head Jacobian, and the Schur complement it reduces to."""
    schur, _ = schur_complement(n_col=n_col)

    # A schematic of the sparsity pattern, drawn at a size one can read.
    size = 3 * n_col
    pattern = np.zeros((size, size))
    index = np.arange(n_col)
    pattern[index, index] = 1.0                            # bulk diagonal
    pattern[index[1:], index[:-1]] = 1.0                   # bulk convection
    pattern[n_col + index, n_col + index] = 1.0            # particle diagonal
    pattern[index, 2 * n_col + index] = 2.0                # J_{0,f}
    pattern[n_col + index, 2 * n_col + index] = 2.0        # J_{p,f}
    pattern[2 * n_col + index, index] = 2.0                # J_{f,0}
    pattern[2 * n_col + index, n_col + index] = 2.0        # J_{f,p}
    pattern[2 * n_col + index, 2 * n_col + index] = 1.0    # J_f = I

    figure, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(pattern, cmap='Blues', vmin=0.0, vmax=2.6,
                   interpolation='nearest')
    for cut in (n_col - 0.5, 2 * n_col - 0.5):
        axes[0].axhline(cut, color='white', linewidth=1.2)
        axes[0].axvline(cut, color='white', linewidth=1.2)
    for position, name in ((0.5 * n_col, 'bulk'), (1.5 * n_col, 'particle'),
                           (2.5 * n_col, 'flux')):
        axes[0].text(position, -1.6, name, ha='center', fontsize=8,
                     color='#404040')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title('the arrow head Jacobian $J$', fontsize=11)

    normalized = np.abs(schur) / np.abs(schur).max()
    image = axes[1].imshow(normalized, cmap='Blues', vmin=0.0, vmax=1.0,
                           interpolation='nearest')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title('the Schur complement $S$, dense', fontsize=11)
    figure.colorbar(image, ax=axes[1], fraction=0.046,
                    label='$|S_{ij}| / \\max |S|$')

    for index, film in enumerate((1e-06, 1e-02, 1e+00)):
        sub, rates = schur_complement(n_col=200, film_diffusion=film)
        row = np.abs(sub[-1, ::-1])
        label = f'film rate {rates["film_rate_bulk"]:.0e}' + r' s$^{-1}$'
        axes[2].semilogy(np.arange(row.size), row, color=COLORS[index],
                         linewidth=1.8, label=label)
    axes[2].set_xlim(0, 40)
    axes[2].set_xlabel('cells upstream')
    axes[2].set_ylabel(r'$|S_{ij}|$')
    axes[2].set_title('one row of $S$', fontsize=11)
    axes[2].legend(frameon=False, fontsize=8)
    style_axis(axes[2])

    figure.tight_layout()

    return figure, axes


def plot_residual_history(film_diffusion=1.0, n_col=500,
                          reductions=(1e-06, 1e-12, 1e-16, 1e-17),
                          figsize=(10.5, 4.0)):
    """What GMRES does on this Schur complement, asked for different digits."""
    problem = _model_problem(n_col=n_col, film_diffusion=film_diffusion)
    initial = np.linalg.norm(
        problem['weight'] * (problem['rhs']
                             - problem['matrix_vector'](problem['guess'])))

    figure, axes = plt.subplots(1, 2, figsize=figsize)
    rows = []

    for index, reduction in enumerate(sorted(reductions, reverse=True)):
        result = gmres(problem['matrix_vector'], problem['rhs'],
                       problem['guess'], reduction * initial, 0, 10,
                       weight=problem['weight'])
        history = result['residuals'] / initial

        axes[0].semilogy(np.arange(history.size), history,
                         color=COLORS[index % len(COLORS)], linewidth=1.8,
                         label=f'asked for {reduction:.0e}')
        axes[0].axhline(reduction, color=COLORS[index % len(COLORS)],
                        linestyle=':', linewidth=1.0)

        rows.append({
            'required reduction': f'{reduction:.0e}',
            'matrix-vector products': result['matrix_vector_products'],
            'inner products': result['inner_products'],
        })

    axes[0].axhline(EPS, color='#606060', linestyle='--', linewidth=1.3)
    axes[0].annotate('  double precision floor', xy=(0.0, EPS),
                     xycoords=('axes fraction', 'data'), xytext=(0, 3),
                     textcoords='offset points', fontsize=8, color='#606060')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('residual, relative to the first')
    axes[0].set_title('one more digit, twenty times the iterations',
                      fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc='lower left')
    style_axis(axes[0])

    table = pd.DataFrame(rows)
    costs = table['inner products'].astype(float)
    bars = axes[1].barh(table['required reduction'], costs,
                        color=[COLORS[i % len(COLORS)]
                               for i in range(len(costs))], height=0.55)
    axes[1].invert_yaxis()
    axes[1].bar_label(bars, labels=[f'  {value:,.0f}' for value in costs],
                      fontsize=9)
    axes[1].set_xscale('log')
    axes[1].set_xlim(10.0, costs.max() * 30.0)
    axes[1].set_xlabel('inner products in one linear solve')
    axes[1].set_ylabel('required reduction')
    axes[1].set_title('the quadratic term is what bites', fontsize=11)
    style_axis(axes[1])

    figure.tight_layout()

    return figure, axes, table


def plot_cost_vs_reduction(film_diffusion=1.0, n_col=500, figsize=(10.5, 3.8)):
    """The cliff, and the three things that push a solve over it."""
    table = cost_against_reduction(film_diffusion=film_diffusion, n_col=n_col)

    figure, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].loglog(table['required reduction'], table['inner products'], 'o-',
                   color=COLORS[1], linewidth=1.8, markersize=5,
                   label='inner products')
    axes[0].loglog(table['required reduction'],
                   table['matrix-vector products'], 's-', color=COLORS[3],
                   linewidth=1.8, markersize=5, label='matrix-vector products')
    axes[0].axvline(EPS, color='#606060', linestyle='--', linewidth=1.3)
    axes[0].annotate('machine epsilon',
                     xy=(EPS, table['inner products'].max()),
                     xytext=(5, -4), textcoords='offset points', fontsize=8,
                     color='#606060')
    axes[0].invert_xaxis()
    axes[0].set_xlabel('reduction GMRES is asked for')
    axes[0].set_ylabel('work in one linear solve')
    axes[0].set_title('the cliff is at double precision', fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    style_axis(axes[0])

    films = np.logspace(-6, 0, 13)
    budgets = [residual_budget(film_diffusion=film, n_col=n_col)
               for film in films]
    rates = np.array([b['film_rate'] for b in budgets])

    axes[1].loglog(rates, [b['schur_norm'] for b in budgets], 'o-',
                   color=COLORS[1], linewidth=1.8, markersize=5,
                   label=r'$\|S\|_2$')
    axes[1].loglog(rates, [b['weight'] for b in budgets], 's-',
                   color=COLORS[2], linewidth=1.8, markersize=5,
                   label=r'error weight $1/(r_{tol}|j_f| + a_{tol})$')
    axes[1].loglog(rates, [b['schur_condition'] for b in budgets], '^-',
                   color=COLORS[0], linewidth=1.8, markersize=5,
                   label=r'$\mathrm{cond}(S)$')
    axes[1].set_xlabel(r'film rate / s$^{-1}$')
    axes[1].set_title('two of the three factors grow;\nthe conditioning does '
                      'not', fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, loc='upper left')
    style_axis(axes[1])

    safeties = np.logspace(-10, -2, 9)
    for index, rate_label in enumerate(('film rate 1.4e6', 'film rate 1.4e2')):
        film = 1.0 if index == 0 else 1e-04
        budget = residual_budget(film_diffusion=film, n_col=n_col,
                                 schur_safety=1.0)
        required = [safety * budget['required_reduction']
                    for safety in safeties]
        axes[2].loglog(safeties, required, 'o-', color=COLORS[index],
                       linewidth=1.8, markersize=5,
                       label=rate_label + r' s$^{-1}$')
    axes[2].axhline(EPS, color='#606060', linestyle='--', linewidth=1.3)
    axes[2].annotate('  unreachable below here', xy=(safeties[0], EPS),
                     xytext=(0, 4), textcoords='offset points', fontsize=8,
                     color='#606060')
    axes[2].axvline(1e-08, color='#606060', linestyle=':', linewidth=1.2)
    axes[2].annotate('1e-8', xy=(1e-08, 0.0),
                     xycoords=('data', 'axes fraction'),
                     xytext=(4, 4), textcoords='offset points', fontsize=8,
                     color='#606060')
    axes[2].set_xlabel('SCHUR_SAFETY')
    axes[2].set_ylabel('required reduction')
    axes[2].set_title('SCHUR_SAFETY moves the requirement one\nfor one '
                      '(position anchored to the measurement)', fontsize=9)
    axes[2].legend(frameon=False, fontsize=8, loc='upper left')
    style_axis(axes[2])

    figure.tight_layout()

    return figure, axes, table


def _tolerance_for(problem, reduction):
    """Absolute tolerance that asks for a given reduction on this problem."""
    initial = np.linalg.norm(
        problem['weight'] * (problem['rhs']
                             - problem['matrix_vector'](problem['guess'])))

    return reduction * initial


def krylov_cost(film_diffusion=1.0, n_col=500, reduction=1e-17,
                lengths=(10, 25, 50, 100, 250, 500), max_restarts=10):
    """Cost of one linear solve against the Krylov space length.

    Run at a requirement past the floor, i.e. in the regime the default
    settings put the reported case in, because that is where MAX_KRYLOV is
    worth anything. Off the cliff every length gives the same handful of
    iterations and the setting does not matter.
    """
    problem = _model_problem(n_col=n_col, film_diffusion=film_diffusion)
    tolerance = _tolerance_for(problem, reduction)

    rows = []
    for length in lengths:
        result = gmres(problem['matrix_vector'], problem['rhs'],
                       problem['guess'], tolerance, length,
                       max_restarts, weight=problem['weight'])
        rows.append({
            'MAX_KRYLOV': length,
            'matrix-vector products': result['matrix_vector_products'],
            'inner products': result['inner_products'],
            'restart cycles used': result['restarts'],
            'converged': result['converged'],
            'basis vectors stored': min(length, n_col) + 1,
        })

    return pd.DataFrame(rows)


def plot_krylov_cost(film_diffusion=1.0, n_col=500, reduction=1e-17,
                     figsize=(10.5, 3.6)):
    """Why a shorter Krylov space can be cheaper than a longer one."""
    table = krylov_cost(film_diffusion=film_diffusion, n_col=n_col,
                        reduction=reduction)

    figure, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].plot(table['MAX_KRYLOV'], table['matrix-vector products'], 'o-',
                 color=COLORS[0], linewidth=1.8, markersize=6)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('MAX_KRYLOV')
    axes[0].set_ylabel('matrix-vector products')
    axes[0].set_title('work that scales linearly', fontsize=10)
    style_axis(axes[0])

    axes[1].plot(table['MAX_KRYLOV'], table['inner products'], 'o-',
                 color=COLORS[1], linewidth=1.8, markersize=6)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('MAX_KRYLOV')
    axes[1].set_ylabel('inner products')
    axes[1].set_title('work that scales quadratically', fontsize=10)
    style_axis(axes[1])

    # The model problem is one component; CADET's Schur complement holds all
    # of them, so the basis it stores is NCOMP times as wide.
    schur_size = n_col * case.N_COMP
    memory = table['basis vectors stored'] * schur_size * 8 / 1024 ** 2
    axes[2].plot(table['MAX_KRYLOV'], memory, 'o-', color=COLORS[2],
                 linewidth=1.8, markersize=6)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('MAX_KRYLOV')
    axes[2].set_ylabel('MB of Krylov basis')
    axes[2].set_title('and memory held for the run\n'
                      f'(Schur complement of {schur_size})', fontsize=10)
    style_axis(axes[2])

    figure.tight_layout()

    return figure, axes, table


def restart_cost(film_diffusion=1.0, n_col=500, reduction=1e-16,
                 max_krylov=None, restarts=(0, 1, 2, 5, 10, 30)):
    """Cost and success of one linear solve against MAX_RESTARTS.

    MAX_RESTARTS only does anything once MAX_KRYLOV is too short to finish in
    one cycle, so it defaults to the same short Krylov space the measured
    sweep uses. The residual column is the point: a solve denied its restarts
    does not fail, it returns.
    """
    max_krylov = RESTART_KRYLOV_SHORT if max_krylov is None else max_krylov
    problem = _model_problem(n_col=n_col, film_diffusion=film_diffusion)
    tolerance = _tolerance_for(problem, reduction)

    rows = []
    for allowed in restarts:
        result = gmres(problem['matrix_vector'], problem['rhs'],
                       problem['guess'], tolerance, max_krylov,
                       allowed, weight=problem['weight'])
        rows.append({
            'MAX_RESTARTS': allowed,
            'matrix-vector products': result['matrix_vector_products'],
            'inner products': result['inner_products'],
            'cycles used': result['restarts'],
            'converged': result['converged'],
            'residual reached': f"{result['residuals'][-1]:.2e}",
        })

    return pd.DataFrame(rows)


def gram_schmidt_quality(film_diffusion=1.0, n_col=500, max_krylov=40,
                         reduction=1e-18):
    """The two orthogonalizations, side by side.

    Both do the same k+1 inner products at iteration k, so an operation count
    cannot tell them apart. What differs is how those inner products are
    organised: modified Gram-Schmidt subtracts one basis vector at a time, so
    it traverses memory 2(k+1) times and cannot be vectorized across the
    basis; classical Gram-Schmidt does one fused multi-dot and one linear
    combination, two traversals whatever k is. That is a throughput
    difference, and it only shows up when the Krylov space is long.

    Run with a requirement nothing can meet, so that both build the same
    number of basis vectors and the comparison is like for like.

    Returns
    -------
    pd.DataFrame
    """
    problem = _model_problem(n_col=n_col, film_diffusion=film_diffusion)
    tolerance = _tolerance_for(problem, reduction)

    variants = (
        ('GS_TYPE 1, modified (default)',
         dict(orthogonalization='modified')),
        ('GS_TYPE 0, classical with reorthogonalization',
         dict(orthogonalization='classical', reorthogonalize=True)),
    )

    rows = []
    for label, options in variants:
        result = gmres(problem['matrix_vector'], problem['rhs'],
                       problem['guess'], tolerance, max_krylov, 0,
                       weight=problem['weight'], **options)
        rows.append({
            'orthogonalization': label,
            'iterations': result['matrix_vector_products'],
            'inner products': result['inner_products'],
            'passes over the basis': result['vector_passes'],
            'reorthogonalizations triggered': result['reorthogonalizations'],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Half two: the same four settings, measured in CADET
# ---------------------------------------------------------------------------

FILM_COEFFICIENTS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0)
SCHUR_SAFETIES = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2)
KRYLOV_LENGTHS = (0, 250, 100, 50, 25, 10)
RESTART_COUNTS = (0, 1, 2, 5, 10, 30)
GS_VARIANTS = ((1, 0), (0, 0), (1, 100), (0, 100))

# MAX_RESTARTS only bites once the Krylov space is too short to finish inside
# one cycle. It is swept twice: at a Krylov space that is long enough (where it
# turns out to do nothing at all) and at one that is not.
RESTART_KRYLOV = 25
RESTART_KRYLOV_SHORT = 5


def _plan():
    """Every run this study needs, as (key, group, label, settings)."""
    pathological = dict(transport_model='LRMP', film_diffusion=1.0)

    plan = [
        ('lrm', 'baseline', 'LRM (no particles, no GMRES)',
         dict(transport_model='LRM')),
        ('lrmp', 'baseline', 'LRMP, default settings', dict(**pathological)),
        ('sparse', 'baseline', 'LRMP, global sparse solver',
         dict(arrow_head=False, **pathological)),
    ]

    for film in FILM_COEFFICIENTS:
        plan.append((f'film_{film:.0e}', 'film', f'{film:.0e}',
                     dict(transport_model='LRMP', film_diffusion=film)))

    for safety in SCHUR_SAFETIES:
        plan.append((f'safety_{safety:.0e}', 'schur_safety', f'{safety:.0e}',
                     dict(schur_safety=safety, **pathological)))

    for length in KRYLOV_LENGTHS:
        plan.append((f'krylov_{length}', 'max_krylov',
                     'full (0)' if length == 0 else str(length),
                     dict(max_krylov=length, **pathological)))

    for allowed in RESTART_COUNTS:
        plan.append((f'restart_{allowed}', 'max_restarts', str(allowed),
                     dict(max_krylov=RESTART_KRYLOV, max_restarts=allowed,
                          **pathological)))

    for allowed in RESTART_COUNTS:
        plan.append((f'restart_short_{allowed}', 'max_restarts_short',
                     str(allowed),
                     dict(max_krylov=RESTART_KRYLOV_SHORT,
                          max_restarts=allowed, **pathological)))

    for gs_type, length in GS_VARIANTS:
        name = 'modified' if gs_type else 'classical'
        krylov = 'full' if length == 0 else str(length)
        plan.append((f'gs_{gs_type}_{length}', 'gs_type',
                     f'{name}, MAX_KRYLOV {krylov}',
                     dict(gs_type=gs_type, max_krylov=length, **pathological)))

    return plan


def measure(force=False, timeout=900.0, verbose=True):
    """Run every configuration once and write the two CSV files.

    Runs are timed, so they are done strictly one after another. The whole set
    takes roughly half an hour, almost all of it in the handful of runs that
    sit on the cliff.

    Runs that are already in the CSV are kept, so adding a configuration to
    _plan() and calling measure() again costs only the new runs. force=True
    discards everything and measures from scratch.

    Parameters
    ----------
    force : bool
        Measure everything again, including runs that are already recorded.
    timeout : float
        Wall clock limit handed to CADET per run, so that a bad setting cannot
        run away. A run that hits it is recorded with no timing.
    verbose : bool
        Print each run as it finishes.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
    """
    import h5py
    from cadet import Cadet

    import utility.cadet_runtime as cadet_runtime
    import utility.stiffness_metrics as metrics

    plan = _plan()
    done = {}
    curves = {}

    if not force and os.path.exists(RUNS_CSV):
        previous = pd.read_csv(RUNS_CSV)
        done = {row['key']: row for row in previous.to_dict('records')}
        curves = pd.read_csv(CURVES_CSV).to_dict('list')

        if all(key in done for key, _, _, _ in plan):
            return load()

    cadet_runtime.configure()
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)

    records = []
    reference = None

    # The deviation column is measured against the LRM, so its outlet is
    # needed even when that run itself is already recorded.
    lrm_file = os.path.join(WORK, 'lrm.h5')
    if 'lrm' in done and os.path.exists(lrm_file):
        with h5py.File(lrm_file, 'r') as handle:
            reference = handle['output/solution/unit_002/SOLUTION_OUTLET'][()]

    for key, group, label, settings in plan:
        if key in done:
            records.append(done[key])
            continue

        model = Cadet()
        model.root = case.get_model(**settings)
        model.root.input.solver.TIMEOUT = float(timeout)
        model.filename = os.path.join(WORK, f'{key}.h5')
        model.save()

        start = time.time()
        info = model.run_simulation()
        wall = time.time() - start

        if info.return_code:
            if verbose:
                print(f'{key:>14} FAILED  {(info.error_message or "")[:60]}',
                      flush=True)
            records.append({
                'key': key, 'group': group, 'label': label,
                'film_diffusion': settings.get('film_diffusion', np.nan),
                'gs_type': settings.get('gs_type', 1),
                'max_krylov': settings.get('max_krylov', 0),
                'max_restarts': settings.get('max_restarts', 10),
                'schur_safety': settings.get('schur_safety', 1e-08),
                'film_rate': case.film_rate(
                    settings.get('film_diffusion', case.FILM_DIFFUSION)),
                'steps': np.nan, 'wall_time': np.nan, 'ms_per_step': np.nan,
                'error_test_failures': np.nan, 'max_deviation': np.nan,
                'failed': True,
            })
            continue

        measured = metrics.measure(model.filename, label=key)
        with h5py.File(model.filename, 'r') as handle:
            times = handle['output/solution/SOLUTION_TIMES'][()]
            outlet = handle['output/solution/unit_002/SOLUTION_OUTLET'][()]

        if key == 'lrm':
            reference = outlet
            curves['time'] = times
            curves['LRM component 0'] = outlet[:, 0]
            curves['LRM component 1'] = outlet[:, 1]
        elif key == 'lrmp':
            curves['LRMP component 0'] = outlet[:, 0]
            curves['LRMP component 1'] = outlet[:, 1]

        records.append({
            'key': key,
            'group': group,
            'label': label,
            'film_diffusion': settings.get('film_diffusion', np.nan),
            'gs_type': settings.get('gs_type', 1),
            'max_krylov': settings.get('max_krylov', 0),
            'max_restarts': settings.get('max_restarts', 10),
            'schur_safety': settings.get('schur_safety', 1e-08),
            'film_rate': case.film_rate(
                settings.get('film_diffusion', case.FILM_DIFFUSION),
                settings.get('par_radius', case.PARTICLE_RADIUS)),
            'steps': measured.n_steps,
            'wall_time': wall,
            'ms_per_step': 1000.0 * wall / measured.n_steps,
            'error_test_failures': measured.n_error_test_failures,
            'max_deviation': (np.abs(outlet - reference).max()
                              if reference is not None else np.nan),
            'failed': False,
        })

        if verbose:
            print(f'{key:>14} {wall:>9.2f} s  {measured.n_steps:>6} steps',
                  flush=True)

    pd.DataFrame(records).to_csv(RUNS_CSV, index=False)
    pd.DataFrame(curves).to_csv(CURVES_CSV, index=False)

    return load()


def load():
    """The measured runs and the two chromatograms."""
    return pd.read_csv(RUNS_CSV), pd.read_csv(CURVES_CSV)


def _baselines(runs):
    return (runs[runs.key == 'lrm'].iloc[0],
            runs[runs.key == 'lrmp'].iloc[0],
            runs[runs.key == 'sparse'].iloc[0])


def plot_same_problem(figsize=(10.5, 3.6)):
    """The two models return the same chromatogram. That is the premise."""
    runs, curves = load()
    _, lrmp, _ = _baselines(runs)

    figure, axes = plt.subplots(1, 2, figsize=figsize)

    for component in (0, 1):
        axes[0].plot(curves['time'] / 60.0,
                     curves[f'LRM component {component}'],
                     color=COLORS[component], linewidth=3.2, alpha=0.35,
                     label=f'LRM, component {component + 1}')
        axes[0].plot(curves['time'] / 60.0,
                     curves[f'LRMP component {component}'],
                     color=COLORS[component], linewidth=1.3, linestyle='--',
                     label=f'LRMP, component {component + 1}')

    axes[0].set_xlabel('time / min')
    axes[0].set_ylabel(r'concentration / mol m$^{-3}$')
    axes[0].set_title('the same answer', fontsize=11)
    axes[0].legend(frameon=False, fontsize=8)
    style_axis(axes[0])

    difference = np.abs(curves['LRMP component 0'] - curves['LRM component 0'])
    axes[1].plot(curves['time'] / 60.0, difference, color=COLORS[1],
                 linewidth=1.6)
    axes[1].set_yscale('log')
    top = float(lrmp.max_deviation) * 3.0
    axes[1].set_ylim(top * 1e-6, top)
    axes[1].set_xlabel('time / min')
    axes[1].set_ylabel('|LRMP - LRM|')
    axes[1].set_title(f'difference, at most {lrmp.max_deviation:.1e} '
                      f'of a feed of {case.FEED:g}', fontsize=11)
    style_axis(axes[1])

    figure.tight_layout()

    return figure, axes


def plot_where_the_time_goes(figsize=(10.5, 3.4)):
    """Same steps, far more cost per step: not the integrator's doing."""
    runs, _ = load()
    lrm, lrmp, _ = _baselines(runs)

    figure, axes = plt.subplots(1, 3, figsize=figsize)
    names = ['LRM', 'LRMP']

    panels = (
        ('time steps taken', [lrm.steps, lrmp.steps], False),
        ('failed error tests',
         [lrm.error_test_failures, lrmp.error_test_failures], False),
        ('ms per step', [lrm.ms_per_step, lrmp.ms_per_step], True),
    )

    for axis, (title, values, log) in zip(axes, panels):
        bars = axis.bar(names, values, color=[COLORS[0], COLORS[1]],
                        width=0.55)
        if log:
            axis.set_yscale('log')
        axis.bar_label(bars, fmt='%.4g', fontsize=9, padding=2)
        axis.set_title(title, fontsize=11)
        axis.margins(y=0.25)
        style_axis(axis)

    axes[2].set_title(f'ms per step  ({lrmp.ms_per_step / lrm.ms_per_step:.0f}'
                      r'$\times$)', fontsize=11)
    figure.tight_layout()

    return figure, axes


def plot_the_cliff(figsize=(10.5, 3.8)):
    """Measured cost against the film rate, and what the digits bought."""
    runs, _ = load()
    film = runs[runs.group == 'film'].sort_values('film_rate')
    lrm, _, _ = _baselines(runs)

    figure, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(film.film_rate, film.ms_per_step, 'o-', color=COLORS[1],
                 linewidth=1.8, markersize=6, label='LRMP, measured')
    axes[0].axhline(lrm.ms_per_step, color=COLORS[0], linestyle=':',
                    linewidth=1.5, label='LRM, which runs no GMRES')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'film rate $3k_f/r_p \cdot (1-\varepsilon_c)'
                       r'/\varepsilon_c$ / s$^{-1}$')
    axes[0].set_ylabel('ms per time step')
    axes[0].set_title('six decades cost nothing, the seventh costs '
                      'everything', fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    style_axis(axes[0])

    axes[1].plot(film.film_rate, film.max_deviation, 'o-', color=COLORS[1],
                 linewidth=1.8, markersize=6)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel(r'film rate / s$^{-1}$')
    axes[1].set_ylabel('max deviation from the LRM')
    axes[1].set_title('and the answer stopped moving long before',
                      fontsize=10)
    style_axis(axes[1])

    figure.tight_layout()

    return figure, axes


def _sweep_panel(axis, rows, baseline, xlabel, title):
    labels = [str(value) for value in rows['label']]
    values = rows['wall_time'].to_numpy(dtype=float)
    colors = [COLORS[1] if value > 10.0 * baseline else COLORS[2]
              for value in values]

    bars = axis.bar(labels, np.nan_to_num(values, nan=0.0), color=colors,
                    width=0.6)
    axis.bar_label(bars, labels=[('did not finish' if np.isnan(value)
                                  else f'{value:.1f}')
                                 for value in values], fontsize=8, padding=2)
    axis.set_yscale('log')
    axis.set_ylim(1.0, 1000.0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel('wall clock / s')
    axis.set_title(title, fontsize=10)
    axis.axhline(baseline, color=COLORS[0], linestyle=':', linewidth=1.4)
    style_axis(axis)

    return axis


def plot_parameter_sweeps(figsize=(11.0, 7.0)):
    """Each of the four settings, swept on the identical model."""
    runs, _ = load()
    lrm, _, _ = _baselines(runs)

    figure, axes = plt.subplots(2, 2, figsize=figsize)

    _sweep_panel(axes[0, 0], runs[runs.group == 'schur_safety'],
                 lrm.wall_time, 'SCHUR_SAFETY',
                 'SCHUR_SAFETY (1e-8 is the usual value, leftmost)')
    _sweep_panel(axes[0, 1], runs[runs.group == 'max_krylov'],
                 lrm.wall_time, 'MAX_KRYLOV',
                 'MAX_KRYLOV (0 is the usual value, leftmost)')
    _sweep_panel(axes[1, 0], runs[runs.group == 'max_restarts_short'],
                 lrm.wall_time, 'MAX_RESTARTS',
                 f'MAX_RESTARTS at MAX_KRYLOV {RESTART_KRYLOV_SHORT}')
    gs_rows = runs[runs.group == 'gs_type'].copy()
    gs_rows['label'] = [value.replace('MAX_KRYLOV ', 'K=')
                        for value in gs_rows['label']]
    _sweep_panel(axes[1, 1], gs_rows, lrm.wall_time, '',
                 'GS_TYPE, at two Krylov space lengths')
    axes[1, 1].tick_params(axis='x', labelsize=8)

    figure.suptitle('dotted line: the LRM, which has no GMRES to configure',
                    fontsize=10, color=COLORS[0])
    figure.tight_layout()

    return figure, axes


def plot_silent_failure(figsize=(11.0, 3.4)):
    """What too few restarts does, and how it announces itself: it does not.

    At MAX_KRYLOV 5 the Schur solve needs more than one cycle. Denied the
    restarts, GMRES returns whatever it has, the unit operation discards the
    flag that says so, and the integrator pays for it without ever being told
    what happened.
    """
    runs, _ = load()
    rows = runs[runs.group == 'max_restarts_short']
    converged = rows[rows.max_restarts >= 5].iloc[0]

    figure, axes = plt.subplots(1, 4, figsize=figsize)

    panels = (
        ('wall clock / s', rows.wall_time, converged.wall_time, False),
        ('time steps taken', rows.steps, converged.steps, False),
        ('failed error tests', rows.error_test_failures,
         converged.error_test_failures, True),
        ('max deviation from the LRM', rows.max_deviation,
         converged.max_deviation, True),
    )

    for axis, (title, values, reference, log) in zip(axes, panels):
        labels = [str(value) for value in rows.max_restarts]
        colors = [COLORS[1] if value > 1.05 * reference else COLORS[2]
                  for value in values]
        axis.bar(labels, values, color=colors, width=0.6)
        if log:
            axis.set_yscale('log')
        axis.axhline(reference, color=COLORS[0], linestyle=':', linewidth=1.4)
        axis.set_xlabel('MAX_RESTARTS')
        axis.set_title(title, fontsize=10)
        axis.margins(y=0.2)
        style_axis(axis)

    figure.suptitle(f'MAX_KRYLOV {RESTART_KRYLOV_SHORT}, everything else at '
                    'its usual value. No warning is issued in any of these '
                    'runs.', fontsize=10)
    figure.tight_layout()

    return figure, axes


def summary(groups=('baseline', 'schur_safety', 'max_krylov', 'max_restarts',
                    'max_restarts_short', 'gs_type')):
    """Every measured run, with the answer it returned."""
    runs, _ = load()
    lrm, _, _ = _baselines(runs)

    rows = runs[runs.group.isin(groups)].copy()

    return pd.DataFrame({
        'group': rows.group,
        'setting': rows.label,
        'GS_TYPE': rows.gs_type,
        'MAX_KRYLOV': rows.max_krylov,
        'MAX_RESTARTS': rows.max_restarts,
        'SCHUR_SAFETY': [f'{value:.0e}' for value in rows.schur_safety],
        'wall clock / s': rows.wall_time.round(2),
        'vs LRM': (rows.wall_time / lrm.wall_time).round(1),
        'steps': rows.steps,
        'max deviation from LRM': [f'{value:.3e}'
                                   for value in rows.max_deviation],
    }).reset_index(drop=True)
