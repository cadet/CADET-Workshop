# -*- coding: utf-8 -*-
"""

Why an LRMP with a very large film coefficient costs 160 times an LRM.

Measurement and figures for the batch elution case reported on the forum. The
runs are slow enough that the notebook should not repeat them, so `measure()`
writes two CSV files and the plots read those. Pass force=True to measure
again.

The finding, in one line: the gap is entirely cost *per step*, the step counts
are the same, and it comes from an unpreconditioned GMRES being asked for eight
digits it does not need.

"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utility.setting_lrm_vs_lrmp as case
from utility.stiffness_metrics import COLORS, style_axis

DATA = os.path.join('data', 'lrmp_cost')
RUNS_CSV = os.path.join(DATA, 'runs.csv')
CURVES_CSV = os.path.join(DATA, 'chromatograms.csv')
WORK = os.path.join('sweeps', 'lrmp_cost')

FILM_COEFFICIENTS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0)
FIXED_RATE_FACTORS = (1e0, 1e2, 1e4, 1e6)

SOLVER_VARIANTS = (
    ('arrow head, SCHUR_SAFETY 1e-8 (default)', dict()),
    ('arrow head, SCHUR_SAFETY 1e-6', dict(schur_safety=1e-6)),
    ('arrow head, SCHUR_SAFETY 1e-4', dict(schur_safety=1e-4)),
    ('arrow head, SCHUR_SAFETY 1e-2', dict(schur_safety=1e-2)),
    ('arrow head, MAX_KRYLOV 100', dict(max_krylov=100)),
    ('global sparse solver', dict(arrow_head=False)),
)


def _plan():
    """Every run this study needs, as (key, group, label, settings)."""
    plan = [
        ('lrm', 'baseline', 'LRM', dict(transport_model='LRM')),
        ('lrmp', 'baseline', 'LRMP, k_f = 1 m/s',
         dict(transport_model='LRMP', film_diffusion=1.0)),
    ]

    for film in FILM_COEFFICIENTS:
        plan.append((f'film_{film:.0e}', 'film', f'{film:.0e}',
                     dict(transport_model='LRMP', film_diffusion=film)))

    for factor in FIXED_RATE_FACTORS:
        plan.append((
            f'fixed_{factor:.0e}', 'fixed_rate', f'{1e-6 * factor:.0e}',
            dict(transport_model='LRMP', film_diffusion=1e-6 * factor,
                 par_radius=case.PARTICLE_RADIUS * factor)))

    for index, (label, extra) in enumerate(SOLVER_VARIANTS):
        plan.append((f'solver_{index}', 'solver', label,
                     dict(transport_model='LRMP', film_diffusion=1.0, **extra)))

    return plan


def measure(force=False):
    """Run everything once and write the two CSV files.

    Runs are timed, so they are done strictly one after another. The whole set
    takes roughly ten minutes, almost all of it in the one pathological case.
    """
    import h5py
    from cadet import Cadet

    import utility.cadet_runtime as cadet_runtime
    import utility.stiffness_metrics as metrics

    if not force and os.path.exists(RUNS_CSV):
        return load()

    cadet_runtime.configure()
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)

    records = []
    curves = {}
    reference = None

    for key, group, label, settings in _plan():
        model = Cadet()
        model.root = case.get_model(**settings)
        model.filename = os.path.join(WORK, f'{key}.h5')
        model.save()

        start = time.time()
        info = model.run_simulation()
        wall = time.time() - start

        if info.return_code:
            print(f'{key}: FAILED {(info.error_message or "")[:70]}')
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

        deviation = (np.abs(outlet - reference).max()
                     if reference is not None else np.nan)

        records.append({
            'key': key,
            'group': group,
            'label': label,
            'film_diffusion': settings.get('film_diffusion', np.nan),
            'par_radius': settings.get('par_radius', case.PARTICLE_RADIUS),
            'film_rate': case.film_rate(
                settings.get('film_diffusion', case.FILM_DIFFUSION),
                settings.get('par_radius', case.PARTICLE_RADIUS)),
            'steps': measured.n_steps,
            'wall_time': wall,
            'ms_per_step': 1000.0 * wall / measured.n_steps,
            'mean_order': float(np.average(
                measured.bdf_order, weights=np.maximum(measured.steps, 0))),
            'error_test_failures': measured.n_error_test_failures,
            'max_deviation': deviation,
        })
        print(f'{key:>14} {wall:>9.2f} s  {measured.n_steps:>6} steps',
              flush=True)

    pd.DataFrame(records).to_csv(RUNS_CSV, index=False)
    pd.DataFrame(curves).to_csv(CURVES_CSV, index=False)

    return load()


def load():
    """The measured runs and the two chromatograms."""
    return pd.read_csv(RUNS_CSV), pd.read_csv(CURVES_CSV)


def _baseline(runs):
    lrm = runs[runs.key == 'lrm'].iloc[0]
    lrmp = runs[runs.key == 'lrmp'].iloc[0]

    return lrm, lrmp


def plot_same_problem(figsize=(11.0, 3.8)):
    """The two models return the same chromatogram. That is the premise."""
    runs, curves = load()
    lrm, lrmp = _baseline(runs)

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

    difference = np.abs(
        curves['LRMP component 0'] - curves['LRM component 0'])
    axes[1].plot(curves['time'] / 60.0, difference, color=COLORS[1],
                 linewidth=1.6)
    axes[1].set_yscale('log')
    # Before the pulse arrives the two solutions agree to denormal numbers, so
    # an unclipped log axis would span hundreds of useless decades.
    top = float(lrmp.max_deviation) * 3.0
    axes[1].set_ylim(top * 1e-6, top)
    axes[1].set_xlabel('time / min')
    axes[1].set_ylabel('|LRMP - LRM|')
    axes[1].set_title(f'difference, at most {lrmp.max_deviation:.1e} '
                      f'of a feed of {case.FEED:g}', fontsize=11)
    style_axis(axes[1])

    figure.tight_layout()

    return figure, axes


def plot_where_the_time_goes(figsize=(10.5, 3.6)):
    """Same steps, same order, 167 times the cost per step."""
    runs, _ = load()
    lrm, lrmp = _baseline(runs)

    figure, axes = plt.subplots(1, 3, figsize=figsize)
    names = ['LRM', 'LRMP']

    panels = (
        ('time steps taken', [lrm.steps, lrmp.steps], False),
        ('mean BDF order', [lrm.mean_order, lrmp.mean_order], False),
        ('ms per step', [lrm.ms_per_step, lrmp.ms_per_step], True),
    )

    for axis, (title, values, log) in zip(axes, panels):
        bars = axis.bar(names, values,
                        color=[COLORS[0], COLORS[1]], width=0.55)
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


def plot_the_cliff(figsize=(10.5, 4.0)):
    """Cost against the film rate: flat for six decades, then a cliff."""
    runs, _ = load()
    film = runs[runs.group == 'film'].sort_values('film_rate')
    fixed = runs[runs.group == 'fixed_rate']
    lrm, _ = _baseline(runs)

    figure, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(film.film_rate, film.ms_per_step, 'o-', color=COLORS[1],
                 linewidth=1.8, markersize=6, label=r'$r_p$ = 5 µm, $k_f$ swept')
    # All four control runs sit at the same film rate by construction, so they
    # overlap into one marker. Say so, otherwise the control is invisible.
    axes[0].plot(fixed.film_rate, fixed.ms_per_step, 's', color=COLORS[2],
                 markersize=9, zorder=5,
                 label=f'control: {len(fixed)} runs, $k_f$ over six decades,\n'
                       'film rate held fixed - all here')
    axes[0].axhline(lrm.ms_per_step, color=COLORS[0], linestyle=':',
                    linewidth=1.5, label='LRM')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'film rate $3k_f/r_p \cdot (1-\varepsilon)/\varepsilon$'
                       r' / s$^{-1}$')
    axes[0].set_ylabel('ms per step')
    axes[0].set_title('the cost is a cliff, not a slope', fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    style_axis(axes[0])

    axes[1].plot(film.film_rate, film.max_deviation, 'o-', color=COLORS[1],
                 linewidth=1.8, markersize=6)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel(r'film rate / s$^{-1}$')
    axes[1].set_ylabel('max deviation from the LRM')
    axes[1].set_title('and the answer stopped moving long before',
                      fontsize=11)
    style_axis(axes[1])

    figure.tight_layout()

    return figure, axes


def plot_the_fixes(figsize=(10.0, 3.6)):
    """Wall clock for each way of solving the identical model."""
    runs, _ = load()
    solver = runs[runs.group == 'solver']
    lrm, _ = _baseline(runs)

    figure, axis = plt.subplots(figsize=figsize)

    labels = list(solver.label)[::-1]
    values = list(solver.wall_time)[::-1]
    colors = [COLORS[1] if value > 100 else COLORS[3] for value in values]

    bars = axis.barh(labels, values, color=colors, height=0.6)
    axis.bar_label(bars, labels=[f'  {value:.1f} s' for value in values],
                   fontsize=9)
    axis.axvline(lrm.wall_time, color=COLORS[0], linestyle=':', linewidth=1.5)
    axis.annotate(f'LRM, {lrm.wall_time:.1f} s',
                  xy=(lrm.wall_time, len(labels) - 0.4), xytext=(4, 0),
                  textcoords='offset points', fontsize=9, color=COLORS[0],
                  ha='left', va='top')
    axis.set_xscale('log')
    axis.set_xlabel('wall clock / s, same model, same answer')
    axis.set_xlim(1.0, 1500.0)
    axis.set_title('every bar solves the identical problem', fontsize=11)
    style_axis(axis)

    figure.tight_layout()

    return figure, axis


def summary():
    """The solver variants as a table, with the answer each one returned."""
    runs, _ = load()
    solver = runs[runs.group == 'solver'].copy()
    lrm, _ = _baseline(runs)

    return pd.DataFrame({
        'how the linear system is solved': solver.label,
        'wall clock / s': solver.wall_time.round(2),
        'vs LRM': (solver.wall_time / lrm.wall_time).round(1),
        'steps': solver.steps,
        'max deviation from LRM': [f'{value:.3e}'
                                   for value in solver.max_deviation],
    }).reset_index(drop=True)
