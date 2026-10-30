# -*- coding: utf-8 -*-
"""

Generate the stored discretization study for "Spatial discretization".

The notebook lets the audience compare spatial methods and resolutions on the
four component SMA load-wash-elute benchmark. Simulating that live costs
seconds to minutes per setting, which is too slow to explore with a slider, so
every setting on the grid below is simulated once, here, and the outlets are
committed as a result file. The notebook then loads them instantly, and can
still rerun any single setting on demand.

The grid is also what defines the slider ranges: the notebook offers exactly
the resolutions found in the result file. Change GRID and rerun to change them.

The runs use the tight tolerances ABSTOL / RELTOL below, so that what the
audience sees is the spatial discretization error, with the time integration
error pushed well below it. That is also why the stored simulation times are
meaningful to compare: they all pay for the same temporal accuracy.

Run from the chapter directory:

    python utility/make_lwe_discretization_results.py

It skips settings that are already stored, so an interrupted run can simply be
started again. Pass --force to recompute everything, e.g. after changing the
settings module or the CADET version.

"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadet import Cadet  # noqa: E402
import utility.cadet_runtime as cadet_runtime  # noqa: E402
import utility.convergence as convergence  # noqa: E402
import utility.setting_Col1D_SMA_4comp_LWE_benchmark1 as lwe  # noqa: E402

# Spatial method (0 = FV with WENO3, otherwise the DG polynomial degree) mapped
# to the axial and particle resolutions offered for it. FV needs many cells to
# get anywhere while DG is accurate on a handful of elements, so the two differ
# by more than an order of magnitude. Each range runs a little past the point
# where refining stops paying off, so that the saturation is visible.
GRID_GRM = {
    0: ((4, 8, 16, 32, 64, 128, 256), (1, 2, 4, 8, 16)),
    1: ((4, 8, 16, 32, 64), (1, 2, 4, 8, 16)),
    2: ((4, 8, 16, 32), (1, 2, 4, 8, 16)),
    3: ((4, 8, 16, 32), (1, 2, 4, 8)),
    4: ((4, 8, 16, 32), (1, 2, 4, 8)),
    5: ((4, 8, 16, 32), (1, 2, 4, 8)),
}

ABSTOL, RELTOL = 1e-12, 1e-10
TIMEOUT = 7200.0

STUDIES = {
    'grm': {
        'particle_type': 'GENERAL_RATE_PARTICLE',
        'grid': GRID_GRM,
        'directory': os.path.join('data', '01_spatial_discretization',
                                  'lwe_discretization'),
        'reference_file': os.path.join('data',
                                       '02_time_discretization',
                                       'ref_LWE.h5'),
        'description': 'general rate model, resolved particle',
    },
}


def method_label(spatial_method):
    """Name of a spatial method: FV, or DG with its polynomial degree."""
    return 'FV' if spatial_method == 0 else f'DG{spatial_method}'


def case_key(spatial_method, n_axial, n_particle, lumped_particle=False):
    """Name under which one setting is stored in the result file."""
    if lumped_particle:
        return f'{method_label(spatial_method)}_ax{n_axial}'

    return f'{method_label(spatial_method)}_ax{n_axial}_par{n_particle}'


def spatial_dofs(spatial_method, n_axial, n_particle, lumped_particle=False):
    """Number of spatial discretization points per component."""
    nodes = 1 if spatial_method == 0 else spatial_method + 1
    if lumped_particle:
        return 2 * nodes * n_axial

    return nodes * n_axial * (1 + nodes * n_particle)


def simulate(spatial_method, n_axial, n_particle, work_directory,
             particle_type='GENERAL_RATE_PARTICLE'):
    """Run one setting and return its outlet, times and simulation time."""
    model = Cadet()
    model.root = lwe.get_model(
        spatial_method_bulk=spatial_method,
        spatial_method_particle=spatial_method,
        particle_type=particle_type,
        axNElem=n_axial,
        parZ=n_particle,
        idas_abstol=ABSTOL,
        idas_reltol=RELTOL,
    )
    model.filename = os.path.join(
        work_directory, case_key(spatial_method, n_axial, n_particle) + '.h5')
    model.save()

    run_info = model.run_simulation(timeout=TIMEOUT)
    if run_info.return_code != 0:
        raise RuntimeError(
            f'{case_key(spatial_method, n_axial, n_particle)} failed: {run_info}')

    times = np.asarray(convergence.get_solution_times(model.filename))
    outlet = np.asarray(convergence.get_outlet(model.filename, unit='000'))
    simulation_time = float(convergence.get_compute_time(model.filename))
    os.remove(model.filename)

    return times, outlet, simulation_time


def load_existing(directory):
    """The runs and outlets stored so far, as a (list of dicts, dict) pair."""
    runs_csv = os.path.join(directory, 'runs.csv')
    solutions_npz = os.path.join(directory, 'solutions.npz')
    if not (os.path.exists(runs_csv) and os.path.exists(solutions_npz)):
        return [], {}

    runs = pd.read_csv(runs_csv).to_dict('records')
    with np.load(solutions_npz) as stored:
        solutions = {name: stored[name] for name in stored.files}

    return runs, solutions


def store(directory, runs, solutions):
    """Write the runs table and the outlets, replacing what is there."""
    frame = pd.DataFrame(runs)
    if frame.empty:
        raise ValueError('Refusing to write an empty runs table.')
    frame = frame.sort_values(['spatial_method', 'n_axial', 'n_particle'])
    frame.to_csv(os.path.join(directory, 'runs.csv'), index=False)
    np.savez_compressed(os.path.join(directory, 'solutions.npz'), **solutions)


def reference_outlet(study, solutions):
    """The reference this study measures against, reusing a stored one."""
    if 'reference' in solutions:
        return np.asarray(solutions['reference'])

    return np.asarray(convergence.get_outlet(study['reference_file'], unit='000'))


def run_study(name, study, force=False):
    """Simulate every setting of one study that is not stored yet."""
    lumped = study['particle_type'] == 'HOMOGENEOUS_PARTICLE'
    directory = study['directory']
    os.makedirs(directory, exist_ok=True)

    print(f'=== {name.upper()}: {study["description"]} ===')
    runs, solutions = ([], {}) if force else load_existing(directory)
    done = {run['key'] for run in runs}

    reference = reference_outlet(study, solutions)
    solutions['reference'] = reference
    # Written out right away, so that a study whose settings are all stored
    # still picks up its reference and particle type. A study that has nothing
    # stored yet is written by the loop below instead.
    for run in runs:
        run.setdefault('particle_type', study['particle_type'])
    if runs:
        store(directory, runs, solutions)
    reference_salt = np.max(abs(reference[:, 0]))
    reference_components = np.max(abs(reference[:, 1:]))

    cases = [(method, n_axial, n_particle)
             for method, (axial, particle) in study['grid'].items()
             for n_axial in axial for n_particle in particle]
    cases.sort(key=lambda case: spatial_dofs(*case, lumped_particle=lumped))

    # Keep the store and the grid in step: a setting that has been taken off
    # the grid is dropped, so that the sliders never offer it again.
    wanted = {case_key(*case, lumped) for case in cases}
    stale = done - wanted
    if stale:
        print(f'Dropping {len(stale)} stored settings that are no longer on the'
              f' grid: {", ".join(sorted(stale))}.')
        runs = [run for run in runs if run['key'] in wanted]
        for key in stale:
            solutions.pop(key, None)
        done &= wanted
        if runs:
            store(directory, runs, solutions)

    todo = [case for case in cases if case_key(*case, lumped) not in done]

    print(f'{len(cases)} settings on the grid, {len(cases) - len(todo)} already '
          f'stored, {len(todo)} to simulate.')
    print(f'Tolerances: abstol {ABSTOL:.0e}, reltol {RELTOL:.0e}.\n')

    failed = []
    start = time.time()
    for number, (method, n_axial, n_particle) in enumerate(todo, 1):
        label = f'{method_label(method):>4} {n_axial:>4} axial'
        label += '' if lumped else f' {n_particle:>3} particle'
        try:
            times, outlet, simulation_time = simulate(
                method, n_axial, n_particle, directory, study['particle_type'])
        except Exception as error:                            # noqa: BLE001
            failed.append((method, n_axial, n_particle, str(error)[:200]))
            print(f'[{number}/{len(todo)}] {label}   FAILED: {str(error)[:120]}',
                  flush=True)
            continue

        key = case_key(method, n_axial, n_particle, lumped)
        solutions[key] = outlet
        solutions['times'] = times
        runs.append({
            'key': key,
            'method': method_label(method),
            'spatial_method': method,
            'n_axial': n_axial,
            'n_particle': n_particle,
            'dofs': spatial_dofs(method, n_axial, n_particle, lumped),
            'particle_type': study['particle_type'],
            'simulation_time': simulation_time,
            'error_salt': float(
                np.max(abs(reference[:, 0] - outlet[:, 0])) / reference_salt),
            'error_components': float(
                np.max(abs(reference[:, 1:] - outlet[:, 1:])) / reference_components),
            'abstol': ABSTOL,
            'reltol': RELTOL,
        })
        store(directory, runs, solutions)

        run = runs[-1]
        print(f'[{number}/{len(todo)}] {label}   N={run["dofs"]:<6}'
              f' {simulation_time:9.2f} s'
              f'   salt {run["error_salt"] * 100:8.4f} %'
              f'   components {run["error_components"] * 100:8.4f} %'
              f'   [{(time.time() - start) / 60:.0f} min elapsed]', flush=True)

    size = os.path.getsize(os.path.join(directory, 'solutions.npz')) / 1024 ** 2
    print(f'\n{len(runs)} settings stored in {directory} ({size:.1f} MB of'
          f' outlets) in {(time.time() - start) / 60:.0f} min.')
    for method, n_axial, n_particle, message in failed:
        print(f'  FAILED {method_label(method)} {n_axial} axial'
              f' {n_particle} particle: {message}')

    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=(*STUDIES, 'both'), default='both',
                        help='which study to generate (default: both)')
    parser.add_argument('--force', action='store_true',
                        help='recompute settings that are already stored')
    arguments = parser.parse_args()

    cadet_runtime.configure()
    chosen = STUDIES if arguments.study == 'both' else {
        arguments.study: STUDIES[arguments.study]}
    for name, study in chosen.items():
        run_study(name, study, arguments.force)
        print()


if __name__ == '__main__':
    main()
