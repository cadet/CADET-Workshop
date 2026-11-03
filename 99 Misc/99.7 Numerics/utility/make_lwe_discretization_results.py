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
import json
import math
import io
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

# The bulk and the particle are separate spatial domains and CADET
# discretizes them separately, so the study offers a method for each. 0 is
# finite volumes (WENO3 in the bulk), anything else is the DG polynomial
# degree.
#
# The two grids below are crossed: every bulk method and resolution is
# combined with every particle method and resolution.
#
# Each ladder starts at 1 and ends where that method first reaches a relative
# error of 1e-3 in its own domain. Those endpoints are measured, not guessed:
# each domain was refined on its own against its own converged run, with the
# other domain held fixed and cheap, so that the fixed partner's error
# cancels in the difference. The measured crossings on the components:
#
#          axial    particle              axial    particle
#   FV        64          32       DG3         3           2
#   DG1        -          16       DG4         3           2
#   DG2        6           3       DG5         3           1
#
# Every method reaches 1e-3 at a cost worth paying, so unlike a tighter
# target this grid needs no exceptions. The gap between the columns is the
# result worth reading: DG5 needs three axial elements where FV needs 64.
#
# DG1 is offered in the particle but not in the bulk. Degree one is the one
# DG that is not worth its cost there: it needs as many axial elements as FV
# needs cells, 64, while carrying two nodes in each of them, so it is beaten
# by both its neighbours on the slider and it dominates the cost of building
# this grid. Inside the particle the ladders are short enough that it still
# earns its place as the low end of the DG family.
#
# The DG ladders run a step or two past their crossing, so that the
# saturation stays visible; the FV ladder is long enough already.
AXIAL_GRID = {
    0: (1, 2, 4, 8, 16, 32, 64),
    2: (1, 2, 3, 4, 6, 8),
    3: (1, 2, 3, 4, 6),
    4: (1, 2, 3, 4, 6),
    5: (1, 2, 3, 4, 6),
}

PARTICLE_GRID = {
    0: (1, 2, 4, 8, 16, 32),
    1: (1, 2, 4, 8, 16),
    2: (1, 2, 3, 4),
    3: (1, 2, 3, 4),
    4: (1, 2, 3, 4),
    5: (1, 2, 3, 4),
}

# A safety net rather than a budget: at this target the whole grid is about
# 945 settings and the most expensive of them takes a few minutes, so nothing
# is normally skipped. It only stops one pathological setting from holding up
# the batch, and the explorer offers to rerun anything missing on demand.
#
# Tightening the target is what makes this bite. Ending the ladders at 1e-5
# instead costs about 61 hours, almost all of it in the corners where BOTH
# domains are refined at once -- a single FV/FV setting accurate to 1e-5 in
# both domains takes around four hours by itself.
#
# The model is fitted to 91 calibration runs and is accurate to a median of
# 1.03x, with no run off by more than 2.2x.
COST_CEILING = 900.0
COST_SCALE, COST_EXPONENT = 0.0136, 1.227


def predicted_cost(bulk_method, particle_method, n_axial, n_particle,
                   lumped_particle=False):
    """Roughly how long one setting takes, in seconds."""
    points = spatial_dofs(bulk_method, particle_method, n_axial, n_particle,
                          lumped_particle)

    return COST_SCALE * points ** COST_EXPONENT

# Each level is simulated here and deleted again; only the summary
# lands in the published directory under data/.
WORK_DIRECTORY = os.path.join('generated', '01_spatial_discretization',
                              'lwe_discretization')

ABSTOL, RELTOL = 1e-12, 1e-10
TIMEOUT = 7200.0
STORE_EVERY = 20

STUDIES = {
    'grm': {
        'particle_type': 'GENERAL_RATE_PARTICLE',
        'axial_grid': AXIAL_GRID,
        'particle_grid': PARTICLE_GRID,
        'directory': os.path.join('data', '01_spatial_discretization',
                                  'lwe_discretization'),
        'reference_file': os.path.join('data',
                                       '02_time_discretization',
                                       'reference', 'ref_LWE.h5'),
        'description': 'general rate model, resolved particle',
    },
}


def method_label(spatial_method):
    """Name of a spatial method: FV, or DG with its polynomial degree."""
    return 'FV' if spatial_method == 0 else f'DG{spatial_method}'


def case_key(bulk_method, particle_method, n_axial, n_particle,
             lumped_particle=False):
    """Name under which one setting is stored in the result file."""
    if lumped_particle:
        return f'{method_label(bulk_method)}_ax{n_axial}'

    return (f'{method_label(bulk_method)}-{method_label(particle_method)}'
            f'_ax{n_axial}_par{n_particle}')


def spatial_dofs(bulk_method, particle_method, n_axial, n_particle,
                 lumped_particle=False):
    """Number of spatial discretization points per component.

    The bulk contributes its own nodes per axial element, and each of those
    carries a particle discretized with the particle method's nodes.
    """
    bulk_nodes = 1 if bulk_method == 0 else bulk_method + 1
    particle_nodes = 1 if particle_method == 0 else particle_method + 1
    if lumped_particle:
        return 2 * bulk_nodes * n_axial

    return bulk_nodes * n_axial * (1 + particle_nodes * n_particle)


def simulate(bulk_method, particle_method, n_axial, n_particle,
             work_directory, particle_type='GENERAL_RATE_PARTICLE'):
    """Run one setting and return its outlet, times and simulation time."""
    model = Cadet()
    model.root = lwe.get_model(
        spatial_method_bulk=bulk_method,
        spatial_method_particle=particle_method,
        particle_type=particle_type,
        axNElem=n_axial,
        parZ=n_particle,
        idas_abstol=ABSTOL,
        idas_reltol=RELTOL,
    )
    model.filename = os.path.join(
        work_directory,
        case_key(bulk_method, particle_method, n_axial, n_particle) + '.h5')
    model.save()

    run_info = model.run_simulation(timeout=TIMEOUT)
    if run_info.return_code != 0:
        raise RuntimeError(
            f'{case_key(bulk_method, particle_method, n_axial, n_particle)}'
            f' failed: {run_info}')

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


def _replace(write, path, temporary=None, attempts=5):
    """Write beside the target and move it into place when it is complete.

    Two reasons, both learned the hard way. A store this size takes a moment
    to write, and a run interrupted midway through would otherwise leave a
    truncated file where the results used to be. And on Windows the target is
    intermittently locked by whatever else has looked at it, which surfaces as
    an OSError on open; writing to a fresh name and retrying the move gets
    past that instead of losing the batch.
    """
    temporary = temporary or path + '.part'
    write(temporary)
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:                              # noqa: PERF203
            if attempt == attempts - 1:
                raise
            print(f'  {os.path.basename(path)} is locked ({error.strerror});'
                  f' retrying in {2 ** attempt} s', flush=True)
            time.sleep(2 ** attempt)


def store(directory, runs, solutions, study=None):
    """Write the runs table and the outlets, replacing what is there."""
    frame = pd.DataFrame(runs)
    if frame.empty:
        raise ValueError('Refusing to write an empty runs table.')
    frame = frame.sort_values(['spatial_method_bulk',
                               'spatial_method_particle',
                               'n_axial', 'n_particle'])
    _replace(lambda path: frame.to_csv(path, index=False),
             os.path.join(directory, 'runs.csv'))
    # savez_compressed appends .npz to any name that does not already end in
    # one, so the scratch file has to carry the suffix itself.
    _replace(lambda path: np.savez_compressed(path, **solutions),
             os.path.join(directory, 'solutions.npz'),
             temporary=os.path.join(directory, 'solutions.part.npz'))
    if study is not None:
        write_convergence_json(directory, runs, study)


def verification_name(spatial_method):
    """Method name in the spelling cadet-verification uses."""
    return 'FVWENO3' if spatial_method == 0 else f'DG_P{spatial_method}'


def write_convergence_json(directory, runs, study):
    """Write the study as a convergence table, in cadet-verification's shape.

    The explorer reads its performance panel from this file alone, so the
    outlets only have to be kept for the reference. One entry per pair of
    methods, holding every resolution of that pair.

    cadet-verification tabulates a single refinement sequence, where the
    experimental order of convergence is the ratio of successive errors. This
    study refines two domains, so the rows are grouped by particle resolution
    and the order is taken along axial refinement within each group, starting
    at 0.0 the way cadet-verification starts each table.
    """
    tables = {}
    for run in sorted(runs, key=lambda row: (row['spatial_method_bulk'],
                                             row['spatial_method_particle'],
                                             row['n_particle'],
                                             row['n_axial'])):
        key = (f'{verification_name(run["spatial_method_bulk"])}'
               f' / {verification_name(run["spatial_method_particle"])}')
        table = tables.setdefault(key, {})
        for column, value in (
                ('$N_d$', str(run['spatial_method_bulk']) or 'FV'),
                ('$N_e^z$', float(run['n_axial'])),
                ('$N_d^r$', str(run['spatial_method_particle'])),
                ('$N_e^r$', float(run['n_particle'])),
                ('Max. error', float(run['error_components'])),
                ('Max. error salt', float(run['error_salt'])),
                ('Sim. time', float(run['simulation_time'])),
                ('DoF', float(run['dofs']))):
            table.setdefault(column, []).append(value)
        table.setdefault('$N_d$', [])[-1] = (
            'FV' if run['spatial_method_bulk'] == 0
            else str(run['spatial_method_bulk']))
        table['$N_d^r$'][-1] = ('FV' if run['spatial_method_particle'] == 0
                                else str(run['spatial_method_particle']))

    for table in tables.values():
        orders = []
        for index, particle in enumerate(table['$N_e^r$']):
            if index == 0 or table['$N_e^r$'][index - 1] != particle:
                orders.append(0.0)
                continue
            before, after = table['Max. error'][index - 1], table['Max. error'][index]
            ratio = table['$N_e^z$'][index] / table['$N_e^z$'][index - 1]
            orders.append(0.0 if not (before > 0 and after > 0 and ratio > 1)
                          else math.log(before / after) / math.log(ratio))
        table['Max. EOC'] = orders

    document = {
        'meta': {
            'name': 'convergence_lwe_discretization',
            'description': ('Spatial discretization of the four component SMA'
                            ' load-wash-elute benchmark, refining the bulk and'
                            ' the particle independently.'),
            'particle_type': study['particle_type'],
            'reference': study['reference_file'].replace(os.sep, '/'),
            'abstol': ABSTOL,
            'reltol': RELTOL,
            'error_norm': ('relative maximum norm against the reference'
                           ' outlet; "Max. error" is over the three proteins'
                           ' and "Max. error salt" over the salt'),
            'eoc': ('along axial refinement at fixed particle resolution,'
                    ' 0.0 at the first entry of each group'),
        },
        'convergence': {key: {'outlet': table}
                        for key, table in sorted(tables.items())},
    }
    _replace(lambda path: io.open(path, 'w', encoding='utf-8').write(
        json.dumps(document, indent=1)),
        os.path.join(directory, 'convergence.json'))


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
        store(directory, runs, solutions, study)
    reference_salt = np.max(abs(reference[:, 0]))
    reference_components = np.max(abs(reference[:, 1:]))

    cases = [(bulk_method, particle_method, n_axial, n_particle)
             for bulk_method, axial in study['axial_grid'].items()
             for particle_method, particle in study['particle_grid'].items()
             for n_axial in axial for n_particle in particle]
    # Cheapest first, so that an interrupted run leaves the useful coarse end
    # of every combination in place.
    cases.sort(key=lambda case: spatial_dofs(*case, lumped_particle=lumped))

    # Keep the store and the grid in step: a setting that has been taken off
    # the grid is dropped, so that the sliders never offer it again. The cost
    # ceiling is deliberately not part of this: it decides what is worth
    # computing now, so a setting that is already stored is kept whatever it
    # would cost to recompute.
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
            store(directory, runs, solutions, study)

    todo = [case for case in cases if case_key(*case, lumped) not in done]
    affordable = [case for case in todo
                  if predicted_cost(*case, lumped_particle=lumped)
                  <= COST_CEILING]
    if len(affordable) < len(todo):
        print(f'{len(todo) - len(affordable)} of the {len(todo)} missing'
              f' settings are predicted to cost more than'
              f' {COST_CEILING:.0f} s each and are skipped; the notebook can'
              f' still rerun any of them on demand.')
        todo = affordable
    expected = sum(predicted_cost(*case, lumped_particle=lumped)
                   for case in todo)

    print(f'{len(cases)} settings on the grid, '
          f'{len(cases) - len(todo)} already stored or skipped, '
          f'{len(todo)} to simulate, predicted {expected / 3600:.1f} h.')
    print(f'Tolerances: abstol {ABSTOL:.0e}, reltol {RELTOL:.0e}.\n')

    failed = []
    start = time.time()
    for number, (bulk_method, particle_method,
                 n_axial, n_particle) in enumerate(todo, 1):
        label = (f'{method_label(bulk_method):>4} bulk'
                 f' / {method_label(particle_method):>4} particle'
                 f' {n_axial:>4} axial')
        label += '' if lumped else f' {n_particle:>3} particle'
        try:
            os.makedirs(WORK_DIRECTORY, exist_ok=True)
            times, outlet, simulation_time = simulate(
                bulk_method, particle_method, n_axial, n_particle,
                WORK_DIRECTORY, study['particle_type'])
        except Exception as error:                            # noqa: BLE001
            failed.append((bulk_method, particle_method, n_axial, n_particle,
                           str(error)[:200]))
            print(f'[{number}/{len(todo)}] {label}   FAILED: {str(error)[:120]}',
                  flush=True)
            continue

        key = case_key(bulk_method, particle_method, n_axial, n_particle,
                       lumped)
        solutions[key] = outlet
        solutions['times'] = times
        runs.append({
            'key': key,
            'method_bulk': method_label(bulk_method),
            'method_particle': method_label(particle_method),
            'spatial_method_bulk': bulk_method,
            'spatial_method_particle': particle_method,
            'n_axial': n_axial,
            'n_particle': n_particle,
            'dofs': spatial_dofs(bulk_method, particle_method, n_axial,
                                 n_particle, lumped),
            'particle_type': study['particle_type'],
            'simulation_time': simulation_time,
            'error_salt': float(
                np.max(abs(reference[:, 0] - outlet[:, 0])) / reference_salt),
            'error_components': float(
                np.max(abs(reference[:, 1:] - outlet[:, 1:])) / reference_components),
            'abstol': ABSTOL,
            'reltol': RELTOL,
        })
        # The whole store is rewritten each time it is checkpointed, so doing
        # it after every run would move tens of gigabytes over a full grid.
        # Every STORE_EVERY runs costs at most that many runs on an
        # interruption, which is minutes, not hours.
        if number % STORE_EVERY == 0 or number == len(todo):
            store(directory, runs, solutions, study)

        run = runs[-1]
        print(f'[{number}/{len(todo)}] {label}   N={run["dofs"]:<6}'
              f' {simulation_time:9.2f} s'
              f'   salt {run["error_salt"] * 100:8.4f} %'
              f'   components {run["error_components"] * 100:8.4f} %'
              f'   [{(time.time() - start) / 60:.0f} min elapsed]', flush=True)

    if todo:
        store(directory, runs, solutions, study)

    size = os.path.getsize(os.path.join(directory, 'solutions.npz')) / 1024 ** 2
    print(f'\n{len(runs)} settings stored in {directory} ({size:.1f} MB of'
          f' outlets) in {(time.time() - start) / 60:.0f} min.')
    for bulk_method, particle_method, n_axial, n_particle, message in failed:
        print(f'  FAILED {method_label(bulk_method)} bulk /'
              f' {method_label(particle_method)} particle,'
              f' {n_axial} axial {n_particle} particle: {message}')

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
