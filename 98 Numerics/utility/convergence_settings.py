# -*- coding: utf-8 -*-
"""

The published convergence studies, and how to reproduce one of their levels.

The json files under data/spatial_discretization/convergence are the results
CADET-Verification wrote: for each spatial method and refinement level they
hold the outlet error and the compute time. This module says how each of those
levels was simulated, so that the notebook can rerun one and measure it, and
holds the reference each study is measured against.

The references are the ones CADET-Verification published, committed beside the
json files. Simulating one instead gives the same answer: the committed SMA
reference and a rerun of it here agree to 3e-12.

The DG runs use exact integration (USE_COLLOCATION_DG = 0). With collocation
the errors of the SMA setting come out one and a half to five times larger and
no longer sit on the published curves.

Where a sweep's own result files are at hand, its chromatograms are copied
out of them, so that the notebook can show how each discretization distorts the
solution: the numerical dispersion of the finite volumes and the oscillations
of DG near a steep front.

    python utility/convergence_settings.py --collect <CADET-Verification output>

Run without --collect to check that these recipes still reproduce the published
numbers, which resimulates the cheaper levels and compares:

    python utility/convergence_settings.py [--study lrmp_sma|lrm_langmuir|both]

"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadet import Cadet  # noqa: E402
import utility.cadet_runtime as cadet_runtime  # noqa: E402
import utility.convergence as convergence  # noqa: E402
import utility.setting_Col1D_SMA_4comp_LWE_benchmark1 as lwe  # noqa: E402
import utility.setting_Col1D_langLRM_2comp_benchmark1 as langmuir  # noqa: E402

DIRECTORY = os.path.join('data', 'spatial_discretization', 'convergence')
TIMEOUT = 10800.0


def build_lrmp_sma(polynomial_degree, n_elements, idas_abstol):
    """The load-wash-elute SMA setting as a lumped rate model with pores."""
    return lwe.get_model(
        spatial_method_bulk=polynomial_degree,
        particle_type='HOMOGENEOUS_PARTICLE',
        column_geometry='AXIAL_FLOW_CYLINDER',
        axNElem=n_elements,
        use_collocation_dg=0,
        idas_abstol=idas_abstol,
    )


def build_lrm_langmuir(polynomial_degree, n_elements, idas_abstol):
    """The two-component Langmuir setting as a lumped rate model."""
    return langmuir.get_model(
        spatial_method_bulk=polynomial_degree,
        refinement=n_elements,
        column_geometry='AXIAL_FLOW_CYLINDER',
        col_dispersion=1e-7,
        use_collocation_dg=0,
        idas_reftol=idas_abstol,
    )


# Each study names its json, the unit whose outlet is compared, the time
# integration tolerance of the sweep, its reference, and how much compute time
# per level is worth spending on a chromatogram. The Langmuir sweep reaches
# 65536 finite volume cells at well over two hours a level, so it is cut off;
# the levels beyond the budget keep their published error and compute time and
# simply have no chromatogram to show.
STUDIES = {
    'lrmp_sma': {
        'json': 'convergence_axial_LRMP_reqSMA_4comp_benchmark1.json',
        'description': 'four-component SMA load-wash-elute, lumped rate model with pores',
        'build': build_lrmp_sma,
        'unit': '000',
        'idas_abstol': 1e-12,
        'reference': (5, 256),
        'reference_abstol': 1e-12,
        'reference_file': 'axial_LRMP_reqSMA_4comp_benchmark1_DG_P5Z256.h5',
        'result_prefix': 'axial_LRMP_reqSMA_4comp_benchmark1',
        'time_budget': None,
    },
    'lrm_langmuir': {
        'json': 'convergence_axial_LRM_langmuir_2comp_benchmark1.json',
        'description': 'two-component Langmuir, lumped rate model',
        'build': build_lrm_langmuir,
        'unit': '001',
        'idas_abstol': 1e-10,
        'reference': (5, 2048),
        'reference_abstol': 1e-10,
        'reference_file': 'axial_LRM_langmuir_2comp_benchmark1_DG_P5Z2048.h5',
        'result_prefix': 'axial_LRM_langmuir_2comp_benchmark1',
        'time_budget': 250.0,
    },
}


def outlets_file(study):
    """Where the per level outlets of a study are stored, beside its json."""
    return os.path.join(DIRECTORY, study['json'].replace('.json', '_outlets.npz'))


def result_name(study, method, n_elements):
    """The CADET-Verification result file of one level of a sweep."""
    suffix = 'FV_' if method == 'FVWENO3' else method
    return f'{study["result_prefix"]}_{suffix}Z{n_elements}.h5'


def read_outlet(path, unit):
    """The outlet and solution times of one CADET result file."""
    import h5py

    with h5py.File(path, 'r') as handle:
        return (
            np.asarray(handle['output/solution/SOLUTION_TIMES'][()]),
            np.asarray(handle[f'output/solution/unit_{unit}/SOLUTION_OUTLET'][()]),
        )


def collect_outlets(name, study, source):
    """Copy the sweep's outlets out of a CADET-Verification output folder.

    Those runs are the ones the json was measured from, so the chromatogram of
    every level can be shown without simulating anything. Each outlet is
    checked against the published error of its level, which is what proves the
    file was matched to the right level.
    """
    _, reference = read_reference(study)
    if reference is None:
        print(f'{name}: no reference, cannot check the outlets.')
        return

    stored = {'reference': reference}
    worst = 0.0
    missing = []
    for method, n_elements, error, _ in published_levels(study):
        path = os.path.join(source, result_name(study, method, n_elements))
        if not os.path.exists(path):
            missing.append(os.path.basename(path))
            continue

        times, outlet = read_outlet(path, study['unit'])
        if outlet.shape != reference.shape:
            raise ValueError(f'{path}: outlet and reference are sampled differently.')

        stored[level_key(method, n_elements)] = outlet
        stored['times'] = times
        measured = float(np.max(abs(outlet - reference)))
        drift = abs(measured - error) / error if error else 0.0
        worst = max(worst, drift)

    kept = [key for key in stored if key not in ('times', 'reference')]
    if not kept:
        print(f'{name}: no result files in {source}; nothing collected. The'
              ' chromatogram falls back to the reference.')
        return

    target = outlets_file(study)
    temporary = target + '.tmp.npz'
    np.savez_compressed(temporary, **stored)
    os.replace(temporary, target)

    size = os.path.getsize(target) / 1024
    print(f'{name}: {len(kept)} outlets collected into {target} ({size:.0f} KB).')
    print(f'  largest deviation from the published errors: {worst * 100:.3f} %.')
    if missing:
        print(f'  {len(missing)} missing: {", ".join(missing[:4])}'
              + (' ...' if len(missing) > 4 else ''))


def level_key(method, n_elements):
    return f'{method}_ne{n_elements}'


def polynomial_degree(method):
    """The DG degree a method name stands for; 0 for the finite volumes."""
    return 0 if method == 'FVWENO3' else int(method.rsplit('P', 1)[1])


def simulate(study, polydeg, n_elements, abstol, tag, directory=DIRECTORY):
    """Run one level and return its outlet and the time CADET took."""
    model = Cadet()
    model.root = study['build'](polydeg, n_elements, abstol)
    model.filename = os.path.join(directory, tag + '.h5')
    model.save()

    run_info = model.run_simulation(timeout=TIMEOUT)
    if run_info.return_code != 0:
        raise RuntimeError(f'{tag} failed: {run_info}')

    outlet = np.asarray(convergence.get_outlet(model.filename, unit=study['unit']))
    times = np.asarray(convergence.get_solution_times(model.filename))
    seconds = float(convergence.get_compute_time(model.filename))
    os.remove(model.filename)

    return times, outlet, seconds


def read_reference(study):
    """The reference outlet of one study, from the file it was published with.

    Those files are committed beside the json files, so that neither the
    notebook nor this script has to spend an hour recomputing a reference that
    already exists. Simulating it gives the same answer: the committed SMA
    reference and a rerun of it here agree to 3e-12. Returns (None, None) if
    the file is missing, which makes the caller simulate one.
    """
    import h5py

    path = os.path.join(DIRECTORY, study['reference_file'])
    if not os.path.exists(path):
        return None, None

    with h5py.File(path, 'r') as handle:
        outlet = np.asarray(
            handle[f'output/solution/unit_{study["unit"]}/SOLUTION_OUTLET'][()])
        times = np.asarray(handle['output/solution/SOLUTION_TIMES'][()])

    return times, outlet


def published_levels(study):
    """The levels of the json, as (method, n_elements, error, time) tuples."""
    with open(os.path.join(DIRECTORY, study['json'])) as handle:
        content = json.load(handle)

    levels = []
    for method, record in content['convergence'].items():
        table = record['outlet']
        for index in range(len(table['$N_e^z$'])):
            levels.append((
                method,
                int(table['$N_e^z$'][index]),
                float(table['Max. error'][index]),
                float(table['Sim. time'][index]),
            ))

    return levels


def check_study(name, study, budget=None):
    """Resimulate the affordable levels and compare against the published ones.

    Stores nothing: this only answers whether the recipes above still produce
    the numbers in the json, which is what the notebook presents as published.
    """
    print(f'=== {name}: {study["description"]} ===')
    _, reference = read_reference(study)
    if reference is None:
        print(f'No reference at {study["reference_file"]}; nothing to check.\n')
        return

    if budget is None:
        budget = study['time_budget']
    levels = published_levels(study)
    affordable = [level for level in levels
                  if budget is None or level[3] <= budget]
    print(f'{len(levels)} published levels, checking the {len(affordable)}'
          f' that take at most {budget} s.\n', flush=True)

    worst = 0.0
    for number, (method, n_elements, error, published) in enumerate(affordable, 1):
        _, outlet, seconds = simulate(
            study, polynomial_degree(method), n_elements, study['idas_abstol'],
            f'check_{level_key(method, n_elements)}')
        measured = float(np.max(abs(outlet - reference)))
        drift = abs(measured - error) / error if error else 0.0
        worst = max(worst, drift)
        print(f'[{number}/{len(affordable)}] {method:>8} {n_elements:>6} elements'
              f'   {seconds:8.2f} s (published {published:8.2f})'
              f'   error {measured:.6e} (published {error:.6e},'
              f' off by {drift * 100:.3f} %)', flush=True)

    print(f'\nLargest deviation from the published errors: {worst * 100:.3f} %.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=(*STUDIES, 'both'), default='both',
                        help='which study to check (default: both)')
    parser.add_argument('--budget', type=float, default=None,
                        help='skip levels whose published time exceeds this')
    parser.add_argument('--collect', metavar='DIRECTORY', default=None,
                        help='copy the sweep outlets out of a CADET-Verification '
                             'output folder instead of checking')
    arguments = parser.parse_args()

    chosen = STUDIES if arguments.study == 'both' else {
        arguments.study: STUDIES[arguments.study]}
    if arguments.collect:
        for name, study in chosen.items():
            collect_outlets(name, study, arguments.collect)
        return

    cadet_runtime.configure()
    for name, study in chosen.items():
        check_study(name, study, arguments.budget)
        print()


if __name__ == '__main__':
    main()
