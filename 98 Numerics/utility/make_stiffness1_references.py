# -*- coding: utf-8 -*-
"""

Generate the converged reference solutions for the "Stiffness 1" case study.

The study measures, for each binding speed, how fine a grid is needed before
the chromatogram is within 1 % of the converged answer. That requires a
converged answer per setting, computed on a grid far finer than any candidate,
which is far too slow to do inside a notebook: at the fastest kinetics a single
3200 cell run takes minutes.

So the references are computed once, here, and committed as small result files.
They hold the outlet chromatogram as csv, about 20 kB each. Csv rather than
h5 because the repository ignores *.h5, and because a reference chromatogram is
data worth reading without a solver installed.

Run from the chapter directory:

    python utility/make_stiffness1_references.py

Delete data/stiffness1_references/ and rerun to regenerate them, e.g. after
changing the settings module or the CADET version.

"""

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadet import Cadet  # noqa: E402
import utility.cadet_runtime as cadet_runtime  # noqa: E402
import utility.setting_Col1D_langmuir_nComp as langmuir  # noqa: E402
import utility.stiffness_metrics as stiffness_metrics  # noqa: E402

# Must match the notebook.
KINETICS_FACTORS = (0.1, 1.0, 10.0, 100.0, 1000.0)
REFERENCE_GRID = 3200
ABSTOL, RELTOL = 1e-8, 1e-6

DIRECTORY = os.path.join('data', 'stiffness1_references')


def reference_name(factor, is_kinetic):
    """File name of the reference for one setting."""
    tag = f'kin_{factor:g}' if is_kinetic else 'equilibrium'

    return tag + f'_n{REFERENCE_GRID}.csv'


def write_reference(simulation, target):
    """Extract the outlet chromatogram of a simulation into a csv file.

    Parameters
    ----------
    simulation : string
        The result file of the reference run.
    target : string
        Where to write the csv.
    """
    import h5py

    with h5py.File(simulation, 'r') as file:
        outlet = file['output/solution/unit_001/SOLUTION_OUTLET'][()]
        times = file['output/solution/SOLUTION_TIMES'][()]

    frame = pd.DataFrame({'time': times})
    for component in range(outlet.shape[1]):
        frame[f'c{component}'] = outlet[:, component]

    frame.to_csv(target, index=False)


def settings():
    """Every setting of the study: the kinetics sweep, then rapid equilibrium."""
    for factor in KINETICS_FACTORS:
        yield factor, True

    yield 1.0, False


def main():
    executable = cadet_runtime.configure()
    print('CADET:', executable)
    os.makedirs(DIRECTORY, exist_ok=True)

    for factor, is_kinetic in settings():
        path = os.path.join(DIRECTORY, reference_name(factor, is_kinetic))
        label = f'kinetics factor {factor:g}' if is_kinetic else 'rapid equilibrium'

        if os.path.exists(path):
            print(f'  {label:24s} already there')
            continue

        model = Cadet()
        model.root = langmuir.get_model(
            spatial_method_bulk=0, n_comp=2, transport_model='LRM',
            n_elements=REFERENCE_GRID, kinetics_factor=factor,
            is_kinetic=is_kinetic, idas_abstol=ABSTOL, idas_reltol=RELTOL)
        model.filename = path[:-4] + '.h5'

        start = time.time()
        measured = stiffness_metrics.run_and_measure(model, label=label,
                                                     every_step=False)
        write_reference(model.filename, path)
        os.remove(model.filename)

        print(f'  {label:24s} {measured.n_steps:7d} steps, '
              f'{time.time() - start:7.1f} s wall, '
              f'{os.path.getsize(path) / 1024:.0f} KB', flush=True)

    print('\nreferences in', DIRECTORY)


if __name__ == '__main__':
    main()
