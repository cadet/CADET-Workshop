# -*- coding: utf-8 -*-
"""

Study: a cyclic reaction, the boundary case of this chapter.

Every other study here found fast processes free, and the eigenvalue argument
in the concept notebook says why: every dissipative term in a chromatography
model is symmetrizable, so it contributes a real negative eigenvalue, and BDF
is stable along that axis at any magnitude.

This study constructs the one thing that argument excludes. A closed cycle
with no reverse reactions,

    X_0 -> X_1 -> ... -> X_(N-1) -> X_0,

has a circulant Jacobian whose eigenvalues sit on a circle through the origin,
at an angle of 90 - 180/N degrees away from the negative real axis. Lengthening
the cycle rotates them off the axis; from six species on they leave the sector
BDF5 is stable in at every magnitude.

Two knobs, because one alone proves nothing:

    n_species  rotates the eigenvalues away from the real axis
    rate       moves them outward, which is what h * lambda has to land on

Short cycles are the control. N = 3 sits at 30 degrees and can never be
destabilized no matter how fast it runs, so if the rate knob changes the
integrator's behaviour at N = 12 but not at N = 3, the cause is the geometry
and not the speed.

This is not chromatography and is not meant to be. A stirred tank has no
spatial discretization, so the Jacobian *is* the reaction network and nothing
else can be blamed for what shows up.

"""

import numpy as np
import pandas as pd

import utility.setting_cstr_cycle as case
import utility.stiffness_demos as demos
import utility.stiffness_study as study_tools
from utility.stiffness_study import Study

BDF_MAX_ORDER = 5              # the highest order IDAS will run
SPECIES = (3, 4, 6, 12, 24)
RATES = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 20.0, 100.0)
COMPONENT_NAMES = tuple(f'X{index}' for index in range(max(SPECIES)))


def critical_eigenvalue(n_species, rate):
    """The eigenvalue furthest from the negative real axis, in 1/s.

    Not the largest one. For an even cycle length the largest eigenvalue is
    -2k, which is real and therefore harmless; the one that can cause trouble
    is the one at the widest angle.
    """
    values = case.eigenvalues(n_species, rate)
    angles = np.abs(np.angle(-values))

    return values[int(np.argmax(angles))]


def _build(n_species, rate):
    return case.get_model(n_species=n_species, rate=rate)


def _label(n_species, rate):
    return f'N = {n_species}, k = {rate:g} / s'


def _median_step(measured):
    """Step size the integrator settled on, over the steps it actually took."""
    taken = measured.steps > 0

    return float(np.median(measured.step_size[taken])) if taken.any() else np.nan


def _order_share(measured, order=BDF_MAX_ORDER):
    """Fraction of the steps that were taken at this BDF order."""
    weight = measured.steps.astype(float)
    total = weight.sum()
    if total <= 0:
        return np.nan

    return float(weight[measured.bdf_order == order].sum() / total)


def _describe(measured, outlet, n_species, rate):
    """Where this setting puts h * lambda, and what the integrator did there."""
    eigenvalue = critical_eigenvalue(n_species, rate)
    step = _median_step(measured)
    scaled = step * eigenvalue

    # The same test the conclusion figure draws, asked of this one point.
    unstable = not demos.bdf_is_stable(BDF_MAX_ORDER, scaled)

    return {
        'angle / deg': round(float(np.degrees(np.abs(np.angle(-eigenvalue)))), 1),
        'median h / s': f'{step:.3g}',
        '|h*lambda|': f'{abs(scaled):.3g}',
        'in the island': 'YES' if unstable else 'no',
        'steps at order 5': f'{_order_share(measured):.0%}',
    }


def make_study():
    """Assemble the Study for this case."""
    return Study(
        name='cycle',
        notebook='stiffness/Stiffness 11 - A cyclic reaction',
        build=_build,
        options={
            'n_species': study_tools.value_options(SPECIES, '{:d}'),
            'rate': study_tools.value_options(RATES, '{:g} / s'),
        },
        label_of=_label,
        velocity=float('nan'),      # a stirred tank has no bed to flow through
        feed_concentration=case.FEED,
        reference=None,             # cycles of different length are not comparable
        component_names=COMPONENT_NAMES,
        describe=_describe,
        defaults={'n_species': 12, 'rate': 1.0},
    )


def predictions(species=SPECIES):
    """What the circulant eigenvalues say before anything is simulated.

    Returns
    -------
    pandas.DataFrame
        Per cycle length: the angle its eigenvalues sit at, whether that is
        outside the sector BDF5 covers at every magnitude, and the step size
        at k = 1 / s that would put the worst eigenvalue inside the unstable
        island.
    """
    alpha = demos.wedge_angle(BDF_MAX_ORDER)

    rows = []
    for n_species in species:
        eigenvalue = critical_eigenvalue(n_species, 1.0)
        angle = float(np.degrees(np.abs(np.angle(-eigenvalue))))

        steps = np.logspace(-3, 3, 2000)
        unstable = ~demos.bdf_is_stable(BDF_MAX_ORDER, steps * eigenvalue)

        rows.append({
            'species in the cycle': n_species,
            'angle / deg': round(angle, 1),
            f'outside A({alpha:.0f} deg)': 'yes' if angle > alpha else 'no',
            'h / s that destabilizes, at k = 1 / s': (
                f'{steps[unstable].min():.2g} ... {steps[unstable].max():.2g}'
                if unstable.any() else 'none, at any step size'),
        })

    return pd.DataFrame(rows)


def show_setting(n_species=12, rate=1.0):
    """Plot the base case and return the eigenvalues it is built from.

    The usual rate table is not meaningful here: a stirred tank has no bed, no
    particles and no film, so the eigenvalues of the reaction network are the
    whole story and are reported directly instead.
    """
    import utility.cadet_runtime as cadet_runtime
    cadet_runtime.configure()

    study = make_study()
    _, times, outlet = study_tools.simulate(study, n_species=n_species,
                                            rate=rate)

    axis = study_tools.plot_chromatogram(
        times, outlet, component_names=study.component_names,
        title=f'a {n_species} species cycle in a stirred tank, '
              f'k = {rate:g} / s')
    axis.figure.tight_layout()

    values = case.eigenvalues(n_species, rate)

    return pd.DataFrame({
        'eigenvalue / s^-1': [f'{value:.4g}' for value in values],
        'angle from -Re axis / deg': np.degrees(
            np.abs(np.angle(-values))).round(1),
    })


def make_explorer():
    """Build the slider callback together with the record it accumulates."""
    return study_tools.make_explorer(make_study())
