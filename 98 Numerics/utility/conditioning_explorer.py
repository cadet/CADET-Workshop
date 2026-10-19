# -*- coding: utf-8 -*-
"""

Study: conditioning, which is the one failure mode in this chapter that is not
about speed at all.

Steric mass action writes its binding flux as a difference of two terms,

    k_a c_p (Lambda - (nu + sigma) q)^nu   -   k_d q c_salt^nu,

and both carry a quantity of order hundreds raised to the power nu. The two
have to cancel to whatever precision the solution needs, out of numbers that
can be enormous.

Two knobs, neither of which changes the chemistry:

    nu               the characteristic charge, which sets how far the
                     magnitudes are blown up. The equilibrium constant is
                     retuned alongside it so the protein keeps eluting at the
                     same salt.
    kinetics_factor  a common factor on k_a and k_d. The equilibrium constant
                     is untouched, so the chromatogram is unchanged, and both
                     terms grow together.

The first study of this chapter established that scaling binding kinetics is
free. Anything that happens here is therefore not stiffness: it is arithmetic.

"""

import utility.setting_Col1D_sma_realistic as case
import utility.stiffness_study as study_tools
from utility.stiffness_study import Study

CHARGES = (4.0, 6.0, 8.0, 10.0, 12.0)
KINETICS_FACTORS = tuple(10.0 ** exponent for exponent in range(0, 15, 2))

N_ELEMENTS = 50
COMPONENT_NAMES = ('salt', 'protein')


def _build(nu, kinetics_factor):
    return case.get_model(nu=nu, kinetics_factor=kinetics_factor,
                          n_elements=N_ELEMENTS)


def _label(nu, kinetics_factor):
    return f'nu = {nu:g}, rate constants x {kinetics_factor:.0e}'


def _describe(measured, outlet, nu, kinetics_factor):
    magnitude = case.flux_magnitude(nu, kinetics_factor)

    return {
        'k_d c_salt^nu': f'{magnitude:.2e}',
        'K_eq': f'{case.equilibrium_constant(nu):.3e}',
        'protein peak': round(float(outlet[:, 1].max()), 5),
    }


def make_study():
    """Assemble the Study for this case."""
    return Study(
        name='conditioning',
        build=_build,
        options={
            'nu': study_tools.value_options(CHARGES, '{:g}'),
            'kinetics_factor': study_tools.value_options(KINETICS_FACTORS,
                                                         '{:.0e}'),
        },
        label_of=_label,
        velocity=case.interstitial_velocity(),
        feed_concentration=case.PROTEIN_FEED,
        component_names=COMPONENT_NAMES,
        describe=_describe,
        defaults={'nu': 8.0, 'kinetics_factor': 1.0},
    )


def show_setting():
    """Plot the base case and return the rates it is made of."""
    return study_tools.show_setting(
        make_study(), nu=8.0, kinetics_factor=1.0,
        title='steric mass action: load, wash, salt gradient')


def make_explorer():
    """Build the slider callback together with the record it accumulates."""
    return study_tools.make_explorer(make_study())
