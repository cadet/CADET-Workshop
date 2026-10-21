# -*- coding: utf-8 -*-
"""

This script estimates the characteristic rates of the processes in a 1D column
model, so that "fast" and "slow" become numbers instead of adjectives.

Every process in the model relaxes at some rate. The integrator has to resolve
the fastest of them; the slow ones cost nothing, which is the point that the
usual "ratio of time scales" phrasing gets wrong. A process that is a thousand
times slower than everything else does not make a system stiff. A process that
is a thousand times faster does.

Rates are given per second, so bigger means faster. They are order of magnitude
estimates from linearized coefficients, not eigenvalues: the point is to see
which line of the table is the largest, not to get three digits.

Note that two of the entries depend on the discretization, not on the physics.
Convection and dispersion are differential operators, and what the integrator
sees is their discrete counterpart, whose rate grows as the grid is refined.

"""

import numpy as np
import pandas as pd


def _axial_node_spacing(discretization, bed_length):
    """Distance between neighbouring axial degrees of freedom."""
    if str(discretization.get('SPATIAL_METHOD', 'FV')).upper().strip("b'") == 'DG':
        n_elements = int(discretization['NELEM'])
        polynomial_degree = int(discretization['POLYDEG'])
        # Nodes within a DG element cluster towards its edges, so the smallest
        # spacing, which is what the eigenvalues see, is roughly this.
        return bed_length / (n_elements * polynomial_degree**2) \
            if polynomial_degree > 1 else bed_length / n_elements

    return bed_length / int(discretization['NCOL'])


def characteristic_rates(column, feed_concentration=None):
    """Estimate the characteristic rate of every process in a column model.

    Parameters
    ----------
    column : dict-like
        The unit operation configuration, i.e. `model.root.input.model.unit_XXX`
        of a COLUMN_MODEL_1D.
    feed_concentration : float
        Representative liquid concentration, used for the adsorption rate, which
        is concentration dependent. Defaults to 1.

    Returns
    -------
    pd.DataFrame
        One row per process, with its rate in 1/s, its time scale in s, and
        whether it is set by the physics or by the discretization.
    """
    if feed_concentration is None:
        feed_concentration = 1.0

    bed_length = float(column['bed_length'])
    total_porosity = float(column.get('total_porosity', 0.4))
    cross_section = float(column.get('cross_section_area',
                                     column.get('inlet_cross_section_area', 1.0)))

    # Interstitial velocity from the flow rate is not available here, so fall
    # back on the velocity the settings are built with.
    velocity = float(column.get('velocity', 0.01))

    node_spacing = _axial_node_spacing(column['discretization'], bed_length)

    rows = [
        ('convection, whole column', velocity / bed_length, 'physics'),
        ('convection, one grid node', velocity / node_spacing, 'discretization'),
    ]

    dispersion = float(column.get('col_dispersion', 0.0))
    if dispersion > 0.0:
        rows.append(('axial dispersion, whole column',
                     dispersion / bed_length**2, 'physics'))
        rows.append(('axial dispersion, one grid node',
                     dispersion / node_spacing**2, 'discretization'))

    particle = column.get('particle_type_000', None)
    if particle is not None:
        adsorption = particle.get('adsorption', {})

        if 'mcl_ka' in adsorption:
            ka = np.max(np.atleast_1d(adsorption['mcl_ka']))
            kd = np.max(np.atleast_1d(adsorption['mcl_kd']))
            rate = ka * feed_concentration + kd
            label = 'binding (Langmuir)'
        elif 'sma_ka' in adsorption:
            ka = np.max(np.atleast_1d(adsorption['sma_ka']))
            kd = np.max(np.atleast_1d(adsorption['sma_kd']))
            rate = ka * feed_concentration + kd
            label = 'binding (steric mass action)'
        else:
            ka = kd = rate = None
            label = None

        if rate is not None:
            if not int(adsorption.get('is_kinetic', 1)):
                rows.append((label + ', rapid equilibrium',
                             np.inf, 'physics'))
            else:
                rows.append((label, float(rate), 'physics'))

        if int(particle.get('has_film_diffusion', 0)):
            film = np.max(np.atleast_1d(particle['film_diffusion']))
            radius = float(particle['par_radius'])
            # Rate seen by the bulk liquid: specific surface times the
            # mass transfer coefficient.
            rows.append(('film mass transfer',
                         3.0 * film / radius * (1.0 - total_porosity) / total_porosity,
                         'physics'))

        if int(particle.get('has_pore_diffusion', 0)):
            pore = np.max(np.atleast_1d(particle['pore_diffusion']))
            radius = float(particle['par_radius'])
            rows.append(('pore diffusion, whole particle',
                         pore / radius**2, 'physics'))

    table = pd.DataFrame(rows, columns=['process', 'rate / s^-1', 'set by'])
    table['time scale / s'] = 1.0 / table['rate / s^-1']

    return table.sort_values('rate / s^-1', ascending=False).reset_index(drop=True)


def summarise(table, simulated_time):
    """One sentence on what the table means for the time integration.

    Parameters
    ----------
    table : pd.DataFrame
        Output of characteristic_rates().
    simulated_time : float
        Length of the simulated process, in seconds.

    Returns
    -------
    string
    """
    finite = table[np.isfinite(table['rate / s^-1'])]
    fastest = finite.iloc[0]
    slowest = finite.iloc[-1]

    return (
        f"fastest process: {fastest['process']} at {fastest['rate / s^-1']:.3g} /s "
        f"({fastest['time scale / s']:.3g} s)\n"
        f"slowest process: {slowest['process']} at {slowest['rate / s^-1']:.3g} /s "
        f"({slowest['time scale / s']:.3g} s)\n"
        f"the simulation covers {simulated_time:g} s, so the fastest process is "
        f"resolved about {simulated_time * fastest['rate / s^-1']:.3g} times over"
    )
