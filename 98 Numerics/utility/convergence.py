# -*- coding: utf-8 -*-
"""
Created January 2023

This script impements evaluation functionalities related to numerical 
convergence analysis for CADET simulations.

@author: Jan Michael Breuer
"""

from cadet import Cadet
import matplotlib.pyplot as plt
import numpy as np
import re

def legendre_gauss_nodes_and_weights(N, nit=100, TOL=1e-12):
    if N < 0:
        raise ValueError(
            "legendre_gauss_nodes_and_weights: N must be at least 0 !")
    if N == 0:
        x0 = 0
        w0 = 2
        return np.array([x0]), np.array([w0])
    elif N == 1:
        x0 = -np.sqrt(1/3)
        w0 = 1
        x1 = -x0
        w1 = 1
        return np.array([x0, x1]), np.array([w0, w1])
    else:
        nodes = np.zeros(N + 1)
        weights = np.zeros(N + 1)

        for j in range((N + 1) // 2):
            xj = -np.cos(np.pi * ((2*j+1) / (2*N+2)))

            for k in range(nit):
                P = legendre(N+1)(xj)
                dP = legendre(N+1).deriv()(xj)
                delta = -P / dP
                xj += delta

                if abs(delta) <= TOL * abs(xj):
                    break
            P = legendre(N+1)(xj)
            dP = legendre(N+1).deriv()(xj)
            nodes[j] = xj
            weights[j] = 2 / ((1.0 - xj**2) * (dP ** 2))

            nodes[N - j] = -xj
            weights[N - j] = weights[j]

        if N % 2 == 0:
            dP = legendre(N+1).deriv()(0.0)
            nodes[N // 2] = 0
            weights[N // 2] = 2 / dP**2

        return nodes, weights


def q_and_L(N, x):
    if N < 1:
        raise ValueError("q_and_L: Number of LGL nodes must be at least 2!")

    L = legendre(N)(x)
    Q = legendre(N+1)(x) - legendre(N-1)(x)
    dQ = legendre(N+1).deriv()(x) - legendre(N-1).deriv()(x)
    return L, Q, dQ


def legendre_gauss_lobatto_nodes_and_weights(N, nit=100, TOL=1e-12):
    if N < 1:
        raise ValueError(
            "legendre_gauss_lobatto_nodes_and_weights: Number of LGL nodes must be at least 2!")
    elif N == 1:
        x0 = -1
        w0 = 1
        x1 = 1
        w1 = 1
        return np.array([x0, x1]), np.array([w0, w1])
    else:
        nodes = np.zeros(N + 1)
        weights = np.zeros(N + 1)
        nodes[0] = -1
        weights[0] = 2.0 / (N*(N+1))
        nodes[N] = 1
        weights[N] = weights[0]

        for j in range(1, int(np.floor((N + 1) / 2))):
            xj = -np.cos(np.pi * (j+1/4) / N - 3 / (8*N*np.pi) * 1 / (j + 1/4))

            for k in range(nit):
                L, Q, dQ = q_and_L(N, xj)
                delta = -Q / dQ
                xj += delta
                if abs(delta) <= TOL * abs(xj):
                    break

            L, Q, dQ = q_and_L(N, xj)
            nodes[j] = xj
            nodes[N-j] = -xj
            weights[j] = 2 / (N*(N+1)*(L**2))

            # nodes[j] = -xj
            weights[N - j] = weights[j]

        if N % 2 == 0:
            L, Q, dQ = q_and_L(N, 0.0)
            nodes[N // 2] = 0.0
            weights[N // 2] = 2 / (N*(N+1)*(L**2))

        return nodes, weights


def test_Gauss_and_Lobatto_nodes():

    # test Legendre-Gauss
    with pytest.raises(ValueError):
        LGnodes, LGweights = legendre_gauss_nodes_and_weights(-1)

    LGnodes, LGweights = legendre_gauss_nodes_and_weights(0)
    np.testing.assert_almost_equal(LGnodes, np.array([0.0]), decimal=14)
    np.testing.assert_almost_equal(LGweights, np.array([2.0]), decimal=14)

    LGnodes, LGweights = legendre_gauss_nodes_and_weights(1)
    np.testing.assert_almost_equal(
        LGnodes, np.array([-np.sqrt(1/3), np.sqrt(1/3)]), decimal=14)
    np.testing.assert_almost_equal(LGweights, np.array([1.0, 1.0]), decimal=14)

    LGnodes, LGweights = legendre_gauss_nodes_and_weights(2)
    np.testing.assert_almost_equal(
        LGnodes, np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)]), decimal=14)
    np.testing.assert_almost_equal(
        LGweights, np.array([5/9, 8/9, 5/9]), decimal=14)

    LGnodes, LGweights = legendre_gauss_nodes_and_weights(4)
    np.testing.assert_almost_equal(
        LGnodes, np.array([-0.906179845938664, -0.538469310105683, 0.0,
                           0.538469310105683, 0.906179845938664]), decimal=14)
    np.testing.assert_almost_equal(LGweights, np.array([
        0.236926885056189, 0.478628670499366, 0.568888888888889,
        0.478628670499366, 0.236926885056189]), decimal=14)

    # test Legendre-Gauss-Lobatto
    with pytest.raises(ValueError):
        LGnodes, LGweights = legendre_gauss_lobatto_nodes_and_weights(-1)
        LGnodes, LGweights = legendre_gauss_lobatto_nodes_and_weights(0)

    LGnodes, LGweights = legendre_gauss_lobatto_nodes_and_weights(1)
    np.testing.assert_almost_equal(
        LGnodes, np.array([-1.0, 1.0]), decimal=14)
    np.testing.assert_almost_equal(LGweights, np.array([1.0, 1.0]), decimal=14)

    LGnodes, LGweights = legendre_gauss_lobatto_nodes_and_weights(2)
    np.testing.assert_almost_equal(
        LGnodes, np.array([-1.0, 0.0, 1.0]), decimal=14)
    np.testing.assert_almost_equal(
        LGweights, np.array([1/3, 4/3, 1/3]), decimal=14)

    LGnodes, LGweights = legendre_gauss_lobatto_nodes_and_weights(4)
    np.testing.assert_almost_equal(
        LGnodes, np.array([-1.0, -np.sqrt(3/7), 0.0, np.sqrt(3/7), 1.0]), decimal=14)
    np.testing.assert_almost_equal(LGweights, np.array([
        1/10, 49/90, 32/45, 49/90, 1/10]), decimal=14)


def get_simulation(simulation):
    """Get CADET object from h5 file.

    Parameters
    ----------
    simulation : Either string specifying an h5 file (with path)
    or CADET object

    Returns
    -------
    CADET object
    """
    if isinstance(simulation, str):
        sim = Cadet()
        sim.filename = simulation
        sim.load()
        return sim

    elif not isinstance(simulation, Cadet):
        raise ValueError(
            "Data provided is neither string to specify an h5 file nor CADET object, but " +
            str(type(simulation))
        )

    return simulation


def sim_go_to(dictionary, keys):
    """Move in Cadet object or h5 groups via dict.

    Parameters
    ----------
    dictionary : dict
         Specifies the structure of Cadet object or h5.
    unit : array of strings
        Specifies the path to take.

    Returns
    -------
    dict or any
        Data or dict or dict at the and of path.
    """
    keys = np.array(keys)

    for key in keys:
        if key in dictionary.keys():
            dictionary = dictionary[key]
        else:
            raise ValueError(
                "Simulation does not contain group: " + key
            )

    return dictionary

def get_cry_bins(simulation, unit='unit_001'):
    sol = np.squeeze(
        sim_go_to(get_simulation(simulation).root,
                  ['input', 'model', unit, 'reaction_bulk', 'CRY_BINS'	]
                  )
        )
    return sol

# TODO? currently only allows split_components_data = false

def get_solution(simulation, unit='unit_001', which='outlet', comp=[-1], **kwargs):
    """Get solution from simulation.

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest.
    unit : string
        specifies the unit of interest (000-999).
    comp : list or int
        list -> specifies the components of interest, [-1] defaults to all components.
        int -> specifies parameter ID for sensitivities.


    Returns
    -------
    np.array
        Solution vector.
    """
    if re.search('sens', which):
        param = 'param_{:03d}'.format(kwargs.get('sensIdx', 0))
        sol = np.squeeze(
            sim_go_to(get_simulation(simulation).root,
                      ['output',
                       'sensitivity',
                       param,
                       unit,
                       which]
                      )
        )
        return sol
    if comp[0] == -1:
        sol = np.squeeze(
            sim_go_to(get_simulation(simulation).root,
                      ['output',
                       'solution',
                       unit,
                       'solution_' + which]
                      )
        )
    else:
        sol = np.zeros(
            shape=(len(np.squeeze(
                sim_go_to(get_simulation(simulation).root,
                          ['output',
                           'solution',
                           unit,
                           'solution_' + which]
                          )
            )[:, 0]),
                len(comp)
            )
        )
        for i in range(0, len(comp)):
            sol[:, i] = np.squeeze(
                sim_go_to(get_simulation(simulation).root,
                          ['output',
                           'solution',
                           unit,
                           'solution_' + which]
                          )[:, comp[i]]
            )

    return sol


def get_outlet(simulation, unit='001'):
    """Get outlet from simulation.

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest.
    unit : string
        specifies the unit of onterest (000-999).


    Returns
    -------
    np.array
        Outlet vector.
    """
    return get_solution(simulation, 'unit_' + unit, 'outlet')


def get_bulk(simulation, unit='001'):
    """Get bulk solution from simulation.

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest.
    unit : string
        specifies the unit of onterest (000-999).


    Returns
    -------
    np.array
        Bulk solution vector.
    """
    return get_solution(simulation, 'unit_' + unit, 'bulk')


def get_solid(simulation, unit='001'):
    """Get solid (particle) solution from simulation.

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest.
    unit : string
        specifies the unit of onterest (000-999).


    Returns
    -------
    np.array
        Solution vector.
    """
    return get_solution(simulation, 'unit_' + unit, 'solid')


def get_particle(simulation, unit='001'):
    """Get liquid particle solution from simulation.

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest.
    unit : string
        specifies the unit of onterest (000-999).


    Returns
    -------
    np.array
        Solution vector.
    """
    return get_solution(simulation, 'unit_' + unit, 'particle')


def get_solution_times(simulation):
    """Get solution times from simulation.

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest.
    unit : string
        specifies the unit of onterest (000-999).


    Returns
    -------
    np.array
        Solution vector.
    """
    return np.squeeze(
        sim_go_to(
            get_simulation(simulation).root, ['output',
                                              'solution',
                                              'solution_times']
        )
    )

def get_commit_hash(simulation):
    """Get compute time from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest

    Returns
    -------
    np.array
        Solution vector.
    """
    
    simRoot = get_simulation(simulation).root
    commit_info = sim_go_to(simRoot, ['meta', 'cadet_commit']).decode('utf-8')
    
    if commit_info == 'GITDIR-NOTFOUND':
        commit_info = 'v.' + sim_go_to(simRoot, ['meta', 'cadet_version']).decode('utf-8') + ' commit'
    
    return commit_info


def get_compute_time(simulation):
    """Get compute time from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest

    Returns
    -------
    np.array
        Solution vector.
    """
    return np.squeeze(
        sim_go_to(
            get_simulation(simulation).root, ['meta',
                                              'time_sim']
        )
    )


def get_compute_times(simulations):

    computeTimes = np.zeros(len(simulations))

    for simulation in range(0, len(simulations)):

        computeTimes[simulation] = get_compute_time(simulations[simulation])

    return np.squeeze(computeTimes)


def get_axial_coordinates(simulation, unit='001'):
    """Get axial coordinates from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest
    unit : string
        specifies the unit of onterest (000-999)


    Returns
    -------
    np.array
        Solution vector.
    """
    return np.squeeze(
        sim_go_to(
            get_simulation(simulation).root, ['output',
                                              'coordinates',
                                              'unit_' + unit,
                                              'axial_coordinates']
        )
    )


def get_particle_coordinates(simulation, unit='001'):
    """Get particle coordinates from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest
    unit : string
        specifies the unit of onterest (000-999)


    Returns
    -------
    np.array
        Solution vector.
    """
    unit_type = sim_go_to(
        get_simulation(simulation).root, ['input',
                                          'model',
                                          'unit_' + unit,
                                          'unit_type']
    )
    if re.search("LUMPED_RATE_MODEL_WITHOUT_PORES", unit_type):
        raise ValueError(
            "LRM does not have particle coordinates!"
        )

    return np.squeeze(
        sim_go_to(
            get_simulation(simulation).root, ['output',
                                              'coordinates',
                                              'unit_' + unit,
                                              'particle_coordinates']
        )
    )


def get_radial_coordinates(simulation, unit='001'):
    """Get radial coordinates from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest
    unit : string
        specifies the unit of onterest (000-999)


    Returns
    -------
    np.array
        Solution vector.
    """
    unit_type = sim_go_to(
        get_simulation(simulation).root, ['input',
                                          'model',
                                          'unit_' + unit,
                                          'unit_type']
    )
    if not re.search("2D", str(unit_type)):
        raise ValueError(
            "Simulation does not have radial coordinates!"
        )

    return np.squeeze(
        sim_go_to(
            get_simulation(simulation).root, ['output',
                                              'coordinates',
                                              'unit_' + unit,
                                              'radial_coordinates']
        )
    )


def get_smoothness_indicator(simulation, unit='001'):
    """Get smoothness indicator from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest
    unit : string
        specifies the unit of onterest (000-999)


    Returns
    -------
    np.array
        Solution vector.
    """
    unit_type = sim_go_to(
        get_simulation(simulation).root, ['input',
                                          'model',
                                          'unit_' + unit,
                                          'unit_type']
    )

    return np.squeeze(
        sim_go_to(
            get_simulation(simulation).root, ['output',
                                              'solution',
                                              'unit_' + unit,
                                              'solution_smoothness_indicator']
        )
    )


def calculate_all_L1_errors(simulations, reference,
                            unit='001', which='outlet',
                            weights=None,
                            comps=-1):
    """Calculate L1 errors of multiple simulations.

    Parameters
    ----------
    simulations : np.ndarray
        Simulations either Cadet objects or specified by h5 names.
    reference : Cadet object or string
        (exact) reference solution specified by Cadet object or h5 name.
    unit : string
        Unit ID 000-999.
    which : string
        Specifies the solution array of interest, i.e. outlet, bulk, ...
    weights : np.array
        L1 error weights
    comps : list
        components that are to be considered. defaults to -1,
        i.e. all components are considered

    Returns
    -------
    np.array
        L1 errors.
    """

    abs_errors = np.array(calculate_all_abs_errors(
        simulations, reference, unit, which))

    if comps == -1:
        comps = range(0, abs_errors.shape[2])

    if len(abs_errors.shape) == 3:  # shape: nMethods, error domain, nComponents
        if weights is None:
            weights = [1.0] * abs_errors.shape[1]

        errors = np.zeros((abs_errors.shape[0]))

        for method_idx in range(0, abs_errors.shape[0]):
            for comp_idx in comps:
                errors[method_idx] += np.sum(np.multiply(
                    abs_errors[method_idx, :, comp_idx], weights))
    # @todo methods und components unterscheidung
    else:
        raise ValueError("not implemented yet for this nMethods, nComp")

    # shape: nMethods, error domain
    return errors


def calculate_all_L2_errors(simulations, reference,
                            unit='001', which='outlet',
                            weights=None,
                            comps=-1):
    """Calculate L2 errors of multiple simulations.

    Parameters
    ----------
    simulations : np.ndarray
        Simulations either Cadet objects or specified by h5 names.
    reference : Cadet object or string
        (exact) reference solution specified by Cadet object or h5 name.
    unit : string
        Unit ID 000-999.
    which : string
        Specifies the solution array of interest, i.e. outlet, bulk, ...
    weights : np.array
        L2 error weights
    comps : list
        components that are to be considered. defaults to -1,
        i.e. all components are considered

    Returns
    -------
    np.array
        L2 errors.
    """

    abs_errors = np.array(calculate_all_abs_errors(
        simulations, reference, unit, which))

    if comps == -1:
        comps = range(0, abs_errors.shape[2])

    if len(abs_errors.shape) == 3:  # shape: nMethods, error domain, nComponents
        if weights is None:
            weights = [1.0] * abs_errors.shape[1]

        errors = np.zeros((abs_errors.shape[0]))

        for method_idx in range(0, abs_errors.shape[0]):
            for comp_idx in comps:
                errors[method_idx] += np.multiply(np.sqrt(np.sum(
                    np.square(abs_errors[method_idx, :, comp_idx]))), weights)
    # @todo methods und components unterscheidung
    else:
        raise ValueError("not implemented yet for this nMethods, nComp")

    # shape: nMethods, error domain
    return errors


def calculate_all_min_vals(simulations, unit='001', which='outlet'):
    """Calculate minimal values of multiple simulations.

    Parameters
    ----------
    simulations : np.ndarray
        Simulations either Cadet objects or specified by h5 names.
    unit : string
        Unit ID 000-999
    which : string
        Specifies the solution array of interest, i.e. outlet, bulk, ...

    Returns
    -------
    np.array
        Minimal values errors.
    """
    errors = []

    for simulation in simulations:

        if which == 'radial_outlet':
            sim = []
            coords = get_radial_coordinates(simulation, unit)
            nRad = len(coords)
            
            for rad in range(nRad):
            
                sim.append(get_solution(
                    simulation, 'unit_'+unit, 'outlet_port_{:03d}'.format(rad))
                    )
            errors.append(np.min(np.array(sim)))
        else:
            errors.append(np.min(get_solution(simulation, 'unit_'+unit, which)))

    return np.array(errors)


def calculate_abs_error(solution, reference):
    """Calculate absolute error of solution.

    Parameters
    ----------
    solution : np.array
        Solution of simulation.
    reference : np.array
        (exact) reference solution.

    Returns
    -------
    np.array
        Absolute error.
    """
    return np.abs(solution - reference)


def calculate_all_abs_errors(simulations, reference, unit='001', which='outlet', comp=[-1], **kwargs):
    """Calculate absolute errors of multiple simulations.

    Parameters
    ----------
    simulations : np.ndarray
        Simulations either Cadet objects or specified by h5 names.
    reference : Cadet object or string
        (exact) reference solution specified by Cadet object or h5 name.
    unit : string
        Unit ID 000-999
    which : string
        Specifies the solution array of interest, i.e. outlet or bulk
    comp : list
        Specifies components of interest, defaults to all.
    kwargs : dict
        Contains other keyword arguments, e.g. interpolation information
        for bulk L1 error.

    Returns
    -------
    np.array
        Absolute errors.
    """
    errors = []

    if isinstance(reference, str) or isinstance(reference, Cadet):

        Idx = 0
        for simulation in simulations:

            if which == 'outlet':
                errors.append(calculate_abs_error(get_solution(simulation, 'unit_'+unit, which, comp),
                                                  get_solution(reference, 'unit_'+unit, which, comp)))
            elif which == 'bulk':
                if all(key in kwargs for key in ('orig_coords', 'domain_end', 'output_coords', 'polyDeg', 'nCells')):
                    errors.append(
                        calculate_abs_error(
                            get_solution(simulation, 'unit_'+unit, which,
                                         comp)[kwargs.get('time_point', -1), :],
                            get_interpolated_solution(
                                get_solution(
                                    reference, 'unit_'+unit, which, comp)[kwargs.get('time_point', -1), :],
                                kwargs['orig_coords'], kwargs['domain_end'], kwargs['output_coords'][Idx], kwargs['polyDeg'], kwargs['nCells'])
                        )
                    )
                else:  # it is assumed that solution is already given at the same discrete points
                    errors.append(calculate_abs_error(get_solution(simulation, 'unit_'+unit, which, comp)[-1, :],
                                                      reference))
            elif which == 'sens_outlet':
                errors.append(calculate_abs_error(get_solution(simulation, 'unit_'+unit, which, comp, **kwargs),
                                                  get_solution(reference, 'unit_'+unit, which, comp, **kwargs)))
            else:
                raise ValueError(
                    "Error for output " + which + "is not defined."
                )
            Idx += 1

    elif isinstance(reference, np.ndarray):

        simIdx = 0
        for simulation in simulations:

            if re.search('outlet', which):
                
                if which == 'radial_outlet':
                    if all(key in kwargs for key in ('polyDeg', 'nCells', 'domain_end', 'ref_coords')):
                        
                        sim = []
                        sim_interpolated = []
                        coords = get_radial_coordinates(simulation, unit)
                        nRad = len(coords)
                        
                        for rad in range(nRad):
                        
                            sim.append(get_solution(
                                simulation, 'unit_'+unit, 'outlet_port_{:03d}'.format(rad), comp)
                                )
                        sim = np.array(sim)
                        for tIdx in range(sim.shape[1]):
                        
                            sim_interpolated.append(get_interpolated_solution(
                                sim[:, tIdx],
                                coords, kwargs['domain_end'], kwargs['ref_coords'], kwargs['polyDeg'], kwargs['nCells'][simIdx]
                                ))
                            
                        sim_interpolated = np.array(sim_interpolated).T
                        errors.append(calculate_abs_error(reference, sim_interpolated))
                        
                    else:  # it is assumed that solution is already given at the same discrete points
                        errors.append(calculate_abs_error(get_solution(simulation, 'unit_'+unit, which, comp)[-1, :],
                                                          reference))
                else:
                    errors.append(calculate_abs_error(get_solution(simulation, 'unit_'+unit, which, comp), reference))
            elif which == 'bulk':
                if all(key in kwargs for key in ('orig_coords', 'domain_end', 'output_coords', 'polyDeg', 'nCells')):
                    errors.append(
                        calculate_abs_error(
                            get_solution(simulation, 'unit_'+unit, which,
                                         comp)[kwargs.get('time_point', -1), :],
                            get_interpolated_solution(
                                reference, kwargs['orig_coords'], kwargs['domain_end'], kwargs['output_coords'], kwargs['polyDeg'], kwargs['nCells'])
                        )
                    )
                else:  # it is assumed that solution is already given at the same discrete points
                    errors.append(calculate_abs_error(get_solution(simulation, 'unit_'+unit, which, comp)[-1, :],
                                                      reference))
            elif which == 'sens_outlet':
                errors.append(calculate_abs_error(get_solution(simulation, 'unit_'+unit, which, comp, **kwargs),
                                                  reference))
            else:
                raise ValueError(
                    "Error for output " + which + "is not defined."
                )
                
            simIdx += 1
    else:
        raise ValueError(
            "Reference is neither np.ndarray nor Cadet object nor string specifying h5 file name."
        )

    return np.array(errors, dtype=object)


def calculate_sse(solution, reference=-1):
    """Calculate sum squared error of solution.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (exact) reference solution.

    Returns
    -------
    np.array
        SSE.
    """
    if (reference == -1):
        np.sum((solution) ** 2, axis=0)
    else:
        return np.sum((solution - reference) ** 2, axis=0)


def calculate_max_error(solution, reference=-1):
    """Calculate max error of solution.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (exact) reference solution.

    Returns
    -------
    np.array
        Max Error.
    """
    solution = np.array(solution)
    if (reference == -1):
        return np.max(np.abs(solution))
    else:
        solution = np.array(reference)
        return np.max(calculate_abs_error(solution, reference))


def calculate_all_max_errors(simulations, reference, unit='001', which='outlet', comp=[-1]):
    """Calculate absolute errors of multiple simulations.

    Parameters
    ----------
    simulations : np.ndarray
        Simulations either Cadet objects or specified by h5 names.
    reference : Cadet object or string
        (exact) reference solution specified by Cadet object or h5 name.
    unit : string
        Unit ID 000-999.
    which : string
        Specifies the solution array of interest, i.e. outlet, bulk, ...
    comp : list
        Specifies components of interest, defaults to all.

    Returns
    -------
    np.array
        Absolute errors.
    """
    abs_errors = np.array(calculate_all_abs_errors(
        simulations, reference, unit, which, comp))
    if len(abs_errors.shape) == 3:  # shape: nMethods, error domain, nComponents
        abs_errors = np.max(abs_errors, axis=len(abs_errors.shape)-1)
    # shape: nMethods, error domain
    return np.max(abs_errors, axis=len(abs_errors.shape)-1)


def calculate_Linf_error(solution, reference=-1):
    """Calculate L_infinity (i.e. maximal) error of solution.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (exact) reference solution.

    Returns
    -------
    np.array
        Max Error.
    """
    return calculate_max_error(solution, reference)


def calculate_weighted_error(solution, reference=-1, weights=1.0):
    """Calculate weighted error of solution.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (~exact) reference solution.
    weights : np.array or scalar
        weights

    Returns
    -------
    np.array
        Max Error.
    """
    if (reference == -1):
        return np.sum(np.abs(solution) * weights)
    else:
        return np.sum(np.abs(solution - reference) * weights)


def calculate_solution_times_sizes(simulation):
    """Calculate the solution times step sizes

    Parameters
    ----------
    simulation : string or Cadet object
        simulation.

    Returns
    -------
    np.array
        Time step sizes.
    """

    solution_times = get_solution_times(simulation)

    return solution_times[1:] - solution_times[:-1]


def qAndL(polyDeg, x):

    L_2 = 1.0
    L_1 = x
    Lder_2 = 0.0
    Lder_1 = 1.0
    Lder = 0.0

    for k in range(2, polyDeg + 1):
        L = ((2 * k - 1) * x * L_1 - (k - 1) * L_2) / k
        Lder = Lder_2 + (2 * k - 1) * L_1
        L_2 = L_1
        L_1 = L
        Lder_2 = Lder_1
        Lder_1 = Lder

    q = ((2.0 * polyDeg + 1) * x * L - polyDeg * L_2) / (polyDeg + 1.0) - L_2
    qder = Lder_1 + (2.0 * polyDeg + 1) * L_1 - Lder_2

    return L, q, qder


def LGL_NodesWeights(polyDeg):
    """Calculate LGL quadraturenodes and  weights on [-1, 1].

    Parameters
    ----------
    nPoints : int
        Number of LGL points

    Returns
    -------
    np.array
        LGL nodes.
    np.array
        LGL weights.
    """
    # tolerance and max #iterations for Newton iteration
    nIterations = 10
    tolerance = 1e-15

    # Legendre polynomial and derivative
    L = 0
    q = 0
    qder = 0

    if polyDeg == 0:
        raise ValueError("Polynomial degree must be at least 1 !")
    elif polyDeg == 1:
        nodes = np.array([-1, 1])
        weights = np.array([1, 1])
    else:
        nodes = np.zeros(polyDeg + 1)
        weights = np.zeros(polyDeg + 1)

        nodes[0] = -1
        nodes[polyDeg] = 1
        weights[0] = 2.0 / (polyDeg * (polyDeg + 1.0))
        weights[polyDeg] = weights[0]

        # use symmetry, only compute half of points and weights
        for j in range(1, math.floor((polyDeg + 1) / 2)):
            # first guess for Newton iteration
            nodes[j] = -math.cos(math.pi * (j + 0.25) / polyDeg -
                                 3 / (8.0 * polyDeg * math.pi * (j + 0.25)))

            # Newton iteration to find roots of Legendre Polynomial
            for k in range(nIterations + 1):
                L, q, qder = qAndL(polyDeg, nodes[j])
                nodes[j] = nodes[j] - q / qder
                if abs(q / qder) <= tolerance * abs(nodes[j]):
                    break

            # calculate weights
            L, q, qder = qAndL(polyDeg, nodes[j])
            weights[j] = 2.0 / (polyDeg * (polyDeg + 1.0) * L**2)
            # copy to second half of points and weights
            nodes[polyDeg - j] = -nodes[j]
            weights[polyDeg - j] = weights[j]

        if polyDeg % 2 == 0:  # for even polyDeg we have an odd number of points which include 0.0
            L, q, qder = qAndL(polyDeg, 0.0)
            nodes[polyDeg // 2] = 0
            weights[polyDeg // 2] = 2.0 / (polyDeg * (polyDeg + 1.0) * L**2)

    return nodes, weights


def barycentric_weights(polyDeg):

    baryWeights = np.ones(polyDeg+1)
    _nodes, _weights = LGL_NodesWeights(polyDeg)

    for j in range(1, polyDeg+1):
        for k in range(j):
            baryWeights[k] *= (_nodes[k] - _nodes[j]) * 1.0
            baryWeights[j] *= (_nodes[j] - _nodes[k]) * 1.0

    baryWeights = np.reciprocal(baryWeights)

    return baryWeights


def map_z_to_xi(z, cellIdx, deltaZ):

    z_l = cellIdx * deltaZ
    xi = 2.0 * (z - z_l) / deltaZ - 1.0

    return xi


def map_xi_to_z(xi, cellIdx, deltaZ):

    z_l = cellIdx * deltaZ
    z = deltaZ * (xi + 1.0) / 2.0 + z_l

    return z


def test_map_z_to_xi():

    polyDeg = 4

    nNodes = polyDeg + 1

    nodes, weights = LGL_NodesWeights(polyDeg)

    x_lIdx = 0
    cellIdx = 2
    stretch = 0.1
    deltaZ = 2 * stretch
    orig_coords = (nodes + 1) * stretch + deltaZ * cellIdx

    mapped_orig_coords = np.zeros(nNodes)

    for node in range(nNodes):
        mapped_orig_coords[node] = map_z_to_xi(
            orig_coords[x_lIdx + node], cellIdx, deltaZ)

    if (abs(mapped_orig_coords - nodes) > 1e-14).any():
        raise ValueError(
            "getSolutionDG: mapped_orig_coords != nodes"
        )


# TODO slice ?
# def interpolate_DGsolution(solution_slice, output_coords, deltaZ=None, nNodes=None):

#     for outputIdx in range(len(output_coords)):

#         coord = output_coords[outputIdx]

#         # get element index and left boundary node index
#         while (cellIdx + 1) * deltaZ < coord:
#             cellIdx += 1
#             x_lIdx += nNodes

#         # Check if output coordinate is in DG coordinates
#         for nodeIdx in range(nNodes):
#             if coord - orig_coords[cellIdx * nNodes + nodeIdx] < 1e-14:
#                 if nodeIdx == 0 and cellIdx != 0:
#                     output_values[outputIdx] = 0.5 * (orig_values[cellIdx * nNodes + nodeIdx] + [cellIdx * nNodes + nodeIdx - 1])
#                 elif nodeIdx == polyDeg and cellIdx != nCells - 1:
#                     output_values[outputIdx] = 0.5 * (orig_values[cellIdx * nNodes + nodeIdx] + [cellIdx * nNodes + nodeIdx + 1])
#                 else:
#                     output_values[outputIdx] = orig_values[cellIdx * nNodes + nodeIdx]

#         # Coordinate not in DG coordinates -> interpolate!

#         # get coordinate w.r.t to reference element
#         mapped_coord = coord * map_z_to_xi(coord, cellIdx, deltaZ)

#         # calculate value at coord using using barycentric form
#         output_values[outputIdx] = (bary_weights / (mapped_coord - nodes) * orig_values).sum() / (bary_weights / (mapped_coord - nodes)).sum()

def get_interpolated_solution(simulation_name, output_coords, polyDeg, nCells, unit='unit_001'):

    orig_coords = get_axial_coordinates(simulation_name, unit=unit)
    orig_values = get_bulk(simulation_name, unit=unit)
    column_length = sim_go_to(get_simulation(simulation_name).root,
                              ['input',
                               'model',
                               unit,
                               'col_length']
                              )
    return get_interpolated_solution(orig_values, orig_coords, column_length, output_coords, polyDeg, nCells)


# all arrays need to be sorted w.r.t spatial direction
def get_interpolated_solution(orig_values, orig_coords, domain_end, output_coords, polyDeg, nCells):
    """Calculate weighted error of solution.

    Parameters
    ----------
    orig_values : np.array
        Solution to be interpolated.
    orig_coords : np.array
        Coordinate vector of solution.
    domain_end : float
        Length of spatial 1D interval
    output_coords : np.array
        Coordinates the solution is interpolated to
    polyDeg : int
        Polynomial degree of solution
    nCells : int
        Number of cells of solution

    Returns
    -------
    np.array
        Solution at output_coords.
    """
    if np.any(output_coords > domain_end) or np.any(output_coords < 0.0):
        raise ValueError(
            "get_interpolated_solution: Output coordinates not within [0, L]"
        )

    orig_values = np.array(orig_values)
    output_coords = np.sort(output_coords)

    # todo: improve runtime by using array chunks in calculation instead of the following recursive loop
    if orig_values.ndim == 3:  # shape: time x space x components
        output_values = np.zeros(
            (orig_values.shape[0], len(output_coords), orig_values.shape[2]))
        for timeIdx in range(orig_values.shape[0]):
            for compIdx in range(orig_values.shape[2]):
                output_values[timeIdx, :, compIdx] = get_interpolated_solution(
                    orig_values[timeIdx, :, compIdx], orig_coords, domain_end, output_coords, polyDeg, nCells)

        return output_values

    elif orig_values.ndim == 2:  # shape: time x space
        output_values = np.zeros((orig_values.shape[0], len(output_coords)))
        for timeIdx in range(orig_values.shape[0]):
            output_values[timeIdx, :] = get_interpolated_solution(
                orig_values[timeIdx, :], orig_coords, domain_end, output_coords, polyDeg, nCells)

        return output_values
    else:
        output_values = np.zeros(len(output_coords))

    deltaZ = domain_end / nCells
    nNodes = polyDeg + 1

    if polyDeg == 0:

        cellIdx = 0
        for outputIdx in range(len(output_coords)):
            while orig_coords[cellIdx] + 0.5 * deltaZ < output_coords[outputIdx]:
                cellIdx += 1
            output_values[outputIdx] = orig_values[cellIdx]

        return output_values

    else:

        nodes, weights = LGL_NodesWeights(polyDeg)
        bary_weights = barycentric_weights(polyDeg)

        cellIdx = 0
        x_lIdx = 0  # index of x_l in current cell w.r.t. orig_values

        for outputIdx in range(len(output_coords)):

            nodeHit = False
            coord = output_coords[outputIdx]

            # get cell index and left boundary node index
            while (cellIdx + 1) * deltaZ < coord:
                cellIdx += 1
                x_lIdx += nNodes

            # Check if output coordinate is in DG coordinates and if so, set value accordingly
            for nodeIdx in range(nNodes):
                if abs(coord - orig_coords[cellIdx * nNodes + nodeIdx]) < 1e-12:
                    nodeHit = True

                    if nodeIdx == 0 and cellIdx != 0:
                        output_values[outputIdx] = 0.5 * (
                            orig_values[cellIdx * nNodes + nodeIdx] + orig_values[cellIdx * nNodes + nodeIdx - 1])
                    elif nodeIdx == polyDeg and cellIdx != nCells - 1:
                        output_values[outputIdx] = 0.5 * (
                            orig_values[cellIdx * nNodes + nodeIdx] + orig_values[cellIdx * nNodes + nodeIdx + 1])
                    else:
                        output_values[outputIdx] = orig_values[cellIdx *
                                                               nNodes + nodeIdx]
                    break
            # Interpolate value if coordinate is not in DG coordinates
            if not nodeHit:
                # get coordinate w.r.t to reference element
                mapped_coord = map_z_to_xi(coord, cellIdx, deltaZ)
                # calculate value at coord using using barycentric form
                output_values[outputIdx] = (bary_weights / (mapped_coord - nodes) * orig_values[cellIdx * nNodes: (
                    cellIdx + 1) * nNodes]).sum() / (bary_weights / (mapped_coord - nodes)).sum()

    return output_values


def test_get_interpolated_DGsolution():

    polyDeg = 4
    nCells = 30
    column_length = 1.0

    deltaZ = column_length / nCells
    nNodes = (polyDeg + 1)
    nPoints = nNodes * nCells
    nodes, weights = LGL_NodesWeights(polyDeg)

    x_l = np.linspace(0.0, column_length, num=nCells, endpoint=False)

    # print("x_l:\n", x_l)

    orig_coords = np.zeros(nPoints)
    orig_values = np.zeros(nPoints)

    for cellIdx in range(nCells):
        orig_coords[cellIdx * nNodes: (cellIdx + 1) *
                    nNodes] = map_xi_to_z(nodes, cellIdx, deltaZ)

    # print("orig_coords:\n", orig_coords)

    # test linear concentration function
    orig_values = orig_coords
    output_coords = np.linspace(
        0.0, column_length, num=2*nPoints, endpoint=True)

    output_values = get_interpolated_solution(
        orig_values, orig_coords, column_length, output_coords, polyDeg, nCells)

    np.testing.assert_almost_equal(output_values, output_coords, decimal=14)

    # test quadratic concentration function
    out_polyDeg = 3
    out_nodes, out_weights = LGL_NodesWeights(polyDeg)
    for cellIdx in range(nCells*2):
        output_coords[cellIdx * nNodes: (cellIdx + 1) *
                      nNodes] = map_xi_to_z(out_nodes, cellIdx, deltaZ/2)
    orig_values = np.square(orig_coords)
    # output_coords = np.linspace(0.0, column_length, num=int(nPoints/2), endpoint=True)
    output_values = get_interpolated_solution(
        orig_values, orig_coords, column_length, output_coords, polyDeg, nCells)

    np.testing.assert_almost_equal(
        output_values, np.square(output_coords), decimal=14)

    # test sinus concentration function (limited accuracy)
    orig_values = np.sin(orig_coords)
    output_values = get_interpolated_solution(
        orig_values, orig_coords, column_length, output_coords, polyDeg, nCells)

    np.testing.assert_almost_equal(
        output_values, np.sin(output_coords), decimal=6)

    # print("output_coords:\n", output_coords)
    # print("output_values:\n", output_values)

    # test two dimensional array
    output_coords = np.linspace(0.0, column_length, num=nPoints, endpoint=True)
    nTime = 10
    orig_values = np.zeros((nTime, nPoints))
    orig_values[0, :] = np.square(orig_coords)
    orig_values[5, :] = np.power(orig_coords, polyDeg)
    compare = np.zeros((nTime, nPoints))
    compare[0, :] = np.square(output_coords)
    compare[5, :] = np.power(output_coords, polyDeg)

    output_values = get_interpolated_solution(
        orig_values, orig_coords, column_length, output_coords, polyDeg, nCells)

    np.testing.assert_almost_equal(output_values, compare, decimal=14)

    # test three dimensional array
    nTime = 12
    nComp = 3
    orig_values = np.zeros((nTime, nPoints, nComp))
    orig_values[0, :, 0] = np.square(orig_coords)
    orig_values[3, :, 0] = np.power(orig_coords, polyDeg)
    orig_values[7, :, 2] = np.power(orig_coords, polyDeg - 1)
    compare = np.zeros((nTime, nPoints, nComp))
    compare[0, :, 0] = np.square(output_coords)
    compare[3, :, 0] = np.power(output_coords, polyDeg)
    compare[7, :, 2] = np.power(output_coords, polyDeg - 1)

    output_values = get_interpolated_solution(
        orig_values, orig_coords, column_length, output_coords, polyDeg, nCells)

    np.testing.assert_almost_equal(output_values, compare, decimal=14)

    # # test runtime for three dimensional array
    # nTime = 1000  # for 10000 already couple seconds -> enhance get_interpolated_solution?
    # nComp = 4
    # orig_values = np.zeros((nTime, nPoints, nComp))
    # orig_values[476, :, 0] = np.square(orig_coords)
    # orig_values[100, :, 1] = np.power(orig_coords, polyDeg)
    # orig_values[20, :, 3] = np.power(orig_coords, polyDeg - 1)
    # compare = np.zeros((nTime, nPoints, nComp))
    # compare[476, :, 0] = np.square(output_coords)
    # compare[100, :, 1] = np.power(output_coords, polyDeg)
    # compare[20, :, 3] = np.power(output_coords, polyDeg - 1)

    # output_values = get_interpolated_solution(
    #     orig_values, orig_coords, column_length, output_coords, polyDeg, nCells)

    # np.testing.assert_almost_equal(output_values, compare, decimal=14)


def get_unique_DGcoordinates(coords):

    return np.unique(coords)


def get_unique_DGsolution(polyDeg, solution):

    nNodes = polyDeg + 1
    solution = np.array(solution)

    if solution.ndim == 1:  # space slice
        return get_unique_DGsolution_slice(polyDeg, solution)

    elif solution.ndim == 2:  # time x space
        nCells = int(solution.shape[1] / nNodes)
        out = np.zeros((solution.shape[0], nNodes * nCells - nCells + 1))
        for timeIdx in range(solution.shape[0]):
            out[timeIdx, :] = get_unique_DGsolution_slice(
                polyDeg, solution[timeIdx, :])

    elif solution.ndim == 3:  # time x space x comp
        nCells = int(solution.shape[1] / nNodes)
        out = np.zeros(
            (solution.shape[0], nNodes * nCells - nCells + 1, solution.shape[2]))
        for compIdx in range(solution.shape[2]):
            for timeIdx in range(solution.shape[0]):
                out[timeIdx, :, compIdx] = get_unique_DGsolution_slice(
                    polyDeg, solution[timeIdx, :, compIdx])

    return out


def get_unique_DGsolution_slice(polyDeg, solution):

    nNodes = polyDeg + 1

    nPoints = len(solution)
    nCells = int(nPoints / nNodes)
    out = []
    out.append(solution[0])  # Boundary interface
    nodeIdx = 1

    for cellIdx in range(nCells):
        for i in range(1, polyDeg):  # inner polyDeg - 1 nodes
            out.append(solution[nodeIdx])
            nodeIdx += 1
        if cellIdx < nCells - 1:  # inner interfaces
            out.append(0.5 * (solution[nodeIdx] + solution[nodeIdx+1]))
            nodeIdx += 2
        else:
            out.append(solution[nodeIdx])  # Boundary interface

    return out


def test_get_unique_DGsolution():

    nCells = 3
    polyDeg = 3
    nComp = 2
    timePoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    nTime = len(timePoints)

    nNodes = polyDeg + 1
    nPoints = nCells*nNodes
    solution = np.zeros((nTime, nCells*nNodes, nComp))
    compare = np.zeros((nTime, nCells*nNodes-nCells+1, nComp))

    # fill timepoints
    for nodeIdx in range(nPoints):
        for compIdx in range(nComp):
            solution[:, nodeIdx, compIdx] = timePoints
            if nodeIdx < nCells*nNodes-nCells+1:
                compare[:, nodeIdx, compIdx] = timePoints

    solution[0, :, 0] = np.ones(nPoints)
    compare[0, :, 0] = np.ones(nCells*nNodes-nCells+1)
    solution[0, :, 1] = [1, 2, 3, 4,
                         5, 6, 7, 8,
                         9, 10, 11, 12]
    compare[0, :, 1] = [1, 2, 3, 4.5, 6, 7, 8.5, 10, 11, 12]
    solution[6, :, 1] = [1, 2, 3, 4,
                         5, 6, 7, 8,
                         9, 10, 11, 12]
    compare[6, :, 1] = [1, 2, 3, 4.5, 6, 7, 8.5, 10, 11, 12]

    # Test single space slice
    np.testing.assert_array_equal(get_unique_DGsolution(
        polyDeg, np.ones(nPoints)), np.ones(nPoints-nCells+1))
    np.testing.assert_array_equal(get_unique_DGsolution(
        polyDeg, solution[0, :, 1]), compare[0, :, 1])

    # Test full time x space x comp array
    test = get_unique_DGsolution(polyDeg, solution)
    np.testing.assert_array_equal(test, compare)

    # Test unique DG coords
    coords = [0, 0.1, 0.5, 0.9, 1.0, 1.0, 1.1,
              1.5, 1.9, 2.0, 2.0, 2.1, 2.5, 2.9, 3.0]
    compare_coords = [0, 0.1, 0.5, 0.9, 1.0,
                      1.1, 1.5, 1.9, 2.0, 2.1, 2.5, 2.9, 3.0]
    np.testing.assert_array_equal(
        get_unique_DGcoordinates(coords), compare_coords)


def calculate_bulk_L1_error(polyDeg, mapJac, solution, reference=-1, weights=-1):
    """Calculate L_1 error of a DG bulk solution time slice.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (exact) reference solution. Default -1, if abs. error is given in field 'solution'
    polyDeg : int
        DG polynomial degree
    mapJac : float or np.array(cell-wise factor)
        Mapping Jacobian for DG, cell-spacing for FV
    weights : np.array
        LGL weights, defaults to -1 to calculate weights

    Returns
    -------
    float
        L1 error.
    """

    solution = np.array(solution)
    nNodes = polyDeg + 1
    nCells = int(solution.shape[0] / nNodes)

    if solution.shape[0] % nNodes:
        raise ValueError(
            "calculate_bulk_L1_error: Length " +
            str(solution.shape[0]) +
            " of input vector not a multiple of number of nodes " + str(nNodes)
        )
    if isinstance(reference, (int, float)):
        absError = np.array(solution)
    else:
        if solution.shape[0] != reference.shape[0]:
            raise ValueError(
                "calculate_bulk_L1_error: Length of reference and approximation vector are not the same"
            )
        reference = np.array(reference)
        absError = np.abs(solution - reference)

    L1error = 0.0
    if polyDeg == 0:
        weights = 1
    elif isinstance(weights, (int, float)):
        nodes, weights = LGL_NodesWeights(polyDeg)

    if isinstance(mapJac, (int, float)):
        for cell in range(nCells):
            L1error += mapJac * np.sum(
                absError[cell * nNodes: (cell + 1) * nNodes] * weights
            )
    else:
        for cell in range(nCells):
            L1error += mapJac[cell] * np.sum(
                absError[cell * nNodes: (cell + 1) * nNodes] * weights
            )

    return L1error


def calculate_summed_bulk_L1_error(solution, reference, polyDeg, mapJac, comp=-1, nComp=1, deltaT=1.0):
    """Calculate L_1 error of DG bulk solution cumulated over time.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (exact) reference solution.
    polyDeg : int
        DG polynomial degree
    mapJac : float
        Mapping Jacobian
    comp : int
        Component index of interest, defaults to -1 to consider all components
    nComp : int
        Number of components
    polyDeg : float
        Timestep size

    Returns
    -------
    float
        L1 error.
    """

    solution = np.array(solution)
    reference = np.array(reference)

    nodes, weights = LGL_NodesWeights(polyDeg)

    L1error = 0.0

    if nComp == 1:  # shape is time, space
        for t in range(solution.shape[0]):
            L1error += calculate_bulk_L1_error(
                solution[t, :], reference[t, :], polyDeg, mapJac, weights)
    elif comp == -1:  # shape is time, space, comp, all components wanted
        for t in range(solution.shape[0]):
            L1error += calculate_bulk_L1_error(
                solution[t, :, :], reference[t, :, :], polyDeg, mapJac, weights)
    else:  # shape is time, space, comp, one component wanted
        for t in range(solution.shape[0]):
            L1error += calculate_bulk_L1_error(
                solution[t, :, comp], reference[t, :, comp], polyDeg, mapJac, weights)


def calculate_L1_error(solution, reference=-1, weights=1.0):
    """Calculate L_1 error of solution.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (exact) reference solution.

    Returns
    -------
    float
        L1 Error.
    """
    return calculate_weighted_error(solution, reference, weights)


def calculate_L2_error(solution, reference=-1, weights=1.0):
    """Calculate L_w error of solution.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (exact) reference solution.

    Returns
    -------
    float
        L2 Error.
    """
    if (reference == -1):
        return np.sqrt(np.sum(np.square(solution))) * weights
    else:
        return np.sqrt(np.sum(np.square(solution - reference))) * weights


def calculate_average_error(solution, reference):
    """Calculate average error of solution.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (exact) reference solution.

    Returns
    -------
    np.array
        Average Error.
    """
    return np.calculate_weighted_error(solution, reference, 1.0)


def generate_1D_name(prefix, axP, axCells, suffix='.h5'):
    """Generate simulation name for bulk discretized models (LRMP, LRM).

    Parameters
    ----------
    prefix : string
        Name prefix.
    axP : int
        axial polynomial degree.
    axCells : int
        axial number of cells.
    suffix : string
        Name suffix, including filetype .h5.

    Returns
    -------
    string
        File name.
    """
    if int(axCells) <= 0:
        return None

    if int(axP) == 0:
        return prefix + "_FV_Z" + str(int(axCells)) + suffix

    elif int(axP) > 0:
        return prefix + "_DG_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + suffix

    elif int(axP) < 0:
        return prefix + "_DGexInt_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + suffix

    else:
        return None


def generate_simulation_names_1D(prefix, methods, disc, suffix='.h5'):
    """Generate simulation names for bulk discretized models (LRMP, LRM).

    Parameters
    ----------
    prefix : string
        Name prefix.
    methods : np.array<int>
        axial polynomial degrees (0 for FV, -int for exInt DGSEM).
    disc : np.array<int>
        axial number of cells.
    suffix : string
        Name suffix, including filetype .h5.

    Returns
    -------
    string
        File names.
    """
    methods = np.array(methods)
    disc = np.array(disc)
    _simulation_names = []
    nMethods = len(methods)
    # Check input parameters and infer data
    if nMethods > 1 and nMethods != disc.shape[0]:
        disc = disc.transpose()
        if nMethods != disc.shape[0]:
            raise ValueError(
                "Method and discretization must have feasible dimensionalities, look up description."
            )

    if nMethods > 1:
        nDisc = disc.shape[1]
    else:
        nDisc = disc.size
        _disc = disc

    # Main loop, create names
    for m in range(0, nMethods):
        if nMethods > 1:
            _disc = disc[m]

        for d in range(0, nDisc):

            name = generate_1D_name(prefix, methods[m], _disc[d], suffix)
            if name is not None:
                _simulation_names.append(name)

    return _simulation_names


def test_generate_simulation_names_1D():

    prefix = "LinearLRM1Comp"
    methods = [1]
    disc = [1, 2, 4]
    names = np.array(generate_simulation_names_1D(prefix, methods, disc))
    expected_names = np.array([prefix+"_DG_P1Z1",
                              prefix+"_DG_P1Z2",
                              prefix+"_DG_P1Z4"])
    np.testing.assert_array_equal(names, expected_names)

    methods = [2, -3]
    disc = [[2, 4, 8], [1, 2, 4]]
    names = np.array(generate_simulation_names_1D(prefix, methods, disc))
    expected_names = np.array([prefix+"_DG_P2Z2",
                              prefix+"_DG_P2Z4",
                              prefix+"_DG_P2Z8",
                              prefix+"_DGexInt_P3Z1",
                              prefix+"_DGexInt_P3Z2",
                              prefix+"_DGexInt_P3Z4"
                               ])
    np.testing.assert_array_equal(names, expected_names)


def generate_2D_name(prefix, axP, axCells, radP, radCells, suffix='.h5'):
    """Generate simulation name for bulk discretized models (2DLRMP, 2DLRM).

    Parameters
    ----------
    prefix : string
        Name prefix.
    axP : int
        axial polynomial degree.
    axCells : int
        axial number of cells.
    radP : int
        radial polynomial degree.
    radCells : int
        radial number of cells.
    suffix : string
        Name suffix, including filetype .h5.

    Returns
    -------
    string
        File name.
    """
    if int(axCells) <= 0 or int(radCells) <= 0:
        return None

    if int(axP) == 0 and int(radP) == 0:
        return prefix + "_FV_axZ" + str(int(axCells)) + "radZ" + str(int(radCells)) + suffix

    elif int(axP) > 0 and int(radP) > 0:
        return prefix + "_DG_axP" + str(int(abs(axP))) + "Z" + str(int(axCells)) + "_radP" + str(int(abs(radP))) + "Z" + str(int(radCells)) + suffix

    else:
        return None


def generate_simulation_names_2D(prefix, ax_methods, ax_disc, rad_methods, rad_disc, suffix='.h5'):
    """Generate simulation names for 2Dbulk discretized models (2DLRMP, 2DLRM).

    Parameters
    ----------
    prefix : string
        Name prefix.
    ax_methods : np.array<int>
        axial polynomial degrees
    ax_disc : np.array<int>
        axial number of cells.
    rad_methods : np.array<int>
        radial polynomial degrees
    rad_disc : np.array<int>
        radial number of cells.
    suffix : string
        Name suffix, including filetype .h5.

    Returns
    -------
    string
        File names.
    """
    ax_methods = np.array(ax_methods)
    ax_disc = np.array(ax_disc)
    _simulation_names = []
    nMethods = len(ax_methods)
    # Check input parameters and infer data
    if nMethods > 1 and nMethods != ax_disc.shape[0]:
        ax_disc = ax_disc.transpose()
        if nMethods != ax_disc.shape[0]:
            raise ValueError(
                "Method and discretization must have feasible dimensionalities, look up description."
            )
    if nMethods > 1 and nMethods != rad_disc.shape[0]:
        rad_disc = rad_disc.transpose()
        if nMethods != rad_disc.shape[0]:
            raise ValueError(
                "Method and discretization must have feasible dimensionalities, look up description."
            )

    if nMethods > 1:
        nDisc = ax_disc.shape[1]
    else:
        nDisc = ax_disc.size
        _ax_disc = ax_disc
        _rad_disc = rad_disc

    # Main loop, create names
    for m in range(0, nMethods):
        if nMethods > 1:
            _ax_disc = ax_disc[m]
            _rad_disc = rad_disc[m]

        for d in range(0, nDisc):

            name = generate_2D_name(
                prefix, ax_methods[m], _ax_disc[d], rad_methods[m], _rad_disc[d], suffix)
            if name is not None:
                _simulation_names.append(name)

    return _simulation_names


def generate_GRM_name(prefix, axP, axCells, parP, parCells, parGSM=True, suffix='.h5'):
    """Generate simulation name for bulk and particle discretized GRM.

    Parameters
    ----------
    prefix : string
        Name prefix.
    axP : int
        axial polynomial degree.
    axCells : int
        axial number of cells.
    parP : int
        particle polynomial degree.
    parCells : int
        particle number of cells.
    parDG : bool
        specifies whether DG should be used in the case of a single particle
        element, as opposed to GSM.
    suffix : string
        Name suffix, including filetype .h5.

    Returns
    -------
    string
        File name.
    """

    if int(axCells) <= 0 or int(parCells) <= 0:
        return None

    if int(axP) == 0 and int(parP) == 0:
        return prefix + "_FV_Z" + str(int(axCells)) + "parZ" + str(int(parCells)) + suffix

    elif int(axP) > 0 and int(parP) > 0:
        if int(parCells) == 1 and parGSM:
            return prefix + "_cDG_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + "_GSM_parP" + str(int(abs(parP))) + "parZ" + str(int(parCells)) + suffix
        else:
            return prefix + "_cDG_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + "_DGexInt_parP" + str(int(abs(parP))) + "parZ" + str(int(parCells)) + suffix
    elif int(axP) < 0 and int(parP) > 0:
        if int(parCells) == 1 and parGSM:
            return prefix + "_DGexInt_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + "_GSM_parP" + str(int(abs(parP))) + "parZ" + str(int(parCells)) + suffix
        else:
            return prefix + "_DGexInt_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + "parP" + str(int(abs(parP))) + "parZ" + str(int(parCells)) + suffix

    elif int(axP) > 0 and int(parP) < 0:
        return prefix + "_cDG_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + "parP" + str(int(abs(parP))) + "parZ" + str(int(parCells)) + suffix

    elif int(axP) < 0 and int(parP) < 0:
        return prefix + "_DGexInt_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + "_cDG_parP" + str(int(abs(parP))) + "parZ" + str(int(parCells)) + suffix

    elif int(parP) == 0:
        raise Exception(
            "Particle polynomial degree must be >= 1 for DG models")
    else:
        return None


def generate_2DGRM_name(prefix, axP, axCells, radP, radCells, parP, parCells, parGSM=True, suffix='.h5'):
    """Generate simulation name 2DGRM.

    Parameters
    ----------
    prefix : string
        Name prefix.
    axP : int
        axial polynomial degree.
    axCells : int
        axial number of cells.
    radP : int
        radial polynomial degree.
    radCells : int
        radial number of cells.
    parP : int
        particle polynomial degree.
    parCells : int
        particle number of cells.
    parDG : bool
        specifies whether DG should be used in the case of a single particle
        element, as opposed to GSM.
    suffix : string
        Name suffix, including filetype .h5.

    Returns
    -------
    string
        File name.
    """

    if int(axCells) <= 0 or int(parCells) <= 0 or int(radCells) <= 0:
        return None

    if int(axP) == 0 and int(parP) == 0 and int(radP) == 0:
        return prefix + "_FV_axZ" + str(int(axCells)) + "radZ" + str(int(radCells)) + "parZ" + str(int(parCells)) + suffix

    elif int(axP) > 0 and int(radP) > 0 and int(parP) > 0:

        name = prefix + "_DG_axP" + str(int(abs(axP))) + "Z" + str(
            int(axCells)) + "_radP" + str(int(abs(radP))) + "Z" + str(int(radCells))

        if int(parCells) == 1 and parGSM:
            return name + "_GSM_parP" + str(int(abs(parP))) + "Z" + str(int(parCells)) + suffix
        else:
            return name + "_parP" + str(int(abs(parP))) + "Z" + str(int(parCells)) + suffix

    else:
        return None
# TODO generate_simulation_names: scalar values as methods input!


def generate_simulation_names_GRM(
        prefix, ax_methods, ax_cells, par_methods, par_cells, suffix='.h5'
):
    """Generate simulation names for bulk and particle discretized GRM.

    Parameters
    ----------
    prefix : string
        Name prefix.
    ax_methods : np.array<int>
        axial polynomial degrees (0 for FV, -int for exInt DGSEM).
    ax_cells : int
        axial number of cells.
    par_methods : np.array<int>
        particle polynomial degrees (0 for FV, -int for collocation DGSEM).
    par_cells : int
        particle number of cells.
    suffix : string
        Name suffix, including filetype .h5.

    Returns
    -------
    string
        File names.
    """
    ax_methods = np.array(ax_methods)
    ax_cells = np.array(ax_cells)
    par_methods = np.array(par_methods)
    par_cells = np.array(par_cells)

    _simulation_names = []
    nMethods = len(ax_methods)
    # Check input parameters and infer data
    if nMethods > 1 and nMethods != ax_cells.shape[0]:
        ax_cells = ax_cells.transpose()
        par_cells = par_cells.transpose()
        if nMethods != ax_cells.shape[0] or ax_methods.shape != par_methods.shape:
            raise ValueError(
                "Methods and discretizations must have feasible dimensionalities, look up description."
            )
    if ax_cells.shape != par_cells.shape:
        raise ValueError(
            "Axial and particle discretizations must have same dimensionalities."
        )

    if (nMethods > 1):
        nDisc = ax_cells.shape[1]
    else:
        nDisc = len(ax_cells)
        _ax_cells = ax_cells
        _par_cells = par_cells

    # Main loop, create names
    for m in range(0, nMethods):
        if (nMethods > 1):
            _ax_cells = ax_cells[m]
            _par_cells = par_cells[m]

        for d in range(0, nDisc):

            name = generate_GRM_name(
                prefix, ax_methods[m], _ax_cells[d], par_methods[m], _par_cells[d], suffix)
            if name is not None:
                _simulation_names.append(name)

    return _simulation_names


def test_generate_simulation_names_GRM():

    prefix = "LinearGRM2Comp"
    ax_methods = [2]
    par_methods = [1]
    ax_cells = [8]
    par_cells = [1]
    names = np.array(generate_simulation_names_GRM(
        prefix, ax_methods, ax_cells, par_methods, par_cells, suffix='.hmpf'))
    expected_names = np.array([prefix+"_DG_P2Z8parP1parZ1.hmpf"])
    np.testing.assert_array_equal(names, expected_names)

    ax_methods = [2]
    par_methods = [1]
    ax_cells = [8, 16, 32]
    par_cells = [1, 2, 4]
    names = np.array(generate_simulation_names_GRM(
        prefix, ax_methods, ax_cells, par_methods, par_cells))
    expected_names = np.array([prefix+"_DG_P2Z8parP1parZ1.h5",
                              prefix+"_DG_P2Z16parP1parZ2.h5",
                              prefix+"_DG_P2Z32parP1parZ4.h5"])
    np.testing.assert_array_equal(names, expected_names)

    ax_methods = [0, 0,  -2, 3]
    par_methods = [-1, 0,  2, 3]
    ax_cells = [[8, 16, 32], [8, 16, 32], [8, 16, 32], [8, 16, 32]]
    par_cells = [[1, 2, 4], [1, 2, 4], [1, 2, 4], [1, 2, 4]]
    names = np.array(generate_simulation_names_GRM(
        prefix, ax_methods, ax_cells, par_methods, par_cells))
    expected_names = np.array([
        prefix+"_FV_Z8parZ1.h5",
        prefix+"_FV_Z16parZ2.h5",
        prefix+"_FV_Z32parZ4.h5",
        prefix+"_DGexInt_P2Z8parP2parZ1.h5",
        prefix+"_DGexInt_P2Z16parP2parZ2.h5",
        prefix+"_DGexInt_P2Z32parP2parZ4.h5",
        prefix+"_DG_P3Z8parP3parZ1.h5",
        prefix+"_DG_P3Z16parP3parZ2.h5",
        prefix+"_DG_P3Z32parP3parZ4.h5",
    ])

    np.testing.assert_array_equal(names, expected_names)


def generate_simulation_names_2DGRM(
        prefix, ax_methods, ax_cells, rad_methods, rad_cells, par_methods, par_cells, suffix='.h5'
):
    """Generate simulation names for bulk and particle discretized GRM.

    Parameters
    ----------
    prefix : string
        Name prefix.
    ax_methods : np.array<int>
        axial polynomial degrees
    ax_cells : int
        axial number of cells.
    rad_methods : np.array<int>
        radial polynomial degrees
    rad_cells : int
        radial number of cells.
    par_methods : np.array<int>
        particle polynomial degrees
    par_cells : int
        particle number of cells.
    suffix : string
        Name suffix, including filetype .h5.

    Returns
    -------
    string
        File names.
    """
    ax_methods = np.array(ax_methods)
    ax_cells = np.array(ax_cells)
    rad_methods = np.array(rad_methods)
    rad_cells = np.array(rad_cells)
    par_methods = np.array(par_methods)
    par_cells = np.array(par_cells)

    _simulation_names = []
    nMethods = len(ax_methods)
    # Check input parameters and infer data
    if nMethods > 1 and nMethods != ax_cells.shape[0]:
        ax_cells = ax_cells.transpose()
        rad_cells = rad_cells.transpose()
        par_cells = par_cells.transpose()
        if nMethods != ax_cells.shape[0] or ax_methods.shape != par_methods.shape:
            raise ValueError(
                "Methods and discretizations must have feasible dimensionalities, look up description."
            )
    if ax_cells.shape != par_cells.shape and ax_cells.shape != rad_cells.shape:
        raise ValueError(
            "Axial, radial, and particle discretizations must have same dimensionalities."
        )

    if (nMethods > 1):
        nDisc = ax_cells.shape[1]
    else:
        nDisc = len(ax_cells)
        _ax_cells = ax_cells
        _rad_cells = rad_cells
        _par_cells = par_cells

    # Main loop, create names
    for m in range(0, nMethods):
        if (nMethods > 1):
            _ax_cells = ax_cells[m]
            _rad_cells = rad_cells[m]
            _par_cells = par_cells[m]

        for d in range(0, nDisc):

            name = generate_2DGRM_name(
                prefix, ax_methods[m], _ax_cells[d], rad_methods[m], _rad_cells[d], par_methods[m], _par_cells[d], suffix)
            if name is not None:
                _simulation_names.append(name)

    return _simulation_names


def generate_simulation_names(
        prefix, ax_methods, ax_cells, rad_methods=None, rad_cells=None,
        par_methods=None, par_cells=None,
        suffix='.h5'):
    """Generates simulation names.

    Parameters
    ----------
    prefix : string
        Prefix of simulation name.
    ax_methods : np.array
        Specifies axial discretization method as DG -> polynomial degree; FV -> 0.
    ax_cells : np.array
        Specifies axial number of cells.
    par_methods : np.array
        Specifies particle discretization method as DG -> polynomial degree; FV -> 0.
    par_cells : np.array
        Specifies particle number of cells.
    rad_methods : np.array
        Specifies radial discretization method
    rad_cells : np.array
        Specifies radial number of cells.

    Returns
    -------
    np.array
        Simulation names for all discretizations.
    """

    if (rad_methods is None or rad_cells is None):
        if (rad_methods == rad_cells):
            if (par_methods is None or par_cells is None):
                if (par_methods == par_cells):
                    return generate_simulation_names_1D(prefix, ax_methods, ax_cells, suffix)
                else:
                    raise ValueError(
                        "Particle discretization and methods must be either None or both specified."
                    )

            else:
                return generate_simulation_names_GRM(
                    prefix, ax_methods, ax_cells, par_methods, par_cells, suffix)
        else:
            raise ValueError(
                "Radial discretization and methods must be either None or both specified."
            )
    else:
        if (par_methods is None or par_cells is None):
            if (par_methods == par_cells):
                return generate_simulation_names_2D(
                    prefix, ax_methods, ax_cells, rad_methods, rad_cells, suffix)
            else:
                raise ValueError(
                    "Particle discretization and methods must be either None or both specified."
                )

        else:
            return generate_simulation_names_2DGRM(
                prefix, ax_methods, ax_cells, rad_methods, rad_cells,
                par_methods, par_cells, suffix)


def test_generate_simulation_names():

    with pytest.raises(ValueError):
        names = np.array(generate_simulation_names("prefix", [0], [2, 4], [1]))

    methods = [2, -3]
    disc = [[2, 4, 8], [1, 2, 4]]
    prefix = "prefix"
    names = np.array(generate_simulation_names(prefix, methods, disc))
    expected_names = np.array([prefix+"_DG_P2Z2.h5",
                              prefix+"_DG_P2Z4.h5",
                              prefix+"_DG_P2Z8.h5",
                              prefix+"_DGexInt_P3Z1.h5",
                              prefix+"_DGexInt_P3Z2.h5",
                              prefix+"_DGexInt_P3Z4.h5"
                               ])
    np.testing.assert_array_equal(names, expected_names)

    methods = [0]
    disc = [2, 4, 8]
    prefix = "prefix"
    names = np.array(generate_simulation_names(prefix, methods, disc))
    expected_names = np.array([prefix+"_FV_Z2.h5",
                              prefix+"_FV_Z4.h5",
                              prefix+"_FV_Z8.h5"
                               ])
    np.testing.assert_array_equal(names, expected_names)


def std_name_prefix(transport_model, binding_model, dyn=None, n_comp=None):
    """Generate uniform name prefices depending on transport and binding model.

    Parameters
    ----------
    transport_model : string
        Applied transport model, i.e. LRMP, GRM, LRM, GRMsd
    binding_model : string
        Applied transport model, e.g. Linear, SMA, ...
    dyn : boolean
        Binding kinetics.
    n_comp : int
        Number of applied components.

    Returns
    -------
    string
        Simulation name prefix.

    """
    tm = ""
    bnd = ""

    if re.search("General|GRM", transport_model):
        tm = "GRM"
        if re.search("surface diffusion|surf|diff|sd", transport_model, re.IGNORECASE):
            tm += "sd"
    elif re.search("LRM|without", transport_model, re.IGNORECASE):
        tm = "LRM"
    elif re.search("LRMP|with", transport_model, re.IGNORECASE):
        tm = "LRMP"
    else:
        raise ValueError(
            "Unrecognized transport model: " + str(transport_model)
        )
    if re.search("2D", transport_model, re.IGNORECASE):
        tm += "2D"

    bnd = "_"
    if binding_model is not None and binding_model != "NONE":
        if dyn is not None:
            bnd += "dyn" if dyn else "req"

        if re.search("linear", binding_model, re.IGNORECASE):
            bnd += "Lin_"
        elif re.search("langmuir", binding_model, re.IGNORECASE):
            bnd += "Langmuir_"
        elif re.search("sma|steric_mass_action", binding_model, re.IGNORECASE):
            bnd += "SMA_"
        else:
            bnd += binding_model + '_'

    if n_comp is not None:
        return tm + bnd + str(int(n_comp)) + "comp_"
    else:
        return tm + bnd


_smoothness_indicator_translator_ = {
    'ALL_ELEMENTS': 'All', 'MODAL_ENERGY_LEGENDRE': 'ME'}

_limiter_translator_ = {'UNLIMITED_FORWARD_RECONSTRUCTION': 'FWD',
                        'MINMOD': 'MINMOD',
                        'SUPERBEE': 'SUPERBEE',
                        'VANALBADA': 'VANALBADA',
                        'RELAXED_VANALBADA': 'RELVANALBADA',
                        'VANLEER': 'VANLEER'}

oscillation_suppression = {
    # smoothness indicator/troubled element detector in {ALL_ELEMENTS, MODAL_ENERGY_LEGENDRE}
    'smoothness_indicator': 'MODAL_ENERGY_LEGENDRE',
    # Only required in some indicators (modal energy). relaxes sudden changes of indicator in time by setting indicator to max(relax*oldIndicator, newIndicator)
    'time_relaxation': 0.0,
    # only required in modal energy indicators, specifies the number of highest modes to be considered in [1,2]
    'num_modes': 2,
    'method': 'SUBCELL_FV_LIMITING',  # 'SUBCELL_FV_LIMITING', SUBCELL_FV
    'subcell_FV_order': 2,  # order of subcell FV scheme in {1, 2}
    'subcell_FV_max_blending': 0.1,  # maximal blending of subcell FV in [0, 1]
    # only needed for modal energy indicator: threshold for shift (for sole near zero values, the modal energy indicator might be misleading)
    'nodal_coefficient_threshold': 0.0,
    # only needed for modal energy indicator: amplitude of shift (for sole near zero values, the modal energy indicator might be misleading)
    'nodal_coefficient_shift': 0.1,
    # threshold for minimal blending, i.e. no blending if indicator is smaller
    'blending_threshold': 0.1,
    # subcell FV scheme boundary treatment; 0 : limiter, 1: central, 2: constant
    'subcell_fv_boundary_treatment': 0,
    # only required for 2nd order subcell FV; see limiter translator for available options
    'subcell_fv_limiter': 'MINMOD',
    'return_smoothness_indicator': True,
}


def oscillation_suppression_prefix(**oscillation_suppression):

    prefix = ''

    if 'method' in oscillation_suppression:
        if oscillation_suppression['method'] == 'SUBCELL_FV_LIMITING':
            prefix = 'subFV'
            if 'subcell_FV_order' in oscillation_suppression and oscillation_suppression['subcell_FV_order'] == 2:

                limiter_translator = {'UNLIMITED_FORWARD_RECONSTRUCTION': 'FWD',
                                      'MINMOD': 'MINMOD',
                                      'SUPERBEE': 'SUPERBEE',
                                      'VANALBADA': 'VANALBADA',
                                      'RELAXED_VANALBADA': 'RELVANALBADA',
                                      'VANLEER': 'VANLEER'}
                # todo check defaults
                prefix += '2Lim' + \
                    limiter_translator[str(oscillation_suppression.get(
                        'subcell_fv_limiter', 'MINMOD'))] + '_'
                prefix += 'Bndry' + \
                    str(oscillation_suppression.get(
                        'subcell_fv_boundary_treatment', 0))+'_'
            else:
                prefix += '_'
        elif oscillation_suppression['method'] == 'SUBCELL_FV':
            prefix = 'pureSubFV'
            if 'subcell_FV_order' in oscillation_suppression and oscillation_suppression['subcell_FV_order'] == 2:

                limiter_translator = {'UNLIMITED_FORWARD_RECONSTRUCTION': 'FWD',
                                      'MINMOD': 'MINMOD',
                                      'SUPERBEE': 'SUPERBEE',
                                      'VANALBADA': 'VANALBADA',
                                      'RELAXED_VANALBADA': 'RELVANALBADA',
                                      'VANLEER': 'VANLEER'}
                # todo check defaults
                prefix += '2Lim' + \
                    limiter_translator[str(oscillation_suppression.get(
                        'subcell_fv_limiter', 'MINMOD'))] + '_'
                prefix += 'Bndry' + \
                    str(oscillation_suppression.get(
                        'subcell_fv_boundary_treatment', 0))+'_'
            else:
                prefix += '_'
        elif oscillation_suppression['method'] != 'NONE':  # todo add weno
            raise Exception("Unknown oscillation suppression mode " +
                            oscillation_suppression['method'] + ".")

    if 'smoothness_indicator' in oscillation_suppression:

        prefix += 'indicator' + \
            _smoothness_indicator_translator_[
                oscillation_suppression['smoothness_indicator']]

        if oscillation_suppression['smoothness_indicator'] != "ALL_ELEMENTS":
            prefix += 'Nmode' + str(oscillation_suppression.get('num_modes', 2)
                                    ) if oscillation_suppression.get('num_modes', 2) == 1 else ""
            prefix += '_maxBlend' + \
                str(oscillation_suppression.get('subcell_FV_max_blending', 0.0))
            prefix += 'Shift' + \
                str(oscillation_suppression.get('nodal_coefficient_shift', 0.0))

            if 'blending_threshold' in oscillation_suppression and oscillation_suppression['blending_threshold'] != 0.0:
                prefix += 'thrBlend' + \
                    str(oscillation_suppression['blending_threshold'])
            if 'time_relaxation' in oscillation_suppression and oscillation_suppression['time_relaxation']:
                prefix += 'TimeRelax'

    return prefix


def calculate_DOFs(discretization, method=np.array([3]), nComp=1,
                   full_DOFs=False, model=None, nBound=None, singleDirection=False):
    """Calculate number of degrees of freedom.

    Parameters
    ----------
    discretization : np.array
        Discretization steps (GRM stored as [[bulk], [particle]]).
    method : np.array
        Discretization method, i.e. polynomial degree(s) of method with FV -> 0.
        Multiple polynomial degrees must be specified for GRM, i.e. [bulk, particle].
    model : string
        Chromatography model (LRMP, LRM, GRM), only required for LRM and LRMP
    nComp : int
        Number of components
    nBound : int
        Number of bound states
    singleDirection : Bool
        Determines whether only a single (the first) direction should be considered, e.g. to determine the refinement rate

    Returns
    -------
    np.array
        Number of Degrees of freedom.

    Raises
    -------
    ValueError
        If any discretization value is smaller than 0.
        If methods dimensionality is larger than two or does not match discretizations dimensionality.

    """
    method = np.array(method)
    discretization = np.array(discretization)
        
    if singleDirection:
        return discretization

    if np.any(discretization <= 0):
        raise ValueError(
            "Discretization must be larger than 0."
        )
    if model is None or not re.search("2D", model):
        if discretization.ndim not in [1, 2] or method.ndim not in [0, 1]:
            if not (discretization.ndim == 0 and method.ndim == 0):
                raise ValueError(
                    "Method and discretization must have feasible dimensionalities, look up description."
                )

    if nBound == None:
        nBound = nComp

    bulk_dof = 0
    par_dof = 0
    flux_dof = 0

    if model is None or not re.search("2D", model):
        
        inlet_dof = nComp if full_DOFs else 0
        
        if discretization.ndim == 2:  # GRM
            bulk_dof = (abs(method[0]) + 1) * discretization[0, :] * nComp
            if full_DOFs:
                if method[0] == 0:  # add flux states for FV discretization
                    flux_dof = bulk_dof
                par_dof = (
                    (abs(method[0]) + 1) * discretization[0, :]
                    * ((abs(method[1]) + 1) * discretization[1, :] * (nComp + nBound))
                )
    
        elif discretization.ndim in [0, 1]:  # LRM or LRMP
            bulk_dof = (abs(method) + 1) * discretization * nComp
            if full_DOFs:
                if model == "LRMP":
                    par_dof = (nComp + nBound) * abs(method+1) * discretization
                    if method == 0:  # add flux states for FV discretization
                        flux_dof = bulk_dof
                elif model == "LRM":
                    par_dof = nBound * abs(method+1) * discretization
                else:
                    raise ValueError(
                        "Unknown transport model \"" + model +
                        "\" or wrong dimensionality of input."
                    )
    
        return inlet_dof + bulk_dof + flux_dof + par_dof
    
    else:
        
        inlet_dof = nComp * (method[1] + 1) * discretization[1, :] if full_DOFs else 0
        
        if discretization.ndim not in [2, 3]:
            raise ValueError(
                "Dimensionality in determination of DOFs unclear"
            )
            
        bulk_dof = (method[0] + 1) * discretization[0, :] * (method[1] + 1) * discretization[1, :] * nComp
            
        if full_DOFs and method[0] == 0: # add flux dofs for FV
            flux_dof = bulk_dof
    
        if discretization.ndim == 3:  # 2D GRM
            if full_DOFs:
                if re.search(model, "LRMP"):
                    par_dof = (nComp + nBound) * bulk_dof
                elif re.search(model, "LRM"):
                    par_dof = (nComp + nBound) * bulk_dof
                elif re.search(model, "GRM"):
                    par_dof = (nComp + nBound) * (method[2] + 1) * discretization[2, :] * bulk_dof
                else:
                    raise ValueError(
                        "Unknown transport model \"" + model +
                        "\" or wrong dimensionality of input."
                    )
            
        return inlet_dof + bulk_dof + par_dof + flux_dof


def test_calculate_DOFs():

    # test GRM
    disc = [[4, 8, 10, 32], [1, 2, 3, 5]]
    method = [3, 2]
    nComp = 3
    nBound = nComp

    inletDOFs = np.array([1, 1, 1, 1]) * nComp
    bulkDOFs = (method[0] + 1) * np.array(disc[0]) * nComp
    fluxDOFs = bulkDOFs
    parDOFs = np.multiply(
        (nComp + nBound) * (method[1] + 1) * np.array(disc[1]),
        (method[0] + 1) * np.array(disc[0])
    )
    DOFs = inletDOFs + bulkDOFs + fluxDOFs + parDOFs

    np.testing.assert_array_equal(
        bulkDOFs,
        calculate_DOFs(disc, method, nComp=nComp, full_DOFs=False)
    )
    np.testing.assert_array_equal(
        DOFs,
        calculate_DOFs(disc, method, nComp=nComp, full_DOFs=True)
    )

    # test LRM
    disc = [4, 8, 10, 32]
    method = [1]
    nComp = 3
    nBound = 4

    inletDOFs = np.array([1, 1, 1, 1]) * nComp
    bulkDOFs = (method[0] + 1) * np.array(disc) * nComp
    parDOFs = (method[0] + 1) * np.array(disc) * nBound
    DOFs = inletDOFs + bulkDOFs + parDOFs

    np.testing.assert_array_equal(
        bulkDOFs,
        calculate_DOFs(disc, method, nComp=nComp, nBound=nBound,
                       full_DOFs=False,
                       model="LRM")
    )
    np.testing.assert_array_equal(
        DOFs,
        calculate_DOFs(disc, method, nComp=nComp, nBound=nBound,
                       full_DOFs=True,
                       model="LRM")
    )

    # test LRMP
    disc = [4, 8, 10, 32]
    method = [1]
    nComp = 3
    nBound = 4

    inletDOFs = np.array([1, 1, 1, 1]) * nComp
    bulkDOFs = (method[0] + 1) * np.array(disc) * nComp
    parDOFs = (method[0] + 1) * np.array(disc) * (nComp + nBound)
    DOFs = inletDOFs + bulkDOFs + parDOFs

    np.testing.assert_array_equal(
        bulkDOFs,
        calculate_DOFs(disc, method, nComp=nComp, nBound=nBound,
                       full_DOFs=False,
                       model="LRMP")
    )
    method = [0]  # FV, i.e. with flux
    inletDOFs = np.array([1, 1, 1, 1]) * nComp
    bulkDOFs = (method[0] + 1) * np.array(disc) * nComp
    fluxDOFs = bulkDOFs
    parDOFs = (method[0] + 1) * np.array(disc) * (nComp + nBound)
    DOFs = inletDOFs + bulkDOFs + fluxDOFs + parDOFs
    np.testing.assert_array_equal(
        DOFs,
        calculate_DOFs(disc, method, nComp=nComp, nBound=nBound,
                       full_DOFs=True,
                       model="LRMP")
    )


def calculate_eoc(discretization, error):
    """Calculate experimental order of convergence.

    Parameters
    ----------
    discretization : np.array
        Discretization steps.
    error : np.array
        Errors.

    Returns
    -------
    np.array
        Experimental order of convergence

    Raises
    -------
    ValueError
        If any discretization value is smaller than 0.
        If any error value is smaller than 0.

    """
    error = np.asarray(error)
    discretization = np.asarray(discretization)

    if np.any(discretization <= 0):
        raise ValueError(
            "Discretization must be larger than 0."
        )
    if np.any(error < 0):
        raise ValueError(
            "Error must be larger or equal than 0 (must be provided as absolute error)."
        )
    if np.any(error < np.finfo(float).eps):
        error[error < np.finfo(float).eps] = np.finfo(float).eps

    ratio_discretization = discretization[:-1]/discretization[1:]
    ratio_error = error[1:]/error[:-1]

    return np.log2(ratio_error)/np.log2(ratio_discretization)


def test_eoc():
    dof = [1, 2, 4, 8, 16]
    error = [0.1, 0.05, 0.025, 0.0125, 0.00625]

    eoc_expected = [1, 1, 1, 1]
    eoc = calculate_eoc(dof, error)

    np.testing.assert_almost_equal(eoc, eoc_expected)

    with pytest.raises(ValueError):
        dof_zero = [1, 2, 4, 8, 0]
        eoc = calculate_eoc(dof_zero, error)

    with pytest.raises(ValueError):
        error_smaller_zero = [-0.1, 0.05, 0.025, 0.0125, 0.00625]
        eoc = calculate_eoc(dof, error_smaller_zero)

    # Test eps
    error_zero = [0.1, 0.05, 0.025, 0.0125, 0]

    eoc_expected = [1, 1, 1, 45.67807191]
    eoc = calculate_eoc(dof, error_zero)
    np.testing.assert_almost_equal(eoc, eoc_expected)


"""
 TODO Function: Automatic refinement until error tolerance is reached
 -> convergence table

"""


# TODO: add normalization to temporal L1 and Linf error !
def convergency_table(method,
                      disc,
                      abs_errors,
                      sim_names=None,
                      error_types=np.array(["max"]),
                      delta=1.0,
                      normalization=1.0,
                      full_DOFs=False,
                      transport_model=None):
    """Calculate convergency table for one spatial method for column outlet.

    Parameters
    ----------
    method : np.array
        Discretization method, i.e. polynomial degree(s) of method with FV -> 0.
        Multiple polynomial degrees must be specified for GRM, i.e. [bulk, particle].
    disc : np.array
        Discretization steps (GRM stored as [[bulk], [radial] [particle]]).
    sim_names : np.array
        String with simulation names, if compute times should be included.
    abs_errors : np.array
        Absolute errors of simulations corresponding to discretization array.
    error_types : np.array
        Contains error types for which convergency is computed.
    delta : float or np array
        discrete step sizes (i.e. spatial cells or outlet time steps).
        Dimensionality: None or nSim or nSim x nCells
    normalization : float
        normalization factor for errors.
    full_DOFs : boolean
        Default recommended for 1.5D models to neglect particle dimension

    Returns
    -------
    np.array
        Header of convergency table
    np.array
        Convergency table

    Raises
    -------
    ValueError
        If discretization is not one or two dimensional.
        If dimensionality of errors and discretization does not match.
        If error type is unknown.

    """
    disc = np.array(disc)
    error_types = np.array(error_types)
    abs_errors = np.abs(np.array(abs_errors, dtype=object))
    header = []
    header.append("$N_e^z$")
    table = []
    # Infer dimensionality of table
    offRow = 1
    if disc.ndim == 2:
        if len(disc) == 3 and (transport_model is not None and re.search("2D", transport_model)): # 2DGRM
            nDisc = disc.shape[1]
            offCol = 3
            header.append("$N_e^r$")
            header.append("$N_e^p$")
            table.append(disc[0])
            table.append(disc[1])
            table.append(disc[2])
        elif len(disc) == 2 and (transport_model is not None and re.search("2D", transport_model)): # 2D LRM or LRMP
            nDisc = disc.shape[1]
            offCol = 2
            header.append("$N_e^r$")
            table.append(disc[0])
            table.append(disc[1])
        elif len(disc) == 2: # GRM
            nDisc = disc.shape[1]
            offCol = 2
            header.append("$N_e^p$")
            table.append(disc[0])
            table.append(disc[1])
    elif (disc.ndim == 1):   # LRMP, LRM
        nDisc = len(disc)
        offCol = 1
        table.append(disc)
    else:
        raise ValueError(
            "Discretization must be array or n x 2 matrix."
        )
    if disc.any() <= 0:
        raise ValueError(
            "Number of cells must be >= 1."
        )
    # Check input
    if abs_errors.ndim == 1:
        if (nDisc != len(abs_errors)):
            raise ValueError(
                "Not the same number of discretizations as errors."
            )
    else:
        if (nDisc != abs_errors.shape[0]):
            abs_errors = abs_errors.transpose()
            if (nDisc != abs_errors.shape[0]):
                raise ValueError(
                    "Not the same number of discretizations as errors."
                )

    DOFs = calculate_DOFs(disc, method, full_DOFs=full_DOFs, model=transport_model)

    # Main loop: calculate all errors and EOC's
    for error_type in range(0, len(error_types)):

        current_errors = np.zeros(nDisc)

        if (re.search("max", error_types[error_type], re.IGNORECASE)
           or re.search("Linf", error_types[error_type], re.IGNORECASE)
           or re.search("Linfty", error_types[error_type], re.IGNORECASE)):

            table_error_name = "Max."
            for d in range(0, nDisc):
                if abs_errors.ndim == 1:
                    current_errors[d] = calculate_Linf_error(abs_errors[d])
                else:
                    current_errors[d] = calculate_Linf_error(abs_errors[d, :])
                if re.search("norm", error_types[error_type], re.IGNORECASE):
                    current_errors[d] /= normalization if isinstance(
                        normalization, (int, float)) else normalization[d]

        elif (re.search("L1_spatial", error_types[error_type], re.IGNORECASE)
              or re.search("L^1_spatial", error_types[error_type], re.IGNORECASE)):

            table_error_name = "$L^1$"

            for d in range(0, nDisc):
                factor = delta if isinstance(
                    delta, (int, float)) else delta[d]
                if abs_errors.ndim == 1:
                    current_errors[d] = calculate_bulk_L1_error(
                        abs(method[0]), factor, abs_errors[d], reference=-1, weights=-1)
                else:
                    current_errors[d] = calculate_bulk_L1_error(
                        abs(method[0]), factor, abs_errors[d, :], reference=-1, weights=-1)
                if re.search("norm", error_types[error_type], re.IGNORECASE):
                    current_errors[d] /= normalization if isinstance(
                        normalization, (int, float)) else normalization[d]

        elif (re.match("L1", error_types[error_type], re.IGNORECASE)
              or re.match("L^1", error_types[error_type], re.IGNORECASE)):

            table_error_name = "$L^1$"
            for d in range(0, nDisc):
                if abs_errors.ndim == 1:
                    current_errors[d] = calculate_L1_error(
                        abs_errors[d], weights=delta)
                else:
                    current_errors[d] = calculate_L1_error(
                        abs_errors[d, :], weights=delta)

        elif (re.match("L2", error_types[error_type], re.IGNORECASE)
              or re.match("L^2", error_types[error_type], re.IGNORECASE)):

            table_error_name = "$L^2$"
            for d in range(0, nDisc):
                if abs_errors.ndim == 1:
                    current_errors[d] = calculate_L2_error(
                        abs_errors[d], weights=delta)
                else:
                    current_errors[d] = calculate_L2_error(
                        abs_errors[d, :], weights=delta)

        else:
            raise ValueError(
                "Unknown error Type " + error_type + "."
            )

        # we assume that all spatial directions are refined at the same rate and thus only consider one direction to compute the EOC
        if transport_model is not None and re.search("2D", transport_model):
            eocDOF = calculate_DOFs(disc[0], method[0], full_DOFs=False, model=transport_model, singleDirection=True)
        else:
            eocDOF = calculate_DOFs(disc, method, full_DOFs=False, model=transport_model)
            
        table.append(current_errors)
        header.append(table_error_name + " error")
        table.append(
            np.insert(
                calculate_eoc(eocDOF, current_errors), 0, 0.0
            )
        )
        header.append(table_error_name + " EOC")

    if sim_names is not None:
        header.append('Sim. time')
        table.append(get_compute_times(sim_names))

    return header, np.array(table).transpose()


def test_convergency_table():

    # GRM test for L1/max error
    expected_order = 3
    method = np.array([1, 1]) * (expected_order - 1)
    discretizations = [[4, 8, 16], [1, 2, 4]]
    full_DOFs = False

    error_type = ["max"]
    initial_errors = np.random.rand(100)
    error_factors = np.zeros(len(discretizations[0]))
    for i in range(0, len(error_factors)):
        error_factors[i] = 2 ** (expected_order * i)
    abs_errors = np.zeros([len(error_factors), len(initial_errors)])

    for factor in range(0, len(error_factors)):
        abs_errors[factor] = initial_errors / error_factors[factor]

    header, table = convergency_table(method=method, disc=discretizations, abs_errors=abs_errors,
                                      error_types=error_type, full_DOFs=full_DOFs)

    np.testing.assert_array_equal(header, np.array(
        ["$N_e^z$", "$N_e^p$", 'Max. error', 'Max. EOC']))
    np.testing.assert_almost_equal(
        table[:, 0], discretizations[0])  # check axial disc
    np.testing.assert_almost_equal(
        table[:, 1], discretizations[1])  # check particle disc
    np.testing.assert_almost_equal(
        table[:, 2], np.amax(abs_errors, axis=1))  # check errors
    np.testing.assert_almost_equal(table[1:, 3], np.ones(
        len(discretizations[0])-1) * expected_order)  # check EOC

    # LRMP/LRM test for L1/max error
    expected_order = 3
    method = (expected_order - 1)
    discretizations = [4, 8, 16]
    full_DOFs = False

    error_type = ["max"]
    initial_errors = np.random.rand(100)
    error_factors = np.zeros(len(discretizations))
    for i in range(0, len(error_factors)):
        error_factors[i] = 2 ** (expected_order * i)
    abs_errors = np.zeros([len(error_factors), len(initial_errors)])

    for factor in range(0, len(error_factors)):
        abs_errors[factor] = initial_errors / error_factors[factor]

    header, table = convergency_table(method=method, disc=discretizations, abs_errors=abs_errors,
                                      error_types=error_type, full_DOFs=full_DOFs)

    np.testing.assert_array_equal(header, np.array(
        ["$N_e^z$", 'Max. error', 'Max. EOC']))
    np.testing.assert_almost_equal(
        table[:, 0], discretizations)  # check axial disc
    np.testing.assert_almost_equal(
        table[:, 1], np.amax(abs_errors, axis=1))  # check errors
    np.testing.assert_almost_equal(table[1:, 2], np.ones(
        len(discretizations)-1) * expected_order)  # check EOC


def calculate_convergence_tables_from_files(
        prefix, reference,
        ax_methods, ax_cells,
        par_methods=None, par_cells=None,
        unit='001', error_types=['max'], which='outlet', full_DOFs=False):
    """Calculate convergency tables from simulation files.

    Parameters
    ----------
    prefix: string
        Prefix of (all) the file names
    reference: np.ndarray or Cadet object or string
        Reference solution, Cadet object or h5 file name
    ax_methods : np.array
        Axial discretization methods, i.e. polynomial degree(s) of method with FV -> 0.
    ax_cells : np.array
        Axial discretization steps, i.e. cells.
    par_methods : np.array
        Particle discretization methods, i.e. polynomial degree(s) of method with FV -> 0.
    par_cells : np.array
        Particle discretization steps, i.e. cells.
    unit : string
        Unit name '000'-'999'
    error_types: np.array
        Contains error types for which convergence is computed.
    which : string
        Specifies which solution is considered, i.e. 'outlet', 'bulk', ...
    full_DOFs: boolean
        Default recommended! Specifies whether or not particle DOFs are considered for convergency.

    Returns
    -------
    np.array
        Header of convergency table
    np.array
        Convergency tables

    """
    tables = []
    ax_methods = np.array(ax_methods)

    if par_methods is None:  # i.e. LRMP or LRM
        par_methods = np.array(ax_methods.size * [None])
        par_cells = np.array(ax_methods.size * [None])

    for m in range(0, ax_methods.size):

        # Get simulation names
        simulation_names = generate_simulation_names(
            prefix=prefix,
            ax_methods=[ax_methods[m]],
            ax_cells=ax_cells[m],
            par_methods=[par_methods[m]],
            par_cells=par_cells[m])

        # Calculate errors
        errors = calculate_all_abs_errors(
            simulation_names, reference, unit=unit, which=which)

        # Calculate convergence tables
        header, table = convergency_table(
            method=[ax_methods[m], par_methods[m]],
            disc=[ax_cells[m], par_cells[m]],
            abs_errors=errors,
            error_types=error_types,
            full_DOFs=full_DOFs
        )

        tables.append(table)

    return header, tables


def get_compute_times_from_files(
        prefix,
        ax_methods, ax_cells,
        par_methods=None, par_cells=None):
    """returns compute times from simulation files.

    Parameters
    ----------
    prefix: string
        Prefix of (all) the file names
    ax_methods : np.array
        Axial discretization methods, i.e. polynomial degree(s) of method with FV -> 0.
    ax_cells : np.array
        Axial discretization steps, i.e. cells.
    par_methods : np.array
        Particle discretization methods, i.e. polynomial degree(s) of method with FV -> 0.
    par_cells : np.array
        Particle discretization steps, i.e. cells.

    Returns
    -------
    np.array
        Compute times per method

    """
    times = []
    ax_methods = np.array(ax_methods)

    if par_methods is None:  # i.e. LRMP or LRM
        par_methods = np.array(ax_methods.size * [None])
        par_cells = np.array(ax_methods.size * [None])

    for m in range(0, ax_methods.size):
        # Get simulation name
        simulation_names = generate_simulation_names(
            prefix=prefix,
            ax_methods=[ax_methods[m]],
            ax_cells=ax_cells[m],
            par_methods=[par_methods[m]],
            par_cells=par_cells[m])

        times.append(get_compute_times(simulation_names))

    return times


def std_plot(x_axis, y_axis, **kwargs):
    """adds a line to the current plot.

    Parameters
    ----------
    x_axis : array
        x values (one line per row)
    y_axis : array
        y values (one line per row)
    shape : array size 2
        shape of plot
    font_size : float
        Scale factor of font sozes, e.g. legend, title, ...
    x_scale : string
        Scale of x-axis, e.g. 'log'

    Returns
    -------
    np.array
        Compute times per method

    """
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)

    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = 3
    if 'marker' not in kwargs:
        kwargs['marker'] = 'o'
    if 'markersize' not in kwargs:
        kwargs['markersize'] = 12

    if x_axis.ndim == 2:
        for i in range(0, x_axis.shape[0]):
            plt.plot(x_axis[i], y_axis[i], **kwargs)

    elif x_axis.ndim == 1:
        plt.plot(x_axis, y_axis, **kwargs)
    else:
        raise ValueError(
            "Unfeasible dimensionality of input, i.e. ndim = " +
            str(x_axis.ndim)
        )


def std_plot_prep(benchmark_plot=True, **kwargs):
    """Create standard plot setting.

    Parameters
    ----------
    x_axis : array
        x values (one line per row)
    y_axis : array
        y values (one line per row)
    shape : array size 2
        shape of plot
    font_size : float
        Scale factor of font sozes, e.g. legend, title, ...
    x_scale : string
        Scale of x-axis, e.g. 'log'

    Returns
    -------
    np.array
        Compute times per method

    """
    if 'title' in kwargs:
        title = kwargs.pop('title')
    else:
        title = None
    if 'x_label' in kwargs:
        x_label = kwargs.pop('x_label')
    else:
        x_label = None
    if 'y_label' in kwargs:
        y_label = kwargs.pop('y_label')
    else:
        y_label = None
    if 'shape' in kwargs:
        shape = kwargs.pop('shape')
    elif benchmark_plot:
        shape = [20, 10]
    else:
        shape = [10, 10]
    if 'font_size_fac' in kwargs:
        font_size_factor = kwargs.pop('font_size_fac')
    else:
        font_size_factor = 1.5
    if 'x_scale' in kwargs:
        x_scale = kwargs.pop('x_scale')
    elif benchmark_plot:
        x_scale = 'log'
    if 'y_scale' in kwargs:
        y_scale = kwargs.pop('y_scale')
    else:
        y_scale = 'log'
    if 'x_lim' in kwargs:
        x_lim = kwargs['x_lim']
    else:
        x_lim = None
    if 'y_lim' in kwargs:
        y_lim = kwargs['y_lim']
    else:
        y_lim = None
    # TODO: ignores ticks right now
    # if 'x_ticks' in kwargs:
    #     x_ticks = kwargs['x_ticks']
    # elif x_lim is not None:
    #     x_ticks = np.logspace(x_lim[0], y_lim[1], num=4, endpoint=True)
    # if 'y_ticks' in kwargs:
    #     y_ticks = kwargs['y_ticks']
    # elif y_lim is not None:
    #     y_ticks = np.logspace(y_lim[0], y_lim[1], num=12, endpoint=True)

    if 'linewidth' not in kwargs and benchmark_plot:
        kwargs['linewidth'] = 3
    if 'marker' not in kwargs:
        kwargs['marker'] = 'o'
    if 'markersize' not in kwargs and benchmark_plot:
        kwargs['markersize'] = 12

    plt.rcParams["figure.figsize"] = (shape[0], shape[1])

    plt.legend(fontsize=15*font_size_factor)
    plt.grid()
    if benchmark_plot or 'y_scale' in kwargs:
        plt.yscale(y_scale)
    if benchmark_plot or 'x_scale' in kwargs:
        plt.xscale(x_scale)
    if x_label is not None:
        plt.xlabel(x_label, fontsize=15*font_size_factor)
    if x_label is not None:
        plt.ylabel(y_label, fontsize=15*font_size_factor)
    if x_lim is not None:
        plt.xlim(x_lim[0], x_lim[1])
        plt.xticks(fontsize=15*font_size_factor)  # todo x_ticks
    else:
        plt.xticks(fontsize=15*font_size_factor)
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])
        plt.yticks(fontsize=15*font_size_factor)  # todo y_ticks
    else:
        plt.yticks(fontsize=15*font_size_factor)
    if title is not None:
        plt.title(title, fontsize=15*font_size_factor)

def std_latex_table(data_frame, latex_columns, math_mode=True, error_digits=3):
    """Create latex table from dataframe.

    Parameters
    ----------
    data_frame : pandas.dataframe
        table
    latex_columns : array
        column names to be included
    math_mode : bool
        print numbers in latex math mode
    error_digits : int
        Number of digits for errors.

    Returns
    -------
    string
        Latex table code

    """
    float_format = '%.' + str(error_digits) + 'e'
    eoc_format = '{:.2f}'
    if math_mode:
        float_format = '$' + float_format + '$'
        eoc_format = '$' + eoc_format + '$'

    # EOC as fixed point number
    eoc_columns = data_frame.filter(like='EOC').columns

    data_frame[eoc_columns] = data_frame[eoc_columns].applymap(
        eoc_format.format)

    data_frame['$N_d$'] = data_frame['$N_d$'].astype(int)
    data_frame['$N_e^z$'] = data_frame['$N_e^z$'].astype(int)
    data_frame['$N_e^p$'] = data_frame['$N_e^p$'].astype(int)

    if math_mode:
        return data_frame.to_latex(index=False,
                                   float_format=float_format,
                                   columns=latex_columns,
                                   escape=False,
                                   formatters={
                                       # needed for math mode $$
                                       '$N_d$': '${:d}$'.format,
                                       '$N_e^z$': '${:d}$'.format,
                                       '$N_e^p$': '${:d}$'.format,
                                   })
    else:
        return data_frame.to_latex(index=False,
                                   float_format=float_format,
                                   columns=latex_columns,
                                   escape=False,
                                   formatters={
                                       '$N_d$': '{:d}'.format,
                                       '$N_e^z$': '{:d}'.format,
                                       '$N_e^p$': '{:d}'.format,
                                   })


def recalculate_results(file_path, models,
                        ax_methods, ax_cells,
                        exact_names,
                        unit='001', which='outlet',
                        par_methods=[None], par_cells=None,
                        rad_methods=[None], rad_cells=None,
                        incl_min_val=True,
                        transport_model=None, ncomp=None, nbound=None,
                        save_path_=None,
                        simulation_names=None,
                        save_names=None,
                        export_results=False,
                        **kwargs):
    """Calculate results (EOC, Sim. time, errors) for saved simulations.

    Parameters
    ----------
    file_path : string
        Specifies path to where simulation files are stored.
    models : array
        Model names
    ax_methods : array
        Axial discretization methods, i.e. 0:FV;Z^+:cDG;Z^-:DG.
    ax_cells : array
        Axial number of cells
    par_methods : array
        Particle discretization methods, i.e. 0:FV;Z^+:cDG;Z^-:DG.
    par_cells : array
        Particle number of cells.
    rad_methods : array
        Radial discretization methods
    rad_cells : array
        Radial number of cells.
    exact_names : array
        Strings with reference solution names (full, without path)
            or np.arrays with reference solutions
    unit : string
        Unit of interest, i.e. '000'-'999'.
    which : string
        Concentration of interest, i.e. 'outlet', 'bulk'.
    transport_model : list or np.array
        Considered transport models, defaults to search string of models
        in variable 'models'.
    ncomp : int
        Number of components for each model, defaults to search string
        '_/d+comp_' in variable 'models'.
    nbound : int
        Number of total bound states for each model, defaults to ncomp
        (mult. parTypes not supported).
    save_path : string
        Specifies the directory to which the tables are saved, defaults to file path
    simulation_names : np.array of strings
        Simulation names, defaults to None, i.e. standard simulation names based on model and discretization are generated
    export_results : bool
        Specifies whether result table should be written to a csv file

    """

    for modelIdx in range(0, len(models)):

        extra_keys = {}        

        # needed for DOF calculation
        if transport_model is None:
            try:
                transport_model = re.search(
                    '2DLRM(?!P)|2DLRMP|2DGRM|LRM(?!P)|LRMP|GRM',
                    models[modelIdx],
                    re.IGNORECASE).group(0)
            except:
                raise ValueError(
                    "Considered transport model neither explicitly specified \
                    nor given in model name: " + models[modelIdx])
        if ncomp is None:
            try:
                ncomp = int(re.search(r'(_)(\d+)(comp)',
                                      models[modelIdx],
                                      re.IGNORECASE).group(2))
            except:
                raise ValueError(
                    "Number of components neither explicitly specified \
                    nor given in model name: " + models[modelIdx])
        if nbound is None:
            nbound = ncomp

        # Data per discretization method
        abs_errors = []
        DoFs = []
        bulk_DoFs = []
        convergence_tables = []
        if incl_min_val:
            min_val = []

        if type(exact_names[modelIdx]) is not str and type(exact_names[modelIdx]) is not None:
            
            reference = exact_names[modelIdx]
            
        elif which == 'radial_outlet':
            
            extra_keys['domain_end'] = sim_go_to(get_simulation(file_path+exact_names[modelIdx]).root,
                      ['input', 'model', 'unit_'+unit, 'col_radius']
                      )
            extra_keys['ref_coords'] = get_radial_coordinates(file_path+exact_names[modelIdx], unit)
            nRad = len(extra_keys['ref_coords'])
            reference = []
            kwargs.update(extra_keys)
            
            for rad in range(nRad):
                reference.append(get_solution(file_path+exact_names[modelIdx], 'unit_'+unit, 'outlet_port_{:03d}'.format(rad), kwargs.get(
                    'comp', [-1])))
            reference=np.array(reference)
        else:
            
            reference = get_solution(file_path+exact_names[modelIdx], 'unit_'+unit, which, kwargs.get(
                'comp', [-1]), **{'sensIdx': kwargs.get('sensIdx', 0)})

        if simulation_names is None:
            simulation_names = []
            for m in range(0, len(ax_methods)):

                if len(ax_methods) > 1:
                    ax_cells_ = ax_cells[m]
                    par_cells_ = None if par_cells is None else par_cells[m]
                    rad_cells_ = None if rad_cells is None else rad_cells[m]
                else:
                    ax_cells_ = ax_cells
                    par_cells_ = par_cells
                    rad_cells_ = rad_cells

                if rad_methods[0] is None:
                    if par_methods[0] is None:
                        simulation_names.append(
                            generate_simulation_names(
                                prefix=file_path+models[modelIdx],
                                ax_methods=[ax_methods[m]], ax_cells=ax_cells_
                            )
                        )
                    else:
                        simulation_names.append(
                            generate_simulation_names(
                                prefix=file_path+models[modelIdx],
                                ax_methods=[ax_methods[m]], ax_cells=ax_cells_,
                                par_methods=[par_methods[m]
                                             ], par_cells=par_cells_
                            )
                        )
                else:
                    if par_methods[0] is None:
                        simulation_names.append(
                            generate_simulation_names(
                                prefix=file_path+models[modelIdx],
                                ax_methods=[ax_methods[m]], ax_cells=ax_cells_,
                                rad_methods=[rad_methods[m]
                                             ], rad_cells=rad_cells_
                            )
                        )
                    else:
                        simulation_names.append(
                            generate_simulation_names(
                                prefix=file_path+models[modelIdx],
                                ax_methods=[ax_methods[m]], ax_cells=ax_cells_,
                                par_methods=[par_methods[m]
                                             ], par_cells=par_cells_,
                                rad_methods=[rad_methods[m]
                                             ], rad_cells=rad_cells_
                            )
                        )

        nTimePoints = len(get_solution_times(simulation_names[0][0]))
        
        error_weight = 1.0/nTimePoints if kwargs.get('time_weighted_error', False) else 1.0
        
        for m in range(0, len(ax_methods)):

            if len(ax_methods) > 1:
                ax_cells_ = ax_cells[m]
                par_cells_ = None if par_cells is None else par_cells[m]
                rad_cells_ = None if rad_cells is None else rad_cells[m]
            else:
                ax_cells_ = ax_cells
                par_cells_ = par_cells
                rad_cells_ = rad_cells

            abs_errors.append(
                calculate_all_abs_errors(
                    simulation_names[m],
                    reference,
                    unit,
                    which=which,
                    polyDeg=rad_methods[m], nCells=rad_cells_,
                    **kwargs
                )
            )
            if incl_min_val:
                min_val.append(
                    calculate_all_min_vals(
                        simulation_names[m],
                        unit,
                        which=which
                    )
                )
            if par_methods[0] is None and rad_methods[0] is None:
                DoFs.append(
                    calculate_DOFs(ax_cells_,
                                   ax_methods[m],
                                   full_DOFs=True,
                                   model=transport_model,
                                   nComp=ncomp, nBound=nbound)
                )
                bulk_DoFs.append(
                    calculate_DOFs(ax_cells_,
                                   ax_methods[m],
                                   full_DOFs=False,
                                   model=transport_model,
                                   nComp=ncomp, nBound=nbound)
                )
                # Convergence Table
                header, table = convergency_table(ax_methods[m],
                                                  ax_cells_,
                                                  abs_errors[m],
                                                  delta=error_weight,
                                                  error_types=kwargs.get('error_types', ['max', 'L1', 'L2']),
                                                  sim_names=simulation_names[m])

            else:
                
                aux_methods = [ax_methods[m]]
                aux_cells = [ax_cells_]
                
                if rad_methods[0] is not None:
                    aux_methods.append(rad_methods[m])
                    aux_cells.append(rad_cells_)
                
                if par_methods[0] is not None:
                    aux_methods.append(par_methods[m])
                    aux_cells.append(par_cells_)
                
                DoFs.append(
                    calculate_DOFs(aux_cells,
                                   aux_methods,
                                   full_DOFs=True,
                                   model=transport_model,
                                   nComp=ncomp, nBound=nbound)
                )
                bulk_DoFs.append(
                    calculate_DOFs(aux_cells,
                                   aux_methods,
                                   full_DOFs=False,
                                   model=transport_model,
                                   nComp=ncomp, nBound=nbound)
                )
                # Convergence Table
                header, table = convergency_table(
                    aux_methods,
                    aux_cells,
                    abs_errors[m],
                    delta=error_weight,
                    error_types=kwargs.get('error_types', ['max', 'L1', 'L2']),
                    sim_names=simulation_names[m],
                    full_DOFs=False,
                    transport_model=transport_model
                )

            if incl_min_val:
                table = np.column_stack((np.array(table), min_val[m]))
                header.append('Min. value')

            convergence_tables.append(table)

            # Export Data
            if save_names == None:
                if ax_methods[m] == 0:
                    save_name = models[modelIdx] + "_FV"
                elif ax_methods[m] < 0:
                    save_name = models[modelIdx] + \
                        "_DGexInt_P" + str(abs(ax_methods[m]))
                else:
                    save_name = models[modelIdx] + \
                        "_DG_P" + str(abs(ax_methods[m]))
                if par_methods[0] is not None:
                    if par_methods[m] < 0:
                        save_name += "_cDG_parP" + str(abs(par_methods[m]))
                    elif abs(par_methods[m]) != abs(ax_methods[m]):
                        save_name += "parP" + str(abs(par_methods[m]))
            else:
                save_name = save_names[m]

            # add (bulk) DoFs to output table
            header = np.insert(header, len(header), 'DoF')
            result = np.hstack(
                (convergence_tables[m], np.atleast_2d(DoFs[m]).T))
            header = np.insert(header, len(header), 'Bulk DoF')
            result = np.hstack(
                (result, np.atleast_2d(bulk_DoFs[m]).T))

            # add method to output table
            header = np.insert(header, 0, '$N_d$')
            arr = np.empty(len(ax_cells_), dtype=object)
            arr[:] = str(int(abs(ax_methods[m])))
            result = np.column_stack((arr, result))

            # export
            if export_results:
                if save_path_ is None:
                    save_path_ = file_path
                pd.DataFrame(result, columns=header).to_csv(
                    save_path_ + save_name + ".csv", index=False)

    return pd.DataFrame(result, columns=header)


def compare_h5_files(file1, file2):
    """Compares two h5 files, i.e. their structure, datatypes and datasets.

    Parameters
    ----------
    fil1 : string
        First file name.
    fil2 : string
        Second file name.
    """

    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        # Compare attributes of root group
        attrs1 = dict(f1.attrs.items())
        attrs2 = dict(f2.attrs.items())
        if attrs1 != attrs2:
            print("Attributes of root group are not equal")
            print("File 1:", attrs1)
            print("File 2:", attrs2)
            return False

        # Recursively compare file structure and datasets
        def compare_items(item1, item2):
            if isinstance(item1, h5py.Group):
                if not isinstance(item2, h5py.Group):
                    print(f"File 2 does not contain group '{item1.name}'")
                    return False
                # Compare group attributes
                attrs1 = dict(item1.attrs.items())
                attrs2 = dict(item2.attrs.items())
                if attrs1 != attrs2:
                    print(f"Attributes of group '{item1.name}' are not equal")
                    print("File 1:", attrs1)
                    print("File 2:", attrs2)
                    return False
                # Compare group keys
                keys1 = set(item1.keys())
                keys2 = set(item2.keys())
                if keys1 != keys2:
                    print(f"Keys in group '{item1.name}' are not equal")
                    print("File 1:", keys1)
                    print("File 2:", keys2)
                    return False
                # Recursively compare group items
                for key in keys1:
                    if not compare_items(item1[key], item2[key]):
                        return False
            elif isinstance(item1, h5py.Dataset):
                if not isinstance(item2, h5py.Dataset):
                    print(f"File 2 does not contain dataset '{item1.name}'")
                    return False
                # Compare dataset attributes
                attrs1 = dict(item1.attrs.items())
                attrs2 = dict(item2.attrs.items())
                if attrs1 != attrs2:
                    print(
                        f"Attributes of dataset '{item1.name}' are not equal")
                    print("File 1:", attrs1)
                    print("File 2:", attrs2)
                    return False
                # Compare dataset shape and data
                if item1.dtype != item2.dtype:
                    # if not (item1.dtype.kind == 'S' and item2.dtype.kind in ('S')):
                    print(
                        f"Dataset '{item1.name}' has different data types: {item1.dtype} and {item2.dtype}")
                    return False
                if item1.shape != item2.shape:
                    if not (  # scalar values might have different shapes
                            (item1.shape == () or all(dim == 1 for dim in item1.shape)) and
                            (item2.shape == () or all(
                                dim == 1 for dim in item2.shape))
                    ):
                        print(
                            f"Dataset '{item1.name}' has different shapes: {item1.shape} and {item2.shape}")
                        return False
                if (item1.shape == () or all(dim == 1 for dim in item1.shape)):
                    if item1[()] != item2[()]:
                        print(
                            f"Dataset '{item1.name}' has different scalar values: {item1[()]} and {item2[()]}")
                        return False
                # else:
                #     print(item1.shape == ())
                #     print(item2)
                #     return False
                elif not (item1[:] == item2[:]).all():
                    print(f"Dataset '{item1.name}' has different data values")
                    return False
            else:
                # Unsupported object type
                print(f"Unsupported object type '{type(item1)}'")
                return False
            return True

        # Compare root group items
        if not compare_items(f1, f2):
            return False

        return True


def mult_sim_rerun(file_path, cadet_path, n_wdh):
    """Rerun simulations and keep best simulation time (for benchmarking).

    Parameters
    ----------
    file_path : String
        Path to files to be rerun.
    cadet_path : String
        Path to cadet executable.
    n_wdh : int
        Number of reruns per simulation
    """
    Cadet.cadet_path = cadet_path
    model = Cadet()
    files = os.listdir(file_path)

    for file in files:  # file = files[0]

        if (re.search('.h5', file)):

            model.filename = file_path + "/" + file
            success = model.run_load()
            if not success.returncode == 0:
                print(success)
                break

            best_sim_time = model.root.meta.time_sim

            for i in range(0, n_wdh):

                success = model.run_load()
                if not success.returncode == 0:
                    print(success)
                    break

                # only keep faster simulation time
                best_sim_time = min(best_sim_time, model.root.meta.time_sim)

            model.load()
            model.root.meta.time_sim = best_sim_time
            model.save()
            print(file + " rerun")

        else:
            print(file + " kein h5")


def delete_h5_files(directory, exclude_files=None):
    """
    Deletes all .h5 files in the specified directory, excluding files whose names (without extension) 
    are in the exclusion list.

    Parameters:
        directory (str): The path of the directory to clean up.
        exclude_files (list, optional): List of file names (without extensions) to exclude from deletion.
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    if not os.path.isdir(directory):
        print(f"Provided path {directory} is not a directory.")
        return

    if exclude_files is None:
        exclude_files = []

    # Convert exclude_files to a set for fast lookup
    exclude_set = set(exclude_files)
    
    deleted_files = 0
    for filename in os.listdir(directory):
        # Extract the base name (without extension)
        base_name, ext = os.path.splitext(filename)
        if ext == '.h5' and base_name not in exclude_set:
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                deleted_files += 1
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    if deleted_files == 0:
        print("No .h5 files deleted in the {directory} directory.")
    else:
        print(f"Deleted {deleted_files} .h5 file(s) in the {directory} directory.")


if __name__ == '__main__':

    test_eoc()
    test_calculate_DOFs()
    test_convergency_table()
    test_get_unique_DGsolution()
    test_map_z_to_xi()
    test_get_interpolated_DGsolution()
    test_Gauss_and_Lobatto_nodes()
