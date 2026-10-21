# -*- coding: utf-8 -*-
"""

This script impements evaluation functionalities related to numerical 
convergence analysis for CADET simulations.

"""

from cadet import Cadet
import numpy as np
from scipy.special import legendre
import math
import pandas as pd
from dataclasses import dataclass
import re
import os
import h5py
import shutil
import platform
from pathlib import Path

def get_cadet_path():
    install_path = None
    
    executable = 'cadet-cli'
    if install_path is None:
        try:
            if platform.system() == 'Windows':
                executable += '.exe'
            executable_path = Path(shutil.which(executable))
        except TypeError:
            raise FileNotFoundError(
                "CADET could not be found. Please set an install path"
            )
        install_path = executable_path.parent.parent
    
    install_path = Path(install_path)
    cadet_bin_path = install_path
    
    if cadet_bin_path.exists():
        print("CADET-Core found: ", cadet_bin_path)
        return cadet_bin_path
        
    else:
        raise FileNotFoundError(
            "CADET could not be found. Please check the path"
        )

_DG_models_ = [
    "GENERAL_RATE_MODEL",
    "LUMPED_RATE_MODEL_WITH_PORES",
    "LUMPED_RATE_MODEL_WITHOUT_PORES"
]
_FV_models_ = [
    "GENERAL_RATE_MODEL",
    "LUMPED_RATE_MODEL_WITH_PORES",
    "LUMPED_RATE_MODEL_WITHOUT_PORES",
    "GENERAL_RATE_MODEL_2D"
]
_methods_ = ["DGSEM", "collocation_DGSEM", "FV"]


@dataclass
class Discretization:

    axial_polynomial_degree: int
    particle_polynomial_degree: int
    radial_polynomial_degree: int
    axial_collocation_DG: bool
    particle_collocation_DG: bool
    radial_collocation_DG: bool
    axial_cells: int
    particle_cells: int
    radial_cells: int

    def __init__(self,
                 axial_polynomial_degree, axial_cells,
                 particle_polynomial_degree=None, particle_cells=None,
                 radial_polynomial_degree=None, radial_cells=None,
                 axial_collocation_DG=True,
                 particle_collocation_DG=False,
                 radial_collocation_DG=True):

        self.axial_polynomial_degree = axial_polynomial_degree
        self.particle_polynomial_degree = particle_polynomial_degree
        self.radial_polynomial_degree = radial_polynomial_degree
        self.axial_cells = axial_cells
        self.particle_cells = particle_cells
        self.radial_cells = radial_cells
        self.axial_collocation_DG = axial_collocation_DG
        self.particle_collocation_DG = particle_collocation_DG
        self.radial_collocation_DG = radial_collocation_DG

    @property
    def axial_method(self):
        if self.axial_polynomial_degree == 0:
            return 0
        elif self.axial_collocation_DG:
            return self.axial_polynomial_degree
        else:
            return -self.axial_polynomial_degree

    @axial_method.setter
    def axial_method(self, axial_method):
        self.axial_polynomial_degree = int(abs(axial_method))
        self.axial_collocation_DG = 1 if axial_method > 0 else 0

    @property
    def particle_method(self):
        if self.particle_polynomial_degree == 0:
            return 0
        elif self.particle_collocation_DG:
            return self.particle_polynomial_degree
        else:
            return -self.particle_polynomial_degree

    @particle_method.setter
    def particle_method(self, particle_method):
        self.particle_polynomial_degree = int(abs(particle_method))
        self.particle_collocation_DG = 1 if particle_method > 0 else 0

    @property
    def radial_method(self):
        if self.radial_polynomial_degree == 0:
            return 0
        elif self.radial_collocation_DG:
            return self.radial_polynomial_degree
        else:
            return -self.radial_polynomial_degree

    @radial_method.setter
    def radial_method(self, radial_method):
        self.radial_polynomial_degree = int(abs(radial_method))
        self.radial_collocation_DG = 1 if radial_method > 0 else 0


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
        sim.load_from_file()
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
                        try:
                            errors.append(
                                calculate_abs_error(
                                    get_solution(simulation, 'unit_'+unit, which, comp)[-1, :],
                                    reference)
                                )
                        except ValueError as e:
                            raise ValueError(f"Caught a ValueError: {e}\n You probably need to provide reference data info such as polyDeg, nCells, ref_coords!")

                else:
                    errors.append(calculate_abs_error(get_solution(simulation, 'unit_'+unit, which, comp), reference))
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


def build_gauss_evaluation_grid(length, n_cells, quad_points):
    """Build a per-cell Gauss-Legendre evaluation grid for bulk error integration.

    Parameters
    ----------
    length : float
        Length of the spatial 1D interval.
    n_cells : int
        Number of (DG or FV) cells.
    quad_points : int
        Number of Gauss-Legendre quadrature points per cell.

    Returns
    -------
    np.array
        Evaluation coordinates, shape (n_cells * quad_points,).
    np.array
        Gauss-Legendre quadrature weights, shape (quad_points,).
    float
        Mapping Jacobian (half the cell width).
    """
    xi_q, w_q = np.polynomial.legendre.leggauss(quad_points)
    h = length / n_cells
    eval_coords = np.empty(n_cells * quad_points)

    write_idx = 0
    for cell_idx in range(n_cells):
        z_left = cell_idx * h
        eval_coords[write_idx:write_idx + quad_points] = z_left + 0.5 * (xi_q + 1.0) * h
        write_idx += quad_points

    return eval_coords, w_q, 0.5 * h


def infer_bulk_n_cells(values, poly_deg, label="reference"):
    """Infer the number of DG/FV cells from a bulk solution's spatial extent.

    Parameters
    ----------
    values : np.array
        Bulk solution time slice, shape (n_points, ...).
    poly_deg : int
        Polynomial degree of the discretization.
    label : string
        Used in the error message if the shape is inconsistent.

    Returns
    -------
    int
        Number of cells.
    """
    n_nodes = poly_deg + 1
    if values.shape[0] % n_nodes != 0:
        raise ValueError(
            f"{label} bulk size {values.shape[0]} is not divisible by {n_nodes} "
            f"(poly_deg={poly_deg})."
        )
    return values.shape[0] // n_nodes


def infer_bulk_poly_deg(simulation, unit='001', default=None):
    """Infer the polynomial degree of a bulk solution's representation.

    Reads the spatial discretization of the given simulation (or reference),
    i.e. the polynomial degree for DG and 0 for FV. This makes a reference
    self-describing: an analytical reference may be stored as a high order DG
    representation (see src/analytical.py) while the simulations that are
    compared against it use any other method.

    Parameters
    ----------
    simulation : string or CADET object
        Simulation or reference to read the discretization from.
    unit : string
        Unit ID (000-999).
    default : int
        Returned if the discretization cannot be determined.

    Returns
    -------
    int or None
        Polynomial degree (0 for FV), or default.
    """
    try:
        disc = sim_go_to(
            get_simulation(simulation).root,
            ['input', 'model', 'unit_' + unit, 'discretization']
        )
    except (ValueError, OSError, AttributeError):
        return default

    method = None
    for key in ('spatial_method', 'SPATIAL_METHOD'):
        if key in disc:
            method = disc[key]
            break
    if method is None:
        return default

    method = np.asarray(method).ravel()[0]
    if isinstance(method, bytes):
        method = method.decode()
    method = str(method).upper()

    if method == 'FV':
        return 0
    if method == 'DG':
        for key in ('polydeg', 'POLYDEG'):
            if key in disc:
                return int(np.asarray(disc[key]).ravel()[0])
    return default


def interpolate_uniform_dg_values(values, eval_coords, length, poly_deg, n_cells, nodes, bary_weights):
    """Barycentrically interpolate a uniform-grid DG/FV bulk solution onto arbitrary coordinates.

    Parameters
    ----------
    values : np.array
        Bulk solution time slice, shape (n_cells * (poly_deg + 1), n_comp).
    eval_coords : np.array
        Coordinates to interpolate onto.
    length : float
        Length of the spatial 1D interval.
    poly_deg : int
        Polynomial degree of the discretization.
    n_cells : int
        Number of cells.
    nodes : np.array
        LGL nodes on [-1, 1] for poly_deg.
    bary_weights : np.array
        Barycentric weights for poly_deg.

    Returns
    -------
    np.array
        Interpolated values, shape (len(eval_coords), n_comp).
    """
    n_nodes = poly_deg + 1
    n_comp = values.shape[1]
    expected_size = n_cells * n_nodes
    if values.shape[0] != expected_size:
        raise ValueError(
            f"Expected values with first dimension {expected_size}, got {values.shape[0]}"
        )

    h = length / n_cells
    cell_idx = np.floor(eval_coords / h).astype(int)
    cell_idx = np.clip(cell_idx, 0, n_cells - 1)
    xi = 2.0 * (eval_coords - cell_idx * h) / h - 1.0

    diff = xi[:, None] - nodes[None, :]
    hit = np.abs(diff) < 1e-13

    safe_diff = np.where(hit, 1.0, diff)
    bary = bary_weights[None, :] / safe_diff
    alpha = bary / np.sum(bary, axis=1, keepdims=True)

    if np.any(hit):
        hit_rows = np.any(hit, axis=1)
        alpha[hit_rows, :] = 0.0
        alpha[hit_rows, np.argmax(hit[hit_rows], axis=1)] = 1.0

    nodal = values.reshape(n_cells, n_nodes, n_comp)
    selected = nodal[cell_idx, :, :]
    return np.einsum("en,enc->ec", alpha, selected, optimize=True)


def calculate_bulk_error_norms(sim_values, ref_values, length, n_cells_sim, n_cells_ref,
                               poly_deg, quad_points, nodes, bary_weights,
                               ref_poly_deg=None, ref_nodes=None, ref_bary_weights=None):
    """Compute Linf, L1, and L2 errors between two bulk DG/FV solution time slices.

    Both solutions are interpolated onto a shared Gauss-Legendre evaluation grid
    (built from the simulation's own cell layout) so that solutions living on
    different cell counts and/or polynomial degrees can be compared.

    Parameters
    ----------
    sim_values : np.array
        Simulation bulk solution time slice, shape (n_cells_sim*(poly_deg+1), n_comp).
    ref_values : np.array
        Reference bulk solution time slice, shape (n_cells_ref*(ref_poly_deg+1), n_comp).
    length : float
        Length of the spatial 1D interval.
    n_cells_sim : int
        Number of cells of the simulation.
    n_cells_ref : int
        Number of cells of the reference.
    poly_deg : int
        Polynomial degree of the simulation.
    quad_points : int
        Number of Gauss-Legendre quadrature points per cell.
    nodes : np.array
        LGL nodes for poly_deg.
    bary_weights : np.array
        Barycentric weights for poly_deg.
    ref_poly_deg : int
        Polynomial degree of the reference, defaults to poly_deg.
    ref_nodes : np.array
        LGL nodes for ref_poly_deg, defaults to nodes.
    ref_bary_weights : np.array
        Barycentric weights for ref_poly_deg, defaults to bary_weights.

    Returns
    -------
    np.array
        Linf error per component.
    np.array
        L1 error per component.
    np.array
        L2 error per component.
    """
    if ref_poly_deg is None:
        ref_poly_deg = poly_deg
        ref_nodes = nodes
        ref_bary_weights = bary_weights

    eval_coords, w_q, jac = build_gauss_evaluation_grid(length, n_cells_sim, quad_points)

    sim_eval = interpolate_uniform_dg_values(
        sim_values, eval_coords, length, poly_deg, n_cells_sim, nodes, bary_weights
    )
    ref_eval = interpolate_uniform_dg_values(
        ref_values, eval_coords, length, ref_poly_deg, n_cells_ref, ref_nodes, ref_bary_weights
    )

    abs_err = np.abs(sim_eval - ref_eval)
    n_comp = abs_err.shape[1]

    linf_errors = np.max(abs_err, axis=0)

    abs_err_reshaped = abs_err.reshape(n_cells_sim, quad_points, n_comp)
    l1_errors = jac * np.einsum("q,cqk->k", w_q, abs_err_reshaped, optimize=True)

    sq_err_reshaped = (abs_err ** 2).reshape(n_cells_sim, quad_points, n_comp)
    l2_errors = np.sqrt(jac * np.einsum("q,cqk->k", w_q, sq_err_reshaped, optimize=True))

    return linf_errors, l1_errors, l2_errors


def get_bulk_time_slice(simulation, unit='001', time_point=-1):
    """Get a bulk solution time slice and the associated column length.

    Parameters
    ----------
    simulation : string or CADET object
        Specifies the simulation of interest.
    unit : string
        Unit ID (000-999).
    time_point : int
        Index into the solution time axis.

    Returns
    -------
    np.array
        Bulk solution at time_point, shape (n_points, n_comp).
    float
        Column length.
    """
    sim = get_simulation(simulation)
    length = float(sim_go_to(sim.root, ['input', 'model', 'unit_' + unit, 'bed_length']))
    bulk = np.array(sim_go_to(sim.root, ['output', 'solution', 'unit_' + unit, 'solution_bulk']))
    return bulk[time_point, :, :], length

import numpy as np


def _get_fv_bulk_geometry_weighting(simulation, unit):
    """Return the geometry weighting of the FV bulk degrees of freedom.

    The FV state of the axial cylinder model consists of ordinary cell
    averages, whereas the radial cylinder shell and frustum models use
    geometry-weighted cell averages

        cbar_i = integral_{cell} A(x) c(x) dx / integral_{cell} A(x) dx

    with A ~ r (radial shell) or A ~ r^2 (frustum), where
    r(x) = r_inner + (x / length) * (r_outer - r_inner) along the normalized
    transport coordinate. Any grid transfer of FV solutions (e.g. restricting
    a fine reference onto a coarse grid) must use the same weighting;
    an unweighted restriction has an O(h^2) error that caps the observable
    EOC at two regardless of the scheme's actual order.

    Parameters
    ----------
    simulation : string or CADET object
        Simulation to read the unit geometry from.
    unit : string
        Unit ID (000-999).

    Returns
    -------
    tuple (int, float or None, float or None)
        (weight_power, r_inner, r_outer): A ~ r^weight_power;
        weight_power = 0 denotes unweighted averages.
    """
    sim = get_simulation(simulation)
    unit_dict = sim_go_to(sim.root, ['input', 'model', 'unit_' + unit])

    def _get(key, default=None):
        for k in (key.lower(), key.upper()):
            if k in unit_dict:
                return unit_dict[k]
        return default

    def _get_str(key, default=''):
        value = _get(key, default)
        value = np.asarray(value).ravel()[0] if np.size(value) else default
        if isinstance(value, bytes):
            value = value.decode()
        return str(value).upper()

    # GEOMETRY is the current interface's field for this; UNIT_TYPE substring matching is kept as a
    # fallback for simulation output generated with the old (pre geometry-unification) interface,
    # where the unit type itself was prefixed (e.g. RADIAL_GENERAL_RATE_MODEL).
    geometry = _get_str('geometry')
    unit_type = _get_str('unit_type')

    if geometry == 'RADIAL_FLOW_CYLINDER_SHELL' or 'RADIAL' in unit_type:
        weight_power = 1
    elif geometry == 'AXIAL_FLOW_FRUSTUM' or 'FRUSTUM' in unit_type:
        weight_power = 2
    else:
        return 0, None, None

    if geometry == 'RADIAL_FLOW_CYLINDER_SHELL':
        # r(xi=0) = inner radius, r(xi=1) = outer radius, derived from the mandatory
        # CROSS_SECTION_AREA_OUTER/CYLINDER_HEIGHT/BED_LENGTH (CROSS_SECTION_AREA_INNER is only an
        # optional double-check field and may not be present).
        height = float(np.asarray(_get('cylinder_height')).ravel()[0])
        r_outer = float(np.asarray(_get('cross_section_area_outer')).ravel()[0]) / (2.0 * np.pi * height)
        r_inner = r_outer - float(np.asarray(_get('bed_length')).ravel()[0])
    elif geometry == 'AXIAL_FLOW_FRUSTUM':
        # r(xi=0) = large-end radius, r(xi=1) = small-end radius (CADET-Core places the large end
        # at the start of the bed and the small end at x = bed_length).
        r_inner = np.sqrt(float(np.asarray(_get('cross_section_area_large_end')).ravel()[0]) / np.pi)
        r_outer = np.sqrt(float(np.asarray(_get('cross_section_area_small_end')).ravel()[0]) / np.pi)
    else:
        # Legacy (pre geometry-unification) interface.
        r_inner = float(np.asarray(_get('col_radius_inner')).ravel()[0])
        r_outer = float(np.asarray(_get('col_radius_outer')).ravel()[0])
    return weight_power, r_inner, r_outer


def _get_fv_grid_faces(simulation, unit, n_cells):
    """Return the physical FV cell face coordinates of a simulation, or None.

    Reads input/model/unit_XXX/discretization/GRID_FACES (present for
    non-equidistant grids); returns None for equidistant grids (no
    GRID_FACES field), in which case callers fall back to a uniform grid.

    Parameters
    ----------
    simulation : string or CADET object
        Simulation to read the grid from.
    unit : string
        Unit ID (000-999).
    n_cells : int
        Expected number of cells (validates the face count).

    Returns
    -------
    np.ndarray or None
        Face coordinates, shape (n_cells + 1,), or None.
    """
    sim = get_simulation(simulation)
    disc = sim_go_to(sim.root, ['input', 'model', 'unit_' + unit, 'discretization'])

    faces = None
    for key in ('grid_faces', 'GRID_FACES'):
        if key in disc:
            faces = np.asarray(disc[key], dtype=float).ravel()
            break
    if faces is None:
        return None
    if faces.size != n_cells + 1:
        raise ValueError(
            f"GRID_FACES of unit {unit} has {faces.size} entries, expected {n_cells + 1}"
        )
    return faces


def _fv_geometry_weight_integral(xi_a, xi_b, weight_power, r_inner, delta_r):
    """Integral of the geometry weight r(xi)^p over [xi_a, xi_b] (normalized coords)."""
    if weight_power == 0 or r_inner is None:
        return xi_b - xi_a
    if weight_power == 1:
        return r_inner * (xi_b - xi_a) + 0.5 * delta_r * (xi_b**2 - xi_a**2)
    # weight_power == 2
    return ((r_inner + delta_r * xi_b)**3 - (r_inner + delta_r * xi_a)**3) / (3.0 * delta_r)


def _fv_reference_cell_averages(ref_values, ref_poly_deg,
                                length, n_cells_ref,
                                sim_n_cells,
                                weight_power=0, r_inner=None, r_outer=None,
                                sim_faces=None, ref_faces=None):
    """Return reference cell averages on the FV simulation grid.

    Parameters
    ----------
    ref_values : ndarray
        Reference solution, flattened over cells/nodes:
        (n_cells_ref * (ref_poly_deg + 1), n_comp) for DG,
        (n_cells_ref, n_comp) for FV.
    ref_poly_deg : int
        Reference polynomial degree. 0 means FV.
    length : float
        Domain length.
    n_cells_ref : int
        Number of reference cells.
    sim_n_cells : int
        Number of FV simulation cells.
    weight_power : int
        Geometry weighting of the FV cell averages: A(r) ~ r^weight_power
        (0: axial/unweighted, 1: radial cylinder shell, 2: frustum), see
        _get_fv_bulk_geometry_weighting.
    r_inner : float
        Inner radius, required for weight_power > 0.
    r_outer : float
        Outer radius, required for weight_power > 0.
    sim_faces, ref_faces : ndarray, optional
        Physical cell face coordinates of the simulation/reference grids
        (shape (n_cells + 1,)); defaults to uniform grids. Required for
        non-equidistant (e.g. equivolume) grids, where the uniform
        assumption would introduce an O(h^2) restriction error that caps
        the observable EOC at two.

    Returns
    -------
    ndarray
        Reference cell averages on the FV simulation grid,
        shape (sim_n_cells, n_comp).
    """
    n_comp = ref_values.shape[1]
    delta_r = (r_outer - r_inner) if (weight_power and r_inner is not None) else None

    # ------------------------------------------------------------------
    # Reference is FV:
    #
    # The FV values are (geometry-weighted) cell averages. To obtain the
    # reference average over a simulation cell, integrate the piecewise-
    # constant reference solution over the simulation cell using exact
    # cell overlap, weighted by the same geometry measure A(r) ~ r^p that
    # defines the FV degrees of freedom.
    # ------------------------------------------------------------------
    if ref_poly_deg == 0:
        # Physical face coordinates; when only one grid provides faces, the
        # other one is built uniformly on the same domain.
        if sim_faces is None and ref_faces is None:
            domain_lo, domain_hi = 0.0, length
        elif sim_faces is not None:
            domain_lo, domain_hi = float(sim_faces[0]), float(sim_faces[-1])
        else:
            domain_lo, domain_hi = float(ref_faces[0]), float(ref_faces[-1])
        span = domain_hi - domain_lo

        if sim_faces is None:
            sim_f = domain_lo + np.arange(sim_n_cells + 1) * (span / sim_n_cells)
        else:
            sim_f = np.asarray(sim_faces, dtype=float)
        if ref_faces is None:
            ref_f = domain_lo + np.arange(n_cells_ref + 1) * (span / n_cells_ref)
        else:
            ref_f = np.asarray(ref_faces, dtype=float)

        ref_left = ref_f[:-1]
        ref_right = ref_f[1:]
        sim_left = sim_f[:-1]
        sim_right = sim_f[1:]

        result = np.zeros((sim_n_cells, n_comp))

        ref_idx = 0

        for i in range(sim_n_cells):
            x_left = sim_left[i]
            x_right = sim_right[i]

            # Advance to the first potentially overlapping reference cell.
            while ref_idx < n_cells_ref - 1 and ref_right[ref_idx] <= x_left:
                ref_idx += 1

            weight_total = 0.0

            j = ref_idx
            while j < n_cells_ref and ref_left[j] < x_right:
                overlap_left = max(x_left, ref_left[j])
                overlap_right = min(x_right, ref_right[j])

                if overlap_right > overlap_left:
                    weight = _fv_geometry_weight_integral(
                        (overlap_left - domain_lo) / span,
                        (overlap_right - domain_lo) / span,
                        weight_power, r_inner, delta_r
                    )
                    result[i, :] += weight * ref_values[j, :]
                    weight_total += weight

                j += 1

            result[i, :] /= weight_total

        return result

    # ------------------------------------------------------------------
    # Reference is DG:
    #
    # Integrate the DG polynomial over every FV simulation cell, using the
    # geometry weight A(r) ~ r^p of the FV degrees of freedom.
    # ------------------------------------------------------------------
    ref_nodes, _ = LGL_NodesWeights(ref_poly_deg)
    ref_bary_weights = barycentric_weights(ref_poly_deg)

    n_nodes_ref = ref_poly_deg + 1

    ref_values = ref_values.reshape(
        n_cells_ref, n_nodes_ref, n_comp
    )

    dx_ref = length / n_cells_ref
    dx_sim = length / sim_n_cells

    # Gauss-Legendre quadrature for cell averaging.
    # A sufficiently high order avoids contaminating the error estimate.
    n_quad = max(2 * ref_poly_deg + 3, 16)
    xi_quad, w_quad = np.polynomial.legendre.leggauss(n_quad)

    result = np.zeros((sim_n_cells, n_comp))

    for i in range(sim_n_cells):
        x_left = i * dx_sim
        x_right = x_left + dx_sim

        # Reference cells that overlap this FV cell.
        j_start = max(0, int(np.floor(x_left / dx_ref)))
        j_end = min(n_cells_ref - 1, int(np.floor(
            np.nextafter(x_right, x_left) / dx_ref
        )))

        integral = np.zeros(n_comp)
        weight_total = 0.0

        for j in range(j_start, j_end + 1):
            ref_x_left = j * dx_ref
            ref_x_right = ref_x_left + dx_ref

            overlap_left = max(x_left, ref_x_left)
            overlap_right = min(x_right, ref_x_right)

            if overlap_right <= overlap_left:
                continue

            # Map quadrature points from the overlap interval to
            # physical coordinates.
            x_quad = (
                0.5 * (overlap_left + overlap_right)
                + 0.5 * (overlap_right - overlap_left) * xi_quad
            )

            # Geometry weight at the quadrature points.
            if weight_power and r_inner is not None:
                w_geom = (r_inner + delta_r * (x_quad / length)) ** weight_power
            else:
                w_geom = np.ones_like(x_quad)

            # Map physical coordinates to the reference cell's
            # reference coordinate [-1, 1].
            xi_ref = (
                2.0 * (x_quad - ref_x_left) / dx_ref - 1.0
            )

            # Barycentric interpolation of the DG polynomial.
            values_at_quad = np.empty(
                (n_quad, n_comp)
            )

            for q, xi in enumerate(xi_ref):
                diff = xi - ref_nodes

                # Exact-node case.
                exact = np.abs(diff) < 1e-14

                if np.any(exact):
                    values_at_quad[q, :] = (
                        ref_values[j, np.argmax(exact), :]
                    )
                else:
                    weights = ref_bary_weights / diff
                    weights /= np.sum(weights)

                    values_at_quad[q, :] = np.sum(
                        weights[:, None] * ref_values[j, :, :],
                        axis=0
                    )

            integral += (
                0.5 * (overlap_right - overlap_left)
                * np.sum(
                    (w_quad * w_geom)[:, None] * values_at_quad,
                    axis=0
                )
            )

            weight_total += (
                0.5 * (overlap_right - overlap_left)
                * np.sum(w_quad * w_geom)
            )

        # Convert the weighted integral into the geometry-weighted
        # average over the FV cell.
        result[i, :] = integral / weight_total

    return result


def _calculate_fv_average_error_norms(
        sim_values, ref_values, length,
        n_cells_sim, n_cells_ref, ref_poly_deg,
        weight_power=0, r_inner=None, r_outer=None,
        sim_faces=None, ref_faces=None):
    """Calculate error norms between FV cell averages and a reference."""

    # FV solution consists of one value per cell.
    sim_averages = sim_values.reshape(n_cells_sim, -1)

    ref_averages = _fv_reference_cell_averages(
        ref_values=ref_values,
        ref_poly_deg=ref_poly_deg,
        length=length,
        n_cells_ref=n_cells_ref,
        sim_n_cells=n_cells_sim,
        weight_power=weight_power,
        r_inner=r_inner,
        r_outer=r_outer,
        sim_faces=sim_faces,
        ref_faces=ref_faces,
    )

    errors = sim_averages - ref_averages

    if sim_faces is not None:
        dx = np.diff(np.asarray(sim_faces, dtype=float))[:, None]
    else:
        dx = length / n_cells_sim

    linf = np.max(np.abs(errors), axis=0)
    l1 = np.sum(np.abs(errors) * dx, axis=0)
    l2 = np.sqrt(np.sum(errors**2 * dx, axis=0))

    return linf, l1, l2


def calculate_bulk_convergence_errors(simulation_names, reference, unit,
                                      ax_method, ref_method, time_point,
                                      quad_points=None):
    """Compute Linf/L1/L2 errors, min. values, cell counts and sim. times.

    DG solutions are compared using the existing quadrature/interpolation
    procedure. FV solutions (polydeg=0) are compared using cell-average
    errors: the FV solution is treated as a set of cell averages, while the
    reference is averaged over each FV simulation cell.

    Parameters
    ----------
    simulation_names : list
        Simulations (Cadet objects or h5 file names), ordered by discretization.
    reference : string or CADET object
        Reference simulation, which may be DG or FV.
    unit : string
        Unit ID (000-999).
    ax_method : int
        Polynomial degree (DG) or 0 (FV) of the simulations.
    ref_method : int
        Polynomial degree (DG) or 0 (FV) of the reference.
    time_point : int
        Index into the solution time axis at which errors are compared.
    quad_points : int
        Number of Gauss-Legendre quadrature points per cell for the DG
        error calculation.

    Returns
    -------
    np.array
        Number of cells per discretization.
    np.array
        Linf errors.
    np.array
        L1 errors.
    np.array
        L2 errors.
    np.array
        Min. values per component.
    np.array
        Simulation compute times.
    """
    poly_deg = abs(ax_method)
    ref_poly_deg = abs(ref_method)

    if quad_points is None:
        quad_points = max(2 * poly_deg + 3, 16)

    # ------------------------------------------------------------------
    # DG machinery. Only needed when the simulation is DG.
    # ------------------------------------------------------------------
    if poly_deg > 0:
        nodes, _ = LGL_NodesWeights(poly_deg)
        bary_weights = barycentric_weights(poly_deg)

        if ref_poly_deg != poly_deg:
            ref_nodes, _ = LGL_NodesWeights(ref_poly_deg)
            ref_bary_weights = barycentric_weights(ref_poly_deg)
        else:
            ref_nodes, ref_bary_weights = nodes, bary_weights
    else:
        nodes = bary_weights = None
        ref_nodes = ref_bary_weights = None

    ref_values, _ = get_bulk_time_slice(reference, unit, time_point)

    n_cells_ref = infer_bulk_n_cells(
        ref_values, ref_poly_deg, label="reference"
    )

    # Geometry weighting of the FV degrees of freedom (radial shell/frustum);
    # required for consistent restriction of the reference onto coarser grids.
    if poly_deg == 0:
        weight_power, r_inner, r_outer = _get_fv_bulk_geometry_weighting(
            reference, unit
        )
    else:
        weight_power, r_inner, r_outer = 0, None, None

    n_disc = len(simulation_names)
    n_comp = ref_values.shape[1]

    n_cells_per_disc = np.zeros(n_disc, dtype=int)
    linf_errors = np.zeros((n_disc, n_comp))
    l1_errors = np.zeros((n_disc, n_comp))
    l2_errors = np.zeros((n_disc, n_comp))
    min_values = np.zeros((n_disc, n_comp))
    sim_times = np.zeros(n_disc)

    for disc_idx, simulation in enumerate(simulation_names):

        sim_values, length = get_bulk_time_slice(
            simulation, unit, time_point
        )

        n_cells_sim = infer_bulk_n_cells(
            sim_values, poly_deg, label="simulation"
        )

        n_cells_per_disc[disc_idx] = n_cells_sim

        # ==============================================================
        # FV simulation: compare cell averages.
        # ==============================================================
        if poly_deg == 0:

            linf, l1, l2 = _calculate_fv_average_error_norms(
                sim_values=sim_values,
                ref_values=ref_values,
                length=length,
                n_cells_sim=n_cells_sim,
                n_cells_ref=n_cells_ref,
                ref_poly_deg=ref_poly_deg,
                weight_power=weight_power,
                r_inner=r_inner,
                r_outer=r_outer,
            )

        # ==============================================================
        # DG simulation: retain the existing pointwise/quadrature
        # comparison.
        # ==============================================================
        else:

            linf, l1, l2 = calculate_bulk_error_norms(
                sim_values, ref_values, length,
                n_cells_sim, n_cells_ref,
                poly_deg, quad_points,
                nodes, bary_weights,
                ref_poly_deg=ref_poly_deg,
                ref_nodes=ref_nodes,
                ref_bary_weights=ref_bary_weights
            )

        linf_errors[disc_idx, :] = linf
        l1_errors[disc_idx, :] = l1
        l2_errors[disc_idx, :] = l2

        min_values[disc_idx, :] = np.min(
            sim_values, axis=0
        )

        sim_times[disc_idx] = get_compute_time(simulation)

    return (
        n_cells_per_disc,
        linf_errors,
        l1_errors,
        l2_errors,
        min_values,
        sim_times,
    )


def interpolate_dg_at_coordinate(values, coordinate, length, poly_deg, n_cells,
                                 nodes, bary_weights,
                                 interface_mode="average", node_tolerance=1e-13):
    """Interpolate a uniform-grid DG solution at one physical spatial coordinate.

    Parameters
    ----------
    values : np.array
        DG solution, shape (n_cells*(poly_deg+1), n_comp) for a single time
        slice or (n_times, n_cells*(poly_deg+1), n_comp) for a full time series.
    coordinate : float
        Physical axial coordinate in [0, length].
    length : float
        Length of the spatial 1D interval.
    poly_deg : int
        DG polynomial degree.
    n_cells : int
        Number of cells.
    nodes : np.array
        LGL nodes on [-1, 1] for poly_deg.
    bary_weights : np.array
        Barycentric weights for poly_deg.
    interface_mode : string
        Handling of coordinates that hit an element interface, where the DG
        solution is double-valued: 'left', 'right', or 'average'.
    node_tolerance : float
        Absolute tolerance for detecting element-interface and DG-node hits.

    Returns
    -------
    np.array
        Interpolated values, shape (n_comp,) for 2D input and
        (n_times, n_comp) for 3D input.
    """
    if interface_mode not in ("left", "right", "average"):
        raise ValueError(
            f"Invalid interface_mode='{interface_mode}'. "
            "Choose one of ['average', 'left', 'right']."
        )

    values = np.asarray(values)
    if values.ndim not in (2, 3):
        raise ValueError(
            f"Expected values to be 2D or 3D, got shape {values.shape}."
        )

    n_nodes = poly_deg + 1
    spatial_axis = values.ndim - 2
    if values.shape[spatial_axis] != n_cells * n_nodes:
        raise ValueError(
            f"Expected {n_cells * n_nodes} spatial DG values, "
            f"got {values.shape[spatial_axis]}."
        )

    # nodal shape: ([n_times,] n_cells, n_nodes, n_comp)
    nodal = values.reshape(values.shape[:spatial_axis] + (n_cells, n_nodes, values.shape[-1]))

    h = length / n_cells
    normalized_element_coordinate = coordinate / h

    # element interfaces (incl. domain boundaries), where the DG solution is
    # double-valued, are treated according to interface_mode
    nearest_interface = int(np.round(normalized_element_coordinate))
    if abs(normalized_element_coordinate - nearest_interface) <= node_tolerance:
        interface_idx = max(0, min(n_cells, nearest_interface))

        if interface_idx == 0:  # left domain boundary, single-valued
            return nodal[..., 0, 0, :]
        if interface_idx == n_cells:  # right domain boundary, single-valued
            return nodal[..., n_cells - 1, -1, :]

        left_value = nodal[..., interface_idx - 1, -1, :]
        right_value = nodal[..., interface_idx, 0, :]
        if interface_mode == "left":
            return left_value
        elif interface_mode == "right":
            return right_value
        return 0.5 * (left_value + right_value)

    cell_idx = int(np.floor(normalized_element_coordinate))
    cell_idx = max(0, min(n_cells - 1, cell_idx))
    xi = 2.0 * (coordinate - cell_idx * h) / h - 1.0

    # exact DG-node hits would yield 0/0 in the barycentric formula
    diff = xi - nodes
    hit = np.abs(diff) <= node_tolerance
    if np.any(hit):
        return nodal[..., cell_idx, int(np.flatnonzero(hit)[0]), :]

    bary = bary_weights / diff
    alpha = bary / np.sum(bary)

    return np.einsum("n,...nc->...c", alpha, nodal[..., cell_idx, :, :], optimize=True)


def calculate_bulk_slice_convergence_errors(simulation_names, reference, unit,
                                            ax_method, ref_method, normed_coord,
                                            interface_mode="average",
                                            node_tolerance=1e-13):
    """Compute temporal Linf/L1/L2 errors at a fixed axial coordinate for an EOC sweep.

    For each simulation, the bulk solution is interpolated at the normalized
    axial coordinate normed_coord for every solution time, yielding an
    outlet-like time series c(z*, t) that is compared against the reference
    interpolated at the same coordinate (normed_coord=1.0 is equivalent to the
    outlet solution). The temporal norms are Linf = max_t |e|,
    L1 = integral |e| dt and L2 = sqrt(integral e^2 dt) (trapezoidal rule),
    matching scripts/bulkSliceEOC.py's reference implementation.

    Parameters
    ----------
    simulation_names : list
        Simulations (Cadet objects or h5 file names), ordered by discretization.
    reference : string or CADET object
        Reference simulation, may share or differ from ax_method's polynomial degree.
    unit : string
        Unit ID (000-999).
    ax_method : int
        Polynomial degree (DG) of the simulations.
    ref_method : int
        Polynomial degree (DG) of the reference.
    normed_coord : float
        Normalized axial evaluation coordinate z/L in [0, 1].
    interface_mode : string
        Handling of coordinates that hit an element interface: 'left', 'right',
        or 'average'.
    node_tolerance : float
        Absolute tolerance for detecting element-interface and DG-node hits.

    Returns
    -------
    np.array
        Number of cells per discretization, shape (nDisc,).
    np.array
        Linf errors, shape (nDisc, nComp).
    np.array
        L1 errors, shape (nDisc, nComp).
    np.array
        L2 errors, shape (nDisc, nComp).
    np.array
        Min. values per component of the interpolated time series, shape (nDisc, nComp).
    np.array
        Simulation compute times, shape (nDisc,).
    """
    if not 0.0 <= normed_coord <= 1.0:
        raise ValueError(
            f"normed_coord={normed_coord} is outside [0, 1]."
        )

    poly_deg = abs(ax_method)
    ref_poly_deg = abs(ref_method)
    if poly_deg == 0 or ref_poly_deg == 0:
        raise ValueError(
            "calculate_bulk_slice_convergence_errors only supports DG discretizations."
        )

    nodes, _ = LGL_NodesWeights(poly_deg)
    bary_weights = barycentric_weights(poly_deg)
    if ref_poly_deg != poly_deg:
        ref_nodes, _ = LGL_NodesWeights(ref_poly_deg)
        ref_bary_weights = barycentric_weights(ref_poly_deg)
    else:
        ref_nodes, ref_bary_weights = nodes, bary_weights

    ref_sim = get_simulation(reference)
    length = float(sim_go_to(ref_sim.root, ['input', 'model', 'unit_' + unit, 'bed_length']))
    ref_bulk = np.array(sim_go_to(ref_sim.root, ['output', 'solution', 'unit_' + unit, 'solution_bulk']))
    ref_times = np.asarray(get_solution_times(ref_sim), dtype=float)
    n_cells_ref = infer_bulk_n_cells(ref_bulk[0, :, :], ref_poly_deg, label="reference")

    coordinate = normed_coord * length
    ref_eval = interpolate_dg_at_coordinate(
        ref_bulk, coordinate, length, ref_poly_deg, n_cells_ref,
        ref_nodes, ref_bary_weights,
        interface_mode=interface_mode, node_tolerance=node_tolerance
    )

    n_disc = len(simulation_names)
    n_comp = ref_bulk.shape[2]

    n_cells_per_disc = np.zeros(n_disc, dtype=int)
    linf_errors = np.zeros((n_disc, n_comp))
    l1_errors = np.zeros((n_disc, n_comp))
    l2_errors = np.zeros((n_disc, n_comp))
    min_values = np.zeros((n_disc, n_comp))
    sim_times = np.zeros(n_disc)

    for disc_idx, simulation in enumerate(simulation_names):
        sim = get_simulation(simulation)
        sim_bulk = np.array(sim_go_to(sim.root, ['output', 'solution', 'unit_' + unit, 'solution_bulk']))
        sim_solution_times = np.asarray(get_solution_times(sim), dtype=float)
        if sim_solution_times.shape != ref_times.shape or not np.allclose(
                sim_solution_times, ref_times, rtol=0.0, atol=node_tolerance):
            raise ValueError(
                "Simulation and reference time vectors are not identical. "
                "The bulk slice temporal EOC analysis requires the same time "
                "points for all simulations and the reference."
            )

        n_cells_sim = infer_bulk_n_cells(sim_bulk[0, :, :], poly_deg, label="simulation")
        n_cells_per_disc[disc_idx] = n_cells_sim

        sim_eval = interpolate_dg_at_coordinate(
            sim_bulk, coordinate, length, poly_deg, n_cells_sim,
            nodes, bary_weights,
            interface_mode=interface_mode, node_tolerance=node_tolerance
        )

        abs_err = np.abs(sim_eval - ref_eval)
        linf_errors[disc_idx, :] = np.max(abs_err, axis=0)
        l1_errors[disc_idx, :] = np.trapezoid(abs_err, x=ref_times, axis=0)
        l2_errors[disc_idx, :] = np.sqrt(np.trapezoid(abs_err ** 2, x=ref_times, axis=0))
        min_values[disc_idx, :] = np.min(sim_eval, axis=0)
        sim_times[disc_idx] = get_compute_time(simulation)

    return n_cells_per_disc, linf_errors, l1_errors, l2_errors, min_values, sim_times


def get_interpolated_solution(orig_values, orig_coords, domain_end, output_coords, polyDeg, nCells):
    """Calculate weighted error of solution.

    Parameters
    ----------
    orig_values : np.array
        Solution to be interpolated, sorted along orig_coords.
    orig_coords : np.array
        Coordinate vector of solution in ascending order.
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
    if np.any(output_coords - domain_end > 1e-17) or np.any(output_coords < 0.0):
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
        coordHit = False
        
        for outputIdx in range(len(output_coords)):
            
            while orig_coords[cellIdx] + 0.5 * deltaZ < output_coords[outputIdx]:
                
                if abs(orig_coords[cellIdx] + 0.5 * deltaZ - output_coords[outputIdx]) < 1E-15:
                    coordHit = True # if the coordinate is hit exactly, we take the average value, except for the right boundary
                    break
                else:
                    cellIdx += 1 # increase the index until the output coord is within the current FV cell

            if coordHit and not cellIdx == len(orig_coords) - 1: # if the coordinate is hit exactly, we take the average value, except for the right boundary
                output_values[outputIdx] = 0.5 * (orig_values[cellIdx] + orig_values[cellIdx + 1])
                
            else:
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


def get_case_insensitive(d, key):
    key = key.lower()
    for k, v in d.items():
        if k.lower() == key:
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            return v
    return None

def generate_bulkDisc_name(disc):
    
    name = get_case_insensitive(disc, 'SPATIAL_METHOD')
    
    suffix = ""
    
    if get_case_insensitive(disc, 'GRID_FACES') is not None:
        
        grid = np.array(get_case_insensitive(disc, 'GRID_FACES'))
        
        is_equidistant = np.allclose(np.diff(grid), np.diff(grid)[0])
        
        suffix = "" if  is_equidistant else "nonEq"
    elif get_case_insensitive(disc, 'AXIAL_GRID_FACES') is not None:
        
        grid = np.array(get_case_insensitive(disc, 'AXIAL_GRID_FACES'))
        
        is_equidistant = np.allclose(np.diff(grid), np.diff(grid)[0])
        
        suffix = "" if  is_equidistant else "nonEq"

    if name == "FV":
        
        if get_case_insensitive(disc, 'RECONSTRUCTION') == "WENO":
            return name + "WENO" + str(get_case_insensitive(disc['weno'], 'WENO_ORDER')) + suffix
        else:
            if get_case_insensitive(disc, 'RECONSTRUCTION') is None:
                 return name + suffix
            return name + get_case_insensitive(disc, 'RECONSTRUCTION') + suffix
        
    elif name == "DG":
          
        # todo once implemented: non-eq
        
        # USE_COLLOCATION_DG = 1 selects collocation DG, 0 selects exact integration DG.
        # If the field is absent, CADET-Core defaults to collocation DG on axial cylinders.
        collocation = get_case_insensitive(disc, "USE_COLLOCATION_DG")
        exInt = False if collocation is None else not collocation
        name = "exIntDG" if exInt else "DG" # "cDG"
        
        polyDeg = get_case_insensitive(disc, "POLYDEG")
        if polyDeg is None:
            polyDeg = get_case_insensitive(disc, "AX_POLYDEG")
        if polyDeg is None:
            return name
        return name + "_P" + str(polyDeg)
    else:
        return name


def generate_parDisc_name(disc, bulkDiscName=None):
    
    name = "par" + get_case_insensitive(disc, 'SPATIAL_METHOD')
    
    suffix = ""
    
    if get_case_insensitive(disc, 'PAR_DISC_TYPE') is not None:
        suffix = "" if get_case_insensitive(disc, 'PAR_DISC_TYPE') == "EQUIDISTANT_PAR" else "nonEq"
    
    if name == "parFV":
        if suffix == "" and re.search("FV", bulkDiscName):
            return ""
        return name + suffix
        
    elif name == "parDG":
        polyDeg = get_case_insensitive(disc, "PAR_POLYDEG")
        if polyDeg is None:
            return "" if suffix == "" and re.search("DG", bulkDiscName) and re.search(f"P{polyDeg}", bulkDiscName) else name + suffix
        return "parP" + str(polyDeg) + suffix
    else:
        return name + suffix

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

    elif int(axP) > 0 and int(parP) == 0:
        return prefix + "_cDG_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + "_FVparZ" + str(int(parCells)) + suffix

    elif int(axP) < 0 and int(parP) == 0:
        return prefix + "_DGexInt_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + "_FVparZ" + str(int(parCells)) + suffix

    elif int(axP) == 0 and int(parP) > 0:
        if int(parCells) == 1 and parGSM:
            return prefix + "_FV_Z" + str(int(axCells)) + "_GSM_parP" + str(int(abs(parP))) + "parZ" + str(int(parCells)) + suffix
        else:
            return prefix + "_FV_Z" + str(int(axCells)) + "_DG_parP" + str(int(abs(parP))) + "parZ" + str(int(parCells)) + suffix

    else: # parP < 0 not supported
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
    full_DOFs : Bool
        Determines whether all DOFs should be considered, i.e. including flux and particle states,
    model : string
        Model such as LRM, GRM, LRMP, MCT, DPFR, COL1D, ...
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

    if model is None:
        nBound = 0
    if nBound == None:
        nBound = 0 if re.search("DPFR", model) else nComp

    bulk_dof = 0
    par_dof = 0
    flux_dof = 0

    if model is None or not (re.search("2D", model) or re.search("MCT", model)):
        
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
    
        elif discretization.ndim in [0, 1]:  # LRM or LRMP or DPFR
            bulk_dof = (abs(method) + 1) * discretization * nComp
            if full_DOFs:
                if model.upper() == "LRMP":
                    par_dof = (nComp + nBound) * abs(method+1) * discretization
                    if method == 0:  # add flux states for FV discretization
                        flux_dof = bulk_dof
                elif model.upper() in ("LRM", "COL1D", "DPFR"):
                    par_dof = nBound * abs(method+1) * discretization
                else:
                    raise ValueError(
                        "Unknown transport model \"" + model +
                        "\" or wrong dimensionality of input."
                    )
    
        return inlet_dof + bulk_dof + flux_dof + par_dof
    
    else:
        
        if re.search("MCT", model):
            return (abs(method) + 1) * discretization * nComp
        
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
                if re.search("LRMP", model):
                    par_dof = (nComp + nBound) * bulk_dof
                elif re.search("LRM", model):
                    par_dof = (nComp + nBound) * bulk_dof
                elif re.search("GRM", model):
                    par_dof = (nComp + nBound) * (method[2] + 1) * discretization[2, :] * bulk_dof
                else:
                    raise ValueError(
                        "Unknown transport model \"" + model +
                        "\" or wrong dimensionality of input."
                    )
            
        return inlet_dof + bulk_dof + par_dof + flux_dof


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
    abs_errors = np.array(abs_errors)
    header = []
    header.append("$N_e^z$")
    table = []
    # Infer dimensionality of table
    if disc.ndim == 2:
        if len(disc) == 3 and (transport_model is not None and re.search("2D", transport_model)): # 2DGRM
            nDisc = disc.shape[1]
            header.append("$N_e^r$")
            header.append("$N_e^p$")
            table.append(disc[0])
            table.append(disc[1])
            table.append(disc[2])
        elif len(disc) == 2 and (transport_model is not None and re.search("2D", transport_model)): # 2D LRM or LRMP
            nDisc = disc.shape[1]
            header.append("$N_e^r$")
            table.append(disc[0])
            table.append(disc[1])
        elif len(disc) == 2: # GRM
            nDisc = disc.shape[1]
            header.append("$N_e^p$")
            table.append(disc[0])
            table.append(disc[1])
    elif (disc.ndim == 1):   # LRMP, LRM, MCT
        nDisc = len(disc)
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


def recalculate_results(file_path, model,
                        ax_method, ax_cells,
                        exact_name,
                        unit='001', which='outlet',
                        par_method=None, par_cells=None,
                        rad_method=None, rad_cells=None,
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

    extra_keys = {}        

    # needed for DOF calculation
    if transport_model is None:
        try:
            transport_model = re.search(
                '2DLRM(?!P)|2DLRMP|2DGRM|LRM(?!P)|LRMP|GRM|COL1D|COL2D|MCT|DPFR',
                model,
                re.IGNORECASE).group(0)
        except:
            raise ValueError(
                "Considered transport model neither explicitly specified \
                nor given in model name: " + model)
    if ncomp is None:
        try:
            ncomp = int(re.search(r'(_)(\d+)(comp)',
                                  model,
                                  re.IGNORECASE).group(2))
        except:
            raise ValueError(
                "Number of components neither explicitly specified \
                nor given in model name: " + model)
    if nbound is None:
        nbound = ncomp

    if which == 'bulk':
        if simulation_names is None:
            simulation_names = generate_simulation_names(
                prefix=file_path + model,
                ax_methods=[ax_method], ax_cells=ax_cells
            )

        reference = file_path + exact_name if isinstance(exact_name, str) else exact_name

        poly_deg = abs(ax_method)
        # The reference is self-describing: read the polynomial degree of its
        # own representation (0 for FV), so that analytical references, which
        # are stored as high order DG representations, can be compared against
        # simulations of any method. Falls back to the simulation's degree,
        # which is the right choice for a self-convergence reference.
        ref_method = kwargs.get('ref_method', None)
        if ref_method is None:
            ref_method = infer_bulk_poly_deg(
                reference, unit=unit, default=ax_method
            )

        # Exactly one of time_point and normed_coord selects the analysis:
        # time_point -> spatial error norms at that solution time index;
        # normed_coord -> temporal error norms at that normalized axial coordinate.
        time_point = kwargs.get('time_point', None)
        normed_coord = kwargs.get('normed_coord', None)
        if (time_point is None) == (normed_coord is None):
            raise ValueError(
                "which='bulk' requires exactly one of 'time_point' (spatial "
                "error norms at that solution time index) or 'normed_coord' "
                "(temporal error norms at that normalized axial coordinate "
                "z/L in [0, 1]), got "
                f"time_point={time_point}, normed_coord={normed_coord}."
            )

        if time_point is not None:
            n_cells_per_disc, linf_errors, l1_errors, l2_errors, min_values, sim_times = (
                calculate_bulk_convergence_errors(
                    simulation_names, reference, unit,
                    ax_method, ref_method, time_point,
                    quad_points=kwargs.get('quad_points', None)
                )
            )
        else:
            n_cells_per_disc, linf_errors, l1_errors, l2_errors, min_values, sim_times = (
                calculate_bulk_slice_convergence_errors(
                    simulation_names, reference, unit,
                    ax_method, ref_method,
                    normed_coord=normed_coord,
                    interface_mode=kwargs.get('interface_mode', 'average'),
                    node_tolerance=kwargs.get('node_tolerance', 1e-13)
                )
            )

        n_comp = linf_errors.shape[1]
        tables = {}
        for comp_idx in range(n_comp):
            linf_eoc = np.insert(calculate_eoc(n_cells_per_disc, linf_errors[:, comp_idx]), 0, 0.0)
            l1_eoc = np.insert(calculate_eoc(n_cells_per_disc, l1_errors[:, comp_idx]), 0, 0.0)
            l2_eoc = np.insert(calculate_eoc(n_cells_per_disc, l2_errors[:, comp_idx]), 0, 0.0)

            tables[comp_idx] = pd.DataFrame({
                '$N_d$': [poly_deg] * len(n_cells_per_disc),
                '$N_e^z$': n_cells_per_disc.tolist(),
                'Max. error': linf_errors[:, comp_idx].tolist(),
                'Max. EOC': linf_eoc.tolist(),
                '$L^1$ error': l1_errors[:, comp_idx].tolist(),
                '$L^1$ EOC': l1_eoc.tolist(),
                '$L^2$ error': l2_errors[:, comp_idx].tolist(),
                '$L^2$ EOC': l2_eoc.tolist(),
                'Sim. time': sim_times.tolist(),
                'Min. value': min_values[:, comp_idx].tolist(),
                'Bulk DoF': (n_cells_per_disc * (poly_deg + 1)).tolist(),
            })

        return tables

    # Data per discretization method
    abs_errors = []
    DoFs = []
    bulk_DoFs = []
    convergence_tables = []
    if incl_min_val:
        min_val = []

    if type(exact_name) is not str and type(exact_name) is not None:
        
        reference = exact_name
        
    elif which == 'radial_outlet':
        
        extra_keys['domain_end'] = sim_go_to(get_simulation(file_path+exact_name).root,
                  ['input', 'model', 'unit_'+unit, 'col_radius']
                  )
        extra_keys['ref_coords'] = get_radial_coordinates(file_path+exact_name, unit)
        nRad = len(extra_keys['ref_coords'])
        reference = []
        kwargs.update(extra_keys)
        
        for rad in range(nRad):
            reference.append(get_solution(file_path+exact_name, 'unit_'+unit, 'outlet_port_{:03d}'.format(rad), kwargs.get(
                'comp', [-1])))
        reference=np.array(reference)
    else:
        
        reference = get_solution(file_path+exact_name, 'unit_'+unit, which, kwargs.get(
            'comp', [-1]), **{'sensIdx': kwargs.get('sensIdx', 0)})

    if simulation_names is None:
        
        simulation_names = []

        if rad_method is None:
            if par_method is None:
                simulation_names.append(
                    generate_simulation_names(
                        prefix=file_path+model,
                        ax_methods=[ax_method], ax_cells=ax_cells
                    )
                )
            else:
                simulation_names.append(
                    generate_simulation_names(
                        prefix=file_path+model,
                        ax_methods=[ax_method], ax_cells=ax_cells,
                        par_methods=[par_method], par_cells=par_cells
                    )
                )
        else:
            if par_method is None:
                simulation_names.append(
                    generate_simulation_names(
                        prefix=file_path+model,
                        ax_methods=[ax_method], ax_cells=ax_cells,
                        rad_methods=[rad_method], rad_cells=rad_cells
                    )
                )
            else:
                simulation_names.append(
                    generate_simulation_names(
                        prefix=file_path+model,
                        ax_methods=[ax_method], ax_cells=ax_cells,
                        par_methods=[par_method], par_cells=par_cells,
                        rad_methods=[rad_method], rad_cells=rad_cells
                    )
                )

    nTimePoints = len(get_solution_times(simulation_names[0]))
    
    error_weight = 1.0/nTimePoints if kwargs.get('time_weighted_error', False) else 1.0
    
    abs_errors = calculate_all_abs_errors(
        simulation_names,
        reference,
        unit,
        which=which,
        polyDeg=rad_method, nCells=rad_cells,
        **kwargs
        )
    if incl_min_val:
        min_val = calculate_all_min_vals(
            simulation_names,
            unit,
            which=which
            )
    if par_method is None and rad_method is None:
        DoFs = calculate_DOFs(ax_cells,
            ax_method,
            full_DOFs=True,
            model=transport_model,
            nComp=ncomp, nBound=nbound
            )
        bulk_DoFs = calculate_DOFs(ax_cells,
            ax_method,
            full_DOFs=False,
            model=transport_model,
            nComp=ncomp, nBound=nbound
            )
        # Convergence Table
        header, table = convergency_table(
            ax_method, ax_cells,
            abs_errors,
            delta=error_weight,
            error_types=kwargs.get('error_types', ['max', 'L1', 'L2']),
            sim_names=simulation_names
            )

    else:
        
        aux_methods = [ax_method]
        aux_cells = [ax_cells]
        
        if rad_method is not None:
            aux_methods.append(rad_method)
            aux_cells.append(rad_cells)
        
        if par_method is not None:
            aux_methods.append(par_method)
            aux_cells.append(par_cells)
        
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
            abs_errors,
            delta=error_weight,
            error_types=kwargs.get('error_types', ['max', 'L1', 'L2']),
            sim_names=simulation_names,
            full_DOFs=False,
            transport_model=transport_model
        )

    if incl_min_val:
        table = np.column_stack((np.array(table), min_val))
        header.append('Min. value')

    convergence_tables = table

    # Export Data
    if save_names == None:
        
        refinement_ID = kwargs['refinement_ID']
        
        disc = sim_go_to(get_simulation(simulation_names[0]).root,
                              ['input',
                               'model',
                               f'unit_{refinement_ID}',
                               'discretization'
                               ]
                )
        
        save_name = model + generate_bulkDisc_name(disc)
            
        if par_method is not None:
            
            disc = sim_go_to(get_simulation(simulation_names[0]).root,
                                  ['input',
                                   'model',
                                   f'unit_{refinement_ID}',
                                   'particle_type_000',
                                   'discretization'
                                   ]
                    )
            
            save_name = save_name + generate_parDisc_name(disc, generate_bulkDisc_name(disc))
    else:
        save_name = save_names

    # add (bulk) DoFs to output table
    header = np.insert(header, len(header), 'DoF')
    result = np.hstack(
        (convergence_tables, np.atleast_2d(DoFs).T))
    header = np.insert(header, len(header), 'Bulk DoF')
    result = np.hstack(
        (result, np.atleast_2d(bulk_DoFs).T))

    # add method to output table
    header = np.insert(header, 0, '$N_d$')
    arr = np.empty(len(ax_cells), dtype=object)
    arr[:] = str(int(abs(ax_method)))
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
    model = Cadet()
    model.install_path = cadet_path
    
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

            model.load_from_file()
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
