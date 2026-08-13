"""The abstract class for forward solvers.

This module provides the abstract class for any forward solver used in
this package, with the basic methods and attributes. Therefore,
different forward methods may be implemented and coupled to inverse
solvers without to specify which one is implemented.

The following routine is also provided:

`add_noise(x, delta)`
    Add noise to an array.
"""

from abc import ABC, abstractmethod
import copy as cp
import numpy as np
from numpy import random as rnd
import pickle

from eispy2d.core import error
from eispy2d.core import configuration as cfg


class ForwardSolver(ABC):
    """Abstract base class for forward solvers.

    This class provides the expected attributes and methods of any
    implementation of a forward solver for electromagnetic scattering.

    Attributes
    ----------
    name : str
        The name of the method. Should be defined within the implementation.
    parallelization : bool
        Whether parallel processing is enabled.
    et : numpy.ndarray
        Total field information. Rows are points in D-domain (C order),
        columns are sources.
    ei : numpy.ndarray
        Incident field information. Rows are points in D-domain (C order),
        columns are sources.
    es : numpy.ndarray
        Scattered field information. Rows correspond to measurement points,
        columns correspond to sources.
    epsilon_r : numpy.ndarray
        Relative permittivity map (rows: y-coordinates, columns: x-coordinates).
    sigma : numpy.ndarray
        Conductivity map in S/m (rows: y-coordinates, columns: x-coordinates).
    configuration : Configuration
        Problem configuration object.

    Methods
    -------
    solve(inputdata, noise=None, PRINT_INFO=False, SAVE_INTERN_FIELD=True)
        Execute the forward solver given a problem input.
    incident_field(resolution, configuration)
        Return the incident field for a given resolution.
    save(file_name, file_path='')
        Save simulation data.
    importdata(file_name, file_path='')
        Import solver data from file.
    """

    def __init__(self, parallelization=False):
        """Create a forward solver object.

        Parameters
        ----------
            None
        """
        self.name = None
        self.parallelization = parallelization

    @abstractmethod
    def solve(self, inputdata, noise=None, PRINT_INFO=False,
              SAVE_INTERN_FIELD=True):
        """Execute the forward solver given a problem input.

        Parameters
        ----------
        inputdata : InputData
            Input data object containing the problem configuration and
            either relative permittivity or conductivity maps.
        noise : float, optional
            Noise level to add to the computed scattered field (percentage).
        PRINT_INFO : bool, default=False
            Whether to print progress information.
        SAVE_INTERN_FIELD : bool, default=True
            Whether to save the internal total field.

        Returns
        -------
        tuple
            (epsilon_r, sigma) arrays with the computed dielectric properties.
            Derived classes may return additional fields.
        """
        if inputdata.rel_permittivity is None and inputdata.conductivity is None:
            raise error.MissingAttributesError('InputData',
                                               'rel_permittivity or conductivity')
        if inputdata.rel_permittivity is not None:
            resolution = inputdata.rel_permittivity.shape
        else:
            resolution = inputdata.conductivity.shape

        if inputdata.rel_permittivity is None:
            epsilon_r = inputdata.configuration.epsilon_rb*np.ones(resolution)
        else:
            epsilon_r = np.copy(inputdata.rel_permittivity)

        if inputdata.conductivity is None:
            sigma = inputdata.configuration.sigma_b*np.ones(resolution)
        else:
            sigma = np.copy(inputdata.conductivity)

        return epsilon_r, sigma

    @abstractmethod
    def incident_field(self, resolution, configuration):
        """Return the incident field for a given resolution.

        Parameters
        ----------
        resolution : tuple of int
            Image resolution (NY, NX).
        configuration : Configuration
            Problem configuration object.

        Returns
        -------
        numpy.ndarray
            Incident field matrix with shape (NY*NX, NS) where NS is the number
            of sources.
        """
        return np.zeros((int, int), dtype=complex)

    @abstractmethod
    def save(self, file_name, file_path=''):
        """Save simulation data."""
        return {'name': self.name,
                'parallelization': self.parallelization}

    @abstractmethod
    def importdata(self, file_name, file_path=''):
        data = cfg.import_dict(file_name, file_path)
        self.name = data['name']
        self.parallelization = data['parallelization']
        return data

    @abstractmethod
    def __str__(self):
        """Print information of the method object."""
        return "Foward Solver: " + self.name + "\n"


def add_noise(x, percentage):
    r"""Add noise to data.

    The noise is implemented as a complex number with fixed magnitude
    and random phase. The user can control the percentage of noise amplitude.

    Parameters
    ----------
    x : array_like
        Data to receive noise.
    percentage : float
        Noise level in percentage.

    Returns
    -------
    xd : array_like
        Corrupted data with added noise.
    """
    phase = np.reshape(2*np.pi*rnd.rand(x.size), x.shape)
    mod = percentage/100*np.abs(x)
    xd = x + mod*np.cos(phase) + 1j*mod*np.sin(phase)
    return xd
