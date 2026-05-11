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
import error
import configuration as cfg


class ForwardSolver(ABC):
    """The abstract class for Forward Solvers.

    This class provides the expected attributes and methods of any
    implementation of a forward solver.

    Attributes
    ----------
        name : str
            The name of the method. It should be defined within the
            implementation of the method.
        et, ei : :class:`numpy.ndarray`
            Matrices containing the total and incident field
            information. The rows are points in D-domain following 'C'
            order. The columns are the sources.

        es : :class:`numpy.ndarray`
            Matrix containing the scattered field information. The rows
            correspond to the measurement points and the columns
            correspond to the sources.

        epsilon_r, sigma : :class:`numpy.ndarray`
            Matrices representing the dielectric properties in each
            pixel of the image. *epsilon_r* stands for the relative
            permitivitty and *sigma* stands for the conductivity (S/m).
            `Obs.:` the rows correspond to the y-coordinates, and the
            columns, to the x-ones.

        configuration : :class:`configuration:Configuration`
            Configuration object.
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
        """Execute the method given a problem input.

        This is the basic model of the simulation routine.

        Parameters
        ----------
            input : :class:`inputdata:InputData`
                An object of InputData type which must contains the
                `resolution` attribute and either `epsilon_r` or
                `sigma` or both.

        Returns
        -------
            es, et, ei : :class:`numpy.ndarray`
                Matrices with the computed scattered, total and incident
                fields, respectively.
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
        """Return the incident field for a given resolution."""
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

    The noise is implemmented as a complex number with fixed magnitude
    and random phase. Therefore, the user can control the percentage
    of noise amplitude.

    Parameters
    ----------
        x : array_like
            Data to receive noise.
        percentage : float
            Noise level in percentage.

    Returns
    -------
        xd : corrupted data
    """
    phase = np.reshape(2*np.pi*rnd.rand(x.size), x.shape)
    mod = percentage/100*np.abs(x)
    xd = x + mod*np.cos(phase) + 1j*mod*np.sin(phase)
    return xd
