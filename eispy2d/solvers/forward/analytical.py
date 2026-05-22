r"""Analytical Solution for Electromagnetic Scattering by Cylinders.

This module provides the analytical solution for electromagnetic scattering
by dielectric and conducting cylinders. It solves the forward problem
following the Forward Solver abstract class using exact mathematical
formulations based on cylindrical wave expansions.

The implementation supports both perfect dielectric and perfect conductor
cylinders, providing exact solutions that can be used for validation of
numerical methods and benchmarking purposes.

References
----------
.. [1] Chew, Weng Cho. "Waves and fields in inhomogeneous media." 
   IEEE press, 1995.

.. [2] Balanis, Constantine A. "Advanced engineering electromagnetics." 
   John Wiley & Sons, 2012.

.. [3] Harrington, Roger F. "Time-harmonic electromagnetic fields." 
   IEEE press, 2001.
"""

import pickle
import numpy as np
from numpy import pi
from scipy.constants import epsilon_0, mu_0
from scipy.special import jv, jvp, hankel2, h2vp
from matplotlib import pyplot as plt

from eispy2d.solvers.base import forward as fwr
from eispy2d.data import inputdata as ipt
from eispy2d.core import configuration as cfg
from eispy2d.core import error


PERFECT_DIELECTRIC_PROBLEM = 'perfect_dieletric'
PERFECT_CONDUCTOR_PROBLEM = 'perfect_conductor'


class Analytical(fwr.ForwardSolver):
    """Analytical Solution for Electromagnetic Scattering by Cylinders.

    This class implements analytical solutions for electromagnetic scattering
    by dielectric and perfectly conducting cylinders using cylindrical wave
    expansions. It provides exact solutions that can be used for validation
    of numerical methods and benchmarking purposes.

    Attributes
    ----------
    name : str
        Name identifier for the solver.
    contrast : float or None
        Relative permittivity contrast of the cylinder.
    radius : float or None
        Radius of the cylinder in wavelengths.
    """

    def __init__(self, contrast=None, radius=None):
        """Create the analytical solver object.

        Parameters
        ----------
        contrast : float, optional
            Relative permittivity contrast of the dielectric cylinder.
            Required for perfect dielectric problems.
        radius : float, optional
            Radius of the cylinder in wavelengths. Required for both
            dielectric and conductor problems.
        """
        super().__init__()
        self.name = "Analytical Solution to Cylinder Scattering"
        self.contrast = contrast
        self.radius = radius

    def incident_field(self, resolution, configuration):
        """Compute the incident field matrix.

        Computes the incident field matrix for plane waves from different
        angles based on the configuration parameters.

        Parameters
        ----------
        resolution : 2-tuple
            The image size of D-domain in pixels (y, x).
        configuration : Configuration
            Configuration object containing problem parameters.

        Returns
        -------
        ei : numpy.ndarray
            Incident field matrix. Rows correspond to points in the image
            following C-order and columns correspond to sources.
        """
        NY, NX = resolution
        phi = cfg.get_angles(configuration.NS)
        x, y = cfg.get_coordinates_ddomain(configuration=configuration,
                                           resolution=resolution)
        kb = configuration.kb
        E0 = configuration.E0
        if isinstance(kb, float) or isinstance(kb, complex):
            ei = E0*np.exp(-1j*kb*(x.reshape((-1, 1))
                                   @ np.cos(phi.reshape((1, -1)))
                                   + y.reshape((-1, 1))
                                   @ np.sin(phi.reshape((1, -1)))))
        else:
            ei = np.zeros((NX*NY, self.configuration.NS, kb.size),
                          dtype=complex)
            for f in range(kb.size):
                ei[:, :, f] = E0*np.exp(-1j*kb[f]*(x.reshape((-1, 1))
                                                   @ np.cos(phi.reshape((1,
                                                                         -1)))
                                                   + y.reshape((-1, 1))
                                                   @ np.sin(phi.reshape((1,
                                                                         -1))))
                                        )
        return ei

    def solve(self, inputdata, noise=None, PRINT_INFO=False,
              COMPUTE_SCATTERED_FIELD=True, SAVE_INTERN_FIELD=True):
        """Solve the analytical scattering problem.

        Determines the problem type (dielectric or conductor) and calls
        the appropriate analytical solution method.

        Parameters
        ----------
        inputdata : InputData
            Input data object containing configuration and problem setup.
        noise : float, optional
            Noise level to add to scattered field (not used in this method).
        PRINT_INFO : bool, default: False
            Print iteration information (not used in analytical solution).
        COMPUTE_SCATTERED_FIELD : bool, default: True
            Compute scattered field (not used in this method).
        SAVE_INTERN_FIELD : bool, default: True
            Save the total field in the D-domain.

        Raises
        ------
        WrongValueInput
            If neither perfect_dielectric nor good_conductor is True.
        """
        
        if inputdata.configuration.perfect_dielectric:
            self.dielectric_cylinder(inputdata,
                                     SAVE_INTERN_FIELD=SAVE_INTERN_FIELD,
                                     SAVE_MAP=True)
        elif inputdata.configuration.good_conductor:
            self.conductor_cylinder(inputdata,
                                    radius_proportion=self.radius,
                                    SAVE_INTERN_FIELD=SAVE_INTERN_FIELD,
                                    SAVE_MAP=True)
        else:
            raise error.WrongValueInput('Analytical.solve',
                                        'inputdata.configuration',
                                        "either attributes "
                                        + "'perfect_dielectric' or "
                                        + "'good_conductor' must be True",
                                        'both are False')

    def dielectric_cylinder(self, inputdata, SAVE_INTERN_FIELD=True,
                            SAVE_MAP=False):
        """Solve scattering by a dielectric cylinder analytically.

        Computes the analytical solution for electromagnetic scattering
        by a dielectric cylinder using cylindrical wave expansions.

        Parameters
        ----------
        inputdata : InputData
            Input data object containing configuration and problem setup.
        SAVE_INTERN_FIELD : bool, default: True
            Save the total field in the D-domain.
        SAVE_MAP : bool, default: False
            Save the relative permittivity map.

        Raises
        ------
        MissingAttributesError
            If radius or contrast attributes are None.
        """
        if self.radius is None:
            raise error.MissingAttributesError('Analytical', 'radius')
        elif self.contrast is None:
            raise error.MissingAttributesError('Analytical', 'contrast')
        # Main constants
        omega = 2*pi*inputdata.configuration.f  # Angular frequency [rad/s]
        epsilon_rd = cfg.get_relative_permittivity(
            self.contrast, inputdata.configuration.epsilon_rb
        )
        epsd = epsilon_rd*epsilon_0  # Cylinder's permittivity [F/m]
        epsb = inputdata.configuration.epsilon_rb*epsilon_0
        mud = mu_0  # Cylinder's permeability [H/m]
        kb = inputdata.configuration.kb  # Wavenumber of background [rad/m]
        kd = omega*np.sqrt(mud*epsd)  # Wavenumber of cylinder [rad/m]
        lambdab = 2*pi/kb  # Wavelength of background [m]
        a = self.radius*lambdab  # Sphere's radius [m]
        thetal = cfg.get_angles(inputdata.configuration.NS)
        thetam = cfg.get_angles(inputdata.configuration.NM)

        # Summing coefficients
        an, cn, n = get_coefficients(kb, kd, a, epsd, epsb)

        # Mesh parameters
        x, y = cfg.get_coordinates_ddomain(
            configuration=inputdata.configuration,
            resolution=inputdata.resolution
        )

        # Total field array
        et = compute_total_field(x, y, a, an, cn, n, kb, kd,
                                 inputdata.configuration.E0, thetal)

        # Map of parameters
        epsilon_r, _ = get_map(x, y, a, inputdata.configuration.epsilon_rb,
                               epsilon_rd)

        # Scatered field
        rho = inputdata.configuration.Ro
        xm, ym = rho*np.cos(thetam), rho*np.sin(thetam)
        es = compute_scattered_field(xm, ym, an, n, kb, thetal,
                                     inputdata.configuration.E0)

        if inputdata.noise is not None and inputdata.noise > 0:
            es = fwr.add_noise(es, inputdata.noise)

        inputdata.scattered_field = np.copy(es)
        if SAVE_INTERN_FIELD:
            inputdata.total_field = np.copy(et)
        if SAVE_MAP:
            inputdata.rel_permittivity = np.copy(epsilon_r)

    def conductor_cylinder(self, inputdata, SAVE_INTERN_FIELD=True,
                           SAVE_MAP=False):
        """Solve scattering by a perfectly conducting cylinder analytically.

        Computes the analytical solution for electromagnetic scattering
        by a perfectly conducting cylinder using cylindrical wave expansions.

        Parameters
        ----------
        inputdata : InputData
            Input data object containing configuration and problem setup.
        SAVE_INTERN_FIELD : bool, default: True
            Save the total field in the D-domain.
        SAVE_MAP : bool, default: False
            Save the conductivity map.

        Raises
        ------
        MissingAttributesError
            If radius attribute is None.
        """
        if self.radius is None:
            raise error.MissingAttributesError('Analytical', 'radius')
        # Main constants
        kb = inputdata.configuration.kb  # Wavenumber of background [rad/m]
        a = self.radius*inputdata.configuration.lambda_b  # Sphere's radius
        thetal = cfg.get_angles(inputdata.configuration.NS)
        thetam = cfg.get_angles(inputdata.configuration.NM)

        # Summing coefficients
        n = np.arange(-1000, 1001)
        criteria = np.abs(jv(n, kb*a))
        n = n[criteria > 1e-25]
        an = -jv(n, kb*a)/hankel2(n, kb*a)
        cn = np.zeros(n.size)

        # Mesh parameters
        x, y = cfg.get_coordinates_ddomain(
            configuration=inputdata.configuration,
            resolution=inputdata.resolution
        )

        # Total field array
        et = compute_total_field(x, y, a, an, cn, n, kb, 1.,
                                 inputdata.configuration.E0, thetal)

        # Map of parameters
        sigma = np.zeros(x.shape)
        sigma[x**2 + y**2 <= a**2] = 1e10

        # Scatered field
        rho = self.configuration.Ro
        xm, ym = rho*np.cos(thetam), rho*np.sin(thetam)
        es = compute_scattered_field(xm, ym, an, n, kb, thetal,
                                     inputdata.configuration.E0)

        if inputdata.noise is not None and inputdata.noise > 0:
            es = fwr.add_noise(es, inputdata.noise)

        inputdata.scattered_field = np.copy(es)
        if SAVE_INTERN_FIELD:
            inputdata.total_field = np.copy(et)
        if SAVE_MAP:
            inputdata.conductivity = np.copy(sigma)

    def save(self, file_name, file_path=''):
        """Save the analytical solver object to file.

        Parameters
        ----------
        file_name : str
            Name of the file to save.
        file_path : str, default: ''
            Path where to save the file.
        """
        data = super().save()
        data['radius'] = self.radius
        data['contrast'] = self.contrast
        with open(file_path + file_name, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        """Import analytical solver data from file.

        Parameters
        ----------
        file_name : str
            Name of the file to import.
        file_path : str, default: ''
            Path of the file to import.
        """
        data = super().importdata(file_name, file_path=file_path)
        self.radius = data['radius']
        self.contrast = data['contrast']

    def __str__(self):
        """Return string representation of the analytical solver.

        Returns
        -------
        str
            String description including radius and contrast if available.
        """
        message = super().__str__()
        if self.radius is not None:
            message += 'Radius: %.2e [wavelengths] ' % self.radius 
        if self.contrast is not None:
            message += 'Contrast: %.1f' % self.contrast
        return message


def cart2pol(x, y):
    """Summarize the method."""
    rho = np.sqrt(x**2+y**2)
    phi = np.arctan2(y, x)
    phi[phi < 0] = 2*pi + phi[phi < 0]
    return rho, phi
 

def get_coefficients(wavenumber_b, wavenumber_d, radius, epsilon_d,
                     epsilon_b):
    """Summarize the method."""
    n = np.arange(-1000, 1001)
    kb, kd = wavenumber_b, wavenumber_d
    a = radius

    criteria = np.abs(jvp(n, kd*a))
    n = n[criteria > 1e-25]

    an = (-jv(n, kb*a)/hankel2(n, kb*a)*(
        (epsilon_d*jvp(n, kd*a)/(epsilon_b*kd*a*jv(n, kd*a))
         - jvp(n, kb*a)/(kb*a*jv(n, kb*a)))
        / (epsilon_d*jvp(n, kd*a)/(epsilon_b*kd*a*jv(n, kd*a))
           - h2vp(n, kb*a)/(kb*a*hankel2(n, kb*a)))
    ))

    cn = 1/jv(n, kd*a)*(jv(n, kb*a)+an*hankel2(n, kb*a))

    return an, cn, n


def rotate_axis(theta, x, y):
    """Summarize the method."""
    T = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    r = np.vstack((x.reshape(-1), y.reshape(-1)))
    rp = T@r
    xp, yp = np.vsplit(rp, 2)
    xp = np.reshape(np.squeeze(xp), x.shape)
    yp = np.reshape(np.squeeze(yp), y.shape)
    return xp, yp


def compute_total_field(x, y, radius, an, cn, N, wavenumber_b, wavenumber_d,
                        magnitude, theta=None):
    """Summarize the method."""
    E0 = magnitude
    kb, kd = wavenumber_b, wavenumber_d
    a = radius

    if theta is None:
        rho, phi = cart2pol(x, y)
        et = np.zeros(rho.shape, dtype=complex)
        i = 0
        for n in N:

            et[rho > a] = et[rho > a] + (
                E0*1j**(-n)*(jv(n, kb*rho[rho > a])
                             + an[i]*hankel2(n, kb*rho[rho > a]))
                * np.exp(1j*n*phi[rho > a])
            )

            et[rho <= a] = et[rho <= a] + (
                E0*1j**(-n)*cn[i]*jv(n, kd*rho[rho <= a])
                * np.exp(1j*n*phi[rho <= a])
            )

            i += 1

    else:
        S = theta.size
        et = np.zeros((x.size, S), dtype=complex)
        for s in range(S):
            xp, yp = rotate_axis(theta[s], x.reshape(-1), y.reshape(-1))
            rho, phi = cart2pol(xp, yp)
            i = 0
            for n in N:

                et[rho > a, s] = et[rho > a, s] + (
                    E0*1j**(-n)*(jv(n, kb*rho[rho > a])
                                 + an[i]*hankel2(n, kb*rho[rho > a]))
                    * np.exp(1j*n*phi[rho > a])
                )

                et[rho <= a, s] = et[rho <= a, s] + (
                    E0*1j**(-n)*cn[i]*jv(n, kd*rho[rho <= a])
                    * np.exp(1j*n*phi[rho <= a])
                )

                i += 1
    return et


def get_map(x, y, radius, epsilon_rb, epsilon_rd):
    """Summarize the method."""
    epsilon_r = epsilon_rb*np.ones(x.shape)
    sigma = np.zeros(x.shape)
    epsilon_r[x**2+y**2 <= radius**2] = epsilon_rd
    return epsilon_r, sigma


def compute_scattered_field(xm, ym, an, n, kb, theta, magnitude):
    """Summarize the method."""
    M, S, N = xm.size, theta.size, round((an.size-1)/2)
    E0 = magnitude
    es = np.zeros((M, S), dtype=complex)
    for s in range(S):
        xp, yp = rotate_axis(theta[s], xm, ym)
        rho, phi = cart2pol(xp, yp)
        for j in range(phi.size):
            es[j, s] = E0*np.sum(1j**(-n)*an*hankel2(n, kb*rho[j])
                                 * np.exp(1j*n*phi[j]))

    return es
