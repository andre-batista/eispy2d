r"""Linear Sampling Method for qualitative electromagnetic inverse scattering.

This module implements the Linear Sampling Method (LSM) [1]_, a qualitative
technique that reconstructs the *support* (shape) of scattering objects
without recovering material properties. For each sampling point in the
investigation domain, the method solves a far-field (or near-field) integral
equation of the first kind; the norm of the solution is large outside the
scatterer and small inside, yielding a sharp indicator image.

The module supports far-field and near-field measurement configurations and
provides optional Tikhonov or singular-value regularization.

Classes
-------
LinearSamplingMethod : dtm.Deterministic
    Main class implementing the Linear Sampling Method.

Functions
---------
standard(x)
    Default indicator function: :math:`-\log_{10}(x)`.
solve(U, s, V, solution, rhs, alpha)
    Numba-accelerated SVD-based solver for the LSM linear system.

Constants
---------
REGULARIZATION, TIKHONOV, SV_CUTOFF, THRESHOLD, FAR_FIELD, INDICATOR
    Serialization keys.

References
----------
.. [1] Colton, D., & Kirsch, A. (1996). A simple method for solving inverse
   scattering problems in the resonance region. Inverse Problems, 12(4),
   383-393.
.. [2] Colton, D., & Kress, R. (2013). Inverse Acoustic and Electromagnetic
   Scattering Theory (3rd ed.). Springer.
"""

import sys
import pickle
import time as tm
import numpy as np
from numpy import pi
from scipy.special import hankel2
from scipy.linalg import svd, norm
from numba import jit

from eispy2d.core import configuration as cfg
from eispy2d.core import result as rst
from eispy2d.solvers.base import deterministic as dtm


REGULARIZATION = 'regularization'
TIKHONOV = 'tikhonov'
SV_CUTOFF = 'sv_cutoff'
THRESHOLD = 'threshold'
FAR_FIELD = 'far_field'
INDICATOR = 'indicator'


def standard(x):
    r"""Default LSM indicator function.

    Transforms the norm of the LSM solution into a positive-valued
    indicator. Large values indicate points *outside* the scatterer;
    small values indicate points *inside*.

    Parameters
    ----------
    x : numpy.ndarray
        Array of solution norms, one value per sampling point.

    Returns
    -------
    numpy.ndarray
        Indicator values :math:`-\log_{10}(x)`, same shape as `x`.
    """
    return -np.log10(x)


class LinearSamplingMethod(dtm.Deterministic):
    """Linear Sampling Method for qualitative inverse scattering.

    This class implements the Linear Sampling Method (LSM), a qualitative
    technique for shape reconstruction of scatterers. The method identifies
    the support of the scatterer without reconstructing material properties.

    Parameters
    ----------
    alias : str, default=''
        Alias name for the algorithm.
    regularization : Regularization, optional
        Regularization method for solving linear systems.
    tikhonov : float, optional
        Tikhonov regularization parameter.
    sv_cutoff : float, optional
        Singular value cutoff threshold.
    threshold : float, optional
        Threshold for indicator function.
    far_field : bool, optional
        Whether to use far-field approximation.
    indicator_function : callable, default=standard
        Function to compute indicator values.
    import_filename : str, optional
        Filename to import algorithm state from.
    import_filepath : str, default=''
        Path to import file.
    """
    def __init__(self, alias='', regularization=None, tikhonov=None,
                 sv_cutoff=None, threshold=None, far_field=None,
                 indicator_function=standard, import_filename=None,
                 import_filepath=''):
        """Initialize the Linear Sampling Method.

        Parameters
        ----------
        alias : str, default: ''
            Short identifier for the solver.
        regularization : Regularization, optional
            Regularization object (e.g. :class:`Tikhonov`). When provided,
            it is used for every sampling point instead of the built-in SVD
            solver.
        tikhonov : float, optional
            Tikhonov parameter for the built-in SVD solver. Ignored when
            `regularization` is provided.
        sv_cutoff : float, optional
            Singular-value cutoff threshold for the built-in SVD solver.
        threshold : float, optional
            Binary threshold applied to the normalized indicator image.
            If set, pixels below this value are set to zero.
        far_field : bool or None, optional
            Force far-field (``True``) or near-field (``False``) kernel.
            If ``None``, the choice is made automatically based on the
            measurement radius :math:`R_o` and wavelength :math:`\lambda_b`.
        indicator_function : callable, default: standard
            Function that maps solution norms to indicator values.
        import_filename : str, optional
            Filename of a previously saved solver state to restore.
        import_filepath : str, default: ''
            Directory containing the import file.
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(alias=alias, parallelization=None)
            self.name = 'Linear Sampling Method'
            self.regularization = regularization
            self.tikhonov = tikhonov
            self.sv_cutoff = sv_cutoff
            self.threshold = threshold
            self.far_field = far_field
            self.indicator = indicator_function

    def solve(self, inputdata, discretization=None, print_info=True,
              print_file=sys.stdout):
        """Solve the inverse scattering problem using the Linear Sampling Method.

        For each point in the investigation domain, an integral equation of
        the first kind is solved; the norm of the solution provides the
        indicator image identifying the scatterer's support.

        Parameters
        ----------
        inputdata : InputData
            Object containing the measured scattered field, problem
            configuration, and optional ground-truth data for error metrics.
        discretization : Discretization or None, optional
            Not used by LSM; kept for API compatibility with the base class.
        print_info : bool, default: True
            Whether to print progress information.
        print_file : file-like object, default: sys.stdout
            Destination for progress messages.

        Returns
        -------
        result : Result
            Result object containing the reconstructed indicator image
            (stored as relative permittivity / conductivity maps) and
            optional error metrics.
        """
        result = super().solve(inputdata, discretization,
                               print_info=print_info, print_file=print_file)
        if self.far_field is not None:
            far_field = self.far_field
        else:
            if (inputdata.configuration.Ro
                    >= 10*inputdata.configuration.lambda_b):
                far_field = True
            else:
                far_field = False
        execution_time = 0.
        tic = tm.time()
        if far_field:
            K = self._far_field_kernel(inputdata)
            rhs = self._far_field_rhs(inputdata)
        else:
            K = self._near_field_kernel(inputdata)
            rhs = self._near_field_rhs(inputdata)
        execution_time += tm.time()-tic
        solution = np.zeros(np.prod(inputdata.resolution))
        tic = tm.time()
        if self.regularization is not None:
            for n in range(solution.size):
                t = self.regularization.solve(K, rhs[:, n].flatten())
                solution[n] = norm(t)
        else:
            U, s, Vh = svd(K, full_matrices=False)
            V = np.conj(Vh).T
            if self.tikhonov is None:
                alpha = 0.
            else:
                alpha = self.tikhonov
            if self.sv_cutoff is not None:
                s = s[s > self.sv_cutoff]
            solve(U, s, V, solution, rhs, alpha)
        solution = self.indicator(solution)
        execution_time += tm.time()-tic
        sol_min, sol_max = np.amin(solution), np.amax(solution)
        contrast = (solution-sol_min)/(sol_max-sol_min)
        contrast = contrast.reshape(inputdata.resolution)
        if self.threshold is not None:
            contrast = contrast > self.threshold
            contrast = contrast.astype(float)
        if not inputdata.configuration.good_conductor:
            result.rel_permittivity = cfg.get_relative_permittivity(
                contrast, inputdata.configuration.epsilon_rb
            )
        if not inputdata.configuration.perfect_dielectric:
            result.conductivity = cfg.get_conductivity(
                contrast, 2*pi*inputdata.configuration.f,
                inputdata.configuration.epsilon_rb,
                inputdata.configuration.sigma_b
            )
        if rst.SHAPE_ERROR in inputdata.indicators:
            groundtruth = cfg.get_contrast_map(
                epsilon_r=inputdata.rel_permittivity,
                sigma=inputdata.conductivity,
                configuration=inputdata.configuration
            )
            result.zeta_s = [rst.compute_zeta_s(groundtruth, contrast)]
        if rst.POSITION_ERROR in inputdata.indicators:
            groundtruth = cfg.get_contrast_map(
                epsilon_r=inputdata.rel_permittivity,
                sigma=inputdata.conductivity,
                configuration=inputdata.configuration
            )
            result.zeta_p = [rst.compute_zeta_p(groundtruth, contrast)]
        if rst.EXECUTION_TIME in inputdata.indicators:
            result.execution_time = execution_time
        return result

    def save(self, file_path=''):
        """Save the Linear Sampling Method state to file.

        Serializes all solver parameters (regularization, thresholds,
        field-approximation mode, indicator function) using pickle.

        Parameters
        ----------
        file_path : str, default: ''
            Directory where the state file is written. The file is named
            after the solver's alias.
        """
        data = super().save(file_path=file_path)
        data[REGULARIZATION] = self.regularization
        data[TIKHONOV] = self.tikhonov
        data[SV_CUTOFF] = self.sv_cutoff
        data[THRESHOLD] = self.threshold
        data[FAR_FIELD] = self.far_field
        data[INDICATOR] = self.indicator
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        """Import Linear Sampling Method state from file.

        Restores all solver parameters previously saved with
        :meth:`save`.

        Parameters
        ----------
        file_name : str
            Name of the file containing the saved solver state.
        file_path : str, default: ''
            Directory containing the file.
        """
        data = super().importdata(file_name, file_path=file_path)
        self.regularization = data[REGULARIZATION]
        self.tikhonov = data[TIKHONOV]
        self.sv_cutoff = data[SV_CUTOFF]
        self.threshold = data[THRESHOLD]
        self.far_field = data[FAR_FIELD]
        self.indicator = data[INDICATOR]

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        super()._print_title(inputdata, discretization, print_file=print_file)
        if self.regularization is not None:
            print(self.regularization, file=print_file)
        else:
            message = 'Regularization: Standard('
            if self.tikhonov is not None:
                message += 'Tikhonov parameter: %.1e' % self.tikhonov
            else:
                message += 'Tikhonov parameter: 0'
            if self.sv_cutoff is not None:
                message += ', Singular value cut-off: %.1e' % self.sv_cutoff
            message += ')'
            print(message, file=print_file)
        if self.threshold is not None:
            print('Threshold: %.2f' % self.threshold, file=print_file)
        if self.far_field is True:
            print('Field approximation: Far', file=print_file)
        elif self.far_field is False:
            print('Field approximation: Near', file=print_file)
        elif self.far_field is None:
            print('Field approximation: automatic', file=print_file)
        print('Indicator function: ' + self.indicator.__name__,
              file=print_file)

    def copy(self, new=None):
        """Create a copy of this Linear Sampling Method instance.

        Parameters
        ----------
        new : LinearSamplingMethod, optional
            Existing instance to copy attributes into. If ``None``,
            a new instance is created and returned.

        Returns
        -------
        LinearSamplingMethod or None
            A new instance when `new` is ``None``; otherwise ``None``
            (the provided instance is modified in place).
        """
        if new is None:
            return LinearSamplingMethod(alias=self.alias,
                                        regularization=self.regularization,
                                        tikhonov=self.tikhonov,
                                        sv_cutoff=self.sv_cutoff,
                                        threshold=self.threshold,
                                        far_field=self.far_field,
                                        indicator_function=self.indicator)
        else:
            super().copy(new)
            self.regularization = new.regularization
            self.tikhonov = new.tikhonov
            self.sv_cutoff = new.sv_cutoff
            self.threshold = new.threshold
            self.far_field = new.far_field
            self.indicator = new.indicator

    def __str__(self):
        message = super().__str__()
        if self.regularization is not None:
            message += str(self.regularization)
        else:
            message += '\nRegularization: Standard('
            if self.tikhonov is not None:
                message += 'Tikhonov parameter: %.1e' % self.tikhonov
            else:
                message += 'Tikhonov parameter: 0'
            if self.sv_cutoff is not None:
                message += ', Singular value cut-off: %.1e' % self.sv_cutoff
            message += ')'
        if self.threshold is not None:
            message += ('\nThreshold: %.2f' % self.threshold)
        if self.far_field is True:
            message += '\nField approximation: Far'
        elif self.far_field is False:
            message += '\nField approximation: Near'
        elif self.far_field is None:
            message += '\nField approximation: automatic'
        message += '\nIndicator function: ' + self.indicator.__name__
        return message

    def _far_field_kernel(self, inputdata):
        NS = inputdata.configuration.NS
        kb = inputdata.configuration.kb
        rho = inputdata.configuration.Ro
        dphi = 2*pi/NS
        E_inf = np.sqrt(rho)/np.exp(-1j*kb*rho)*inputdata.scattered_field
        return E_inf*dphi

    def _far_field_rhs(self, inputdata):
        NM = inputdata.configuration.NM
        kb = inputdata.configuration.kb
        theta = cfg.get_angles(NM)
        x, y = cfg.get_coordinates_ddomain(
            configuration=inputdata.configuration,
            resolution=inputdata.resolution
        )
        x, y = x.flatten(), y.flatten()
        N = x.size
        r = np.sqrt(x**2 + y**2)
        psi = np.arctan2(y, x)
        psi[psi<0] = 2*pi + psi[psi<0]
        Phi = np.zeros((NM, N), dtype=complex)
        for n in range(N):
            Phi[:, n] = (-1j/4*np.sqrt(2/(pi*kb))
                         * np.exp(1j*pi/4)
                         * np.exp(1j*kb*r[n]*np.cos(theta - psi[n])))
        return Phi

    def _near_field_kernel(self, inputdata):
        NS = inputdata.configuration.NS
        dphi = 2*pi/NS
        return inputdata.scattered_field*dphi

    def _near_field_rhs(self, inputdata):
        NM = inputdata.configuration.NM
        kb = inputdata.configuration.kb
        rho = inputdata.configuration.Ro
        theta = cfg.get_angles(NM)
        x, y = cfg.get_coordinates_ddomain(
            configuration=inputdata.configuration,
            resolution=inputdata.resolution
        )
        x, y = x.flatten(), y.flatten()
        N = x.size
        Phi = np.zeros((NM, N), dtype=complex)
        for n in range(N):
            Phi[:, n] = (
                -1j/4*hankel2(0, kb*np.sqrt((rho*np.cos(theta)-x[n])**2
                                            + (rho*np.sin(theta)-y[n])**2))
            )
        return Phi


@jit(nopython=True)
def solve(U, s, V, solution, rhs, alpha):
    """Compute LSM solution norms via SVD with Tikhonov regularization.

    For each sampling point (column of `rhs`), solves the regularized
    system and stores the norm of the solution in `solution`.

    Parameters
    ----------
    U : numpy.ndarray
        Left singular vectors, shape ``(NM, P)``.
    s : numpy.ndarray
        Singular values, shape ``(P,)``.
    V : numpy.ndarray
        Right singular vectors (already conjugate-transposed), shape
        ``(N, P)`` where ``N`` is the number of domain elements.
    solution : numpy.ndarray
        Output array of shape ``(N_sampling,)`` that is filled in place
        with the solution norms.
    rhs : numpy.ndarray
        Right-hand-side matrix of shape ``(NM, N_sampling)`` whose
        columns are the Green's-function vectors for each sampling point.
    alpha : float
        Tikhonov regularization parameter.
    """
    N = solution.size
    P = s.size
    for n in range(N):
        t = s[0]/(s[0]**2+alpha) * np.sum(rhs[:, n]*np.conj(U[:, 0])) * V[:, 0]
        for j in range(1, P):
            t += s[j]/(s[j]**2+alpha) * np.sum(rhs[:, n]*np.conj(U[:, j]))* V[:, j]
        solution[n] = np.sqrt(np.sum(np.abs(t)**2))

