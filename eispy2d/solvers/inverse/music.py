r"""Multiple Signal Classification (MUSIC) imaging method.

This module implements the MUSIC algorithm [1]_ for qualitative electromagnetic
inverse scattering. The method uses singular value decomposition of the
measured scattered-field matrix to split the signal space into a *signal
subspace* and a *noise subspace*. A point :math:`z` in the investigation
domain belongs to the support of the scatterer when the Green's function
vector at :math:`z` has a large projection onto the signal subspace
(equivalently, a small projection onto the noise subspace), yielding a
sharp indicator image.

Classes
-------
MUSIC : dtm.Deterministic
    Main class implementing the MUSIC imaging algorithm.

Functions
---------
solve(U, GS, x)
    Numba-accelerated computation of the MUSIC indicator values.

Constants
---------
SV_CUTOFF, THRESHOLD
    Serialization keys.

References
----------
.. [1] Devaney, A. J. (2000). Super-resolution processing of multi-static
   data using time reversal and MUSIC. Preprint.
.. [2] Ammari, H., Iakovleva, E., & Lesselier, D. (2005). A MUSIC algorithm
   for locating small inclusions buried in a half-space from the scattering
   amplitude at a fixed frequency. SIAM Multiscale Modeling & Simulation,
   3(3), 597-628.
"""

import sys
import pickle
import time as tm
import numpy as np
from numpy import linalg as lag
from numpy import pi
from scipy.special import hankel2
from scipy.linalg import svd, norm
from numba import jit

from eispy2d.core import configuration as cfg
from eispy2d.core import result as rst
from eispy2d.solvers.base import deterministic as dtm


SV_CUTOFF = 'sv_cutoff'
THRESHOLD = 'threshold'


class MUSIC(dtm.Deterministic):
    """Multiple Signal Classification imaging method.

    This class implements the MUSIC algorithm for qualitative imaging,
    which uses singular value decomposition to identify scatterer locations.

    Parameters
    ----------
    alias : str, default=''
        Alias name for the algorithm.
    sv_cutoff : int or float, optional
        Singular value cutoff (int: number of values, float: threshold).
    threshold : float, optional
        Threshold for indicator function.
    import_filename : str, optional
        Filename to import algorithm state from.
    import_filepath : str, default=''
        Path to import file.
    """
    def __init__(self, alias='', sv_cutoff=None, threshold=None,
                 import_filename=None, import_filepath=''):
        """Initialize the MUSIC imaging method.

        Parameters
        ----------
        alias : str, default: ''
            Short identifier for the solver.
        sv_cutoff : int or float, optional
            Singular-value cutoff for partitioning signal and noise subspaces:

            - ``int``: retain the first `sv_cutoff` singular vectors.
            - ``float``: retain singular vectors whose singular value is at
              least `sv_cutoff`.
            - ``None``: use all singular vectors (no partition).
        threshold : float, optional
            Binary threshold applied to the normalized indicator image.
            If set, pixels below this value are set to zero.
        import_filename : str, optional
            Filename of a previously saved solver state to restore.
        import_filepath : str, default: ''
            Directory containing the import file.
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(alias=alias, parallelization=None)
            self.name = 'Multiple Signal Classification Imaging'
            self.sv_cutoff = sv_cutoff
            self.threshold = threshold

    def solve(self, inputdata, discretization=None, print_info=True,
              print_file=sys.stdout):
        """Solve the inverse scattering problem using the MUSIC algorithm.

        Computes the MUSIC indicator image by projecting the columns of
        the discretization Green's function matrix onto the signal subspace
        of the measured scattered-field matrix.

        Parameters
        ----------
        inputdata : InputData
            Object containing the measured scattered field, problem
            configuration, and optional ground-truth data for error metrics.
        discretization : Discretization
            Object providing the Green's function matrix ``GS`` and domain
            geometry utilities.
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
        execution_time = 0.
        tic = tm.time()
        U, sv, _ = lag.svd(inputdata.scattered_field, full_matrices=False)
        x = np.zeros(discretization.GS.shape[1])
        if self.sv_cutoff is None:
            pass
        elif type(self.sv_cutoff) is int and self.sv_cutoff < sv.size:
            U = U[:, :self.sv_cutoff]
        elif type(self.sv_cutoff) is float:
            U = U[:, sv >= self.sv_cutoff]
        solve(U, discretization.GS, x)
        execution_time += tm.time()-tic
        x = (x-np.amin(x))/(np.amax(x)-np.amin(x))
        contrast = discretization.contrast_image(x, inputdata.resolution)
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
        """Save the MUSIC solver state to file.

        Serializes the singular-value cutoff and threshold parameters
        using pickle.

        Parameters
        ----------
        file_path : str, default: ''
            Directory where the state file is written. The file is named
            after the solver's alias.
        """
        data = super().save(file_path=file_path)
        data[SV_CUTOFF] = self.sv_cutoff
        data[THRESHOLD] = self.threshold
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        """Import MUSIC solver state from file.

        Restores the solver parameters previously saved with :meth:`save`.

        Parameters
        ----------
        file_name : str
            Name of the file containing the saved solver state.
        file_path : str, default: ''
            Directory containing the file.
        """
        data = super().importdata(file_name, file_path=file_path)
        self.sv_cutoff = data[SV_CUTOFF]
        self.threshold = data[THRESHOLD]

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        super()._print_title(inputdata, discretization, print_file=print_file)
        message = 'Singular value cut-off: '
        if self.sv_cutoff is None:
            message += 'None'
        elif type(self.sv_cutoff) is int:
            message += 'First %d values' % self.sv_cutoff
        else:
            message += '%.1e' % self.sv_cutoff
        if self.threshold is not None:
            message = '\nThreshold: %.2f' % self.threshold
        print(message, file=print_file)

    def copy(self, new=None):
        """Create a copy of this MUSIC instance.

        Parameters
        ----------
        new : MUSIC, optional
            Existing instance to copy attributes into. If ``None``,
            a new instance is created and returned.

        Returns
        -------
        MUSIC or None
            A new instance when `new` is ``None``; otherwise ``None``
            (the provided instance is modified in place).
        """
        if new is None:
            return MUSIC(alias=self.alias, sv_cutoff=self.sv_cutoff,
                         threshold=self.threshold)
        else:
            super().copy(new)
            self.sv_cutoff = new.sv_cutoff
            self.threshold = new.threshold

    def __str__(self):
        message = super().__str__()
        message += '\nSingular value cut-off: '
        if self.sv_cutoff is None:
            message += 'None'
        elif type(self.sv_cutoff) is int:
            message += 'First %d values' % self.sv_cutoff
        else:
            message += '%.1e' % self.sv_cutoff
        if self.threshold is not None:
            message = '\nThreshold: %.2f' % self.threshold
        return message


@jit(nopython=True)
def solve(U, GS, x):
    """Compute MUSIC indicator values for all domain sampling points.

    For each column of `GS` (one per domain point), computes the sum
    of squared projections onto the columns of `U` (signal subspace).
    Points with large values are inside the support of the scatterer.

    Parameters
    ----------
    U : numpy.ndarray
        Signal subspace matrix, shape ``(NM, L)``, where `L` is the
        number of retained singular vectors.
    GS : numpy.ndarray
        Green's function matrix, shape ``(NM, ND)``, where ``ND`` is
        the number of domain discretization points.
    x : numpy.ndarray
        Output indicator array of shape ``(ND,)``, filled in place.
    """
    for n in range(x.size):
        den = 0
        for j in range(U.shape[1]):
            den += np.abs(np.sum(np.conj(U[:, j])*GS[:, n]))**2
        x[n] = den

