"""Back-Propagation Algorithm for Electromagnetic Inverse Scattering.

This module implements the Back-Propagation algorithm for solving two-dimensional
electromagnetic inverse scattering problems. The algorithm uses scattered field
measurements to reconstruct the contrast function of the scattering object.

The Back-Propagation method is a non-iterative technique that provides a fast
approximation to the inverse scattering problem by computing the contrast
directly from the scattered field measurements using Green's function relationships.

References
----------
.. [1] Devaney, A. J. "A filtered backpropagation algorithm for diffraction 
   tomography." Ultrasonic imaging 4.4 (1982): 336-350.

.. [2] Chew, Weng Cho, and Yih-Min Wang. "Reconstruction of two-dimensional 
   permittivity distribution using the distorted Born iterative method." 
   IEEE transactions on medical imaging 9.2 (1990): 218-225.
"""

# Standard libraries
import sys
import pickle
import time as tm
import numpy as np
from numpy import pi
from numba import jit

# Developed libraries
import deterministic as dtm
import mom_cg_fft as mom
import configuration as cfg
import result as rst
import fftproduct

FORWARD = 'forward'

class BackPropagation(dtm.Deterministic):
    """Back-Propagation Algorithm for Electromagnetic Inverse Scattering.

    This class implements the Back-Propagation algorithm, a non-iterative method
    for solving electromagnetic inverse scattering problems. The algorithm
    reconstructs the contrast function directly from scattered field measurements
    using Green's function relationships.

    The method provides a fast approximation to the inverse problem by computing
    the contrast in a single step, making it suitable for real-time applications
    or as an initial guess for iterative methods.

    Parameters
    ----------
    forward : ForwardSolver, default: mom.MoM_CG_FFT()
        Forward solver object used to compute incident fields and Green's functions.
    alias : str, default: 'backprop'
        Alias name for the algorithm instance.
    import_filename : str, optional
        Filename to import previously saved algorithm state.
    import_filepath : str, default: ''
        Path to the file containing saved algorithm state.

    Attributes
    ----------
    name : str
        Name of the algorithm ('Back-Propagation').
    forward : ForwardSolver
        Forward solver instance used for field computations.
    """

    def __init__(self, forward=mom.MoM_CG_FFT(), alias='backprop',
                 import_filename=None, import_filepath=''):
        """Initialize the Back-Propagation algorithm.

        Parameters
        ----------
        forward : ForwardSolver, default: mom.MoM_CG_FFT()
            Forward solver object used to compute incident fields and Green's functions.
        alias : str, default: 'backprop'
            Alias name for the algorithm instance.
        import_filename : str, optional
            Filename to import previously saved algorithm state.
        import_filepath : str, default: ''
            Path to the file containing saved algorithm state.
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(alias=alias, parallelization=False)
            self.name = 'Back-Propagation'
            self.forward = forward

    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout):
        """Solve the inverse scattering problem using Back-Propagation.

        This method implements the Back-Propagation algorithm to reconstruct
        the contrast function from scattered field measurements. The algorithm
        computes the solution in a single step using Green's function relationships.

        Parameters
        ----------
        inputdata : InputData
            Input data object containing scattered field measurements and
            configuration parameters.
        discretization : Discretization
            Discretization object containing domain information and Green's
            function matrix.
        print_info : bool, default: True
            Whether to print algorithm progress information.
        print_file : file-like object, default: sys.stdout
            File object to write progress information to.

        Returns
        -------
        result : Result
            Result object containing the reconstructed contrast, fields,
            and error metrics.
        """
        result = super().solve(inputdata, discretization,
                               print_info=print_info, print_file=print_file)

        NY, NX = discretization.elements

        tic = tm.time()
        incident_field = self.forward.incident_field(discretization.elements,
                                                     inputdata.configuration)
        gamma = compute_gamma(inputdata.scattered_field,
                              discretization.GS)
        current = compute_current(inputdata.scattered_field, discretization.GS,
                                  gamma)
        prod = fftproduct.FFTProduct(discretization)
        total_field = incident_field + prod.compute(current)
        contrast = compute_contrast(total_field, current, NX, NY)
        execution_time = tm.time()-tic


        scattered_field = discretization.scattered_field(
            contrast=contrast, total_field=total_field
        )
        contrast = discretization.contrast_image(contrast,
                                                 inputdata.resolution)

        result.update_error(inputdata, scattered_field=scattered_field,
                            total_field=total_field, contrast=contrast)
        result.scattered_field = scattered_field
        result.total_field = total_field

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
        if rst.EXECUTION_TIME in inputdata.indicators:
            result.execution_time = execution_time
        if rst.NUMBER_ITERATIONS in inputdata.indicators:
            result.number_iterations = 1

        return result

    def save(self, file_path=''):
        """Save the Back-Propagation algorithm state to file.

        Parameters
        ----------
        file_path : str, default: ''
            Path where to save the algorithm state file.
        """
        data = super().save(file_path=file_path)
        data[FORWARD] = self.forward
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        """Import Back-Propagation algorithm state from file.

        Parameters
        ----------
        file_name : str
            Name of the file containing saved algorithm state.
        file_path : str, default: ''
            Path to the file containing saved algorithm state.
        """
        data = super().importdata(file_name, file_path=file_path)
        self.forward = data[FORWARD]

    def copy(self, new=None):
        """Create a copy of the Back-Propagation algorithm.

        Parameters
        ----------
        new : BackPropagation, optional
            Existing BackPropagation object to copy attributes to.
            If None, creates a new instance.

        Returns
        -------
        BackPropagation or None
            If new is None, returns a new BackPropagation instance.
            If new is provided, modifies it in place and returns None.
        """
        if new is None:
            return BackPropagation(self.forward, self.alias)
        else:
            super().copy(new)
            self.forward = new.forward

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        """Print algorithm title and configuration information.

        Parameters
        ----------
        inputdata : InputData
            Input data object containing problem configuration.
        discretization : Discretization
            Discretization object containing domain information.
        print_file : file-like object, default: sys.stdout
            File object to write information to.
        """
        super()._print_title(inputdata, discretization, print_file=print_file)
        print(self.forward, file=print_file)

    def __str__(self):
        """Return string representation of the Back-Propagation algorithm.

        Returns
        -------
        str
            String description including algorithm name and forward solver.
        """
        message = super().__str__()
        message += str(self.forward)
        return message

@jit(nopython=True)
def compute_gamma(Es, GS):
    """Compute gamma parameter for Back-Propagation algorithm.

    Computes the gamma parameter used in the Back-Propagation algorithm,
    which represents the optimal scaling factor for the current density
    estimation.

    Parameters
    ----------
    Es : numpy.ndarray
        Scattered field measurements matrix. Shape (NM, NS) where NM is
        the number of measurement points and NS is the number of sources.
    GS : numpy.ndarray
        Green's function matrix from sources to measurement points.
        Shape (NM, ND) where ND is the number of domain points.

    Returns
    -------
    numpy.ndarray
        Gamma parameter array with shape (NS,).
    """
    aux = GS @ GS.T.conjugate() @ Es
    num = np.sum(Es * np.conjugate(aux), axis=0)
    dem = np.sum(np.abs(aux)**2, axis=0)
    return num/dem

@jit(nopython=True)
def compute_current(Es, GS, gamma):
    """Compute current density from scattered field measurements.

    Computes the current density in the scattering domain using the
    Back-Propagation algorithm. The current density is estimated using
    the adjoint of the Green's function matrix scaled by the gamma parameter.

    Parameters
    ----------
    Es : numpy.ndarray
        Scattered field measurements matrix. Shape (NM, NS) where NM is
        the number of measurement points and NS is the number of sources.
    GS : numpy.ndarray
        Green's function matrix from sources to measurement points.
        Shape (NM, ND) where ND is the number of domain points.
    gamma : numpy.ndarray
        Gamma scaling parameter array with shape (NS,).

    Returns
    -------
    numpy.ndarray
        Current density matrix with shape (ND, NS).
    """
    aux = 0j*np.ones((GS.shape[1], gamma.size))
    for n in range(GS.shape[1]):
        aux[n, :] = gamma
    return aux * (GS.T.conjugate()  @ Es)
    # return np.tile(gamma, (GS.shape[1], 1)) * (GS.T.conjugate()  @ Es)

@jit(nopython=True)
def compute_contrast(E, J, NX, NY):
    """Compute contrast function from total field and current density.

    Computes the contrast function (relative permittivity or conductivity
    contrast) using the relationship between total field and current density
    in the scattering domain.

    Parameters
    ----------
    E : numpy.ndarray
        Total field matrix with shape (ND, NS) where ND is the number
        of domain points and NS is the number of sources.
    J : numpy.ndarray
        Current density matrix with shape (ND, NS).
    NX : int
        Number of pixels in the x-direction.
    NY : int
        Number of pixels in the y-direction.

    Returns
    -------
    numpy.ndarray
        Contrast function matrix with shape (NY, NX).
    """
    Econj = np.conj(E)
    num = np.sum(J * Econj, 1)
    den = np.sum(Econj * E, 1)
    return np.reshape(num/den, (NY, NX))