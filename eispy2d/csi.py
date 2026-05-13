r"""
Contrast Source Inversion Method for Electromagnetic Inverse Scattering

This module implements the Contrast Source Inversion (CSI) method for solving
nonlinear electromagnetic inverse scattering problems. The CSI method is an
iterative algorithm that simultaneously reconstructs both the contrast function
and the contrast sources (induced currents) in the investigation domain.

The method is based on minimizing a cost functional that includes both data
error (difference between measured and computed scattered fields) and object
error (consistency between contrast and current distributions). The algorithm
uses conjugate gradient optimization to efficiently solve the nonlinear
inverse problem.

Classes
-------
ContrastSourceInversion : Class implementing the CSI method for nonlinear inverse scattering.

Functions
---------
Optimized computational functions (JIT-compiled with Numba):
get_data_error(Es, GS, J) : Compute data error term
get_object_error(chi, E, J) : Compute object error term
get_normalization_s(Es) : Compute scattered field normalization
get_normalization_d(chi, Ei) : Compute domain normalization
get_gradient(GS, rho, eta_s, r, GDaXr, eta_d) : Compute gradient
get_gamma(g, glast) : Compute conjugate gradient coefficient
update_direction(g, gamma, v) : Update search direction
compute_constant(rho, GSv, eta_s, r, v_XGDv, eta_d) : Compute step size
update_current(J, alpha, v) : Update current distribution
update_total_field(E, alpha, GDv) : Update total field
compute_contrast(J, E) : Compute contrast from current and field
evaluate_objective_function(rho, eta_s, r, eta_d) : Evaluate cost function

Constants
---------
FORWARD : str
    Dictionary key for forward solver
STOP_CRITERIA : str
    Dictionary key for stopping criteria

Notes
-----
The CSI method minimizes the following cost functional:

.. math::
    F = F_S + F_D = \frac{\|\mathbf{E}^s - \mathbf{G}^s \mathbf{J}\|^2}{\|\mathbf{E}^s\|^2} + 
    \frac{\|\chi \mathbf{E} - \mathbf{J}\|^2}{\|\chi \mathbf{E}^i\|^2}

where :math:`F_S` is the data error term, :math:`F_D` is the object error term,
:math:`\mathbf{J}` is the contrast source, :math:`\chi` is the contrast function,
and :math:`\mathbf{E}` is the total electric field.

The method uses conjugate gradient optimization with Polak-Ribière updates
for efficient convergence to the solution.

References
----------
.. [1] van den Berg, Peter M., and Roy E. Kleinman. "A contrast source
   inversion method." Inverse problems 13.6 (1997): 1607.
.. [2] Abubakar, Aria, et al. "A robust iterative method for Born inversion."
   IEEE Transactions on Geoscience and Remote Sensing 42.2 (2004): 342-354.

Examples
--------
>>> # Create CSI method with MoM forward solver
>>> csi = ContrastSourceInversion(stop_criteria=my_criteria)
>>> result = csi.solve(input_data, discretization)

>>> # Create CSI with custom forward solver
>>> csi = ContrastSourceInversion(stop_criteria=my_criteria,
...                               forward_solver=my_forward_solver)
>>> result = csi.solve(input_data, discretization)
"""

# Standard libraries
import time as tm
import numpy as np
import sys
import pickle
from numba import jit

# Developed libraries
import eispy2d.configuration as cfg
import eispy2d.inputdata as ipt
import eispy2d.result as rst
import eispy2d.deterministic as dtm
import eispy2d.collocation as clc
import eispy2d.mom_cg_fft as mom
import eispy2d.regularization as reg
import eispy2d.backprop as bp
import eispy2d.fftproduct as fftproduct

FORWARD = 'forward'
STOP_CRITERIA = 'stop criteria'


class ContrastSourceInversion(dtm.Deterministic):
    r"""
    Contrast Source Inversion method for nonlinear electromagnetic inverse scattering.
    
    This class implements the Contrast Source Inversion (CSI) method, which
    simultaneously reconstructs both the contrast function and the contrast
    sources (induced currents) in electromagnetic inverse scattering problems.
    The method uses an iterative conjugate gradient optimization approach to
    minimize a cost functional with both data and object error terms.
    
    The CSI method solves the nonlinear inverse problem by minimizing:
    
    .. math::
        F = F_S + F_D = \frac{\|\mathbf{E}^s - \mathbf{G}^s \mathbf{J}\|^2}{\|\mathbf{E}^s\|^2} + 
        \frac{\|\chi \mathbf{E} - \mathbf{J}\|^2}{\|\chi \mathbf{E}^i\|^2}
    
    where:
    - :math:`F_S` is the data error term (scattered field misfit)
    - :math:`F_D` is the object error term (current-contrast consistency)
    - :math:`\mathbf{J}` is the contrast source (induced current)
    - :math:`\chi` is the contrast function
    - :math:`\mathbf{E}` is the total electric field
    - :math:`\mathbf{E}^s` is the scattered field
    - :math:`\mathbf{E}^i` is the incident field
    - :math:`\mathbf{G}^s` is the scattered field Green's function
    
    Parameters
    ----------
    stop_criteria : object
        Stopping criteria object defining convergence conditions
        (maximum iterations, error tolerance, etc.)
    forward_solver : object, default=mom.MoM_CG_FFT()
        Forward solver implementation for computing electromagnetic fields
    alias : str, default='csi'
        Alias name for the method used in saving/loading
    import_filename : str, optional
        Filename to import method parameters from
    import_filepath : str, default=''
        Path to import file
    
    Attributes
    ----------
    name : str
        Human-readable name of the method
    forward : object
        Forward solver for electromagnetic field computation
    stop_criteria : object
        Stopping criteria configuration
    
    Methods
    -------
    solve(inputdata, discretization, print_info=True, print_file=sys.stdout, initial_guess=None)
        Solve the inverse scattering problem using CSI
    save(file_path='')
        Save method configuration to file
    importdata(file_name, file_path='')
        Import method configuration from file
    copy(new=None)
        Create a copy of the method
    
    Notes
    -----
    The CSI method offers several advantages:
    - Simultaneous reconstruction of contrast and sources
    - Robust convergence properties
    - Efficient conjugate gradient optimization
    - Automatic field normalization for stability
    
    The algorithm uses the Polak-Ribière conjugate gradient formula:
    
    .. math::
        \gamma^{(k+1)} = \frac{(\mathbf{g}^{(k+1)} - \mathbf{g}^{(k)})^H \mathbf{g}^{(k+1)}}{\|\mathbf{g}^{(k)}\|^2}
    
    where :math:`\mathbf{g}^{(k)}` is the gradient at iteration k.
    
    References
    ----------
    .. [1] van den Berg, Peter M., and Roy E. Kleinman. "A contrast source
       inversion method." Inverse problems 13.6 (1997): 1607.
    .. [2] Abubakar, Aria, et al. "A robust iterative method for Born inversion."
       IEEE Transactions on Geoscience and Remote Sensing 42.2 (2004): 342-354.
    
    Examples
    --------
    >>> # Basic usage with default forward solver
    >>> csi = ContrastSourceInversion(stop_criteria=my_stop_criteria)
    >>> result = csi.solve(input_data, discretization)
    
    >>> # Using custom forward solver
    >>> csi = ContrastSourceInversion(stop_criteria=my_stop_criteria,
    ...                               forward_solver=my_forward_solver)
    >>> result = csi.solve(input_data, discretization)
    
    >>> # Import from saved configuration
    >>> csi = ContrastSourceInversion(stop_criteria=my_stop_criteria,
    ...                               import_filename='csi_config.pkl')
    """

    def __init__(self, stop_criteria, forward_solver=mom.MoM_CG_FFT(),
                 alias='csi', import_filename=None, import_filepath=''):
        r"""
        Initialize the Contrast Source Inversion method.
        
        Creates a new CSI instance with specified stopping criteria and
        forward solver for electromagnetic field computation.
        
        Parameters
        ----------
        stop_criteria : object
            Stopping criteria object that defines convergence conditions
            such as maximum iterations, error tolerance, or divergence limits
        forward_solver : object, default=mom.MoM_CG_FFT()
            Forward solver implementation for computing electromagnetic fields.
            Must implement the incident_field() method and support the
            required discretization formats
        alias : str, default='csi'
            Alias name for the method used in file operations and identification
        import_filename : str, optional
            If provided, import method parameters from this file instead
            of using the provided parameters
        import_filepath : str, default=''
            Path to the import file
            
        Examples
        --------
        >>> # Create CSI with default forward solver
        >>> csi = ContrastSourceInversion(stop_criteria=my_criteria)
        
        >>> # Create CSI with custom forward solver
        >>> csi = ContrastSourceInversion(stop_criteria=my_criteria,
        ...                               forward_solver=my_forward_solver)
        
        >>> # Import from saved configuration
        >>> csi = ContrastSourceInversion(stop_criteria=my_criteria,
        ...                               import_filename='csi_config.pkl')
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(alias=alias, parallelization=None)
            self.name = 'Contrast Source Inversion'
            self.forward = forward_solver
            self.stop_criteria = stop_criteria

    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout, initial_guess=None):
        """
        Solve the nonlinear inverse scattering problem using CSI.
        
        Applies the Contrast Source Inversion method to reconstruct the
        electromagnetic properties of unknown scatterers. The method
        simultaneously updates both the contrast function and contrast
        sources using conjugate gradient optimization.
        
        Parameters
        ----------
        inputdata : inputdata.InputData
            Input data object containing:
            - scattered_field: Measured scattered electric field
            - configuration: Problem configuration (frequency, geometry, etc.)
            - resolution: Target resolution for reconstruction
            - indicators: List of performance indicators to compute
        discretization : object
            Discretization object containing:
            - elements: Grid dimensions (NY, NX)
            - GS: Green's function matrix for scattered field
            - contrast_image: Method for contrast imaging
            - total_image: Method for total field imaging
            - solve: Method for solving linear systems
        print_info : bool, default=True
            Whether to print iteration information during solving
        print_file : file-like object, default=sys.stdout
            File object for printing iteration information
        initial_guess : numpy.ndarray, optional
            Initial guess for the contrast function. If None, uses
            backpropagation method for initialization
            
        Returns
        -------
        result.Result
            Result object containing:
            - rel_permittivity: Reconstructed relative permittivity
            - conductivity: Reconstructed conductivity (if applicable)
            - scattered_field: Reconstructed scattered field
            - total_field: Reconstructed total field
            - total_error: Final objective function value
            - data_error: Data error evolution
            - execution_time: Total execution time
            - number_iterations: Number of iterations performed
            - number_evaluations: Number of function evaluations
            
        Notes
        -----
        The CSI algorithm performs the following steps:
        
        1. **Initialization**: Compute initial contrast and current using
           backpropagation or provided initial guess
        2. **Data Error**: Compute :math:`\\boldsymbol{\\rho} = \\mathbf{E}^s - \\mathbf{G}^s \\mathbf{J}`
        3. **Object Error**: Compute :math:`\\mathbf{r} = \\chi \\mathbf{E} - \\mathbf{J}`
        4. **Objective Function**: Evaluate :math:`F = F_S + F_D`
        5. **Gradient**: Compute gradient with respect to current :math:`\\mathbf{J}`
        6. **Conjugate Direction**: Update search direction using Polak-Ribière
        7. **Step Size**: Compute optimal step size minimizing cost function
        8. **Update**: Update current, total field, and contrast
        9. **Convergence**: Check stopping criteria and repeat if needed
        
        The method uses FFT-based products for efficient computation of
        domain interactions and automatic normalization for numerical stability.
        
        Algorithm features:
        - **Simultaneous reconstruction**: Both contrast and sources updated
        - **Robust convergence**: Conjugate gradient with automatic step size
        - **Efficient computation**: FFT-based operations for large problems
        - **Flexible initialization**: Backpropagation or custom initial guess
        
        Examples
        --------
        >>> # Basic usage with automatic initialization
        >>> csi = ContrastSourceInversion(stop_criteria=my_criteria)
        >>> result = csi.solve(input_data, discretization)
        >>> print(f"Final error: {result.total_error}")
        
        >>> # Using custom initial guess
        >>> initial_contrast = np.random.random(discretization.elements)
        >>> result = csi.solve(input_data, discretization, 
        ...                   initial_guess=initial_contrast)
        
        >>> # Solve with custom output file
        >>> with open('csi_log.txt', 'w') as f:
        ...     result = csi.solve(input_data, discretization, 
        ...                       print_file=f)
        """
        result = super().solve(inputdata, discretization,
                               print_info=print_info, print_file=print_file)

        # First-Order Born Approximation
        tic = tm.time()
        if initial_guess is None:
            contrast, chi, current = self._get_initial_guess(inputdata,
                                                             discretization)
        else:
            contrast = discretization.contrast_image(initial_guess,
                                                     discretization.elements)
            chi = np.diag(contrast.flatten(), 0)
            regularization = reg.LeastSquares(cutoff=1e-5)
            current = discretization.solve(
                scattered_field=inputdata.scattered_field,
                linear_solver=regularization
            )
        execution_time = tm.time()-tic

        fftp = fftproduct.FFTProduct(discretization=discretization,
                                     adjoint=False)
        fftpa = fftproduct.FFTProduct(discretization=discretization,
                                      adjoint=True)

        # If the same object is used for different resolution instances,
        # then some parameters may need to be updated within the inverse
        # solver. So, the next line ensures it:
        current_evaluations = 0
        iteration = 0
        objective_function = np.inf
        base, power = 1, 0

        N, NS = np.prod(discretization.elements), inputdata.configuration.NS
        direction = np.zeros((N, NS), dtype=complex)
        last_gradient = np.ones((N, NS), dtype=complex)
        incident_field = self.forward.incident_field(discretization.elements,
                                                     inputdata.configuration)
        normalization_s = get_normalization_s(inputdata.scattered_field)

        while (not self.stop_criteria.stop(current_evaluations, iteration,
                                           objective_function)):

            iteration_message = 'Iteration: %d - ' % (iteration+1)

            tic = tm.time()
            data_error = self._get_data_error(inputdata.scattered_field,
                                              discretization.GS, current)
            total_field = incident_field + fftp.compute(current)
            object_error = self._get_object_error(chi, total_field, current)
            normalization_d = get_normalization_d(chi, incident_field)
            objective_function = self._evaluate_objective_function(
                data_error, normalization_s, object_error, normalization_d
            )
            gradient = self._get_gradient(discretization.GS, data_error,
                                          normalization_s, object_error, fftpa,
                                          chi, normalization_d)
            gamma = get_gamma(gradient, last_gradient)
            direction = self._update_direction(gradient, gamma, direction)
            constant = self._get_constant(data_error, discretization.GS,
                                          direction, normalization_s,
                                          object_error, chi, fftp,
                                          normalization_d)
            current = update_current(current, constant, direction)
            total_field = self._update_total_field(total_field, constant, fftp,
                                                   direction)
            contrast = self._compute_contrast(current, total_field)
            chi = np.diag(contrast.flatten(), 0) + 0j
            contrast = contrast.reshape(discretization.elements)
            last_gradient = gradient.copy()
            execution_time +=  tm.time()-tic
            contrast = np.diag(chi, 0)
            contrast = discretization.contrast_image(contrast,
                                                     inputdata.resolution)

            if inputdata.configuration.good_conductor:
                contrast = 1j*contrast.imag
            if inputdata.configuration.perfect_dielectric:
                contrast = contrast.real

            result.update_error(
                inputdata,
                scattered_field=data_error-inputdata.scattered_field,
                total_field=discretization.total_image(total_field,
                                                       inputdata.resolution),
                contrast=contrast, objective_function=objective_function
            )

            if print_info:
                if iteration+1 >= base*10**power:
                    if base == 9:
                        base = 1
                        power += 1
                    else:
                        base += 1
                    iteration_message = result.last_error_message(
                        iteration_message
                    )
                    print(iteration_message, file=print_file)
            current_evaluations += 1
            iteration += 1

        if print_info and iteration != base*10**power:
            iteration_message = result.last_error_message(iteration_message)
            print(iteration_message, file=print_file)

        # Remember: results stores the estimated scattered field. Not
        # the given one.
        result.scattered_field = data_error-inputdata.scattered_field
        result.total_field = total_field

        if not inputdata.configuration.good_conductor:
            result.rel_permittivity = cfg.get_relative_permittivity(
                contrast, inputdata.configuration.epsilon_rb
            )
        if not inputdata.configuration.perfect_dielectric:
            result.conductivity = cfg.get_conductivity(
                contrast, 2*np.pi*inputdata.configuration.f,
                inputdata.configuration.epsilon_rb,
                inputdata.configuration.sigma_b
            )
        if rst.EXECUTION_TIME in inputdata.indicators:
            result.execution_time = execution_time
        if rst.NUMBER_ITERATIONS in inputdata.indicators:
            result.number_iterations = iteration
        if rst.NUMBER_EVALUATIONS in inputdata.indicators:
            result.number_evaluations = current_evaluations

        return result

    def _get_initial_guess(self, inputdata, discretization):
        """
        Generate initial guess using backpropagation method.
        
        Computes initial estimates for contrast function and contrast sources
        using the backpropagation algorithm as a starting point for the CSI
        iterative process.
        
        Parameters
        ----------
        inputdata : inputdata.InputData
            Input data object containing problem configuration
        discretization : object
            Discretization object containing grid information
            
        Returns
        -------
        tuple
            (contrast, chi, current) where:
            - contrast: Initial contrast function estimate
            - chi: Diagonal matrix form of contrast
            - current: Initial contrast source estimate
        """
        initial_guess = bp.BackPropagation()
        temporary = inputdata.copy()
        temporary.resolution = discretization.elements
        temporary.indicators = []
        initial_guess = initial_guess.solve(temporary, discretization,
                                            print_info=False)
        contrast = cfg.get_contrast_map(
            epsilon_r=initial_guess.rel_permittivity,
            configuration=inputdata.configuration
        )
        chi = np.diag(contrast.flatten(), 0) + 0j
        current = chi @ initial_guess.total_field
        contrast = discretization.contrast_image(contrast,
                                                 inputdata.resolution)
        return contrast, chi, current

    def _get_object_error(self, chi, total_field, current):
        """
        Compute object error term for CSI cost function.
        
        Parameters
        ----------
        chi : numpy.ndarray
            Contrast function diagonal matrix
        total_field : numpy.ndarray
            Total electric field
        current : numpy.ndarray
            Contrast source (induced current)
            
        Returns
        -------
        numpy.ndarray
            Object error term (chi * E - J)
        """
        return get_object_error(chi, total_field, current)

    def _get_data_error(self, scattered_field, green_function_s, current):
        """
        Compute data error term for CSI cost function.
        
        Parameters
        ----------
        scattered_field : numpy.ndarray
            Measured scattered electric field
        green_function_s : numpy.ndarray
            Scattered field Green's function matrix
        current : numpy.ndarray
            Contrast source (induced current)
            
        Returns
        -------
        numpy.ndarray
            Data error term (Es - GS * J)
        """
        return get_data_error(scattered_field, green_function_s, current)

    def _get_gradient(self, green_function_s, data_error, normalization_s,
                      object_error, fftpa, chi, normalization_d):
        """
        Compute gradient of CSI cost function.
        
        Parameters
        ----------
        green_function_s : numpy.ndarray
            Scattered field Green's function matrix
        data_error : numpy.ndarray
            Data error term
        normalization_s : float
            Scattered field normalization factor
        object_error : numpy.ndarray
            Object error term
        fftpa : object
            Adjoint FFT product operator
        chi : numpy.ndarray
            Contrast function diagonal matrix
        normalization_d : float
            Domain normalization factor
            
        Returns
        -------
        numpy.ndarray
            Gradient with respect to contrast source
        """
        GDaXr = fftpa.compute(np.conj(chi) @ object_error)
        return get_gradient(green_function_s, data_error, normalization_s,
                            object_error, GDaXr, normalization_d)

    def _update_direction(self, gradient, gamma, direction):
        """
        Update conjugate gradient search direction.
        
        Parameters
        ----------
        gradient : numpy.ndarray
            Current gradient
        gamma : numpy.ndarray
            Conjugate gradient coefficient
        direction : numpy.ndarray
            Previous search direction
            
        Returns
        -------
        numpy.ndarray
            Updated search direction
        """
        N = gradient.shape[0]
        gamma = np.tile(gamma.reshape((1, -1)), (N, 1))
        return update_direction(gradient, gamma, direction)

    def _get_constant(self, data_error, green_function_s, direction,
                      normalization_s, object_error, chi, fftp,
                      normalization_d):
        """
        Compute optimal step size for CSI update.
        
        Parameters
        ----------
        data_error : numpy.ndarray
            Data error term
        green_function_s : numpy.ndarray
            Scattered field Green's function matrix
        direction : numpy.ndarray
            Search direction
        normalization_s : float
            Scattered field normalization factor
        object_error : numpy.ndarray
            Object error term
        chi : numpy.ndarray
            Contrast function diagonal matrix
        fftp : object
            FFT product operator
        normalization_d : float
            Domain normalization factor
            
        Returns
        -------
        numpy.ndarray
            Optimal step size constants
        """
        N = green_function_s.shape[1]
        XGDv = chi @ fftp.compute(direction)
        v_XGDv = direction - XGDv
        GSv = green_function_s @ direction
        constant = compute_constant(data_error, GSv, normalization_s,
                                    object_error, v_XGDv, normalization_d)
        return np.tile(constant.reshape((1, -1)), (N, 1))

    def _update_total_field(self, total_field, constant, fftp, direction):
        """
        Update total electric field using optimization step.
        
        This method updates the total electric field by adding the
        optimal step size times the search direction convolved with
        the Green's function.
        
        Parameters
        ----------
        total_field : numpy.ndarray
            Current total electric field
        constant : numpy.ndarray
            Optimal step size constants
        fftp : object
            FFT product operator for Green's function convolution
        direction : numpy.ndarray
            Search direction
            
        Returns
        -------
        numpy.ndarray
            Updated total electric field
        """
        GDv = fftp.compute(direction)
        return update_total_field(total_field, constant, GDv)

    def _compute_contrast(self, current, total_field):
        """
        Compute contrast function from current and total field.
        
        This method computes the contrast function (relative permittivity
        minus 1) using the current density and total electric field.
        
        Parameters
        ----------
        current : numpy.ndarray
            Current density distribution
        total_field : numpy.ndarray
            Total electric field
            
        Returns
        -------
        numpy.ndarray
            Complex contrast function
        """
        return compute_contrast(current, total_field) + 0j

    def _evaluate_objective_function(self, data_error, normalization_s,
                                     object_error, normalization_d):
        """
        Evaluate CSI objective function.
        
        This method computes the total cost function value as the sum
        of data fidelity and object regularization terms.
        
        Parameters
        ----------
        data_error : numpy.ndarray
            Data error term (scattered field residual)
        normalization_s : float
            Scattered field normalization factor
        object_error : numpy.ndarray
            Object error term (state equation residual)
        normalization_d : float
            Domain normalization factor
            
        Returns
        -------
        float
            Total objective function value
        """
        return evaluate_objective_function(data_error, normalization_s,
                                           object_error, normalization_d)

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        """
        Print algorithm title and configuration.
        
        This method prints the CSI algorithm title along with forward
        solver and stopping criteria information.
        
        Parameters
        ----------
        inputdata : InputData
            Input data object containing problem configuration
        discretization : Discretization
            Discretization object containing mesh information
        print_file : file-like object, optional
            Output file stream (default: sys.stdout)
        """
        super()._print_title(inputdata, discretization, print_file=print_file)
        print(self.forward, file=print_file)
        print(self.stop_criteria, file=print_file)

    def save(self, file_path=''):
        """
        Save CSI algorithm state to file.
        
        This method saves the complete algorithm state including forward
        solver configuration, stopping criteria, and inherited solver data.
        
        Parameters
        ----------
        file_path : str, optional
            Base path for saving algorithm state files
            
        Returns
        -------
        dict
            Dictionary containing algorithm state data
        """
        data = super().save(file_path=file_path)
        data[FORWARD] = self.forward
        data[STOP_CRITERIA] = self.stop_criteria
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)
        return data

    def importdata(self, file_name, file_path=''):
        """
        Import CSI algorithm state from file.
        
        This method loads the complete algorithm state including forward
        solver configuration and stopping criteria from a saved file.
        
        Parameters
        ----------
        file_name : str
            Name of the file containing algorithm state
        file_path : str, optional
            Path to the directory containing the state file
            
        Returns
        -------
        dict
            Dictionary containing imported algorithm state data
        """
        data = super().importdata(file_name, file_path=file_path)
        self.forward = data[FORWARD]
        self.stop_criteria= data[STOP_CRITERIA]

    def copy(self, new=None):
        """
        Create a copy of the CSI algorithm instance.
        
        This method creates a deep copy of the CSI algorithm with the same
        configuration parameters.
        
        Parameters
        ----------
        new : ContrastSourceInversion, optional
            Existing instance to copy configuration into
            
        Returns
        -------
        ContrastSourceInversion or None
            New algorithm instance if new is None, otherwise None
        """
        if new is None:
            return ContrastSourceInversion(self.stop_criteria,
                                           forward_solver=self.forward,
                                           alias=self.alias)
        else:
            super().copy(new)
            new.forward = self.forward
            new.stop_criteria = self.stop_criteria

    def __str__(self):
        """
        String representation of the CSI algorithm.
        
        Returns
        -------
        str
            Human-readable string representation including forward solver
            and stopping criteria information
        """
        message = super().__str__()
        message += str(self.forward)
        message += str(self.stop_criteria)
        return message


@jit(nopython=True)
def get_data_error(Es, GS, J):
    """
    Compute data error term for CSI algorithm.
    
    This function computes the data error as the difference between
    the measured scattered field and the forward model prediction.
    
    Parameters
    ----------
    Es : numpy.ndarray
        Measured scattered electric field
    GS : numpy.ndarray
        Scattered field Green's function matrix
    J : numpy.ndarray
        Current contrast source
        
    Returns
    -------
    numpy.ndarray
        Data error term
    """
    return Es - GS @ J

@jit(nopython=True)
def get_object_error(chi, E, J):
    """
    Compute object error term for CSI algorithm.
    
    This function computes the object error as the difference between
    the contrast times the total field and the contrast source.
    
    Parameters
    ----------
    chi : numpy.ndarray
        Contrast function diagonal matrix
    E : numpy.ndarray
        Total electric field
    J : numpy.ndarray
        Current contrast source
        
    Returns
    -------
    numpy.ndarray
        Object error term
    """
    return chi @ E - J

@jit(nopython=True)
def get_normalization_s(Es):
    """
    Compute scattered field normalization factor.
    
    This function computes the normalization factor for the scattered
    field as the sum of squared magnitudes.
    
    Parameters
    ----------
    Es : numpy.ndarray
        Scattered electric field
        
    Returns
    -------
    float
        Scattered field normalization factor
    """
    return np.sum(np.abs(Es)**2)

@jit(nopython=True)
def get_normalization_d(chi, Ei):
    """
    Compute domain normalization factor.
    
    This function computes the normalization factor for the domain
    as the sum of squared magnitudes of the contrast times incident field.
    
    Parameters
    ----------
    chi : numpy.ndarray
        Contrast function diagonal matrix
    Ei : numpy.ndarray
        Incident electric field
        
    Returns
    -------
    float
        Domain normalization factor
    """
    return np.sum(np.abs(chi @ Ei)**2)

@jit(nopython=True)
def get_gradient(GS, rho, eta_s, r, GDaXr, eta_d):
    """
    Compute gradient for CSI optimization.
    
    This function computes the gradient of the CSI objective function
    with respect to the contrast source.
    
    Parameters
    ----------
    GS : numpy.ndarray
        Scattered field Green's function matrix
    rho : numpy.ndarray
        Data residual
    eta_s : float
        Scattered field normalization factor
    r : numpy.ndarray
        Object residual
    GDaXr : numpy.ndarray
        Green's function convolution result
    eta_d : float
        Domain normalization factor
        
    Returns
    -------
    numpy.ndarray
        Gradient vector
    """
    return - GS.conj().T @ rho / eta_s - (r - GDaXr) / eta_d

@jit(nopython=True)
def get_gamma(g, glast):
    """
    Compute Polak-Ribiere conjugate gradient parameter.
    
    This function computes the beta parameter for the Polak-Ribiere
    conjugate gradient method.
    
    Parameters
    ----------
    g : numpy.ndarray
        Current gradient
    glast : numpy.ndarray
        Previous gradient
        
    Returns
    -------
    numpy.ndarray
        Beta parameter for direction update
    """
    return (np.sum(g * np.conj(g-glast), axis=0)
            / np.sum(glast * np.conj(glast), axis=0))

@jit(nopython=True)
def update_direction(g, gamma, v):
    """
    Update search direction using conjugate gradient.
    
    This function updates the search direction using the current
    gradient and the conjugate gradient parameter.
    
    Parameters
    ----------
    g : numpy.ndarray
        Current gradient
    gamma : numpy.ndarray
        Conjugate gradient parameter
    v : numpy.ndarray
        Previous search direction
        
    Returns
    -------
    numpy.ndarray
        Updated search direction
    """
    return g + gamma*v

@jit(nopython=True)
def compute_constant(rho, GSv, eta_s, r, v_XGDv, eta_d):
    """
    Compute optimal step size for CSI update.
    
    This function computes the optimal step size that minimizes
    the CSI objective function along the search direction.
    
    Parameters
    ----------
    rho : numpy.ndarray
        Data residual
    GSv : numpy.ndarray
        Scattered field Green's function times direction
    eta_s : float
        Scattered field normalization factor
    r : numpy.ndarray
        Object residual
    v_XGDv : numpy.ndarray
        Direction minus contrast convolution result
    eta_d : float
        Domain normalization factor
        
    Returns
    -------
    numpy.ndarray
        Optimal step size
    """
    t1 = np.sum(rho * np.conj(GSv), axis=0)/eta_s
    t2 = np.sum(r * np.conj(v_XGDv), axis=0)/eta_d
    t3 = np.sum(np.abs(GSv)**2)/eta_s
    t4 = np.sum(np.abs(v_XGDv)**2)/eta_d
    return (t1 + t2)/(t3 + t4)

@jit(nopython=True)
def update_current(J, alpha, v):
    """
    Update contrast source with optimal step size.
    
    This function updates the contrast source by taking an optimal
    step along the search direction.
    
    Parameters
    ----------
    J : numpy.ndarray
        Current contrast source
    alpha : numpy.ndarray
        Optimal step size
    v : numpy.ndarray
        Search direction
        
    Returns
    -------
    numpy.ndarray
        Updated contrast source
    """
    return J + alpha * v

@jit(nopython=True)
def update_total_field(E, alpha, GDv):
    """
    Update total electric field with optimal step size.
    
    This function updates the total electric field by taking an optimal
    step along the direction of the Green's function convolution.
    
    Parameters
    ----------
    E : numpy.ndarray
        Current total electric field
    alpha : numpy.ndarray
        Optimal step size
    GDv : numpy.ndarray
        Green's function convolution of search direction
        
    Returns
    -------
    numpy.ndarray
        Updated total electric field
    """
    return E + alpha * GDv

@jit(nopython=True)
def compute_contrast(J, E):
    """
    Compute contrast function from contrast source and total field.
    
    This function computes the contrast function by solving the
    least squares problem for the relationship J = χE.
    
    Parameters
    ----------
    J : numpy.ndarray
        Contrast source
    E : numpy.ndarray
        Total electric field
        
    Returns
    -------
    numpy.ndarray
        Complex contrast function
    """
    den = np.sum(np.abs(E)**2, axis=1)
    num = J * np.conj(E)
    Xr = np.sum(np.real(num), axis=1)/den
    Xi = np.sum(np.imag(num), axis=1)/den
    return Xr + 1j*Xi

@jit(nopython=True)
def evaluate_objective_function(rho, eta_s, r, eta_d):
    """
    Evaluate the CSI objective function.
    
    This function computes the total cost function as the sum of
    normalized data fidelity and object constraint terms.
    
    Parameters
    ----------
    rho : numpy.ndarray
        Data residual term
    eta_s : float
        Scattered field normalization factor
    r : numpy.ndarray
        Object residual term
    eta_d : float
        Domain normalization factor
        
    Returns
    -------
    float
        Total objective function value
    """
    return np.sum(np.abs(rho)**2)/eta_s + np.sum(np.abs(r)**2)/eta_d