r"""
Conjugated Gradient Method for Electromagnetic Inverse Scattering

This module implements the Conjugated Gradient Method (CGM) for solving
nonlinear electromagnetic inverse scattering problems. The method is based
on gradient-based optimization techniques and uses the conjugate gradient
algorithm to iteratively reconstruct the electromagnetic properties of
scatterers from scattered field measurements.

The implementation includes various initialization strategies, step size
optimization methods, and stopping criteria for robust convergence.

Classes
-------
ConjugatedGradientMethod : Extends deterministic.Deterministic
    Main implementation of the conjugated gradient method

Constants
---------
INITIAL_GUESS : str
    Dictionary key for initial guess strategy
BACKGROUND : str
    Background initial guess strategy
BACKPROPAGATION : str
    Backpropagation initial guess strategy
IMAGE : str
    Image-based initial guess strategy
QUALITATIVE : str
    Qualitative initial guess strategy
STEP : str
    Dictionary key for step size method
FIXED : str
    Fixed step size method
OPTIMUM : str
    Optimum step size method
STOP_CRITERIA : str
    Dictionary key for stopping criteria

References
----------
.. [1] Lobel, P., et al. "Conjugate gradient method for solving inverse
   scattering with experimental data." IEEE Antennas and Propagation
   Magazine 38.3 (1996): 48-51.

Examples
--------
>>> # Create CGM with background initial guess
>>> cgm = ConjugatedGradientMethod(initial_guess='background',
...                                step='optimum',
...                                stop_criteria=my_criteria)
>>> result = cgm.solve(input_data, discretization)

>>> # Create CGM with qualitative initial guess
>>> cgm = ConjugatedGradientMethod(initial_guess='qualitative',
...                                step='fixed',
...                                stop_criteria=my_criteria)
>>> result = cgm.solve(input_data, discretization)
"""

# Standard libraries
import time as tm
import numpy as np
from scipy.linalg import norm, inv
from scipy import sparse as sps
from scipy import optimize as opt
import sys
import pickle

# Developed libraries
from eispy2d.core import configuration as cfg
from eispy2d.data import inputdata as ipt
from eispy2d.data import result as rst
from eispy2d.solvers.base import deterministic as dtm
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.solvers.inverse import osm

INITIAL_GUESS = 'initial_guess'
BACKGROUND = 'background'
BACKPROPAGATION = 'backpropagation'
IMAGE = 'image'
QUALITATIVE = 'qualitative'
STEP = 'step'
FIXED = 'fixed'
OPTIMUM = 'optimum'
STOP_CRITERIA = 'stop_criteria'

class ConjugatedGradientMethod(dtm.Deterministic):
    r"""
    Conjugated Gradient Method for nonlinear inverse scattering.
    
    This class implements the Conjugated Gradient Method (CGM) for solving
    nonlinear electromagnetic inverse scattering problems. The method uses
    gradient-based optimization with conjugate gradient updates to iteratively
    reconstruct the electromagnetic properties of unknown scatterers from
    scattered field measurements.
    
    The algorithm minimizes the following objective function:
    
    .. math::
        J(\\chi) = \|\mathbf{E}^s - \mathbf{G}^s \\chi \mathbf{L}^{-1} \mathbf{E}^i\|^2
    
    where :math:`\\chi` is the contrast function, :math:`\mathbf{E}^s` is the
    scattered field, :math:`\mathbf{G}^s` is the Green's function matrix,
    and :math:`\mathbf{L}^{-1}` is the inverse of the Lippmann-Schwinger
    operator.
    
    Parameters
    ----------
    initial_guess : str
        Strategy for initial guess:
        - 'background': Start with background medium
        - 'backpropagation': Use backpropagation algorithm
        - 'image': Use direct image reconstruction
        - 'qualitative': Use qualitative method (OSM)
    step : str
        Step size computation method:
        - 'fixed': Fixed step size based on gradient
        - 'optimum': Optimal step size via line search
    stop_criteria : object
        Stopping criteria object defining convergence conditions
    alias : str, default='cgm'
        Alias name for the method
    import_filename : str, optional
        Filename to import method parameters from
    import_filepath : str, default=''
        Path to import file
    
    Attributes
    ----------
    name : str
        Human-readable name of the method
    initial_guess : str
        Initial guess strategy
    step : str
        Step size computation method
    stop_criteria : object
        Stopping criteria configuration
    
    Methods
    -------
    solve(inputdata, discretization, print_info=True, print_file=sys.stdout)
        Solve the inverse scattering problem
    save(file_path='')
        Save method configuration to file
    importdata(file_name, file_path='')
        Import method configuration from file
    copy(new=None)
        Create a copy of the method
    
    Notes
    -----
    The conjugate gradient method is particularly effective for problems where
    the gradient computation is efficient. The method uses the Polak-Ribière
    formula for computing conjugate directions:
    
    .. math::
        \mathbf{d}^{(k+1)} = -\mathbf{g}^{(k+1)} + \beta^{(k+1)} \mathbf{d}^{(k)}
    
    where :math:`\beta^{(k+1)} = \frac{(\mathbf{g}^{(k+1)} - \mathbf{g}^{(k)})^T \mathbf{g}^{(k+1)}}{\|\mathbf{g}^{(k)}\|^2}`
    
    References
    ----------
    .. [1] Lobel, P., et al. "Conjugate gradient method for solving inverse
       scattering with experimental data." IEEE Antennas and Propagation
       Magazine 38.3 (1996): 48-51.
    .. [2] Nocedal, J., & Wright, S. J. (2006). Numerical optimization.
       Springer Science & Business Media.
    
    Examples
    --------
    >>> # Basic usage with background initial guess
    >>> cgm = ConjugatedGradientMethod(initial_guess='background',
    ...                                step='optimum',
    ...                                stop_criteria=my_stop_criteria)
    >>> result = cgm.solve(input_data, discretization)
    
    >>> # Using qualitative initial guess
    >>> cgm = ConjugatedGradientMethod(initial_guess='qualitative',
    ...                                step='fixed',
    ...                                stop_criteria=my_stop_criteria)
    >>> result = cgm.solve(input_data, discretization)
    
    >>> # Import from saved configuration
    >>> cgm = ConjugatedGradientMethod(import_filename='cgm_config.pkl')
    """

    def __init__(self, initial_guess, step, stop_criteria,
                 alias='cgm', import_filename=None, import_filepath=''):
        r"""
        Initialize the Conjugated Gradient Method.
        
        Creates a new instance of the CGM with specified initialization
        strategy, step size method, and stopping criteria.
        
        Parameters
        ----------
        initial_guess : str
            Strategy for initial guess:
            - 'background': Start with background medium (zero contrast)
            - 'backpropagation': Use backpropagation algorithm for initialization
            - 'image': Use direct image reconstruction as initial guess
            - 'qualitative': Use qualitative method (OSM) for initialization
        step : str
            Step size computation method:
            - 'fixed': Fixed step size based on gradient and residual
            - 'optimum': Optimal step size via line search optimization
        stop_criteria : object
            Stopping criteria object that defines convergence conditions
            (e.g., maximum iterations, error tolerance)
        alias : str, default='cgm'
            Alias name for the method used in saving/loading
        import_filename : str, optional
            If provided, import method parameters from this file
        import_filepath : str, default=''
            Path to the import file
            
        Examples
        --------
        >>> # Create CGM with background initial guess and optimal step
        >>> cgm = ConjugatedGradientMethod(initial_guess='background',
        ...                                step='optimum',
        ...                                stop_criteria=my_criteria)
        
        >>> # Create CGM with qualitative initial guess and fixed step
        >>> cgm = ConjugatedGradientMethod(initial_guess='qualitative',
        ...                                step='fixed',
        ...                                stop_criteria=my_criteria)
        
        >>> # Import from saved configuration
        >>> cgm = ConjugatedGradientMethod(initial_guess='background',
        ...                                step='optimum',
        ...                                stop_criteria=my_criteria,
        ...                                import_filename='cgm_config.pkl')
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(alias=alias, parallelization=None)
            self.name = 'Conjugated Gradient Method'
            self.initial_guess = initial_guess
            self.step = step
            self.stop_criteria = stop_criteria

    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout):
        r"""
        Solve the nonlinear inverse scattering problem using CGM.
        
        Applies the conjugated gradient method to iteratively reconstruct
        the electromagnetic properties of unknown scatterers from scattered
        field measurements. The method minimizes the data misfit using
        gradient-based optimization with conjugate gradient updates.
        
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
            - GD: Green's function matrix for domain interaction
        print_info : bool, default=True
            Whether to print iteration information during solving
        print_file : file-like object, default=sys.stdout
            File object for printing iteration information
            
        Returns
        -------
        result.Result
            Result object containing:
            - rel_permittivity: Reconstructed relative permittivity
            - conductivity: Reconstructed conductivity (if applicable)
            - total_error: Final data misfit error
            - data_error: Data error evolution
            - execution_time: Total execution time
            - number_iterations: Number of iterations performed
            - number_evaluations: Number of function evaluations
            
        Notes
        -----
        The algorithm implements the following steps:
        
        1. **Initialization**: Set initial contrast guess based on strategy
        2. **Gradient Computation**: Compute gradient of objective function
        3. **Conjugate Direction**: Update search direction using Polak-Ribière formula
        4. **Step Size**: Compute optimal or fixed step size
        5. **Update**: Update contrast function
        6. **Convergence Check**: Check stopping criteria
        
        The objective function being minimized is:
        
        .. math::
            J(\\chi) = \|\mathbf{E}^s - \mathbf{G}^s \\chi (\mathbf{I} - \mathbf{G}^d \\chi)^{-1} \mathbf{E}^i\|^2
        
        where :math:`\\chi` is the contrast function matrix.
        
        Initial guess strategies:
        - **background**: Zero contrast (background medium)
        - **backpropagation**: Backpropagation-based initialization
        - **image**: Direct image reconstruction
        - **qualitative**: Orthogonality Sampling Method (OSM)
        
        Step size methods:
        - **fixed**: :math:`\alpha = \frac{\mathbf{v}^H \boldsymbol{\rho}}{\|\mathbf{v}\|^2}`
        - **optimum**: Line search optimization
        
        Examples
        --------
        >>> cgm = ConjugatedGradientMethod(initial_guess='background',
        ...                                step='optimum',
        ...                                stop_criteria=my_criteria)
        >>> result = cgm.solve(input_data, discretization)
        >>> print(f"Final error: {result.total_error}")
        >>> print(f"Iterations: {result.number_iterations}")
        
        >>> # Solve with custom output
        >>> with open('cgm_log.txt', 'w') as f:
        ...     result = cgm.solve(input_data, discretization, 
        ...                       print_file=f)
        """
        result = super().solve(inputdata, discretization,
                               print_info=print_info, print_file=print_file)

        NY, NX = discretization.elements
        N = NX*NY
        NS = inputdata.configuration.NS
        dx, dy = inputdata.configuration.Lx/NX, inputdata.configuration.Ly/NY
        dS = dx*dy
        Es = inputdata.scattered_field
        GS, GD = discretization.GS, discretization.GD
        forward_solver = mom.MoM_CG_FFT()
        Ei = forward_solver.incident_field((NY, NX), inputdata.configuration)

        # If the same object is used for different resolution instances,
        # then some parameters may need to be updated within the inverse
        # solver. So, the next line ensures it:
        execution_time = 0.
        current_evaluations = 0
        iteration = 0
        objective_function = np.inf
        base, power = 1, 0
        
        tic = tm.time()
        if self.initial_guess == BACKGROUND:
            X = sps.dia_matrix((N, N),dtype=complex)
        elif self.initial_guess == BACKPROPAGATION:
            gamma = norm(np.reshape(GS.conj().T @ Es, (-1, 1)))**2/norm(np.reshape(GS @ GS.conj().T @ Es, (-1, 1)))**2
            w0 = gamma*GS.conj().T @ Es
            X = sps.dia_matrix(np.diag(1/NS*np.sum(w0/Ei,1)),dtype=complex)
        elif self.initial_guess == IMAGE:
            X = cfg.get_contrast_map(epsilon_r=inputdata.rel_permittivity,
                                     sigma=inputdata.conductivity,
                                     configuration=inputdata.configuration)
            X = discretization.contrast_image(X, (NY, NX))
            X = sps.dia_matrix(np.diag(np.reshape(X, -1)), dtype=complex)
        elif self.initial_guess == QUALITATIVE:
            method = osm.OrthogonalitySamplingMethod()
            temp = inputdata.copy()
            temp.resolution = discretization.elements
            result = method.solve(temp, discretization, print_info=False)
            X = cfg.get_contrast_map(epsilon_r=result.rel_permittivity,
                                     sigma=result.conductivity,
                                     configuration=result.configuration)
            X = sps.dia_matrix(np.diag(X.flatten()), dtype=complex)
        
        d = np.zeros((N, 1), dtype=complex)
        g = np.ones((N, 1), dtype=complex)
        
        cnvg = []
        I = sps.eye(N, dtype=complex)
        LC = inv(I-GD@X)
        rho = Es-GS@X@LC@Ei
        cnvg.append([norm(rho.reshape(-1))**2, 0.])
        execution_time +=  tm.time()-tic
        last_iteration_printed = False

        while (not self.stop_criteria.stop(current_evaluations, iteration,
                                           objective_function)):

            tic = tm.time()
            
            # Computing the gradient
            gradJ = np.zeros((N, 1), dtype=complex)
            for l in range(NS):
                gsrho = GS.conj().T@rho[:,l]
                gradJ = gradJ - np.reshape(2*np.conj(sps.spdiags(LC@Ei[:,l], 0, N, N)@LC)@gsrho, (-1, 1))
            
            g_last = np.copy(g)
            g = -gradJ
            
            # Computing the optimum direction
            d = g + np.vdot(g-g_last, g*dS)/norm(g_last)**2*d
            D = sps.spdiags(d.reshape(-1), 0, N, N)
            
            # Computing v matrix
            v = GS@LC.T@D@LC@Ei
            
            # Computing step
            if self.step == 'fixed':
                alpha = 0
                for l in range(NS):
                    alpha += np.vdot(v[:, l], rho[:, l]*dx)
                alpha = alpha/norm(v.reshape(-1))**2
            elif self.step == 'optimum':
                def fx(x, rho, v):
                    return norm(np.reshape(rho-x*v,(-1,1)))**2
                xopt = opt.minimize_scalar(fx, args=(rho, v))
                alpha = xopt.x
                current_evaluations += xopt.nfev
            
            # Computing next contrast
            X = X + alpha*D
    
            # Computing the inverse matriz
            LC = inv(I-GD@X)
    
            # Computing the residual
            # rho = es-gs@C@LC@ei
            rho = rho-alpha*v
    
            # Computing the objective function
            J = norm(rho.reshape(-1))**2
            current_evaluations += 1
            
            DT = tm.time()-tic
            execution_time += DT
            # iteration_message += ('Cost function: %.2e' %J
            #                       + ' - norm(g): %.2e' %norm(g)
            #                       + ' - time: %.1f sec' %DT)
            cnvg.append([J, norm(g)])
            # iteration_message = result.last_error_message(iteration_message)
            # if print_info:
            #     print(iteration_message, file=print_file)
                
            if print_info:
                if iteration+1 >= base*10**power:
                    if base == 9:
                        base = 1
                        power += 1
                    else:
                        base += 1
                    iteration_message = 'Iteration: %d - ' % (iteration+1)
                    iteration_message += ('Cost function: %.2e' %J
                                          + ' - norm(g): %.2e' %norm(g)
                                          + ' - time: %.1f sec' %DT)
                    iteration_message = result.last_error_message(
                        iteration_message
                    )
                    print(iteration_message, file=print_file)
                    last_iteration_printed = True
                else:
                    last_iteration_printed = False

            iteration += 1

        if print_info and not last_iteration_printed:
            iteration_message = 'Iteration: %d - ' % iteration
            iteration_message += ('Cost function: %.2e' %J
                                  + ' - norm(g): %.2e' %norm(g)
                                  + ' - time: %.1f sec' %DT)
            iteration_message = result.last_error_message(iteration_message)
            print(iteration_message, file=print_file)

        cnvg = np.array(cnvg)
        cnvg = cnvg[:, 0].flatten()

        contrast=discretization.contrast_image(np.reshape(X.data, (NY, NX)), inputdata.resolution)
        result.update_error(inputdata,
                            scattered_field=None,
                            total_field=None,
                            contrast=contrast,
                            objective_function=cnvg)

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

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        """
        Print method title and configuration information.
        
        Prints the method title along with key configuration parameters
        including initial guess strategy, step size method, and stopping criteria.
        
        Parameters
        ----------
        inputdata : inputdata.InputData
            Input data object containing problem configuration
        discretization : object
            Discretization object containing grid information
        print_file : file-like object, default=sys.stdout
            File object for printing information
            
        Examples
        --------
        >>> cgm._print_title(input_data, discretization)
        Conjugated Gradient Method
        Initial guess: background
        Step: optimum
        Maximum iterations: 100
        """
        super()._print_title(inputdata, discretization, print_file=print_file)
        print('Initial guess: ' + self.initial_guess, file=print_file)
        print('Step: ' + self.step, file=print_file)
        print(self.stop_criteria, file=print_file)

    def save(self, file_path=''):
        """
        Save the CGM configuration to a file.
        
        Saves the complete method configuration including initial guess strategy,
        step size method, and stopping criteria using pickle serialization.
        
        Parameters
        ----------
        file_path : str, default=''
            Path where the configuration file will be saved
            
        Examples
        --------
        >>> cgm.save('/path/to/save/')
        >>> cgm.save()  # Save in current directory
        """
        data = super().save(file_path=file_path)
        data[INITIAL_GUESS] = self.initial_guess
        data[STEP] = self.step
        data[STOP_CRITERIA] = self.stop_criteria
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        """
        Import CGM configuration from a saved file.
        
        Loads a previously saved CGM configuration including initial guess
        strategy, step size method, and stopping criteria.
        
        Parameters
        ----------
        file_name : str
            Name of the file to import from
        file_path : str, default=''
            Path to the file location
            
        Examples
        --------
        >>> cgm = ConjugatedGradientMethod(initial_guess='background',
        ...                                step='optimum',
        ...                                stop_criteria=my_criteria)
        >>> cgm.importdata('cgm_config.pkl', '/path/to/files/')
        """
        data = super().importdata(file_name, file_path=file_path)
        self.initial_guess = data[INITIAL_GUESS]
        self.step = data[STEP]
        self.stop_criteria= data[STOP_CRITERIA]

    def copy(self, new=None):
        """
        Create a copy of the CGM instance.
        
        Creates either a new independent instance or copies configuration
        to an existing instance.
        
        Parameters
        ----------
        new : ConjugatedGradientMethod or None, default=None
            If None, creates a new independent instance
            If provided, copies configuration to this instance
            
        Returns
        -------
        ConjugatedGradientMethod or None
            New instance if new=None, otherwise None
            
        Examples
        --------
        >>> # Create independent copy
        >>> cgm_copy = cgm.copy()
        
        >>> # Copy configuration to existing instance
        >>> cgm_new = ConjugatedGradientMethod(initial_guess='background',
        ...                                    step='fixed',
        ...                                    stop_criteria=other_criteria)
        >>> cgm.copy(cgm_new)  # cgm_new now has cgm's configuration
        """
        if new is None:
            return ConjugatedGradientMethod(self.initial_guess, self.step,
                                            self.stop_criteria, alias=self.alias)
        else:
            super().copy(new)
            new.initial_guess = self.initial_guess
            new.step = self.step
            new.stop_criteria = self.stop_criteria

    def __str__(self):
        """
        Return string representation of the CGM configuration.
        
        Creates a formatted string containing the method configuration
        including initial guess strategy, step size method, and stopping criteria.
        
        Returns
        -------
        str
            Formatted string representation of the CGM configuration
            
        Examples
        --------
        >>> print(cgm)
        Conjugated Gradient Method
        Initial guess: background
        Step: optimum
        Maximum iterations: 100
        Error tolerance: 1e-4
        """
        message = super().__str__()
        message += 'Initial guess: ' + self.initial_guess + '\n'
        message += 'Step: ' + self.step + '\n'
        message += str(self.stop_criteria)
        return message