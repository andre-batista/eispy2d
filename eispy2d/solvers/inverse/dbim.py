r"""Distorted Born Iterative Method for Electromagnetic Inverse Scattering.

This module implements the Distorted Born Iterative Method (DBIM) [1]_ for solving
two-dimensional electromagnetic inverse scattering problems. The DBIM is a nonlinear
iterative method that couples forward and inverse solvers to reconstruct the
electromagnetic properties of unknown scatterers from scattered field measurements.

The method is based on the Born approximation with distorted incident fields,
where the Green's function is iteratively updated to account for the estimated
contrast function. This approach provides better convergence properties compared
to the standard Born Iterative Method, particularly for high-contrast scatterers.

Classes
-------
DistortedBornIterativeMethod
    Main implementation of the DBIM algorithm

Functions
---------
update_greenf
    Update Green's function matrix with current contrast estimate
faster_computation
    Optimized computation for Green's function update (JIT-compiled)

Examples
--------
>>> import dbim
>>> import forward
>>> import regularization
>>> import stopcriteria
>>> # Create solver components
>>> forward_solver = forward.MomentMethod()
>>> reg_solver = regularization.TikhonovRegularization(alpha=1e-3)
>>> stop_criteria = stopcriteria.MaxIterations(max_iterations=20)
>>> # Create DBIM solver
>>> solver = dbim.DistortedBornIterativeMethod(
...     forward_solver=forward_solver,
...     regularization=reg_solver,
...     stop_criteria=stop_criteria
... )
>>> # Solve inverse problem
>>> result = solver.solve(input_data, discretization)

References
----------
.. [1] W. C. Chew and Y. M. Wang, "Reconstruction of two-dimensional 
       permittivity distribution using the distorted Born iterative method," 
       in IEEE Transactions on Medical Imaging, vol. 9, no. 2, pp. 218-225, 
       June 1990, doi: 10.1109/42.56334.
"""

# Standard libraries
import time as tm
import numpy as np
from scipy.linalg import norm
from numpy.linalg import inv
import sys
import pickle
from numba import jit

# Developed libraries
from eispy2d.core import configuration as cfg
from eispy2d.core import inputdata as ipt
from eispy2d.core import result as rst
from eispy2d.solvers.base import deterministic as dtm
from eispy2d.discretization import collocation as clc


FORWARD = 'forward'
REGULARIZATION = 'regularization'
STOP_CRITERIA = 'stop criteria'


class DistortedBornIterativeMethod(dtm.Deterministic):
    r"""Distorted Born Iterative Method for electromagnetic inverse scattering.

    This class implements the Distorted Born Iterative Method (DBIM) [1]_, a
    nonlinear iterative algorithm for solving electromagnetic inverse scattering
    problems. The method couples forward and inverse solvers iteratively,
    updating the Green's function at each iteration to account for the
    estimated contrast function.
    
    The DBIM algorithm works by:
    1. Starting with Born approximation (incident field as total field)
    2. Solving linear inverse problem to get initial contrast estimate
    3. Updating Green's function using current contrast estimate
    4. Solving forward problem with updated Green's function
    5. Computing residual scattered field
    6. Solving linear inverse problem for contrast update
    7. Repeating until convergence or maximum iterations
    
    The method provides improved convergence compared to standard BIM,
    particularly for high-contrast scatterers, by using distorted incident
    fields that better approximate the true total field.

    Parameters
    ----------
    forward_solver : Forward
        Forward solver implementation for computing total electric field
        and scattered field from given contrast distribution
    regularization : Regularization
        Regularization method for solving the linear inverse problem
        (e.g., Tikhonov regularization, truncated SVD)
    stop_criteria : StopCriteria
        Stopping criteria object defining when to terminate iterations
        (e.g., maximum iterations, relative error threshold)
    alias : str, default='dbim'
        Alias name for the algorithm instance
    import_filename : str, optional
        Filename to import previously saved algorithm state
    import_filepath : str, default=''
        Path to directory containing import file

    Attributes
    ----------
    name : str
        Algorithm name ('Distorted Born Iterative Method')
    forward : Forward
        Forward solver instance
    regularization : Regularization
        Regularization method instance
    stop_criteria : StopCriteria
        Stopping criteria instance
    alias : str
        Algorithm alias for identification and file naming

    Methods
    -------
    solve(inputdata, discretization, print_info=True, print_file=sys.stdout, initial_guess=None)
        Solve electromagnetic inverse scattering problem using DBIM
    save(file_path='')
        Save algorithm state to file
    importdata(file_name, file_path='')
        Import algorithm state from file
    copy(new=None)
        Create copy of algorithm instance
    
    Examples
    --------
    >>> # Create DBIM solver with components
    >>> forward_solver = MomentMethod()
    >>> regularization = TikhonovRegularization(alpha=1e-3)
    >>> stop_criteria = MaxIterations(max_iterations=20)
    >>> solver = DistortedBornIterativeMethod(
    ...     forward_solver=forward_solver,
    ...     regularization=regularization,
    ...     stop_criteria=stop_criteria
    ... )
    >>> # Solve inverse problem
    >>> result = solver.solve(input_data, discretization)
    >>> print(f"Converged in {result.number_iterations} iterations")
    
    References
    ----------
    .. [1] W. C. Chew and Y. M. Wang, "Reconstruction of two-dimensional 
       permittivity distribution using the distorted Born iterative method," 
       in IEEE Transactions on Medical Imaging, vol. 9, no. 2, pp. 218-225, 
       June 1990, doi: 10.1109/42.56334.
    """

    def __init__(self, forward_solver, regularization, stop_criteria,
                 alias='dbim', import_filename=None, import_filepath=''):
        """Initialize the Distorted Born Iterative Method solver.

        Parameters
        ----------
        forward_solver : Forward
            Forward solver implementation for computing electromagnetic fields.
            Must implement methods for computing incident field and solving
            the forward scattering problem.
        regularization : Regularization
            Regularization method for solving the linear inverse problem.
            Common choices include Tikhonov regularization, truncated SVD,
            or other regularization schemes.
        stop_criteria : StopCriteria
            Stopping criteria object that determines when to terminate
            the iterative process. Can be based on maximum iterations,
            relative error, objective function value, etc.
        alias : str, default='dbim'
            Alias name for the algorithm instance, used for identification
            and file naming when saving/loading algorithm state.
        import_filename : str, optional
            If provided, imports algorithm state from this file instead
            of initializing with provided parameters.
        import_filepath : str, default=''
            Directory path where the import file is located.
            
        Notes
        -----
        If `import_filename` is provided, the algorithm state is loaded
        from file and other parameters are ignored. Otherwise, a new
        instance is created with the provided solver components.
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(alias=alias, parallelization=None)
            self.name = 'Distorted Born Iterative Method'
            self.forward = forward_solver
            self.regularization = regularization
            self.stop_criteria = stop_criteria

    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout, initial_guess=None):
        """Solve electromagnetic inverse scattering problem using DBIM.

        This method implements the complete DBIM algorithm, iteratively
        updating the contrast function and Green's function until
        convergence or maximum iterations are reached.

        Algorithm Steps:
        1. Initialize with Born approximation (incident field as total field)
        2. Solve linear inverse problem for initial contrast estimate
        3. Enter iterative loop:
           a. Update Green's function using current contrast
           b. Solve forward problem with updated Green's function
           c. Compute residual scattered field
           d. Solve linear inverse problem for contrast update
           e. Check stopping criteria
        4. Return final reconstruction results

        Parameters
        ----------
        inputdata : InputData
            Input data object containing:
            - scattered_field: Measured scattered field data
            - configuration: Problem configuration (frequency, background, etc.)
            - resolution: Desired output resolution
            - indicators: Performance metrics to compute
        discretization : Discretization
            Discretization object containing:
            - elements: Discretization grid points
            - GS: Scattered field Green's function matrix
            - GD: Domain Green's function matrix
            - Methods for field interpolation and contrast imaging
        print_info : bool, default=True
            Whether to print iteration information during solving
        print_file : file-like object, default=sys.stdout
            Output stream for printing iteration information
        initial_guess : array_like, optional
            Initial guess for contrast function. If None, uses Born
            approximation with incident field as total field.

        Returns
        -------
        Result
            Result object containing:
            - rel_permittivity: Reconstructed relative permittivity
            - conductivity: Reconstructed conductivity (if applicable)
            - scattered_field: Final computed scattered field
            - total_field: Final computed total field
            - execution_time: Algorithm execution time
            - number_iterations: Number of iterations performed
            - number_evaluations: Number of function evaluations
            - error_history: Convergence history

        Notes
        -----
        The algorithm automatically handles both good conductor and
        perfect dielectric cases based on the configuration. For
        good conductors, only conductivity is reconstructed. For
        perfect dielectrics, only permittivity is reconstructed.

        The Green's function update is the key difference from standard
        BIM, providing better convergence for high-contrast scatterers
        by using distorted incident fields.

        Examples
        --------
        >>> # Solve with default settings
        >>> result = solver.solve(input_data, discretization)
        >>> print(f"Converged in {result.number_iterations} iterations")
        
        >>> # Solve with initial guess and custom output
        >>> with open('output.txt', 'w') as f:
        ...     result = solver.solve(input_data, discretization,
        ...                          initial_guess=my_guess,
        ...                          print_file=f)
        """
        result = super().solve(inputdata, discretization,
                               print_info=print_info, print_file=print_file)

        # First-Order Born Approximation
        tic = tm.time()
        if initial_guess is None:
            total_field = self.forward.incident_field(discretization.elements,
                                                      inputdata.configuration)
            contrast = discretization.solve(
                scattered_field=inputdata.scattered_field,
                total_field=total_field,
                linear_solver=self.regularization
            )
            contrast = discretization.contrast_image(contrast,
                                                     discretization.elements)
        else:
            contrast = (discretization.contrast_image(initial_guess,
                                                      discretization.elements)
                        + 0j)
            
        greenf_s = discretization.GS.copy()
        execution_time = tm.time()-tic

        # If the same object is used for different resolution instances,
        # then some parameters may need to be updated within the inverse
        # solver. So, the next line ensures it:
        current_evaluations = 0
        iteration = 0
        objective_function = np.inf

        while (not self.stop_criteria.stop(current_evaluations, iteration,
                                           objective_function)):

            iteration_message = 'Iteration: %d - ' % (iteration+1)
            tic = tm.time()
            greenf_s = update_greenf(contrast, discretization.GD, greenf_s)
            execution_time +=  tm.time()-tic
            

            if not inputdata.configuration.good_conductor:
                rel_permittivity = cfg.get_relative_permittivity(
                    contrast, inputdata.configuration.epsilon_rb
                )
            else:
                rel_permittivity = None

            if not inputdata.configuration.perfect_dielectric:
                conductivity = cfg.get_conductivity(
                    contrast, 2*np.pi*inputdata.configuration.f,
                    inputdata.configuration.epsilon_rb,
                    inputdata.configuration.sigma_b
                )
            else:
                conductivity = None

            solution = ipt.InputData(
                name='aux', configuration=inputdata.configuration,
                rel_permittivity=rel_permittivity,
                conductivity=conductivity
            )

            tic = tm.time()
            self.forward.solve(solution, noise=0., PRINT_INFO=False,
                               SAVE_INTERN_FIELD=True)

            scattered_field = (inputdata.scattered_field
                               - solution.scattered_field)
            
            kernel = clc.kernel_GSE(greenf_s, solution.total_field)
            dX = self.regularization.solve(kernel, scattered_field.flatten()) 

            contrast = contrast + dX.reshape(discretization.elements)

            # The variable `execution_time` will record only the time
            # expended by the forward and linear routines.
            execution_time +=  tm.time()-tic

            objective_function = norm(inputdata.scattered_field
                                      - solution.scattered_field)**2

            result.update_error(
                inputdata, scattered_field=solution.scattered_field,
                total_field=discretization.total_image(solution.total_field,
                                                       inputdata.resolution),
                contrast=discretization.contrast_image(contrast,
                                                       inputdata.resolution),
                objective_function=objective_function)

            iteration_message = result.last_error_message(iteration_message)

            if print_info:
                print(iteration_message, file=print_file)

            current_evaluations += 1
            iteration += 1

        # Remember: results stores the estimated scattered field. Not
        # the given one.
        result.scattered_field = solution.scattered_field
        result.total_field = solution.total_field

        contrast=discretization.contrast_image(contrast, inputdata.resolution)

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
        """Print algorithm title and configuration information.

        This method prints the DBIM algorithm title along with
        forward solver, regularization, and stopping criteria details.

        Parameters
        ----------
        inputdata : InputData
            Input data object containing problem configuration
        discretization : Discretization
            Discretization object containing mesh information
        print_file : file-like object, default=sys.stdout
            Output stream for printing information
        """
        super()._print_title(inputdata, discretization, print_file=print_file)
        print(self.forward, file=print_file)
        print(self.regularization, file=print_file)
        print(self.stop_criteria, file=print_file)

    def save(self, file_path=''):
        """Save DBIM algorithm state to file.

        This method saves the complete algorithm state including forward
        solver, regularization method, and stopping criteria to a file
        for later restoration.

        Parameters
        ----------
        file_path : str, default=''
            Base path for saving algorithm state files. The algorithm
            alias will be appended to create the full filename.

        Returns
        -------
        dict
            Dictionary containing algorithm state data

        Notes
        -----
        The saved file can be loaded later using the `importdata` method
        or by specifying `import_filename` during initialization.
        """
        data = super().save(file_path=file_path)
        data[FORWARD] = self.forward
        data[REGULARIZATION] = self.regularization
        data[STOP_CRITERIA] = self.stop_criteria
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)
        return data

    def importdata(self, file_name, file_path=''):
        """Import DBIM algorithm state from file.

        This method loads the complete algorithm state including forward
        solver, regularization method, and stopping criteria from a
        previously saved file.

        Parameters
        ----------
        file_name : str
            Name of the file containing algorithm state data
        file_path : str, default=''
            Path to the directory containing the state file

        Returns
        -------
        dict
            Dictionary containing imported algorithm state data

        Notes
        -----
        This method restores the complete algorithm configuration,
        allowing resumption of work with previously saved settings.
        """
        data = super().importdata(file_name, file_path=file_path)
        self.forward = data[FORWARD]
        self.regularization = data[REGULARIZATION]
        self.stop_criteria= data[STOP_CRITERIA]
        return data

    def copy(self, new=None):
        """Create a copy of the DBIM algorithm instance.

        This method creates a deep copy of the algorithm with the same
        configuration parameters, allowing independent use of multiple
        solver instances.

        Parameters
        ----------
        new : DistortedBornIterativeMethod, optional
            Existing instance to copy configuration into. If None,
            creates and returns a new instance.

        Returns
        -------
        DistortedBornIterativeMethod or None
            New algorithm instance if `new` is None, otherwise None
            (configuration is copied into `new` parameter)

        Notes
        -----
        When `new` is None, returns a completely independent new instance.
        When `new` is provided, copies configuration into that instance.
        """
        if new is None:
            return DistortedBornIterativeMethod(self.forward,
                                                self.regularization,
                                                self.stop_criteria,
                                                alias=self.alias)
        else:
            super().copy(new)
            new.forward = self.forward
            new.regularization = self.regularization
            new.stop_criteria = self.stop_criteria

    def __str__(self):
        """Return string representation of the DBIM algorithm.

        Returns
        -------
        str
            String representation including algorithm details and
            configuration of forward solver, regularization method,
            and stopping criteria.
        """
        message = super().__str__()
        message += str(self.forward)
        message += str(self.regularization)
        message += str(self.stop_criteria)
        return message


def update_greenf(contrast, greenf_d, greenf_s):
    """Update Green's function matrix using current contrast estimate.

    This function updates the scattered field Green's function matrix
    to account for the current contrast function estimate. This is the
    key step that distinguishes DBIM from standard Born methods.

    The update is performed using the formula:

    .. math::
        G_S^{new} = (I - G_D \\chi)^{-1} G_S

    where \(G_S\) is the scattered field Green's function,
    \(G_D\) is the domain Green's function, \(\chi\) is the
    contrast function, and \(I\) is the identity matrix.
    Parameters
    ----------
    contrast : numpy.ndarray
        Current contrast function estimate as a 2D array representing
        the spatial distribution of contrast values
    greenf_d : numpy.ndarray
        Domain Green's function matrix relating contrast sources to
        total field within the investigation domain
    greenf_s : numpy.ndarray
        Current scattered field Green's function matrix

    Returns
    -------
    numpy.ndarray
        Updated scattered field Green's function matrix that incorporates
        the current contrast estimate

    Notes
    -----
    This function creates a diagonal matrix from the contrast function
    and uses the optimized JIT-compiled `faster_computation` function
    to perform the matrix inversion and multiplication efficiently.

    The updated Green's function provides a better approximation to the
    true scattering behavior, leading to improved convergence compared
    to methods that use fixed Green's functions.

    Examples
    --------
    >>> contrast = np.array([[1.0, 1.5], [1.2, 1.8]])
    >>> updated_gs = update_greenf(contrast, greenf_d, greenf_s)
    >>> print(updated_gs.shape)
    (num_receivers, num_domain_points)
    """
    X = np.diag(contrast.flatten(), k=0)
    I = np.eye(contrast.size)
    return faster_computation(X, I, greenf_d, greenf_s)

@jit(nopython=True)
def faster_computation(X, I, GD, GS):
    """Optimized computation for Green's function update.

    This JIT-compiled function performs the core matrix operations
    required for updating the Green's function in the DBIM algorithm.
    The computation is optimized using Numba for maximum performance.

    The function computes:
    .. math::
        G_S^{new} = [(I - G_D \\chi)^{-1} G_S^T]^T

    where the transposition is used to ensure proper matrix dimensions.

    Parameters
    ----------
    X : numpy.ndarray
        Diagonal matrix representation of the contrast function,
        created from the flattened contrast array
    I : numpy.ndarray
        Identity matrix of size matching the contrast function
    GD : numpy.ndarray
        Domain Green's function matrix
    GS : numpy.ndarray
        Scattered field Green's function matrix

    Returns
    -------
    numpy.ndarray
        Updated scattered field Green's function matrix

    Notes
    -----
    This function is JIT-compiled with Numba for high performance.
    The matrix inversion is performed using standard linear algebra
    operations. For large problems, this operation can be computationally
    intensive and may benefit from specialized solvers.

    The function assumes that the matrix (I - GD@X) is invertible.
    In practice, this is generally true for physical scattering problems,
    but numerical issues may arise for very high contrast values.

    Examples
    --------
    >>> X = np.diag([1.0, 1.5, 1.2])
    >>> I = np.eye(3)
    >>> result = faster_computation(X, I, GD, GS)
    """
    return np.transpose(inv(I - GD@X) @ GS.T)