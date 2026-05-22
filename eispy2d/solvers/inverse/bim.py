r"""Born Iterative Method for Electromagnetic Inverse Scattering.

This module implements the Born Iterative Method (BIM) for solving nonlinear
electromagnetic inverse scattering problems. The BIM is a classical iterative
approach that couples forward and inverse solvers to reconstruct the material
properties of scattering objects.

The method works by iteratively improving the solution using the Born
approximation, where each iteration involves solving a linearized inverse
problem and updating the total field using a forward solver.

The implementation follows the Deterministic solver abstract class and provides
a complete framework for nonlinear inverse scattering reconstruction.

Classes
-------
BornIterativeMethod : dtm.Deterministic
    Main class implementing the Born Iterative Method.

Constants
---------
FORWARD : str
    Key for forward solver in serialization.
REGULARIZATION : str
    Key for regularization method in serialization.
STOP_CRITERIA : str
    Key for stop criteria in serialization.

References
----------
.. [1] Wang, Y. M., and Weng Cho Chew. "An iterative solution of the
   two‐dimensional electromagnetic inverse scattering problem."
   International Journal of Imaging Systems and Technology 1.1 (1989):
   100-108.
"""

# Standard libraries
import time as tm
import numpy as np
from scipy.linalg import norm
import sys
import pickle

# Developed libraries
from eispy2d.core import configuration as cfg
from eispy2d.data import inputdata as ipt
from eispy2d.data import result as rst
from eispy2d.solvers.base import deterministic as dtm


FORWARD = 'forward'
REGULARIZATION = 'regularization'
STOP_CRITERIA = 'stop criteria'


class BornIterativeMethod(dtm.Deterministic):
    r"""Born Iterative Method for Nonlinear Electromagnetic Inverse Scattering.

    This class implements the Born Iterative Method (BIM), a classical iterative
    approach for solving nonlinear electromagnetic inverse scattering problems.
    The method couples forward and inverse solvers in an iterative process to
    reconstruct material properties from scattered field measurements.

    The algorithm works by:
    1. Starting with the incident field as initial guess
    2. Solving the linearized inverse problem using the current total field
    3. Updating the total field using the forward solver
    4. Repeating until convergence criteria are met

    Parameters
    ----------
    forward_solver : ForwardSolver
        Forward solver implementation for computing total electric field.
    regularization : Regularization
        Regularization method for the linear inverse problem.
    stop_criteria : StopCriteria
        Convergence criteria for the iterative process.
    alias : str, default: 'bim'
        Alias name for the method instance.
    import_filename : str, optional
        Filename to import previously saved method state.
    import_filepath : str, default: ''
        Path to the file containing saved method state.

    Attributes
    ----------
    name : str
        Name of the method ('Born Iterative Method').
    forward : ForwardSolver
        Forward solver instance for field computation.
    regularization : Regularization
        Regularization method for linear inverse problem.
    stop_criteria : StopCriteria
        Convergence criteria for the iterative process.

    Methods
    -------
    solve(inputdata, discretization, print_info=True, print_file=sys.stdout)
        Solve the nonlinear inverse scattering problem.
    save(file_path='')
        Save the method state to file.
    importdata(file_name, file_path='')
        Import method state from file.
    copy(new=None)
        Create a copy of the method instance.

    References
    ----------
    .. [1] Wang, Y. M., and Weng Cho Chew. "An iterative solution of the
       two‐dimensional electromagnetic inverse scattering problem."
       International Journal of Imaging Systems and Technology 1.1 (1989):
       100-108.
    """

    def __init__(self, forward_solver, regularization, stop_criteria,
                 alias='bim', import_filename=None, import_filepath=''):
        r"""Initialize the Born Iterative Method.

        Creates a new BIM instance with the specified forward solver,
        regularization method, and convergence criteria.

        Parameters
        ----------
        forward_solver : ForwardSolver
            Forward solver implementation for computing the total electric field
            in the scattering domain. Must implement the ForwardSolver interface.
        regularization : Regularization
            Regularization method for solving the linearized inverse problem.
            Used to handle ill-conditioning in the linear system.
        stop_criteria : StopCriteria
            Convergence criteria object that determines when to stop the
            iterative process. Typically based on relative change in solution
            or maximum number of iterations.
        alias : str, default: 'bim'
            Alias name for this method instance, used for identification
            and file naming.
        import_filename : str, optional
            Filename to import previously saved method state. If provided,
            other parameters are ignored and state is loaded from file.
        import_filepath : str, default: ''
            Path to the file containing saved method state.

        Notes
        -----
        The method uses the standard Born iterative approach where:
        
        1. The incident field serves as the initial guess for total field
        2. At each iteration, the linear inverse problem is solved using
           the current total field estimate
        3. The forward solver computes the new total field
        4. The process continues until convergence criteria are met

        The mathematical formulation involves solving:
        
        .. math::
            \\chi^{(n+1)} = \\mathcal{L}^{-1}[E_s, E_t^{(n)}]
            
        where :math:`\\chi` is the contrast, :math:`E_s` is the scattered field,
        :math:`E_t^{(n)}` is the total field at iteration n, and :math:`\\mathcal{L}^{-1}`
        is the linearized inverse operator.
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(alias=alias, parallelization=None)
            self.name = 'Born Iterative Method'
            self.forward = forward_solver
            self.regularization = regularization
            self.stop_criteria = stop_criteria

    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout):
        """Solve the nonlinear inverse scattering problem using BIM.

        Implements the Born Iterative Method to reconstruct material properties
        from scattered field measurements. The method iteratively refines the
        solution by coupling forward and inverse solvers.

        Parameters
        ----------
        inputdata : InputData
            Input data object containing scattered field measurements,
            configuration parameters, and other problem-specific information.
        discretization : Discretization
            Discretization object containing domain information, Green's
            function matrices, and methods for field computation.
        print_info : bool, default: True
            Whether to print iteration progress information to output.
        print_file : file-like object, default: sys.stdout
            File object to write progress information to.

        Returns
        -------
        result : Result
            Result object containing the reconstructed material properties,
            fields, error metrics, and algorithm performance information.

        Notes
        -----
        The algorithm follows these steps:

        1. **Initialization**: Start with incident field as initial total field
        2. **Iterative Process**: For each iteration:
           
           a. Solve linearized inverse problem:
              :math:`\\chi^{(k)} = \\mathcal{R}[G_S^H (E_s - G_S E_t^{(k-1)} \\chi^{(k-1)})]`
           
           b. Update material properties and create auxiliary InputData
           
           c. Solve forward problem to get new total field:
              :math:`E_t^{(k)} = E_i + G_D \\chi^{(k)} E_t^{(k)}`
           
           d. Compute objective function and check convergence
           
        3. **Termination**: Stop when convergence criteria are satisfied

        The method handles both dielectric and conducting materials, automatically
        converting between contrast and physical parameters (permittivity,
        conductivity) as needed.

        **Convergence Criteria**: The algorithm stops when the stop criteria
        object determines convergence based on iteration count, objective
        function value, or relative change in solution.

        **Error Tracking**: The method computes and tracks various error metrics
        including relative permittivity error, objective function value, and
        field reconstruction errors.
        """
        result = super().solve(inputdata, discretization,
                               print_info=print_info, print_file=print_file)

        # First-Order Born Approximation
        total_field = self.forward.incident_field(discretization.elements,
                                                  inputdata.configuration)

        # If the same object is used for different resolution instances,
        # then some parameters may need to be updated within the inverse
        # solver. So, the next line ensures it:
        execution_time = 0.
        current_evaluations = 0
        iteration = 0
        objective_function = np.inf

        while (not self.stop_criteria.stop(current_evaluations, iteration,
                                           objective_function)):

            iteration_message = 'Iteration: %d - ' % (iteration+1)
            tic = tm.time()
            contrast = discretization.solve(
                scattered_field=inputdata.scattered_field,
                total_field=total_field,
                linear_solver=self.regularization
            )
            contrast = discretization.contrast_image(contrast,
                                                     discretization.elements)

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

            self.forward.solve(solution, noise=0., PRINT_INFO=False,
                               SAVE_INTERN_FIELD=True)
            
            total_field = solution.total_field
            scattered_field = discretization.scattered_field(contrast=contrast,
                                                             total_field=total_field)
            objective_function = norm(inputdata.scattered_field
                                      - scattered_field)**2

            # The variable `execution_time` will record only the time
            # expended by the forward and linear routines.
            execution_time +=  tm.time()-tic

            result.update_error(inputdata,
                                scattered_field=scattered_field,
                                total_field=discretization.total_image(total_field,
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
        result.scattered_field = scattered_field
        result.total_field = total_field

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

        Prints the algorithm header including forward solver, regularization
        method, and stop criteria information.

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
        print(self.regularization, file=print_file)
        print(self.stop_criteria, file=print_file)

    def save(self, file_path=''):
        """Save the Born Iterative Method state to file.

        Saves the complete method configuration including forward solver,
        regularization method, and stop criteria to a file for later retrieval.

        Parameters
        ----------
        file_path : str, default: ''
            Path where to save the method state file. The file will be
            saved with the method's alias as the filename.

        Notes
        -----
        The method saves all necessary information to recreate the BIM
        instance, including:
        - Forward solver configuration
        - Regularization method parameters
        - Stop criteria settings
        - Base class attributes (alias, etc.)

        The file is saved using pickle serialization format.
        """
        data = super().save(file_path=file_path)
        data[FORWARD] = self.forward
        data[REGULARIZATION] = self.regularization
        data[STOP_CRITERIA] = self.stop_criteria
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        """Import Born Iterative Method state from file.

        Loads a previously saved BIM configuration from file, restoring
        all method parameters and settings.

        Parameters
        ----------
        file_name : str
            Name of the file containing saved method state.
        file_path : str, default: ''
            Path to the file containing saved method state.

        Notes
        -----
        This method restores the complete BIM configuration including:
        - Forward solver with all its parameters
        - Regularization method configuration
        - Stop criteria settings
        - Base class attributes

        The loaded configuration can be used to recreate the exact same
        method instance as was saved.
        """
        data = super().importdata(file_name, file_path=file_path)
        self.forward = data[FORWARD]
        self.regularization = data[REGULARIZATION]
        self.stop_criteria= data[STOP_CRITERIA]

    def copy(self, new=None):
        """Create a copy of the Born Iterative Method instance.

        Creates a new BIM instance with the same configuration as the current
        one, or copies the current configuration to an existing instance.

        Parameters
        ----------
        new : BornIterativeMethod, optional
            Existing BIM instance to copy configuration to. If None,
            creates a new instance.

        Returns
        -------
        BornIterativeMethod or None
            If new is None, returns a new BIM instance with copied configuration.
            If new is provided, modifies it in place and returns None.

        Notes
        -----
        The copy includes all method components:
        - Forward solver (reference copy)
        - Regularization method (reference copy)
        - Stop criteria (reference copy)
        - Base class attributes (alias, etc.)

        This method is useful for creating multiple instances with the same
        configuration or for backup purposes before modifying parameters.
        """
        if new is None:
            return BornIterativeMethod(self.forward, self.regularization,
                                       self.stop_criteria, alias=self.alias)
        else:
            super().copy(new)
            self.forward = new.forward
            self.regularization = new.regularization
            self.stop_criteria = new.stop_criteria

    def __str__(self):
        """Return string representation of the Born Iterative Method.

        Creates a comprehensive string description of the BIM instance
        including all its components and configuration.

        Returns
        -------
        str
            String representation including:
            - Base class information (name, alias, etc.)
            - Forward solver details
            - Regularization method information
            - Stop criteria configuration

        Notes
        -----
        This method provides a complete overview of the BIM configuration,
        which is useful for debugging, logging, and understanding the
        method setup.
        """
        message = super().__str__()
        message += str(self.forward)
        message += str(self.regularization)
        message += str(self.stop_criteria)
        return message
            