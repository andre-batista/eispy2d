r"""Extended Contrast Source Inversion (ECSI) Method.

This module implements the Extended Contrast Source Inversion (ECSI) method 
for solving electromagnetic inverse scattering problems. ECSI is an advanced 
iterative nonlinear solver that extends the classical Contrast Source Inversion 
(CSI) approach by incorporating enhanced optimization techniques.

The method couples forward and inverse solvers in an iterative process, 
alternately updating the contrast function and current density distribution 
within the scattering domain. It employs conjugate gradient techniques for 
efficient convergence and can handle both lossy and lossless media.

Key Features
------------
- Iterative nonlinear optimization for contrast reconstruction
- Conjugate gradient-based direction updates
- Support for both dielectric and conductive objects
- Flexible stopping criteria
- Integration with various forward solvers

Classes
-------
ExtendedContrastSourceInversion
    Main implementation of the ECSI algorithm extending the deterministic solver base class.

Functions
---------
get_gamma
    Compute the Polak-Ribière conjugate gradient parameter.
compute_constant_j
    Calculate the step size for current density updates.
update_contrast
    Update the contrast function using computed step size and direction.
get_gradient_x
    Compute the gradient of the objective function with respect to contrast.
compute_constant_x
    Calculate the step size for contrast updates.

Notes
-----
The implementation uses Numba JIT compilation for performance-critical functions
and supports both perfect dielectric and good conductor approximations.

References
----------
.. [1] P. M. V. D. Berg, A. L. V. Broekhoven, and A. Abubakar, “Extended 
       contrast source inversion,” Inverse Problems, vol. 15, no. 5, 
       pp. 1325–1344, Oct. 1999, doi: 10.1088/0266-5611/15/5/315.
"""

# Standard libraries
import time as tm
import numpy as np
import sys
import pickle
from numba import jit

# Developed libraries
from eispy2d.core import configuration as cfg
from eispy2d.core import result as rst
from eispy2d.solvers.base import deterministic as dtm
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.solvers.inverse import regularization as reg
from eispy2d.solvers.inverse import csi as csi
from eispy2d.solvers.inverse import backprop as bp
from eispy2d.solvers.forward import fftproduct as fftproduct

FORWARD = 'forward'
STOP_CRITERIA = 'stop criteria'


class ExtendedContrastSourceInversion(dtm.Deterministic):
    r"""Extended Contrast Source Inversion (ECSI) Method.

    This class implements the Extended Contrast Source Inversion (ECSI) algorithm,
    an advanced iterative nonlinear solver for electromagnetic inverse scattering
    problems. ECSI extends the classical Contrast Source Inversion (CSI) approach
    by incorporating enhanced optimization techniques and conjugate gradient methods.

    The method alternately updates the contrast function (representing material
    properties) and current density distribution within the scattering domain.
    It employs a cost function minimization approach with conjugate gradient
    optimization for efficient convergence.

    Mathematical Foundation
    -----------------------
    The ECSI method minimizes the cost function:

    .. math::
        F(\\chi, \\mathbf{J}) = \\frac{||\\mathbf{E}^s - \\mathbf{G}^s \\mathbf{J}||^2}{||\\mathbf{E}^s||^2} + 
        \\frac{||\\mathbf{J} - \\chi \\mathbf{E}^{tot}||^2}{||\\chi \\mathbf{E}^{inc}||^2}

    where:
    - :math:`\\chi` is the contrast function
    - :math:`\\mathbf{J}` is the current density
    - :math:`\\mathbf{E}^s` is the scattered electric field
    - :math:`\\mathbf{E}^{tot}` is the total electric field
    - :math:`\\mathbf{G}^s` is the scattered field Green's function

    Attributes
    ----------
    forward : :class:`forward.Forward`
        An implementation of the abstract forward solver class which computes
        the total electric field for a given contrast distribution.
    stop_criteria : :class:`stopcriteria.StopCriteria`
        Object defining the stopping criteria for the iterative algorithm,
        including maximum iterations and convergence thresholds.
    name : str
        Human-readable name of the algorithm ('Extended Contrast Source Inversion').

    Parameters
    ----------
    stop_criteria : :class:`stopcriteria.StopCriteria`
        Stopping criteria configuration for the iterative algorithm.
    forward_solver : :class:`forward.Forward`, optional
        Forward solver implementation. Default is MoM_CG_FFT().
    alias : str, optional
        Short identifier for the algorithm. Default is 'ecsi'.
    import_filename : str, optional
        Name of file to import saved algorithm state from. Default is None.
    import_filepath : str, optional
        Path to directory containing import file. Default is empty string.

    Examples
    --------
    >>> import stopcriteria as sc
    >>> import mom_cg_fft as mom
    >>> 
    >>> # Create stopping criteria
    >>> stop_crit = sc.MaxIterations(max_iterations=50)
    >>> 
    >>> # Create ECSI solver
    >>> solver = ExtendedContrastSourceInversion(
    ...     stop_criteria=stop_crit,
    ...     forward_solver=mom.MoM_CG_FFT()
    ... )
    >>> 
    >>> # Solve inverse problem
    >>> result = solver.solve(input_data, discretization)

    Notes
    -----
    - The algorithm supports both perfect dielectric and good conductor approximations
    - Uses Numba JIT compilation for performance-critical computations
    - Employs conjugate gradient techniques for efficient convergence
    - Can handle complex-valued contrast functions for lossy media

    References
    ----------
    .. [1] P. M. V. D. Berg, A. L. V. Broekhoven, and A. Abubakar, “Extended 
        contrast source inversion,” Inverse Problems, vol. 15, no. 5, 
        pp. 1325–1344, Oct. 1999, doi: 10.1088/0266-5611/15/5/315.
    """

    def __init__(self, stop_criteria, forward_solver=mom.MoM_CG_FFT(),
                 alias='ecsi', import_filename=None, import_filepath=''):
        r"""Initialize the Extended Contrast Source Inversion solver.

        Creates an ECSI solver instance with specified stopping criteria and
        forward solver. The solver can be initialized from scratch or loaded
        from a previously saved state.

        Parameters
        ----------
        stop_criteria : :class:`stopcriteria.StopCriteria`
            Object defining the stopping criteria for the iterative algorithm.
            This includes maximum iterations, convergence thresholds, and
            other termination conditions.
        forward_solver : :class:`forward.Forward`, optional
            Forward solver implementation for computing the total electric field.
            Default is :class:`mom_cg_fft.MoM_CG_FFT`.
        alias : str, optional
            Short identifier for this solver instance. Used in result naming
            and file operations. Default is 'ecsi'.
        import_filename : str, optional
            Name of file containing previously saved solver state. If provided,
            the solver will be initialized from this saved state. Default is None.
        import_filepath : str, optional
            Directory path containing the import file. Only used if import_filename
            is provided. Default is empty string (current directory).

        Raises
        ------
        FileNotFoundError
            If import_filename is specified but the file cannot be found.
        ValueError
            If the loaded file contains invalid or corrupted solver state.

        Examples
        --------
        >>> import stopcriteria as sc
        >>> import mom_cg_fft as mom
        >>> 
        >>> # Initialize with stopping criteria
        >>> stop_crit = sc.MaxIterations(max_iterations=100)
        >>> solver = ExtendedContrastSourceInversion(
        ...     stop_criteria=stop_crit,
        ...     forward_solver=mom.MoM_CG_FFT()
        ... )
        >>> 
        >>> # Initialize from saved file
        >>> solver = ExtendedContrastSourceInversion(
        ...     stop_criteria=stop_crit,
        ...     import_filename='saved_ecsi.pkl',
        ...     import_filepath='/path/to/saved/files/'
        ... )

        Notes
        -----
        If both import_filename and other parameters are provided, the solver
        will be initialized from the saved file and other parameters will be ignored.
        """
        if import_filename is not None:
            self.importdata(import_filename, import_filepath)
        else:
            super().__init__(alias=alias, parallelization=None)
            self.name = 'Extended Contrast Source Inversion'
            self.forward = forward_solver
            self.stop_criteria = stop_criteria

    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout, initial_guess=None):
        r"""Solve the electromagnetic inverse scattering problem using ECSI.

        Performs iterative reconstruction of the contrast function and current
        density distribution using the Extended Contrast Source Inversion algorithm.
        The method alternates between updating the current density (J) and the
        contrast function (\\chi) using conjugate gradient optimization.

        Parameters
        ----------
        inputdata : :class:`inputdata.InputData`
            Object containing the measured scattered field data and problem
            configuration including frequency, measurement points, and incident
            field parameters.
        discretization : :class:`discretization.Discretization`
            Discretization scheme defining the computational grid, Green's
            functions, and spatial sampling of the investigation domain.
        print_info : bool, optional
            Whether to print iteration progress information. Default is True.
        print_file : file-like object, optional
            File object for printing output. Default is sys.stdout.
        initial_guess : :class:`numpy.ndarray`, optional
            Initial guess for the contrast function. If None, backpropagation
            method is used to generate initial guess. Default is None.

        Returns
        -------
        result : :class:`result.Result`
            Object containing the reconstructed contrast function, relative
            permittivity, conductivity, total field, and convergence information.

        Raises
        ------
        ValueError
            If input data or discretization are invalid.
        RuntimeError
            If the algorithm fails to converge within specified criteria.

        Notes
        -----
        The algorithm implements the following iterative procedure:

        1. **Initialization**: Generate initial guess using backpropagation if not provided
        2. **Current Update**: Minimize data and object error functionals with respect to J
        3. **Contrast Update**: Minimize object error functional with respect to \\chi
        4. **Convergence Check**: Evaluate stopping criteria and continue if necessary

        The cost function minimized is:

        .. math::
            F(\\chi, \\mathbf{J}) = \\frac{||\\mathbf{E}^s - \\mathbf{G}^s \\mathbf{J}||^2}{||\\mathbf{E}^s||^2} + 
            \\frac{||\\mathbf{J} - \\chi \\mathbf{E}^{tot}||^2}{||\\chi \\mathbf{E}^{inc}||^2}

        Examples
        --------
        >>> import inputdata as ipt
        >>> import discretization as dsc
        >>> import stopcriteria as sc
        >>> 
        >>> # Create problem setup
        >>> data = ipt.InputData(frequency=1e9, ...)
        >>> disc = dsc.Discretization(grid_size=(64, 64), ...)
        >>> stop_crit = sc.MaxIterations(max_iterations=100)
        >>> 
        >>> # Create and run solver
        >>> solver = ExtendedContrastSourceInversion(stop_criteria=stop_crit)
        >>> result = solver.solve(data, disc, print_info=True)
        >>> 
        >>> # Access results
        >>> contrast = result.contrast
        >>> permittivity = result.rel_permittivity
        >>> convergence_info = result.number_iterations
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
            chi = np.diag(contrast.flatten(), 0) + 0j
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
        last_message_printed = False

        N, NS = np.prod(discretization.elements), inputdata.configuration.NS
        direction_j = np.zeros((N, NS), dtype=complex)
        direction_x = np.zeros(N, dtype=complex)
        last_gradient_j = np.ones((N, NS), dtype=complex)
        last_gradient_x = np.ones(N, dtype=complex)
        incident_field = self.forward.incident_field(discretization.elements,
                                                     inputdata.configuration)
        normalization_s = csi.get_normalization_s(inputdata.scattered_field)

        while (not self.stop_criteria.stop(current_evaluations, iteration,
                                           objective_function)):

            iteration_message = 'Iteration: %d - ' % (iteration+1)

            tic = tm.time()
            data_error = self._get_data_error(inputdata.scattered_field,
                                              discretization.GS, current)
            total_field = incident_field + fftp.compute(current)
            object_error = self._get_object_error(chi, total_field, current)
            normalization_d = csi.get_normalization_d(chi, incident_field)
            objective_function = self._evaluate_objective_function(
                data_error, normalization_s, object_error, normalization_d
            )
            gradient_j = self._get_gradient_j(discretization.GS, data_error,
                                              normalization_s, object_error,
                                              fftpa, chi, normalization_d)
            gamma_j = get_gamma(gradient_j, last_gradient_j)
            direction_j = self._update_direction(gradient_j, gamma_j, direction_j)
            constant_j = self._get_constant_j(gradient_j, discretization.GS,
                                              direction_j,normalization_s, chi,
                                              fftpa, normalization_d)
            current = csi.update_current(current, constant_j, direction_j)
            total_field = self._update_total_field(current, incident_field,
                                                   fftp)
            gradient_x = self._get_gradient_x(normalization_d, current,
                                              total_field, chi)
            gamma_x = get_gamma(gradient_x, last_gradient_x)
            direction_x = self._update_direction(gradient_x, gamma_x,
                                                 direction_x)
            constant_x = self._get_constant_x(direction_x, total_field, chi,
                                              current, incident_field)
            contrast = self._update_contrast(chi, constant_x, direction_x)
            chi = np.diag(contrast.flatten(), 0) + 0j
            last_gradient_j = gradient_j.copy()
            last_gradient_x = gradient_x.copy()
            execution_time +=  tm.time()-tic
            contrast = contrast.reshape(discretization.elements)
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
                    last_message_printed = True
                else:
                    last_message_printed = False

            current_evaluations += 1
            iteration += 1

        if print_info and not last_message_printed:
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
        r"""Generate initial guess for contrast and current using backpropagation.

        Creates an initial estimate of the contrast function and current density
        distribution using the backpropagation method. This provides a reasonable
        starting point for the iterative ECSI algorithm.

        Parameters
        ----------
        inputdata : :class:`inputdata.InputData`
            Input data containing scattered field measurements and configuration.
        discretization : :class:`discretization.Discretization`
            Discretization scheme defining the computational grid and operators.

        Returns
        -------
        contrast : :class:`numpy.ndarray`
            Initial contrast function estimate resized to match input resolution.
        chi : :class:`numpy.ndarray`
            Diagonal matrix representation of the contrast function.
        current : :class:`numpy.ndarray`
            Initial current density distribution.

        Notes
        -----
        The method uses backpropagation to generate an initial relative permittivity
        estimate, which is then converted to contrast and current density:

        .. math::
            \\chi = \\frac{\\epsilon_r - \\epsilon_{rb}}{\\epsilon_{rb}}

        where :math:`\\epsilon_r` is the relative permittivity and :math:`\\epsilon_{rb}` 
        is the background relative permittivity.
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
        r"""Calculate the object error functional.

        Computes the object error term in the ECSI cost function, which measures
        the consistency between the current density and the contrast-field relationship.

        Parameters
        ----------
        chi : :class:`numpy.ndarray`
            Diagonal matrix representation of the contrast function.
        total_field : :class:`numpy.ndarray`
            Total electric field distribution.
        current : :class:`numpy.ndarray`
            Current density distribution.

        Returns
        -------
        :class:`numpy.ndarray`
            Object error vector: :math:`\\mathbf{J} - \\chi \\mathbf{E}^{tot}`.
        """
        return csi.get_object_error(chi, total_field, current)

    def _get_data_error(self, scattered_field, green_function_s, current):
        r"""Calculate the data error functional.

        Computes the data error term in the ECSI cost function, which measures
        the discrepancy between measured and predicted scattered field data.

        Parameters
        ----------
        scattered_field : :class:`numpy.ndarray`
            Measured scattered electric field data.
        green_function_s : :class:`numpy.ndarray`
            Scattered field Green's function matrix.
        current : :class:`numpy.ndarray`
            Current density distribution.

        Returns
        -------
        :class:`numpy.ndarray`
            Data error vector: :math:`\\mathbf{E}^s - \\mathbf{G}^s \\mathbf{J}`.
        """
        return csi.get_data_error(scattered_field, green_function_s, current)

    def _get_gradient_j(self, green_function_s, data_error, normalization_s,
                      object_error, fftpa, chi, normalization_d):
        r"""Compute gradient of cost function with respect to current density.

        Calculates the gradient of the ECSI cost function with respect to the
        current density J, which is used in the conjugate gradient optimization
        for current updates.

        Parameters
        ----------
        green_function_s : :class:`numpy.ndarray`
            Scattered field Green's function matrix.
        data_error : :class:`numpy.ndarray`
            Data error vector from scattered field comparison.
        normalization_s : float
            Normalization factor for scattered field data.
        object_error : :class:`numpy.ndarray`
            Object error vector from current-contrast consistency.
        fftpa : :class:`fftproduct.FFTProduct`
            Adjoint FFT product operator for efficient computation.
        chi : :class:`numpy.ndarray`
            Diagonal matrix representation of contrast function.
        normalization_d : float
            Normalization factor for object error.

        Returns
        -------
        :class:`numpy.ndarray`
            Gradient vector with respect to current density.

        Notes
        -----
        The gradient combines contributions from both data and object error terms:

        .. math::
            \\nabla_J F = \\frac{2}{\\eta_s} \\mathbf{G}^{s*} \\mathbf{e}_s + 
            \\frac{2}{\\eta_d} \\mathbf{e}_o + \\frac{2}{\\eta_d} \\mathbf{G}^{D*} \\chi^* \\mathbf{e}_o
        """
        GDaXr = fftpa.compute(np.conj(chi) @ object_error)
        return csi.get_gradient(green_function_s, data_error, normalization_s,
                                object_error, GDaXr, normalization_d)

    def _get_gradient_x(self, normalization_d, current, total_field, chi):
        r"""Compute gradient of cost function with respect to contrast.

        Calculates the gradient of the ECSI cost function with respect to the
        contrast function \chi, which is used in the conjugate gradient optimization
        for contrast updates.

        Parameters
        ----------
        normalization_d : float
            Normalization factor for object error.
        current : :class:`numpy.ndarray`
            Current density distribution.
        total_field : :class:`numpy.ndarray`
            Total electric field distribution.
        chi : :class:`numpy.ndarray`
            Diagonal matrix representation of contrast function.

        Returns
        -------
        :class:`numpy.ndarray`
            Gradient vector with respect to contrast function.

        Notes
        -----
        The gradient is computed as:

        .. math::
            \\nabla_\\chi F = \\frac{2}{\\eta_d} \\left( \\frac{\\mathbf{J} \\cdot \\mathbf{E}^{tot*}}{||\\mathbf{E}^{tot}||^2} - \\chi \\right)
        """
        contrast = np.diag(chi, 0)
        return get_gradient_x(normalization_d, current, total_field, contrast)

    def _update_direction(self, gradient, gamma, direction):
        r"""Update search direction using conjugate gradient method.

        Computes the new search direction for the conjugate gradient optimization
        using the Polak-Ribière formula.

        Parameters
        ----------
        gradient : :class:`numpy.ndarray`
            Current gradient vector.
        gamma : float
            Conjugate gradient parameter (beta).
        direction : :class:`numpy.ndarray`
            Previous search direction.

        Returns
        -------
        :class:`numpy.ndarray`
            Updated search direction: :math:`\\mathbf{d}^{new} = -\\mathbf{g} + \\gamma \\mathbf{d}^{old}`.
        """
        return csi.update_direction(gradient, gamma, direction)

    def _get_constant_j(self, gradient_j, green_function_s, direction,
                        normalization_s, chi, fftpa, normalization_d):
        r"""Calculate optimal step size for current density update.

        Determines the optimal step size (alpha) for updating the current density
        along the conjugate gradient direction by minimizing the cost function.

        Parameters
        ----------
        gradient_j : :class:`numpy.ndarray`
            Gradient with respect to current density.
        green_function_s : :class:`numpy.ndarray`
            Scattered field Green's function matrix.
        direction : :class:`numpy.ndarray`
            Search direction for current density update.
        normalization_s : float
            Normalization factor for scattered field data.
        chi : :class:`numpy.ndarray`
            Diagonal matrix representation of contrast function.
        fftpa : :class:`fftproduct.FFTProduct`
            Adjoint FFT product operator.
        normalization_d : float
            Normalization factor for object error.

        Returns
        -------
        float
            Optimal step size for current density update.

        Notes
        -----
        The step size is computed by minimizing the quadratic approximation
        of the cost function along the search direction.
        """
        gv = gradient_j * np.conj(direction)
        GSv = green_function_s @ direction
        v_XGDva = direction - chi @ fftpa.compute(direction)
        constant = compute_constant_j(gv, GSv, normalization_s,
                                      v_XGDva, normalization_d)
        return constant
    
    def _get_constant_x(self, direction_x, total_field, chi, current,
                        incident_field):
        r"""Calculate optimal step size for contrast update.

        Determines the optimal step size (alpha) for updating the contrast
        function along the conjugate gradient direction by minimizing the cost function.

        Parameters
        ----------
        direction_x : :class:`numpy.ndarray`
            Search direction for contrast update.
        total_field : :class:`numpy.ndarray`
            Total electric field distribution.
        chi : :class:`numpy.ndarray`
            Diagonal matrix representation of contrast function.
        current : :class:`numpy.ndarray`
            Current density distribution.
        incident_field : :class:`numpy.ndarray`
            Incident electric field distribution.

        Returns
        -------
        float
            Optimal step size for contrast update.

        Notes
        -----
        The step size is computed by solving a quadratic equation derived from
        the cost function minimization along the search direction.
        """
        D = np.diag(direction_x.flatten(), 0)
        return compute_constant_x(D, total_field, chi, current, incident_field)

    def _update_total_field(self, current, incident_field, fftp):
        r"""Update total electric field from current density.

        Computes the total electric field by combining the incident field
        with the scattered field generated by the current density.

        Parameters
        ----------
        current : :class:`numpy.ndarray`
            Current density distribution.
        incident_field : :class:`numpy.ndarray`
            Incident electric field distribution.
        fftp : :class:`fftproduct.FFTProduct`
            FFT product operator for efficient Green's function computation.

        Returns
        -------
        :class:`numpy.ndarray`
            Updated total electric field: :math:`\\mathbf{E}^{tot} = \\mathbf{E}^{inc} + \\mathbf{G}^D \\mathbf{J}`.
        """
        GDJ = fftp.compute(current)
        return incident_field + GDJ

    def _update_contrast(self, chi, constant_x, direction_x):
        r"""Update contrast function using computed step size and direction.

        Updates the contrast function along the conjugate gradient direction
        with the optimal step size.

        Parameters
        ----------
        chi : :class:`numpy.ndarray`
            Current diagonal matrix representation of contrast function.
        constant_x : float
            Optimal step size for contrast update.
        direction_x : :class:`numpy.ndarray`
            Search direction for contrast update.

        Returns
        -------
        :class:`numpy.ndarray`
            Updated contrast function: :math:`\\chi^{new} = \\chi^{old} + \\alpha \\mathbf{d}_x`.
        """
        X = np.diag(chi, 0)
        return update_contrast(X, constant_x, direction_x)

    def _evaluate_objective_function(self, data_error, normalization_s,
                                     object_error, normalization_d):
        r"""Evaluate the ECSI objective function.

        Computes the value of the cost function combining data and object error terms.

        Parameters
        ----------
        data_error : :class:`numpy.ndarray`
            Data error vector from scattered field comparison.
        normalization_s : float
            Normalization factor for scattered field data.
        object_error : :class:`numpy.ndarray`
            Object error vector from current-contrast consistency.
        normalization_d : float
            Normalization factor for object error.

        Returns
        -------
        float
            Value of the objective function.

        Notes
        -----
        The objective function is:

        .. math::
            F = \\frac{||\\mathbf{e}_s||^2}{\\eta_s} + \\frac{||\\mathbf{e}_o||^2}{\\eta_d}
        """
        return csi.evaluate_objective_function(data_error, normalization_s,
                                               object_error, normalization_d)

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        r"""Print algorithm title and configuration information.

        Prints the solver title along with forward solver and stopping criteria
        information to the specified output stream.

        Parameters
        ----------
        inputdata : :class:`inputdata.InputData`
            Input data object containing problem configuration.
        discretization : :class:`discretization.Discretization`
            Discretization scheme for the problem.
        print_file : file-like object, optional
            Output stream for printing. Default is sys.stdout.
        """
        super()._print_title(inputdata, discretization, print_file=print_file)
        print(self.forward, file=print_file)
        print(self.stop_criteria, file=print_file)

    def save(self, file_path=''):
        r"""Save solver state to file.

        Saves the current solver configuration including forward solver and
        stopping criteria to a pickle file for later restoration.

        Parameters
        ----------
        file_path : str, optional
            Directory path where the file will be saved. Default is current directory.

        Notes
        -----
        The file is saved with the name specified by the solver's alias attribute.
        """
        data = super().save(file_path=file_path)
        data[FORWARD] = self.forward
        data[STOP_CRITERIA] = self.stop_criteria
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        r"""Load solver state from file.

        Restores the solver configuration from a previously saved pickle file.

        Parameters
        ----------
        file_name : str
            Name of the file containing saved solver state.
        file_path : str, optional
            Directory path containing the file. Default is current directory.

        Raises
        ------
        FileNotFoundError
            If the specified file cannot be found.
        """
        data = super().importdata(file_name, file_path=file_path)
        self.forward = data[FORWARD]
        self.stop_criteria= data[STOP_CRITERIA]

    def copy(self, new=None):
        r"""Create a copy of the solver.

        Creates a new solver instance with the same configuration or copies
        configuration from another solver instance.

        Parameters
        ----------
        new : :class:`ExtendedContrastSourceInversion`, optional
            Another solver instance to copy configuration from. If None,
            creates a new instance with current configuration.

        Returns
        -------
        :class:`ExtendedContrastSourceInversion`
            New solver instance with copied configuration.
        """
        if new is None:
            return ExtendedContrastSourceInversion(self.stop_criteria,
                                                   forward_solver=self.forward,
                                                   alias=self.alias)
        else:
            super().copy(new)
            self.forward = new.forward
            self.stop_criteria = new.stop_criteria

    def __str__(self):
        r"""Return string representation of the solver.

        Returns
        -------
        str
            String description including solver name, forward solver, and stopping criteria.
        """
        message = super().__str__()
        message += str(self.forward)
        message += str(self.stop_criteria)
        return message


@jit(nopython=True)
def get_gamma(g, glast):
    r"""Compute the Polak-Ribière conjugate gradient parameter.

    Calculates the beta parameter for the Polak-Ribière conjugate gradient method
    used to update search directions in the optimization process.

    Parameters
    ----------
    g : :class:`numpy.ndarray`
        Current gradient vector.
    glast : :class:`numpy.ndarray`
        Previous gradient vector.

    Returns
    -------
    float
        Polak-Ribière parameter: :math:`\\beta = \\frac{\\Re(\\mathbf{g}^T (\\mathbf{g} - \\mathbf{g}_{old}))}{||\\mathbf{g}_{old}||^2}`

    Notes
    -----
    This function is JIT-compiled with Numba for performance. The Polak-Ribière
    method generally provides better convergence properties than the Fletcher-Reeves
    method for nonlinear optimization problems.
    """
    return (np.real(np.sum(g * np.conj(g-glast)))
            / np.sum(glast * np.conj(glast)))

@jit(nopython=True)
def compute_constant_j(gv, GSv, eta_s, v_XGDva, eta_d):
    r"""Compute optimal step size for current density update.

    Calculates the optimal step size for updating the current density in the
    ECSI algorithm by minimizing the cost function along the search direction.

    Parameters
    ----------
    gv : :class:`numpy.ndarray`
        Product of gradient and conjugate direction for current density.
    GSv : :class:`numpy.ndarray`
        Green's function applied to search direction.
    eta_s : float
        Normalization factor for scattered field data.
    v_XGDva : :class:`numpy.ndarray`
        Auxiliary vector for object error computation.
    eta_d : float
        Normalization factor for object error.

    Returns
    -------
    float
        Optimal step size for current density update.

    Notes
    -----
    This function is JIT-compiled with Numba for performance. The step size
    minimizes the quadratic approximation of the cost function along the
    conjugate gradient direction.
    """
    t1 = np.sum(gv)
    t2 = np.sum(np.abs(GSv)**2) / eta_s
    t3 = np.sum(np.abs(v_XGDva)**2) / eta_d
    return -np.real(t1)/(t2 + t3) 

@jit(nopython=True)
def update_contrast(X, alpha, d):
    r"""Update contrast function with step size and direction.

    Updates the contrast function using the computed step size and search direction.

    Parameters
    ----------
    X : :class:`numpy.ndarray`
        Current contrast function values.
    alpha : float
        Step size for the update.
    d : :class:`numpy.ndarray`
        Search direction vector.

    Returns
    -------
    :class:`numpy.ndarray`
        Updated contrast function: :math:`X_{new} = X_{old} + \\alpha d`

    Notes
    -----
    This function is JIT-compiled with Numba for performance. It performs
    a simple linear update of the contrast function.
    """
    return X + alpha*d

@jit(nopython=True)
def get_gradient_x(eta_d, J, E, X):
    r"""Compute gradient of cost function with respect to contrast.

    Calculates the gradient of the ECSI cost function with respect to the
    contrast function using the current field and current density distributions.

    Parameters
    ----------
    eta_d : float
        Normalization factor for object error.
    J : :class:`numpy.ndarray`
        Current density distribution.
    E : :class:`numpy.ndarray`
        Electric field distribution.
    X : :class:`numpy.ndarray`
        Current contrast function values.

    Returns
    -------
    :class:`numpy.ndarray`
        Gradient vector with respect to contrast function.

    Notes
    -----
    This function is JIT-compiled with Numba for performance. The gradient
    is computed using the relationship between current density and contrast:

    .. math::
        \\nabla_X F = \\eta_d \\left( \\frac{\\mathbf{J} \\cdot \\mathbf{E}^*}{||\\mathbf{E}||^2} - X \\right)
    """
    gx = eta_d * (np.sum(J*np.conj(E), axis=1)/np.sum(np.abs(E)**2, axis=1)
                  - X.flatten())
    return gx

@jit(nopython=True)
def compute_constant_x(D, E, chi, J, Ei):
    r"""Compute optimal step size for contrast update.

    Calculates the optimal step size for updating the contrast function by
    solving a quadratic optimization problem along the search direction.

    Parameters
    ----------
    D : :class:`numpy.ndarray`
        Diagonal matrix representation of search direction.
    E : :class:`numpy.ndarray`
        Total electric field distribution.
    chi : :class:`numpy.ndarray`
        Current contrast function as diagonal matrix.
    J : :class:`numpy.ndarray`
        Current density distribution.
    Ei : :class:`numpy.ndarray`
        Incident electric field distribution.

    Returns
    -------
    float
        Optimal step size for contrast update.

    Notes
    -----
    This function is JIT-compiled with Numba for performance. The step size
    is computed by solving the quadratic equation:

    .. math::
        \\alpha = \\frac{-(aC - Ac) + \\sqrt{(aC - Ac)^2 - 4(aB - Ab)(bC - Bc)}}{2(aB - Ab)}

    where the coefficients are derived from the cost function minimization.
    """
    DE = D@E
    DEi = D@Ei
    XE_J = chi@E - J
    XEi = chi@Ei
    a = np.sum(np.abs(DE)**2)
    b = np.real(np.sum(XE_J*np.conj(DE)))
    c = np.sum(np.abs(XE_J)**2)
    A = np.sum(np.abs(DEi)**2)
    B = np.real(np.sum((XEi)*np.conj(DEi)))
    C = np.sum(np.abs(XEi)**2)
    return ((-(a*C-A*c) + np.sqrt((a*C-A*c)**2-4*(a*B-A*b)*(b*C-B*c)))
            / (2*(a*B-A*b)))