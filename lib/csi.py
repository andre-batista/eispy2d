r"""The Born Iterative Method.

This module implements the Born Iterative Method [1]_ as a derivation of
Solver class. The object contains an object of a forward solver and
one of linear inverse solver. The method solves the nonlinear
inverse problem iteratively. The implemented in
:class:`BornIterativeMethod`

References
----------
.. [1] Wang, Y. M., and Weng Cho Chew. "An iterative solution of the
   two-dimensional electromagnetic inverse scattering problem."
   International Journal of Imaging Systems and Technology 1.1 (1989):
   100-108.
"""

# Standard libraries
import time as tm
import numpy as np
import sys
import pickle
from numba import jit

# Developed libraries
import configuration as cfg
import inputdata as ipt
import result as rst
import deterministic as dtm
import collocation as clc
import mom_cg_fft as mom
import regularization as reg
import backprop as bp
import fftproduct

FORWARD = 'forward'
STOP_CRITERIA = 'stop criteria'


class ContrastSourceInversion(dtm.Deterministic):
    r"""The Born Interative Method (BIM).

    This class implements a classical nonlinear inverse solver [1]_. The
    method is based on coupling forward and inverse solvers in an
    iterative process. Therefore, it depends on the definition of a
    forward solver implementation and an linear inverse one.

    Attributes
    ----------
        forward : :class:`forward.Forward`:
            An implementation of the abstract class which defines a
            forward method which solves the total electric field.

        inverse : :class:`inverse.Inverse`:
            An implementation of the abstract class which defines method
            for solving the linear inverse scattering problem.

        MAX_IT : int
            The number of iterations.

        sc_measure : str
            Stop criterion for the algorithm. The algorithm will stop
            when the amount of variation from the current iteration in
            respect to the last one is below some threshold percentage:

            .. math:: \frac{|\zeta^i-\zeta^{i-1}|}{\zeta^{i-1}}*100
                      \leq \eta

        stopcriterion_measure : float
            Threshold criterion for stop the algorithm.

        divergence_tolerance : int, default: 5
            Number of iterations in which it will be accepted a
            divergence occurrence, i.e., the new solution has a larger
            evaluation than the previous considering the stop criterion
            measure.

    References
    ----------
    .. [1] Wang, Y. M., and Weng Cho Chew. "An iterative solution of the
       two dimensional electromagnetic inverse scattering problem."
       International Journal of Imaging Systems and Technology 1.1 (1989):
       100-108.
    """

    def __init__(self, stop_criteria, forward_solver=mom.MoM_CG_FFT(),
                 alias='csi', import_filename=None, import_filepath=''):
        r"""Create the object.

        Parameters
        ----------
            configuration : :class:`configuration.Configuration`
                It may be either an object of problem configuration or
                a string to a pre-saved file or a 2-tuple with the file
                name and path, respectively.

            version : str
                A string naming the version of this method. It may be
                useful when using different implementation of forward
                and inverse solvers.

            forward_solver : :class:`forward.Forward`
                An implementation of the abstract class Forward which
                defines a method for computing the total intern field.

            inverse_solver : :class:`inverse.Inverse`
                An implementation of the abstract class Inverse which
                defines a method for solving the linear inverse problem.

            maximum_iterations : int, default: 10
                Maximum number of iterations.

            stopcriterion_measure : str, default: None
                Define the measure for stop criterion. The algorithm
                will stop when the amount of variation from the current
                iteration in respect to the last one is below some
                threshold percentage:

                .. math:: \frac{|\zeta^i-\zeta^{i-1}|}{\zeta^{i-1}}*100
                          \leq \eta

            stopcriterion_measure : float, default: 1e-3
                Threshold criterion for stop the algorithm.
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
        """Solve a nonlinear inverse problem.

        Parameters
        ----------
            instance : :class:`inputdata.InputData`
                An object which defines a case problem with scattered
                field and some others information.

            print_info : bool
                Print or not the iteration information.
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
        return get_object_error(chi, total_field, current)

    def _get_data_error(self, scattered_field, green_function_s, current):
        return get_data_error(scattered_field, green_function_s, current)

    def _get_gradient(self, green_function_s, data_error, normalization_s,
                      object_error, fftpa, chi, normalization_d):
        GDaXr = fftpa.compute(np.conj(chi) @ object_error)
        return get_gradient(green_function_s, data_error, normalization_s,
                            object_error, GDaXr, normalization_d)

    def _update_direction(self, gradient, gamma, direction):
        N = gradient.shape[0]
        gamma = np.tile(gamma.reshape((1, -1)), (N, 1))
        return update_direction(gradient, gamma, direction)

    def _get_constant(self, data_error, green_function_s, direction,
                      normalization_s, object_error, chi, fftp,
                      normalization_d):
        N = green_function_s.shape[1]
        XGDv = chi @ fftp.compute(direction)
        v_XGDv = direction - XGDv
        GSv = green_function_s @ direction
        constant = compute_constant(data_error, GSv, normalization_s,
                                    object_error, v_XGDv, normalization_d)
        return np.tile(constant.reshape((1, -1)), (N, 1))

    def _update_total_field(self, total_field, constant, fftp, direction):
        GDv = fftp.compute(direction)
        return update_total_field(total_field, constant, GDv)

    def _compute_contrast(self, current, total_field):
        return compute_contrast(current, total_field) + 0j

    def _evaluate_objective_function(self, data_error, normalization_s,
                                     object_error, normalization_d):
        return evaluate_objective_function(data_error, normalization_s,
                                           object_error, normalization_d)

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        super()._print_title(inputdata, discretization, print_file=print_file)
        print(self.forward, file=print_file)
        print(self.stop_criteria, file=print_file)

    def save(self, file_path=''):
        data = super().save(file_path=file_path)
        data[FORWARD] = self.forward
        data[STOP_CRITERIA] = self.stop_criteria
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path=file_path)
        self.forward = data[FORWARD]
        self.stop_criteria= data[STOP_CRITERIA]

    def copy(self, new=None):
        if new is None:
            return ContrastSourceInversion(self.stop_criteria,
                                           forward_solver=self.forward,
                                           alias=self.alias)
        else:
            super().copy(new)
            self.forward = new.forward
            self.stop_criteria = new.stop_criteria

    def __str__(self):
        message = super().__str__()
        message += str(self.forward)
        message += str(self.stop_criteria)
        return message


@jit(nopython=True)
def get_data_error(Es, GS, J):
    return Es - GS @ J

@jit(nopython=True)
def get_object_error(chi, E, J):
    return chi @ E - J

@jit(nopython=True)
def get_normalization_s(Es):
    return np.sum(np.abs(Es)**2)

@jit(nopython=True)
def get_normalization_d(chi, Ei):
    return np.sum(np.abs(chi @ Ei)**2)

@jit(nopython=True)
def get_gradient(GS, rho, eta_s, r, GDaXr, eta_d):
    return - GS.conj().T @ rho / eta_s - (r - GDaXr) / eta_d

@jit(nopython=True)
def get_gamma(g, glast):
    return (np.sum(g * np.conj(g-glast), axis=0)
            / np.sum(glast * np.conj(glast), axis=0))

@jit(nopython=True)
def update_direction(g, gamma, v):
    return g + gamma*v

@jit(nopython=True)
def compute_constant(rho, GSv, eta_s, r, v_XGDv, eta_d):
    t1 = np.sum(rho * np.conj(GSv), axis=0)/eta_s
    t2 = np.sum(r * np.conj(v_XGDv), axis=0)/eta_d
    t3 = np.sum(np.abs(GSv)**2)/eta_s
    t4 = np.sum(np.abs(v_XGDv)**2)/eta_d
    return (t1 + t2)/(t3 + t4)

@jit(nopython=True)
def update_current(J, alpha, v):
    return J + alpha * v

@jit(nopython=True)
def update_total_field(E, alpha, GDv):
    return E + alpha * GDv

@jit(nopython=True)
def compute_contrast(J, E):
    den = np.sum(np.abs(E)**2, axis=1)
    num = J * np.conj(E)
    Xr = np.sum(np.real(num), axis=1)/den
    Xi = np.sum(np.imag(num), axis=1)/den
    return Xr + 1j*Xi

@jit(nopython=True)
def evaluate_objective_function(rho, eta_s, r, eta_d):
    return np.sum(np.abs(rho)**2)/eta_s + np.sum(np.abs(r)**2)/eta_d