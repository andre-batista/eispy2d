r"""The Born Iterative Method.

This module implements the Born Iterative Method [1]_ as a derivation of
Solver class. The object contains an object of a forward solver and
one of linear inverse solver. The method solves the nonlinear
inverse problem iteratively. The implemented in
:class:`BornIterativeMethod`

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
import sys
import pickle
from numba import jit
from scipy.optimize import minimize_scalar

# Developed libraries
import configuration as cfg
import result as rst
import deterministic as dtm
import mom_cg_fft as mom
import regularization as reg
import csi
import ecsi
import backprop as bp
import fftproduct

FORWARD = 'forward'
STOP_CRITERIA = 'stop criteria'
EXPONENT = 'exponent'


class MRContrastSourceInversion(dtm.Deterministic):
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
       two‐dimensional electromagnetic inverse scattering problem."
       International Journal of Imaging Systems and Technology 1.1 (1989):
       100-108.
    """

    def __init__(self, stop_criteria, exponent=1.,
                 forward_solver=mom.MoM_CG_FFT(),
                 alias='mrcsi', import_filename=None, import_filepath=''):
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
            self.name = ('Multiplicative Regularization '
                         + 'Contrast Source Inversion')
            self.forward = forward_solver
            self.exponent = exponent
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
        direction_j = np.zeros((N, NS), dtype=complex)
        direction_x = np.zeros(N, dtype=complex)
        last_gradient_j = np.ones((N, NS), dtype=complex)
        last_gradient_x = np.ones(N, dtype=complex)
        exponent = self.exponent
        dx = inputdata.configuration.Lx/discretization.elements[1]
        dy = inputdata.configuration.Ly/discretization.elements[0]
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
            gamma_j = ecsi.get_gamma(gradient_j, last_gradient_j)
            direction_j = self._update_direction(gradient_j, gamma_j, direction_j)
            constant_j = self._get_constant_j(gradient_j, discretization.GS,
                                              direction_j,normalization_s, chi,
                                              fftpa, normalization_d)
            current = csi.update_current(current, constant_j, direction_j)
            total_field = self._update_total_field(current, incident_field,
                                                   fftp)
            
            delta_square = self._compute_delta_square(chi, incident_field,
                                                      current, fftp,
                                                      normalization_d)

            gradient_contrast = np.gradient(
                np.diag(chi).reshape(discretization.elements), dx, dy
            )
            data_error = self._get_data_error(inputdata.scattered_field,
                                              discretization.GS, current)

            gradient_x = self._get_gradient_x(chi, total_field, current,
                                              normalization_d,
                                              gradient_contrast, dx, dy,
                                              exponent, data_error,
                                              normalization_s, delta_square)

            gamma_x = ecsi.get_gamma(gradient_x, last_gradient_x)
            direction_x = self._update_direction(gradient_x, gamma_x,
                                                 direction_x)
            constant_x = self._get_constant_x(direction_x, total_field, chi,
                                              current, incident_field,
                                              discretization, dx, dy,
                                              data_error, normalization_s,
                                              gradient_contrast, delta_square,
                                              exponent)
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
        return csi.get_object_error(chi, total_field, current)

    def _get_data_error(self, scattered_field, green_function_s, current):
        return csi.get_data_error(scattered_field, green_function_s, current)

    def _get_gradient_j(self, green_function_s, data_error, normalization_s,
                      object_error, fftpa, chi, normalization_d):
        GDaXr = fftpa.compute(np.conj(chi) @ object_error)
        return csi.get_gradient(green_function_s, data_error, normalization_s,
                                object_error, GDaXr, normalization_d)

    def _get_gradient_x(self, chi, total_field, current, normalization_d,
                        gradient_contrast, dx, dy, exponent, data_error,
                        normalization_s, delta_square):
        return get_gradient_x(chi, total_field, current, normalization_d,
                              gradient_contrast[0], gradient_contrast[1], dx,
                              dy, exponent, data_error, normalization_s,
                              delta_square)

    def _update_direction(self, gradient, gamma, direction):
        return csi.update_direction(gradient, gamma, direction)

    def _get_constant_j(self, gradient_j, green_function_s, direction,
                        normalization_s, chi, fftpa, normalization_d):
        gv = gradient_j * np.conj(direction)
        GSv = green_function_s @ direction
        v_XGDva = direction - chi @ fftpa.compute(direction)
        constant = ecsi.compute_constant_j(gv, GSv, normalization_s,
                                           v_XGDva, normalization_d)
        return constant
    
    def _get_constant_x(self, direction_x, total_field, chi, current,
                        incident_field, discretization, dx, dy, data_error,
                        normalization_s, gradient_contrast, delta_square,
                        exponent):

        D = np.diag(direction_x.flatten(), 0)
        gradd = np.gradient(direction_x.reshape(discretization.elements),
                            dx, dy)
        norm_res = np.sum(np.abs(data_error)**2)
        
        def fun(x):
            t1 = (norm_res/normalization_s
                  + np.sum(np.abs((chi + x*D)@total_field-current)**2)
                  / np.sum(np.abs((chi + x*D)@incident_field)**2))
            t2 = np.trapz(
                np.trapz(
                    np.sqrt((gradient_contrast[0]+x*gradd[0])**2
                            + (gradient_contrast[1]+x*gradd[1])**2
                            + delta_square)**exponent, dx=dy
                ), dx=dx
            )
            return t1*t2
        
        sol = minimize_scalar(fun, method='brent')
        return sol.x

    def _update_total_field(self, current, incident_field, fftp):
        GDJ = fftp.compute(current)
        return incident_field + GDJ

    def _update_contrast(self, chi, constant_x, direction_x):
        X = np.diag(chi, 0)
        return ecsi.update_contrast(X, constant_x, direction_x)

    def _evaluate_objective_function(self, data_error, normalization_s,
                                     object_error, normalization_d):
        return csi.evaluate_objective_function(data_error, normalization_s,
                                               object_error, normalization_d)

    def _compute_delta_square(self, chi, incident_field, current, fftp,
                              normalization_d):
        GDJ = fftp.compute(current)
        return compute_delta_square(chi, incident_field, current, GDJ,
                                    normalization_d)

    def _print_title(self, inputdata, discretization, print_file=sys.stdout):
        super()._print_title(inputdata, discretization, print_file=print_file)
        print(self.forward, file=print_file)
        print(self.stop_criteria, file=print_file)
        print('p = {:.2}'.format(self.exponent), file=print_file)

    def save(self, file_path=''):
        data = super().save(file_path=file_path)
        data[FORWARD] = self.forward
        data[STOP_CRITERIA] = self.stop_criteria
        data[EXPONENT] = self.exponent
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path=file_path)
        self.forward = data[FORWARD]
        self.stop_criteria= data[STOP_CRITERIA]
        self.exponent = data[EXPONENT]

    def copy(self, new=None):
        if new is None:
            return MRContrastSourceInversion(self.stop_criteria,
                                             forward_solver=self.forward, 
                                             exponent=self.exponent,
                                             alias=self.alias)
        else:
            super().copy(new)
            self.forward = new.forward
            self.stop_criteria = new.stop_criteria
            self.exponent = new.exponent

    def __str__(self):
        message = super().__str__()
        message += str(self.forward)
        message += str(self.stop_criteria)
        message += 'p = {:.2}'.format(self.exponent)
        return message


@jit(nopython=True)
def compute_delta_square(chi, Ei, J, GDJ, eta_d):
    return np.sum(np.abs(chi @ Ei - J + chi @ GDJ)**2)/eta_d

@jit(nopython=True)
def get_gradient_x(chi, E, J, eta_d, gradXx, gradXy, dx, dy, p, rho, eta_s, d2):
    aux1 = gradXx**2 + gradXy**2 + d2
    Ftv = np.trapz(np.trapz(np.sqrt(aux1)**p, dx=dy), dx=dx)
    gd = -np.sum((chi@E - J)*np.conj(E), axis=1)
    aux2 = np.sqrt(aux1)**(p-2)
    gtv = p/2*(aux2*gradXx + aux2*gradXy)
    gtv = gtv.flatten()
    F = np.sum(np.abs(rho)**2)/eta_s + d2
    return (eta_d*Ftv*gd + F*gtv)/np.sum(np.abs(E)**2)