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
from scipy.linalg import norm
import sys
import pickle

# Developed libraries
import configuration as cfg
import inputdata as ipt
import result as rst
import deterministic as dtm


FORWARD = 'forward'
REGULARIZATION = 'regularization'
STOP_CRITERIA = 'stop criteria'


class BornIterativeMethod(dtm.Deterministic):
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

    def __init__(self, forward_solver, regularization, stop_criteria,
                 alias='bim'):
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
        super().__init__(alias=alias, parallelization=None)
        self.name = 'Born Iterative Method'
        self.forward = forward_solver
        self.regularization = regularization
        self.stop_criteria = stop_criteria

    def solve(self, inputdata, discretization, print_info=True,
              print_file=sys.stdout):
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
        super()._print_title(inputdata, discretization, print_file=print_file)
        print(self.forward, file=print_file)
        print(self.regularization, file=print_file)
        print(self.stop_criteria, file=print_file)

    def save(self, file_path=''):
        data = super().save(file_path=file_path)
        data[FORWARD] = self.forward
        data[REGULARIZATION] = self.regularization
        data[STOP_CRITERIA] = self.stop_criteria
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path=file_path)
        self.forward = data[FORWARD]
        self.regularization = data[REGULARIZATION]
        self.stop_criteria= data[STOP_CRITERIA]

    def copy(self, new=None):
        if new is None:
            return BornIterativeMethod(self.forward, self.regularization,
                                       self.stop_criteria, alias=self.alias)
        else:
            super().copy(new)
            self.forward = new.forward
            self.regularization = new.regularization
            self.stop_criteria = new.stop_criteria

    def __str__(self):
        message = super().__str__()
        message += str(self.forward)
        message += str(self.regularization)
        message += str(self.stop_criteria)
        return message
            