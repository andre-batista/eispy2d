r"""The Conjugated Gradient Method.

This module implements the Conjugated Gradient Method [1]_ as a derivation of
Solver class. The method solves the nonlinear inverse problem iteratively.

References
----------
.. [1] Lobel, P., et al. "Conjugate gradient method for solving inverse
   scattering with experimental data." IEEE Antennas and Propagation
   Magazine 38.3 (1996): 48-51.   
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
import configuration as cfg
import inputdata as ipt
import result as rst
import deterministic as dtm
import mom_cg_fft as mom
import osm

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

    def __init__(self, initial_guess, step, stop_criteria,
                 alias='cgm', import_filename=None, import_filepath=''):
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
            self.name = 'Conjugated Gradient Method'
            self.initial_guess = initial_guess
            self.step = step
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
        super()._print_title(inputdata, discretization, print_file=print_file)
        print('Initial guess: ' + self.initial_guess, file=print_file)
        print('Step: ' + self.step, file=print_file)
        print(self.stop_criteria, file=print_file)

    def save(self, file_path=''):
        data = super().save(file_path=file_path)
        data[INITIAL_GUESS] = self.initial_guess
        data[STEP] = self.step
        data[STOP_CRITERIA] = self.stop_criteria
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        data = super().importdata(file_name, file_path=file_path)
        self.initial_guess = data[INITIAL_GUESS]
        self.step = data[STEP]
        self.stop_criteria= data[STOP_CRITERIA]

    def copy(self, new=None):
        if new is None:
            return ConjugatedGradientMethod(self.initial_guess, self.step,
                                            self.stop_criteria, alias=self.alias)
        else:
            super().copy(new)
            self.initial_guess = new.initial_guess
            self.step = new.step
            self.stop_criteria = new.stop_criteria

    def __str__(self):
        message = super().__str__()
        message += 'Initial guess: ' + self.initial_guess + '\n'
        message += 'Step: ' + self.step + '\n'
        message += str(self.stop_criteria)
        return message