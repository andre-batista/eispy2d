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
from scipy.linalg import norm
from scipy.sparse import spdiags
from numpy.linalg import svd
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


class SubspaceBasedOptimizationMethod(dtm.Deterministic):
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

    def __init__(self, stop_criteria, cutoff_index=5,
                 forward_solver=mom.MoM_CG_FFT(),  alias='som',
                 import_filename=None, import_filepath=''):
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
            self.name = 'Subspace-based Optimization Method'
            self.forward = forward_solver
            self.stop_criteria = stop_criteria
            self.cutoff_index = cutoff_index

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
            contrast, current = self._get_initial_guess(inputdata,
                                                             discretization)
        else:
            contrast = discretization.contrast_image(initial_guess,
                                                     discretization.elements)

        execution_time = tm.time()-tic

        fftp = fftproduct.FFTProduct(discretization=discretization,
                                     adjoint=False)
        fftpa = fftproduct.FFTProduct(discretization=discretization,
                                      conjugate=True)

        # If the same object is used for different resolution instances,
        # then some parameters may need to be updated within the inverse
        # solver. So, the next line ensures it:
        current_evaluations = 0
        iteration = 0
        objective_function = np.inf
        base, power = 1, 0
        last_message_printed = False

        N, NS = np.prod(discretization.elements), inputdata.configuration.NS
        incident_field = self.forward.incident_field(discretization.elements,
                                                     inputdata.configuration)
        L = self.cutoff_index
        
        output = initial_parameters(discretization.GS, L,
                                    inputdata.scattered_field)
        J_po=output[0]
        Gs_V_ne=output[1]
        alpha_ne = output[2]
        alpha_neo=output[3]
        rho=output[4]
        grad=output[5]
        del_dat=output[6]
        E_s_norm_sq=output[7]
        J_po_norm_sq=output[8]
        
        X = contrast.reshape((-1, 1))
        Xe = np.tile(X, (1, NS))
        Ei = incident_field
        B = Xe * (Ei + fftp.compute(J_po)) - J_po 
        del_sta = Gs_V_ne @ alpha_ne - Xe * fftp.compute(Gs_V_ne@alpha_ne) - B
        E_po = Ei + fftp.compute(J_po)
        grado = np.zeros((N-L, NS))
        rhoo = np.zeros((N-L, NS))

        while (not self.stop_criteria.stop(current_evaluations, iteration,
                                           objective_function)):

            iteration_message = 'Iteration: %d - ' % (iteration+1)

            tic = tm.time()

            grad = self._get_gradient(discretization.GS, Gs_V_ne, del_dat,
                                      E_s_norm_sq, Xe, del_sta,fftpa,
                                      J_po_norm_sq)

            rho = self._get_rho(iteration+1, grad, grado, N, L, rhoo)

            alpha_ne = self._get_alpha(discretization.GS, Gs_V_ne, rho,
                                       del_dat, E_s_norm_sq, Xe, fftp, del_sta,
                                       J_po_norm_sq, alpha_neo)

            J = compute_J(J_po, Gs_V_ne, alpha_ne)
            Et = Ei + fftp.compute(J)
            X = compute_X(Et, J, J_po_norm_sq)
            Xe = np.tile(X.reshape((-1, 1)), (1, NS))
            
            grado = grad.copy()
            alpha_neo = alpha_ne.copy()
            rhoo = rho.copy()

            B, del_dat, del_sta = self._update_last_parameters(
                Xe, E_po, J_po, discretization.GS, Gs_V_ne, alpha_ne,
                inputdata.scattered_field, fftp
            )

            objective_function = compute_objective_function(del_dat,
                                                            E_s_norm_sq,
                                                            del_sta,
                                                            J_po_norm_sq)

            execution_time +=  tm.time()-tic
            contrast = X.reshape(discretization.elements)
            contrast = discretization.contrast_image(contrast,
                                                     inputdata.resolution)
            total_field = Et
            scattered_field = discretization.GS@J

            if inputdata.configuration.good_conductor:
                contrast = 1j*contrast.imag
            if inputdata.configuration.perfect_dielectric:
                contrast = contrast.real

            if inputdata.total_field is not None:
                total_field = discretization.total_image(total_field,
                                                         inputdata.resolution)
            else:
                total_field = total_field

            result.update_error(inputdata, scattered_field=scattered_field,
                                total_field=total_field, contrast=contrast,
                                objective_function=objective_function)

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
        result.scattered_field = scattered_field
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
        return contrast, current

    def _get_gradient(self, GS, Gs_V_ne, del_dat, E_s_norm_sq, Xe, del_sta,
                      fftpa, J_po_norm_sq):
        L = self.cutoff_index
        N = GS.shape[1]
        t1, t2a, t2c = gradient_terms(GS, Gs_V_ne, del_dat, E_s_norm_sq, N, L,
                                      del_sta, J_po_norm_sq)
        t2b = -Gs_V_ne.conj().T @ fftpa.compute(np.conj(Xe)*(del_sta))
        t2 = (t2a + t2b) / t2c
        return  t1 + t2

    def _get_rho(self, iteration, grad, grado, N, L, rhoo):
        if iteration == 1:
            return grad.copy() 
        else:
            return compute_rho(grad, grado, N, L, rhoo)
        
    def _get_alpha(self, GS, Gs_V_ne, rho, del_dat, E_s_norm_sq, Xe, fftp,
                   del_sta, J_po_norm_sq, alpha_neo):
        N = GS.shape[1]
        L = self.cutoff_index
        GDGs_V_nerho =  fftp.compute(Gs_V_ne@rho)
        return compute_alpha(GS, Gs_V_ne, rho, del_dat, E_s_norm_sq, Xe,
                             GDGs_V_nerho, del_sta, J_po_norm_sq, alpha_neo, N,
                             L)

    def _update_last_parameters(self, Xe, E_po, J_po, GS, Gs_V_ne, alpha_ne, Es, fftp):
        Gs_V_nealpha_ne = Gs_V_ne@alpha_ne
        GDGs_V_nealpha_ne = fftp.compute(Gs_V_nealpha_ne)
        return update_last_parameters(Xe, E_po, J_po, GS, Gs_V_nealpha_ne, Es,
                                      GDGs_V_nealpha_ne)


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
            return SubspaceBasedOptimizationMethod(
                self.stop_criteria, forward_solver=self.forward,
                cutoff_index=self.cutoff_index, alias=self.alias
            )
        else:
            super().copy(new)
            self.forward = new.forward
            self.stop_criteria = new.stop_criteria
            self.cutoff_index = new.cutoff_index

    def __str__(self):
        message = super().__str__()
        message += str(self.forward)
        message += str(self.stop_criteria)
        message += 'Cut-off index: %d' % self.cutoff_index
        return message


def initial_parameters(GS, L, Es):
    NS = Es.shape[1]
    N = GS.shape[1]
    Gs_U, Gs_S, Gs_V = svd(GS)
    Gs_V = Gs_V.conj().T
    Gs_S_tile = 0j*np.ones((L, NS))
    for l in range(L):
        Gs_S_tile[l, :] = Gs_S[l]
    alpha_po = Gs_U[:, :L].conj().T @ Es / Gs_S_tile
    J_po = Gs_V[:, :L] @ alpha_po
    Gs_V_ne = Gs_V[:, L:]
    alpha_ne = 0j*np.ones((N-L, NS))
    alpha_neo = 0j*np.ones((N-L, NS))
    rho = 0j*np.ones((N-L, NS))
    grad = 0j*np.ones((N-L, NS))
    del_dat = GS @ (Gs_V_ne @ alpha_ne) + GS @ J_po - Es
    E_s_norm_sq = np.sum(np.abs(Es)**2, axis=0)
    J_po_norm_sq = np.sum(np.abs(J_po)**2, axis=0)

    return (J_po, Gs_V_ne, alpha_ne, alpha_neo, rho, grad, del_dat,
            E_s_norm_sq, J_po_norm_sq)

@jit(nopython=True)
def gradient_terms(GS, Gs_V_ne, del_dat, E_s_norm_sq, N, L, del_sta,
                   J_po_norm_sq):
    E_s_norm_sq_tile = 0j*np.ones((N-L, E_s_norm_sq.size))
    for n in range(N-L):
        E_s_norm_sq_tile[n, :] = E_s_norm_sq
    t1 = np.conj(GS@Gs_V_ne).T @ (del_dat) / E_s_norm_sq_tile
    t2a = Gs_V_ne.conj().T @ (del_sta)
    t2c = 0j*np.ones((N-L, J_po_norm_sq.size))
    for n in range(N-L):
        t2c[n, :] = J_po_norm_sq
    return t1, t2a, t2c

@jit(nopython=True)
def compute_rho(grad, grado, N, L, rhoo):
    
    aux = np.real(np.sum(np.conj(grad - grado)*grad,axis=0))/np.sum(np.abs(grado)**2, axis=0)
    aux_tile = 0j*np.ones((N-L, aux.size))
    for n in range(N-L):
        aux_tile[n, :] = aux
    return (grad + aux_tile*rhoo)

@jit(nopython=True)
def compute_alpha(GS, Gs_V_ne, rho, del_dat, E_s_norm_sq, Xe, GDGs_V_nerho,
                  del_sta, J_po_norm_sq, alpha_neo, N, L):
    aux1 = Gs_V_ne@rho
    aux2 = GS@aux1
    aux3 = aux1 - Xe*GDGs_V_nerho
    num = (-np.sum(np.conj(aux2)*del_dat, axis=0)/E_s_norm_sq
           - np.sum(np.conj(aux3)*del_sta, axis=0)/J_po_norm_sq)
    den = (np.sum(np.abs(aux2)**2, axis=0)/E_s_norm_sq
           + np.sum(np.abs(aux3)**2, axis=0)/J_po_norm_sq)
    aux4 = num/den
    aux4_tile = 0j*np.ones((N-L, aux4.size))
    for n in range(N-L):
        aux4_tile[n, :] = aux4
    return alpha_neo + aux4_tile * rho

@jit(nopython=True)
def compute_J(J_po, Gs_V_ne, alpha_ne):
    return J_po + Gs_V_ne @ alpha_ne

@jit(nopython=True)
def compute_X(Et, J, J_po_norm_sq):
    N = J.shape[0]
    Etconj = Et.conj()
    den = 0j*np.ones((N, J_po_norm_sq.size))
    for n in range(N):
        den[n, :] = J_po_norm_sq
    chi_num = np.sum(Etconj*J/den, axis=1)
    chi_den = np.sum(Etconj*Et/den, axis=1)
    return chi_num/chi_den

@jit(nopython=True)
def update_last_parameters(Xe, E_po, J_po, GS, Gs_V_nealpha_ne, Es,
                           GDGs_V_nealpha_ne):   
    B = Xe*E_po - J_po
    del_dat = GS@Gs_V_nealpha_ne + GS@J_po - Es
    del_sta = Gs_V_nealpha_ne - Xe*GDGs_V_nealpha_ne - B
    return B, del_dat, del_sta

@jit(nopython=True)
def compute_objective_function(del_dat, E_s_norm_sq, del_sta, J_po_norm_sq):
    objectf1 = np.sum(np.sum((np.abs(del_dat))**2, axis=0)/E_s_norm_sq)
    objectf2 = np.sum(np.sum((np.abs(del_sta))**2, axis=0)/J_po_norm_sq)
    return objectf1 + objectf2