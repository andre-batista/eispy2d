r"""Subspace-Based Optimization Method (SOM) for inverse scattering.

This module implements the Subspace-Based Optimization Method [1]_ as a
derivation of the Solver class. The key idea is to decompose the space of
contrast sources (induced currents) into two orthogonal complementary
subspaces: a *signal subspace*, whose contribution to the contrast source
is determined analytically via spectral analysis of the data operator
(without any optimization), and a *noise subspace*, whose contribution is
recovered by a conjugate-gradient optimization. This decomposition
significantly accelerates convergence and endows the algorithm with
robustness against measurement noise. The implemented class is
:class:`SubspaceBasedOptimizationMethod`.

References
----------
.. [1] X. Chen, "Subspace-Based Optimization Method for Solving
   Inverse-Scattering Problems," IEEE Transactions on Geoscience and
   Remote Sensing, vol. 48, no. 1, pp. 42-49, Jan. 2010,
   doi: 10.1109/TGRS.2009.2025122.
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
from eispy2d.core import configuration as cfg
from eispy2d.core import inputdata as ipt
from eispy2d.core import result as rst
from eispy2d.solvers.base import deterministic as dtm
from eispy2d.discretization import collocation as clc
from eispy2d.solvers.forward import mom_cg_fft as mom
from eispy2d.solvers.inverse import regularization as reg
from eispy2d.solvers.inverse import backprop as bp
from eispy2d.solvers.forward import fftproduct as fftproduct

FORWARD = 'forward'
STOP_CRITERIA = 'stop criteria'


class SubspaceBasedOptimizationMethod(dtm.Deterministic):
    r"""Subspace-Based Optimization Method (SOM).

    This class implements the Subspace-Based Optimization Method [1]_
    for solving nonlinear electromagnetic inverse scattering problems.
    The space of contrast sources is split into two orthogonal
    complementary subspaces via singular-value decomposition (SVD) of
    the data operator:

    * **Signal subspace** (first ``cutoff_index`` singular vectors):
      the projection of the contrast source onto this subspace is
      determined analytically from the measured scattered field, without
      any optimization step.
    * **Noise subspace** (remaining singular vectors): the projection
      onto this subspace is recovered iteratively by a
      conjugate-gradient minimization of a normalized cost functional.

    This partitioning makes convergence significantly faster than
    contrast-source inversion and yields robustness against noise.

    Attributes
    ----------
        forward : :class:`forward.Forward`:
            An implementation of the abstract Forward class used to
            compute the incident electric field.

        stop_criteria : stop-criterion object
            Object controlling the termination condition of the
            iterative loop.

        cutoff_index : int
            Number of singular values/vectors retained to span the
            signal subspace (denoted *L* in [1]_). Increasing this
            value incorporates more spectral content into the
            analytically determined component of the contrast source.

    References
    ----------
    .. [1] X. Chen, "Subspace-Based Optimization Method for Solving
       Inverse-Scattering Problems," IEEE Transactions on Geoscience
       and Remote Sensing, vol. 48, no. 1, pp. 42-49, Jan. 2010,
       doi: 10.1109/TGRS.2009.2025122.
    """

    def __init__(self, stop_criteria, cutoff_index=5,
                 forward_solver=mom.MoM_CG_FFT(),  alias='som',
                 import_filename=None, import_filepath=''):
        r"""Create the object.

        Parameters
        ----------
            stop_criteria : stop-criterion object
                Object that controls the termination of the iterative
                loop. It must expose a ``stop(evaluations, iteration,
                objective_function)`` method returning ``True`` when
                convergence is detected.

            cutoff_index : int, default: 5
                Number of dominant singular values/vectors (denoted *L*
                in [1]_) used to span the signal subspace. The
                corresponding contrast-source component is determined
                analytically; the remainder is found by optimization.

            forward_solver : :class:`forward.Forward`, optional
                An implementation of the abstract Forward class used to
                compute the incident electric field. Defaults to
                :class:`mom_cg_fft.MoM_CG_FFT`.

            alias : str, default: ``'som'``
                Short identifier for the solver, used when saving
                results to disk.

            import_filename : str or None, default: None
                If provided, the solver state is restored from a
                previously saved file instead of being initialized
                from scratch.

            import_filepath : str, default: ``''``
                Directory path where the import file is located.
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
        """Solve the nonlinear inverse scattering problem.

        Executes the SOM iterative loop. At each iteration, the
        noise-subspace coefficients are updated by a conjugate-gradient
        step while the signal-subspace component of the contrast source
        remains fixed (determined analytically from the SVD of the data
        operator). The contrast profile is then updated in closed form
        from the total contrast source.

        Parameters
        ----------
            inputdata : :class:`inputdata.InputData`
                Object containing the measured scattered field, the
                problem configuration, and any target indicators.

            discretization : discretization object
                Object that describes the spatial discretization of the
                investigation domain (grid elements, Green's functions,
                interpolation methods).

            print_info : bool, default: True
                Whether to print iteration progress to ``print_file``.

            print_file : file-like object, default: ``sys.stdout``
                Destination for iteration messages.

            initial_guess : array-like or None, default: None
                Initial contrast profile. If ``None``, the
                back-propagation method is used to generate a first
                estimate.

        Returns
        -------
        result : :class:`result.Result`
            Object containing the reconstructed contrast, relative
            permittivity, conductivity, scattered field, and optional
            performance indicators (execution time, iteration count).
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
        """Save the SOM solver state to file.

        Serializes the forward solver and stop criteria using pickle.

        Parameters
        ----------
        file_path : str, default: ''
            Directory where the state file is written. The file is named
            after the solver's alias.
        """
        data = super().save(file_path=file_path)
        data[FORWARD] = self.forward
        data[STOP_CRITERIA] = self.stop_criteria
        with open(file_path + self.alias, 'wb') as datafile:
            pickle.dump(data, datafile)

    def importdata(self, file_name, file_path=''):
        """Import SOM solver state from file.

        Restores the forward solver and stop criteria previously saved
        with :meth:`save`.

        Parameters
        ----------
        file_name : str
            Name of the file containing the saved solver state.
        file_path : str, default: ''
            Directory containing the file.
        """
        data = super().importdata(file_name, file_path=file_path)
        self.forward = data[FORWARD]
        self.stop_criteria= data[STOP_CRITERIA]

    def copy(self, new=None):
        """Create a copy of this SOM instance.

        Parameters
        ----------
        new : SubspaceBasedOptimizationMethod, optional
            Existing instance to copy attributes into. If ``None``,
            a new instance is created and returned.

        Returns
        -------
        SubspaceBasedOptimizationMethod or None
            A new instance when `new` is ``None``; otherwise ``None``
            (the provided instance is modified in place).
        """
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
    r"""Initialize the SOM subspace decomposition parameters.

    Performs the SVD of the data operator ``GS`` and extracts the signal
    subspace (first ``L`` singular vectors). Returns all arrays needed to
    start the main SOM iterative loop.

    Parameters
    ----------
    GS : numpy.ndarray
        Scattering Green's function matrix, shape ``(NM, N)``.
    L : int
        Number of dominant singular values/vectors retained for the signal
        subspace (denoted *L* in [1]_).
    Es : numpy.ndarray
        Measured scattered field matrix, shape ``(NM, NS)``.

    Returns
    -------
    tuple
        ``(J_po, Gs_V_ne, alpha_ne, alpha_neo, rho, grad, del_dat,
        E_s_norm_sq, J_po_norm_sq)``

        - **J_po** *(N, NS)*: Signal-subspace component of the contrast source.
        - **Gs_V_ne** *(N, N-L)*: Noise-subspace right singular vectors.
        - **alpha_ne** *(N-L, NS)*: Noise-subspace coefficients (initialized to zero).
        - **alpha_neo** *(N-L, NS)*: Previous-step noise-subspace coefficients.
        - **rho** *(N-L, NS)*: CG search direction (initialized to zero).
        - **grad** *(N-L, NS)*: Gradient (initialized to zero).
        - **del_dat** *(NM, NS)*: Initial data-equation residual.
        - **E_s_norm_sq** *(NS,)*: Column-wise squared norms of ``Es``.
        - **J_po_norm_sq** *(NS,)*: Column-wise squared norms of ``J_po``.
    """
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
    r"""Compute the three gradient building blocks for the SOM CG step.

    Parameters
    ----------
    GS : numpy.ndarray
        Scattering Green's function matrix, shape ``(NM, N)``.
    Gs_V_ne : numpy.ndarray
        Noise-subspace right singular vectors, shape ``(N, N-L)``.
    del_dat : numpy.ndarray
        Data-equation residual :math:`G_S J - E_s`, shape ``(NM, NS)``.
    E_s_norm_sq : numpy.ndarray
        Column-wise squared norms of the scattered field, shape ``(NS,)``.
    N : int
        Total number of domain discretization points.
    L : int
        Signal-subspace dimension.
    del_sta : numpy.ndarray
        Domain-equation residual, shape ``(N, NS)``.
    J_po_norm_sq : numpy.ndarray
        Column-wise squared norms of ``J_po``, shape ``(NS,)``.

    Returns
    -------
    tuple
        ``(t1, t2a, t2c)`` intermediate gradient arrays used in
        :meth:`SubspaceBasedOptimizationMethod._get_gradient`.
    """
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
    r"""Compute the Polak–Ribière conjugate-gradient search direction.

    Updates the CG search direction :math:`\rho` for the noise-subspace
    coefficients using the Polak–Ribière formula:

    .. math::

        \rho^{(k)} = g^{(k)} + \beta^{(k)}\rho^{(k-1)},\quad
        \beta^{(k)} = \frac{\Re\bigl[(g^{(k)}-g^{(k-1)})^* g^{(k)}\bigr]}
                           {\|g^{(k-1)}\|^2}

    Parameters
    ----------
    grad : numpy.ndarray
        Current gradient, shape ``(N-L, NS)``.
    grado : numpy.ndarray
        Previous-step gradient, shape ``(N-L, NS)``.
    N : int
        Total number of domain points.
    L : int
        Signal-subspace dimension.
    rhoo : numpy.ndarray
        Previous-step search direction, shape ``(N-L, NS)``.

    Returns
    -------
    numpy.ndarray
        Updated search direction, shape ``(N-L, NS)``.
    """
    
    aux = np.real(np.sum(np.conj(grad - grado)*grad,axis=0))/np.sum(np.abs(grado)**2, axis=0)
    aux_tile = 0j*np.ones((N-L, aux.size))
    for n in range(N-L):
        aux_tile[n, :] = aux
    return (grad + aux_tile*rhoo)

@jit(nopython=True)
def compute_alpha(GS, Gs_V_ne, rho, del_dat, E_s_norm_sq, Xe, GDGs_V_nerho,
                  del_sta, J_po_norm_sq, alpha_neo, N, L):
    r"""Update the noise-subspace coefficients via the optimal step size.

    Computes the step size :math:`\mu` that minimizes the SOM cost
    functional along the search direction :math:`\rho` and returns the
    updated coefficients:

    .. math::

        \alpha_{\mathrm{ne}}^{(k)} =
            \alpha_{\mathrm{ne}}^{(k-1)} + \mu\,\rho^{(k)}

    Parameters
    ----------
    GS : numpy.ndarray
        Scattering Green's function matrix, shape ``(NM, N)``.
    Gs_V_ne : numpy.ndarray
        Noise-subspace right singular vectors, shape ``(N, N-L)``.
    rho : numpy.ndarray
        CG search direction, shape ``(N-L, NS)``.
    del_dat : numpy.ndarray
        Data-equation residual, shape ``(NM, NS)``.
    E_s_norm_sq : numpy.ndarray
        Scattered-field squared norms, shape ``(NS,)``.
    Xe : numpy.ndarray
        Tiled contrast vector, shape ``(N, NS)``.
    GDGs_V_nerho : numpy.ndarray
        Domain Green's function applied to ``Gs_V_ne @ rho``, shape ``(N, NS)``.
    del_sta : numpy.ndarray
        Domain-equation residual, shape ``(N, NS)``.
    J_po_norm_sq : numpy.ndarray
        Signal-subspace contrast-source squared norms, shape ``(NS,)``.
    alpha_neo : numpy.ndarray
        Previous-step noise-subspace coefficients, shape ``(N-L, NS)``.
    N : int
        Total number of domain points.
    L : int
        Signal-subspace dimension.

    Returns
    -------
    numpy.ndarray
        Updated noise-subspace coefficients, shape ``(N-L, NS)``.
    """
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
    r"""Assemble the total contrast source from signal and noise components.

    .. math::

        J = J_{\mathrm{po}} + G_S V_{\mathrm{ne}}\,\alpha_{\mathrm{ne}}

    Parameters
    ----------
    J_po : numpy.ndarray
        Signal-subspace contrast source, shape ``(N, NS)``.
    Gs_V_ne : numpy.ndarray
        Noise-subspace right singular vectors, shape ``(N, N-L)``.
    alpha_ne : numpy.ndarray
        Noise-subspace coefficients, shape ``(N-L, NS)``.

    Returns
    -------
    numpy.ndarray
        Total contrast source, shape ``(N, NS)``.
    """
    return J_po + Gs_V_ne @ alpha_ne

@jit(nopython=True)
def compute_X(Et, J, J_po_norm_sq):
    r"""Update the contrast profile from total field and contrast source.

    Computes the least-squares update of the contrast vector in closed
    form:

    .. math::

        \chi_n =
            \frac{\sum_s E_t^*(z_n,s)\,J(z_n,s)/\|J_{\mathrm{po}}\|_s^2}
                 {\sum_s |E_t(z_n,s)|^2/\|J_{\mathrm{po}}\|_s^2}

    Parameters
    ----------
    Et : numpy.ndarray
        Total electric field, shape ``(N, NS)``.
    J : numpy.ndarray
        Total contrast source, shape ``(N, NS)``.
    J_po_norm_sq : numpy.ndarray
        Signal-subspace contrast-source squared norms, shape ``(NS,)``.

    Returns
    -------
    numpy.ndarray
        Updated contrast vector, shape ``(N,)``.
    """
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
    r"""Recompute residuals and auxiliary variables after updating :math:`\alpha_{\mathrm{ne}}`.

    Parameters
    ----------
    Xe : numpy.ndarray
        Tiled contrast vector, shape ``(N, NS)``.
    E_po : numpy.ndarray
        Signal-subspace total field :math:`E_i + G_D J_{\mathrm{po}}`,
        shape ``(N, NS)``.
    J_po : numpy.ndarray
        Signal-subspace contrast source, shape ``(N, NS)``.
    GS : numpy.ndarray
        Scattering Green's function matrix, shape ``(NM, N)``.
    Gs_V_nealpha_ne : numpy.ndarray
        Product :math:`G_S V_{\mathrm{ne}}\alpha_{\mathrm{ne}}`,
        shape ``(N, NS)``.
    Es : numpy.ndarray
        Measured scattered field, shape ``(NM, NS)``.
    GDGs_V_nealpha_ne : numpy.ndarray
        Domain Green's function applied to :math:`G_S V_{\mathrm{ne}}\alpha_{\mathrm{ne}}`,
        shape ``(N, NS)``.

    Returns
    -------
    tuple
        ``(B, del_dat, del_sta)`` updated auxiliary arrays.
    """   
    B = Xe*E_po - J_po
    del_dat = GS@Gs_V_nealpha_ne + GS@J_po - Es
    del_sta = Gs_V_nealpha_ne - Xe*GDGs_V_nealpha_ne - B
    return B, del_dat, del_sta

@jit(nopython=True)
def compute_objective_function(del_dat, E_s_norm_sq, del_sta, J_po_norm_sq):
    r"""Evaluate the SOM normalized cost functional.

    Computes the sum of the normalized data-equation error and the
    normalized domain-equation error:

    .. math::

        \mathcal{F} =
            \sum_s \frac{\|\Delta d_s\|^2}{\|E_s\|^2}
            + \sum_s \frac{\|\Delta \mathrm{sta}_s\|^2}{\|J_{\mathrm{po},s}\|^2}

    Parameters
    ----------
    del_dat : numpy.ndarray
        Data-equation residual, shape ``(NM, NS)``.
    E_s_norm_sq : numpy.ndarray
        Column-wise squared norms of the scattered field, shape ``(NS,)``.
    del_sta : numpy.ndarray
        Domain-equation residual, shape ``(N, NS)``.
    J_po_norm_sq : numpy.ndarray
        Column-wise squared norms of the signal-subspace contrast source,
        shape ``(NS,)``.

    Returns
    -------
    float
        Current value of the objective functional.
    """
    objectf1 = np.sum(np.sum((np.abs(del_dat))**2, axis=0)/E_s_norm_sq)
    objectf2 = np.sum(np.sum((np.abs(del_sta))**2, axis=0)/J_po_norm_sq)
    return objectf1 + objectf2