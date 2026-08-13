r"""Regularization methods for ill-posed linear inverse problems.

This module provides a collection of regularization algorithms used to
stabilize the solution of ill-conditioned linear systems of the form
:math:`Kx = y` that arise in electromagnetic inverse scattering. All
concrete classes inherit from the abstract base class
:class:`Regularization` and expose a uniform ``solve(K, y)`` interface.

Classes
-------
Regularization : ABC
    Abstract base class defining the regularization interface.
Tikhonov : Regularization
    Tikhonov (L2) regularization with fixed, Morozov, or L-curve parameter
    selection.
Landweber : Regularization
    Iterative Landweber regularization.
ConjugatedGradient : Regularization
    Conjugated-gradient (CGLS) regularization.
LeastSquares : Regularization
    Least-squares solution with spectral cut-off (pseudo-inverse).
SingularValueDecomposition : Regularization
    SVD-based regularization combining Tikhonov damping and spectral
    cut-off.

Functions
---------
tikhonov(K, y, alpha)
    Tikhonov regularized solution of :math:`Kx = y`.
mozorov_choice(K, y, delta)
    Morozov discrepancy-principle parameter selection.
lcurve_choice(K, y, bounds, number_terms)
    L-curve parameter selection.
landweber(K, y, x, iterations)
    Landweber iterative solution.
conjugated_gradient(K, y, iterations)
    Conjugated-gradient iterative solution.
least_squares(K, y, cutoff)
    Least-squares solution via :func:`numpy.linalg.lstsq`.
svd(K, y, alpha, min_sv, U, s, V)
    SVD-based regularized solution.

References
----------
.. [1] Kirsch, Andreas. *An Introduction to the Mathematical Theory
   of Inverse Problems*. Vol. 120. Springer, 2011.
"""

import numpy as np
from numba import jit, prange
from numpy import linalg as lag
from abc import ABC, abstractmethod

from eispy2d.core import error

TIK_FIXED = 'fixed'
TIK_MOZOROV = 'mozorov'
TIK_LCURVE = 'lcurve'


class Regularization(ABC):
    """Abstract base class for regularization methods.

    All regularization strategies must inherit from this class and
    implement the :meth:`solve` method, which accepts a coefficient
    matrix and a right-hand-side vector (or matrix) and returns the
    regularized solution.
    """
    def __init__(self):
        """Initialize base regularization object."""
        pass
    @abstractmethod
    def solve(self, K, y):
        """Solve the regularized linear system :math:`Kx = y`.

        Parameters
        ----------
        K : numpy.ndarray
            Coefficient matrix (kernel), shape ``(M, N)``.
        y : numpy.ndarray
            Right-hand side, shape ``(M,)`` or ``(M, P)``.

        Returns
        -------
        numpy.ndarray
            Regularized solution, shape ``(N,)`` or ``(N, P)``.
        """
    @abstractmethod
    def __str__(self):
        return 'Regularization Method: '


class Tikhonov(Regularization):
    r"""Tikhonov (L2) regularization for ill-posed linear systems.

    Solves :math:`Kx = y` by minimizing the Tikhonov functional:

    .. math::

        \|Kx - y\|^2 + \alpha \|x\|^2

    The optimal solution is :math:`x^\alpha = (K^*K + \alpha I)^{-1}K^*y`.
    The regularization parameter :math:`\alpha` can be specified directly
    or determined automatically via the Morozov discrepancy principle or
    the L-curve criterion.

    Parameters
    ----------
    choice : int, float, or {'fixed', 'mozorov', 'lcurve'}
        Regularization parameter selection strategy:

        - ``int`` or ``float``: fixed regularization parameter value.
        - ``'fixed'``: use a fixed parameter value (requires `parameter`).
        - ``'mozorov'``: Morozov discrepancy principle.
        - ``'lcurve'``: L-curve criterion.
    parameter : float, optional
        Fixed regularization parameter value. Required when
        ``choice='fixed'``.

    Attributes
    ----------
    alpha : float or None
        Active regularization parameter. ``None`` when the parameter
        is determined automatically at solve time.
    choice : str
        Selected parameter-choice strategy.

    Raises
    ------
    MissingInputError
        If ``choice='fixed'`` and `parameter` is ``None``.
    WrongTypeInput
        If `parameter` is provided with an unsupported type.
    """
    def __init__(self, choice, parameter=None):
        r"""Initialize Tikhonov regularization.

        Parameters
        ----------
        choice : int, float, or {'fixed', 'mozorov', 'lcurve'}
            Regularization parameter selection strategy:
            - If int or float: fixed regularization parameter value.
            - 'fixed': use a fixed parameter value (must provide `parameter`).
            - 'mozorov': use Morozov's discrepancy principle.
            - 'lcurve': use L-curve criterion.
        parameter : float, optional
            Fixed regularization parameter value (required if choice='fixed').

        Raises
        ------
        MissingInputError
            If choice='fixed' but parameter is None.
        WrongTypeInput
            If parameter has wrong type when choice='fixed'.
        """
        super().__init__()
        if type(choice) is int or type(choice) is float:
            self.alpha = choice
            self.choice = TIK_FIXED
        elif choice == TIK_FIXED:
            if parameter is None:
                raise error.MissingInputError('Tikhonov.__init__', 'parameter')
            elif type(parameter) is float or type(parameter) is int:
                self.alpha = parameter
                self.choice = choice
            else:
                raise error.WrongTypeInput('Tikhonov.__init__', 'parameter',
                                           'float', str(type(parameter)))
        elif choice == TIK_MOZOROV:
            self.choice = choice
            self.alpha = None
        elif choice == TIK_LCURVE:
            self.choice = choice
            self.alpha = None
    def solve(self, K, y):
        r"""Solve the linear system using Tikhonov regularization.

        Parameters
        ----------
        K : :class:`numpy.ndarray`
            Coefficient matrix (kernel) with shape (M, N).
        y : :class:`numpy.ndarray`
            Right-hand side vector or matrix with shape (M,) or (M, P).

        Returns
        -------
        :class:`numpy.ndarray`
            Solution vector or matrix with shape (N,) or (N, P).

        Notes
        -----
        The solution is computed as:
        - For fixed parameter: :math:`x = (K^*K + \alpha I)^{-1}K^*y`
        - For Morozov: automatically selects :math:`\alpha` using discrepancy principle
        - For L-curve: automatically selects :math:`\alpha` using L-curve criterion
        """
        if self.choice == TIK_FIXED:
            if y.ndim == 1:
                return tikhonov(K, y, self.alpha)
            elif y.dim == 2:
                x = np.zeros((K.shape[1], y.shape[1]), dtype=K.dtype)
                for n in range(y.shape[1]):
                    x[:, n] = tikhonov(K, y[:, n].flatten(), self.alpha)
                return x
        elif self.choice == TIK_MOZOROV:
            if y.ndim == 1:
                alpha = mozorov_choice(K, y)
                return tikhonov(K, y, alpha)
            elif y.ndim == 2:
                x = np.zeros((K.shape[1], y.shape[1]), dtype=K.dtype)
                for n in range(y.shape[1]):
                    alpha = mozorov_choice(K, y[:, n].flatten())
                    x[:, n] = tikhonov(K, y[:, n].flatten(), alpha)
                return x
        elif self.choice == TIK_LCURVE:
            if y.ndim == 1:
                alpha = lcurve_choice(K, y)
                return tikhonov(K, y, alpha)
            elif y.ndim == 2:
                x = np.zeros((K.shape[1], y.shape[1]), dtype=K.dtype)
                for n in range(y.shape[1]):
                    alpha = lcurve_choice(K, y[:, n].flatten())
                    x[:, n] = tikhonov(K, y[:, n].flatten(), alpha)
                return x
    def __str__(self):
        message = super().__str__()
        message += 'Tikhonov\n'
        message += 'Choice strategy: ' + self.choice + '\n'
        if self.alpha is not None:
            message += 'Parameter value: %.3e' % self.alpha
        return message


class Landweber(Regularization):
    r"""Iterative Landweber regularization.

    Approximates the solution of :math:`Kx = y` by fixed-point
    iteration:

    .. math::

        x_{n+1} = x_n + a\,K^*(y - K\,x_n), \quad a = \|K\|^{-2}

    The number of iterations acts as an implicit regularization
    parameter: early stopping prevents over-fitting to noise.

    Parameters
    ----------
    iterations : int
        Number of Landweber iterations to perform.

    Attributes
    ----------
    M : int
        Number of iterations.
    """
    def __init__(self, iterations):
        r"""Initialize Landweber regularization.

        Parameters
        ----------
        iterations : int
            Number of iterations for the Landweber method.
        """
        super().__init__()
        self.M = iterations
    def solve(self, K, y):
        r"""Solve the linear system using Landweber iteration.

        Parameters
        ----------
        K : :class:`numpy.ndarray`
            Coefficient matrix (kernel) with shape (M, N).
        y : :class:`numpy.ndarray`
            Right-hand side vector or matrix with shape (M,) or (M, P).

        Returns
        -------
        :class:`numpy.ndarray`
            Solution vector or matrix with shape (N,) or (N, P).

        Notes
        -----
        The Landweber iteration is given by:

        .. math::
            x_{n+1} = x_n + a K^*(y - Kx_n)

        where :math:`a = 1/\|K\|^2`.
        """
        if y.ndim == 1:
            x = np.zeros(K.shape[1], dtype=K.dtype)
            return landweber(K, y, x, self.M)
        elif y.ndim == 2:
            x = np.zeros((K.shape[1], y.shape[1]), dtype=K.dtype)
            for n in range(y.shape[1]):
                t = np.zeros(K.shape[1], dtype=K.dtype)
                x[:, n] = landweber(K, y[:, n].flatten(), t, self.M)
            return x
    def __str__(self):
        message = super().__str__()
        message += 'Landweber\n'
        message += 'Number of iterations: %d' % self.M
        return message


class ConjugatedGradient(Regularization):
    r"""Conjugated-gradient (CGLS) regularization.

    Iteratively solves the normal equations :math:`K^*K\,x = K^*y`
    using the conjugated-gradient method. Early stopping (controlled
    by the number of iterations) provides implicit regularization.

    Parameters
    ----------
    iterations : int
        Maximum number of CG iterations.

    Attributes
    ----------
    M : int
        Number of iterations.
    """
    def __init__(self, iterations):
        r"""Initialize Conjugated Gradient regularization.

        Parameters
        ----------
        iterations : int
            Number of iterations for the CG method.
        """
        super().__init__()
        self.M = iterations
    def solve(self, K, y):
        r"""Solve the linear system using Conjugated Gradient method.

        Parameters
        ----------
        K : :class:`numpy.ndarray`
            Coefficient matrix (kernel) with shape (M, N).
        y : :class:`numpy.ndarray`
            Right-hand side vector or matrix with shape (M,) or (M, P).

        Returns
        -------
        :class:`numpy.ndarray`
            Solution vector or matrix with shape (N,) or (N, P).

        Notes
        -----
        The CG method solves the normal equations:
        :math:`K^*K x = K^*y` using an iterative approach.
        """
        if y.ndim == 1:
            return conjugated_gradient(K, y, self.M)
        elif y.ndim == 2:
            x = np.zeros((K.shape[1], y.shape[1]), dtype=K.dtype)
            for n in range(y.shape[1]):
                x[:, n] = conjugated_gradient(K, y[:, n].flatten(), self.M)
            return x
    def __str__(self):
        message = super().__str__()
        message += 'Conjugated Gradient\n'
        message += 'Number of iterations: %d' % self.M
        return message


class LeastSquares(Regularization):
    r"""Least-squares regularization with spectral cut-off.

    Computes the minimum-norm least-squares solution of :math:`Kx = y`
    via :func:`numpy.linalg.lstsq`. Singular values below the specified
    `cutoff` (rcond) threshold are treated as zero, which acts as a
    spectral truncation regularizer (TSVD).

    Parameters
    ----------
    cutoff : float, optional
        Relative cut-off threshold for singular values (``rcond``
        argument of :func:`numpy.linalg.lstsq`). If ``None``, the
        NumPy default is used.

    Attributes
    ----------
    cutoff : float or None
        Active singular-value cut-off ratio.
    """
    def __init__(self, cutoff=None):
        r"""Initialize Least Squares regularization with spectral cutoff.

        Parameters
        ----------
        cutoff : float, optional
            Cutoff threshold for singular values (rcond parameter).
            If None, uses default from numpy.linalg.lstsq.
        """
        super().__init__()
        self.cutoff = cutoff
    def solve(self, K, y):
        r"""Solve the linear system using least squares with spectral cutoff.

        Parameters
        ----------
        K : :class:`numpy.ndarray`
            Coefficient matrix (kernel) with shape (M, N).
        y : :class:`numpy.ndarray`
            Right-hand side vector or matrix with shape (M,) or (M, P).

        Returns
        -------
        :class:`numpy.ndarray`
            Solution vector or matrix with shape (N,) or (N, P).

        Notes
        -----
        Uses numpy.linalg.lstsq with specified rcond cutoff for
        regularization by truncating small singular values.
        """
        if y.ndim == 1:
            return least_squares(K, y, self.cutoff)
        elif y.ndim == 2:
            x = np.zeros((K.shape[1], y.shape[1]), dtype=K.dtype)
            for n in range(y.shape[1]):
                x[:, n] = least_squares(K, y[:, n].flatten(), self.cutoff)
            return x
    def __str__(self):
        message = super().__str__()
        message += 'Least Squares\n'
        message += 'Cut-Off ratio: %.3e' % self.alpha
        return message


class SingularValueDecomposition(Regularization):
    r"""SVD-based regularization with Tikhonov damping and spectral cut-off.

    Computes the regularized pseudo-inverse of :math:`K` via singular
    value decomposition, combining Tikhonov damping and a spectral
    cut-off filter:

    .. math::

        x = \sum_{n:\,s_n \geq s_{\min}}
            \frac{s_n}{s_n^2 + \alpha}\,(U_n^*\,y)\,V_n

    where :math:`\alpha` is the Tikhonov parameter and :math:`s_{\min}`
    is the cut-off threshold for singular values.

    Parameters
    ----------
    tikhonov : float, default: 0.0
        Tikhonov regularization parameter :math:`\alpha`. Use ``0.0``
        for pure spectral truncation.
    cutoff : float, default: 0.0
        Minimum singular value threshold :math:`s_{\min}`. Singular
        values below this threshold are discarded.

    Attributes
    ----------
    tikhonov : float
        Active Tikhonov parameter.
    cutoff : float
        Active singular-value cut-off.
    """
    def __init__(self, tikhonov=.0, cutoff=.0):
        r"""Initialize SVD-based regularization.

        Parameters
        ----------
        tikhonov : float, default: 0.0
            Tikhonov regularization parameter.
        cutoff : float, default: 0.0
            Cutoff threshold for singular values.
        """
        super().__init__()
        self.tikhonov = tikhonov
        self.cutoff = cutoff
    def solve(self, K=None, y=None, U=None, s=None, V=None):
        r"""Solve the linear system using SVD-based regularization.

        Parameters
        ----------
        K : :class:`numpy.ndarray`, optional
            Coefficient matrix (kernel) with shape (M, N). Required if U, s, V not provided.
        y : :class:`numpy.ndarray`, optional
            Right-hand side vector or matrix with shape (M,) or (M, P).
        U : :class:`numpy.ndarray`, optional
            Left singular vectors matrix from SVD.
        s : :class:`numpy.ndarray`, optional
            Singular values vector from SVD.
        V : :class:`numpy.ndarray`, optional
            Right singular vectors matrix from SVD.

        Returns
        -------
        :class:`numpy.ndarray`
            Solution vector or matrix with shape (N,) or (N, P).

        Notes
        -----
        The solution is computed using singular value decomposition:

        .. math::
            x = \sum_{n} \frac{s_n}{s_n^2 + \alpha} (U_n^* y) V_n

        where :math:`\alpha` is the Tikhonov parameter and singular values
        below `cutoff` are truncated.
        """
        if K is not None and y is not None:
            if y.ndim == 1:
                return svd(K=K, y=y, alpha=self.tikhonov, min_sv=self.cutoff)
            elif y.ndim == 2:
                x = np.zeros((K.shape[1], y.shape[1]), dtype=K.dtype)
                for n in range(y.shape[1]):
                    x[:, n] = svd(K=K, y=y[:, n].flatten(),
                                  alpha=self.tikhonov, min_sv=self.cutoff)
                return x
        elif (y is not None and U is not None and s is not None
                and V is not None):
            if y.ndim == 1:
                return svd(U=U, s=s, V=V, y=y, alpha=self.tikhonov,
                           min_sv=self.cutoff)
            elif y.ndim == 2:
                x = np.zeros((V.shape[1], y.shape[1]), dtype=K.dtype)
                for n in range(y.shape[1]):
                    x[:, n] = svd(U=U, s=s, V=V, y=y[:, n].flatten(),
                                  alpha=self.tikhonov, min_sv=self.cutoff)
                return x
    def __str__(self):
        message = super().__str__()
        message += 'Singular Value Decomposition\n'
        message += 'Tikhonov Regularization Parameter: %.1e\n' % self.tikhonov
        message += 'Singular value cut-off ratio: %.1e\n' % self.cutoff
        return message 


@jit(nopython=True)
def tikhonov(K, y, alpha):
    r"""Perform the Tikhonov regularization.

    Solve the linear ill-posed system through Tikhonov regularization
    [1]_. The solution is given according to:

    .. math:: (K^*K + \alpha I)x = K^*y

    Parameters
    ----------
    K : :class:`numpy.ndarray`
        The coefficient matrix (kernel).
    y : :class:`numpy.ndarray`
        The right-hand-side array.
    alpha : float
        Regularization parameter.

    Returns
    -------
    :class:`numpy.ndarray`
        The regularized solution vector.

    References
    ----------
    .. [1] Kirsch, Andreas. An introduction to the mathematical theory
       of inverse problems. Vol. 120. Springer Science & Business Media,
       2011.
    """
    x = lag.solve(K.conj().T@K + alpha*np.eye(K.shape[1]), K.conj().T@y)
    return x


@jit(nopython=True, parallel=True)
def mozorov_choice(K, y, delta=1e-3):
    r"""Apply the Discrepancy Principle of Morozov [1].

    Compute the regularization parameter according to the starting guess
    of Newton's method for solving the Discrepancy Principle of Morozov
    defined in [1].

    Parameters
    ----------
    K : :class:`numpy.ndarray`
        Coefficient matrix (kernel).
    y : :class:`numpy.ndarray`
        Right-hand-side array.
    delta : float, default: 1e-3
        Noise level of the problem.

    Returns
    -------
    float
        Optimal regularization parameter :math:`\alpha`.

    Notes
    -----
    The Discrepancy Principle of Morozov is defined according to
    the zero of the following monotone function:

    .. math:: \phi(\alpha) = \|Kx^{\alpha,\delta}-y^{\delta}\|^2 - \delta^2

    The initial guess of Newton's method to determine the zero is:

    .. math:: \alpha = \frac{\delta\|K\|^2}{\|y^\delta\| - \delta}

    References
    ----------
    .. [1] Kirsch, Andreas. An introduction to the mathematical theory
           of inverse problems. Vol. 120. Springer Science & Business
           Media, 2011.
    """
    # Auxiliar variables
    KsK = np.conj(K.T)@K
    Ksy = np.conj(K.T)@y
    eye = np.eye(K.shape[1])

    # Initial guess of frequency interval
    x0 = np.log10(delta*lag.norm(K)**2/(lag.norm(y)-delta))
    xmax = x0+5
    xmin = x0-5

    # Error of the initial guess
    fa = (lag.norm(y - K@lag.solve(KsK + 10**xmin*eye, Ksy))-delta**2)**2
    fb = (lag.norm(y - K@lag.solve(KsK + 10**xmax*eye, Ksy))-delta**2)**2

    # Find interval
    evals = 2
    while fb < fa:
        xmin = xmax
        fa = fb
        xmax = 2*xmax
        fb = (lag.norm(y - K@lag.solve(KsK + 10**xmax*eye, Ksy))-delta**2)**2
        evals += 1
    if evals <= 3:
        xmin = np.log10(delta*lag.norm(K)**2/(lag.norm(y)-delta))-5
    else:
        xmin = xmin/2

    # Solve the frequency
    xa = xmax - .618*(xmax-xmin)
    xb = xmin + .618*(xmax-xmin)
    fa = (lag.norm(y - K@lag.solve(KsK + 10**xa*eye, Ksy))-delta**2)**2
    fb = (lag.norm(y - K@lag.solve(KsK + 10**xb*eye, Ksy))-delta**2)**2

    while (xmax-xmin) > 1e-3:
        if fa > fb:
            xmin = xa
            xa = xb
            xb = xmin + 0.618*(xmax-xmin)
            fa = fb
            fb = (lag.norm(y - K@lag.solve(KsK + 10**xb*eye, Ksy))-delta**2)**2

        else:
            xmax = xb
            xb = xa
            xa = xmax - 0.618*(xmax-xmin)
            fb = fa
            fa = (lag.norm(y - K@lag.solve(KsK + 10**xa*eye, Ksy))-delta**2)**2

    return 10**((xmin+xmax)/2)


@jit(nopython=True, parallel=True)
def lcurve_choice(K, y, bounds=(-20, 0), number_terms=21):
    r"""Determine the regularization parameter through L-curve.

    The regularization parameter is determined according to the L-curve.
    The L-curve is the graph between error and solution norms. The
    values are normalized and the chosen point is the one in which its
    distance from (0, 0) is minimum.

    Parameters
    ----------
    K : :class:`numpy.ndarray`
        Coefficient matrix (kernel).
    y : :class:`numpy.ndarray`
        Right-hand-side array.
    bounds : 2-tuple, default: (-20, 0)
        Minimum and maximum value of the exponential form of the
        regularization parameter (log10 scale).
    number_terms : int, default: 21
        Number of samples on the L-curve.

    Returns
    -------
    float
        Optimal regularization parameter :math:`\alpha`.
    """
    # Auxiliar variables
    KsK = np.conj(K.T)@K
    Ksy = np.conj(K.T)@y
    eye = np.eye(K.shape[1])

    f1, f2 = np.zeros(number_terms), np.zeros(number_terms)
    alpha = 10**np.linspace(bounds[0], bounds[1], number_terms)

    # Compute objective-functions
    for i in prange(number_terms):
        x = lag.solve(KsK + alpha[i]*eye, Ksy)
        f1[i] = lag.norm(y-K@x)
        f2[i] = lag.norm(x)

    # Normalization
    f1 = (f1-np.amin(f1))/(np.amax(f1)-np.amin(f1))
    f2 = (f2-np.amin(f2))/(np.amax(f2)-np.amin(f2))

    # Best solution (Closest solution to the utopic one)
    knee = np.argmin(np.sqrt(f1**2 + f2**2))
    return alpha[knee]


@jit(nopython=True, parallel=True)
def landweber(K, y, x, iterations):
    r"""Perform the Landweber regularization.

    Solve the linear ill-posed system through Landweber regularization
    [1]_. The algorithm formula is:

    .. math:: x_{n+1} = x_n + a K^*(y - K x_n)

    Parameters
    ----------
    K : :class:`numpy.ndarray`
        The coefficient matrix (kernel).
    y : :class:`numpy.ndarray`
        The right-hand-side array.
    x : :class:`numpy.ndarray`
        Initial guess for the solution.
    iterations : int
        Number of iterations.

    Returns
    -------
    :class:`numpy.ndarray`
        The regularized solution vector.

    References
    ----------
    .. [1] Kirsch, Andreas. An introduction to the mathematical theory
       of inverse problems. Vol. 120. Springer Science & Business Media,
       2011.
    """
    a = 1/lag.norm(K)**2
    for m in range(iterations):
        x = x + a*K.T.conj()@(y-K@x)
    return x


@jit(nopython=True, parallel=True)
def conjugated_gradient(K, y, iterations):
    r"""Perform the Conjugated-Gradient (CG) regularization.

    Solve the linear ill-posed system through CG regularization [1]_.

    Parameters
    ----------
    K : :class:`numpy.ndarray`
        The coefficient matrix (kernel).
    y : :class:`numpy.ndarray`
        The right-hand-side array.
    iterations : int
        Number of iterations.

    Returns
    -------
    :class:`numpy.ndarray`
        The regularized solution vector.

    References
    ----------
    .. [1] Kirsch, Andreas. An introduction to the mathematical theory
       of inverse problems. Vol. 120. Springer Science & Business Media,
       2011.
    """
    p = -K.conj().T@y
    x = 0j*np.zeros(K.shape[1])
    for m in range(iterations):
        Kp = K@p
        res = K@x-y
        tm = np.sum(res * np.conj(Kp))/np.sum(np.abs(Kp)**2)
        # tm = np.inner(res, np.conj(Kp))/lag.norm(Kp)**2
        x = x - tm*p
        Kres = K.conj().T@(K@x-y)
        gamma = np.sum(np.abs(Kres)**2)/np.sum(np.abs(K.conj().T@res))
        # gamma = (lag.norm(Kres)**2/lag.norm(K.conj().T@res)**2)
        p = Kres + gamma*p
    return x


@jit(nopython=True)
def least_squares(K, y, cutoff):
    r"""Return the Spectral Cut-off solution to a linear matrix equation.

    See explanation at `<https://numpy.org/doc/stable/reference
    /generated/numpy.linalg.lstsq.html>`_

    Parameters
    ----------
    K : :class:`numpy.ndarray`
        The coefficient matrix (kernel).
    y : :class:`numpy.ndarray`
        The right-hand-side array.
    cutoff : float
        Truncation level (rcond) for singular values.

    Returns
    -------
    :class:`numpy.ndarray`
        The least squares solution vector.
    """
    return lag.lstsq(K, y, rcond=cutoff)[0]

@jit(nopython=True)
def svd(K=None, y=None, alpha=None, min_sv=None, U=None, s=None, V=None):
    r"""Solve linear system using SVD with Tikhonov regularization and spectral cutoff.

    Parameters
    ----------
    K : :class:`numpy.ndarray`, optional
        Coefficient matrix (kernel). If provided, computes SVD.
    y : :class:`numpy.ndarray`, optional
        Right-hand side vector.
    alpha : float, optional
        Tikhonov regularization parameter.
    min_sv : float, optional
        Minimum singular value threshold (spectral cutoff).
    U : :class:`numpy.ndarray`, optional
        Left singular vectors (if precomputed).
    s : :class:`numpy.ndarray`, optional
        Singular values (if precomputed).
    V : :class:`numpy.ndarray`, optional
        Right singular vectors (if precomputed).

    Returns
    -------
    :class:`numpy.ndarray` or tuple
        If K and y provided: solution vector.
        If only K provided: (U, s, V) SVD components.
        If U, s, V, y provided: solution vector.

    Notes
    -----
    The solution is computed as:

    .. math::
        x = \sum_{n} \frac{s_n}{s_n^2 + \alpha} (U_n^* y) V_n

    where singular values below `min_sv` are truncated.
    """
    if K is not None and y is None:
        U, s, Vh = lag.svd(K)
        V = np.transpose(np.conj(Vh))
        return U, s, V
    elif K is not None and y is not None:
        if alpha is None:
            alpha = 0.
        if min_sv is None:
            min_sv = 1e-50
        U, s, Vh = lag.svd(K)
        V = np.transpose(np.conj(Vh))
        x = s[0]/(s[0]**2 + alpha)*np.sum(y*np.conj(U[0, :]))*V[0, :]
        for n in range(1, s.size):
            if s[n] < min_sv:
                break
            x += s[n]/(s[n]**2 + alpha)*np.sum(y*np.conj(U[n, :]))*V[n, :]
        return x
    elif (K is None and y is not None and U is not None and s is not None
            and V is not None):
        if alpha is None:
            alpha = 0.
        if min_sv is None:
            min_sv = 1e-50
        x = s[0]/(s[0]**2 + alpha)*np.sum(y*np.conj(U[0, :]))*V[0, :]
        for n in range(1, s.size):
            if s[n] < min_sv:
                break
            x += s[n]/(s[n]**2 + alpha)*np.sum(y*np.conj(U[n, :]))*V[n, :]
        return x
