import numpy as np
from numba import jit, prange
from numpy import linalg as lag
from abc import ABC, abstractmethod

import error

TIK_FIXED = 'fixed'
TIK_MOZOROV = 'mozorov'
TIK_LCURVE = 'lcurve'


class Regularization(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def solve(self, K, y):
        pass
    @abstractmethod
    def __str__(self):
        return 'Regularization Method: '


class Tikhonov(Regularization):
    def __init__(self, choice, parameter=None):
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
    def __init__(self, iterations):
        super().__init__()
        self.M = iterations
    def solve(self, K, y):
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
    def __init__(self, iterations):
        super().__init__()
        self.M = iterations
    def solve(self, K, y):
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
    def __init__(self, cutoff=None):
        super().__init__()
        self.cutoff = cutoff
    def solve(self, K, y):
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
    def __init__(self, tikhonov=.0, cutoff=.0):
        super().__init__()
        self.tikhonov = tikhonov
        self.cutoff = cutoff
    def solve(self, K=None, y=None, U=None, s=None, V=None):
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
        A : :class:`numpy.ndarray`
            The coefficient matrix.

        beta : :class:`numpy.ndarray`
            The right-hand-side array.

        alpha : float
            Regularization parameter.

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
            Coefficient matrix.

        y : :class:`numpy.ndarray`
            Right-hand-side array.

        delta : float
            Noise level of problem.

    Notes
    -----
        The Discrepancy Principle of Morozov is defined according to
        the zero of the following monotone function:

        .. math:: \phi(\alpha) = ||Kx^{\alpha,\delta}-y^{\delta}||^2-\delta^2

        The initial guess of Newton's method to determine the zero is:

        .. math:: \alpha = \frac{\delta||K||^2}{||y^\delta-\delta}

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
    """Determine the regularization parameter through L-curve.

    The regularization parameter is determined according to the L-curve.
    The L-curve is the graph between error and solution norms. The
    values are normalized and the chosen point is the one in which its
    distance from (0, 0) is minimum.

    Parameters
    ----------
        A : 2-d :class:`numpy.ndarray`
            Coefficient matrix.

        b : 1-d :class:`numpy.ndarray`
            Right-hand-side.

        bounds : 2-tuple
            Minimum and maximum value of the exponential form of the
            regularization parameter.

        number_terms : int
            Number of samples at the L-curve.
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

    .. math:: x_{n+1} = x_n + aK^{*}(y-Kx_n)

    Parameters
    ----------
        K : :class:`numpy.ndarray`
            The coefficient matrix.

        y : :class:`numpy.ndarray`
            The right-hand-side array.

        iterations : int
            Number of iterations.

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
            The coefficient matrix.

        y : :class:`numpy.ndarray`
            The right-hand-side array.
        
        iterations : int
            Number of iterations

        x0 : :class:`numpy.ndarray`
            Initial guess of solution.

        delta : float
            Error tolerance level.

        print_info : bool
            Print iteration information.

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
    """Return the Spectral Cut-off solution to a linear matrix equation.

    See explanation at `<https://numpy.org/doc/stable/reference
    /generated/numpy.linalg.lstsq.html>`_

    Parameters
    ----------
        K : :class:`numpy.ndarray`
            The coefficient matrix.

        y : :class:`numpy.ndarray`
            The right-hand-side array.

        alpha : float
            Truncation level of singular values.
    """
    return lag.lstsq(K, y, rcond=cutoff)[0]

@jit(nopython=True)
def svd(K=None, y=None, alpha=None, min_sv=None, U=None, s=None, V=None):
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
