import warnings
import numpy as np
from numpy import std
from matplotlib import pyplot as plt
from scipy.stats import shapiro, boxcox, fligner, friedmanchisquare, ttest_ind
from scipy.stats import mannwhitneyu, wilcoxon, ttest_1samp, kruskal
from scipy.stats import f as Fdist
from scipy.stats import t as Tdist
from scipy.optimize import curve_fit
from statsmodels import stats
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.oneway import anova_oneway as anova
from statsmodels.stats.power import tt_solve_power as power_tt
from statsmodels.stats.power import tt_ind_solve_power as power_ttind
from statsmodels.stats.power import TTestPower
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.api import graphics
import pingouin as pg

import error
import result as rst


def rcbd(data, alpha=0.05):
    r"""Randomized Complete Block Design"""
    if type(data) is list:
        y = np.array(data)
    else:
        y = np.copy(data)
    a, b = y.shape
    N = a*b
    yid = np.sum(y, axis=1)
    ydj = np.sum(y, axis=0)
    ydd = np.sum(y)

    SST = np.sum(y**2) - ydd**2/N
    SSTreat = 1/b*np.sum(yid**2)-ydd**2/N
    SSBlocks = 1/a*np.sum(ydj**2)-ydd**2/N
    SSE = SST-SSTreat-SSBlocks

    DOFnum = a-1
    DOFden = (a-1)*(b-1)

    MSTreat = SSTreat/(a-1)
    MSE = SSE/((a-1)*(b-1))

    F0 = MSTreat/MSE

    alpha = 0.05

    Fcrit = Fdist.ppf(1-alpha, DOFnum, DOFden)

    pvalue = Fdist.sf(F0, DOFnum, DOFden)

    return F0, pvalue, F0 < Fcrit


def residuals(data, blocked=False):
    if type(data) is np.ndarray:
        y = data.tolist()
    else:
        y = data

    a = len(data)
    b = np.zeros(a, dtype=int)
    for i in range(a):
        b[i] = len(y[i])
    N = np.sum(b)

    if blocked is False:
        means = np.zeros(a)
        for i in range(a):
            means[i] = np.mean(y[i])
        res = np.zeros(N)
        n = 0
        for i in range(a):
            for j in range(b[i]):
                res[n] = y[i][j] - means[i]

    else:
        yid = np.zeros(a)
        ydj = np.zeros(np.amax(b))
        ydd = 0
        for i in range(a):
            yid[i] = np.sum(y[i])
            for j in range(b[i]):
                ydj[j] += y[i][j]
                ydd += y[i][j]

        yidb = yid/b
        ydjb = ydj/a
        yddb = ydd/N

        res = np.zeros(N)
        n = 0
        for i in range(a):
            for j in range(b[i]):
                res[n] = y[i][j] - yidb[i] - ydjb[j] + yddb
                n += 1

    return res


def ttest_paired(y1, y2, alternative='two-sided', alpha=0.05):
    if y1.size != y2.size:
        raise error.WrongTypeInput('ttest_paired', 'y1 and y2',
                                   'y1.size == y2.size', 'y1.size == %d'
                                   % y1.size + ' and y2.size == %d' % y2.size)
    n = y1.size
    d = y1-y2
    dh = np.sum(d)/n
    Sd = np.sqrt(np.sum((d-dh)**2)/(n-1))
    t0 = dh/(Sd/np.sqrt(n))
    if alternative == 'two-sided':
        ta = Tdist.ppf(alpha/2, n-1)
        tb = Tdist.ppf(1-alpha/2, n-1)
        H0 = ta < t0 and t0 < tb
        pvalue = 2*Tdist.cdf(-np.abs(t0), n-1)
        confint = (dh-np.abs(ta)*Sd/np.sqrt(n), dh+tb*Sd/np.sqrt(n))
    elif alternative == 'less':
        tc = Tdist.ppf(alpha, n-1)
        H0 = tc < t0
        pvalue = Tdist.cdf(t0, n-1)
        confint = ("-inf", dh+tc*Sd/np.sqrt(n))
    elif alternative == 'greater':
        tc = Tdist.ppf(1-alpha, n-1)
        H0 = t0 < tc
        pvalue = Tdist.sf(t0, n-1)
        confint = (dh+np.abs(tc)*Sd/np.sqrt(n), "inf")
    return t0, H0, pvalue, confint


def factorial_analysis(data, alpha=0.05, group_names=None, ylabel=None):
    r"""Perform factorial analysis.

    Given a data set with some amount of factors and levels, the method
    performs the factorial analysis in order to find evidences for
    impact of single factors (main effects) and combination among them
    (interaction effects) [1]_.

    In this current version, it only supports two or three factors and
    balanced data.

    Parameters
    ----------
        data : :class:`numpy.ndarray`
            The data set in array format in which each dimension
            represents a factor and the number of elements represents
            the number of levels of respective factor. The shape must be
            either (a, b, n), for two-factors, or (a, b, c, n), for
            three factors.  *Obs.*: the last dimension is the number of
            samples for each combination of factors-levels.

        alpha : float, default: 0.05
            Significance level.

        group_names : list, default: None
            Factor names for plot purposes.

        ylabel : str
            Y-axis label for plot purposes (meaning of the data).

    Returns
    -------
        null_hypothesis : list
            The list with the results of the null hypothesis of the
            statistic tests. If `True`, means that the test failed to
            reject the null hypothesis; if `False`, means the null
            hypothesis was rejected. For two-factor anaylsis, the each
            element represents the test on the following factors
            `[A, B, AB]`. For three-factor,
            `[A, B, C, AB, AC, BC, ABC]`.

        pvalues : list
            The list with the p-values of each test. The order follows
            the same defined for `null_hypothesis`.

        shapiro_pvalue: float
            The p-value of the Shapiro-Wilk's test for normality of
            residuals assumption. A p-value less than 0.05 means the
            rejection of the assumption.

        fligner_pvalue: float
            The p-value of the Fligner-Killen's test for homoscedascity
            (variance equality) of samples. A p-value less than 0.05
            means the rejection of the assumption.

        fig : :class:`matplotlib.figure.Figure`
            A plot showing the normality and homoscedascity assumption.
            The graphic way to anaylise the assumptions.

        transformation : None or str
            If `None`, no transformation was applied on the data in
            order to fix it for following the assumption. Otherwise,
            it is a string saying the type of transformation.

    References
    ----------
    .. [1] Montgomery, Douglas C. Design and analysis of experiments.
       John wiley & sons, 2017.
    """
    NF = data.ndim-1

    # Two-Factor Analysis
    if NF == 2:

        # Number of levels and samples
        a, b, n = data.shape

        # Computing residuals and separing samples
        res = np.zeros((a, b, n))
        samples = []
        for i in range(a):
            for j in range(b):
                res[i, j, :] = data[i, j, :] - np.mean(data[i, j, :])
                samples.append(data[i, j, :])

        # Check normality assumption
        if shapiro(res.flatten())[1] < .05:

            # For Box-Cox transformation, it is required positive data.
            if np.amin(res) <= 0 and np.amin(res) < np.amin(data):
                delta = -np.amin(res) + 1
            elif np.amin(data) <= 0 and np.amin(data) <= np.amin(res):
                delta = -np.amin(data) + 1
            else:
                delta = 0

            # In case of non-normality, the Box-Cox transformation
            # is performed.
            _, lmbda = boxcox(res.flatten() + delta)
            y = boxcox(data.flatten() + delta, lmbda=lmbda)
            y = y.reshape((a, b, n))
            transformation = 'boxcox, lambda=%.3e' % lmbda
            res = np.zeros((a, b, n))
            samples = []
            for i in range(a):
                for j in range(b):
                    res[i, j, :] = y[i, j, :] - np.mean(y[i, j, :])
                    samples.append(y[i, j, :])
        else:
            y = np.copy(data)
            transformation = None

        # Save results of assumptions.
        _, shapiro_pvalue = shapiro(res.flatten())
        _, fligner_pvalue = fligner(*samples)

        # Plot normality and homoscedascity
        fig, axes, lgd_size = rst.get_figure(2, len(group_names))
        normalitiyplot(res.flatten(), axes=axes[0])
        homoscedasticityplot(y.reshape((-1, n)), axes=axes[1],
                             title='Homoscedascity', ylabel=ylabel,
                             names=group_names, legend_fontsize=lgd_size)

        # Means
        yhi = np.sum(y, axis=(1, 2))/(b*n)
        yhj = np.sum(y, axis=(0, 2))/(a*n)
        yhij = np.sum(y, axis=2)/n
        yh = np.sum(y)/(a*b*n)

        # Square sums
        SSA = b*n*np.sum((yhi-yh)**2)
        SSB = a*n*np.sum((yhj-yh)**2)
        SSAB = 0
        SSE = 0
        for i in range(a):
            for j in range(b):
                SSAB += n*(yhij[i, j] - yhi[i] - yhj[j] + yh)**2
                SSE += np.sum((y[i, j, :]-yhij[i, j])**2)

        # Degrees of freedom
        dfA, dfB, dfAB, dfE = a-1, b-1, (a-1)*(b-1), a*b*(n-1)

        # Means of square sums
        MSA = SSA/(a-1)
        MSB = SSB/(b-1)
        MSAB = SSAB/(a-1)/(b-1)
        MSE = SSE/(a*b)/(n-1)

        # Statistics
        F0A, F0B, F0AB = MSA/MSE, MSB/MSE, MSAB/MSE

        # Critical values
        FCA = Fdist.ppf(1-alpha, dfA, dfE)
        FCB = Fdist.ppf(1-alpha, dfB, dfE)
        FCAB = Fdist.ppf(1-alpha, dfAB, dfE)

        # Hypothesis tests
        null_hypothesis = [F0A < FCA, F0B < FCB, F0AB < FCAB]

        # P-value computation
        pvalue_a = 1-Fdist.cdf(F0A, dfA, dfE)
        pvalue_b = 1-Fdist.cdf(F0B, dfB, dfE)
        pvalue_ab = 1-Fdist.cdf(F0AB, dfAB, dfE)

        return (null_hypothesis, [pvalue_a, pvalue_b, pvalue_ab],
                shapiro_pvalue, fligner_pvalue, fig, transformation)

    # Three-factor analysis
    elif NF == 3:

        # Number of levels and samples
        a, b, c, n = data.shape

        # Computing residuals and separing samples
        res = np.zeros((a, b, c, n))
        samples = []
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    res[i, j, k, :] = (data[i, j, k, :]
                                       - np.mean(data[i, j, k, :]))
                    samples.append(data[i, j, k, :])

        # Check normality assumption
        if shapiro(res.flatten())[1] < .05:

            # For Box-Cox transformation, it is required positive data.
            if np.amin(res) <= 0 and np.amin(res) < np.amin(data):
                delta = -np.amin(res) + 1
            elif np.amin(data) <= 0 and np.amin(data) <= np.amin(res):
                delta = -np.amin(data) + 1
            else:
                delta = 0

            # In case of non-normality, the Box-Cox transformation
            # is performed.
            _, lmbda = boxcox(res.flatten() + delta)
            y = boxcox(data.flatten() + delta, lmbda=lmbda)
            y = y.reshape((a, b, c, n))
            transformation = 'boxcox, lambda=%.3e' % lmbda
            res = np.zeros((a, b, c, n))
            samples = []
            for i in range(a):
                for j in range(b):
                    for k in range(c):
                        res[i, j, k, :] = (y[i, j, k, :]
                                           - np.mean(y[i, j, k, :]))
                        samples.append(y[i, j, k, :])

        else:
            y = np.copy(data)
            transformation = None

        # Save results of assumptions.
        _, shapiro_pvalue = shapiro(res.flatten())
        _, fligner_pvalue = fligner(*samples)

        # Plot normality and homoscedascity
        fig, axes, lgd_size = rst.get_figure(2, len(group_names))
        normalitiyplot(res.flatten(), axes=axes[0])
        homoscedasticityplot(y.reshape((-1, n)), axes=axes[1],
                             title='Homocedascity', ylabel=ylabel,
                             names=group_names, legend_fontsize=lgd_size)

        # Sums
        ydddd = np.sum(y)
        yiddd = np.sum(y, axis=(1, 2, 3))
        ydjdd = np.sum(y, axis=(0, 2, 3))
        yddkd = np.sum(y, axis=(0, 1, 3))
        yijdd = np.sum(y, axis=(2, 3))
        yidkd = np.sum(y, axis=(1, 3))
        ydjkd = np.sum(y, axis=(0, 3))
        yijkd = np.sum(y, axis=3)

        # Square sums
        SST = np.sum(y**2) - ydddd**2/(a*b*c*n)
        SSA = 1/(b*c*n)*np.sum(yiddd**2) - ydddd**2/(a*b*c*n)
        SSB = 1/(a*c*n)*np.sum(ydjdd**2) - ydddd**2/(a*b*c*n)
        SSC = 1/(a*b*n)*np.sum(yddkd**2) - ydddd**2/(a*b*c*n)
        SSAB = 1/(c*n)*np.sum(yijdd**2) - ydddd**2/(a*b*c*n)-SSA-SSB
        SSAC = 1/(b*n)*np.sum(yidkd**2) - ydddd**2/(a*b*c*n)-SSA-SSC
        SSBC = 1/(a*n)*np.sum(ydjkd**2) - ydddd**2/(a*b*c*n)-SSB-SSC
        SSABC = (1/n*np.sum(yijkd**2)-ydddd**2/(a*b*c*n)-SSA-SSB-SSC-SSAB-SSAC
                 - SSBC)
        SSE = SST-SSABC-SSA-SSB-SSC-SSAB-SSAC-SSBC

        # Means of square sums
        MSA = SSA/(a-1)
        MSB = SSB/(b-1)
        MSC = SSC/(c-1)
        MSAB = SSAB/(a-1)/(b-1)
        MSAC = SSAC/(a-1)/(c-1)
        MSBC = SSBC/(b-1)/(c-1)
        MSABC = SSABC/(a-1)/(b-1)/(c-1)
        MSE = SSE/(a*b*c)/(n-1)

        # Statistics
        F0A, F0B, F0C = MSA/MSE, MSB/MSE, MSC/MSE
        F0AB, F0AC, F0BC, F0ABC = MSAB/MSE, MSAC/MSE, MSBC/MSE, MSABC/MSE

        # Degrees of freedom
        dfA, dfB, dfC = a-1, b-1, c-1
        dfAB, dfAC, dfBC = dfA*dfB, dfA*dfC, dfB*dfC
        dfABC, dfE = dfA*dfB*dfC, a*b*c*(n-1)

        # Critical values
        FCA = Fdist.ppf(1-alpha, dfA, dfE)
        FCB = Fdist.ppf(1-alpha, dfB, dfE)
        FCC = Fdist.ppf(1-alpha, dfC, dfE)
        FCAB = Fdist.ppf(1-alpha, dfAB, dfE)
        FCAC = Fdist.ppf(1-alpha, dfAC, dfE)
        FCBC = Fdist.ppf(1-alpha, dfBC, dfE)
        FCABC = Fdist.ppf(1-alpha, dfABC, dfE)

        # Hypothesis tests
        null_hypothesis = [F0A < FCA, F0B < FCB, F0C < FCC, F0AB < FCAB,
                           F0AC < FCAC, F0BC < FCBC, F0ABC < FCABC]

        # P-value computation
        pvalue_a = 1-Fdist.cdf(F0A, dfA, dfE)
        pvalue_b = 1-Fdist.cdf(F0B, dfB, dfE)
        pvalue_c = 1-Fdist.cdf(F0C, dfC, dfE)
        pvalue_ab = 1-Fdist.cdf(F0AB, dfAB, dfE)
        pvalue_ac = 1-Fdist.cdf(F0AC, dfAC, dfE)
        pvalue_bc = 1-Fdist.cdf(F0BC, dfBC, dfE)
        pvalue_abc = 1-Fdist.cdf(F0ABC, dfABC, dfE)

        return (null_hypothesis, [pvalue_a, pvalue_b, pvalue_c, pvalue_ab,
                                  pvalue_ac, pvalue_bc, pvalue_abc],
                shapiro_pvalue, fligner_pvalue, fig, transformation)

    # Future implementations will address more factors.
    else:
        return None


def ttest_ind_nonequalvar(y1, y2, alternative='two-sided', alpha=0.05):
    r"""Perform T-Test on independent samples with non-equal variances.

    Statistic test which compares two independent sample without
    assuming variance equality [1]_. The *two-sided* test is performed.

    Parameters
    ----------
        y1, y2 : :class:`numpy.ndarray`
            1-d arrays representing the samples.

        alpha : float, default: 0.05
            Significance level.

    Returns
    -------
        null_hypothesis : bool
            Result of the null hypothesis test. If `True`, it means that
            the test has failed to reject the null hypothesis. If
            `False`, it means that the null hypothesis has been rejected
            at 1-`alpha` confidence level.

        t0 : float
            T statistic.

        pvalue: float

        nu : float
            Degrees of freedom.

        confint : tuple of 2-float
            Confidence interval (lower and upper bounds) of the true
            mean difference.

    References
    ----------
    .. [1] Montgomery, Douglas C. Design and analysis of experiments.
       John wiley & sons, 2017.
    """
    # Samples sizes
    n1, n2 = y1.size, y2.size

    # Estimated means
    y1h, y2h = np.mean(y1), np.mean(y2)

    # Estimated variances
    S12, S22 = np.sum((y1-y1h)**2)/(n1-1), np.sum((y2-y2h)**2)/(n2-1)

    # T-statistics
    t0 = (y1h-y2h)/np.sqrt(S12/n1 + S22/n2)

    # Degrees of freedom
    nu = (S12/n1 + S22/n2)**2/((S12/n1)**2/(n1-1) + (S22/n2)**2/(n2-1))

    # Critical values
    ta, tb = Tdist.ppf(alpha/2, nu), Tdist.ppf(1-alpha/2, nu)

    # Hypothesis test
    null_hypothesis = ta < t0 and tb > t0

    # Confidence level
    confint = (y1h-y2h-ta*np.sqrt(S12/n1 + S22/n2),
               y1h-y2h+tb*np.sqrt(S12/n1 + S22/n2))

    # P-value computation
    pvalue = 2*Tdist.cdf(-np.abs(t0), nu)

    return null_hypothesis, t0, pvalue, nu, confint


def dunnetttest(y0, y):
    r"""Perform all-to-one comparisons through Dunnett's test.

    The Dunnett's test is a procedure for comparing a set of :math:`a-1`
    treatments against a single one called the control group [1]_. The
    test is a modification of the usual t-test where, in each
    comparison, the null hypothesis is the equality of means. The
    significance level is fixed in 0.05.

    Parameters
    ----------
        y0 : :class:`numpy.ndarray`
            Control sample (1-d array).

        y : list or :class:`numpy.ndarray`
            :math:`a-1` treatments to be compared. The argument must be
            either a list of Numpy arrays or a matrix with shape
            (a-1, n).

    Returns
    -------
        null_hypothesis : list of bool
            List of boolean values indicating the result of the null
            hypothesis. If `True`, it means that the test has failed in
            rejecting the null hypothesis. If `False`, then the null
            hypothesis of equality of means for the respective
            comparison has been reject at a 0.05 significance level.

    References
    ----------
    .. [1] Montgomery, Douglas C. Design and analysis of experiments.
       John wiley & sons, 2017.
    """
    # Avoiding insignificant messages for the analysis
    warnings.filterwarnings('ignore', message='Covariance of the parameters '
                            + 'could not be estimated')

    # Columns of the statistic table (a-1 predefined values)
    Am1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Rows of the statistic table (predefined degrees of freedom)
    F = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                  16, 17, 18, 19, 20, 24, 30, 40, 60, 120, 1e20])

    # Critical values for Dunnett's Test for 0.05 significance level
    D = np.array([[2.57, 3.03, 3.29, 3.48, 3.62, 3.73, 3.82, 3.90, 3.97],
                  [2.45, 2.86, 3.10, 3.26, 3.39, 3.49, 3.57, 3.64, 3.71],
                  [2.36, 2.75, 2.97, 3.12, 3.24, 3.33, 3.41, 3.47, 3.53],
                  [2.31, 2.67, 2.88, 3.02, 3.13, 3.22, 3.29, 3.35, 3.41],
                  [2.26, 2.61, 2.81, 2.95, 3.05, 3.14, 3.20, 3.26, 3.32],
                  [2.23, 2.57, 2.76, 2.89, 2.99, 3.07, 3.14, 3.19, 3.24],
                  [2.20, 2.53, 2.72, 2.84, 2.94, 3.02, 3.08, 3.14, 3.19],
                  [2.18, 2.50, 2.68, 2.81, 2.90, 2.98, 3.04, 3.09, 3.14],
                  [2.16, 2.48, 2.65, 2.78, 2.87, 2.94, 3.00, 3.06, 3.10],
                  [2.14, 2.46, 2.63, 2.75, 2.84, 2.91, 2.97, 3.02, 3.07],
                  [2.13, 2.44, 2.61, 2.73, 2.82, 2.89, 2.95, 3.00, 3.04],
                  [2.12, 2.42, 2.59, 2.71, 2.80, 2.87, 2.92, 2.97, 3.02],
                  [2.11, 2.41, 2.58, 2.69, 2.78, 2.85, 2.90, 2.95, 3.00],
                  [2.10, 2.40, 2.56, 2.68, 2.76, 2.83, 2.89, 2.94, 2.98],
                  [2.09, 2.39, 2.55, 2.66, 2.75, 2.81, 2.87, 2.92, 2.96],
                  [2.09, 2.38, 2.54, 2.65, 2.73, 2.80, 2.86, 2.90, 2.95],
                  [2.06, 2.35, 2.51, 2.61, 2.70, 2.76, 2.81, 2.86, 2.90],
                  [2.04, 2.32, 2.47, 2.58, 2.66, 2.72, 2.77, 2.82, 2.86],
                  [2.02, 2.29, 2.44, 2.54, 2.62, 2.68, 2.73, 2.77, 2.81],
                  [2.00, 2.27, 2.41, 2.51, 2.58, 2.64, 2.69, 2.73, 2.77],
                  [1.98, 2.24, 2.38, 2.47, 2.55, 2.60, 2.65, 2.69, 2.73],
                  [1.96, 2.21, 2.35, 2.44, 2.51, 2.57, 2.61, 2.65, 2.69]])

    # Compute the sum of square for both input types
    if type(y) is list:
        a = 1 + len(y)
        N = y0.size
        n = []
        for i in range(len(y)):
            N += y[i].size
            n.append(y[i].size)
        SSE = np.sum((y0-np.mean(y0))**2)
        yh = np.zeros(len(y))
        for i in range(len(y)):
            yh[i] = np.mean(y[i])
            SSE += np.sum((y[i]-yh[i])**2)
    else:
        a = 1 + y.shape[0]
        N = y0.size + y.size
        SSE = np.sum((y0-np.mean(y0))**2)
        yh = np.zeros(y.shape[0])
        n = y.shape[1]*np.ones(y.shape[0])
        for i in range(y.shape[0]):
            yh[i] = np.mean(y[i, :])
            SSE += np.sum((y[i, :]-yh[i])**2)

    # Mean square error and degrees of freedom
    MSE = SSE/(N-a)
    f = N-a

    # If the number of comparisons is equal to one of the columns of the
    # table of critical values, then we check if the number of degrees
    # of freedom is also available. If isn't, we approximate a value
    # by curve fitting procedure with the closest number of degrees of
    # freedom.
    if a-1 < 10:
        if np.any(F-f == 0):
            j = np.argwhere(F-f == 0)[0][0]
            d = D[j, a-2]
        else:
            popt, _ = curve_fit(fittedcurve, F[:], D[:, a-1],
                                p0=[4.132, -1.204, 1.971],
                                absolute_sigma=False, maxfev=20000)
            d = fittedcurve(f, popt[0], popt[1], popt[2])

    # If the number of comparisons is greater than the available, then
    # we approximate a value through curve fitting.
    else:
        for i in range(F.size):
            if F-f >= 0:
                break
        popt, _ = curve_fit(fittedcurve, Am1, D[i, :],
                            absolute_sigma=False, maxfev=20000)
        d = fittedcurve(a-1, popt[0], popt[1], popt[2])

    null_hypothesis = []
    y0h = np.mean(y0)
    na = y0.size

    # Hypothesis test
    for i in range(a-1):
        if np.abs(yh[i]-y0h) > d*np.sqrt(MSE*(1/n[i]+1/na)):
            null_hypothesis.append(False)
        else:
            null_hypothesis.append(True)

    return null_hypothesis


def fittedcurve(x, a, b, c):
    """Evalute standard curve for linear regression in Dunnett's test.

    This routine computes the function :math:`ax^b+c` which is used for
    curve fitting in Dunnett's test.
    """
    return a*x**b+c


def data_transformation(data, residuals_check=False, blocked=False):
    """Try data transformation for normal distribution assumptions.

    Currently, it only implements the Log and Square-Root
    transformations. The normality assumption may be tested on the data
    or in the residuals.

    Parameters
    ----------
        data : either :class:`numpy.ndarray` or list
            If `residuals` is `False`, then the argument must be an 1-d
            array with the sample to be tested. Otherwise, it must be
            a list of arrays.

        residuals : bool
            If `True`, the transformation will be tried over the
            residuals of the observations. Otherwise, the transformation
            will be tried over the own sample.

    Returns
    -------
        If the transformation succeeds, then it returns the transformed
        data and a string containing the type of transformation.
        Otherwise, it returns `None`.
    """
    # Try transformation over the data
    if not residuals_check:

        # Log Transformation
        if all(d != 0 for d in data) and shapiro(np.log(data))[1] > .05:
            return np.log(data), 'log'

        # Square-root transformation
        elif all(d > 0 for d in data) and shapiro(np.sqrt(data))[1] > .05:
            return np.sqrt(data), 'sqrt'

        # If both transformations fail
        else:
            return None

    # Try transformation over the residuals
    else:

        res = residuals(np.log(data), blocked=blocked)

        # Try Log Transformation
        if shapiro(res)[1] > .05:
            for m in range(len(data)):
                data[m] = np.log(data[m])
            return data, 'log'

        # Compute Square-Root Transformation
        res = residuals(np.sqrt(data), blocked=blocked)

        # Try Square-Root Transformation
        if shapiro(res)[1] > .05:
            for m in range(len(data)):
                data[m] = np.sqrt(data[m])
            return data, 'sqrt'

        # If both transformations fail
        else:
            return None


def normalitiyplot(data, axes=None, title=None, fontsize=10):
    """Graphic investigation of normality assumption.

    This routine plots a figure comparing a sample to a standard normal
    distribution for the purpose of investigating the assumption of
    normality. This routine does not show any plot. It only draws the
    graphic.

    Parameters
    ----------
        data : :class:`numpy.ndarray`
            An 1-d array representing the sample.

        axes : :class:`matplotlib.Axes.axes`, default: None
            A specified axes for plotting the graphics. If none is
            provided, then one will be created and returned.

        title : str, default: None
            A possible title to the plot.

    Returns
    -------
        fig : :class:`matplotlib.figure.Figure`

    Example
    -------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> y1 = np.random.normal(size=30)
    >>> y2 = np.random.normal(size=60)
    >>> fig = plt.figure()
    >>> axes1 = fig.add_subplot(1, 2, 1)
    >>> normalityplot(y1, axes=axes1, title='Sample 1')
    >>> axes2 = fig.add_subplot(1, 2, 1)
    >>> normalityplot(y2, axes=axes2, title='Sample 2')
    >>> plt.show()
    """
    # If no axes is provided, a figure is created.
    if axes is None:
        fig, axes = plt.subplots()
    else:
        fig = plt.gcf()

    # QQ Plot
    pg.qqplot(data, dist='norm', ax=axes)

    if title is not None:
        axes.set_title(title, fontsize=fontsize)    
    axes.xaxis.label.set_size(fontsize)
    axes.yaxis.label.set_size(fontsize)
    axes.tick_params(labelsize=fontsize)
    axes.grid(True)

    return fig, axes


def homoscedasticityplot(data, axes=None, title=None, ylabel=None, names=None,
                         legend_fontsize=None):
    """Graphic investigation of homoscedasticity assumption.

    This routine plots a figure comparing variance of samples for the
    purpose of investigating the assumption of homoscedasticity
    (samples with equal variance). Each samples is positioned in the
    x-axis in the correspondent value of its own mean. This routine does
    not show any plot. It only draws the graphic.

    Parameters
    ----------
        data : either :class:`numpy.ndarray` or list
            A 2-d array with the samples in which each row is a single
            sample or a list of 1-d arrays.

        axes : :class:`matplotlib.Axes.axes`, default: None
            A specified axes for plotting the graphics. If none is
            provided, then one will be created and returned.

        title : str, default: None
            A possible title to the plot.

        ylabel : str, default: None
            The label of the y-axis which represent the unit of the
            data.

        names : list of str, default: None
            A list with the name of the samples for legend purpose.

    Returns
    -------
        fig : :class:`matplotlib.figure.Figure`

    Example
    -------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> y1 = np.random.normal(loc=2., size=30)
    >>> y2 = np.random.normal(loc=4., size=60)
    >>> homoscedasticityplot([y1, y2], title='Samples',
                             names=['Sample 1', 'Sample 2'])
    >>> plt.show()
    """
    if axes is None:
        fig, axes = plt.subplots()
    else:
        fig = None

    if type(data) is list:
        for i in range(len(data)):
            if names is None:
                axes.plot(np.mean(data[i])*np.ones(data[i].size),
                          data[i]-np.mean(data[i]), 'o')
            else:
                axes.plot(np.mean(data[i])*np.ones(data[i].size),
                          data[i]-np.mean(data[i]), 'o', label=names[i])

    else:
        for i in range(data.shape[0]):
            if names is None:
                axes.plot(np.mean(data[i, :])*np.ones(data.shape[1]),
                          data[i, :]-np.mean(data[i, :]), 'o')
            else:
                axes.plot(np.mean(data[i, :])*np.ones(data.shape[1]),
                          data[i, :]-np.mean(data[i, :]), 'o', label=names[i])

    axes.grid()
    axes.set_xlabel('Means')
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if title is not None:
        axes.set_title(title)
    if names is not None:
        if legend_fontsize is not None:
            axes.legend(fontsize=legend_fontsize)
        else:
            axes.legend()

    if title is not None:
        axes.set_title(title)

    return axes


def confint(data, alpha=.05, alternative="two-sided"):
    if shapiro(data)[1] > alpha:
        normality_hypothesis, transformation, x = True, None, data
    else:
        out = data_transformation(data)
        if out is None:
            normality_hypothesis, transformation, x = False, None, data
        else:
            normality_hypothesis, transformation, x = True, out[1], out[0]
    cfi = DescrStatsW(x).tconfint_mean(alpha=alpha, alternative=alternative)
    return cfi, normality_hypothesis, transformation


def confintplot(data, axes=None, xlabel=None, ylabel=None, title=None,
                fontsize=10, confidence_level=0.95, xscale=None):
    """Plot the confidence interval of means.

    This routine plots a figure comparing the confidence interval of
    means among samples. The confidence intervals are computed at a
    0.95 confidence level. This routine does not show any plot. It only
    draws the graphic.

    Parameters
    ----------
        data : either :class:`numpy.ndarray` or list
            A 2-d array with the samples in which each *column* is a
            single sample or a list of 1-d arrays.

        axes : :class:`matplotlib.Axes.axes`, default: None
            A specified axes for plotting the graphics. If none is
            provided, then one will be created and returned.

        title : str, default: None
            A possible title to the plot.

        xlabel : str, default: None
            The label of the x-axis which represent the unit of the
            data.

        ylabel : list of str, default: None
            A list with the name of the samples.

    Returns
    -------
        fig : :class:`matplotlib.figure.Figure`

    Example
    -------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> y1 = np.random.normal(loc=2., size=30)
    >>> y2 = np.random.normal(loc=4., size=60)
    >>> y3 = np.random.normal(loc=6., size=10)
    >>> confintplot([y1, y2, y3], title='Samples',
                    ylabel=['Sample 1', 'Sample 2', 'Sample 3'])
    >>> plt.show()
    """
    if type(data) is np.ndarray:
        y = []
        for i in range(data.shape[1]):
            info = DescrStatsW(data[:, i])
            cf = info.tconfint_mean(alpha=1-confidence_level)
            y.append((cf[0], info.mean, cf[1]))
    elif type(data) is list:
        y = []
        for i in range(len(data)):
            info = DescrStatsW(data[i])
            cf = info.tconfint_mean(alpha=1-confidence_level)
            y.append((cf[0], info.mean, cf[1]))
    else:
        raise error.WrongTypeInput('confintplot', 'data', 'list or ndarray',
                                   str(type(data)))

    if axes is None:
        fig, axes = plt.subplots()
    else:
        fig = None

    for i in range(len(y)):
        axes.plot(y[i][::2], [i, i], 'k')
        axes.plot(y[i][0], i, '|k', markersize=20)
        axes.plot(y[i][2], i, '|k', markersize=20)
        axes.plot(y[i][1], i, 'ok')

    axes.tick_params(labelsize=fontsize)

    plt.grid()
    if xlabel is not None:
        axes.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        axes.set_yticks(range(len(y)))
        axes.set_yticklabels(ylabel, fontsize=fontsize)
        axes.set_ylim(ymin=-1, ymax=len(y))
    if title is not None:
        axes.set_title(title, fontsize=fontsize)
    if xscale is not None:
        axes.set_xscale(xscale)

    return fig, axes


def violinplot(data, axes=None, labels=None, xlabel=None, ylabel=None,
               color='royalblue', yscale=None, title=None, show=False,
               file_name=None, file_path='', file_format='eps'):
    """Improved violinplot routine.

    *Obs*: if no axes is provided, then a figure will be created and
    showed or saved.

    Parameters
    ----------
        data : list of :class:`numpy.ndarray`
            A list of 1-d arrays meaning the samples.

        axes : :class:`matplotlib.Axes.axes`, default: None
            A specified axes for plotting the graphics. If none is
            provided, then one will be created and showed or saved.

        labels : list of str, default: None
            Names of the samples.

        xlabel : str, default: None

        ylabel : list of str, default: None

        color : str, default: 'b'
            Color of boxes. Check some `here <https://matplotlib.org/
            3.1.1/gallery/color/named_colors.html>`_

        yscale : None or {'linear', 'log', 'symlog', 'logit', ...}
            Scale of y-axis. Check some options `here <https://
            matplotlib.org/3.1.1/api/_as_gen/
            matplotlib.pyplot.yscale.html>`

        title : str, default: None
            A possible title to the plot.

        show : bool
            If `True`, then the figure is shown. Otherwise, the figure
            is saved.

        file_name : str
            File name when saving the figure.

        file_path : str
            Path to the saved figure.

        file_format : {'eps', 'png', 'pdf', 'svg'}
            Format of the saved figure.

    Example
    -------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> y1 = np.random.normal(loc=2., size=30)
    >>> y2 = np.random.normal(loc=4., size=60)
    >>> y3 = np.random.normal(loc=6., size=10)
    >>> violinplot([y1, y2, y3], title='Samples',
                   labels=['Sample 1', 'Sample 2', 'Sample 3'],
                   xlabel='Samples', ylabel='Unit', color='tab:blue',
                   show=True)
    """
    plot_opts = {'violin_fc': color,
                 'violin_ec': 'w',
                 'violin_alpha': .2}

    if axes is not None:
        if yscale is not None:
            axes.set_yscale(yscale)

        graphics.violinplot(data,
                               ax=axes,
                               labels=labels,
                               plot_opts=plot_opts)

        if xlabel is not None:
            axes.set_xlabel(xlabel)
        if ylabel is not None:
            axes.set_ylabel(ylabel)
        if title is not None:
            axes.set_title(title)
        axes.grid()

    else:
        if yscale is not None:
            plt.yscale(yscale)

        graphics.violinplot(data, labels=labels, plot_opts=plot_opts)

        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if title is not None:
            plt.title(title)
        plt.grid()

        if show:
            plt.show()
        else:
            if file_name is not None:
                raise error.MissingInputError('boxplot', 'file_name')

            plt.savefig(file_path + file_name + '.' + file_format,
                        format=file_format)
            plt.close()


def compare1sample(x, offset=0.):
    if shapiro(x) < .05:
        out = data_transformation(x)
        if out is None:
            nonnormal = True
            transf = None
        else:
            x, transf = out
            nonnormal = False
    if not nonnormal:
        statistic, pvalue = ttest_1samp(x, offset)
    else:
        statistic, pvalue = wilcoxon(x, offset*np.ones(len(x)))
    if pvalue > .05:
        alternative = 'two-sided'
    else:
        if np.mean(x) > offset:
            alternative = 'greater'
        else:
            alternative = 'less'
    if not nonnormal:
        statistic, pvalue = ttest_1samp(x, offset, alternative=alternative)
    else:
        statistic, pvalue = wilcoxon(x, offset*np.ones(len(x)),
                                     alternative=alternative)
    if not nonnormal:
        delta = (TTestPower().solve_power(nobs=len(x), alpha=.05, power=.8,
                                          alternative=alternative)/std(x))
    else:
        delta = None
    return statistic, pvalue, alternative, nonnormal, transf, delta


def compare2samples(x1, x2, paired=False):
    if not paired:
        if shapiro(x1)[1] < .05:
            out = data_transformation(x1)
            if out is None:
                nonnormal = True
                transf = None
            else:
                x1, transf = out
                if transf == 'log':
                    x2 = np.log(x2)
                elif transf == 'sqrt':
                    x2 = np.sqrt(x2)
                if shapiro(x2) < .05:
                    nonnormal = True
                else:
                    nonnormal = False
        elif shapiro(x2)[1] < .05:
            out = data_transformation(x2)
            if out is None:
                nonnormal = True
                transf = None
            else:
                x2, transf = out
                if transf == 'log':
                    x1 = np.log(x1)
                elif transf == 'sqrt':
                    x1 = np.sqrt(x1)
                if shapiro(x1)[1] < .05:
                    nonnormal = True
                else:
                    nonnormal = False
        else:
            nonnormal = False
            transf = None
        if not nonnormal:
            if fligner(x1, x2)[1] > .05:
                equal_var = True
            else:
                equal_var = False
            alternative = 'two-sided'
            statistic, pvalue = ttest_ind(x1, x2, equal_var=equal_var)
            if pvalue < 0.05:
                if np.mean(x1) < np.mean(x2):
                    alternative = 'less'
                    statistic, pvalue = ttest_ind(x1, x2, equal_var=equal_var,
                                                  alternative='less')
                else:
                    alternative = 'greater'
                    statistic, pvalue = ttest_ind(x1, x2, equal_var=equal_var,
                                                  alternative='greater')
            if equal_var:
                try:
                    delta = power_ttind(nobs1=len(x1), alpha=0.05, power=0.8,
                                        ratio=len(x2)/len(x1),
                                        alternative=alternative)*np.std(x1)
                except:
                    delta = None
            else:
                delta = None
                    
        else:
            alternative = 'two-sided'
            statistic, pvalue = mannwhitneyu(x1, x2, alternative=alternative)
            if pvalue < .05:
                alternative = 'less'
                statistic, pvalue = mannwhitneyu(x1, x2,
                                                 alternative=alternative)
                if pvalue > .05:
                    alternative = 'greater'
                    statistic, pvalue = mannwhitneyu(x1, x2,
                                                     alternative=alternative)
            delta, equal_var = None, None
    else:
        if (type(x1) is np.ndarray and type(x2) is np.ndarray
                and x1.size != x2.size):
            raise error.Error('For paired comparison, both samples must have '
                              + 'the same size')
        elif type(x1) is list and type(x2) is list and len(x1) != len(x2):
            raise error.Error('For paired comparison, both samples must have '
                              + 'the same size')
        if shapiro(x1-x2)[1] < .05:
            out = data_transformation(x1-x2)
            if out is None:
                nonnormal = True
                xd = x1-x2
                transf = None
            else:
                xd, transf = out
                nonnormal = False
        else:
            xd = x1-x2
            nonnormal = False
            transf = None
        if not nonnormal:
            alternative = 'two-sided'
            statistic, pvalue = ttest_1samp(xd, 0, alternative=alternative)
            if pvalue < .05:
                alternative = 'less'
                statistic, pvalue = ttest_1samp(xd, 0, alternative='less')
                if pvalue > .05:
                    alternative = 'greater'
                    statistic, pvalue = ttest_1samp(xd, 0,
                                                    alternative='greater')
            try:
                delta = power_tt(nobs=len(xd), alpha=0.05, power=0.8,
                                 alternative=alternative)*np.std(xd)
            except Exception:
                delta = None
        else:
            alternative = 'two-sided'
            statistic, pvalue = wilcoxon(xd, alternative='two-sided')
            if pvalue < .05:
                alternative = 'less'
                statistic, pvalue = wilcoxon(xd, alternative='less')
                if pvalue > .05:
                    alternative = 'greater'
                    statistic, pvalue = wilcoxon(xd, alternative='greater')
            delta = None
        equal_var = None
    return statistic, pvalue, alternative, delta, nonnormal, transf, equal_var


def compare_multiple(data, all2all=False, all2one=None, paired=False):
    if type(data) is np.ndarray:
        samples = [data[n, :] for n in range(data.shape[0])]
    elif type(data) is not list and all(type(d) is np.ndarray for d in data):
        raise error.WrongTypeInput('statistics.compare_multiple', 'data',
                                   'list or numpy.ndarray', str(type(data)))
    else:
        samples = data.copy()
    NS = len(samples)
    res = residuals(samples, blocked=paired)
    if shapiro(res)[1] < .05:
        out = data_transformation(samples, residuals_check=True,
                                  blocked=paired)
        if out is None:
            nonnormal, transf = True, None
        else:
            nonnormal = False
            samples, transf = out
    else:
        nonnormal = False
        transf = None
    if not nonnormal:
        if fligner(*samples)[1] > .05:
            homocedascity = True
            if paired:
                statistic, pvalue, _ = rcbd(samples, alpha=0.05)
            else:
                output = anova(samples, use_var='equal')
                statistic, pvalue = output.statistic, output.pvalue
        else:
            if paired:
                statistic, pvalue, _ = rcbd(samples, alpha=0.05)
            else:
                output = anova(samples, use_var='unequal')
                statistic, pvalue = output.statistic, output.pvalue
            homocedascity = False
    else:
        if paired:
            statistic, pvalue = friedmanchisquare(*samples)
        else:
            statistic, pvalue = kruskal(*samples)
        homocedascity = None
    if all2all:
        all2all_out = []
        if not nonnormal:
            if paired:
                alpha = 0.05/(NS*(NS-1)/2)
                for i in range(NS-1):
                    for j in range(i+1, NS):
                        output = ttest_paired(samples[i], samples[j],
                                              alpha=alpha)
                        all2all_out.append((output[1], output[2], output[3])) # H0, p-value, confint
            elif homocedascity:
                aux, groups = [], []
                for n in range(NS):
                    for i in range(len(samples[n])):
                        aux.append(samples[n][i])
                        groups.append('%d' %n)
                aux, groups = np.array(aux), np.array(groups)
                output = MultiComparison(aux, groups).tukeyhsd(alpha=.05)
                
                for n in range(len(output.reject)):
                    all2all_out.append((not output.reject[n],
                                        output.pvalues[n],
                                        output.confint[n])) # H0, p-value, confint
            else:
                alpha = 0.05/(NS*(NS-1)/2)
                for i in range(NS-1):
                    for j in range(i+1, NS):
                        output = ttest_ind_nonequalvar(samples[i], samples[j],
                                                       alpha=alpha)
                        all2all_out.append((output[0], output[2], output[4])) # H0, p-value, confint
                
        else:
            if paired:
                for i in range(NS-1):
                    for j in range(i+1, NS):
                        all2all_out.append(wilcoxon(samples[i], samples[j])[1]) # p-value
            else:
                for i in range(NS-1):
                    for j in range(i+1, NS):
                        all2all_out.append(mannwhitneyu(samples[i],
                                                        samples[j])[1]) # p-value
    else:
        all2all_out = None
    if all2one is not None:
        if type(all2one) is not int:
            raise error.WrongTypeInput('statistics.compare_multiple',
                                       'one2all', 'int', str(type(all2one)))
        elif all2one < 0 or all2one >= NS:
            raise error.WrongValueInput('statistics.compare_multiple',
                                        'all2one', '0 <= all2one < %d' % NS,
                                        str(all2one))
        if not nonnormal:
            if paired:
                alpha = 0.05/(NS-1)
                all2one_out = []
                for n in range(NS):
                    if n != all2one:
                        output = ttest_paired(samples[all2one], samples[n],
                                              alpha=alpha)
                        all2one_out.append(output[1], output[2], output[3]) # H0, p-value, confint
            elif homocedascity:
                y0 = samples[all2one]
                y = []
                for n in range(NS):
                    if n != all2one:
                        y.append(samples[n])
                all2one_out = dunnetttest(y0, y) # H0
            else:
                alpha = 0.05/(NS-1)
                all2one_out = []
                for n in range(NS):
                    if n != all2one:
                        output = ttest_ind_nonequalvar(samples[all2one],
                                                       samples[n], alpha)
                        all2one_out.append(output[0], output[2], output[4]) # H0, p-value, confint
        else:
            all2one_out = []
            if paired:
                for n in range(NS):
                    if n != all2one:
                        all2one_out.append(wilcoxon(samples[all2one],
                                                    samples[n])[1]) # p-value
            else:
                for n in range(NS):
                    if n != all2one:
                        all2one_out.append(mannwhitneyu(samples[all2one],
                                                        samples[n])[1]) # p-value
    else:
        all2one_out = None
    return (statistic, pvalue, nonnormal, transf, homocedascity, all2all_out,
            all2one_out)
      
