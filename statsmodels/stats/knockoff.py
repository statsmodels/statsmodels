"""
The Knockoff method is an approach for controlling the False Discovery
Rate (FDR).  It controls FDR for parameter estimates that are
potentially strongly dependent, such as coefficient estimates in a
multiple regression model.  The knockoff approach does not require
standard errors, so one application is to provide inference for LASSO
and other regularized or step-wise regression fits.

The approach is applicable whenever the test statistic can be computed
entirely from x'y and x'x, where x is the design matrix and y is the
vector of responses.

Reference
---------
Rina Foygel Barber, Emmanuel Candes (2015).  Controlling the False
Discovery Rate via Knockoffs.  Annals of Statistics 43:5.
http://statweb.stanford.edu/~candes/papers/FDR_regression.pdf
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.compat.numpy import np_new_unique


class KnockoffTester(object):
    """
    Stub for KnockoffTester class.

    Any implementation of the class must take data and calculate test
    statistics that compare the evidence for the effect of a given
    variable relative to its corresponding knockoff variable.  Higher
    values for these statistics imply greater evidence that the effect
    is real.
    """

    def __init__(self):
        raise NotImplementedError

    def stats(self):
        raise NotImplementedError


class KnockoffCorrelation(KnockoffTester):
    """
    Marginal correlation selection for knockoff analysis.

    Parameters
    ----------
    endog : array-like
        The dependent variable of the regression
    exog : array-like
        The independent variables of the regression

    Notes
    -----
    This class implements the marginal correlation approach to
    constructing test statistics for a knockoff analysis, as
    desscribed under (1) in section 2.2 of the Barber and Candes
    paper.
    """

    def __init__(self, endog, exog):
        exog1, exog2, sl = _design_knockoff_equi(exog)
        self.exog1 = exog1
        self.exog2 = exog2
        self.endog = endog - np.mean(endog)

    def stats(self):
        s1 = np.dot(self.exog1.T, self.endog)
        s2 = np.dot(self.exog2.T, self.endog)
        return np.abs(s1) - np.abs(s2)


class KnockoffForward(KnockoffTester):
    """
    Forward selection for knockoff analysis.

    Parameters
    ----------
    endog : array-like
        The dependent variable of the regression
    exog : array-like
        The independent variables of the regression
    pursuit : bool
        If True, 'basis pursuit' is used, which amounts to performing
        a full regression at each selection step to adjust the working
        residual vector.  If False (the default), the residual is
        adjusted by regressing out each selected variable marginally.
        Setting pursuit=True will be considerably slower, but may give
        better results when exog is not orthogonal.

    Notes
    -----
    This class implements the forward selection approach to
    constructing test statistics for a knockoff analysis, as
    desscribed under (5) in section 2.2 of the Barber and Candes
    paper.
    """

    def __init__(self, endog, exog, pursuit=False):

        exog1, exog2, sl = _design_knockoff_equi(exog)
        self.exog = np.concatenate((exog1, exog2), axis=1)
        self.endog = endog - np.mean(endog)
        self.pursuit = pursuit

    def stats(self):
        nvar = self.exog.shape[1]
        rv = self.endog.copy()
        vl = [(i, self.exog[:, i]) for i in range(nvar)]
        z = np.empty(nvar)
        past = []
        for i in range(nvar):
            dp = np.r_[[np.abs(np.dot(rv, x[1])) for x in vl]]
            j = np.argmax(dp)
            z[vl[j][0]] = nvar - i - 1
            x = vl[j][1]
            del vl[j]
            if self.pursuit:
                for v in past:
                    x -= np.dot(x, v)*v
                past.append(x)
            rv -= np.dot(rv, x) * x
        z1 = z[0:nvar//2]
        z2 = z[nvar//2:]
        st = np.where(z1 > z2, z1, z2) * np.sign(z1 - z2)
        return st


class KnockoffOLS(KnockoffTester):
    """
    OLS regression for knockoff analysis.

    Parameters
    ----------
    endog : array-like
        The dependent variable of the regression
    exog : array-like
        The independent variables of the regression

    Notes
    -----
    This class implements the ordinary least squares regression
    approach to constructing test statistics for a knockoff analysis,
    as desscribed under (2) in section 2.2 of the Barber and Candes
    paper.
    """

    def __init__(self, endog, exog):

        nobs, nvar = exog.shape
        exog1, exog2, _ = _design_knockoff_equi(exog)
        exog = np.concatenate((exog1, exog2), axis=1)
        endog = endog - np.mean(endog)

        model = sm.OLS(endog, exog)
        result = model.fit()
        self._stats = (np.abs(result.params[0:nvar]) -
                       np.abs(result.params[nvar:]))

    def stats(self):
        return self._stats


class Knockoff(object):
    """
    Control FDR for various estimation procedures.

    The knockoff method of Barber and Candes is an approach for
    controlling the FDR of a variety of estimation procedures,
    including correlation coefficients, OLS regression, and LASSO
    regression.

    Parameters
    ----------
    tester : KnockoffTester instance
        An instance of a KnockoffTester class that can compute test
        statistics satisfying the conditions of the knockoff
        procedure.
    design_method: string
        The method used to construct the knockoff design matrix.

    Returns an instance of the Knockoff class.  The `fdr` attribute
    holds the estimated false discovery rates.
    """

    def __init__(self, tester, design_method='equi'):
        self.stats = tester.stats()

        unq, inv, cnt = np_new_unique(self.stats, return_inverse=True,
                                      return_counts=True)

        # The denominator of the FDR
        cc = np.cumsum(cnt)
        denom = len(self.stats) - cc + cnt
        denom[denom < 1] = 1

        # The numerator of the FDR
        ii = np.searchsorted(unq, -unq, side='right') - 1
        numer = cc[ii]
        numer[ii < 0] = 0

        # The knockoff+ estimated FDR
        fdr = (1 + numer) / denom

        self.fdr = fdr[inv]
        self._ufdr = fdr
        self._unq = unq

    def threshold(self, tfdr):
        """
        Returns the threshold statistic for a given target FDR.
        """

        if np.min(self._ufdr) <= tfdr:
            return self._unq[self._ufdr <= tfdr][0]
        else:
            return np.inf


def _design_knockoff_sdp(exog):
    """
    Use semidefinite programming to construct a knockoff design
    matrix.

    Requires cvxopt to be installed.
    """

    try:
        from cvxopt import solvers, matrix
    except ImportError:
        raise ValueError("SDP knockoff designs require installation of cvxopt")

    nobs, nvar = exog.shape

    # Standardize exog
    xnm = np.sum(exog**2, 0)
    xnm = np.sqrt(xnm)
    exog /= xnm

    Sigma = np.dot(exog.T, exog)

    c = matrix(-np.ones(nvar))

    h0 = np.concatenate((np.zeros(nvar), np.ones(nvar)))
    h0 = matrix(h0)
    G0 = np.concatenate((-np.eye(nvar), np.eye(nvar)), axis=0)
    G0 = matrix(G0)

    h1 = (2 * Sigma)
    h1 = matrix(h1)
    i, j = np.diag_indices(nvar)
    G1 = np.zeros((nvar*nvar, nvar))
    G1[i*nvar + j, i] = 1
    G1 = matrix(G1)

    solvers.options['show_progress'] = False
    sol = solvers.sdp(c, G0, h0, [G1], [h1])
    sl = np.asarray(sol['x']).ravel()

    xcov = np.dot(exog.T, exog)
    exogn = _get_knmat(exog, xcov, sl)

    return exog, exogn, sl


def _design_knockoff_equi(exog):
    """
    Construct an equivariant design matrix for knockoff analysis.

    Follows the 'equi-correlated knockoff approach of equation 2.4 in
    Barber and Candes.

    Constructs a pair of design matrices exogs, exogn such that exogs
    is a scaled/centered version of the input matrix exog, exogn is
    another matrix of the same shape with cov(exogn) = cov(exogs), and
    the covariances between corresponding columns of exogn and exogs
    are as small as possible.
    """

    nobs, nvar = exog.shape

    if nobs < 2*nvar:
        msg = "The equivariant knockoff can ony be used when n >= 2*p"
        raise ValueError(msg)

    # Standardize exog
    xnm = np.sum(exog**2, 0)
    xnm = np.sqrt(xnm)
    exog /= xnm

    xcov = np.dot(exog.T, exog)
    ev, _ = np.linalg.eig(xcov)
    evmin = np.min(ev)

    sl = min(2*evmin, 1)
    sl = sl * np.ones(nvar)

    exogn = _get_knmat(exog, xcov, sl)

    return exog, exogn, sl


def _get_knmat(exog, xcov, sl):
    # Utility function, see equation 2.2 of Barber & Candes.

    nobs, nvar = exog.shape

    ash = np.linalg.inv(xcov)
    ash *= -np.outer(sl, sl)
    i, j = np.diag_indices(nvar)
    ash[i, j] += 2 * sl

    umat = np.random.normal(size=(nobs, nvar))
    u, _ = np.linalg.qr(exog)
    umat -= np.dot(u, np.dot(u.T, umat))
    umat, _ = np.linalg.qr(umat)

    ashr, xc, _ = np.linalg.svd(ash, 0)
    ashr *= np.sqrt(xc)
    ashr = ashr.T

    ex = (sl[:, None] * np.linalg.solve(xcov, exog.T)).T
    exogn = exog - ex + np.dot(umat, ashr)

    return exogn
