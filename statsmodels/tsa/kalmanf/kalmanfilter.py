"""
State Space Analysis using the Kalman Filter

References
-----------
Durbin., J and Koopman, S.J.  `Time Series Analysis by State Space Methods`.
    Oxford, 2001.

Hamilton, J.D.  `Time Series Analysis`.  Princeton, 1994.

Harvey, A.C. `Forecasting, Structural Time Series Models and the Kalman Filter`.
    Cambridge, 1989.

Notes
-----
This file follows Hamilton's notation pretty closely.
The ARMA Model class follows Durbin and Koopman notation.
Harvey uses Durbin and Koopman notation.
"""
# Anderson and Moore `Optimal Filtering` provides a more efficient algorithm
# namely the information filter
# if the number of series is much greater than the number of states
# e.g., with a DSGE model.  See also
# http://www.federalreserve.gov/pubs/oss/oss4/aimindex.html
# Harvey notes that the square root filter will keep P_t pos. def. but
# is not strictly needed outside of the engineering (long series)
from __future__ import print_function
from statsmodels.compat.python import lzip, lmap, callable, range
import numpy as np
from numpy import dot, identity, kron, log, zeros, pi, eye, issubdtype, ones
from numpy.linalg import inv, pinv
from scipy import optimize
from statsmodels.tools.tools import chain_dot
from . import kalman_loglike

# Fast filtering and smoothing for multivariate state space models
# and The Riksbank -- Strid and Walentin (2008)
# Block Kalman filtering for large-scale DSGE models
# but this is obviously macro model specific


def _init_diffuse(T,R):
    m = T.shape[1] # number of states
    r = R.shape[1] # should also be the number of states?
    Q_0 = dot(inv(identity(m**2)-kron(T,T)),dot(R,R.T).ravel('F'))
    return zeros((m,1)), Q_0.reshape(r,r,order='F')


def kalmansmooth(F, A, H, Q, R, y, X, xi10):
    pass


def kalmanfilter(F, A, H, Q, R, y, X, xi10, ntrain, history=False):
    """
    Returns the negative log-likelihood of y conditional on the information set

    Assumes that the initial state and all innovations are multivariate
    Gaussian.

    Parameters
    -----------
    F : array-like
        The (r x r) array holding the transition matrix for the hidden state.
    A : array-like
        The (nobs x k) array relating the predetermined variables to the
        observed data.
    H : array-like
        The (nobs x r) array relating the hidden state vector to the
        observed data.
    Q : array-like
        (r x r) variance/covariance matrix on the error term in the hidden
        state transition.
    R : array-like
        (nobs x nobs) variance/covariance of the noise in the observation
        equation.
    y : array-like
        The (nobs x 1) array holding the observed data.
    X : array-like
        The (nobs x k) array holding the predetermined variables data.
    xi10 : array-like
        Is the (r x 1) initial prior on the initial state vector.
    ntrain : int
        The number of training periods for the filter.  This is the number of
        observations that do not affect the likelihood.


    Returns
    -------
    likelihood
        The negative of the log likelihood
    history or priors, history of posterior
        If history is True.

    Notes
    -----
    No input checking is done.
    """
# uses log of Hamilton 13.4.1
    F = np.asarray(F)
    H = np.atleast_2d(np.asarray(H))
    n = H.shape[1]  # remember that H gets transposed
    y = np.asarray(y)
    A = np.asarray(A)
    X = np.asarray(X)
    if y.ndim == 1: # note that Y is in rows for now
        y = y[:,None]
    nobs = y.shape[0]
    xi10 = np.atleast_2d(np.asarray(xi10))
#    if xi10.ndim == 1:
#        xi10[:,None]
    if history:
        state_vector = [xi10]
    Q = np.asarray(Q)
    r = xi10.shape[0]
# Eq. 12.2.21, other version says P0 = Q
#    p10 = np.dot(np.linalg.inv(np.eye(r**2)-np.kron(F,F)),Q.ravel('F'))
#    p10 = np.reshape(P0, (r,r), order='F')
# Assume a fixed, known intial point and set P0 = Q
#TODO: this looks *slightly * different than Durbin-Koopman exact likelihood
# initialization p 112 unless I've misunderstood the notational translation.
    p10 = Q

    loglikelihood = 0
    for i in range(nobs):
        HTPHR = np.atleast_1d(np.squeeze(chain_dot(H.T,p10,H)+R))
#        print HTPHR
#        print HTPHR.ndim
#        print HTPHR.shape
        if HTPHR.ndim == 1:
            HTPHRinv = 1./HTPHR
        else:
            HTPHRinv = np.linalg.inv(HTPHR) # correct
#        print A.T
#        print X
#        print H.T
#        print xi10
#        print y[i]
        part1 = y[i] - np.dot(A.T,X) - np.dot(H.T,xi10) # correct
        if i >= ntrain: # zero-index, but ntrain isn't
            HTPHRdet = np.linalg.det(np.atleast_2d(HTPHR)) # correct
            part2 = -.5*chain_dot(part1.T,HTPHRinv,part1) # correct
#TODO: Need to test with ill-conditioned problem.
            loglike_interm = (-n/2.) * np.log(2*np.pi) - .5*\
                        np.log(HTPHRdet) + part2
            loglikelihood += loglike_interm

        # 13.2.15 Update current state xi_t based on y
        xi11 = xi10 + chain_dot(p10, H, HTPHRinv, part1)
        # 13.2.16 MSE of that state
        p11 = p10 - chain_dot(p10, H, HTPHRinv, H.T, p10)
        # 13.2.17 Update forecast about xi_{t+1} based on our F
        xi10 = np.dot(F,xi11)
        if history:
            state_vector.append(xi10)
        # 13.2.21 Update the MSE of the forecast
        p10 = chain_dot(F,p11,F.T) + Q
    if not history:
        return -loglikelihood
    else:
        return -loglikelihood, np.asarray(state_vector[:-1])


def updatematrices(params, y, xi10, ntrain, penalty, upperbound, lowerbound):
    """
    TODO: change API, update names

    This isn't general.  Copy of Luca's matlab example.
    """
    paramsorig = params
    # are the bounds binding?
    params = np.min((np.max((lowerbound,params),axis=0),upperbound), axis=0)
    rho = params[0]
    sigma1 = params[1]
    sigma2 = params[2]

    F = np.array([[rho, 0],[0,0]])
    cholQ = np.array([[sigma1,0],[0,sigma2]])
    H = np.ones((2,1))
    q = np.dot(cholQ,cholQ.T)
    loglike = kalmanfilter(F,0,H,q,0, y, 0, xi10, ntrain)
    loglike = loglike + penalty*np.sum((paramsorig-params)**2)
    return loglike


class KalmanFilter(object):
    """
    Kalman Filter code intended for use with the ARMA model.

    Notes
    -----
    The notation for the state-space form follows Durbin and Koopman (2001).

    The observation equations is

    .. math:: y_{t} = Z_{t}\\alpha_{t} + \\epsilon_{t}

    The state equation is

    .. math:: \\alpha_{t+1} = T_{t}\\alpha_{t} + R_{t}\\eta_{t}

    For the present purposed \epsilon_{t} is assumed to always be zero.
    """

    @classmethod
    def T(cls, params, r, k, p):  # F in Hamilton
        """
        The coefficient matrix for the state vector in the state equation.

        Its dimension is r+k x r+k.

        Parameters
        ----------
        r : int
            In the context of the ARMA model r is max(p,q+1) where p is the
            AR order and q is the MA order.
        k : int
            The number of exogenous variables in the ARMA model, including
            the constant if appropriate.
        p : int
            The AR coefficient in an ARMA model.

        References
        ----------
        Durbin and Koopman Section 3.7.
        """
        arr = zeros((r, r), dtype=params.dtype, order="F")
        # allows for complex-step derivative
        params_padded = zeros(r, dtype=params.dtype,
                              order="F")
        # handle zero coefficients if necessary
        # NOTE: squeeze added for cg optimizer
        params_padded[:p] = params[k:p + k]
        arr[:, 0] = params_padded
        # first p params are AR coeffs w/ short params
        arr[:-1, 1:] = eye(r - 1)
        return arr

    @classmethod
    def R(cls, params, r, k, q, p):  # R is H in Hamilton
        """
        The coefficient matrix for the state vector in the observation equation.

        Its dimension is r+k x 1.

        Parameters
        ----------
        r : int
            In the context of the ARMA model r is max(p,q+1) where p is the
            AR order and q is the MA order.
        k : int
            The number of exogenous variables in the ARMA model, including
            the constant if appropriate.
        q : int
            The MA order in an ARMA model.
        p : int
            The AR order in an ARMA model.

        References
        ----------
        Durbin and Koopman Section 3.7.
        """
        arr = zeros((r, 1), dtype=params.dtype, order="F")
        # this allows zero coefficients
        # dtype allows for compl. der.
        arr[1:q + 1, :] = params[p + k:p + k + q][: ,None]
        arr[0] = 1.0
        return arr

    @classmethod
    def Z(cls, r):
        """
        Returns the Z selector matrix in the observation equation.

        Parameters
        ----------
        r : int
            In the context of the ARMA model r is max(p,q+1) where p is the
            AR order and q is the MA order.

        Notes
        -----
        Currently only returns a 1 x r vector [1,0,0,...0].  Will need to
        be generalized when the Kalman Filter becomes more flexible.
        """
        arr = zeros((1, r), order="F")
        arr[:, 0] = 1.
        return arr

    @classmethod
    def geterrors(cls, y, k, k_ar, k_ma, k_lags, nobs, Z_mat, m, R_mat, T_mat,
                  paramsdtype):
        """
        Returns just the errors of the Kalman Filter
        """
        if issubdtype(paramsdtype, np.float64):
            return kalman_loglike.kalman_filter_double(
                y, k, k_ar, k_ma, k_lags, int(nobs), Z_mat, R_mat, T_mat)[0]
        elif issubdtype(paramsdtype, np.complex128):
            return kalman_loglike.kalman_filter_complex(
                y, k, k_ar, k_ma, k_lags, int(nobs), Z_mat, R_mat, T_mat)[0]
        else:
            raise TypeError("dtype %s is not supported "
                            "Please file a bug report" % paramsdtype)

    @classmethod
    def _init_kalman_state(cls, params, arma_model):
        """
        Returns the system matrices and other info needed for the
        Kalman Filter recursions
        """
        paramsdtype = params.dtype
        y = arma_model.endog.copy().astype(paramsdtype)
        k = arma_model.k_exog + arma_model.k_trend
        nobs = arma_model.nobs
        k_ar = arma_model.k_ar
        k_ma = arma_model.k_ma
        k_lags = arma_model.k_lags

        if arma_model.transparams:
            newparams = arma_model._transparams(params)
        else:
            newparams = params  # don't need a copy if not modified.

        if k > 0:
            y -= dot(arma_model.exog, newparams[:k])

        # system matrices
        Z_mat = cls.Z(k_lags)
        m = Z_mat.shape[1]  # r
        R_mat = cls.R(newparams, k_lags, k, k_ma, k_ar)
        T_mat = cls.T(newparams, k_lags, k, k_ar)
        return (y, k, nobs, k_ar, k_ma, k_lags,
                newparams, Z_mat, m, R_mat, T_mat, paramsdtype)

    @classmethod
    def loglike(cls, params, arma_model, set_sigma2=True):
        """
        The loglikelihood for an ARMA model using the Kalman Filter recursions.

        Parameters
        ----------
        params : array
            The coefficients of the ARMA model, assumed to be in the order of
            trend variables and `k` exogenous coefficients, the `p` AR
            coefficients, then the `q` MA coefficients.
        arma_model : `statsmodels.tsa.arima.ARMA` instance
            A reference to the ARMA model instance.
        set_sigma2 : bool, optional
            True if arma_model.sigma2 should be set.
            Note that sigma2 will be computed in any case,
            but it will be discarded if set_sigma2 is False.

        Notes
        -----
        This works for both real valued and complex valued parameters. The
        complex values being used to compute the numerical derivative. If
        available will use a Cython version of the Kalman Filter.
        """
        # TODO: see section 3.4.6 in Harvey for computing the derivatives in the
        # recursion itself.
        # TODO: this won't work for time-varying parameters
        (y, k, nobs, k_ar, k_ma, k_lags, newparams, Z_mat, m, R_mat, T_mat,
         paramsdtype) = cls._init_kalman_state(params, arma_model)
        if issubdtype(paramsdtype, np.float64):
            loglike, sigma2 =  kalman_loglike.kalman_loglike_double(
                y, k, k_ar, k_ma, k_lags, int(nobs),
                Z_mat, R_mat, T_mat)
        elif issubdtype(paramsdtype, np.complex128):
            loglike, sigma2 =  kalman_loglike.kalman_loglike_complex(
                y, k, k_ar, k_ma, k_lags, int(nobs),
                Z_mat.astype(complex), R_mat, T_mat)
        else:
            raise TypeError("This dtype %s is not supported "
                            " Please files a bug report." % paramsdtype)
        if set_sigma2:
            arma_model.sigma2 = sigma2

        return loglike
