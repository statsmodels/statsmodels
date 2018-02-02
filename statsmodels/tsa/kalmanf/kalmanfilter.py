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
#Anderson and Moore `Optimal Filtering` provides a more efficient algorithm
# namely the information filter
# if the number of series is much greater than the number of states
# e.g., with a DSGE model.  See also
# http://www.federalreserve.gov/pubs/oss/oss4/aimindex.html
# Harvey notes that the square root filter will keep P_t pos. def. but
# is not strictly needed outside of the engineering (long series)
from __future__ import print_function
from statsmodels.compat.python import lzip, lmap, callable, range
import numpy as np
from numpy import dot, identity, kron, log, zeros, pi, exp, eye, issubdtype, ones
from numpy.linalg import inv, pinv
from statsmodels.tools.tools import chain_dot
from . import kalman_loglike

#Fast filtering and smoothing for multivariate state space models
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

#TODO: this works if it gets refactored, but it's not quite as accurate
# as KalmanFilter
#    def loglike_exact(self, params):
#        """
#        Exact likelihood for ARMA process.
#
#        Notes
#        -----
#        Computes the exact likelihood for an ARMA process by modifying the
#        conditional sum of squares likelihood as suggested by Shephard (1997)
#        "The relationship between the conditional sum of squares and the exact
#        likelihood for autoregressive moving average models."
#        """
#        p = self.p
#        q = self.q
#        k = self.k
#        y = self.endog.copy()
#        nobs = self.nobs
#        if self.transparams:
#            newparams = self._transparams(params)
#        else:
#            newparams = params
#        if k > 0:
#            y -= dot(self.exog, newparams[:k])
#        if p != 0:
#            arcoefs = newparams[k:k+p][::-1]
#            T = KalmanFilter.T(arcoefs)
#        else:
#            arcoefs = 0
#        if q != 0:
#            macoefs = newparams[k+p:k+p+q][::-1]
#        else:
#            macoefs = 0
#        errors = [0] * q # psuedo-errors
#        rerrors = [1] * q # error correction term
#        # create pseudo-error and error correction series iteratively
#        for i in range(p,len(y)):
#            errors.append(y[i]-sum(arcoefs*y[i-p:i])-\
#                                sum(macoefs*errors[i-q:i]))
#            rerrors.append(-sum(macoefs*rerrors[i-q:i]))
#        errors = np.asarray(errors)
#        rerrors = np.asarray(rerrors)
#
#        # compute bayesian expected mean and variance of initial errors
#        one_sumrt2 = 1 + np.sum(rerrors**2)
#        sum_errors2 = np.sum(errors**2)
#        mup = -np.sum(errors * rerrors)/one_sumrt2
#
#        # concentrating out the ML estimator of "true" sigma2 gives
#        sigma2 = 1./(2*nobs)  * (sum_errors2 - mup**2*(one_sumrt2))
#
#        # which gives a variance of the initial errors of
#        sigma2p = sigma2/one_sumrt2
#
#        llf = -(nobs-p)/2. * np.log(2*pi*sigma2) - 1./(2*sigma2)*sum_errors2 \
#                + 1./2*log(one_sumrt2) + 1./(2*sigma2) * mup**2*one_sumrt2
#        Z_mat = KalmanFilter.Z(r)
#        R_mat = KalmanFilter.R(newparams, r, k, q, p)
#        T_mat = KalmanFilter.T(newparams, r, k, p)
#        # initial state and its variance
#        alpha = zeros((m,1))
#        Q_0 = dot(inv(identity(m**2)-kron(T_mat,T_mat)),
#                dot(R_mat,R_mat.T).ravel('F'))
#        Q_0 = Q_0.reshape(r,r,order='F')
#        P = Q_0
#        v = zeros((nobs,1))
#        F = zeros((nobs,1))
#        B = array([T_mat, 0], dtype=object)
#
#
#        for i in xrange(int(nobs)):
#            v_mat = (y[i],0) - dot(z_mat,B)
#
#        B_0 = (T,0)
#        v_t = (y_t,0) - z*B_t
#        llf = -nobs/2.*np.log(2*pi*sigma2) - 1/(2.*sigma2)*se_n - \
#            1/2.*logdet(Sigma_a) + 1/(2*sigma2)*s_n_prime*sigma_a*s_n
#        return llf
#


class StateSpaceModel(object):
    """
    Generic StateSpaceModel class. Meant to be a base class.

    This class lays out the methods that are to be defined by any child
    class.

    Parameters
    ----------
    endog : array-like
        An `nobs` x `p` array of observations
    exog : array-like, optional
        An `nobs` x `k` array of exogenous variables.
    **kwargs
        Anything provided to the constructor will be attached as an
        attribute.

    Notes
    -----
    The state space model is assumed to be of the form

    y[t] = Z[t].dot(alpha[t]) + epsilon[t]
    alpha[t+1] = T[t].dot(alpha[t]) + R[t].dot(eta[t])

    where

    epsilon[t] ~ N(0, H[t])
    eta[t] ~ N(0, Q[t])
    alpha[0] ~ N(a[0], P[0])

    Where y is the `p` x 1 observations vector, and alpha is the `m` x 1
    state vector.

    References
    -----------
    Durbin, J. and S.J. Koopman. 2001. `Time Series Analysis by State Space
        Methods.` Oxford.
    """
    def __init__(self, endog, exog=None, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

        endog = np.asarray(endog)
        if endog.ndim == 1:
            endog = endog[:,None]
        self.endog = endog
        p = endog.shape[1]
        self.p = nobs
        self.nobs = endog.shape[0]
        if exog:
            self.exog = exog

    def T(self, params):
        pass

    def R(self, params):
        pass

    def Z(self, params):
        pass

    def H(self, params):
        pass

    def Q(self, params):
        pass

    def _univariatefilter(self, params, init_state, init_var):
        """
        Implements the Kalman Filter recursions. Optimized for univariate case.
        """
        y = self.endog
        nobs = self.nobs

        R = self.R
        T = self.T
        Z = self.Z
        H = self.H
        Q = self.Q
        if not init_state and not init_var:
            alpha, P = _init_diffuse(T,R)
        #NOTE: stopped here

    def _univariatefilter_update(self):
        pass
        # does the KF but calls _update after each loop to update the matrices
        # for time-varying coefficients

    def kalmanfilter(self, params, init_state=None, init_var=None):
        """
        Runs the Kalman Filter
        """
        # determine if
        if self.p == 1:
            return _univariatefilter(init_state, init_var)
        else:
            raise ValueError("No multivariate filter written yet")


    def _updateloglike(self, params, xi10, ntrain, penalty, upperbounds, lowerbounds,
            F,A,H,Q,R, history):
        """
        """
        paramsorig = params
        # are the bounds binding?
        if penalty:
            params = np.min((np.max((lowerbounds, params), axis=0),upperbounds),
                axis=0)
        #TODO: does it make sense for all of these to be allowed to be None?
        if F != None and callable(F):
            F = F(params)
        elif F == None:
            F = 0
        if A != None and callable(A):
            A = A(params)
        elif A == None:
            A = 0
        if H != None and callable(H):
            H = H(params)
        elif H == None:
            H = 0
        print(callable(Q))
        if Q != None and callable(Q):
            Q = Q(params)
        elif Q == None:
            Q = 0
        if R != None and callable(R):
            R = R(params)
        elif R == None:
            R = 0
        X = self.exog
        if X == None:
            X = 0
        y = self.endog
        loglike = kalmanfilter(F,A,H,Q,R,y,X, xi10, ntrain, history)
        # use a quadratic penalty function to move away from bounds
        if penalty:
            loglike += penalty * np.sum((paramsorig-params)**2)
        return loglike

#        r = self.r
#        n = self.n
#        F = np.diagonal(np.ones(r-1), k=-1) # think this will be wrong for VAR
                                            # cf. 13.1.22 but think VAR
#        F[0] = params[:p] # assumes first p start_params are coeffs
                                # of obs. vector, needs to be nxp for VAR?
#        self.F = F
#        cholQ = np.diag(start_params[p:]) # fails for bivariate
                                                        # MA(1) section
                                                        # 13.4.2
#        Q = np.dot(cholQ,cholQ.T)
#        self.Q = Q
#        HT = np.zeros((n,r))
#        xi10 = self.xi10
#        y = self.endog
#        ntrain = self.ntrain
 #       loglike = kalmanfilter(F,H,y,xi10,Q,ntrain)

    def fit_kalman(self, start_params, xi10, ntrain=1, F=None, A=None, H=None,
            Q=None,
            R=None, method="bfgs", penalty=True, upperbounds=None,
            lowerbounds=None):
        """
        Parameters
        ----------
        method : str
            Only "bfgs" is currently accepted.
        start_params : array-like
            The first guess on all parameters to be estimated.  This can
            be in any order as long as the F,A,H,Q, and R functions handle
            the parameters appropriately.
        xi10 : array-like
            The (r x 1) vector of initial states.  See notes.
        F,A,H,Q,R : functions or array-like, optional
            If functions, they should take start_params (or the current
            value of params during iteration and return the F,A,H,Q,R matrices).
            See notes.  If they are constant then can be given as array-like
            objects.  If not included in the state-space representation then
            can be left as None.  See example in class docstring.
        penalty : bool,
            Whether or not to include a penalty for solutions that violate
            the bounds given by `lowerbounds` and `upperbounds`.
        lowerbounds : array-like
            Lower bounds on the parameter solutions.  Expected to be in the
            same order as `start_params`.
        upperbounds : array-like
            Upper bounds on the parameter solutions.  Expected to be in the
            same order as `start_params`
        """
        y = self.endog
        ntrain = ntrain
        _updateloglike = self._updateloglike
        params = start_params
        if method.lower() == 'bfgs':
            (params, llf, score, cov_params, func_calls, grad_calls,
            warnflag) = optimize.fmin_bfgs(_updateloglike, params,
                    args = (xi10, ntrain, penalty, upperbounds, lowerbounds,
                        F,A,H,Q,R, False), gtol= 1e-8, epsilon=1e-5,
                        full_output=1)
            #TODO: provide more options to user for optimize
        # Getting history would require one more call to _updatelikelihood
        self.params = params
        self.llf = llf
        self.gradient = score
        self.cov_params = cov_params # how to interpret this?
        self.warnflag = warnflag

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
    def T(cls, params, r, k, p): # F in Hamilton
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
        #NOTE: squeeze added for cg optimizer
        params_padded[:p] = params[k:p+k]
        arr[:,0] = params_padded   # first p params are AR coeffs w/ short params
        arr[:-1,1:] = eye(r-1)
        return arr

    @classmethod
    def R(cls, params, r, k, q, p): # R is H in Hamilton
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
        arr[1:q+1,:] = params[p+k:p+k+q][:,None]
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
        arr = zeros((1,r), order="F")
        arr[:,0] = 1.
        return arr

    @classmethod
    def geterrors(cls, y, k, k_ar, k_ma, k_lags, nobs, Z_mat, m, R_mat, T_mat,
                  paramsdtype):
        """
        Returns just the errors of the Kalman Filter
        """
        if issubdtype(paramsdtype, np.float64):
            return kalman_loglike.kalman_filter_double(y, k, k_ar, k_ma,
                                k_lags, int(nobs), Z_mat, R_mat, T_mat)[0]
        elif issubdtype(paramsdtype, np.complex128):
            return kalman_loglike.kalman_filter_complex(y, k, k_ar, k_ma,
                                k_lags, int(nobs), Z_mat, R_mat, T_mat)[0]
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
        m = Z_mat.shape[1] # r
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
        #TODO: see section 3.4.6 in Harvey for computing the derivatives in the
        # recursion itself.
        #TODO: this won't work for time-varying parameters
        (y, k, nobs, k_ar, k_ma, k_lags, newparams, Z_mat, m, R_mat, T_mat,
                paramsdtype) = cls._init_kalman_state(params, arma_model)
        if issubdtype(paramsdtype, np.float64):
            loglike, sigma2 =  kalman_loglike.kalman_loglike_double(y, k,
                                    k_ar, k_ma, k_lags, int(nobs), Z_mat,
                                    R_mat, T_mat)
        elif issubdtype(paramsdtype, np.complex128):
            loglike, sigma2 =  kalman_loglike.kalman_loglike_complex(y, k,
                                    k_ar, k_ma, k_lags, int(nobs),
                                    Z_mat.astype(complex),
                                    R_mat, T_mat)
        else:
            raise TypeError("This dtype %s is not supported "
                            " Please files a bug report." % paramsdtype)
        if set_sigma2:
            arma_model.sigma2 = sigma2

        return loglike


if __name__ == "__main__":
    import numpy as np
    from scipy.linalg import block_diag
    import numpy as np
    # Make our observations as in 13.1.13
    np.random.seed(54321)
    nobs = 600
    y = np.zeros(nobs)
    rho = [.5, -.25, .35, .25]
    sigma = 2.0 # std dev. or noise
    for i in range(4,nobs):
        y[i] = np.dot(rho,y[i-4:i][::-1]) + np.random.normal(scale=sigma)
    y = y[100:]

    # make an MA(2) observation equation as in example 13.3
    # y = mu + [1 theta][e_t e_t-1]'
    mu = 2.
    theta = .8
    rho = np.array([1, theta])
    np.random.randn(54321)
    e = np.random.randn(101)
    y = mu + rho[0]*e[1:]+rho[1]*e[:-1]
    # might need to add an axis
    r = len(rho)
    x = np.ones_like(y)

    # For now, assume that F,Q,A,H, and R are known
    F = np.array([[0,0],[1,0]])
    Q = np.array([[1,0],[0,0]])
    A = np.array([mu])
    H = rho[:,None]
    R = 0

    # remember that the goal is to solve recursively for the
    # state vector, xi, given the data, y (in this case)
    # we can also get a MSE matrix, P, associated with *each* observation

    # given that our errors are ~ NID(0,variance)
    # the starting E[e(1),e(0)] = [0,0]
    xi0 = np.array([[0],[0]])
    # with variance = 1 we know that
#    P0 = np.eye(2)  # really P_{1|0}

# Using the note below
    P0 = np.dot(np.linalg.inv(np.eye(r**2)-np.kron(F,F)),Q.ravel('F'))
    P0 = np.reshape(P0, (r,r), order='F')

    # more generally, if the eigenvalues for F are in the unit circle
    # (watch out for rounding error in LAPACK!) then
    # the DGP of the state vector is var/cov stationary, we know that
    # xi0 = 0
    # Furthermore, we could start with
    # vec(P0) = np.dot(np.linalg.inv(np.eye(r**2) - np.kron(F,F)),vec(Q))
    # where vec(X) = np.ravel(X, order='F') with a possible [:,np.newaxis]
    # if you really want a "2-d" array
    # a fortran (row-) ordered raveled array
    # If instead, some eigenvalues are on or outside the unit circle
    # xi0 can be replaced with a best guess and then
    # P0 is a positive definite matrix repr the confidence in the guess
    # larger diagonal elements signify less confidence


    # we also know that y1 = mu
    # and MSE(y1) = variance*(1+theta**2) = np.dot(np.dot(H.T,P0),H)

    state_vector = [xi0]
    forecast_vector = [mu]
    MSE_state = [P0]    # will be a list of matrices
    MSE_forecast = []
    # must be numerical shortcuts for some of this...
    # this should be general enough to be reused
    for i in range(len(y)-1):
        # update the state vector
        sv = state_vector[i]
        P = MSE_state[i]
        HTPHR = np.dot(np.dot(H.T,P),H)+R
        if np.ndim(HTPHR) < 2: # we have a scalar
            HTPHRinv = 1./HTPHR
        else:
            HTPHRinv = np.linalg.inv(HTPHR)
        FPH = np.dot(np.dot(F,P),H)
        gain_matrix = np.dot(FPH,HTPHRinv)  # correct
        new_sv = np.dot(F,sv)
        new_sv += np.dot(gain_matrix,y[i] - np.dot(A.T,x[i]) -
                np.dot(H.T,sv))
        state_vector.append(new_sv)
        # update the MSE of the state vector forecast using 13.2.28
        new_MSEf = np.dot(np.dot(F - np.dot(gain_matrix,H.T),P),F.T - np.dot(H,
            gain_matrix.T)) + np.dot(np.dot(gain_matrix,R),gain_matrix.T) + Q
        MSE_state.append(new_MSEf)
        # update the in sample forecast of y
        forecast_vector.append(np.dot(A.T,x[i+1]) + np.dot(H.T,new_sv))
        # update the MSE of the forecast
        MSE_forecast.append(np.dot(np.dot(H.T,new_MSEf),H) + R)
    MSE_forecast = np.array(MSE_forecast).squeeze()
    MSE_state = np.array(MSE_state)
    forecast_vector = np.array(forecast_vector)
    state_vector = np.array(state_vector).squeeze()

##########
#    Luca's example
    # choose parameters governing the signal extraction problem
    rho = .9
    sigma1 = 1
    sigma2 = 1
    nobs = 100

# get the state space representation (Hamilton's notation)\
    F = np.array([[rho, 0],[0, 0]])
    cholQ = np.array([[sigma1, 0],[0,sigma2]])
    H = np.ones((2,1))

# generate random data
    np.random.seed(12345)
    xihistory = np.zeros((2,nobs))
    for i in range(1,nobs):
        xihistory[:,i] = np.dot(F,xihistory[:,i-1]) + \
                np.dot(cholQ,np.random.randn(2,1)).squeeze()
                # this makes an ARMA process?
                # check notes, do the math
    y = np.dot(H.T, xihistory)
    y = y.T

    params = np.array([rho, sigma1, sigma2])
    penalty = 1e5
    upperbounds = np.array([.999, 100, 100])
    lowerbounds = np.array([-.999, .001, .001])
    xi10 = xihistory[:,0]
    ntrain = 1
    bounds = lzip(lowerbounds,upperbounds) # if you use fmin_l_bfgs_b
#    results = optimize.fmin_bfgs(updatematrices, params,
#        args=(y,xi10,ntrain,penalty,upperbounds,lowerbounds),
#        gtol = 1e-8, epsilon=1e-10)
#    array([ 0.83111567,  1.2695249 ,  0.61436685])


    F = lambda x : np.array([[x[0],0],[0,0]])
    def Q(x):
        cholQ = np.array([[x[1],0],[0,x[2]]])
        return np.dot(cholQ,cholQ.T)
    H = np.ones((2,1))
#    ssm_model = StateSpaceModel(y)  # need to pass in Xi10!
#    ssm_model.fit_kalman(start_params=params, xi10=xi10, F=F, Q=Q, H=H,
#            upperbounds=upperbounds, lowerbounds=lowerbounds)
# why does the above take 3 times as many iterations than direct max?

    # compare directly to matlab output
    from scipy import io
#    y_matlab = io.loadmat('./kalman_y.mat')['y'].reshape(-1,1)
#    ssm_model2 = StateSpaceModel(y_matlab)
#    ssm_model2.fit_kalman(start_params=params, xi10=xi10, F=F, Q=Q, H=H,
#            upperbounds=upperbounds, lowerbounds=lowerbounds)

# matlab output
#    thetaunc = np.array([0.7833, 1.1688, 0.5584])
#    np.testing.assert_almost_equal(ssm_model2.params, thetaunc, 4)
    # maybe add a line search check to make sure we didn't get stuck in a local
    # max for more complicated ssm?



# Examples from Durbin and Koopman
    import zipfile
    try:
        dk = zipfile.ZipFile('/home/skipper/statsmodels/statsmodels-skipper/scikits/statsmodels/sandbox/tsa/DK-data.zip')
    except:
        raise IOError("Install DK-data.zip from http://www.ssfpack.com/DKbook.html or specify its correct local path.")
    with dk.open('Nile.dat') as f:
        nile = f.readlines()
    nile = [float(_.strip()) for _ in nile[1:]]
    nile = np.asarray(nile)
#    v = np.zeros_like(nile)
#    a = np.zeros_like(nile)
#    F = np.zeros_like(nile)
#    P = np.zeros_like(nile)
#    P[0] = 10.**7
#    sigma2e = 15099.
#    sigma2n = 1469.1
#    for i in range(len(nile)):
#        v[i] = nile[i] - a[i] # Kalman filter residual
#        F[i] = P[i] + sigma2e # the variance of the Kalman filter residual
#        K = P[i]/F[i]
#        a[i+1] = a[i] + K*v[i]
#        P[i+1] = P[i]*(1.-K) + sigma2n

    nile_ssm = StateSpaceModel(nile)
    R = lambda params : np.array(params[0])
    Q = lambda params : np.array(params[1])
#    nile_ssm.fit_kalman(start_params=[1.0,1.0], xi10=0, F=[1.], H=[1.],
#                Q=Q, R=R, penalty=False, ntrain=0)

# p. 162 univariate structural time series example
    with dk.open('Seatbelt.dat') as f:
        seatbelt = f.readlines()
    seatbelt = [lmap(float,_.split()) for _ in seatbelt[2:]]
    sb_ssm = StateSpaceModel(seatbelt)
    s = 12 # monthly data
# s p.
    H = np.zeros((s+1,1)) # Z in DK, H' in Hamilton
    H[::2] = 1.
    lambdaj = np.r_[1:6:6j]
    lambdaj *= 2*np.pi/s
    T = np.zeros((s+1,s+1))
    C = lambda j : np.array([[np.cos(j), np.sin(j)],[-np.sin(j), np.cos(j)]])
    Cj = [C(j) for j in lambdaj] + [-1]
#NOTE: the above is for handling seasonality
#TODO: it is just a rotation matrix.  See if Robert's link has a better way
#http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=5F5145BE25D61F87478B25AD1493C8F4?doi=10.1.1.110.5134&rep=rep1&type=pdf&ei=QcetSefqF4GEsQPnx4jSBA&sig2=HjJILSBPFgJTfuifbvKrxw&usg=AFQjCNFbABIxusr-NEbgrinhtR6buvjaYA
    from scipy import linalg
    F = linalg.block_diag(*Cj) # T in DK, F in Hamilton
    R = np.eye(s-1)
    sigma2_omega = 1.
    Q = np.eye(s-1) * sigma2_omega
