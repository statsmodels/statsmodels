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

from scipy import optimize
import numpy as np
from numpy import dot, identity, kron, log, zeros, pi, exp
from numpy.linalg import inv, pinv
from scikits.statsmodels import chain_dot #Note that chain_dot is a bit slower
from scikits.statsmodels.model import LikelihoodModel
from scikits.statsmodels.regression import yule_walker, GLS
from scipy.linalg import block_diag
from scikits.statsmodels.tsa.tsatools import lagmat

#Fast filtering and smoothing for multivariate state space models
# and The Riksbank -- Strid and Walentin (2008)
# Block Kalman filtering for large-scale DSGE models
# but this is obviously macro model specific

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

class StateSpaceModel(object):
    def __init__(self, endog, exog=None):
        """
        Parameters
        ----------
        endog : array-like
            A (nobs x n) array of observations.
        exog : array-like, optional
            A (nobs x k) array of covariates.

        Notes
        -----
        exog are not handled right now.
        Created with a (V)ARMA in mind, but not really general yet.
        """
        endog = np.asarray(endog)
        if endog.ndim == 1:
            endog = endog[:,None]
        self.endog = endog
        n = endog.shape[1]
        self.n = n
        self.nobs = endog.shape[0]
        self.exog = exog
#        xi10 = np.ararray(xi10)
#        if xi10.ndim == 1:
#            xi10 = xi10[:,None]
#        self.xi10 = xi10
#        self.ntrain = ntrain
#        self.p = ARMA[0]
#        self.q = ARMA[1]
#        self.pq = max(ARMA)
#        self.r = xi10.shape[1]
#        self.A = A
#        self.Q = Q
#        self.F = F
#        self.Hmat =
#        if n == 1:
#            F =

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
        print callable(Q)
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
        xi10 : arry-like
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

#class ARMA(StateSpaceModel):
class ARMA(LikelihoodModel):
    """
    ARMA model using the exact Kalman Filter

    Notes
    -----
    Assumes model is stationary
    """
    def __init__(self, endog, exog=None, constant=True, order=(0,0)):
        if exog is None and constant:
            exog = np.ones((len(endog),1))
            constant = False # to skip next logic
        elif exog is not None and constant:
            exog = add_constant(exog,prepend=True)
        super(ARMA, self).__init__(endog, exog)
        p,q = map(int,order) #TODO: will need to use these to ensure that
                    # some of params stays zero
        r = max(p,q+1)
        if exog is not None:
            k = exog.shape[1]  # number of exogenous variables, incl. const.
            Z = np.zeros((self.nobs, k+r))
            Z[:,:k] = exog
            Z[:,k] = 1.
        else:
            k = 0
            Z = np.zeros((self.nobs, r))
            Z[:,0] = 1. # this is inefficient for no constant because
                        # z_i = z_j for all i,j
        self.Z = Z
        self.k = k
        #NOTE: above is for a stationary ARMA, no seasonality
        #NOTE: Z is H' in Hamilton
        self.r = r
        self.p = p
        self.q = q

    #TODO: this assumes no constant.
    def T(self,params): # F in Hamilton
        """
        The coefficient matrix for the state vector in the state equation.

        Its dimension is r+k x r+k.

        Reference
        ---------
        Durbin and Koopman Section 3.7.
        """
        r = self.r
        k = self.k
        p = self.p
        arr = np.zeros((r,r))
        params_padded = np.zeros(r) # handle zero coefficients if necessary
        #NOTE: squeeze added for cg optimizer
        params_padded[:p] = params[k:p+k]
#        arr[:,0] = params[:r]   # first r params are AR coeffs
        arr[:,0] = params_padded   # first p params are AR coeffs w/ short params
        arr[:-1,1:] = np.eye(r-1)
        arr = block_diag(np.eye(k),arr)
        return arr

    def R(self, params): # R is H in Hamilton
        """
        The coefficient matrix for the state vector in the observation equation.

        Its dimension is r+k x 1.

        Reference
        ---------
        Durbin and Koopman Section 3.7.
        """
        r = self.r
        k = self.k
        q = self.q
        p = self.p
#        arr = params[-r-1:-1].tolist()  # last r-2 to -1 params are MA coeffs
        arr = np.zeros(r-1) # this allows zero coefficients
#NOTE: squeeze added because
        arr[:q] = params[p:p+q]  # p to p+q short params are MA coeffs
#        arr[0] = 1.
        arr = np.hstack(([0]*k + [1.0] ,arr))[:,None]
#        arr = np.asarray([0]*k + arr)[:,None]
        return arr

    def loglike1(self, params):
        """
        Compute the loglikelihood using the exact Kalman filter
        """
#TODO: Define functions within loglike to save call to getattr?
#NOTE: there is more overhead, so the below is more efficient
        Z = self.Z
        r = self.r
        k = self.k
        nobs = self.nobs

        # initialization
        # cf. Harvey section 3.3.4, Durbin-Koopman 5.1 - 5.6, and Hamilton 13.2
        # DK is probably the most comprehensive treatment
        # only one initialization step for stationary ARMA model

        # treat delta as diffuse for initialization
        # doesn't necessarily matter for stationary ARMA model
        alpha = np.zeros((r+k,1)) # initial state is null vector
        #TODO: what about constant?  check Harvey
        #Harvey includes constant terms, but can just stack into alpha_0
        # and estimate like anything else, see 3.2.3a
        #TODO: alpha gets beta stacked on top of it for exog != None

        R_mat = self.R(params)
        m = Z.shape[1] # r + k
        T_mat = self.T(params)
        Q_0 = np.dot(inv(identity(m**2)-np.kron(T_mat,T_mat)),
                    np.dot(R_mat,R_mat.T).ravel('F'))
        Q_0 = Q_0.reshape(r,r,order='F')
        P = Q_0

        # update the initialized states, etc. d times
        # where d == the number of states with unknown means and variances
        # in the case of the ARMA-style models this is always d == 1

#        loglikelihood = -nobs/2 * np.log(2*np.pi)
#NOTE: do this later since it's constant, speed this up
        loglikelihood = 0.
        sigma2 = 0

        # loop the below for d,...,n
        for i in xrange(int(nobs)):
            #TODO: history stores these values
            # Predict
            v = y[i] - np.dot(Z[i,None],alpha) # one-step forecast error
            F = chain_dot(Z[i,None], P, Z[i,None].T) #+H=0 for ARMA, var. forecast err
#NOTE: monitor the above for convergence to steady-state and switching to
# fast recursion
            Finv = 1./F # always scalar for univariate series
            K = chain_dot(T_mat,P,Z[i,None].T,Finv) # Kalman Gain Matrix
            # update state
            alpha = np.dot(T_mat, alpha) + np.dot(K,v)
#            Q_0 = params[-1]  # doesn't need to be in a loop

            L = T_mat - np.dot(K,Z[i,None])
            P = chain_dot(T_mat, P, L.T) + np.dot(R_mat,R_mat.T)

#            loglikelihood -= 1/2. * (np.log(F) + chain_dot(v.T,Finv,v))
# OR
# we could concentrate the likelihood as in Harvey Section 3.4
# where L_c = sum(log(f_t) + T*log(sigma_2))
# where sigma_2 = 1/T * sum(v_t**2/f_t)
            loglikelihood += np.log(F)
        sigma2 = 1./nobs * np.sum(v**2/F)
        loglike = -.5 *(loglikelihood + nobs*np.log(sigma2))
        loglike -= nobs/2. * (np.log(2*np.pi)+1)
#        print params, loglike - nobs/2. * np.log(2*np.pi + 1)
        return loglike.squeeze()
#TODO: check for Steady-State convergence to reuse terms


#TODO: this is going to bonk on trend terms and constant terms
#will return zeros because of zeros_like
    def _transparams(self, params):
        """
        Transforms params to induce stationarity/invertability.

        Reference
        ---------
        Jones(1980)
        """
        p,q,k = self.p, self.q, self.k
        newparams = np.zeros_like(params) # = params.copy() # no copy below
            # AR Coeffs
        if p != 0:
            newparams[k:k+p] = ((1-exp(-params[k:k+p]))/(1+exp(-params[k:k+p]))).copy()
            tmp = ((1-exp(-params[k:k+p]))/(1+exp(-params[k:k+p]))).copy()

                # levinson-durbin to get pacf
            for j in range(1,p):
                a = newparams[k+j]
                for kiter in range(j):
                    tmp[kiter] -= a * newparams[k+j-kiter-1]
                newparams[k:k+j] = tmp[:j]
#                params[k:k+p] = newparams

            # MA Coeffs
        if q != 0:
            newparams[k+p:] = ((1-exp(-params[k+p:k+p+q]))/\
                             (1+exp(-params[k+p:k+p+q]))).copy()
            tmp = ((1-exp(-params[k+p:k+p+q]))/\
                        (1+exp(-params[k+p:k+p+q]))).copy()

            # levinson-durbin to get macf
            for j in range(1,q):
                b = newparams[k+p+j]
                for kiter in range(j):
                    tmp[kiter] += b * newparams[k+p+j-kiter-1]
                newparams[k+p:k+p+j] = tmp[:j]
#                params[k+p:k+p+q] = newparams
                #TODO: might be able to speed up the above, but shouldn't be too much
        return newparams



    def _invtransparams(self, start_params):
        """
        Inverse of the Jones reparameterization
        """
        p,q,k = self.p, self.q, self.k
        newparams = start_params.copy()
        arcoefs = newparams[k:k+p]
        macoefs = newparams[k+p:]
        # AR coeffs
        if p != 0:
            tmp = arcoefs.copy()
#            newparams = arcoefs.copy()
            for j in range(p-1,0,-1):
                a = arcoefs[j]
                for kiter in range(j):
                    tmp[kiter] = (arcoefs[kiter]+a*arcoefs[j-kiter-1])/(1-a**2)
                arcoefs[:j] = tmp[:j]
            invarcoefs = -log((1-arcoefs)/(1+arcoefs))
            newparams[k:k+p] = invarcoefs
        # MA coeffs
        if q != 0:
            tmp = macoefs.copy()
#            newparams = macoefs.copy()
            for j in range(q-1,0,-1):
                b = macoefs[j]
                for kiter in range(j):
                    tmp[kiter] = (macoefs[kiter]-b *macoefs[j-kiter-1])/(1-b**2)
                macoefs[:j] = tmp[:j]
            invmacoefs = -log((1-macoefs)/(1+macoefs))
            newparams[k+p:k+p+q] = invmacoefs
        return newparams

    def loglike(self, params):

#TODO: see section 3.4.6 in Harvey for computing the derivatives in the
# recursion itself.
        Z = self.Z
        y = self.endog.copy() #TODO: remove copy if you can
        k = self.k
        if k > 0:
            y -= np.dot(self.exog, params[:k])
        nobs = self.nobs
        m = Z.shape[1] # r + k
        alpha = zeros((m,1)) # if constant (I-T)**-1 * c
        p = self.p
        q = self.q
        r = self.r

        #NOTE: reparameterization suggested in Jones (1980)
        if self.transparams:
            newparams = self._transparams(params)
        else:
            newparams = params  # don't need a copy if not modified.

        R_mat = self.R(newparams)
        T_mat = self.T(newparams)
        Q_0 = dot(inv(identity(m**2)-kron(T_mat,T_mat)),
                            dot(R_mat,R_mat.T).ravel('F'))
#        Q_0 = dot(pinv(identity((m-k)**2)-kron(T_mat[k:,k:],
#                    T_mat[k:,k:])), dot(R_mat[k:],R_mat[k:].T).ravel('F'))
        #TODO: above is only valid if Eigenvalues of T_mat are inside the
        # unit circle, if not then Q_0 = kappa * eye(m**2)
        # w/ kappa some large value say 1e7, but DK recommends not doing this
        # for a diffuse prior
        Q_0 = Q_0.reshape(r,r,order='F')
        #TODO: should order be m,m if const?
#        if k >= 1:
#            kappa = 1
            # kappa = 1e6
#            P = block_diag(identity(k)*kappa,Q_0)
#        else:
#            P = Q_0
        P = Q_0
        sigma2 = 0
        loglikelihood = 0
        v = zeros((nobs,1))
#TODO: replace this with kiter, see AR model transparams
        F = zeros((nobs,1))
        #NOTE: can only do quick recursions if Z is time-invariant
        #so could have recursions for pure ARMA vs ARMAX
        for i in xrange(int(nobs)):
            # Predict
            Z_mat = Z[i,None]
            v_mat = y[i] - dot(Z_mat,alpha) # one-step forecast error
            v[i] = v_mat
            F_mat = dot(dot(Z_mat, P), Z_mat.T)
            F[i] = F_mat
#            print F[i]
            Finv = 1./F_mat # always scalar for univariate series
            K = dot(dot(dot(T_mat,P),Z_mat.T),Finv) # Kalman Gain Matrix
            # update state
            alpha = dot(T_mat, alpha) + dot(K,v_mat)
            L = T_mat - dot(K,Z_mat)
            P = dot(dot(T_mat, P), L.T) + dot(R_mat, R_mat.T)
            loglikelihood += log(F_mat)

        sigma2 = 1./nobs * np.sum(v**2 / F)
        loglike = -.5 *(loglikelihood + nobs*log(sigma2))
        loglike -= nobs/2. * (log(2*pi) + 1)
#        print params, loglike
        self.sigma2 = sigma2
        return loglike.squeeze()


    def fit(self, start_params=None, transparams=True):
        self.transparams = transparams
        r = self.r
        p = self.p
        q = self.q
        k = self.k
#        start_params = np.zeros((r*2)) # use something else?
                                       #NOTE: this is incorrect
                                       # it needs to include the
                                       # variance terms
        # should be something like r * 2 (with zeros) + k + # of terms in Q
#        start_params = np.zeros((r*2 + k + 1)) # use something else?
#        start_params[-1] = self.endog.std()**2  # estimate of state variance
                                                # zero is bad first guess for
                                                # optimization
        loglike = lambda params: -self.loglike(params)
        # specify which coefficients should be zero according to (p,q)
        # AR restrictions
#        bounds = [(None,)*2]*p + [(0.0,)*2] * (r-p)
        # MA restrictions
#        bounds += [(1.0,1.0)] + [(None,)*2]*q + [(0.0,)*2] * (r-(q+1))
        # Exog restrictions
#        bounds += [(None,)*2]*k
        # Variance restriction, should it be positive instead of 0,inf?
#        bounds += [(0.0,None)]
        # could drop one parameter then?
#        results = optimize.fmin_l_bfgs_b(loglike, start_params,
#                    approx_grad=True, m=30, pgtol=1e-12, factr=10.0,
#                    bounds = bounds, iprint=0)

#TODO: rewrite for bfgs.  Using stata coeffs, the Kalman loglike looks correct
# off by just a small amount, so it's the optimization that looks bad
#TODO: reparameterize variance so that it's always positive
#TODO: write analytic gradient
#NOTE: done below
#        self.results = results
#        params = results[0]
#        llf = results[1]
#        self.params = params
#        self.llf = llf
        if start_params is not None:
            start_params = np.asarray(start_params)
            arcoefs = start_params[k:k+p]
            macoefs = start_params[k+p:k+p+q]
            # reparameterize
        else:
            start_params = zeros((p+q+k))
            endog = self.endog
            exog = self.exog
#TODO: don't forget constant and deterministic parts
# tag all these parts on at the end.  Interested in time series properties.
#TODO: replace this with a call to Conditional Sum of Squares Kalman Filter
            if q != 0:
#TODO: Hannan and Rissanen (1982)
# Fit a long AR process by maximizing AR BIC, then get resids and regress
# y_t on resids and lagged y_t
                resid = endog[p:] - np.dot(lagmat(endog, p, trim='both')[:,1:],
                                    arcoefs)
                macoefs = yule_walker(resid, order=q)[0]
                start_params[k+p:k+p+q] = macoefs
            if p != 0:
                arcoefs = yule_walker(endog, order=p)[0]
                start_params[k:k+p] = arcoefs
            if k != 0:
                start_params[:k] += GLS(endog, exog).fit().params
# inverse of reparameterization for Jones (1980)
        if transparams:
            start_params = self._invtransparams(start_params)

#            # AR coeffs
#            if p != 0:
#                tmp = arcoefs.copy()
#                newparams = arcoefs.copy()
#                for j in range(p-1,0,-1):
#                    a = newparams[j]
#                    for k in range(j):
#                        tmp[k] = (newparams[k] + a * newparams[j-k-1])/(1-a**2)
#                    newparams[:j] = tmp[:j]
#                invarcoefs = -log((1-newparams)/(1+newparams))
#                start_params[k:k+p] = invarcoefs
#                # MA coeffs
#            if q != 0:
#                tmp = macoefs.copy()
#                newparams = macoefs.copy()
#                for j in range(q-1,0,-1):
#                    b = newparams[j]
#                    for k in range(j):
#                        tmp[k] = (newparams[k] - b * newparams[j-k-1])/(1-b**2)
#                    newparams[:j] = tmp[:j]
#                invmacoefs = -log((1-newparams)/(1+newparams))
#                start_params[k+p:k+p+q] = invmacoefs
#

#        results = optimize.fmin_bfgs(loglike, start_params, gtol=1e-2,
#                        full_output=1, maxiter=35, disp=1, norm=np.inf)
        bounds = [(None,)*2]*(p+q+k)
#        bounds = [(-250,250)]*(p+q+k) # to avoid exp overflow with reparam
        results = optimize.fmin_l_bfgs_b(loglike, start_params, approx_grad=True,
                    m=30, pgtol=1e-7, factr=1e3, bounds=bounds, iprint=1)
#        results = optimize.fmin_powell(loglike, start_params, full_output=1,
#                disp=1)
        self.results = results
        if transparams:
            newparams = self._transparams(results[0])
#            # copied from loglike
#            # doesn't modify params in place
#            resparams = results[0]
#            newparams = np.zeros_like(resparams)
#            # AR Coeffs
#            if p != 0:
#                newparams[k:k+p] = ((1-exp(-resparams[k:k+p]))/(1+exp(-resparams[k:k+p]))).copy()
#                tmp = ((1-exp(-resparams[k:k+p]))/(1+exp(-resparams[k:k+p]))).copy()
#
#                # levinson-durbin to get pacf
#                for j in range(1,p):
#                    a = newparams[k+j]
#                    for kiter in range(j):
#                        tmp[kiter] -= a * newparams[k+j-kiter-1]
#                    newparams[k:k+j] = tmp[:j]
##                params[k:k+p] = newparams
#
#            # MA Coeffs
#            if q != 0:
#                newparams[k+p:] = ((1-exp(-resparams[k+p:k+p+q]))/\
#                                (1+exp(-resparams[k+p:k+p+q]))).copy()
#                tmp = ((1-exp(-resparams[k+p:k+p+q]))/\
#                        (1+exp(-resparams[k+p:k+p+q]))).copy()
#
#                # levinson-durbin to get macf
#                for j in range(1,q):
#                    b = newparams[k+p+j]
#                    for kiter in range(j):
#                        tmp[kiter] += b * newparams[k+p+j-kiter-1]
#                    newparams[k+p:j] = tmp[:j]
##TODO: remember that loglike equals fmax - nobs/2. *(np.log2*pi+1)
            self.params = newparams





if __name__ == "__main__":
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
    bounds = zip(lowerbounds,upperbounds) # if you use fmin_l_bfgs_b
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
        dk = zipfile.ZipFile('./DK-data.zip')
    except:
        raise IOError("Install DK-data.zip from http://www.ssfpack.com/DKbook.html")
    nile = dk.open('Nile.dat').readlines()
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
    seatbelt = dk.open('Seatbelt.dat').readlines()
    seatbelt = [map(float,_.split()) for _ in seatbelt[2:]]
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

    # simulate arma process
    np.random.seed(12345)
    y = np.zeros(10000)
    errors = np.random.randn(10000)
    y[0] = np.random.randn()
    # params = .75, .25 ARMA(1,1)
    for i in range(1,len(y)):
        y[i] = .75 * y[i-1] + errors[i] + .25*errors[i-1]
    arma = ARMA(y, constant=False, order=(1,1))
#    arma.fit(start_params = [.75, .25])
#    arma.fit()

    y_arma22 = np.zeros(10000)
    errors = np.random.randn(10000)
    y_arma22[:2] = np.random.randn(2)
    for i in range(2,10000):
        y_arma22[i] = .85 * y_arma22[i-1] - .35*y_arma22[i-2] + errors[i] + .25 * errors[i-1]\
                - .9 * errors[i-2]
    arma22 = ARMA(y_arma22, constant=False, order=(2,2))
    arma22.fit()

# Stata gets [.7515029, .2266321, .9981944] with BFGS, but it's practically
# there on the first iteration...how?
#NOTE: CSS or method mentioned above for ARMA
# with loglikelihood of -14171.91
# our loglike gives -14171.92, so the problem is in the optimizer

#    y_ar = np.zeros(10000)
#    y_ar[0] = np.random.randn()
#    for i in range(1,len(y_ar)):
#        y_ar[i] = y_ar[i-1] * .75 + np.random.randn()
#    ar = ARMA(y_ar, constant=False, order=(1,0))
#    ar.fit()

    import scikits.statsmodels as sm
    data = sm.datasets.sunspots.load()
    ar = ARMA(data.endog, constant=False, order=(9,0))
    ar.fit()


# References
# Harvey (1989)
# Jones (1980) - details missing observations and transformation for stationarity
# he suggests using
# a = [1-exp(-u_k)]/[1+exp(-u_k)] and maximizing wrt to uk but using the
# a's in the actual likelihood.  Same as Hamilton.

#NOTE: if you use the OLS AR estimate and the limited memory BFGS it converges
# if you do the same with just bfgs, it does not converge for the MA coefficient
# it needs better starting values.  Can you invert the MA component and get
# a better starting value?  Still jumps, need the reparameterization.
