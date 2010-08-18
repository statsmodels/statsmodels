"""
This is the VAR class refactored from pymaclab.
"""
from __future__ import division
import copy as COP
import scipy as S
import numpy as np
from numpy import matlib as MAT
from scipy import linalg as LIN #TODO: get rid of this once cleaned up
from scipy import linalg, sparse
from scipy.stats import norm, ss as sumofsq
from scikits.statsmodels.regression import yule_walker
from scikits.statsmodels import GLS, OLS
from scikits.statsmodels.tools import chain_dot
from scikits.statsmodels.tsa.tsatools import lagmat
from scikits.statsmodels.tsa.stattools import add_trend, _autolag
from scikits.statsmodels.model import LikelihoodModelResults, LikelihoodModel
from scikits.statsmodels.decorators import *
from scikits.statsmodels.compatibility import np_slogdet
try:
    from numdifftools import Jacobian, Hessian
except:
    raise Warning("You need to install numdifftools to try out the AR model")


__all__ = ['AR', 'VAR2']

#TODO: move this somewhere to be reused.
def irf(params, shock, omega, nperiods=100, ortho=True):
    """
    Returns the impulse response function for a given shock.

    Parameters
    -----------
    shock : array-like
        An array of shocks must be provided that is shape (neqs,)

    If params is None, uses the model params. Note that no normalizing is
    done to the parameters.  Ie., this assumes the the coefficients are
    the identified structural coefficients.

    Notes
    -----
    TODO: Allow for common recursive structures.
    """
    shock = np.asarray(shock)
    params = np.asanyarray(params)
    neqs = params.shape[0]
    if shock.shape[0] != neqs:
        raise ValueError("Each shock must be specified even if it's zero")
    if shock.ndim > 1:
        shock = np.squeeze(shock)
    for i in range(laglen, laglen+nperiods):
        pass

#TODO: stopped here


#TODO: maxlike isn't working very well for higher lag orders.
#TODO: move this
#NOTE: writing this now to be used with ADF test so that
# _autolag can be generalized to VAR.
class AR(LikelihoodModel):
    def __init__(self, endog, exog=None):
        """
        Autoregressive AR(p) Model
        """
        super(AR, self).__init__(endog, exog)
        if endog.ndim == 1:
            endog = endog[:,None]
        elif endog.ndim > 1 and endog.shape[1] != 1:
            raise ValueError("Only the univariate case is implemented")
        self.endog = endog  # overwrite endog
        if exog is not None:
            raise ValueError("Exogenous variables are not supported for AR.")

    def initialize(self):
        pass

    def loglike(self, params):
        """
        The loglikelihood of an AR(p) process

        Notes
        -----
        Contains constant term.  If the model is fit by OLS then this returns
        the conditonal maximum likelihood.  If it is fit by MLE then the
        (exact) unconditional maximum likelihood is returned.
        """
        #TODO: Math is on Hamilton ~pp 124-5
        #will need to be amended for inclusion of exogenous variables
        nobs = self.nobs
        avobs = self.avobs
        Y = self.Y
        X = self.X
        if self.method == "ols":
            ssr = sumofsq(Y.squeeze()-np.dot(X,params))
            sigma2 = ssr/avobs
            return -avobs/2 * (np.log(2*np.pi) + np.log(sigma2)) -\
                    ssr/(2*sigma2)
        endog = self.endog
        penalty = self.penalty
        laglen = self.laglen

# Try reparamaterization:
# just goes to the edge of the boundary for Newton
# reparameterize to ensure stability -- Hamilton 5.9.1
#        if not np.all(params==0):
#            params = params/(1+np.abs(params))

        if isinstance(params,tuple):
            # broyden (all optimize.nonlin return a tuple until rewrite commit)
            params = np.asarray(params)

        usepenalty = False
        # http://en.wikipedia.org/wiki/Autoregressive_model
        roots = np.roots(np.r_[1,-params[1:]])
        mask = np.abs(roots) >= 1
        if np.any(mask) and penalty:
            mask = np.r_[False, mask]
#            signs = np.sign(params)
#            np.putmask(params, mask, .9999)
#            params *= signs
            usepenalty = True

        yp = endog[:laglen]
        lagstart = self.trendorder
        exog = self.exog
        if exog is not None:
            lagstart += exog.shape[1]
#            xp = exog[:laglen]
        if self.trendorder == 1 and lagstart == 1:
            c = [params[0]] * laglen # constant-only no exogenous variables
        else:
            c = np.dot(X[:laglen, :lagstart], params[:lagstart])
        mup = np.asarray(c/(1-np.sum(params[lagstart:])))
        diffp = yp-mup[:,None]

        # get inv(Vp) Hamilton 5.3.7
        params0 = np.r_[-1, params[lagstart:]]

        p = len(params) - lagstart
        p1 = p+1
        Vpinv = np.zeros((p,p))
        for i in range(lagstart,p1):
            for j in range(lagstart,p1):
                if i <= j and j <= p:
                    part1 = np.sum(params0[:i] * params0[j-i:j])
                    part2 = np.sum(params0[p1-j:p1+i-j]*params0[p1-i:])
                    Vpinv[i-1,j-1] = part1 - part2
        Vpinv = Vpinv + Vpinv.T - np.diag(Vpinv.diagonal())
        # this is correct to here

        diffpVpinv = np.dot(np.dot(diffp.T,Vpinv),diffp).item()
        ssr = sumofsq(Y.squeeze() -np.dot(X,params))

        # concentrating the likelihood means that sigma2 is given by
        sigma2 = 1./avobs * (diffpVpinv + ssr)
        logdet = np_slogdet(Vpinv)[1] #TODO: add check for singularity
        loglike = -1/2.*(nobs*(np.log(2*np.pi) + np.log(sigma2)) - \
                logdet + diffpVpinv/sigma2 + ssr/sigma2)

        if usepenalty:
        # subtract a quadratic penalty since we min the negative of loglike
        #NOTE: penalty coefficient should increase with iterations
        # this uses a static one of 1e3
            print "Penalized!"
            loglike -= 1000 *np.sum((mask*params)**2)
        return loglike

    def score(self, params):
        """
        Notes
        -----
        Need to generalize for AR(p) and for a constant.
        Not correct yet.  Returns numerical gradient.  Depends on package
        numdifftools.
        """
        y = self.Y
        ylag = self.X
        nobs = self.nobs
#        diffsumsq = sumofsq(y-np.dot(ylag,params))
#        dsdr = 1/nobs * -2 *np.sum(ylag*(y-np.dot(ylag,params))[:,None])+\
#                2*params*ylag[0]**2
#        sigma2 = 1/nobs*(diffsumsq-ylag[0]**2*(1-params**2))
#        gradient = -nobs/(2*sigma2)*dsdr + params/(1-params**2) + \
#                1/sigma2*np.sum(ylag*(y-np.dot(ylag, params))[:,None])+\
#                .5*sigma2**-2*diffsumsq*dsdr+\
#                ylag[0]**2*params/sigma2 +\
#                ylag[0]**2*(1-params**2)/(2*sigma2**2)*dsdr
        if self.penalty:
            pass
        j = Jacobian(self.loglike)
        return np.squeeze(j(params))
#        return gradient


    def information(self, params):
        """
        Not Implemented Yet
        """
        return

    def hessian(self, params):
        """
        Returns numerical hessian for now.  Depends on numdifftools.
        """
        h = Hessian(self.loglike)
        return h(params)

    def _stackX(self, laglen, trend):
        """
        Private method to build the RHS matrix for estimation.

        Columns are trend terms, then exogenous, then lags.
        """
        endog = self.endog
        exog = self.exog
        X = lagmat(endog, maxlag=laglen, trim='both')[:,1:]
        if exog is not None:
            X = np.column_stack((exog[laglen:,:], X))
        # Handle trend terms
        if trend == 'c':
            trendorder = 1
        elif trend == 'nc':
            trendorder = 0
        elif trend == 'ct':
            trendorder = 2
        elif trend == 'ctt':
            trendorder = 3
        if trend != 'nc':
            X = add_trend(X,prepend=True, trend=trend)
        self.trendorder = trendorder
        return X

    def fit(self, maxlag=None, method='ols', ic=None, trend='c', demean=False,
            penalty=False, start_params=None, solver=None, maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        """
        Fit the unconditional maximum likelihood of an AR(p) process.

        Parameters
        ----------
        start_params : array-like, optional
            A first guess on the parameters.  Defaults is a vector of zeros.
        method : str {'ols', 'yw'. 'mle', 'umle'}, optional
            ols - Ordinary Leasy Squares
            yw - Yule-Walker
            mle - conditional maximum likelihood
            umle - unconditional (exact) maximum likelihood
        solver : str or None, optional
            Unconstrained solvers:
                Default is 'bfgs', 'newton' (newton-raphson), 'ncg'
                (Note that previous 3 are not recommended at the moment.)
                and 'powell'
            Constrained solvers:
                'bfgs-b', 'tnc'
            See notes.
        maxiter : int, optional
            The maximum number of function evaluations. Default is 35.
        tol = float
            The convergence tolerance.  Default is 1e-08.
        penalty : bool
            Whether or not to use a penalty function.  Default is False,
            though this is ignored at the moment and the penalty is always
            used if appropriate.  See notes.

        Notes
        -----
        The unconstrained solvers use a quadratic penalty (regardless if
        penalty kwd is True or False) in order to ensure that the solution
        stays within (-1,1).  The constrained solvers default to using a bound
        of (-.999,.999).

        See also
        --------
        scikits.statsmodels.model.LikelihoodModel.fit for more information
        on using the solvers.

        The below is the docstring from
        scikits.statsmodels.LikelihoodModel.fit
        """
        self.penalty = penalty
        method = method.lower()
        self.method = method
        nobs = self.nobs
        if maxlag is None:
            maxlag = int(round(12*(nobs/100.)**(1/4.)))
        if demean:
            endog = self.endog.copy() # have to copy if demeaning
            mean = endog.mean()
            endog -= mean
            self.endog_mean = mean
        else:
            endog = self.endog
        exog = self.exog
        laglen = maxlag # stays this if ic is None

        # select lag length
        if ic is not None:
            ic = ic.lower()
            if ic not in ['aic','bic','hqic','t-stat']:
                raise ValueError("ic option %s not understood" % ic)
#            icbest, bestlag = _autolag(AR, endog, None, 1, maxlag, ic)
            # make Y and X with same nobs to compare ICs
            Y = endog[maxlag:]
            self.Y = Y  # attach to get correct fit stats
            X = self._stackX(maxlag, trend)
            self.X = X
            startlag = self.trendorder # trendorder set in the above call
            if exog is not None:
                startlag += exog.shape[1] # add dim happens in super?
            results = {}
            if ic != 't-stat':
                for lag in range(startlag,maxlag+1):
                    # have to reinstantiate the model to keep comparable models
                    endog_tmp = endog[maxlag-lag:]
                    fit = AR(endog_tmp).fit(maxlag=lag, demean=demean)
                    results[lag] = eval('fit.'+ic)
                bestic, bestlag = min((res, k) for k,res in results.iteritems())
            else:
                pass
            laglen = bestlag

        # change to what was chosen by fit method
        self.laglen = laglen
        avobs = nobs - laglen
        self.avobs = avobs
                                                    # Model code

        # redo estimation for best lag
        # LHS
        Y = endog[laglen:,:]
        # make lagged RHS
        X = self._stackX(laglen, trend)
#        X = lagmat(endog, maxlag=laglen, trim='both')[:,1:]
#        if exog is not None:
#            X = np.column_stack((self.exog[laglen:,:], X))
#        # Handle constant, etc.
#        if trend == 'c':
#            trendorder = 1
#        elif trend == 'nc':
#            trendorder = 0
#        elif trend == 'ct':
#            trendorder = 2
#        elif trend == 'ctt':
#            trendorder = 3
#        if trend != 'nc':
#            X = add_trend(X,prepend=True, trend=trend)
#        self.trendorder = trendorder
        self.Y = Y
        self.X = X
        self.df_resid = avobs - laglen - self.trendorder # for compatiblity with

        if solver:
            solver = solver.lower()
#TODO: allow user-specified penalty function
#        if penalty and method not in ['bfgs_b','tnc','cobyla','slsqp']:
#            minfunc = lambda params : -self.loglike(params) - \
#                    self.penfunc(params)
#        else:
        if method == "mle":
            if not solver: # make default?
                solver = 'newton'
            if not start_params:
                start_params = np.zeros((X.shape[1]))
            if solver in ['newton', 'bfgs', 'ncg']:
                return super(AR, self).fit(start_params=start_params, method=solver,
                    maxiter=maxiter, full_output=full_output, disp=disp,
                    callback=callback, **kwargs)
#                return retvals
        elif method == "umle":
#TODO: move this stuff up to LikelihoodModel.fit
            minfunc = lambda params: -self.loglike(params)
            bounds = [(-.999,.999)]   # assume stationarity
            if start_params == None:
                start_params = np.array([0]) # assumes AR(1)
            if method == 'bfgs-b':
                retval = optimize.fmin_l_bfgs_b(minfunc, start_params,
                        approx_grad=True, bounds=bounds)
                self.params, self.llf = retval[0:2]
            if method == 'tnc':
                retval = optimize.fmin_tnc(minfunc, start_params,
                        approx_grad=True, bounds = bounds)
                self.params = retval[0]
            if method == 'powell':
                retval = optimize.fmin_powell(minfunc,start_params)
                self.params = retval[None]
#TODO: write regression tests for Pauli's branch so that
# new line_search and optimize.nonlin can get put in.
# http://projects.scipy.org/scipy/ticket/791
#            if method == 'broyden':
#                retval = optimize.broyden2(minfunc, [.5], verbose=True)
#                self.results = retvar
        elif method == "ols":
            arfit = OLS(Y,X).fit()
            params = arfit.params
            omega = None
            self.params = params
        elif method == "yw":
            params, omega = yule_walker(endog, order=maxlag,
                    method="mle", demean=False)
            # how to handle inference after Yule-Walker?
            self.params = params
            self.omega = omega
        pinv_exog = np.linalg.pinv(X)
        normalized_cov_params = np.dot(pinv_exog, pinv_exog.T)
        arfit = ARResults(self, params, normalized_cov_params)
        return arfit

    fit.__doc__ += LikelihoodModel.fit.__doc__


class ARResults(LikelihoodModelResults):

    _cache = {} # for scale setter

    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(ARResults, self).__init__(model, params, normalized_cov_params,
                scale)
        self._cache = resettable_cache()
        self.nobs = model.nobs
        self.avobs = model.avobs
        self.X = model.X # copy?
        self.Y = model.Y
        self.laglen = model.laglen
        self.trendorder = model.trendorder

    @cache_writable()
    def sigma(self):
        #TODO: allow for DOF correction if exog is included
        return 1./self.avobs * self.ssr

    @cache_writable()   # for compatability with RegressionResults
    def scale(self):
        return self.sigma

    @cache_readonly
    def bse(self):
        if self.model.method == "ols":
            ols_scale = self.ssr/(self.avobs - self.laglen - self.trendorder)
            return np.sqrt(np.diag(self.cov_params(scale=ols_scale)))
        else:
            return np.sqrt(np.diag(self.cov_params()))

    @cache_readonly
    def t(self):    # overwrite t()
        return self.params/self.bse

    @cache_readonly
    def aic(self):
# Lutkepohl
#        return np.log(self.sigma) + 1./self.model.avobs * self.laglen
# Include constant as estimated free parameter and double the loss
        return np.log(self.sigma) + 2 * (1 + self.laglen)/self.avobs
# Stata defintion
#        avobs = self.avobs
#        return -2 * self.llf/avobs + 2 * (self.laglen+self.trendorder)/avobs

    @cache_readonly
    def hqic(self):
        avobs = self.avobs
# Lutkepohl
#        return np.log(self.sigma)+ 2 * np.log(np.log(avobs))/avobs * self.laglen
# R uses all estimated parameters rather than just lags
        return np.log(self.sigma) + 2 * np.log(np.log(avobs))/avobs * \
                (1 + self.laglen)
# Stata
#        avobs = self.avobs
#        return -2 * self.llf/avobs + 2 * np.log(np.log(avobs))/avobs * \
#                (self.laglen + self.trendorder)

    @cache_readonly
    def fpe(self):
        avobs = self.avobs
        laglen = self.laglen
        return ((avobs+laglen+1)/(avobs-laglen-1)) * self.sigma

    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params)

    @cache_readonly
    def bic(self):
        avobs = self.avobs
# Lutkepohl
#        return np.log(self.sigma) + np.log(avobs)/avobs * self.laglen
# Include constant as est. free parameter
        return np.log(self.sigma) + (1 + self.laglen) * np.log(avobs)/avobs
# Stata
#        return -2 * self.llf/avobs + np.log(avobs)/avobs * (self.laglen + \
#                self.trendorder)

    @cache_readonly
    def resid(self):
        model = self.model
        return self.Y.squeeze() - np.dot(self.X, self.params)

    @cache_readonly
    def ssr(self):
        resid = self.resid
        return np.dot(resid, resid)

    @cache_readonly
    def roots(self):
        return np.roots(np.r_[1, -self.params[1:]])

class ARIMA(LikelihoodModel):
    def __init__(self, endog, exog=None):
        """
        ARIMA Model
        """
        super(ARIMA, self).__init__(endog, exog)
        if endog.ndim == 1:
            endog = endog[:,None]
        elif endog.ndim > 1 and endog.shape[1] != 1:
            raise ValueError("Only the univariate case is implemented")
        self.endog = endog # overwrite endog
        if exog is not None:
            raise ValueError("Exogenous variables are not yet supported.")

    def fit(self, order=(0,0,0), method="ssm"):
        """
        Notes
        -----
        Current method being developed is the state-space representation.

        Box and Jenkins outline many more procedures.
        """
        if not hasattr(order, '__iter__'):
            raise ValueError("order must be an iterable sequence.  Got type \
%s instead" % type(order))
        p,d,q = order
        if d > 0:
            raise ValueError("Differencing not implemented yet")
            # assume no constant, ie mu = 0
            # unless overwritten then use w_bar for mu
            Y = np.diff(endog, d, axis=0) #TODO: handle lags?




# Refactor of VAR to be like statsmodels
#inherit GLS, SUR?
#class VAR2(object):
class VAR2(LikelihoodModel):
    def __init__(self, endog, exog=None):
        """
        Vector Autoregression (VAR, VARX) models.

        Parameters
        ----------
        endog
        exog
        laglen

        Notes
        -----
        Exogenous variables are not supported yet
        """
        nobs = float(endog.shape[0])
        self.nobs = nobs
        self.nvars = endog.shape[1] # should this be neqs since we might have
                                   # exogenous data?
                                   #NOTE: Yes
        self.neqs = endog.shape[1]
        super(VAR2, self).__init__(endog, exog)

    def loglike(self, params, omega):
        """
        Returns the value of the VAR(p) log-likelihood.

        Parameters
        ----------
        params : array-like
            The parameter estimates
        omega : ndarray
            Sigma hat matrix.  Each element i,j is the average product of the
            OLS residual for variable i and the OLS residual for variable j or
            np.dot(resid.T,resid)/avobs.  There should be no correction for the
            degrees of freedom.


        Returns
        -------
        loglike : float
            The value of the loglikelihood function for a VAR(p) model

        Notes
        -----
        The loglikelihood function for the VAR(p) is

        .. math:: -\left(\frac{T}{2}\right)\left(\ln\left|\Omega\right|-K\ln\left(2\pi\right)-K\right)
        """
        params = np.asarray(params)
        omega = np.asarray(omega)
        logdet = np_slogdet(omega)
        if logdet[0] == -1:
            raise ValueError("Omega matrix is not positive definite")
        elif logdet[0] == 0:
            raise ValueError("Omega matrix is singluar")
        else:
            logdet = logdet[1]
        avobs = self.avobs
        neqs = self.neqs
        return -(avobs/2.)*(neqs*np.log(2*np.pi)+logdet+neqs)

#TODO: IRF, lag length selection
    def fit(self, method="ols", structural=None, dfk=None, maxlag=None,
            ic=None, trend="c"):
        """
        Fit the VAR model

        Parameters
        ----------
        method : str
            "ols" fit equation by equation with OLS
            "yw" fit with yule walker
            "mle" fit with unconditional maximum likelihood
            Only OLS is currently implemented.
        structural : str, optional
            If 'BQ' - Blanchard - Quah identification scheme is used.
            This imposes long run restrictions. Not yet implemented.
        dfk : int or Bool optional
            Small-sample bias correction.  If None, dfk = 0.
            If True, dfk = neqs * nlags + number of exogenous variables.  The
            user can also provide a number for dfk. Omega is divided by (avobs -
            dfk).
        maxlag : int, optional
            The highest lag order for lag length selection according to `ic`.
            The default is 12 * (nobs/100.)**(1./4).  If ic=None, maxlag
            is the number of lags that are fit for each equation.
        ic : str {"aic","bic","hq", "fpe"} or None, optional
            Information criteria to maximize for lag length selection.
            Not yet implemented for VAR.
        trend, str {"c", "ct", "ctt", "nc"}
            "c" - add constant
            "ct" - constant and trend
            "ctt" - constant, linear and quadratic trend
            "nc" - co constant, no trend
            Note that these are prepended to the columns of the dataset.


        Notes
        -----
        Not sure what to do with structural. Restrictions would be on
        coefficients or on omega.  So should it be short run (array),
        long run (array), or sign (str)?  Recursive?
        """
        if dfk is None:
            self.dfk = 0
        elif dkf is True:
            self.dfk = self.X.shape[1] #TODO: change when we accept
                                          # equations for endog and exog
        else:
            self.dfk = dfk

        nobs = int(self.nobs)

        self.avobs = nobs - maxlag # available obs (sample - pre-sample)


#        #recast indices to integers #TODO: really?  Is it easier to just use
                                     # floats in other places or import
                                     # division?

        # need to recompute after lag length selection
        avobs = int(self.avobs)
        if maxlag is None:
            maxlag = round(12*(nobs/100.)**(1/4.))
        self.laglen = maxlag #TODO: change when IC selection is sorted
#        laglen = se
        nvars = int(self.nvars)
        neqs = int(self.neqs)
        endog = self.endog
        laglen = maxlag
        Y = endog[laglen:,:]

        # Make lagged endogenous RHS
        X = np.zeros((avobs,nvars*laglen))
        for x1 in xrange(laglen):
            X[:,x1*nvars:(x1+1)*nvars] = endog[(laglen-1)-x1:(nobs-1)-x1,:]
#NOTE: the above loop is faster than lagmat
#        assert np.all(X == lagmat(endog, laglen-1, trim="backward")[:-laglen])

        # Prepend Exogenous variables
        if self.exog is not None:
            X = np.column_stack((self.exog[laglen:,:], X))

        # Handle constant, etc.
        if trend == 'c':
            trendorder = 1
        elif trend == 'nc':
            trendorder = 0
        elif trend == 'ct':
            trendorder = 2
        elif trend == 'ctt':
            trendorder = 3
        if trend != 'nc':
            X = add_trend(X,prepend=True, trend=trend)
        self.trendorder = trendorder

        self.Y = Y
        self.X = X

# Two ways to do block diagonal, but they are slow
# diag
#        diag_X = linalg.block_diag(*[X]*nvars)
#Sparse: Similar to SUR
#        spdiag_X = sparse.lil_matrix(diag_X.shape)
#        for i in range(nvars):
#            spdiag_X[i*shape0:shape0*(i+1),i*shape1:(i+1)*shape1] = X
#        spX = sparse.kron(sparse.eye(20,20),X).todia()
#        results = GLS(Y,diag_X).fit()

        lagstart = trendorder
        if self.exog is not None:
            lagstart += self.exog.shape[1] #TODO: is there a variable that
                                           #      holds exog.shapep[1]?


#NOTE: just use GLS directly
        results = []
        for y in Y.T:
            results.append(GLS(y,X).fit())
        params = np.vstack((_.params for _ in results))

#TODO: For coefficient restrictions, will have to use SUR


#TODO: make a separate SVAR class or this is going to get really messy
        if structural and structural.lower() == 'bq':
            phi = np.swapaxes(params.reshape(neqs,laglen,neqs), 1,0)
            I_phi_inv = np.linalg.inv(np.eye(n) - phi.sum(0))
            omega = np.dot(results.resid.T,resid)/(avobs - self.dfk)
            shock_var = chain_dot(I_phi_inv, omega, I_phi_inv.T)
            R = np.linalg.cholesky(shock_var)
            phi_normalize = np.dot(I_phi_inv,R)
            params = np.zeros_like(phi)
            #TODO: apply a dot product along an axis?
            for i in range(laglen):
                params[i] = np.dot(phi_normalize, phi[i])
                params = np.swapaxes(params, 1,0).reshape(neqs,laglen*neqs)
        return VARMAResults(self, results, params)


# Setting standard VAR options
VAR_opts = {}
VAR_opts['IRF_periods'] = 20

#from scikits.statsmodels.sandbox.output import SimpleTable

#TODO: correct results if fit by 'BQ'
class VARMAResults(object):
    """
    Holds the results for VAR models.

    Parameters
    -----------
    model
    results
    params

    Attributes
    ----------
    aic (Lutkepohl 2004)
    avobs : float
        Available observations for estimation.  The size of the whole sample
        less the pre-sample observations needed for lags.
    bic : float

    df_resid : float
        Residual degrees of freedom.
    dfk : float
        Degrees of freedom correction.  Not currently used. MLE estimator of
        omega is used everywhere.
    fittedvalues
    fpe (Lutkepohl 2005, p 146-7).
        See notes.
    laglen
    model
    ncoefs
    neqs
    nobs : int
        Total number of observations in the sample.
    omega : ndarray
        Sigma hat matrix.  Each element i,j is the average product of the OLS
        residual for variable i and the OLS residual for variable j or
        np.dot(resid.T,resid)/avobs.  There is no correction for the degrees
        of freedom.  This is the maximum likelihood estimator of Omega.
    omega_beta_gls
    omega_beta_gls_va
    omega_beta_ols
    omega_beta_va
    params : array
        The fitted parameters for each equation.  Note that the rows are the
        equations and that each row holds lags first then variables, so
        it is the first lag for `neqs` variables, the second lag for `neqs`
        variables, etc. exogenous variables and then the trend variables are
        prepended as columns.
    results : list
        Each entry is the equation by equation OLS results if VAR was fit by
        OLS.

    Methods
    -------

    Notes
    ------
    FPE formula

    \left[\frac{T+Kp+t}{T-Kp-t}\right]^{K}$$\left|\Omega\right|

    Where T = `avobs`
          K = `neqs`
          p = `laglength`
          t = `trendorder`
    """
    def __init__(self, model, results, params):
        self.results = results # most of this won't work, keep?
        self.model = model
        self.avobs = model.avobs
        self.dfk = model.dfk
        self.neqs = model.neqs
        self.laglen = model.laglen
        self.nobs = model.nobs
# it's lag1 of y1, lag1 of y2, lag1 of y3 ... lag2 of y1, lag2 of y2 ...
        self.params = params
        self.ncoefs = self.params.shape[1]
        self.df_resid = model.avobs - self.ncoefs # normalize sigma by this
        self.trendorder = model.trendorder

    @cache_readonly
    def fittedvalues(self):
        np.column_stack((_.fittedvalues for _ in results))
#        return self.results.fittedvalues.reshape(-1, self.neqs, order='F')

    @cache_readonly
    def resid(self):
        results = self.results
        return np.column_stack((_.resid for _ in results))
#        return self._results.resid.reshape(-1,self.neqs,order='F')

    @cache_readonly
    def omega(self):
        resid = self.resid
        return np.dot(resid.T,resid)/(self.avobs - self.dfk)
#TODO: include dfk correction anywhere or not?  No small sample bias?

    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params, self.omega)

#    @cache_readonly
#    def omega(self): # variance of residuals across equations
#        resid = self.resid
#        return np.dot(resid.T,resid)/(self.avobs - self.dfk)

#    @cache_readonly
#    def omega_beta_ols(self): # the covariance of each equation (check)
#        ncoefs = self.params.shape[1]
#        XTXinv = self._results.normalized_cov_params[:ncoefs,:ncoefs]
#        # above is iXX in old VAR
#        obols = map(np.multiply, [XTXinv]*self.neqs, np.diag(self.omega))
#        return np.asarray(obols)

#    @cache_readonly
#    def omega_beta_va(self):
#        return map(np.diag, self.omega_beta_ols)

#    @cache_readonly
#    def omega_beta_gls(self):
#        X = self.model.X
#        resid = self.resid
#        neqs = self.neqs
#        ncoefs = self.ncoefs
#        XTXinv = self._results.normalized_cov_params[:ncoefs,:ncoefs]
        # Get GLS Covariance
        # this is just a list of length nvars, with each
        # XeeX where e is the residuals for that equation
        # really just a scaling argument
#        XeeX = [chain_dot(X.T, resid[:,i][:,None], resid[:,i][:,None].T,
#            X) for i in range(neqs)]
#        obgls = np.array(map(chain_dot, [XTXinv]*neqs, XeeX,
#                [XTXinv]*neqs))
#        return obgls

#    @cache_readonly
#    def omega_beta_gls_va(self):
#        return map(np.diag, self.omega_beta_gls_va)

# the next three properties have rounding error stemming from fittedvalues
# dot vs matrix multiplication vs. old VAR, test with another package

#    @cache_readonly
#    def ssr(self):
#        return self.results.ssr # rss in old VAR

#    @cache_readonly
#    def root_MSE(self):
#        avobs = self.avobs
#        laglen = self.laglen
#        neqs = self.neqs
#        trendorder = self.trendorder
#        return np.sqrt(np.diag(self.omega*avobs/(avobs-neqs*laglen-
#            trendorder)))

    @cache_readonly
    def rsquared(self):
        results = self.results
        return np.vstack((_.rsquared for _ in results))

    @cache_readonly
    def bse(self):
        results = self.results
        return np.vstack((_.bse for _ in results))

    @cache_readonly
    def aic(self):
        logdet = np_slogdet(self.omega)
        if logdet[0] == -1:
            raise ValueError("Omega matrix is not positive definite")
        elif logdet[0] == 0:
            raise ValueError("Omega matrix is singluar")
        else:
            logdet = logdet[1]
        neqs = self.neqs
        trendorder = self.trendorder
        return logdet+ (2/self.avobs) * (self.laglen*neqs**2+trendorder*neqs)

    @cache_readonly
    def bic(self):
        logdet = np_slogdet(self.omega)
        if logdet[0] == -1:
            raise ValueError("Omega matrix is not positive definite")
        elif logdet[0] == 0:
            raise ValueError("Omega matrix is singluar")
        else:
            logdet = logdet[1]
        avobs = self.avobs
        neqs = self.neqs
        trendorder = self.trendorder
        return logdet+np.log(avobs)/avobs*(self.laglen*neqs**2 +
                neqs*trendorder)

    @cache_readonly
    def hqic(self):
        logdet = np_slogdet(self.omega)
        if logdet[0] == -1:
            raise ValueError("Omega matrix is not positive definite")
        elif logdet[0] == 0:
            raise ValueError("Omega matrix is singluar")
        else:
            logdet = logdet[1]
        avobs = self.avobs
        laglen = self.laglen
        neqs = self.neqs
        trendorder = self.trendorder
        return logdet + 2*np.log(np.log(avobs))/avobs * (laglen*neqs**2 +
                trendorder * neqs)
#TODO: do the above correctly handle extra exogenous variables?

    @cache_readonly
    def df_eq(self):
#TODO: change when we accept coefficient restrictions
        return self.ncoefs

    @cache_readonly
    def fpe(self):
        detomega = self.detomega
        avobs = self.avobs
        neqs = self.neqs
        laglen = self.laglen
        trendorder = self.trendorder
        return ((avobs+neqs*laglen+trendorder)/(avobs-neqs*laglen-
            trendorder))**neqs * detomega

    @cache_readonly
    def detomega(self):
        return np.linalg.det(self.omega)

#    @wrap
#    def wrap(self, attr, *args):
#        return self.__getattribute__(attr, *args)

    @cache_readonly
    def bse(self):
        return np.sqrt(np.diag(self.cov_params)).reshape(self.neqs, -1,
                order = 'F')

    @cache_readonly
    def z(self):
        return self.params/self.bse

    @cache_readonly
    def pvalues(self):
        return norm.sf(np.abs(self.z))*2

    @cache_readonly
    def cov_params(self):
        #NOTE: Cov(Vec(B)) = (Z'Z)^-1 kron Omega
        X = self.model.X
        return np.kron(np.linalg.inv(np.dot(X.T,X)), self.omega)
#TODO: this might need to be changed when order is changed and with exog

    #could this just be a standalone function?
    def irf(self, shock, params=None, nperiods=100):
        """
        Make the impulse response function.

        Parameters
        -----------
        shock : array-like
            An array of shocks must be provided that is shape (neqs,)

        If params is None, uses the model params. Note that no normalizing is
        done to the parameters.  Ie., this assumes the the coefficients are
        the identified structural coefficients.

        Notes
        -----
        TODO: Allow for common recursive structures.
        """
        neqs = self.neqs
        shock = np.asarray(shock)
        if shock.shape[0] != neqs:
            raise ValueError("Each shock must be specified even if it's zero")
        if shock.ndim > 1:
            shock = np.squeeze(shock)   # more robust check vs neqs
        if params == None:
            params = self.params
        laglen = int(self.laglen)
        nobs = self.nobs
        avobs = self.avobs
#Needed?
#        useconst = self._useconst
        responses = np.zeros((neqs,laglen+nperiods))
        responses[:,laglen] = shock # shock in the first period with
                                    # all lags set to zero
        for i in range(laglen,laglen+nperiods):
            # flatten lagged responses to broadcast with
            # current layout of params
            # each equation is in a row
            # row i needs to be y1_t-1 y2_t-1 y3_t-1 ... y1_t-2 y1_t-2...
            laggedres = responses[:,i-laglen:i][:,::-1].ravel('F')
            responses[:,i] = responses[:,i] + np.sum(laggedres * params,
                    axis = 1)
        return responses

    def summary(self, endog_names=None, exog_names=None):
        """
        Summary of VAR model
        """
        import time
        from scikits.statsmodels.iolib import SimpleTable
        model = self.model

        if endog_names is None:
            endog_names = self.model.endog_names

        # take care of exogenous names
        if model.exog is not None and exog_names is None:
            exog_names = model.exog_names
        elif exog_names is not None:
            if len(exog_names) != model.exog.shape[1]:
                raise ValueError("The number of exog_names does not match the \
size of model.exog")
        else:
            exog_names = []

        lag_names = []
        # take care of lagged endogenous names
        laglen = self.laglen
        for i in range(1,laglen+1):
            for ename in endog_names:
                lag_names.append('L'+str(i)+'.'+ename)
        # put them together
        Xnames = exog_names + lag_names

        # handle the constant name
        trendorder = self.trendorder
        if trendorder != 0:
            Xnames.insert(0, 'const')
        if trendorder > 1:
            Xnames.insert(0, 'trend')
        if trendorder > 2:
            Xnames.insert(0, 'trend**2')
        Xnames *= self.neqs


        modeltype = model.__class__.__name__
        t = time.localtime()

        ncoefs = self.ncoefs #TODO: change when we allow coef restrictions
        part1_fmt = dict(
            data_fmts = ["%s"],
            empty_cell = '',
            colwidths = 15,
            colsep=' ',
            row_pre = '',
            row_post = '',
            table_dec_above='=',
            table_dec_below='',
            header_dec_below=None,
            header_fmt = '%s',
            stub_fmt = '%s',
            title_align='c',
            header_align = 'r',
            data_aligns = "r",
            stubs_align = "l",
            fmt = 'txt'
        )
        part2_fmt = dict(
            data_fmts = ["%#12.6g","%#12.6g","%#10.4g","%#5.4g"],
            empty_cell = '',
            colwidths = None,
            colsep='    ',
            row_pre = '',
            row_post = '',
            table_dec_above='-',
            table_dec_below='-',
            header_dec_below=None,
            header_fmt = '%s',
            stub_fmt = '%s',
            title_align='c',
            header_align = 'r',
            data_aligns = 'r',
            stubs_align = 'l',
            fmt = 'txt'
        )

        part3_fmt = dict(
            #data_fmts = ["%#12.6g","%#12.6g","%#10.4g","%#5.4g"],
            #data_fmts = ["%#10.4g","%#10.4g","%#10.4g","%#6.4g"],
            data_fmts = ["%#15.6F","%#15.6F","%#15.3F","%#14.3F"],
            empty_cell = '',
            #colwidths = 10,
            colsep='  ',
            row_pre = '',
            row_post = '',
            table_dec_above='=',
            table_dec_below='=',
            header_dec_below='-',
            header_fmt = '%s',
            stub_fmt = '%s',
            title_align='c',
            header_align = 'r',
            data_aligns = 'r',
            stubs_align = 'l',
            fmt = 'txt'
        )

        # Header information
        part1title = "Summary of Regression Results"
        part1data = [[modeltype],
                     ["OLS"], #TODO: change when fit methods change
                     [time.strftime("%a, %d, %b, %Y", t)],
                     [time.strftime("%H:%M:%S", t)]]
        part1header = None
        part1stubs = ('Model:',
                     'Method:',
                     'Date:',
                     'Time:')
        part1 = SimpleTable(part1data, part1header, part1stubs, title=
                part1title, txt_fmt=part1_fmt)

        #TODO: do we want individual statistics or should users just
        # use results if wanted?
        # Handle overall fit statistics
        part2Lstubs = ('No. of Equations:',
                       'Nobs:',
                       'Log likelihood:',
                       'AIC:')
        part2Rstubs = ('BIC:',
                       'HQIC:',
                       'FPE:',
                       'Det(Omega_mle):')
        part2Ldata = [[self.neqs],[self.nobs],[self.llf],[self.aic]]
        part2Rdata = [[self.bic],[self.hqic],[self.fpe],[self.detomega]]
        part2Lheader = None
        part2L = SimpleTable(part2Ldata, part2Lheader, part2Lstubs,
                txt_fmt = part2_fmt)
        part2R = SimpleTable(part2Rdata, part2Lheader, part2Rstubs,
                txt_fmt = part2_fmt)
        part2L.extend_right(part2R)

        # Handle coefficients
        part3data = []
        part3data = zip([self.params.ravel()[i] for i in range(len(Xnames))],
                [self.bse.ravel()[i] for i in range(len(Xnames))],
                [self.z.ravel()[i] for i in range(len(Xnames))],
                [self.pvalues.ravel()[i] for i in range(len(Xnames))])
        part3header = ('coefficient','std. error','z-stat','prob')
        part3stubs = Xnames
        part3 = SimpleTable(part3data, part3header, part3stubs, title=None,
                txt_fmt = part3_fmt)


        table = str(part1) +'\n'+str(part2L) + '\n' + str(part3)
        return table


###############THE VECTOR AUTOREGRESSION CLASS (WORKS)###############
class VAR:
    """
    This is the vector autoregression class. It supports estimation,
    doing IRFs and other stuff.
    """
    def __init__(self,laglen=1,data='none',set_useconst='const'):
        self.VAR_attr = {}
        self.getdata(data)
        if set_useconst == 'noconst':
            self.setuseconst(0)
        elif set_useconst == 'const':
            self.setuseconst(1)
        self.setlaglen(laglen)

    def getdata(self,data):
        self.data = np.mat(data)
        self.VAR_attr['nuofobs'] = self.data.shape[0]
        self.VAR_attr['veclen'] = self.data.shape[1]

    def setuseconst(self,useconst):
        self.VAR_attr['useconst'] = useconst

    def setlaglen(self,laglen):
        self.VAR_attr['laglen'] = laglen
        self.VAR_attr['avobs'] = self.VAR_attr['nuofobs'] - self.VAR_attr['laglen']

    #This is the OLS function and does just that, no input
    def ols(self):
        self.ols_results = {}
        data = self.data
        VAR_attr = self.VAR_attr
        veclen = VAR_attr['veclen']
        laglen = VAR_attr['laglen']
        nuofobs = VAR_attr['nuofobs']
        avobs = VAR_attr['avobs']
        useconst = VAR_attr['useconst']
        y = data[laglen:,:]
        X = MAT.zeros((avobs,veclen*laglen))
        for x1 in range(0,laglen,1):
            X[:,x1*veclen:(x1+1)*veclen] = data[(laglen-1)-x1:(nuofobs-1)-x1,:]
        if self.VAR_attr['useconst'] == 1:
            X = np.hstack((MAT.ones((avobs,1)),X[:,:]))
        self.ols_results['y'] = y
        self.ols_results['X'] = X

        if useconst == 1:
            beta = MAT.zeros((veclen,1+veclen*laglen))
        else:
            beta = MAT.zeros((veclen,veclen*laglen))
        XX = X.T*X
        try:
            iXX = XX.I
        except:
            err = TS_err('Singular Matrix')
            return
        self.iXX = iXX
        for i in range(0,veclen,1):
            Xy = X.T*y[:,i]
            beta[i,:] = (iXX*Xy).T
        self.ols_results['beta'] = beta
        yfit = X*beta.T
        self.ols_results['yfit'] = yfit
        resid = y-yfit
        self.ols_results['resid'] = resid
        #Separate out beta's into B matrices
        self.ols_results['BBs'] = {}
        if self.VAR_attr['useconst'] == 0:
            i1 = 1
            for x1 in range(0,veclen*laglen,laglen):
                self.ols_results['BBs']['BB'+str(i1)] = beta[:,x1:x1+veclen]
                i1 = i1 + 1
        elif self.VAR_attr['useconst'] == 1:
            self.ols_results['BBs']['BB0'] = beta[:,0]
            i1 = 1
            for x in range(0,veclen*laglen,laglen):
                self.ols_results['BBs']['BB'+str(i1)] = beta[:,x+1:x+veclen+1]
                i1 = i1 + 1

        #Make variance-covariance matrix of residuals
        omega = (resid.T*resid)/avobs
        self.ols_results['omega'] = omega
        #Make variance-covariance matrix for est. coefficients under OLS
        omega_beta_ols = (np.multiply(iXX,np.diag(omega)[0]),)
        for i1 in range(1,veclen,1):
            omega_beta_ols = omega_beta_ols+(np.multiply(iXX,np.diag(omega)[i1]),)
        self.ols_results['omega_beta'] = omega_beta_ols
        #Extract just the diagonal variances of omega_beta_ols
        omega_beta_ols_va = (np.diag(omega_beta_ols[0]),)
        for i1 in range(1,veclen,1):
            omega_beta_ols_va = omega_beta_ols_va+(np.diag(omega_beta_ols[i1]),)
        self.ols_results['omega_beta_va'] = omega_beta_ols_va
        #Make variance-covariance matrix of est. coefficients under GLS
        XeeX = X.T*resid[:,0]*resid[:,0].T*X
        self.X = X  #TODO: REMOVE
        self.resid = resid #TODO: REMOVE
        self.XeeX = XeeX #TODO: REMOVE
        omega_beta_gls = (iXX*XeeX*iXX,)
        for i1 in range(1,veclen,1):
            XeeX = X.T*resid[:,i1]*resid[:,i1].T*X
            omega_beta_gls = omega_beta_gls+(iXX*XeeX*iXX,)
        self.ols_results['omega_beta_gls'] = omega_beta_gls
        #Extract just the diagonal variances of omega_beta_gls
        omega_beta_gls_va = np.diag(omega_beta_gls[0])
        for i1 in range(1,veclen,1):
            omega_beta_gls_va = (omega_beta_gls_va,)+(np.diag(omega_beta_gls[i1]),)
        self.ols_results['omega_beta_gls_va'] = omega_beta_gls_va
        sqresid = np.power(resid,2)
        rss = np.sum(sqresid)
        self.ols_results['rss'] = rss
        AIC = S.linalg.det(omega)+(2.0*laglen*veclen**2)/nuofobs
        self.ols_results['AIC'] = AIC
        BIC = S.linalg.det(omega)+(np.log(nuofobs)/nuofobs)*laglen*veclen**2
        self.ols_results['BIC'] = BIC

    def do_irf(self,spos=1,plen=VAR_opts['IRF_periods']):
        self.IRF_attr = {}
        self.IRF_attr['spos'] = spos
        self.IRF_attr['plen'] = plen
        VAR_attr = self.VAR_attr
        veclen = VAR_attr['veclen']
        laglen = VAR_attr['laglen']
        nuofobs = VAR_attr['nuofobs']
        avobs = VAR_attr['avobs']
        useconst = VAR_attr['useconst']
        self.irf_results = {}
        # Strip out the means of vector y
        data = self.data
        dmeans = np.mat(np.average(data,0))
        dmeans = np.mat(dmeans.tolist()*nuofobs)
        self.dmeans = dmeans
        dmdata = data - dmeans
        self.IRF_attr['dmdata'] = dmdata
        # Do OLS on de-meaned series and collect BBs
        if self.VAR_attr['useconst'] == 1:
            self.setuseconst(0)
            self.data = dmdata
            self.ols_comp()
            self.IRF_attr['beta'] = COP.deepcopy(self.ols_comp_results['beta'])
            self.ols()
            omega = COP.deepcopy(self.ols_results['omega'])
            self.IRF_attr['omega'] = COP.deepcopy(self.ols_results['omega'])
            self.setuseconst(1)
            self.data = data
            self.ols_comp()
        elif self.VAR_attr['useconst'] == 0:
            self.data = dmdata
            self.ols_comp()
            self.IRF_attr['beta'] = COP.deepcopy(self.ols_comp_results['beta'])
            self.ols()
            omega = self.ols_results['omega']
            self.IRF_attr['omega'] = COP.deepcopy(self.ols_results['omega'])
            self.data = data
            self.ols_comp()

        # Calculate Cholesky decomposition of omega
        A0 = np.mat(S.linalg.cholesky(omega))
        A0 = A0.T
        A0 = A0.I
        self.IRF_attr['A0'] = A0
        beta = self.IRF_attr['beta']

        #Calculate IRFs using ols_comp_results
        ee = MAT.zeros((veclen*laglen,1))
        ee[spos-1,:] = 1
        shock = np.vstack((A0.I*ee[0:veclen,:],ee[veclen:,:]))
        Gam = shock.T
        Gam_2 = Gam[0,0:veclen]
        for x1 in range(0,plen,1):
            Gam_X = beta*Gam[x1,:].T
            Gam_2 = np.vstack((Gam_2,Gam_X.T[:,0:veclen]))
            Gam = np.vstack((Gam,Gam_X.T))

        self.irf_results['Gammas'] = Gam_2

    #This does OLS in first-order companion form (stacked), no input
    def ols_comp(self):
        self.ols_comp_results = {}
        self.make_y_X_stack()
        VAR_attr = self.VAR_attr
        veclen = VAR_attr['veclen']
        laglen = VAR_attr['laglen']
        nuofobs = VAR_attr['nuofobs']
        avobs = VAR_attr['avobs']
        useconst = VAR_attr['useconst']
        veclen = veclen*laglen
        laglen = 1
        X = self.ols_comp_results['X']
        y = self.ols_comp_results['y']

        if useconst == 1:
            beta = MAT.zeros((veclen,1+veclen*laglen))
        else:
            beta = MAT.zeros((veclen,veclen*laglen))
        XX = X.T*X
        iXX = XX.I
        for i in range(0,veclen,1):
            Xy = X.T*y[:,i]
            beta[i,:] = (iXX*Xy).T

        veclen2 = VAR_attr['veclen']
        for x1 in range(0,beta.shape[0],1):
            for x2 in range(0,beta.shape[1],1):
                if beta[x1,x2] < 1e-7 and x1 > veclen2-1:
                    beta[x1,x2] = 0.0

        for x1 in range(0,beta.shape[0],1):
            for x2 in range(0,beta.shape[1],1):
                if  0.9999999 < beta[x1,x2] < 1.0000001 and x1 > veclen2-1:
                    beta[x1,x2] = 1.0

        self.ols_comp_results['beta'] = beta
        yfit = X*beta.T
        self.ols_comp_results['yfit'] = yfit
        resid = y-yfit
        self.ols_comp_results['resid'] = resid
        #Make variance-covariance matrix of residuals
        omega = (resid.T*resid)/avobs
        self.ols_comp_results['omega'] = omega
        #Make variance-covariance matrix for est. coefficients under OLS
        omega_beta_ols = (np.diag(omega)[0]*iXX,)
        for i1 in range(1,veclen,1):
            omega_beta_ols = omega_beta_ols + (np.diag(omega)[i1]*iXX,)
        self.ols_comp_results['omega_beta'] = omega_beta_ols
        #Extract just the diagonal variances of omega_beta_ols
        omega_beta_ols_va = (np.diag(omega_beta_ols[0]),)
        for i1 in range(1,veclen,1):
            omega_beta_ols_va = omega_beta_ols_va+(np.diag(omega_beta_ols[i1]),)
        self.ols_comp_results['omega_beta_va'] = omega_beta_ols_va
        #Make variance-covariance matrix of est. coefficients under GLS
        XeeX = X.T*resid[:,0]*resid[:,0].T*X
        omega_beta_gls = (iXX*XeeX*iXX,)
        for i1 in range(1,veclen,1):
            XeeX = X.T*resid[:,i1]*resid[:,i1].T*X
            omega_beta_gls = omega_beta_gls+(iXX*XeeX*iXX,)
        self.ols_comp_results['omega_beta_gls'] = omega_beta_gls
        #Extract just the diagonal variances of omega_beta_gls
        omega_beta_gls_va = (np.diag(omega_beta_gls[0]),)
        for i1 in range(1,veclen,1):
            omega_beta_gls_va = omega_beta_gls_va+(np.diag(omega_beta_gls[i1]),)
        self.ols_comp_results['omega_beta_gls_va'] = omega_beta_gls_va
        sqresid = np.power(resid,2)
        rss = np.sum(sqresid)
        self.ols_comp_results['rss'] = rss
        AIC = S.linalg.det(omega[0:veclen,0:veclen])+(2.0*laglen*veclen**2)/nuofobs
        self.ols_comp_results['AIC'] = AIC
        BIC = S.linalg.det(omega[0:veclen,0:veclen])+(np.log(nuofobs)/nuofobs)*laglen*veclen**2
        self.ols_comp_results['BIC'] = BIC

    #This is a function which sets lag by AIC, input is maximum lags over which to search
    def lag_by_AIC(self,lags):
        laglen = self.VAR_attr['laglen']
        lagrange = np.arange(1,lags+1,1)
        i1 = 0
        for i in lagrange:
            self.laglen = i
            self.ols()
            if i1 == 0:
                AIC = self.AIC
                i1 = i1 + 1
            else:
                AIC = append(AIC,self.AIC)
                i1 = i1 + 1
        index = argsort(AIC)
        i1 = 1
        for element in index:
            if element == 0:
                lag_AIC = i1
                break
            else:
                i1 = i1 + 1
        self.VAR_attr['lag_AIC'] = lag_AIC
        self.VAR_attr['laglen'] = lag_AIC

    #This is a function which sets lag by BIC, input is maximum lags over which to search
    def lag_by_BIC(self,lags):
        laglen = self.VAR_attr['laglen']
        lagrange = np.arange(1,lags+1,1)
        i1 = 0
        for i in lagrange:
            self.laglen = i
            self.ols()
            if i1 == 0:
                BIC = self.BIC
                i1 = i1 + 1
            else:
                BIC = append(BIC,self.BIC)
                i1 = i1 + 1
        index = argsort(BIC)
        i1 = 1
        for element in index:
            if element == 0:
                lag_BIC = i1
                break
            else:
                i1 = i1 + 1
        self.VAR_attr['lag_BIC'] = lag_BIC
        self.VAR_attr['laglen'] = lag_BIC

    #Auxiliary function, creates the y and X matrices for OLS_comp
    def make_y_X_stack(self):
        VAR_attr = self.VAR_attr
        veclen = VAR_attr['veclen']
        laglen = VAR_attr['laglen']
        nuofobs = VAR_attr['nuofobs']
        avobs = VAR_attr['avobs']
        useconst = VAR_attr['useconst']
        data = self.data

        y = data[laglen:,:]
        X = MAT.zeros((avobs,veclen*laglen))
        for x1 in range(0,laglen,1):
            X[:,x1*veclen:(x1+1)*veclen] = data[(laglen-1)-x1:(nuofobs-1)-x1,:]
        if self.VAR_attr['useconst'] == 1:
            X = np.hstack((MAT.ones((avobs,1)),X[:,:]))
        try:
            self.ols_results['y'] = y
            self.ols_results['X'] = X
        except:
            self.ols()
            self.ols_results['y'] = y
            self.ols_results['X'] = X

        if useconst == 0:
            y_stack = MAT.zeros((avobs,veclen*laglen))
            y_stack_1 = MAT.zeros((avobs,veclen*laglen))
            y_stack[:,0:veclen] = y
            y_stack_1 = X

            y_stack = np.hstack((y_stack[:,0:veclen],y_stack_1[:,0:veclen*(laglen-1)]))

            self.ols_comp_results['X'] = y_stack_1
            self.ols_comp_results['y'] = y_stack
        else:
            y_stack = MAT.zeros((avobs,veclen*laglen))
            y_stack_1 = MAT.zeros((avobs,1+veclen*laglen))
            y_stack_1[:,0] = MAT.ones((avobs,1))[:,0]
            y_stack[:,0:veclen] = y
            y_stack_1 = X

            y_stack = np.hstack((y_stack[:,0:veclen],y_stack_1[:,1:veclen*(laglen-1)+1]))

            self.ols_comp_results['X'] = y_stack_1
            self.ols_comp_results['y'] = y_stack
"""***********************************************************"""

if __name__ == "__main__":
    import numpy as np
    import scikits.statsmodels as sm
#    vr = VAR(data = np.random.randn(50,3))
#    vr.ols()
#    vr.ols_comp()
#    vr.do_irf()
#    print vr.irf_results.keys()
#    print vr.ols_comp_results.keys()
#    print vr.ols_results.keys()
#    print dir(vr)
    np.random.seed(12345)
    data = np.random.rand(50,3)
    vr = VAR(data = data, laglen=2)
    vr2 = VAR2(endog = data)
    dataset = sm.datasets.macrodata.load()
    data = dataset.data
    XX = data[['realinv','realgdp','realcons']].view((float,3))
    XX = np.diff(np.log(XX), axis=0)
    vrx = VAR(data=XX,laglen=2)
    vrx.ols() # fit
    for i,j in vrx.ols_results.items():
        setattr(vrx, i,j)
    vrx2 = VAR2(endog=XX)
    res = vrx2.fit(maxlag=2)
    vrx3 = VAR2(endog=XX)

    varx = VAR2(endog=XX, exog=np.diff(np.log(data['realgovt']), axis=0))
    resx = varx.fit(maxlag=2)

    sunspots = sm.datasets.sunspots.load()
# Why does R demean the data by defaut?
    ar_ols = AR(sunspots.endog)
    res_ols = ar_ols.fit(maxlag=2, demean=False)
#    ar_mle = AR(sunspots.endog)
#    res_mle = ar_mle.fit(maxlag=1, method="mle", solver="bfgs", maxiter=500,
#            gtol=1e-10, penalty=True)
#    res_mle2 = ar_mle.fit(maxlag=1, method="mle", maxiter=500, penalty=True,
#            tol=1e-13)
#    ar_umle = AR(sunspots.endog)
#    ar_umle.fit(maxlag=4, method="umle")
    ar_yw = AR(sunspots.endog)
    res_yw = ar_yw.fit(maxlag=4, method="yw")

#    # Timings versus talkbox
#    from timeit import default_timer as timer
#    print "Time AR fit vs. talkbox"
#    # generate a long series of AR(2) data
#
#    nobs = 1000000
#    y = np.empty(nobs)
#    y[0:2] = 0
#    for i in range(2,nobs):
#        y[i] = .25 * y[i-1] - .75 * y[i-2] + np.random.rand()
#
#    mod_sm = AR(y)
#    t = timer()
#    res_sm = mod_sm.fit(method="yw", trend="nc", demean=False, maxlag=2)
#    t_end = timer()
#    print str(t_end - t) + " seconds for sm.AR with yule-walker, 2 lags"
#    try:
#        import scikits.talkbox as tb
#    except:
#        raise ImportError("You need scikits.talkbox installed for timings")
#    t = timer()
#    mod_tb = tb.lpc(y, 2)
#    t_end = timer()
#    print str(t_end - t) + " seconds for talkbox.lpc"
#    print """For higher lag lengths ours quickly fills up memory and starts
#thrashing the swap.  Should we include talkbox C code or Cythonize the
#Levinson recursion algorithm?"""

# some data for an example in Box Jenkins
    IBM = np.asarray([460,457,452,459,462,459,463,479,493,490.])
    w = np.diff(IBM)
    theta = .5
