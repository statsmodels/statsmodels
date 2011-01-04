
import numpy as np
from scikits.statsmodels.decorators import (cache_readonly, cache_writable,
            resettable_cache)
from scipy import optimize
from numpy import dot, identity, kron, log, zeros, pi, exp, eye, abs, empty
from numpy.linalg import inv, pinv
from scikits.statsmodels import add_constant
from scikits.statsmodels.model import (LikelihoodModel, LikelihoodModelResults,
                                        GenericLikelihoodModel)
from scikits.statsmodels.regression import yule_walker, GLS
from tsatools import lagmat
from var import AR
from scikits.statsmodels.sandbox.regression.numdiff import approx_fprime, \
        approx_hess, approx_hess_cs
from kalmanf import KalmanFilter
from scipy.stats import t
from scipy.signal import lfilter
try:
    from kalmanf import kalman_loglike
    fast_kalman = 1
except:
    fast_kalman = 0

class ARMA(GenericLikelihoodModel):
    """
    ARMA model wrapper

    Parameters
    ----------
    endog : array-like
        The endogenous variable.
    exog : array-like, optional
        An optional arry of exogenous variables.
    """
    def __init__(self, endog, exog=None):
        super(ARMA, self).__init__(endog, exog)
        if exog is not None:
            k = exog.shape[1]  # number of exogenous variables, incl. const.
        else:
            k = 0

    def _fit_start_params(self, order):
        """
        Get starting parameters for fit.

        Parameters
        ----------
        order : iterable
            (p,q,k) - AR lags, MA lags, and number of exogenous variables
            including the constant.

        Returns
        -------
        start_params : array
            A first guess at the starting parameters.

        Notes
        -----
        If necessary, fits an AR process with the laglength selected according to        best BIC.  Obtain the residuals.  Then fit an ARMA(p,q) model via OLS
        using these residuals for a first approximation.  Uses a separate OLS
        regression to find the coefficients of exogenous variables.

        References
        ----------
        Hannan, E.J. and Rissanen, J.  1982.  "Recursive estimation of mixed
            autoregressive-moving average order."  `Biometrika`.  69.1.
        """
        p,q,k = order
        start_params = zeros((p+q+k))
        endog = self.endog.copy() # copy because overwritten
        exog = self.exog
        if k != 0:
            ols_params = GLS(endog, exog).fit().params
            start_params[:k] = ols_params
            endog -= np.dot(exog, ols_params).squeeze()
        if q != 0:
            if p != 0:
                armod = AR(endog).fit(ic='bic', trend='nc')
                arcoefs_tmp = armod.params
                p_tmp = armod.laglen
                resid = endog[p_tmp:] - np.dot(lagmat(endog, p_tmp,
                                trim='both'), arcoefs_tmp)
                X = np.column_stack((lagmat(endog,p,'both')[p_tmp+(q-p):],
                    lagmat(resid,q,'both'))) # stack ar lags and resids
                coefs = GLS(endog[p_tmp+q:], X).fit().params
                start_params[k:k+p+q] = coefs
            else:
                start_params[k+p:k+p+q] = yule_walker(endog, order=q)[0]
        if q==0 and p != 0:
            arcoefs = yule_walker(endog, order=p)[0]
            start_params[k:k+p] = arcoefs
        return start_params

    def score(self, params):
        """
        Compute the score function at params.

        Notes
        -----
        This is a numerical approximation.
        """
        loglike = self.loglike
        if self.transparams:
            params = self._invtransparams(params)
#        return approx_fprime(params, loglike, epsilon=1e-5)
        return approx_fprime_cs(params, loglike, epsilon=1e-5)

    def hessian(self, params):
        """
        Compute the Hessian at params,

        Notes
        -----
        This is a numerical approximation.
        """
        loglike = self.loglike
        if self.transparams:
            params = self._invtransparams(params)
#        return approx_hess_cs(params, loglike, epsilon=1e-5)
        return approx_hess(params, loglike, epsilon=1e-5)

    def _transparams(self, params):
        """
        Transforms params to induce stationarity/invertability.

        Reference
        ---------
        Jones(1980)
        """
        p,q,k = self.p, self.q, self.k
        newparams = np.zeros_like(params)

        # just copy exogenous parameters
        if k != 0:
            newparams[:k] = params[:k]

        # AR Coeffs
        if p != 0:
            newparams[k:k+p] = ((1-exp(-params[k:k+p]))/\
                                    (1+exp(-params[k:k+p]))).copy()
            tmp = ((1-exp(-params[k:k+p]))/(1+exp(-params[k:k+p]))).copy()

            # levinson-durbin to get pacf
            for j in range(1,p):
                a = newparams[k+j]
                for kiter in range(j):
                    tmp[kiter] -= a * newparams[k+j-kiter-1]
                newparams[k:k+j] = tmp[:j]

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
            for j in range(q-1,0,-1):
                b = macoefs[j]
                for kiter in range(j):
                    tmp[kiter] = (macoefs[kiter]-b *macoefs[j-kiter-1])/(1-b**2)
                macoefs[:j] = tmp[:j]
            invmacoefs = -log((1-macoefs)/(1+macoefs))
            newparams[k+p:k+p+q] = invmacoefs
        return newparams

    def loglike_kalman(self, params):
        """
        Compute exact loglikelihood for ARMA(p,q) model using the Kalman Filter.
        """
        return KalmanFilter.loglike(params, self)

    def loglike_css(self, params):
        """
        Conditional Sum of Squares likelihood function.
        """
        p = self.p
        q = self.q
        k = self.k
        y = self.endog.copy().astype(params.dtype)
        nobs = self.nobs
        # how to handle if empty?
        if self.transparams:
            newparams = self._transparams(params)
        else:
            newparams = params
        if k > 0:
            y -= dot(self.exog, newparams[:k])
# the order of p determines how many zeros errors to set for lfilter
        b,a = np.r_[1,-newparams[k:k+p]], np.r_[1,newparams[k+p:]]
        zi = np.zeros((max(p,q)), dtype=params.dtype)
        for i in range(p):
            zi[i] = sum(-b[:i+1][::-1] * y[:i+1])
        errors = lfilter(b,a, y, zi=zi)[0][p:]

        ssr = np.dot(errors,errors)
#        sigma2 = ssr/(nobs-2*p) # 2 times p because we drop p observations then
#                                # est. p more
        sigma2 = ssr/(nobs-p)  # not 2 times because gretl doesn't?
        self.sigma2 = sigma2
        llf = -(nobs-p)/2.*(log(2*pi) + log(sigma2)) - ssr/(2*sigma2)
        return llf

    def fit(self, order, start_params=None, trend='c', method = "css-mle",
            transparams=True, solver=None, maxiter=35, full_output=1,
            disp=1, callback=None, **kwargs):
        """
        Fits ARMA(p,q) model using exact maximum likelihood via Kalman filter.

        Parameters
        ----------
        start_params : array-like, optional
            Starting parameters for ARMA(p,q).  If None, the default is given
            by ARMA._fit_start_params.  See there for more information.
        transparams : bool, optional
            Whehter or not to transform the parameters to ensure stationarity.
            Uses the transformation suggested in Jones (1980).  If False,
            no checking for stationarity or invertibility is done.
        method : str {'css-mle','mle','css'}
            This is the loglikelihood to maximize.  If "css-mle", the conditional
            sum of squares likelihood is maximized and its values are used as
            starting values for the computation of the exact likelihood via the
            Kalman filter.  If "mle", the exact likelihood is maximized via the
            Kalman Filter.  If "css" the conditional sum of squares likelihood
            is maximized.  All three methods use `start_params` as starting
            parameters.  See above for more information.
        trend : str {'c','nc'}
            Whehter to include a constant or not.  'c' includes constant,
            'nc' no constant.
        solver : str or None, optional
            Solver to be used.  The default is 'l_bfgs' (limited memory Broyden-
            Fletcher-Goldfarb-Shanno).  Other choices are 'bfgs', 'newton'
            (Newton-Raphson), 'nm' (Nelder-Mead), 'cg' - (conjugate gradient),
            'ncg' (non-conjugate gradient), and 'powell'.
            The limited memory BFGS uses m=30 to approximate the Hessian,
            projected gradient tolerance of 1e-7 and factr = 1e3.  These
            cannot currently be changed for l_bfgs.  See notes for more
            information.
        maxiter : int, optional
            The maximum number of function evaluations. Default is 35.
        tol : float
            The convergence tolerance.  Default is 1e-08.
        full_output : bool, optional
            If True, all output from solver will be available in
            the Results object's mle_retvals attribute.  Output is dependent
            on the solver.  See Notes for more information.
        disp : bool, optional
            If True, convergence information is output.
        callback : function, optional
            Called after each iteration as callback(xk) where xk is the current
            parameter vector.
        kwargs
            See Notes for keyword arguments that can be passed to fit.

        Returns
        -------
        ARMAResults class

        See also
        --------
        scikits.statsmodels.model.LikelihoodModel.fit for more information
        on using the solvers.

        Notes
        ------
        The below is the docstring from
        scikits.statsmodels.LikelihoodModel.fit
        """
        # enforce invertibility
        self.transparams = transparams

        self.method = method.lower()

        # get model order
        p,q = map(int,order)
        r = max(p,q+1)
        self.p = p
        self.q = q
        self.r = r
        endog = self.endog
        exog = self.exog

        # handle exogenous variables
        if exog is None and trend == 'c':   # constant only
            exog = np.ones((len(endog),1))
        elif exog is not None and trend == 'c': # constant plus exogenous
            exog = add_constant(exog, prepend=True)
        elif exog is not None and trend == 'nc':
            # make sure it's not holding constant from last run
            if exog.var() == 0:
                exog = None
        if exog is not None:    # exog only
            k = exog.shape[1]
        else:   # no exogenous variables
            k = 0
            exog = None # set back so can rerun model
        self.exog = exog    # overwrites original exog from __init__
        self.k = k


        # choose objective function
        if method.lower() in ['mle','css-mle']:
            loglike = lambda params: -self.loglike_kalman(params)
            self.loglike = self.loglike_kalman
        if method.lower() == 'css':
            loglike = lambda params: -self.loglike_css(params)
            self.loglike = self.loglike_css

        if start_params is not None:
            start_params = np.asarray(start_params)

        else:
            if method.lower() != 'css-mle': # use Hannan-Rissanen start_params
                start_params = self._fit_start_params((p,q,k))
            else:   # use Hannan-Rissanen to get CSS start_params
                func = lambda params: -self.loglike_css(params)
                #start_params = [.1]*(p+q+k) # different one for k?
                start_params = self._fit_start_params((p,q,k))
                if transparams:
                    start_params = self._invtransparams(start_params)
                bounds = [(None,)*2]*(p+q+k)
                mlefit = optimize.fmin_l_bfgs_b(func, start_params,
                            approx_grad=True, m=12, pgtol=1e-7, factr=1e3,
                            bounds = bounds, iprint=-1)
                start_params = self._transparams(mlefit[0])

        if transparams: # transform initial parameters to ensure invertibility
            start_params = self._invtransparams(start_params)

        if solver is None:  # use default limited memory bfgs
            bounds = [(None,)*2]*(p+q+k)
            mlefit = optimize.fmin_l_bfgs_b(loglike, start_params,
                    approx_grad=True, m=12, pgtol=1e-8, factr=1e2,
                    bounds=bounds, iprint=3)
            self.mlefit = mlefit
            params = mlefit[0]

        else:   # call the solver from LikelihoodModel
            mlefit = super(ARMA, self).fit(start_params, method=solver,
                        maxiter=maxiter, full_output=full_output, disp=disp,
                        callback = callback, **kwargs)
            self.mlefit = mlefit
            params = mlefit.params

        if transparams: # transform parameters back
            params = self._transparams(params)

        self.transparams = False # set to false so methods don't expect transf.

        normalized_cov_params = None

        return ARMAResults(self, params, normalized_cov_params)

    fit.__doc__ += LikelihoodModel.fit.__doc__


class ARMAResults(LikelihoodModelResults):
    """
    Class to hold results from fitting an ARMA model.
    """
    _cache = {}

    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(ARMAResults, self).__init__(model, params, normalized_cov_params,
                scale)
        self.sigma2 = model.sigma2
        self.nobs = model.nobs
        self.k = model.k
        self.p = model.p
        self.q = model.q
        self._cache = resettable_cache()

    @cache_readonly
    def arroots(self):
        return np.roots(np.r_[1,-self.arparams])**-1

    @cache_readonly
    def maroots(self):
        return np.roots(np.r_[1,self.maparams])**-1

#    @cache_readonly
#    def arfreq(self):
#        return (np.log(arroots/abs(arroots))/(2j*pi)).real

#NOTE: why don't root finding functions work well?
#    @cache_readonly
#    def mafreq(eslf):
#        return


    @cache_readonly
    def arparams(self):
        k = self.k
        return self.params[k:k+self.p]

    @cache_readonly
    def maparams(self):
        k = self.k
        p = self.p
        return self.params[k+p:]

    @cache_readonly
    def llf(self):
        #TODO: needs to carry a method attribute to see which one to use
        return self.model.loglike(self.params)

    @cache_readonly
    def bse(self):
        #TODO: see note above
        if not fast_kalman or self.model.method == "css":
            return np.sqrt(np.diag(-inv(approx_hess_cs(self.params,
                self.model.loglike, epsilon=1e-5))))
        else:
            return np.sqrt(np.diag(-inv(approx_hess(self.params,
                self.model.loglike, epsilon=1e-3)[0])))


    def cov_params(self): # add scale argument?
        func = self.model.loglike
        x0 = self.params
        if not fast_kalman or self.model.method == "css":
            return -inv(approx_hess_cs(x0, func))
        else:
            return -inv(approx_hess(x0, func, epsilon=1e-3)[0])

    def t(self):    # overwrites t() because there is no cov_params
        return self.params/self.bse

    @cache_readonly
    def aic(self):
        return -2*self.llf + 2*(self.q+self.p+self.k+1)

    @cache_readonly
    def bic(self):
        nobs = self.nobs
        p = self.p
        if self.model.method == "css":
            nobs -= p
        return -2*self.llf + np.log(nobs)*(self.q+p+self.k+1)

    @cache_readonly
    def hqic(self):
        nobs = self.nobs
        if self.model.method == "css":
            nobs -= self.p
        return -2*self.llf + 2*(self.q+self.p+self.k+1)*np.log(np.log(nobs))

    @cache_readonly
    def fittedvalues(self):
        model = self.model
        endog = model.endog.copy()
        p = self.p
        exog = model.exog # this is a copy
        if exog is not None:
            if model.method == "css" and p > 0:
                exog = exog[p:]
        if model.method == "css" and p > 0:
            endog = endog[p:]
        fv = endog - self.resid
        # add deterministic part back in
        k = self.k
#TODO: this needs to be commented out for MLE with constant

#        if k != 0:
#            fv += dot(exog, self.params[:k])
        return fv

#TODO: make both of these get errors into functions or methods?
    @cache_readonly
    def resid(self):
        model = self.model
        params = self.params
        y = model.endog.copy()

        #demean for exog != None
        k = model.k
        if k > 0:
            y -= dot(model.exog, params[:k])

        r = model.r
        p = model.p
        q = model.q

        if self.model.method != "css":
            #TODO: move get errors to cython-ized Kalman filter
            nobs = self.nobs

            Z_mat = KalmanFilter.Z(r)
            m = Z_mat.shape[1]
            R_mat = KalmanFilter.R(params, r, k, q, p)
            T_mat = KalmanFilter.T(params, r, k, p)

            #initial state and its variance
            alpha = zeros((m,1))
            Q_0 = dot(inv(identity(m**2)-kron(T_mat,T_mat)),
                                dot(R_mat,R_mat.T).ravel('F'))
            Q_0 = Q_0.reshape(r,r,order='F')
            P = Q_0

            resids = empty((nobs,1), dtype=params.dtype)
            for i in xrange(int(nobs)):
                # Predict
                v_mat = y[i] - dot(Z_mat,alpha) # one-step forecast error
                resids[i] = v_mat
                F_mat = dot(dot(Z_mat, P), Z_mat.T)
                Finv = 1./F_mat # always scalar for univariate series
                K = dot(dot(dot(T_mat,P),Z_mat.T),Finv) # Kalman Gain Matrix
                # update state
                alpha = dot(T_mat, alpha) + dot(K,v_mat)
                L = T_mat - dot(K,Z_mat)
                P = dot(dot(T_mat, P), L.T) + dot(R_mat, R_mat.T)
        else:
            b,a = np.r_[1,-params[k:k+p]], np.r_[1,params[k+p:]]
            zi = np.zeros((max(p,q)))
            for i in range(p):
                zi[i] = sum(-b[:i+1][::-1] * y[:i+1])
            e = lfilter(b,a, y, zi=zi)
            resids = e[0][p:]
        return resids.squeeze()

    @cache_readonly
    def pvalues(self):
        # TODO: is this correct for ARMA?
        df_resid = self.nobs - (self.k+self.q+self.p)
        return t.sf(np.abs(self.t()), df_resid) * 2


if __name__ == "__main__":
    import numpy as np
    import scikits.statsmodels as sm

    # simulate arma process
    from scikits.statsmodels.tsa.arima_process import arma_generate_sample
    y = arma_generate_sample([1., -.75],[1.,.25], nsample=1000)
    arma = ARMA(y)
    res = arma.fit(trend='nc', order=(1,1))

    np.random.seed(12345)
    y_arma22 = arma_generate_sample([1.,-.85,.35],[1,.25,-.9], nsample=1000)
    arma22 = ARMA(y_arma22)
    res22 = arma22.fit(trend = 'nc', order=(2,2))

    # test CSS
    arma22_css = ARMA(y_arma22)
    res22css = arma22_css.fit(trend='nc', order=(2,2), method='css')


    data = sm.datasets.sunspots.load()
    ar = ARMA(data.endog)
    resar = ar.fit(trend='nc', order=(9,0))

    y_arma31 = arma_generate_sample([1,-.75,-.35,.25],[.1], nsample=1000)

    arma31css = ARMA(y_arma31)
    res31css = arma31css.fit(order=(3,1), method="css", trend="nc",
            transparams=True)

    y_arma13 = arma_generate_sample([1., -.75],[1,.25,-.5,.8], nsample=1000)
    arma13css = ARMA(y_arma13)
    res13css = arma13css.fit(order=(1,3), method='css', trend='nc')


# check css for p < q and q < p
    y_arma41 = arma_generate_sample([1., -.75, .35, .25, -.3],[1,-.35],
                    nsample=1000)
    arma41css = ARMA(y_arma41)
    res41css = arma41css.fit(order=(4,1), trend='nc', method='css')

    y_arma14 = arma_generate_sample([1, -.25], [1., -.75, .35, .25, -.3],
                    nsample=1000)
    arma14css = ARMA(y_arma14)
    res14css = arma14css.fit(order=(4,1), trend='nc', method='css')


