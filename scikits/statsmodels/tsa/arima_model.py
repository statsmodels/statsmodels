import numpy as np
from scikits.statsmodels.tools.decorators import (cache_readonly,
        cache_writable, resettable_cache)
from scipy import optimize
from numpy import dot, identity, kron, log, zeros, pi, exp, eye, abs, empty
from numpy.linalg import inv, pinv
import scikits.statsmodels.base.model as base
import scikits.statsmodels.tsa.base.tsa_model as tsbase
import scikits.statsmodels.base.wrapper as wrap
from scikits.statsmodels.regression.linear_model import yule_walker, GLS
from scikits.statsmodels.tsa.tsatools import (lagmat, add_trend,
        _ar_transparams, _ar_invtransparams, _ma_transparams,
        _ma_invtransparams)
from scikits.statsmodels.tsa.vector_ar import util
from scikits.statsmodels.tsa.ar_model import AR
from scikits.statsmodels.tsa.arima_process import arma2ma
from scikits.statsmodels.sandbox.regression.numdiff import (approx_fprime,
        approx_fprime_cs, approx_hess, approx_hess_cs)
from scikits.statsmodels.tsa.kalmanf import KalmanFilter
from scipy.stats import t, norm
from scipy.signal import lfilter
try:
    from kalmanf import kalman_loglike
    fast_kalman = 1
except:
    fast_kalman = 0

class ARMA(tsbase.TimeSeriesModel):
    """
    Autoregressive Moving Average ARMA(p,q) Model

    Parameters
    ----------
    endog : array-like
        The endogenous variable.
    exog : array-like, optional
        An optional arry of exogenous variables. This should *not* include a
        constant or trend. You can specify this in the `fit` method.
    """
    def __init__(self, endog, exog=None, dates=None, freq=None):
        super(ARMA, self).__init__(endog, exog, dates, freq)
        if exog is not None:
            k_exog = exog.shape[1]  # number of exog. variables excl. const
        else:
            k_exog = 0
        self.k_exog = k_exog

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
        If necessary, fits an AR process with the laglength selected according
        to best BIC.  Obtain the residuals.  Then fit an ARMA(p,q) model via
        OLS using these residuals for a first approximation.  Uses a separate
        OLS regression to find the coefficients of exogenous variables.

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
                p_tmp = armod.k_ar
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
        #if self.transparams:
        #    params = self._invtransparams(params)
        #return approx_fprime(params, loglike, epsilon=1e-5)
        return approx_fprime_cs(params, loglike)

    def hessian(self, params):
        """
        Compute the Hessian at params,

        Notes
        -----
        This is a numerical approximation.
        """
        loglike = self.loglike
        #if self.transparams:
        #    params = self._invtransparams(params)
        if not fast_kalman or self.method == "css":
            return approx_hess_cs(params, loglike, epsilon=1e-5)
        else:
            return approx_hess(params, self.loglike, epsilon=1e-3)[0]


    def _transparams(self, params):
        """
        Transforms params to induce stationarity/invertability.

        Reference
        ---------
        Jones(1980)
        """
        k_ar,k_ma = self.k_ar, self.k_ma
        k = self.k_exog + self.k_trend
        newparams = np.zeros_like(params)

        # just copy exogenous parameters
        if k != 0:
            newparams[:k] = params[:k]

        # AR Coeffs
        if k_ar != 0:
            newparams[k:k+k_ar] = _ar_transparams(params[k:k+k_ar].copy())

        # MA Coeffs
        if k_ma != 0:
            newparams[k+k_ar:] = _ma_transparams(params[k+k_ar:].copy())
        return newparams

    def _invtransparams(self, start_params):
        """
        Inverse of the Jones reparameterization
        """
        k_ar,k_ma = self.k_ar, self.k_ma
        k = self.k_exog + self.k_trend
        newparams = start_params.copy()
        arcoefs = newparams[k:k+k_ar]
        macoefs = newparams[k+k_ar:]
        # AR coeffs
        if k_ar != 0:
            newparams[k:k+k_ar] = _ar_invtransparams(arcoefs)

        # MA coeffs
        if k_ma != 0:
            newparams[k+k_ar:k+k_ar+k_ma] = _ma_invtransparams(macoefs)
        return newparams

    def _get_predict_start(self, start):
        # do some defaults
        if start is None:
            if 'mle' in self.method:
                start = 0
            else:
                start = self.k_ar

        if 'mle' not in self.method:
            if start < self.k_ar:
                raise ValueError("Start must be >= k_ar")

        return super(ARMA, self)._get_predict_start(start)

    def geterrors(self, params, start=None, end=None):

        #start = self._get_predict_start(start) # will be an index of a date
        #end, out_of_sample = self._get_predict_end(end)

        if 'mle' in self.method:
            (y, k, nobs, k_ar, k_ma, k_lags, newparams, Z_mat, m, R_mat,
            T_mat, paramsdtype) = KalmanFilter._init_kalman_state(params, self)
            errors = KalmanFilter.geterrors(y,k,k_ar,k_ma, k_lags, nobs,
                    Z_mat, m, R_mat, T_mat, paramsdtype)
            if isinstance(errors, tuple):
                errors = errors[0] # non-cython version returns a tuple
        else:
            y = self.endog.copy()
            k = self.k_exog + self.k_trend
            if k > 0:
                y -= dot(self.exog, params[:k])

            k_ar = self.k_ar
            k_ma = self.k_ma

            results = self._results
            arparams, maparams = results.arparams, results.maparams
            b,a = np.r_[1,-arparams], np.r_[1,maparams]
            zi = zeros((max(k_ar, k_ma)))
            for i in range(k_ar):
                zi[i] = sum(-b[:i+1][::-1]*y[:i+1])
            e = lfilter(b,a,y,zi=zi)
            errors = e[0][k_ar:]
        return errors.squeeze()

    def predict(self, start=None, end=None, method='static', confint=False,
            fcasterr=False):
        """
        """
        self._check_is_fit()
        results = self._results
        method = self.method

        start = self._get_predict_start(start) # will be an index of a date
        end, out_of_sample = self._get_predict_end(end)

        if end < start:
            raise ValueError("end is before start")
        if end == start + out_of_sample:
            return np.array([])

        k_ar = self.k_ar


        predictedvalues = np.zeros(end+1-start + out_of_sample)
        fittedvalues = results.fittedvalues #get them all then trim

        fv_start = start
        if 'mle' not in method:
            fv_start -= k_ar # start is in terms of endog index
        pv_end = min(len(predictedvalues), len(fittedvalues) - fv_start)
        fv_end = min(len(fittedvalues), end+1)

        predictedvalues[:pv_end] = fittedvalues[fv_start:fv_end]

        # do out of sample fitting
        #params = results.params
        #k_trend = self.k_trend

        return predictedvalues


    def loglike_kalman(self, params):
        """
        Compute exact loglikelihood for ARMA(p,q) model using the Kalman Filter.
        """
        return KalmanFilter.loglike(params, self)

    def loglike_css(self, params):
        """
        Conditional Sum of Squares likelihood function.
        """
        k_ar = self.k_ar
        k_ma = self.k_ma
        k = self.k_exog + self.k_trend
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
        b,a = np.r_[1,-newparams[k:k+k_ar]], np.r_[1,newparams[k+k_ar:]]
        zi = np.zeros((max(k_ar,k_ma)), dtype=params.dtype)
        for i in range(k_ar):
            zi[i] = sum(-b[:i+1][::-1] * y[:i+1])
        errors = lfilter(b,a, y, zi=zi)[0][k_ar:]

        ssr = np.dot(errors,errors)
        sigma2 = ssr/nobs
        self.sigma2 = sigma2
        llf = -nobs/2.*(log(2*pi) + log(sigma2)) - ssr/(2*sigma2)
        return llf

    def fit(self, order, start_params=None, trend='c', method = "css-mle",
            transparams=True, solver=None, maxiter=35, full_output=1,
            disp=5, callback=None, **kwargs):
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
            This is the loglikelihood to maximize.  If "css-mle", the
            conditional sum of squares likelihood is maximized and its values
            are used as starting values for the computation of the exact
            likelihood via the Kalman filter.  If "mle", the exact likelihood
            is maximized via the Kalman Filter.  If "css" the conditional sum
            of squares likelihood is maximized.  All three methods use
            `start_params` as starting parameters.  See above for more
            information.
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
            If True, convergence information is printed.  For the default
            l_bfgs_b solver, disp controls the frequency of the output during
            the iterations. disp < 0 means no output in this case.
        callback : function, optional
            Called after each iteration as callback(xk) where xk is the current
            parameter vector.
        kwargs
            See Notes for keyword arguments that can be passed to fit.

        Returns
        -------
        `scikits.statsmodels.tsa.arima.ARMAResults` class

        See also
        --------
        scikits.statsmodels.model.LikelihoodModel.fit for more information
        on using the solvers.

        Notes
        ------
        If fit by 'mle', it is assumed for the Kalman Filter that the initial
        unkown state is zero, and that the inital variance is
        P = dot(inv(identity(m**2)-kron(T,T)),dot(R,R.T).ravel('F')).reshape(r,
        r, order = 'F')

        The below is the docstring from
        `scikits.statsmodels.LikelihoodModel.fit`
        """
        # enforce invertibility
        self.transparams = transparams

        self.method = method.lower()

        # get model order
        k_ar,k_ma = map(int,order)
        k_lags = max(k_ar,k_ma+1)
        self.k_ar = k_ar
        self.k_ma = k_ma
        self.k_lags = k_lags
        endog = self.endog
        exog = self.exog
        k_exog = self.k_exog
        self.nobs = len(endog) # this is overwritten if method is 'css'

        # handle exogenous variables
        k_trend = 1 # overwritten if no constant
        if exog is None and trend == 'c':   # constant only
            exog = np.ones((len(endog),1))
        elif exog is not None and trend == 'c': # constant plus exogenous
            exog = add_trend(exog, trend='c', prepend=True)
        elif exog is not None and trend == 'nc':
            # make sure it's not holding constant from last run
            if exog.var() == 0:
                exog = None
            k_trend = 0
        if trend == 'nc':
            k_trend = 0
        self.k_trend = k_trend
        self.exog = exog    # overwrites original exog from __init__
        exog_names = self.exog_names or []
        if len(exog_names) != k_trend + k_ma + k_ar + k_exog:
            if exog is not None:
                exog_names = self._data._get_names(exog[:,k_trend:]) or []
            else:
                exog_names = []
            ar_lag_names = util.make_lag_names(self.endog_names, k_ar, k_trend)
            ar_lag_names[k_trend:] = [''.join(('ar.', i))
                                      for i in ar_lag_names[k_trend:]]
            ma_lag_names = util.make_lag_names(self.endog_names, k_ma, 0)
            ma_lag_names = [''.join(('ma.', i)) for i in ma_lag_names]
            self.exog_names = ar_lag_names + ma_lag_names + exog_names
        k = k_trend + k_exog


        # choose objective function
        if method.lower() in ['mle','css-mle']:
            loglike = lambda params: -self.loglike_kalman(params)
            self.loglike = self.loglike_kalman
        if method.lower() == 'css':
            loglike = lambda params: -self.loglike_css(params)
            self.loglike = self.loglike_css
            self.nobs = len(endog) - k_ar #nobs for CSS excludes pre-sample

        if start_params is not None:
            start_params = np.asarray(start_params)

        else:
            if method.lower() != 'css-mle': # use Hannan-Rissanen start_params
                start_params = self._fit_start_params((k_ar,k_ma,k))
            else:   # use Hannan-Rissanen to get CSS start_params
                func = lambda params: -self.loglike_css(params)
                #start_params = [.1]*(k_ar+k_ma+k_exog) # different one for k?
                start_params = self._fit_start_params((k_ar,k_ma,k))
                if transparams:
                    start_params = self._invtransparams(start_params)
                bounds = [(None,)*2]*(k_ar+k_ma+k)
                mlefit = optimize.fmin_l_bfgs_b(func, start_params,
                            approx_grad=True, m=12, pgtol=1e-7, factr=1e3,
                            bounds = bounds, iprint=-1)
                start_params = self._transparams(mlefit[0])

        if transparams: # transform initial parameters to ensure invertibility
            start_params = self._invtransparams(start_params)

        if solver is None:  # use default limited memory bfgs
            bounds = [(None,)*2]*(k_ar+k_ma+k)
            mlefit = optimize.fmin_l_bfgs_b(loglike, start_params,
                    approx_grad=True, m=12, pgtol=1e-8, factr=1e2,
                    bounds=bounds, iprint=disp)
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

        normalized_cov_params = None #TODO: fix this
        armafit = ARMAResults(self, params, normalized_cov_params)
        self._results = armafit
        return ARMAResultsWrapper(armafit)

    fit.__doc__ += base.LikelihoodModel.fit.__doc__


class ARMAResults(tsbase.TimeSeriesModelResults):
    """
    Class to hold results from fitting an ARMA model.

    Parameters
    ----------
    model : ARMA instance
        The fitted model instance
    params : array
        Fitted parameters
    normalized_cov_params : array, optional
        The normalized variance covariance matrix
    scale : float, optional
        Optional argument to scale the variance covariance matrix.

    Returns
    --------
    **Attributes**

    aic : float
        Akaikie Information Criterion
        :math:`-2*llf+2*(df_model+1)`
    arparams : array
        The parameters associated with the AR coefficients in the model.
    arroots : array
        The roots of the AR coefficients are the solution to
        (1 - arparams[0]*z - arparams[1]*z**2 -...- arparams[p-1]*z**k_ar) = 0
        Stability requires that the roots in modulus lie outside the unit
        circle.
    bic : float
        Bayes Information Criterion
        -2*llf + log(nobs)*(df_model+1)
        Where if the model is fit using conditional sum of squares, the
        number of observations `nobs` does not include the `p` pre-sample
        observations.
    bse : array
        The standard errors of the parameters. These are computed using the
        numerical Hessian.
    df_model : array
        The model degrees of freedom = `k_exog` + `k_trend` + `k_ar` + `k_ma`
    df_resid : array
        The residual degrees of freedom = `nobs` - `df_model`
    fittedvalues : array
        The predicted values of the model.
    hqic : float
        Hannan-Quinn Information Criterion
        -2*llf + 2*(`df_model`)*log(log(nobs))
        Like `bic` if the model is fit using conditional sum of squares then
        the `k_ar` pre-sample observations are not counted in `nobs`.
    k_ar : int
        The number of AR coefficients in the model.
    k_exog : int
        The number of exogenous variables included in the model. Does not
        include the constant.
    k_ma : int
        The number of MA coefficients.
    k_trend : int
        This is 0 for no constant or 1 if a constant is included.
    llf : float
        The value of the log-likelihood function evaluated at `params`.
    maparams : array
        The value of the moving average coefficients.
    maroots : array
        The roots of the MA coefficients are the solution to
        (1 + maparams[0]*z + maparams[1]*z**2 + ... + maparams[q-1]*z**q) = 0
        Stability requires that the roots in modules lie outside the unit
        circle.
    model : ARMA instance
        A reference to the model that was fit.
    nobs : float
        The number of observations used to fit the model. If the model is fit
        using exact maximum likelihood this is equal to the total number of
        observations, `n_totobs`. If the model is fit using conditional
        maximum likelihood this is equal to `n_totobs` - `k_ar`.
    n_totobs : float
        The total number of observations for `endog`. This includes all
        observations, even pre-sample values if the model is fit using `css`.
    params : array
        The parameters of the model. The order of variables is the trend
        coefficients and the `k_exog` exognous coefficients, then the
        `k_ar` AR coefficients, and finally the `k_ma` MA coefficients.
    pvalues : array
        The p-values associated with the t-values of the coefficients. Note
        that the coefficients are assumed to have a Student's T distribution.
    resid : array
        The model residuals. If the model is fit using 'mle' then the
        residuals are created via the Kalman Filter. If the model is fit
        using 'css' then the residuals are obtained via `scipy.signal.lfilter`
        adjusted such that the first `k_ma` residuals are zero. These zero
        residuals are not returned.
    scale : float
        This is currently set to 1.0 and not used by the model or its results.
    sigma2 : float
        The variance of the residuals. If the model is fit by 'css',
        sigma2 = ssr/nobs, where ssr is the sum of squared residuals. If
        the model is fit by 'mle', then sigma2 = 1/nobs * sum(v**2 / F)
        where v is the one-step forecast error and F is the forecast error
        variance. See `nobs` for the difference in definitions depending on the
        fit.
    """
    _cache = {}

    #TODO: use this for docstring when we fix nobs issue


    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        super(ARMAResults, self).__init__(model, params, normalized_cov_params,
                scale)
        self.sigma2 = model.sigma2
        nobs = model.nobs
        self.nobs = nobs
        k_exog = model.k_exog
        self.k_exog = k_exog
        k_trend = model.k_trend
        self.k_trend = k_trend
        k_ar = model.k_ar
        self.k_ar = k_ar
        self.n_totobs = len(model.endog)
        k_ma = model.k_ma
        self.k_ma = k_ma
        df_model = k_exog + k_trend + k_ar + k_ma
        self.df_model = df_model
        self.df_resid = self.nobs - df_model
        self._cache = resettable_cache()

    @cache_readonly
    def arroots(self):
        return np.roots(np.r_[1,-self.arparams])**-1

    @cache_readonly
    def maroots(self):
        return np.roots(np.r_[1,self.maparams])**-1

    #@cache_readonly
    #def arfreq(self):
    #    return (np.log(arroots/abs(arroots))/(2j*pi)).real

    #NOTE: why don't root finding functions work well?
    #@cache_readonly
    #def mafreq(eslf):
    #    return


    @cache_readonly
    def arparams(self):
        k = self.k_exog + self.k_trend
        return self.params[k:k+self.k_ar]

    @cache_readonly
    def maparams(self):
        k = self.k_exog + self.k_trend
        k_ar = self.k_ar
        return self.params[k+k_ar:]

    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params)

    @cache_readonly
    def bse(self):
        params = self.params
        hess = self.model.hessian(params)
        if len(params) == 1: # can't take an inverse
            return np.sqrt(-1./hess)
        return np.sqrt(np.diag(-inv(hess)))

    def cov_params(self): # add scale argument?
        params = self.params
        hess = self.model.hessian(params)
        return -inv(hess)

    @cache_readonly
    def aic(self):
        return -2*self.llf + 2*(self.df_model+1)

    @cache_readonly
    def bic(self):
        nobs = self.nobs
        return -2*self.llf + np.log(nobs)*(self.df_model+1)

    @cache_readonly
    def hqic(self):
        nobs = self.nobs
        return -2*self.llf + 2*(self.df_model+1)*np.log(np.log(nobs))

    @cache_readonly
    def fittedvalues(self):
        model = self.model
        endog = model.endog.copy()
        k_ar = self.k_ar
        exog = model.exog # this is a copy
        if exog is not None:
            if model.method == "css" and k_ar > 0:
                exog = exog[k_ar:]
        if model.method == "css" and k_ar > 0:
            endog = endog[k_ar:]
        fv = endog - self.resid
        # add deterministic part back in
        k = self.k_exog + self.k_trend
    #TODO: this needs to be commented out for MLE with constant

    #    if k != 0:
    #        fv += dot(exog, self.params[:k])
        return fv

    @cache_readonly
    def resid(self):
        return self.model.geterrors(self.params)

    @cache_readonly
    def pvalues(self):
    #TODO: same for conditional and unconditional?
        df_resid = self.df_resid
        return t.sf(np.abs(self.tvalues), df_resid) * 2

    def forecast(self, steps=1, exog=None, alpha=.05):
        """
        Out-of-sample forecasts

        Parameters
        ----------
        steps : int
            The number of out of sample forecasts from the end of the
            sample.
        exog : array
            If the model is an ARMAX, you must provide out of sample
            values for the exogenous variables. This should not include
            the constant.
        alpha : float
            The confidence intervals for the forecasts are (1 - alpha) %

        Returns
        -------
        forecast : array
            Array of out of sample forecasts
        stderr : array
            Array of the standard error of the forecasts.
        conf_int : array
            2d array of the confidence interval for the forecast
        """
        k_trend, k_exog = self.k_trend, self.k_exog
        if exog is None and k_exog > 0:
            raise ValueError("You must provide exog for ARMAX")

        q, p = self.k_ma, self.k_ar
        maparams, arparams = self.maparams[::-1], self.arparams[::-1]

        if q:
            resid = np.zeros(2*q)
            resid[:q] = self.resid[-q:] #only need last q
        else:
            i = -1 # since we don't run first loop below

        y = self.model.endog
        if k_trend == 1:
            mu = self.params[0] * (1-arparams.sum()) # use expectation
                                                     # not constant
            mu = np.array([mu]*steps) # repeat it so you can slice if exog
        else:
            mu = np.zeros(steps)

        if k_exog > 0:
            mu += np.dot(self.params[k_trend:k_exog+k_trend], exog)


        endog = np.zeros(p+steps-1)
        if p:
            endog[:p] = y[-p:] #only need p

        forecast = np.zeros(steps)
        for i in range(q):
            fcast = mu[i] + np.dot(arparams,endog[i:i+p]) + \
                          np.dot(maparams,resid[i:i+q])
            forecast[i] = fcast
            endog[i+p] = fcast

        for i in range(i+1,steps-1):
            fcast = mu[i] + np.dot(arparams,endog[i:i+p])
            forecast[i] = fcast
            endog[i+p] = fcast

        #need to do one more without updating endog
        forecast[-1] = mu[-1] + np.dot(arparams,endog[steps-1:])

        #take care of exogenous

        # compute the standard errors
        sigma2 = self.sigma2
        ma_rep = arma2ma(np.r_[1,-arparams[::-1]],
                         np.r_[1, maparams[::-1]], nobs=steps)


        fcasterr = np.sqrt(sigma2 * np.cumsum(ma_rep**2))

        const = norm.ppf(1 - alpha/2.)
        conf_int = np.c_[forecast - const*fcasterr, forecast + const*fcasterr]

        return forecast, fcasterr, conf_int

class ARMAResultsWrapper(wrap.ResultsWrapper):
    _attrs = {}
    _wrap_attrs = wrap.union_dicts(tsbase.TimeSeriesResultsWrapper._wrap_attrs,
                                    _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(
                        tsbase.TimeSeriesResultsWrapper._wrap_methods,
                        _methods)
wrap.populate_wrapper(ARMAResultsWrapper, ARMAResults)

if __name__ == "__main__":
    import numpy as np
    import scikits.statsmodels.api as sm

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
