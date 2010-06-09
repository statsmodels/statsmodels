import numpy as np
from scipy.stats import t, norm
from scipy import optimize, derivative
from tools import recipr
from contrast import ContrastResults

class Model(object):
    """
    A (predictive) statistical model. The class Model itself is not to be used.

    Model lays out the methods expected of any subclass.

    Parameters
    ----------
    endog : array-like
        Endogenous response variable.
    exog : array-like
        Exogenous design.

    Methods
    -------
    fit
        Call a models fit method
    predict
        Return fitted response values for a model.  If the model has

    Notes
    -----
    `endog` and `exog` are references to any data provided.  So if the data is
    already stored in numpy arrays and it is changed then `endog` and `exog`
    will change as well.
    """

    _results = None

    def __init__(self, endog, exog=None):
        endog = np.asarray(endog)
        endog = np.squeeze(endog) # for consistent outputs if endog is (n,1)

##        # not sure if we want type conversion, needs tests with integers
##        if np.issubdtype(endog.dtype, int):
##            endog = endog.astype(float)
##        if np.issubdtype(exog.dtype, int):
##            endog = exog.astype(float)
        if not exog is None:
            exog = np.asarray(exog)
            if exog.ndim == 1:
                exog = exog[:,None]
            if exog.ndim != 2:
                raise ValueError, "exog is not 1d or 2d"
            if endog.shape[0] != exog.shape[0]:
                raise ValueError, "endog and exog matrices are not aligned."
            if np.any(exog.var(0) == 0):
                # assumes one constant in first or last position
                const_idx = np.where(exog.var(0) == 0)[0].item()
                if const_idx == exog.shape[1] - 1:
                    exog_names = ['x%d' % i for i in range(1,exog.shape[1])]
                    exog_names += ['const']
                else:
                    exog_names = ['x%d' % i for i in range(exog.shape[1])]
                    exog_names[const_idx] = 'const'
                self.exog_names = exog_names
            self.endog_names = ['y']
        self.endog = endog
        self.exog = exog
        self.nobs = float(self.endog.shape[0])

    def fit(self):
        """
        Fit a model to data.
        """
        raise NotImplementedError

    def predict(self, design):
        """
        After a model has been fit predict returns the fitted values.  If
        the model has not been fit, then fit is called.
        """
        raise NotImplementedError

class LikelihoodModel(Model):
    """
    Likelihood model is a subclass of Model.
    """

    def __init__(self, endog, exog=None):
        super(LikelihoodModel, self).__init__(endog, exog)
        self.initialize()

    def initialize(self):
        """
        Initialize (possibly re-initialize) a Model instance. For
        instance, the design matrix of a linear model may change
        and some things must be recomputed.
        """
        pass
#TODO: if the intent is to re-initialize the model with new data then
# this method needs to take inputs...

    def loglike(self, params):
        """
        Log-likelihood of model.
        """
        raise NotImplementedError

    def score(self, params):
        """
        Score vector of model.

        The gradient of logL with respect to each parameter.
        """
        raise NotImplementedError

    def information(self, params):
        """
        Fisher information matrix of model

        Returns -Hessian of loglike evaluated at params.
        """
        raise NotImplementedError

    def hessian(self, params):
        """
        The Hessian matrix of the model
        """
        raise NotImplementedError

    def fit(self, start_params=None, method='newton', maxiter=35, full_output=1,
            disp=1, fargs=(), callback=None, **kwargs):
        """
        Fit method for likelihood based models

        Parameters
        ----------
        start_params : array-like, optional
            An optional
        method : str
            Method can be 'newton', 'bfgs', 'powell', 'cg', or 'ncg'.
            The default is newton.  See scipy.optimze for more information.
        """
        methods = ['newton', 'nm', 'bfgs', 'powell', 'cg', 'ncg']
        if start_params is None:
            start_params = [0]*self.exog.shape[1] # will fail for shape (K,)
        if method.lower() not in methods:
            raise ValueError, "Unknown fit method %s" % method
        method = method.lower()
        f = lambda params: -self.loglike(params)
        score = lambda params: -self.score(params)
        try:
            hess = lambda params: -self.hessian(params)
        except:
            hess = None
        if method == 'newton':
            tol = kwargs.get('tol', 1e-8)
            score = lambda params: self.score(params) # these are positive?
            hess = lambda params: self.hessian(params) # but neg for others?
            iterations = 0
            start = np.array(start_params)
#TODO: do we want to keep a history?
            history = [np.inf, start]
            while (iterations < maxiter and np.all(np.abs(history[-1] - \
                    history[-2])>tol)):
                H = hess(history[-1])
#                H = self.hessian(history[-1])
                newparams = history[-1] - np.dot(np.linalg.inv(H),
                        score(history[-1]))
                history.append(newparams)
                if callback is not None:
                    callback(newparams)
                iterations += 1
            fval = f(newparams, *fargs) # this is the negative likelihood
            if iterations == maxiter:
                warnflag = 1
                if disp:
                    print "Warning: Maximum number of iterations has been \
exceeded."
                    print "         Current function value: %f" % fval
                    print "         Iterations: %d" % iterations
            else:
                warnflag = 0
                if disp:
                    print "Optimization terminated successfully."
                    print "         Current function value: %f" % fval
                    print "         Iterations %d" % iterations
            if full_output:
                xopt, fopt, niter, gopt, hopt = (newparams, f(newparams),
                    iterations, score(newparams), hess(newparams))
                converged = not warnflag
                retvals = {'fopt' : fopt, 'iterations' : niter, 'score' : gopt,
                        'Hessian' : hopt, 'warnflag' : warnflag,
                        'converged' : converged}
            else:
                retvals = newparams
        elif method == 'nm':    # Nelder-Mead
            xtol = kwargs.get('xtol', 0.0001)
            ftol = kwargs.get('ftol', 0.0001)
            maxfun = kwargs.get('maxfun', None)
            retvals = optimize.fmin(f, start_params, args=fargs, xtol=xtol,
                        ftol=ftol, maxiter=maxiter, maxfun=maxfun,
                        full_output=full_output, disp=disp, retall=0,
                        callback=callback)
            if full_output:
                xopt, fopt, niter, fcalls, warnflag = retvals
                converged = not warnflag
                retvals = {'fopt' : fopt, 'iterations' : niter,
                    'fcalls' : fcalls, 'warnflag' : warnflag,
                    'converged' : converged}
        elif method == 'bfgs':
            gtol = kwargs.get('gtol', 1.0000000000000001e-05)
            norm = kwargs.get('norm', np.Inf)
            epsilon = kwargs.get('epsilon', 1.4901161193847656e-08)
            retvals = optimize.fmin_bfgs(f, start_params, score, args=fargs,
                            gtol=gtol, norm=norm, epsilon=epsilon,
                            maxiter=maxiter, full_output=full_output,
                            disp=disp, retall=0, callback=callback)
            if full_output:
                xopt, fopt, gopt, Hinv, fcalls, gcalls, warnflag = retvals
                converged = not warnflag
                retvals = {'fopt' : fopt, 'gopt' : gopt, 'Hinv' : Hinv,
                        'fcalls' : fcalls, 'gcalls' : gcalls, 'warnflag' :
                        warnflag, 'converged' : converged}
        elif method == 'ncg':
            fhess_p = kwargs.get('fhess_p', None)
            avextol = kwargs.get('avextol', 1.0000000000000001e-05)
            epsilon = kwargs.get('epsilon', 1.4901161193847656e-08)
            retvals = optimize.fmin_ncg(f, start_params, score, fhess_p=fhess_p,
                            fhess=hess, args=fargs, avextol=avextol,
                            epsilon=epsilon, maxiter=maxiter,
                            full_output=full_output, disp=disp, retall=0,
                            callback=callback)
            if full_output:
                xopt, fopt, fcalls, gcalls, hcalls, warnflag = retvals
                converged = not warnflag
                retvals = {'fopt' : fopt, 'fcalls' : fcalls, 'gcalls' : gcalls,
                    'hcalls' : hcalls, 'warnflag' : warnflag,
                    'converged' : converged}
        elif method == 'cg':
            gtol = kwargs.get('gtol', 1.0000000000000001e-05)
            norm = kwargs.get('norm', np.Inf)
            epsilon = kwargs.get('epsilon', 1.4901161193847656e-08)
            retvals = optimize.fmin_cg(f, start_params, score,
                            gtol=gtol, norm=norm,
                            epsilon=epsilon, maxiter=maxiter,
                            full_output=full_output, disp=disp, retall=0,
                            callback=callback)
            if full_output:
                xopt, fopt, fcalls, gcalls, warnflag = retvals
                converged = not warnflag
                retvals = {'fopt' : fopt, 'fcalls' : fcalls, 'gcalls' : gcalls,
                    'warnflag' : warnflag, 'converged' : converged}
        elif method == 'powell':
            xtol = kwargs.get('xtol', 0.0001)
            ftol = kwargs.get('ftol', 0.0001)
            maxfun = kwargs.get('maxfun', None)
            start_direc = kwargs.get('start_direc', None)
            retvals = optimize.fmin_powell(f, start_params, args=fargs,
                            xtol=xtol, ftol=ftol, maxiter=maxiter,
                            maxfun=maxfun, full_output=full_output, disp=disp,
                            retall=0, callback=callback, direc=start_direc)
            if full_output:
                xopt, fopt, direc, niter, fcalls, warnflag = retvals
                converged = not warnflag
                retvals = {'fopt' : fopt, 'direc' : direc, 'iterations' : niter,
                    'fcalls' : fcalls, 'warnflag' : warnflag}
        if not full_output:
            xopt = retvals

        if method == 'bfgs' and full_output:
            Hinv = retvals.get('Hinv', 0)
        elif method == 'newton' and full_output:
            Hinv = np.linalg.inv(-hopt)
        else:
            try:
                Hinv = np.linalg.inv(-1*self.hessian(xopt))
            except:
                Hinv = None
#TODO: add Hessian approximation and change the above if needed
        mlefit = LikelihoodModelResults(self, xopt, Hinv, scale=1.)
#TODO: hardcode scale?

        if isinstance(retvals, dict):
            mlefit.mle_retvals = retvals
        optim_settings = {'optimizer' : method, 'start_params' : start_params,
            'maxiter' : maxiter, 'full_output' : full_output, 'disp' : disp,
            'fargs' : fargs, 'callback' : callback}
        optim_settings.update(kwargs)
        mlefit.mle_settings = optim_settings
        self._results = mlefit
        return mlefit

class Results(object):
    """
    Class to contain model results
    """
    def __init__(self, model, params, **kwd):
        """
        Parameters
        ----------
        model : class instance
            the previously specified model instance
        params : array
            parameter estimates from the fit model
        """
        self.__dict__.update(kwd)
        self.initialize(model, params, **kwd)

    def initialize(self, model, params, **kwd):
        self.params = params
        self.model = model
#TODO: public method?

class LikelihoodModelResults(Results):
    """ Class to contain results from likelihood models """
    def __init__(self, model, params, normalized_cov_params=None, scale=1.):
        """
        Parameters
        -----------
        params : 1d array_like
            parameter estimates from estimated model
        normalized_cov_params : 2d array
           Normalized (before scaling) covariance of params
            normalized_cov_paramss is also known as the hat matrix or H
            (Semiparametric regression, Ruppert, Wand, Carroll; CUP 2003)
        scale : float
            For (some subset of models) scale will typically be the
            mean square error from the estimated model (sigma^2)

        Notes
        --------
        The covariance of params is given by scale times
        normalized_cov_params
        """
        super(LikelihoodModelResults, self).__init__(model, params)
        self.normalized_cov_params = normalized_cov_params
        self.scale = scale

    def normalized_cov_params(self):
        raise NotImplementedError

    def t(self, column=None):
        """
        Return the t-statistic for a given parameter estimate.

        Parameters
        ----------
        column : array-like
            The columns for which you would like the t-value.
            Note that this uses Python's indexing conventions.

        See also
        ---------
        Use t_test for more complicated t-statistics.

        Examples
        --------
        >>> import scikits.statsmodels as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> results.t()
        array([ 0.17737603, -1.06951632, -4.13642736, -4.82198531, -0.22605114,
        4.01588981, -3.91080292])
        >>> results.t([1,2,4])
        array([-1.06951632, -4.13642736, -0.22605114])
        >>> import numpy as np
        >>> results.t(np.array([1,2,4]))
        array([-1.06951632, -4.13642736, -0.22605114])

        """

        if self.normalized_cov_params is None:
            raise ValueError, 'need covariance of parameters for computing T\
 statistics'

        if column is None:
            column = range(self.params.shape[0])

        column = np.asarray(column)
        _params = self.params[column]
        _cov = self.cov_params(column=column)
        if _cov.ndim == 2:
            _cov = np.diag(_cov)
#        _t = _params * recipr(np.sqrt(_cov))
# repicr drops precision for MNLogit?
        _t = _params / np.sqrt(_cov)
        return _t


    def cov_params(self, r_matrix=None, column=None, scale=None, other=None):
        """
        Returns the variance/covariance matrix.

        The variance/covariance matrix can be of a linear contrast
        of the estimates of params or all params multiplied by scale which
        will usually be an estimate of sigma^2.  Scale is assumed to be
        a scalar.

        Parameters
        -----------
        r_matrix : array-like
            Can be 1d, or 2d.  Can be used alone or with other.
        column :  array-like, optional
            Must be used on its own.  Can be 0d or 1d see below.
        scale : float, optional
            Can be specified or not.  Default is None, which means that
            the scale argument is taken from the model.
        other : array-like, optional
            Can be used when r_matrix is specified.

        Returns
        -------
        (The below are assumed to be in matrix notation.)

        cov : ndarray

        If no argument is specified returns the covariance matrix of a model
        (scale)*(X.T X)^(-1)

        If contrast is specified it pre and post-multiplies as follows
        (scale) * r_matrix (X.T X)^(-1) r_matrix.T

        If contrast and other are specified returns
        (scale) * r_matrix (X.T X)^(-1) other.T

        If column is specified returns
        (scale) * (X.T X)^(-1)[column,column] if column is 0d

        OR

        (scale) * (X.T X)^(-1)[column][:,column] if column is 1d

        """
        if self.normalized_cov_params is None:
            raise ValueError, 'need covariance of parameters for computing \
(unnormalized) covariances'
        if column is not None and (r_matrix is not None or other is not None):
            raise ValueError, 'Column should be specified without other \
arguments.'
        if other is not None and r_matrix is None:
            raise ValueError, 'other can only be specified with r_matrix'
        if scale is None:
            scale = self.scale
        if column is not None:
            column = np.asarray(column)
            if column.shape == ():
                return self.normalized_cov_params[column, column] * scale
            else:
                return self.normalized_cov_params[column][:,column] * scale
        elif r_matrix is not None:
            r_matrix = np.asarray(r_matrix)
            if r_matrix.shape == ():
                raise ValueError, "r_matrix should be 1d or 2d"
            if other is None:
                other = r_matrix
            else:
                other = np.asarray(other)
            tmp = np.dot(r_matrix, np.dot(self.normalized_cov_params,
                np.transpose(other)))
            return tmp * scale
        if r_matrix is None and column is None:
            return self.normalized_cov_params * scale

#TODO: make sure this works as needed for GLMs
    def t_test(self, r_matrix, scale=None):
        """
        Compute a tcontrast/t-test for a row vector array.

        Parameters
        ----------
        r_matrix : array-like
            A length p row vector specifying the linear restrictions.
        scale : float, optional
            An optional `scale` to use.  Default is the scale specified
            by the model fit.

        scale : scalar

        Examples
        --------
        >>> import numpy as np
        >>> import scikits.statsmodels as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> r = np.zeros_like(results.params)
        >>> r[4:6] = [1,-1]
        >>> print r
        [ 0.  0.  0.  0.  1. -1.  0.]

        r tests that the coefficients on the 5th and 6th independent
        variable are the same.

        >>>T_Test = results.t_test(r)
        >>>print T_test
        <T contrast: effect=-1829.2025687192481, sd=455.39079425193762, t=-4.0167754636411717, p=0.0015163772380899498, df_denom=9>
        >>> T_test.effect
        -1829.2025687192481
        >>> T_test.sd
        455.39079425193762
        >>> T_test.t
        -4.0167754636411717
        >>> T_test.p
        0.0015163772380899498

        See also
        ---------
        t : method to get simpler t values
        f_test : for f tests

        """
        r_matrix = np.squeeze(np.asarray(r_matrix))

        if self.normalized_cov_params is None:
            raise ValueError, 'Need covariance of parameters for computing \
T statistics'
        if r_matrix.ndim == 1:
            if r_matrix.shape[0] != self.params.shape[0]:
                raise ValueError, 'r_matrix and params are not aligned'
        elif r_matrix.ndim >1:
            if r_matrix.shape[1] != self.params.shape[0]:
                raise ValueError, 'r_matrix and params are not aligned'

        _t = _sd = None

        _effect = np.dot(r_matrix, self.params)
        _sd = np.sqrt(self.cov_params(r_matrix=r_matrix))
        if _sd.ndim > 1:
            _sd = np.diag(_sd)
        _t = _effect * recipr(_sd)
        return ContrastResults(effect=_effect, t=_t, sd=_sd,
                df_denom=self.model.df_resid)

#TODO: untested for GLMs?
    def f_test(self, r_matrix, scale=1.0, invcov=None):
        """
        Compute an Fcontrast/F-test for a contrast matrix.

        Here, matrix `r_matrix` is assumed to be non-singular. More precisely,

        r_matrix (pX pX.T) r_matrix.T

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.

        Parameters
        -----------
        r_matrix : array-like
            q x p array where q is the number of restrictions to test and
            p is the number of regressors in the full model fit.
            If q is 1 then f_test(r_matrix).fvalue is equivalent to
            the square of t_test(r_matrix).t
        scale : float, optional
            Default is 1.0 for no scaling.
        invcov : array-like, optional
            A qxq matrix to specify an inverse covariance
            matrix based on a restrictions matrix.

        Examples
        --------
        >>> import numpy as np
        >>> import scikits.statsmodels as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> A = np.identity(len(results.params))
        >>> A = A[:-1,:]

        This tests that each coefficient is jointly statistically
        significantly different from zero.

        >>> print results.f_test(A)
        <F contrast: F=330.28533923463488, p=4.98403052872e-10, df_denom=9, df_num=6>

        Compare this to

        >>> results.F
        330.2853392346658
        >>> results.F_p
        4.98403096572e-10

        >>> B = np.array(([0,1,-1,0,0,0,0],[0,0,0,0,1,-1,0]))

        This tests that the coefficient on the 2nd and 3rd regressors are
        equal and jointly that the coefficient on the 5th and 6th regressors
        are equal.

        >>> print results.f_test(B)
        <F contrast: F=9.740461873303655, p=0.00560528853174, df_denom=9, df_num=2>

        See also
        --------
        scikits.statsmodels.contrasts
        scikits.statsmodels.model.t_test

        """
        r_matrix = np.asarray(r_matrix)
        r_matrix = np.atleast_2d(r_matrix)

        if self.normalized_cov_params is None:
            raise ValueError, 'need covariance of parameters for computing F statistics'

        cparams = np.dot(r_matrix, self.params)

        q = r_matrix.shape[0]
        if invcov is None:
            invcov = np.linalg.inv(self.cov_params(r_matrix=r_matrix,
                scale=scale))
        F = np.add.reduce(np.dot(invcov, cparams) * cparams, 0) * \
                recipr((q * self.scale))
        return ContrastResults(F=F, df_denom=self.model.df_resid,
                    df_num=invcov.shape[0])

    def conf_int(self, alpha=.05, cols=None):
        """
        Returns the confidence interval of the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
            The `alpha` level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.
        cols : array-like, optional
            `cols` specifies which confidence intervals to return

        Returns
        --------
        conf_int : array
            Each row contains [lower, upper] confidence interval

        Examples
        --------
        >>> import scikits.statsmodels as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> results.conf_int()
        array([[ -1.77029035e+02,   2.07152780e+02],
        [ -1.11581102e-01,   3.99427438e-02],
        [ -3.12506664e+00,  -9.15392966e-01],
        [ -1.51794870e+00,  -5.48505034e-01],
        [ -5.62517214e-01,   4.60309003e-01],
        [  7.98787515e+02,   2.85951541e+03],
        [ -5.49652948e+06,  -1.46798779e+06]])

        >>> results.conf_int(cols=(1,2))
        array([[-0.1115811 ,  0.03994274],
        [-3.12506664, -0.91539297]])

        Notes
        -----
        The confidence interval is based on Student's t distribution for all
        models except RLM and GLM, which uses the standard normal distribution.

        """
        #TODO: simplify structure, DRY
        if self.__class__.__name__ in ['RLMResults','GLMResults','DiscreteResults']:
            dist = norm
        else:
            dist = t
        if cols is None and dist == t:
            lower = self.params - dist.ppf(1-alpha/2,self.model.df_resid) *\
                    self.bse
            upper = self.params + dist.ppf(1-alpha/2,self.model.df_resid) *\
                    self.bse
        elif cols is None and dist == norm:
            lower = self.params - dist.ppf(1-alpha/2)*self.bse
            upper = self.params + dist.ppf(1-alpha/2)*self.bse
        elif cols is not None and dist == t:
            cols = np.asarray(cols)
            lower = self.params[cols] - dist.ppf(1-\
                        alpha/2,self.model.df_resid) *self.bse[cols]
            upper = self.params[cols] + dist.ppf(1-\
                        alpha/2,self.model.df_resid) *self.bse[cols]
        elif cols is not None and dist == norm:
            cols = np.asarray(cols)
            lower = self.params[cols] - dist.ppf(1-alpha/2)*self.bse[cols]
            upper = self.params[cols] + dist.ppf(1-alpha/2)*self.bse[cols]
        return np.asarray(zip(lower,upper))




