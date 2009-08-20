import numpy as np
from scipy.stats import t, norm
from scipy import optimize, derivative
from models.contrast import ContrastResults
from models.tools import recipr

import numpy.lib.recfunctions as nprf

class Model(object):
    """
    A (predictive) statistical model. The class Model itself does nothing
    but lays out the methods expected of any subclass.
    """

    _results = None

    def __init__(self, endog, exog=None):
        self._endog = np.asarray(endog)
        self._exog = np.asarray(exog)
        self.nobs = float(endog.shape[0])

    def fit(self):
        """
        Fit a model to data.
        """
        raise NotImplementedError

#FIXME:  Why not have it as a model method if we're going to put it here anyway?
#        Putting it in results gives infinite recursion with results property
    def predict(self, design):
        """
        After a model has been fit, results are (assumed to be) stored
        in self.results, which itself should have a predict method.
        """
        raise NotImplementedErrror
#        return self.results.predict(design)

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

    def loglike(self, params):
        """
        Log-likelihood of model.
        """
        raise NotImplementedError

    def score(self, params):
        """
        Score function of model.

        The gradient of logL with respect to params.
        """
        return derivative(self.loglike, params, dx=1e-04, n=1, order=3)

    def information(self, params):
        """
        Fisher information function of model = - Hessian of logL with respect
        to params.
        """
        raise NotImplementedError

#TODO: newton's method is not correctly implemented yet
    def fit(self, params, method='newton'):
        if method is 'newton':
            results = self.newton(params)
        else:
            raise ValueError("Unknown fit method.")
        self._results = results

#FIXME: This does not work as implemented
#FIXME: Params should be a first guess on the params
#       so supplied or default guess?
    def newton(self, params):
        #JP this is not newton, it's fmin
# it also doesn't work
# tried fmin, fmin_ncg, fmin_powell
# converges to wrong estimates
# will probably need to write own root-finding routine
#FIXME: should we call it MLE for Maximum Likelihood Estimator?
# SS no this isn't used anywhere right now, but needs attention for
# MLE
        # is this used anywhere
        # results should be attached to self
        f = lambda params: -self.loglike(params)
        score = lambda params: -self.score(params)
#        xopt, fopt, iter, funcalls, warnflag =\
#          optimize.fmin(f, params, full_output=True)
#        xopt, fopt, fcalls, gcalls, hcalls, warnflag = \
#                optimize.fmin_ncg(f, params, score)
        converge = not warnflag
        extras = dict(iter=iter, evaluations=funcalls, converge=converge)
#        return LikelihoodModelResults(self, params, llf=fopt, **extras)
        return LikelihoodModelResults(self, xopt)


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

        Comments
        --------

        The covariance of params is given by scale times
        normalized_cov_params
        """

        #JP what are the minimum attributes and methods that are used of model
        # i.e. can this be called with an almost empty model
        # what's the smallest duck that quacks like a model - list
# SS the minimum is data.endog, data.exog (except for AR)
# some models take more kwd arguments
# the minimum needed for results are the inits here
        super(LikelihoodModelResults, self).__init__(model, params)
        self.normalized_cov_params = normalized_cov_params
        self.scale = scale

    def normalized_cov_params(self):
        raise NotImplementedError

#    def scale(self):    #JP very bad, first scale is an attribute, now a method
#        raise NotImplementedError
                        # SS It's not used I don't think and can be removed

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
        >>>from models.datasets.longley.data import load
        >>>from models.tools import add_constant
        >>>import models
        >>>data = load()
        >>>data.exog = add_constant(data.exog)
        >>>results = models.OLS(data.endog, data.exog).fit()
        >>>results.t()
        array([ 0.17737603, -1.06951632, -4.13642736, -4.82198531, -0.22605114,
        4.01588981, -3.91080292])
        >>>results.t([1,2,4])
        array([-1.06951632, -4.13642736, -0.22605114])
        >>>import numpy as np
        >>>results.t(np.array([1,2,4]))
        array([-1.06951632, -4.13642736, -0.22605114])
        """

        if self.normalized_cov_params is None:
            raise ValueError, 'need covariance of parameters for computing T statistics'

        if column is None:
            column = range(self.params.shape[0])

        column = np.asarray(column)
        _params = self.params[column]
        _cov = self.cov_params(column=column)
        if _cov.ndim == 2:
            _cov = np.diag(_cov)
        _t = _params * recipr(np.sqrt(_cov))
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
#            if np.shape(scale) == ():   # can be a scalar or array
# TODO: Algorithm for HCCM
# the first return is needed if scale is not a scalar
#                scale=np.eye(len(self._model._endog))*scale
#            return np.dot(np.dot(self.calc_params, np.array(scale)),
#                self.calc_params.T)
            return self.normalized_cov_params * scale

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
        -------
        >>>import numpy as np
        >>>import models
        >>>from models.datasets.longley.data import load
        >>>from models.tools import add_constant
        >>>data = load()
        >>>data.exog = add_constant(data.exog)
        >>>results = models.OLS(data.endog, data.exog).fit()
        >>>r = np.zeros(len(results.params))
        >>>r[4:6] = [1,-1]
        >>>print r
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
        `t` method to get simpler t values
        `f_test` method for f tests
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
        _t = _effect * recipr(_sd)
        return ContrastResults(effect=_effect, t=_t, sd=_sd,
                df_denom=self.model.df_resid)

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
            If q is 1 then f_test(r_matrix).fvalue is equivalent to the square of
            t_test(r_matrix).t
        scale : float, optional
            Default is 1.0 for no scaling.
        invcov : array-like, optional
            A qxq matrix to specify an inverse covariance
            matrix based on a restrictions matrix.

        Examples
        ---------
        >>>import numpy as np
        >>>import models
        >>>from models.datasets.longley.data import load
        >>>data = load()
        >>>data.exog = models.tools.add_constant(data.exog)
        >>>results = models.OLS(data.endog, data.exog).fit()
        >>>A = np.identity(len(results.params))
        >>>A = A[:-1,:]

        This tests that each coefficient is jointly statistically
        significantly different from zero.

        >>>print results.f_test(A)
        <F contrast: F=330.28533923463488, p=4.98403052872e-10, df_denom=9, df_num=6>

        Compare this to
        >>>results.F
        330.2853392346658
        >>>results.F_p
        4.98403096572e-10

        >>>B = np.array(([0,1,-1,0,0,0,0],[0,0,0,0,1,-1,0]))

        This tests that the coefficient on the 2nd and 3rd regressors are
        equal and jointly that the coefficient on the 5th and 6th regressors
        are equal.

        >>>print results.f_test(B)
        <F contrast: F=9.740461873303655, p=0.00560528853174, df_denom=9, df_num=2>

        See also
        --------
        models.contrasts
        models.model.t_test

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

        Example
        -------
        >>>import models
        >>>from models.datasets.longley.data import load
        >>>data = load()
        >>>data.exog = models.tools.add_constant(data.exog)
        >>>results = models.OLS(data.endog, data.exog).fit()
        >>>results.conf_int()
        array([[ -1.77029035e+02,   2.07152780e+02],
       [ -1.11581102e-01,   3.99427438e-02],
       [ -3.12506664e+00,  -9.15392966e-01],
       [ -1.51794870e+00,  -5.48505034e-01],
       [ -5.62517214e-01,   4.60309003e-01],
       [  7.98787515e+02,   2.85951541e+03],
       [ -5.49652948e+06,  -1.46798779e+06]])

        >>>results.conf_int(cols=(1,2))
        array([[-0.1115811 ,  0.03994274],
       [-3.12506664, -0.91539297]])

        Notes
        -----
        The confidence interval is based on Student's t distribution for all
        models except RLM and GLM, which uses the standard normal distribution.
        """
        if self.__class__.__name__ in ['RLMResults','GLMResults']:
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




