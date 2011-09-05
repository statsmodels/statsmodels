"""
Generalized linear models currently supports estimation using the one-parameter
exponential families

References
----------
Gill, Jeff. 2000. Generalized Linear Models: A Unified Approach.
    SAGE QASS Series.

Green, PJ. 1984.  "Iteratively reweighted least squares for maximum
    likelihood estimation, and some robust and resistant alternatives."
    Journal of the Royal Statistical Society, Series B, 46, 149-192.

Hardin, J.W. and Hilbe, J.M. 2007.  "Generalized Linear Models and
    Extensions."  2nd ed.  Stata Press, College Station, TX.

McCullagh, P. and Nelder, J.A.  1989.  "Generalized Linear Models." 2nd ed.
    Chapman & Hall, Boca Rotan.
"""

import numpy as np
import families
from scikits.statsmodels.tools.tools import rank
from scikits.statsmodels.regression.linear_model import WLS
from scikits.statsmodels.base.model import (LikelihoodModel,
        LikelihoodModelResults)
from scikits.statsmodels.tools.decorators import (cache_readonly,
        resettable_cache)

from scipy.stats import t

__all__ = ['GLM']

class GLM(LikelihoodModel):
    '''
    Generalized Linear Models class

    GLM inherits from statsmodels.LikelihoodModel

    Parameters
    -----------
    endog : array-like
        1d array of endogenous response variable.  This array can be
        1d or 2d.  Binomial family models accept a 2d array with two columns.
        If supplied, each observation is expected to be [success, failure].
    exog : array-like
        n x p design / exogenous data array
    family : family class instance
        The default is Gaussian.  To specify the binomial distribution
        family = sm.family.Binomial()
        Each family can take a link instance as an argument.  See
        statsmodels.family.family for more information.


    Attributes
    -----------
    df_model : float
        `p` - 1, where `p` is the number of regressors including the intercept.
    df_resid : float
        The number of observation `n` minus the number of regressors `p`.
    endog : array
        See Parameters.
    exog : array
        See Parameters.
    history : dict
        Contains information about the iterations.
    iteration : int
        The number of iterations that fit has run.  Initialized at 0.
    family : family class instance
        A pointer to the distribution family of the model.
    mu : array
        The estimated mean response of the transformed variable.
    normalized_cov_params : array
        `p` x `p` normalized covariance of the design / exogenous data.
    pinv_wexog : array
        For GLM this is just the pseudo inverse of the original design.
    scale : float
        The estimate of the scale / dispersion.  Available after fit is called.
    scaletype : str
        The scaling used for fitting the model.  Available after fit is called.
    weights : array
        The value of the weights after the last iteration of fit.

    Examples
    --------
    >>> import scikits.statsmodels.api as sm
    >>> data = sm.datasets.scotland.load()
    >>> data.exog = sm.add_constant(data.exog)

    Instantiate a gamma family model with the default link function.

    >>> gamma_model = sm.GLM(data.endog, data.exog,        \
                             family=sm.families.Gamma())

    >>> gamma_results = gamma_model.fit()
    >>> gamma_results.params
    array([  4.96176830e-05,   2.03442259e-03,  -7.18142874e-05,
         1.11852013e-04,  -1.46751504e-07,  -5.18683112e-04,
        -2.42717498e-06,  -1.77652703e-02])
    >>> gamma_results.scale
    0.0035842831734919055
    >>> gamma_results.deviance
    0.087388516416999198
    >>> gamma_results.pearson_chi2
    0.086022796163805704
    >>> gamma_results.llf
    -83.017202161073527

    See also
    --------
    statsmodels.families.*

    Notes
    -----
    Only the following combinations make sense for family and link ::

                   + ident log logit probit cloglog pow opow nbinom loglog logc
      Gaussian     |   x    x                        x
      inv Gaussian |   x    x                        x
      binomial     |   x    x    x     x       x     x    x           x      x
      Poission     |   x    x                        x
      neg binomial |   x    x                        x          x
      gamma        |   x    x                        x

    Not all of these link functions are currently available.

    Endog and exog are references so that if the data they refer to are already
    arrays and these arrays are changed, endog and exog will change.


    **Attributes**

    df_model : float
        Model degrees of freedom is equal to p - 1, where p is the number
        of regressors.  Note that the intercept is not reported as a
        degree of freedom.
    df_resid : float
        Residual degrees of freedom is equal to the number of observation n
        minus the number of regressors p.
    endog : array
        See above.  Note that endog is a reference to the data so that if
        data is already an array and it is changed, then `endog` changes
        as well.
    exposure : array-like
        Include ln(exposure) in model with coefficient constrained to 1.
    exog : array
        See above.  Note that endog is a reference to the data so that if
        data is already an array and it is changed, then `endog` changes
        as well.
    history : dict
        Contains information about the iterations. Its keys are `fittedvalues`,
        `deviance`, and `params`.
    iteration : int
        The number of iterations that fit has run.  Initialized at 0.
    family : family class instance
        The distribution family of the model. Can be any family in
        scikits.statsmodels.families.  Default is Gaussian.
    mu : array
        The mean response of the transformed variable.  `mu` is the value of
        the inverse of the link function at eta, where eta is the linear
        predicted value of the WLS fit of the transformed variable.  `mu` is
        only available after fit is called.  See
        statsmodels.families.family.fitted of the distribution family for more
        information.
    normalized_cov_params : array
        The p x p normalized covariance of the design / exogenous data.
        This is approximately equal to (X.T X)^(-1)
    offset : array-like
        Include offset in model with coefficient constrained to 1.
    pinv_wexog : array
        The pseudoinverse of the design / exogenous data array.  Note that
        GLM has no whiten method, so this is just the pseudo inverse of the
        design.
        The pseudoinverse is approximately equal to (X.T X)^(-1)X.T
    scale : float
        The estimate of the scale / dispersion of the model fit.  Only
        available after fit is called.  See GLM.fit and GLM.estimate_scale
        for more information.
    scaletype : str
        The scaling used for fitting the model.  This is only available after
        fit is called.  The default is None.  See GLM.fit for more information.
    weights : array
        The value of the weights after the last iteration of fit.  Only
        available after fit is called.  See statsmodels.families.family for
        the specific distribution weighting functions.

    '''

    def __init__(self, endog, exog, family=None, offset=None, exposure=None):
        endog = np.asarray(endog)
        exog = np.asarray(exog)
        if endog.shape[0] != len(exog):
            msg = "Size of endog (%s) does not match the shape of exog (%s)"
            raise ValueError(msg % (endog.size, len(exog)))
        if family is None:
            family = families.Gaussian()
        if offset is not None:
            offset = np.asarray(offset)
            if offset.shape[0] != endog.shape[0]:
                raise ValueError("offset is not the same length as endog")
            self.offset = offset
        if exposure is not None:
            exposure = np.log(exposure)
            if exposure.shape[0] != endog.shape[0]:
                raise ValueError("exposure is not the same length as endog")
            self.exposure = exposure
        self.endog = endog
        self.exog = exog
        self.family = family
        self.initialize()


    def initialize(self):
        """
        Initialize a generalized linear model.
        """
#TODO: intended for public use?
        self.history = { 'fittedvalues' : [], 'params' : [np.inf],
                         'deviance' : [np.inf]}
        self.iteration = 0
        self.pinv_wexog = np.linalg.pinv(self.exog)
        self.normalized_cov_params = np.dot(self.pinv_wexog,
                                        np.transpose(self.pinv_wexog))
        self.df_model = rank(self.exog)-1
        self.df_resid = self.exog.shape[0] - rank(self.exog)

    def score(self, params):
        """
        Score matrix.  Not yet implemeneted
        """
        raise NotImplementedError

    def loglike(self, *args):
        """
        Loglikelihood function.

        Each distribution family has its own loglikelihood function.
        See statsmodels.families.family
        """
        return self.family.loglike(*args)

    def information(self, params):
        """
        Fisher information matrix.  Not yet implemented.
        """
        raise NotImplementedError

    def _update_history(self, tmp_result, mu):
        """
        Helper method to update history during iterative fit.
        """
        self.history['params'].append(tmp_result.params)
        self.history['fittedvalues'].append(tmp_result.fittedvalues)
        self.history['deviance'].append(self.family.deviance(self.endog, mu))

    def estimate_scale(self, mu):
        """
        Estimates the dispersion/scale.

        Type of scale can be chose in the fit method.

        Parameters
        ----------
        mu : array
            mu is the mean response estimate

        Returns
        --------
        Estimate of scale

        Notes
        -----
        The default scale for Binomial and Poisson families is 1.  The default
        for the other families is Pearson's Chi-Square estimate.

        See also
        --------
        statsmodels.glm.fit for more information
        """
        if not self.scaletype:
            if isinstance(self.family, (families.Binomial, families.Poisson)):
                return 1.
            #make it so you can run from source tree
#            famstring = self.family.__str__().lower()
#            if 'poisson' in famstring or \
#                ('binomial' in famstring and 'negative' not in famstring):
#                return 1.
            else:
                resid = self.endog - mu
                return ((np.power(resid, 2) / self.family.variance(mu)).sum() \
                    / self.df_resid)

        if isinstance(self.scaletype, float):
            return np.array(self.scaletype)

        if isinstance(self.scaletype, str):
            if self.scaletype.lower() == 'x2':
                resid = self.endog - mu
                return ((np.power(resid, 2) / self.family.variance(mu)).sum() \
                    / self.df_resid)
            elif self.scaletype.lower() == 'dev':
                return self.family.deviance(self.endog, mu)/self.df_resid
            else:
                raise ValueError("Scale %s with type %s not understood" %\
                    (self.scaletype,type(self.scaletype)))

        else:
            raise ValueError("Scale %s with type %s not understood" %\
                (self.scaletype, type(self.scaletype)))

    def predict(self, exog, params=None, linear=False):
        """
        Return predicted values for a design matrix

        Parameters
        ----------
        exog : array-like
            Design / exogenous data
        params : array-like, optional after fit has been called
            Parameters / coefficients of a GLM.
        linear : bool
            If True, returns the linear predicted values.  If False,
            returns the value of the inverse of the model's link function at
            the linear predicted values.

        Returns
        -------
        An array of fitted values

        Notes
        -----
        If the model as not yet been fit, params is not optional.
        """
        if self._results is None and params is None:
            raise ValueError("If the model has not been fit, then you must \
specify the params argument.")
        if self._results is not None:
            params = self.results.params
        if linear:
            return np.dot(exog, params)
        else:
            return self.family.fitted(np.dot(exog, params))

    def fit(self, maxiter=100, method='IRLS', tol=1e-8, scale=None):
        '''
        Fits a generalized linear model for a given family.

        parameters
        ----------
        maxiter : int, optional
            Default is 100.
        method : string
            Default is 'IRLS' for iteratively reweighted least squares.  This
            is currently the only method available for GLM fit.
        scale : string or float, optional
            `scale` can be 'X2', 'dev', or a float
            The default value is None, which uses `X2` for Gamma, Gaussian,
            and Inverse Gaussian.
            `X2` is Pearson's chi-square divided by `df_resid`.
            The default is 1 for the Binomial and Poisson families.
            `dev` is the deviance divided by df_resid
        tol : float
            Convergence tolerance.  Default is 1e-8.
        '''
        endog = self.endog
        if endog.ndim > 1 and endog.shape[1] == 2:
            data_weights = endog.sum(1) # weights are total trials
        else:
            data_weights = np.ones((endog.shape[0]))
        self.data_weights = data_weights
        if np.shape(self.data_weights) == () and self.data_weights>1:
            self.data_weights = self.data_weights *\
                    np.ones((endog.shape[0]))
        self.scaletype = scale
        if isinstance(self.family, families.Binomial):
# this checks what kind of data is given for Binomial.
# family will need a reference to endog if this is to be removed from the
# preprocessing
            self.endog = self.family.initialize(self.endog)

        if hasattr(self, 'offset'):
            offset = self.offset
        elif hasattr(self, 'exposure'):
            offset = self.exposure
        else:
            offset = 0
        #TODO: would there ever be both and exposure and an offset?

        mu = self.family.starting_mu(self.endog)
        wlsexog = self.exog
        eta = self.family.predict(mu)
        self.iteration += 1
        dev = self.family.deviance(self.endog, mu)
        if np.isnan(dev):
            raise ValueError("The first guess on the deviance function \
returned a nan.  This could be a boundary problem and should be reported.")
        else:
            self.history['deviance'].append(dev)
            # first guess on the deviance is assumed to be scaled by 1.
        while((np.fabs(self.history['deviance'][self.iteration]-\
                    self.history['deviance'][self.iteration-1])) > tol and \
                    self.iteration < maxiter):
            self.weights = data_weights*self.family.weights(mu)
            wlsendog = eta + self.family.link.deriv(mu) * (self.endog-mu) \
                 - offset
            wls_results = WLS(wlsendog, wlsexog, self.weights).fit()
            eta = np.dot(self.exog, wls_results.params) + offset
            mu = self.family.fitted(eta)
            self._update_history(wls_results, mu)
            self.scale = self.estimate_scale(mu)
            self.iteration += 1
        self.mu = mu
        glm_results = GLMResults(self, wls_results.params,
                wls_results.normalized_cov_params, self.scale)
        return glm_results

class GLMResults(LikelihoodModelResults):
    '''
    Class to contain GLM results.

    GLMResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelReesults

    Returns
    -------
    **Attributes**

    aic : float
        Akaike Information Criterion
        -2 * `llf` + 2*(`df_model` + 1)
    bic : float
        Bayes Information Criterion
        `deviance` - `df_resid` * log(`nobs`)
    deviance : float
        See statsmodels.families.family for the distribution-specific deviance
        functions.
    df_model : float
        See GLM.df_model
    df_resid : float
        See GLM.df_resid
    fittedvalues : array
        Linear predicted values for the fitted model.
        dot(exog, params)
    llf : float
        Value of the loglikelihood function evalued at params.
        See statsmodels.families.family for distribution-specific loglikelihoods.
    model : class instance
        Pointer to GLM model instance that called fit.
    mu : array
        See GLM docstring.
    nobs : float
        The number of observations n.
    normalized_cov_params : array
        See GLM docstring
    null_deviance : float
        The value of the deviance function for the model fit with a constant
        as the only regressor.
    params : array
        The coefficients of the fitted model.  Note that interpretation
        of the coefficients often depends on the distribution family and the
        data.
    pearson_chi2 : array
        Pearson's Chi-Squared statistic is defined as the sum of the squares
        of the Pearson residuals.
    pinv_wexog : array
        See GLM docstring.
    pvalues : array
        The two-tailed p-values for the parameters.
    resid_anscombe : array
        Anscombe residuals.  See statsmodels.families.family for distribution-
        specific Anscombe residuals.
    resid_deviance : array
        Deviance residuals.  See statsmodels.families.family for distribution-
        specific deviance residuals.
    resid_pearson : array
        Pearson residuals.  The Pearson residuals are defined as
        (`endog` - `mu`)/sqrt(VAR(`mu`)) where VAR is the distribution
        specific variance function.  See statsmodels.families.family and
        statsmodels.families.varfuncs for more information.
    resid_response : array
        Respnose residuals.  The response residuals are defined as
        `endog` - `fittedvalues`
    resid_working : array
        Working residuals.  The working residuals are defined as
        `resid_response`/link'(`mu`).  See statsmodels.family.links for the
        derivatives of the link functions.  They are defined analytically.
    scale : array
        The estimate of the scale / dispersion for the model fit.
        See GLM.fit and GLM.estimate_scale for more information.
    stand_errors : array
        The standard errors of the fitted GLM.   #TODO still named bse

    See Also
    --------
    statsmodels.LikelihoodModelResults
    '''
#TODO: add a z value function to LLMResults

    def __init__(self, model, params, normalized_cov_params, scale):
        super(GLMResults, self).__init__(model, params,
                normalized_cov_params=normalized_cov_params, scale=scale)
        self.model._results = self.model.results = self # TODO: get rid of this
                                                      # since results isn't a
                                                      # property for GLM
        # above is needed for model.predict
        self.family = model.family
        self._endog = model.endog
        self.nobs = model.endog.shape[0]
        self.mu = model.mu
        self._data_weights = model.data_weights
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        self.pinv_wexog = model.pinv_wexog
        self._cache = resettable_cache()
# are these intermediate results needed or can we just call the model's attributes?

    @cache_readonly
    def resid_response(self):
        return self._data_weights * (self._endog-self.mu)

    @cache_readonly
    def resid_pearson(self):
        return np.sqrt(self._data_weights) * (self._endog-self.mu)/\
                        np.sqrt(self.family.variance(self.mu))

    @cache_readonly
    def resid_working(self):
        val = (self.resid_response / self.family.link.deriv(self.mu))
        val *= self._data_weights
        return val

    @cache_readonly
    def resid_anscombe(self):
        return self.family.resid_anscombe(self._endog, self.mu)

    @cache_readonly
    def resid_deviance(self):
        return self.family.resid_dev(self._endog, self.mu)


    @cache_readonly
    def pvalues(self):
        return t.sf(np.abs(self.tvalues), self.df_resid)*2

    @cache_readonly
    def pearson_chi2(self):
        chisq =  (self._endog- self.mu)**2 / self.family.variance(self.mu)
        chisq *= self._data_weights
        chisqsum = np.sum(chisq)
        return chisqsum

    @cache_readonly
    def fittedvalues(self):
        return self.mu

    @cache_readonly
    def null(self):
        endog = self._endog
        model = self.model
        exog = np.ones((len(endog),1))
        if hasattr(model, 'offset'):
            return GLM(endog, exog, offset=model.offset,
                family=self.family).fit().mu
        elif hasattr(model, 'exposure'):
            return GLM(endog, exog, exposure=model.exposure,
                    family=self.family).fit().mu
        else:
            return WLS(endog, exog,
                weights=self._data_weights).fit().fittedvalues

    @cache_readonly
    def deviance(self):
        return self.family.deviance(self._endog, self.mu)

    @cache_readonly
    def null_deviance(self):
        return self.family.deviance(self._endog, self.null)

    @cache_readonly
    def llf(self):
        _modelfamily = self.family
        if isinstance(_modelfamily, families.NegativeBinomial):
            val = _modelfamily.loglike(self.model.endog,
                        fittedvalues = np.dot(self.model.exog,self.params))
        else:
            val = _modelfamily.loglike(self._endog, self.mu,
                                    scale=self.scale)
        return val

    @cache_readonly
    def aic(self):
        return -2 * self.llf + 2*(self.df_model+1)

    @cache_readonly
    def bic(self):
        return self.deviance - self.df_resid*np.log(self.nobs)

    def summary2(self, yname=None, xnames=None, title=0, alpha=.05,
                returns='print'):
        """
        This is for testing the new summary setup
        """
        from scikits.statsmodels.iolib.summary import summary as smry
        return smry(self, yname=yname, xname=xnames, title=0, alpha=.05, returns='print')


#TODO: write summary method to use output.py in sandbox
    def summary(self, yname=None, xname=None, title='Generalized linear model',
                returns='text'):
        """
        Print a table of results or returns SimpleTable() instance which
        summarizes the Generalized linear model results.

        Parameters
        -----------
        yname : string
                optional, Default is `Y`
        xname : list of strings
                optional, Default is `X.#` for # in p the number of regressors
        title : string
                optional, Defualt is 'Generalized linear model'
        returns : string
                  'text', 'table', 'csv', 'latex', 'html'

        Returns
        -------
        Defualt :
        returns='print'
                Prints the summarirized results

        Option :
        returns='text'
                Prints the summarirized results

        Option :
        returns='table'
                 SimpleTable instance : summarizing the fit of a linear model.

        Option :
        returns='csv'
                returns a string of csv of the results, to import into a spreadsheet

        Option :
        returns='latex'
        Not implimented yet

        Option :
        returns='HTML'
        Not implimented yet


        Examples (needs updating)
        --------
        >>> import scikits.statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> ols_results = sm.OLS(data.endog, data.exog).results
        >>> print ols_results.summary()
        ...

        Notes
        -----
        stand_errors are not implimented.
        conf_int calculated from normal dist.
        """
        import time as Time
        from scikits.statsmodels.iolib import SimpleTable
        from scikits.statsmodels.stats.stattools import jarque_bera, omni_normtest, durbin_watson

        if yname is None:
            yname = 'Y'
        if xname is None:
            xname = ['x%d' % i for i in range(self.model.exog.shape[1])]

        #List of results used in summary
        #yname = yname
        #xname = xname
        time = Time.localtime()
        dist_family = self.model.family.__class__.__name__
        aic = self.aic
        bic = self.bic
        deviance = self.deviance
        df_model = self.df_model
        df_resid = self.df_resid
        fittedvalues = self.fittedvalues
        llf = self.llf
        mu = self.mu
        nobs = self.nobs
        normalized_cov_params = self.normalized_cov_params
        null_deviance = self.null_deviance
        params = self.params
        pearson_chi2 = self.pearson_chi2
        pinv_wexog = self.pinv_wexog
        resid_anscombe = self.resid_anscombe
        resid_deviance = self.resid_deviance
        resid_pearson = self.resid_pearson
        resid_response = self.resid_response
        resid_working = self.resid_working
        scale = self.scale
#TODO   #stand_errors = self.stand_errors
        stand_errors = self.bse  #[' ' for x in range(len(self.params))]
#Added note about conf_int
        pvalues = self.pvalues
        conf_int = self.conf_int()
        cov_params = self.cov_params()
        #f_test() = self.f_test()
        t = self.tvalues
        #t_test = self.t_test()



        table_1l_fmt = dict(
            data_fmts = ["%s", "%s", "%s", "%s", "%s"],
            empty_cell = '',
            colwidths = 15,
            colsep='   ',
            row_pre = '  ',
            row_post = '  ',
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
        # Note table_1l_fmt over rides the below formating. in extend_right? JP
        table_1r_fmt = dict(
            data_fmts = ["%s", "%s", "%s", "%s", "%1s"],
            empty_cell = '',
            colwidths = 12,
            colsep='   ',
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

        table_2_fmt = dict(
            data_fmts = ["%s", "%s", "%s", "%s"],
            #data_fmts = ["%#12.6g","%#12.6g","%#10.4g","%#5.4g"],
            #data_fmts = ["%#10.4g","%#6.4f", "%#6.4f"],
            #data_fmts = ["%#15.4F","%#15.4F","%#15.4F","%#14.4G"],
            empty_cell = '',
            colwidths = 13,
            colsep=' ',
            row_pre = '  ',
            row_post = '   ',
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
        ########  summary table 1   #######
        table_1l_title = title
        table_1l_header = None
        table_1l_stubs = ('Model Family:',
                          'Method:',
                          'Dependent Variable:',
                          'Date:',
                          'Time:',
                          )
        table_1l_data = [
                         [dist_family],
                         ['IRLS'],
                         [yname],
                         [Time.strftime("%a, %d %b %Y",time)],
                         [Time.strftime("%H:%M:%S",time)],
                        ]
        table_1l = SimpleTable(table_1l_data,
                            table_1l_header,
                            table_1l_stubs,
                            title=table_1l_title,
                            txt_fmt = table_1l_fmt)
        table_1r_title = None
        table_1r_header = None
        table_1r_stubs = ('# of obs:',
                          'Df residuals:',
                          'Df model:',
                          'Scale:',
                          'Log likelihood:'
                          )
        table_1r_data = [
                         [nobs],
                         [df_resid],
                         [df_model],
                         ["%#6.4f" % (scale,)],
                         ["%#6.4f" % (llf,)]
                        ]
        table_1r = SimpleTable(table_1r_data,
                            table_1r_header,
                            table_1r_stubs,
                            title=table_1r_title,
                            txt_fmt = table_1r_fmt)

        ########  summary table 2   #######
#TODO add % range to confidance interval column header
        table_2header = ('coefficient', 'stand errors', 't-statistic',
        'Conf. Interval')
        table_2stubs = xname
        table_2data = zip(["%#6.4f" % (params[i]) for i in range(len(xname))],
                          ["%#6.4f" % stand_errors[i] for i in range(len(xname))],
                          ["%#6.4f" % (t[i]) for i in range(len(xname))],
                          [""" [%#6.3f, %#6.3f]""" % tuple(conf_int[i]) for i in
                                                             range(len(xname))])


        #dfmt={'data_fmt':["%#12.6g","%#12.6g","%#10.4g","%#5.4g"]}
        table_2 = SimpleTable(table_2data,
                            table_2header,
                            table_2stubs,
                            title=None,
                            txt_fmt = table_2_fmt)

        ########  Return Summary Tables ########
        # join table table_s then print
        if returns == 'text':
            table_1l.extend_right(table_1r)
            return str(table_1l) + '\n' +  str(table_2)
        elif returns == 'print':
            table_1l.extend_right(table_1r)
            print(str(table_1l) + '\n' +  str(table_2))
        elif returns == 'tables':
            return [table_1l, table_1r, table_2]
            #return [table_1, table_2 ,table_3L, notes]
        elif returns == 'csv':
            return table_1.as_csv() + '\n' + table_2.as_csv() + '\n' + \
                   table_3L.as_csv()
        elif returns == 'latex':
            print('not avalible yet')
        elif returns == html:
            print('not avalible yet')

if __name__ == "__main__":
    import scikits.statsmodels.api as sm
    import numpy as np
    data = sm.datasets.longley.load()
    #data.exog = add_constant(data.exog)
    GLMmod = GLM(data.endog, data.exog).fit()
    GLMT = GLMmod.summary(returns='tables')
##    GLMT[0].extend_right(GLMT[1])
##    print(GLMT[0])
##    print(GLMT[2])
    GLMTp = GLMmod.summary(title='Test GLM')


    """
From Stata
. webuse beetle
. glm r i.beetle ldose, family(binomial n) link(cloglog)

Iteration 0:   log likelihood = -79.012269
Iteration 1:   log likelihood =  -76.94951
Iteration 2:   log likelihood = -76.945645
Iteration 3:   log likelihood = -76.945645

Generalized linear models                          No. of obs      =        24
Optimization     : ML                              Residual df     =        20
                                                   Scale parameter =         1
Deviance         =  73.76505595                    (1/df) Deviance =  3.688253
Pearson          =   71.8901173                    (1/df) Pearson  =  3.594506

Variance function: V(u) = u*(1-u/n)                [Binomial]
Link function    : g(u) = ln(-ln(1-u/n))           [Complementary log-log]

                                                   AIC             =   6.74547
Log likelihood   = -76.94564525                    BIC             =  10.20398

------------------------------------------------------------------------------
             |                 OIM
           r |      Coef.   Std. Err.      z    P>|z|     [95% Conf. Interval]
-------------+----------------------------------------------------------------
      beetle |
          2  |  -.0910396   .1076132    -0.85   0.398    -.3019576    .1198783
          3  |  -1.836058   .1307125   -14.05   0.000     -2.09225   -1.579867
             |
       ldose |   19.41558   .9954265    19.50   0.000     17.46458    21.36658
       _cons |  -34.84602    1.79333   -19.43   0.000    -38.36089   -31.33116
------------------------------------------------------------------------------
"""

    #NOTE: wfs dataset has been removed due to a licensing issue
    # example of using offset
    #data = sm.datasets.wfs.load()
    # get offset
    #offset = np.log(data.exog[:,-1])
    #exog = data.exog[:,:-1]

    # convert dur to dummy
    #exog = sm.tools.categorical(exog, col=0, drop=True)
    # drop reference category
    # convert res to dummy
    #exog = sm.tools.categorical(exog, col=0, drop=True)
    # convert edu to dummy
    #exog = sm.tools.categorical(exog, col=0, drop=True)
    # drop reference categories and add intercept
    #exog = sm.add_constant(exog[:,[1,2,3,4,5,7,8,10,11,12]])

    #endog = np.round(data.endog)
    #mod = sm.GLM(endog, exog, family=sm.families.Poisson()).fit()

    #res1 = GLM(endog, exog, family=sm.families.Poisson(),
    #                        offset=offset).fit(tol=1e-12, maxiter=250)
    #exposuremod = GLM(endog, exog, family=sm.families.Poisson(),
    #                        exposure = data.exog[:,-1]).fit(tol=1e-12,
    #                                                        maxiter=250)
    #assert(np.all(res1.params == exposuremod.params))
