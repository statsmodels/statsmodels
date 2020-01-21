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

from . import families

from statsmodels.tools.decorators import (cache_readonly,
                                          cached_data, cached_value)
from statsmodels.compat.pandas import Appender

import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
import statsmodels.regression._tools as reg_tools

from statsmodels.graphics._regressionplots_doc import (
    _plot_added_variable_doc,
    _plot_partial_residuals_doc,
    _plot_ceres_residuals_doc)

# need import in module instead of lazily to copy `__doc__`
from . import _prediction as pred
from statsmodels.genmod._prediction import PredictionResults

from statsmodels.tools.sm_exceptions import (PerfectSeparationError,
                                             DomainWarning,
                                             HessianInversionWarning)

from numpy.linalg.linalg import LinAlgError

__all__ = ['GLM', 'PredictionResults']


def _check_convergence(criterion, iteration, atol, rtol):
    return np.allclose(criterion[iteration], criterion[iteration + 1],
                       atol=atol, rtol=rtol)


class GLM(base.LikelihoodModel):
    __doc__ = """
    Generalized Linear Models

    GLM inherits from statsmodels.base.model.LikelihoodModel

    Parameters
    ----------
    endog : array_like
        1d array of endogenous response variable.  This array can be 1d or 2d.
        Binomial family models accept a 2d array with two columns. If
        supplied, each observation is expected to be [success, failure].
    exog : array_like
        A nobs x k array where `nobs` is the number of observations and `k`
        is the number of regressors. An intercept is not included by default
        and should be added by the user (models specified using a formula
        include an intercept by default). See `statsmodels.tools.add_constant`.
    family : family class instance
        The default is Gaussian.  To specify the binomial distribution
        family = sm.family.Binomial()
        Each family can take a link instance as an argument.  See
        statsmodels.family.family for more information.
    offset : array_like or None
        An offset to be included in the model.  If provided, must be
        an array whose length is the number of rows in exog.
    exposure : array_like or None
        Log(exposure) will be added to the linear prediction in the model.
        Exposure is only valid if the log link is used. If provided, it must be
        an array with the same length as endog.
    freq_weights : array_like
        1d array of frequency weights. The default is None. If None is selected
        or a blank value, then the algorithm will replace with an array of 1's
        with length equal to the endog.
        WARNING: Using weights is not verified yet for all possible options
        and results, see Notes.
    var_weights : array_like
        1d array of variance (analytic) weights. The default is None. If None
        is selected or a blank value, then the algorithm will replace with an
        array of 1's with length equal to the endog.
        WARNING: Using weights is not verified yet for all possible options
        and results, see Notes.
    %(extra_params)s

    Attributes
    ----------
    df_model : float
        Model degrees of freedom is equal to p - 1, where p is the number
        of regressors.  Note that the intercept is not reported as a
        degree of freedom.
    df_resid : float
        Residual degrees of freedom is equal to the number of observation n
        minus the number of regressors p.
    endog : ndarray
        See Notes.  Note that `endog` is a reference to the data so that if
        data is already an array and it is changed, then `endog` changes
        as well.
    exposure : array_like
        Include ln(exposure) in model with coefficient constrained to 1. Can
        only be used if the link is the logarithm function.
    exog : ndarray
        See Notes.  Note that `exog` is a reference to the data so that if
        data is already an array and it is changed, then `exog` changes
        as well.
    freq_weights : ndarray
        See Notes. Note that `freq_weights` is a reference to the data so that
        if data is already an array and it is changed, then `freq_weights`
        changes as well.
    var_weights : ndarray
        See Notes. Note that `var_weights` is a reference to the data so that
        if data is already an array and it is changed, then `var_weights`
        changes as well.
    iteration : int
        The number of iterations that fit has run.  Initialized at 0.
    family : family class instance
        The distribution family of the model. Can be any family in
        statsmodels.families.  Default is Gaussian.
    mu : ndarray
        The mean response of the transformed variable.  `mu` is the value of
        the inverse of the link function at lin_pred, where lin_pred is the
        linear predicted value of the WLS fit of the transformed variable.
        `mu` is only available after fit is called.  See
        statsmodels.families.family.fitted of the distribution family for more
        information.
    n_trials : ndarray
        See Notes. Note that `n_trials` is a reference to the data so that if
        data is already an array and it is changed, then `n_trials` changes
        as well. `n_trials` is the number of binomial trials and only available
        with that distribution. See statsmodels.families.Binomial for more
        information.
    normalized_cov_params : ndarray
        The p x p normalized covariance of the design / exogenous data.
        This is approximately equal to (X.T X)^(-1)
    offset : array_like
        Include offset in model with coefficient constrained to 1.
    scale : float
        The estimate of the scale / dispersion of the model fit.  Only
        available after fit is called.  See GLM.fit and GLM.estimate_scale
        for more information.
    scaletype : str
        The scaling used for fitting the model.  This is only available after
        fit is called.  The default is None.  See GLM.fit for more information.
    weights : ndarray
        The value of the weights after the last iteration of fit.  Only
        available after fit is called.  See statsmodels.families.family for
        the specific distribution weighting functions.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.scotland.load(as_pandas=False)
    >>> data.exog = sm.add_constant(data.exog)

    Instantiate a gamma family model with the default link function.

    >>> gamma_model = sm.GLM(data.endog, data.exog,
    ...                      family=sm.families.Gamma())

    >>> gamma_results = gamma_model.fit()
    >>> gamma_results.params
    array([-0.01776527,  0.00004962,  0.00203442, -0.00007181,  0.00011185,
           -0.00000015, -0.00051868, -0.00000243])
    >>> gamma_results.scale
    0.0035842831734919055
    >>> gamma_results.deviance
    0.087388516416999198
    >>> gamma_results.pearson_chi2
    0.086022796163805704
    >>> gamma_results.llf
    -83.017202161073527

    See Also
    --------
    statsmodels.genmod.families.family.Family
    :ref:`families`
    :ref:`links`

    Notes
    -----
    Only the following combinations make sense for family and link:

     ============= ===== === ===== ====== ======= === ==== ====== ====== ====
     Family        ident log logit probit cloglog pow opow nbinom loglog logc
     ============= ===== === ===== ====== ======= === ==== ====== ====== ====
     Gaussian      x     x   x     x      x       x   x     x      x
     inv Gaussian  x     x                        x
     binomial      x     x   x     x      x       x   x           x      x
     Poisson       x     x                        x
     neg binomial  x     x                        x        x
     gamma         x     x                        x
     Tweedie       x     x                        x
     ============= ===== === ===== ====== ======= === ==== ====== ====== ====

    Not all of these link functions are currently available.

    Endog and exog are references so that if the data they refer to are already
    arrays and these arrays are changed, endog and exog will change.

    statsmodels supports two separate definitions of weights: frequency weights
    and variance weights.

    Frequency weights produce the same results as repeating observations by the
    frequencies (if those are integers). Frequency weights will keep the number
    of observations consistent, but the degrees of freedom will change to
    reflect the new weights.

    Variance weights (referred to in other packages as analytic weights) are
    used when ``endog`` represents an an average or mean. This relies on the
    assumption that that the inverse variance scales proportionally to the
    weight--an observation that is deemed more credible should have less
    variance and therefore have more weight. For the ``Poisson`` family--which
    assumes that occurrences scale proportionally with time--a natural practice
    would be to use the amount of time as the variance weight and set ``endog``
    to be a rate (occurrences per period of time). Similarly, using a
    compound Poisson family, namely ``Tweedie``, makes a similar assumption
    about the rate (or frequency) of occurrences having variance proportional to
    time.

    Both frequency and variance weights are verified for all basic results with
    nonrobust or heteroscedasticity robust ``cov_type``. Other robust
    covariance types have not yet been verified, and at least the small sample
    correction is currently not based on the correct total frequency count.

    Currently, all residuals are not weighted by frequency, although they may
    incorporate ``n_trials`` for ``Binomial`` and ``var_weights``

    +---------------+----------------------------------+
    | Residual Type | Applicable weights               |
    +===============+==================================+
    | Anscombe      | ``var_weights``                  |
    +---------------+----------------------------------+
    | Deviance      | ``var_weights``                  |
    +---------------+----------------------------------+
    | Pearson       | ``var_weights`` and ``n_trials`` |
    +---------------+----------------------------------+
    | Reponse       | ``n_trials``                     |
    +---------------+----------------------------------+
    | Working       | ``n_trials``                     |
    +---------------+----------------------------------+

    WARNING: Loglikelihood and deviance are not valid in models where
    scale is equal to 1 (i.e., ``Binomial``, ``NegativeBinomial``, and
    ``Poisson``). If variance weights are specified, then results such as
    ``loglike`` and ``deviance`` are based on a quasi-likelihood
    interpretation. The loglikelihood is not correctly specified in this case,
    and statistics based on it, such AIC or likelihood ratio tests, are not
    appropriate.
    """ % {'extra_params': base._missing_param_doc}
    # Maximum number of endogenous variables when using a formula
    _formula_max_endog = 2

    def __init__(self, endog, exog, family=None, offset=None,
                 exposure=None, freq_weights=None, var_weights=None,
                 missing='none', **kwargs):

        if (family is not None) and not isinstance(family.link,
                                                   tuple(family.safe_links)):

            import warnings
            warnings.warn(("The %s link function does not respect the domain "
                           "of the %s family.") %
                          (family.link.__class__.__name__,
                           family.__class__.__name__),
                          DomainWarning)

        if exposure is not None:
            exposure = np.log(exposure)
        if offset is not None:  # this should probably be done upstream
            offset = np.asarray(offset)

        if freq_weights is not None:
            freq_weights = np.asarray(freq_weights)
        if var_weights is not None:
            var_weights = np.asarray(var_weights)

        self.freq_weights = freq_weights
        self.var_weights = var_weights

        super(GLM, self).__init__(endog, exog, missing=missing,
                                  offset=offset, exposure=exposure,
                                  freq_weights=freq_weights,
                                  var_weights=var_weights, **kwargs)
        self._check_inputs(family, self.offset, self.exposure, self.endog,
                           self.freq_weights, self.var_weights)
        if offset is None:
            delattr(self, 'offset')
        if exposure is None:
            delattr(self, 'exposure')

        self.nobs = self.endog.shape[0]

        # things to remove_data
        self._data_attr.extend(['weights', 'mu', 'freq_weights',
                                'var_weights', 'iweights', '_offset_exposure',
                                'n_trials'])
        # register kwds for __init__, offset and exposure are added by super
        self._init_keys.append('family')

        self._setup_binomial()
        # internal usage for recreating a model
        if 'n_trials' in kwargs:
            self.n_trials = kwargs['n_trials']

        # Construct a combined offset/exposure term.  Note that
        # exposure has already been logged if present.
        offset_exposure = 0.
        if hasattr(self, 'offset'):
            offset_exposure = self.offset
        if hasattr(self, 'exposure'):
            offset_exposure = offset_exposure + self.exposure
        self._offset_exposure = offset_exposure

        self.scaletype = None

    def initialize(self):
        """
        Initialize a generalized linear model.
        """
        self.df_model = np.linalg.matrix_rank(self.exog) - 1

        if (self.freq_weights is not None) and \
           (self.freq_weights.shape[0] == self.endog.shape[0]):
            self.wnobs = self.freq_weights.sum()
            self.df_resid = self.wnobs - self.df_model - 1
        else:
            self.wnobs = self.exog.shape[0]
            self.df_resid = self.exog.shape[0] - self.df_model - 1

    def _check_inputs(self, family, offset, exposure, endog, freq_weights,
                      var_weights):

        # Default family is Gaussian
        if family is None:
            family = families.Gaussian()
        self.family = family

        if exposure is not None:
            if not isinstance(self.family.link, families.links.Log):
                raise ValueError("exposure can only be used with the log "
                                 "link function")
            elif exposure.shape[0] != endog.shape[0]:
                raise ValueError("exposure is not the same length as endog")

        if offset is not None:
            if offset.shape[0] != endog.shape[0]:
                raise ValueError("offset is not the same length as endog")

        if freq_weights is not None:
            if freq_weights.shape[0] != endog.shape[0]:
                raise ValueError("freq weights not the same length as endog")
            if len(freq_weights.shape) > 1:
                raise ValueError("freq weights has too many dimensions")

        # internal flag to store whether freq_weights were not None
        self._has_freq_weights = (self.freq_weights is not None)
        if self.freq_weights is None:
            self.freq_weights = np.ones((endog.shape[0]))
            # TODO: check do we want to keep None as sentinel for freq_weights

        if np.shape(self.freq_weights) == () and self.freq_weights > 1:
            self.freq_weights = (self.freq_weights *
                                 np.ones((endog.shape[0])))

        if var_weights is not None:
            if var_weights.shape[0] != endog.shape[0]:
                raise ValueError("var weights not the same length as endog")
            if len(var_weights.shape) > 1:
                raise ValueError("var weights has too many dimensions")

        # internal flag to store whether var_weights were not None
        self._has_var_weights = (var_weights is not None)
        if var_weights is None:
            self.var_weights = np.ones((endog.shape[0]))
            # TODO: check do we want to keep None as sentinel for var_weights
        self.iweights = np.asarray(self.freq_weights * self.var_weights)

    def _get_init_kwds(self):
        # this is a temporary fixup because exposure has been transformed
        # see #1609, copied from discrete_model.CountModel
        kwds = super(GLM, self)._get_init_kwds()
        if 'exposure' in kwds and kwds['exposure'] is not None:
            kwds['exposure'] = np.exp(kwds['exposure'])
        return kwds

    def loglike_mu(self, mu, scale=1.):
        """
        Evaluate the log-likelihood for a generalized linear model.
        """
        return self.family.loglike(self.endog, mu, self.var_weights,
                                   self.freq_weights, scale)

    def loglike(self, params, scale=None):
        """
        Evaluate the log-likelihood for a generalized linear model.
        """
        lin_pred = np.dot(self.exog, params) + self._offset_exposure
        expval = self.family.link.inverse(lin_pred)
        if scale is None:
            scale = self.estimate_scale(expval)
        llf = self.family.loglike(self.endog, expval, self.var_weights,
                                  self.freq_weights, scale)
        return llf

    def score_obs(self, params, scale=None):
        """score first derivative of the loglikelihood for each observation.

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.

        Returns
        -------
        score_obs : ndarray, 2d
            The first derivative of the loglikelihood function evaluated at
            params for each observation.
        """

        score_factor = self.score_factor(params, scale=scale)
        return score_factor[:, None] * self.exog

    def score(self, params, scale=None):
        """score, first derivative of the loglikelihood function

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.

        Returns
        -------
        score : ndarray_1d
            The first derivative of the loglikelihood function calculated as
            the sum of `score_obs`
        """
        score_factor = self.score_factor(params, scale=scale)
        return np.dot(score_factor, self.exog)

    def score_factor(self, params, scale=None):
        """weights for score for each observation

        This can be considered as score residuals.

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.

        Returns
        -------
        score_factor : ndarray_1d
            A 1d weight vector used in the calculation of the score_obs.
            The score_obs are obtained by `score_factor[:, None] * exog`
        """
        mu = self.predict(params)
        if scale is None:
            scale = self.estimate_scale(mu)

        score_factor = (self.endog - mu) / self.family.link.deriv(mu)
        score_factor /= self.family.variance(mu)
        score_factor *= self.iweights * self.n_trials

        if not scale == 1:
            score_factor /= scale

        return score_factor

    def hessian_factor(self, params, scale=None, observed=True):
        """Weights for calculating Hessian

        Parameters
        ----------
        params : ndarray
            parameter at which Hessian is evaluated
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.

        Returns
        -------
        hessian_factor : ndarray, 1d
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`
        """

        # calculating eim_factor
        mu = self.predict(params)
        if scale is None:
            scale = self.estimate_scale(mu)

        eim_factor = 1 / (self.family.link.deriv(mu)**2 *
                          self.family.variance(mu))
        eim_factor *= self.iweights * self.n_trials

        if not observed:
            if not scale == 1:
                eim_factor /= scale
            return eim_factor

        # calculating oim_factor, eim_factor is with scale=1

        score_factor = self.score_factor(params, scale=1.)
        if eim_factor.ndim > 1 or score_factor.ndim > 1:
            raise RuntimeError('something wrong')

        tmp = self.family.variance(mu) * self.family.link.deriv2(mu)
        tmp += self.family.variance.deriv(mu) * self.family.link.deriv(mu)

        tmp = score_factor * tmp
        # correct for duplicatee iweights in oim_factor and score_factor
        tmp /= self.iweights * self.n_trials
        oim_factor = eim_factor * (1 + tmp)

        if tmp.ndim > 1:
            raise RuntimeError('something wrong')

        if not scale == 1:
            oim_factor /= scale

        return oim_factor

    def hessian(self, params, scale=None, observed=None):
        """Hessian, second derivative of loglikelihood function

        Parameters
        ----------
        params : ndarray
            parameter at which Hessian is evaluated
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned (default).
            If false then the expected information matrix is returned.

        Returns
        -------
        hessian : ndarray
            Hessian, i.e. observed information, or expected information matrix.
        """
        if observed is None:
            if getattr(self, '_optim_hessian', None) == 'eim':
                observed = False
            else:
                observed = True

        tmp = getattr(self, '_tmp_like_exog', np.empty_like(self.exog))

        factor = self.hessian_factor(params, scale=scale, observed=observed)
        np.multiply(self.exog.T, factor, out=tmp.T)
        return -tmp.T.dot(self.exog)

    def information(self, params, scale=None):
        """
        Fisher information matrix.
        """
        return self.hessian(params, scale=scale, observed=False)

    def _deriv_mean_dparams(self, params):
        """
        Derivative of the expected endog with respect to the parameters.

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to the parameter vector.
        """
        lin_pred = self.predict(params, linear=True)
        idl = self.family.link.inverse_deriv(lin_pred)
        dmat = self.exog * idl[:, None]
        return dmat

    def _deriv_score_obs_dendog(self, params, scale=None):
        """derivative of score_obs w.r.t. endog

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.

        Returns
        -------
        derivative : ndarray_2d
            The derivative of the score_obs with respect to endog. This
            can is given by `score_factor0[:, None] * exog` where
            `score_factor0` is the score_factor without the residual.
        """
        mu = self.predict(params)
        if scale is None:
            scale = self.estimate_scale(mu)

        score_factor = 1 / self.family.link.deriv(mu)
        score_factor /= self.family.variance(mu)
        score_factor *= self.iweights * self.n_trials

        if not scale == 1:
            score_factor /= scale

        return score_factor[:, None] * self.exog

    def score_test(self, params_constrained, k_constraints=None,
                   exog_extra=None, observed=True):
        """score test for restrictions or for omitted variables

        The covariance matrix for the score is based on the Hessian, i.e.
        observed information matrix or optionally on the expected information
        matrix..

        Parameters
        ----------
        params_constrained : array_like
            estimated parameter of the restricted model. This can be the
            parameter estimate for the current when testing for omitted
            variables.
        k_constraints : int or None
            Number of constraints that were used in the estimation of params
            restricted relative to the number of exog in the model.
            This must be provided if no exog_extra are given. If exog_extra is
            not None, then k_constraints is assumed to be zero if it is None.
        exog_extra : None or array_like
            Explanatory variables that are jointly tested for inclusion in the
            model, i.e. omitted variables.
        observed : bool
            If True, then the observed Hessian is used in calculating the
            covariance matrix of the score. If false then the expected
            information matrix is used.

        Returns
        -------
        chi2_stat : float
            chisquare statistic for the score test
        p-value : float
            P-value of the score test based on the chisquare distribution.
        df : int
            Degrees of freedom used in the p-value calculation. This is equal
            to the number of constraints.

        Notes
        -----
        not yet verified for case with scale not equal to 1.
        """

        if exog_extra is None:
            if k_constraints is None:
                raise ValueError('if exog_extra is None, then k_constraints'
                                 'needs to be given')

            score = self.score(params_constrained)
            hessian = self.hessian(params_constrained, observed=observed)

        else:
            # exog_extra = np.asarray(exog_extra)
            if k_constraints is None:
                k_constraints = 0

            ex = np.column_stack((self.exog, exog_extra))
            k_constraints += ex.shape[1] - self.exog.shape[1]

            score_factor = self.score_factor(params_constrained)
            score = (score_factor[:, None] * ex).sum(0)
            hessian_factor = self.hessian_factor(params_constrained,
                                                 observed=observed)
            hessian = -np.dot(ex.T * hessian_factor, ex)

        from scipy import stats
        # TODO check sign, why minus?
        chi2stat = -score.dot(np.linalg.solve(hessian, score[:, None]))
        pval = stats.chi2.sf(chi2stat, k_constraints)
        # return a stats results instance instead?  Contrast?
        return chi2stat, pval, k_constraints

    def _update_history(self, tmp_result, mu, history):
        """
        Helper method to update history during iterative fit.
        """
        history['params'].append(tmp_result.params)
        history['deviance'].append(self.family.deviance(self.endog, mu,
                                                        self.var_weights,
                                                        self.freq_weights,
                                                        self.scale))
        return history

    def estimate_scale(self, mu):
        """
        Estimate the dispersion/scale.

        Type of scale can be chose in the fit method.

        Parameters
        ----------
        mu : ndarray
            mu is the mean response estimate

        Returns
        -------
        Estimate of scale

        Notes
        -----
        The default scale for Binomial, Poisson and Negative Binomial
        families is 1.  The default for the other families is Pearson's
        Chi-Square estimate.

        See Also
        --------
        statsmodels.genmod.generalized_linear_model.GLM.fit
        """
        if not self.scaletype:
            if isinstance(self.family, (families.Binomial, families.Poisson,
                                        families.NegativeBinomial)):
                return 1.
            else:
                return self._estimate_x2_scale(mu)

        if isinstance(self.scaletype, float):
            return np.array(self.scaletype)

        if isinstance(self.scaletype, str):
            if self.scaletype.lower() == 'x2':
                return self._estimate_x2_scale(mu)
            elif self.scaletype.lower() == 'dev':
                return (self.family.deviance(self.endog, mu, self.var_weights,
                                             self.freq_weights, 1.) /
                        (self.df_resid))
            else:
                raise ValueError("Scale %s with type %s not understood" %
                                 (self.scaletype, type(self.scaletype)))
        else:
            raise ValueError("Scale %s with type %s not understood" %
                             (self.scaletype, type(self.scaletype)))

    def _estimate_x2_scale(self, mu):
        resid = np.power(self.endog - mu, 2) * self.iweights
        return np.sum(resid / self.family.variance(mu)) / self.df_resid

    def estimate_tweedie_power(self, mu, method='brentq', low=1.01, high=5.):
        """
        Tweedie specific function to estimate scale and the variance parameter.
        The variance parameter is also referred to as p, xi, or shape.

        Parameters
        ----------
        mu : array_like
            Fitted mean response variable
        method : str, defaults to 'brentq'
            Scipy optimizer used to solve the Pearson equation. Only brentq
            currently supported.
        low : float, optional
            Low end of the bracketing interval [a,b] to be used in the search
            for the power. Defaults to 1.01.
        high : float, optional
            High end of the bracketing interval [a,b] to be used in the search
            for the power. Defaults to 5.

        Returns
        -------
        power : float
            The estimated shape or power.
        """
        if method == 'brentq':
            from scipy.optimize import brentq

            def psi_p(power, mu):
                scale = ((self.iweights * (self.endog - mu) ** 2 /
                          (mu ** power)).sum() / self.df_resid)
                return (np.sum(self.iweights * ((self.endog - mu) ** 2 /
                               (scale * (mu ** power)) - 1) *
                               np.log(mu)) / self.freq_weights.sum())
            power = brentq(psi_p, low, high, args=(mu))
        else:
            raise NotImplementedError('Only brentq can currently be used')
        return power

    def predict(self, params, exog=None, exposure=None, offset=None,
                linear=False):
        """
        Return predicted values for a design matrix

        Parameters
        ----------
        params : array_like
            Parameters / coefficients of a GLM.
        exog : array_like, optional
            Design / exogenous data. Is exog is None, model exog is used.
        exposure : array_like, optional
            Exposure time values, only can be used with the log link
            function.  See notes for details.
        offset : array_like, optional
            Offset values.  See notes for details.
        linear : bool
            If True, returns the linear predicted values.  If False,
            returns the value of the inverse of the model's link function at
            the linear predicted values.

        Returns
        -------
        An array of fitted values

        Notes
        -----
        Any `exposure` and `offset` provided here take precedence over
        the `exposure` and `offset` used in the model fit.  If `exog`
        is passed as an argument here, then any `exposure` and
        `offset` values in the fit will be ignored.

        Exposure values must be strictly positive.
        """

        # Use fit offset if appropriate
        if offset is None and exog is None and hasattr(self, 'offset'):
            offset = self.offset
        elif offset is None:
            offset = 0.

        if exposure is not None and not isinstance(self.family.link,
                                                   families.links.Log):
            raise ValueError("exposure can only be used with the log link "
                             "function")

        # Use fit exposure if appropriate
        if exposure is None and exog is None and hasattr(self, 'exposure'):
            # Already logged
            exposure = self.exposure
        elif exposure is None:
            exposure = 0.
        else:
            exposure = np.log(exposure)

        if exog is None:
            exog = self.exog

        linpred = np.dot(exog, params) + offset + exposure
        if linear:
            return linpred
        else:
            return self.family.fitted(linpred)

    def get_distribution(self, params, scale=1, exog=None, exposure=None,
                         offset=None):
        """
        Return a random number generator for the predictive distribution.

        Parameters
        ----------
        params : array_like
            The model parameters.
        scale : scalar
            The scale parameter.
        exog : array_like
            The predictor variable matrix.

        Returns
        -------
        gen
            Frozen random number generator object.  Use the ``rvs`` method to
            generate random values.

        Notes
        -----
        Due to the behavior of ``scipy.stats.distributions objects``, the
        returned random number generator must be called with ``gen.rvs(n)``
        where ``n`` is the number of observations in the data set used
        to fit the model.  If any other value is used for ``n``, misleading
        results will be produced.
        """

        fit = self.predict(params, exog, exposure, offset, linear=False)

        import scipy.stats.distributions as dist

        if isinstance(self.family, families.Gaussian):
            return dist.norm(loc=fit, scale=np.sqrt(scale))

        elif isinstance(self.family, families.Binomial):
            return dist.binom(n=1, p=fit)

        elif isinstance(self.family, families.Poisson):
            return dist.poisson(mu=fit)

        elif isinstance(self.family, families.Gamma):
            alpha = fit / float(scale)
            return dist.gamma(alpha, scale=scale)

        else:
            raise ValueError("get_distribution not implemented for %s" %
                             self.family.name)

    def _setup_binomial(self):
        # this checks what kind of data is given for Binomial.
        # family will need a reference to endog if this is to be removed from
        # preprocessing
        self.n_trials = np.ones((self.endog.shape[0]))  # For binomial
        if isinstance(self.family, families.Binomial):
            tmp = self.family.initialize(self.endog, self.freq_weights)
            self.endog = tmp[0]
            self.n_trials = tmp[1]
            self._init_keys.append('n_trials')

    def fit(self, start_params=None, maxiter=100, method='IRLS', tol=1e-8,
            scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None,
            full_output=True, disp=False, max_start_irls=3, **kwargs):
        """
        Fits a generalized linear model for a given family.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is family-specific and is given by the
            ``family.starting_mu(endog)``. If start_params is given then the
            initial mean will be calculated as ``np.dot(exog, start_params)``.
        maxiter : int, optional
            Default is 100.
        method : str
            Default is 'IRLS' for iteratively reweighted least squares.
            Otherwise gradient optimization is used.
        tol : float
            Convergence tolerance.  Default is 1e-8.
        scale : str or float, optional
            `scale` can be 'X2', 'dev', or a float
            The default value is None, which uses `X2` for Gamma, Gaussian,
            and Inverse Gaussian.
            `X2` is Pearson's chi-square divided by `df_resid`.
            The default is 1 for the Binomial and Poisson families.
            `dev` is the deviance divided by df_resid
        cov_type : str
            The type of parameter estimate covariance matrix to compute.
        cov_kwds : dict-like
            Extra arguments for calculating the covariance of the parameter
            estimates.
        use_t : bool
            If True, the Student t-distribution is used for inference.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
            Not used if methhod is IRLS.
        disp : bool, optional
            Set to True to print convergence messages.  Not used if method is
            IRLS.
        max_start_irls : int
            The number of IRLS iterations used to obtain starting
            values for gradient optimization.  Only relevant if
            `method` is set to something other than 'IRLS'.
        atol : float, optional
            (available with IRLS fits) The absolute tolerance criterion that
            must be satisfied. Defaults to ``tol``. Convergence is attained
            when: :math:`rtol * prior + atol > abs(current - prior)`
        rtol : float, optional
            (available with IRLS fits) The relative tolerance criterion that
            must be satisfied. Defaults to 0 which means ``rtol`` is not used.
            Convergence is attained when:
            :math:`rtol * prior + atol > abs(current - prior)`
        tol_criterion : str, optional
            (available with IRLS fits) Defaults to ``'deviance'``. Can
            optionally be ``'params'``.
        wls_method : str, optional
            (available with IRLS fits) options are 'lstsq', 'pinv' and 'qr'
            specifies which linear algebra function to use for the irls
            optimization. Default is `lstsq` which uses the same underlying
            svd based approach as 'pinv', but is faster during iterations.
            'lstsq' and 'pinv' regularize the estimate in singular and
            near-singular cases by truncating small singular values based
            on `rcond` of the respective numpy.linalg function. 'qr' is
            only valid for cases that are not singular nor near-singular.
        optim_hessian : {'eim', 'oim'}, optional
            (available with scipy optimizer fits) When 'oim'--the default--the
            observed Hessian is used in fitting. 'eim' is the expected Hessian.
            This may provide more stable fits, but adds assumption that the
            Hessian is correctly specified.

        Notes
        -----
        If method is 'IRLS', then an additional keyword 'attach_wls' is
        available. This is currently for internal use only and might change
        in future versions. If attach_wls' is true, then the final WLS
        instance of the IRLS iteration is attached to the results instance
        as `results_wls` attribute.
        """
        self.scaletype = scale

        if method.lower() == "irls":
            if cov_type.lower() == 'eim':
                cov_type = 'nonrobust'
            return self._fit_irls(start_params=start_params, maxiter=maxiter,
                                  tol=tol, scale=scale, cov_type=cov_type,
                                  cov_kwds=cov_kwds, use_t=use_t, **kwargs)
        else:
            self._optim_hessian = kwargs.get('optim_hessian')
            self._tmp_like_exog = np.empty_like(self.exog)
            fit_ = self._fit_gradient(start_params=start_params,
                                      method=method,
                                      maxiter=maxiter,
                                      tol=tol, scale=scale,
                                      full_output=full_output,
                                      disp=disp, cov_type=cov_type,
                                      cov_kwds=cov_kwds, use_t=use_t,
                                      max_start_irls=max_start_irls,
                                      **kwargs)
            del self._optim_hessian
            del self._tmp_like_exog
            return fit_

    def _fit_gradient(self, start_params=None, method="newton",
                      maxiter=100, tol=1e-8, full_output=True,
                      disp=True, scale=None, cov_type='nonrobust',
                      cov_kwds=None, use_t=None, max_start_irls=3,
                      **kwargs):
        """
        Fits a generalized linear model for a given family iteratively
        using the scipy gradient optimizers.
        """

        # fix scale during optimization, see #4616
        scaletype = self.scaletype
        self.scaletype = 1.

        if (max_start_irls > 0) and (start_params is None):
            irls_rslt = self._fit_irls(start_params=start_params,
                                       maxiter=max_start_irls,
                                       tol=tol, scale=1., cov_type='nonrobust',
                                       cov_kwds=None, use_t=None,
                                       **kwargs)
            start_params = irls_rslt.params
            del irls_rslt

        rslt = super(GLM, self).fit(start_params=start_params, tol=tol,
                                    maxiter=maxiter, full_output=full_output,
                                    method=method, disp=disp, **kwargs)

        # reset scaletype to original
        self.scaletype = scaletype

        mu = self.predict(rslt.params)
        scale = self.estimate_scale(mu)

        if rslt.normalized_cov_params is None:
            cov_p = None
        else:
            cov_p = rslt.normalized_cov_params / scale

        if cov_type.lower() == 'eim':
            oim = False
            cov_type = 'nonrobust'
        else:
            oim = True

        try:
            cov_p = np.linalg.inv(-self.hessian(rslt.params, observed=oim)) / scale
        except LinAlgError:
            from warnings import warn
            warn('Inverting hessian failed, no bse or cov_params '
                 'available', HessianInversionWarning)
            cov_p = None

        results_class = getattr(self, '_results_class', GLMResults)
        results_class_wrapper = getattr(self, '_results_class_wrapper', GLMResultsWrapper)
        glm_results = results_class(self, rslt.params,
                                    cov_p,
                                    scale,
                                    cov_type=cov_type, cov_kwds=cov_kwds,
                                    use_t=use_t)

        # TODO: iteration count is not always available
        history = {'iteration': 0}
        if full_output:
            glm_results.mle_retvals = rslt.mle_retvals
            if 'iterations' in rslt.mle_retvals:
                history['iteration'] = rslt.mle_retvals['iterations']
        glm_results.method = method
        glm_results.fit_history = history

        return results_class_wrapper(glm_results)

    def _fit_irls(self, start_params=None, maxiter=100, tol=1e-8,
                  scale=None, cov_type='nonrobust', cov_kwds=None,
                  use_t=None, **kwargs):
        """
        Fits a generalized linear model for a given family using
        iteratively reweighted least squares (IRLS).
        """
        attach_wls = kwargs.pop('attach_wls', False)
        atol = kwargs.get('atol')
        rtol = kwargs.get('rtol', 0.)
        tol_criterion = kwargs.get('tol_criterion', 'deviance')
        wls_method = kwargs.get('wls_method', 'lstsq')
        atol = tol if atol is None else atol

        endog = self.endog
        wlsexog = self.exog
        if start_params is None:
            start_params = np.zeros(self.exog.shape[1], np.float)
            mu = self.family.starting_mu(self.endog)
            lin_pred = self.family.predict(mu)
        else:
            lin_pred = np.dot(wlsexog, start_params) + self._offset_exposure
            mu = self.family.fitted(lin_pred)
        self.scale = self.estimate_scale(mu)
        dev = self.family.deviance(self.endog, mu, self.var_weights,
                                   self.freq_weights, self.scale)
        if np.isnan(dev):
            raise ValueError("The first guess on the deviance function "
                             "returned a nan.  This could be a boundary "
                             " problem and should be reported.")

        # first guess on the deviance is assumed to be scaled by 1.
        # params are none to start, so they line up with the deviance
        history = dict(params=[np.inf, start_params], deviance=[np.inf, dev])
        converged = False
        criterion = history[tol_criterion]
        # This special case is used to get the likelihood for a specific
        # params vector.
        if maxiter == 0:
            mu = self.family.fitted(lin_pred)
            self.scale = self.estimate_scale(mu)
            wls_results = lm.RegressionResults(self, start_params, None)
            iteration = 0
        for iteration in range(maxiter):
            self.weights = (self.iweights * self.n_trials *
                            self.family.weights(mu))
            wlsendog = (lin_pred + self.family.link.deriv(mu) * (self.endog-mu)
                        - self._offset_exposure)
            wls_mod = reg_tools._MinimalWLS(wlsendog, wlsexog,
                                            self.weights, check_endog=True,
                                            check_weights=True)
            wls_results = wls_mod.fit(method=wls_method)
            lin_pred = np.dot(self.exog, wls_results.params)
            lin_pred += self._offset_exposure
            mu = self.family.fitted(lin_pred)
            history = self._update_history(wls_results, mu, history)
            self.scale = self.estimate_scale(mu)
            if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
                msg = "Perfect separation detected, results not available"
                raise PerfectSeparationError(msg)
            converged = _check_convergence(criterion, iteration + 1, atol,
                                           rtol)
            if converged:
                break
        self.mu = mu

        if maxiter > 0:  # Only if iterative used
            wls_method2 = 'pinv' if wls_method == 'lstsq' else wls_method
            wls_model = lm.WLS(wlsendog, wlsexog, self.weights)
            wls_results = wls_model.fit(method=wls_method2)

        glm_results = GLMResults(self, wls_results.params,
                                 wls_results.normalized_cov_params,
                                 self.scale,
                                 cov_type=cov_type, cov_kwds=cov_kwds,
                                 use_t=use_t)

        glm_results.method = "IRLS"
        glm_results.mle_settings = {}
        glm_results.mle_settings['wls_method'] = wls_method
        glm_results.mle_settings['optimizer'] = glm_results.method
        if (maxiter > 0) and (attach_wls is True):
            glm_results.results_wls = wls_results
        history['iteration'] = iteration + 1
        glm_results.fit_history = history
        glm_results.converged = converged
        return GLMResultsWrapper(glm_results)

    def fit_regularized(self, method="elastic_net", alpha=0.,
                        start_params=None, refit=False, **kwargs):
        r"""
        Return a regularized fit to a linear regression model.

        Parameters
        ----------
        method : {'elastic_net'}
            Only the `elastic_net` approach is currently implemented.
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.
        start_params : array_like
            Starting values for `params`.
        refit : bool
            If True, the model is refit using only the variables that
            have non-zero coefficients in the regularized fit.  The
            refitted model is not regularized.
        **kwargs
            Additional keyword arguments used when fitting the model.

        Returns
        -------
        GLMResults
            An array or a GLMResults object, same type returned by `fit`.

        Notes
        -----
        The penalty is the ``elastic net`` penalty, which is a
        combination of L1 and L2 penalties.

        The function that is minimized is:

        .. math::

            -loglike/n + alpha*((1-L1\_wt)*|params|_2^2/2 + L1\_wt*|params|_1)

        where :math:`|*|_1` and :math:`|*|_2` are the L1 and L2 norms.

        Post-estimation results are based on the same data used to
        select variables, hence may be subject to overfitting biases.

        The elastic_net method uses the following keyword arguments:

        maxiter : int
            Maximum number of iterations
        L1_wt  : float
            Must be in [0, 1].  The L1 penalty has weight L1_wt and the
            L2 penalty has weight 1 - L1_wt.
        cnvrg_tol : float
            Convergence threshold for line searches
        zero_tol : float
            Coefficients below this threshold are treated as zero.
        """

        if kwargs.get("L1_wt", 1) == 0:
            return self._fit_ridge(alpha, start_params)

        from statsmodels.base.elastic_net import fit_elasticnet

        if method != "elastic_net":
            raise ValueError("method for fit_regularied must be elastic_net")

        defaults = {"maxiter": 50, "L1_wt": 1, "cnvrg_tol": 1e-10,
                    "zero_tol": 1e-10}
        defaults.update(kwargs)

        result = fit_elasticnet(self, method=method,
                                alpha=alpha,
                                start_params=start_params,
                                refit=refit,
                                **defaults)

        self.mu = self.predict(result.params)
        self.scale = self.estimate_scale(self.mu)

        return result

    def _fit_ridge(self, alpha, start_params, method="newton-cg"):

        if start_params is None:
            start_params = np.zeros(self.exog.shape[1])

        def fun(x):
            return -(self.loglike(x) / self.nobs - np.sum(alpha * x**2) / 2)

        def grad(x):
            return -(self.score(x) / self.nobs - alpha * x)

        from scipy.optimize import minimize
        from statsmodels.base.elastic_net import (RegularizedResults,
            RegularizedResultsWrapper)

        mr = minimize(fun, start_params, jac=grad, method=method)
        params = mr.x

        if not mr.success:
            import warnings
            ngrad = np.sqrt(np.sum(mr.jac**2))
            msg = "GLM ridge optimization may have failed, |grad|=%f" % ngrad
            warnings.warn(msg)

        results = RegularizedResults(self, params)
        results = RegularizedResultsWrapper(results)

        return results

    def fit_constrained(self, constraints, start_params=None, **fit_kwds):
        """fit the model subject to linear equality constraints

        The constraints are of the form   `R params = q`
        where R is the constraint_matrix and q is the vector of
        constraint_values.

        The estimation creates a new model with transformed design matrix,
        exog, and converts the results back to the original parameterization.


        Parameters
        ----------
        constraints : formula expression or tuple
            If it is a tuple, then the constraint needs to be given by two
            arrays (constraint_matrix, constraint_value), i.e. (R, q).
            Otherwise, the constraints can be given as strings or list of
            strings.
            see t_test for details
        start_params : None or array_like
            starting values for the optimization. `start_params` needs to be
            given in the original parameter space and are internally
            transformed.
        **fit_kwds : keyword arguments
            fit_kwds are used in the optimization of the transformed model.

        Returns
        -------
        results : Results instance
        """

        from patsy import DesignInfo
        from statsmodels.base._constraints import fit_constrained

        # same pattern as in base.LikelihoodModel.t_test
        lc = DesignInfo(self.exog_names).linear_constraint(constraints)
        R, q = lc.coefs, lc.constants

        # TODO: add start_params option, need access to tranformation
        #       fit_constrained needs to do the transformation
        params, cov, res_constr = fit_constrained(self, R, q,
                                                  start_params=start_params,
                                                  fit_kwds=fit_kwds)
        # create dummy results Instance, TODO: wire up properly
        res = self.fit(start_params=params, maxiter=0)  # we get a wrapper back
        res._results.params = params
        res._results.cov_params_default = cov
        cov_type = fit_kwds.get('cov_type', 'nonrobust')
        if cov_type != 'nonrobust':
            res._results.normalized_cov_params = cov / res_constr.scale
        else:
            res._results.normalized_cov_params = None
        res._results.scale = res_constr.scale
        k_constr = len(q)
        res._results.df_resid += k_constr
        res._results.df_model -= k_constr
        res._results.constraints = lc
        res._results.k_constr = k_constr
        res._results.results_constrained = res_constr
        return res


class GLMResults(base.LikelihoodModelResults):
    """
    Class to contain GLM results.

    GLMResults inherits from statsmodels.LikelihoodModelResults

    Attributes
    ----------
    df_model : float
        See GLM.df_model
    df_resid : float
        See GLM.df_resid
    fit_history : dict
        Contains information about the iterations. Its keys are `iterations`,
        `deviance` and `params`.
    model : class instance
        Pointer to GLM model instance that called fit.
    nobs : float
        The number of observations n.
    normalized_cov_params : ndarray
        See GLM docstring
    params : ndarray
        The coefficients of the fitted model.  Note that interpretation
        of the coefficients often depends on the distribution family and the
        data.
    pvalues : ndarray
        The two-tailed p-values for the parameters.
    scale : float
        The estimate of the scale / dispersion for the model fit.
        See GLM.fit and GLM.estimate_scale for more information.
    stand_errors : ndarray
        The standard errors of the fitted GLM.   #TODO still named bse

    See Also
    --------
    statsmodels.base.model.LikelihoodModelResults
    """

    def __init__(self, model, params, normalized_cov_params, scale,
                 cov_type='nonrobust', cov_kwds=None, use_t=None):
        super(GLMResults, self).__init__(
                model,
                params,
                normalized_cov_params=normalized_cov_params,
                scale=scale)
        self.family = model.family
        self._endog = model.endog
        self.nobs = model.endog.shape[0]
        self._freq_weights = model.freq_weights
        self._var_weights = model.var_weights
        self._iweights = model.iweights
        if isinstance(self.family, families.Binomial):
            self._n_trials = self.model.n_trials
        else:
            self._n_trials = 1
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        self._cache = {}
        # are these intermediate results needed or can we just
        # call the model's attributes?

        # for remove data and pickle without large arrays
        self._data_attr.extend(['results_constrained', '_freq_weights',
                                '_var_weights', '_iweights'])
        self.data_in_cache = getattr(self, 'data_in_cache', [])
        self.data_in_cache.extend(['null', 'mu'])
        self._data_attr_model = getattr(self, '_data_attr_model', [])
        self._data_attr_model.append('mu')

        # robust covariance
        from statsmodels.base.covtype import get_robustcov_results
        if use_t is None:
            self.use_t = False    # TODO: class default
        else:
            self.use_t = use_t

        # temporary warning
        ct = (cov_type == 'nonrobust') or (cov_type.upper().startswith('HC'))
        if self.model._has_freq_weights and not ct:
            import warnings
            from statsmodels.tools.sm_exceptions import SpecificationWarning
            warnings.warn('cov_type not fully supported with freq_weights',
                          SpecificationWarning)

        if self.model._has_var_weights and not ct:
            import warnings
            from statsmodels.tools.sm_exceptions import SpecificationWarning
            warnings.warn('cov_type not fully supported with var_weights',
                          SpecificationWarning)

        if cov_type == 'nonrobust':
            self.cov_type = 'nonrobust'
            self.cov_kwds = {'description': 'Standard Errors assume that the' +
                             ' covariance matrix of the errors is correctly ' +
                             'specified.'}

        else:
            if cov_kwds is None:
                cov_kwds = {}
            get_robustcov_results(self, cov_type=cov_type, use_self=True,
                                  use_t=use_t, **cov_kwds)

    @cached_data
    def resid_response(self):
        """
        Respnose residuals.  The response residuals are defined as
        `endog` - `fittedvalues`
        """
        return self._n_trials * (self._endog-self.mu)

    @cached_data
    def resid_pearson(self):
        """
        Pearson residuals.  The Pearson residuals are defined as
        (`endog` - `mu`)/sqrt(VAR(`mu`)) where VAR is the distribution
        specific variance function.  See statsmodels.families.family and
        statsmodels.families.varfuncs for more information.
        """
        return (np.sqrt(self._n_trials) * (self._endog-self.mu) *
                np.sqrt(self._var_weights) /
                np.sqrt(self.family.variance(self.mu)))

    @cached_data
    def resid_working(self):
        """
        Working residuals.  The working residuals are defined as
        `resid_response`/link'(`mu`).  See statsmodels.family.links for the
        derivatives of the link functions.  They are defined analytically.
        """
        # Isn't self.resid_response is already adjusted by _n_trials?
        val = (self.resid_response * self.family.link.deriv(self.mu))
        val *= self._n_trials
        return val

    @cached_data
    def resid_anscombe(self):
        """
        Anscombe residuals.  See statsmodels.families.family for distribution-
        specific Anscombe residuals. Currently, the unscaled residuals are
        provided. In a future version, the scaled residuals will be provided.
        """
        import warnings
        warnings.warn('Anscombe residuals currently unscaled. In a future '
                      'release, they will be scaled.', category=FutureWarning)
        return self.family.resid_anscombe(self._endog, self.fittedvalues,
                                          var_weights=self._var_weights,
                                          scale=1.)

    @cached_data
    def resid_anscombe_scaled(self):
        """
        Scaled Anscombe residuals.  See statsmodels.families.family for
        distribution-specific Anscombe residuals.
        """
        return self.family.resid_anscombe(self._endog, self.fittedvalues,
                                          var_weights=self._var_weights,
                                          scale=self.scale)

    @cached_data
    def resid_anscombe_unscaled(self):
        """
        Unscaled Anscombe residuals.  See statsmodels.families.family for
        distribution-specific Anscombe residuals.
        """
        return self.family.resid_anscombe(self._endog, self.fittedvalues,
                                          var_weights=self._var_weights,
                                          scale=1.)

    @cached_data
    def resid_deviance(self):
        """
        Deviance residuals.  See statsmodels.families.family for distribution-
        specific deviance residuals.
        """
        dev = self.family.resid_dev(self._endog, self.fittedvalues,
                                    var_weights=self._var_weights,
                                    scale=1.)
        return dev

    @cached_value
    def pearson_chi2(self):
        """
        Pearson's Chi-Squared statistic is defined as the sum of the squares
        of the Pearson residuals.
        """
        chisq = (self._endog - self.mu)**2 / self.family.variance(self.mu)
        chisq *= self._iweights * self._n_trials
        chisqsum = np.sum(chisq)
        return chisqsum

    @cached_data
    def fittedvalues(self):
        """
        Linear predicted values for the fitted model.
        dot(exog, params)
        """
        return self.mu

    @cached_data
    def mu(self):
        """
        See GLM docstring.
        """
        return self.model.predict(self.params)

    @cache_readonly
    def null(self):
        """
        Fitted values of the null model
        """
        endog = self._endog
        model = self.model
        exog = np.ones((len(endog), 1))

        kwargs = model._get_init_kwds()
        kwargs.pop('family')
        if hasattr(self, '_offset_exposure'):
            return GLM(endog, exog, family=self.family,
                       **kwargs).fit().fittedvalues
        else:
            # correct if fitted is identical across observations
            wls_model = lm.WLS(endog, exog,
                               weights=self._iweights * self._n_trials)
            return wls_model.fit().fittedvalues

    @cache_readonly
    def deviance(self):
        """
        See statsmodels.families.family for the distribution-specific deviance
        functions.
        """
        return self.family.deviance(self._endog, self.mu, self._var_weights,
                                    self._freq_weights)

    @cache_readonly
    def null_deviance(self):
        """The value of the deviance function for the model fit with a constant
        as the only regressor."""
        return self.family.deviance(self._endog, self.null, self._var_weights,
                                    self._freq_weights)

    @cache_readonly
    def llnull(self):
        """
        Log-likelihood of the model fit with a constant as the only regressor
        """
        return self.family.loglike(self._endog, self.null,
                                   var_weights=self._var_weights,
                                   freq_weights=self._freq_weights,
                                   scale=self.scale)

    @cached_value
    def llf(self):
        """
        Value of the loglikelihood function evalued at params.
        See statsmodels.families.family for distribution-specific
        loglikelihoods.
        """
        _modelfamily = self.family
        if (isinstance(self.family, families.Gaussian) and
                isinstance(self.family.link, families.links.Power) and
                (self.family.link.power == 1.)):
            scale = (np.power(self._endog - self.mu, 2) * self._iweights).sum()
            scale /= self.model.wnobs
        else:
            scale = self.scale
        val = _modelfamily.loglike(self._endog, self.mu,
                                   var_weights=self._var_weights,
                                   freq_weights=self._freq_weights,
                                   scale=scale)
        return val

    @cached_value
    def aic(self):
        """
        Akaike Information Criterion
        -2 * `llf` + 2*(`df_model` + 1)
        """
        return -2 * self.llf + 2 * (self.df_model + 1)

    @cached_value
    def bic(self):
        """
        Bayes Information Criterion
        `deviance` - `df_resid` * log(`nobs`)
        """
        return (self.deviance -
                (self.model.wnobs - self.df_model - 1) *
                np.log(self.model.wnobs))

    @Appender(pred.get_prediction_glm.__doc__)
    def get_prediction(self, exog=None, exposure=None, offset=None,
                       transform=True, linear=False,
                       row_labels=None):

        import statsmodels.regression._prediction as linpred

        pred_kwds = {'exposure': exposure, 'offset': offset, 'linear': True}

        # two calls to a get_prediction duplicates exog generation if patsy
        res_linpred = linpred.get_prediction(self, exog=exog,
                                             transform=transform,
                                             row_labels=row_labels,
                                             pred_kwds=pred_kwds)

        pred_kwds['linear'] = False
        res = pred.get_prediction_glm(self, exog=exog, transform=transform,
                                      row_labels=row_labels,
                                      linpred=res_linpred,
                                      link=self.model.family.link,
                                      pred_kwds=pred_kwds)

        return res

    def get_hat_matrix_diag(self, observed=True):
        """
        Compute the diagonal of the hat matrix

        Parameters
        ----------
        observed : bool
            If true, then observed hessian is used in the hat matrix
            computation. If false, then the expected hessian is used.
            In the case of a canonical link function both are the same.

        Returns
        -------
        hat_matrix_diag : ndarray
            The diagonal of the hat matrix computed from the observed
            or expected hessian.
        """
        weights = self.model.hessian_factor(self.params, observed=observed)
        wexog = np.sqrt(weights)[:, None] * self.model.exog

        hd = (wexog * np.linalg.pinv(wexog).T).sum(1)
        return hd

    def get_influence(self, observed=True):
        """
        Get an instance of GLMInfluence with influence and outlier measures

        Parameters
        ----------
        observed : bool
            If true, then observed hessian is used in the hat matrix
            computation. If false, then the expected hessian is used.
            In the case of a canonical link function both are the same.

        Returns
        -------
        infl : GLMInfluence instance
            The instance has methods to calculate the main influence and
            outlier measures as attributes.

        See Also
        --------
        statsmodels.stats.outliers_influence.GLMInfluence
        """
        from statsmodels.stats.outliers_influence import GLMInfluence

        weights = self.model.hessian_factor(self.params, observed=observed)
        weights_sqrt = np.sqrt(weights)
        wexog = weights_sqrt[:, None] * self.model.exog
        wendog = weights_sqrt * self.model.endog

        # using get_hat_matrix_diag has duplicated computation
        hat_matrix_diag = self.get_hat_matrix_diag(observed=observed)
        infl = GLMInfluence(self, endog=wendog, exog=wexog,
                         resid=self.resid_pearson,
                         hat_matrix_diag=hat_matrix_diag)
        return infl

    @Appender(base.LikelihoodModelResults.remove_data.__doc__)
    def remove_data(self):
        # GLM has alias/reference in result instance
        self._data_attr.extend([i for i in self.model._data_attr
                                if '_data.' not in i])
        super(self.__class__, self).remove_data()

        # TODO: what are these in results?
        self._endog = None
        self._freq_weights = None
        self._var_weights = None
        self._iweights = None
        self._n_trials = None

    @Appender(_plot_added_variable_doc % {'extra_params_doc': ''})
    def plot_added_variable(self, focus_exog, resid_type=None,
                            use_glm_weights=True, fit_kwargs=None,
                            ax=None):

        from statsmodels.graphics.regressionplots import plot_added_variable

        fig = plot_added_variable(self, focus_exog,
                                  resid_type=resid_type,
                                  use_glm_weights=use_glm_weights,
                                  fit_kwargs=fit_kwargs, ax=ax)

        return fig

    @Appender(_plot_partial_residuals_doc % {'extra_params_doc': ''})
    def plot_partial_residuals(self, focus_exog, ax=None):

        from statsmodels.graphics.regressionplots import plot_partial_residuals

        return plot_partial_residuals(self, focus_exog, ax=ax)

    @Appender(_plot_ceres_residuals_doc % {'extra_params_doc': ''})
    def plot_ceres_residuals(self, focus_exog, frac=0.66, cond_means=None,
                             ax=None):

        from statsmodels.graphics.regressionplots import plot_ceres_residuals

        return plot_ceres_residuals(self, focus_exog, frac,
                                    cond_means=cond_means, ax=ax)

    def summary(self, yname=None, xname=None, title=None, alpha=.05):
        """
        Summarize the Regression Results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is `var_#` for ## in
            the number of regressors. Must match the number of parameters in
            the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary results
        """

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Model Family:', [self.family.__class__.__name__]),
                    ('Link Function:', [self.family.link.__class__.__name__]),
                    ('Method:', [self.method]),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Iterations:',
                     ["%d" % self.fit_history['iteration']]),
                    ]

        top_right = [('No. Observations:', None),
                     ('Df Residuals:', None),
                     ('Df Model:', None),
                     ('Scale:', ["%#8.5g" % self.scale]),
                     ('Log-Likelihood:', None),
                     ('Deviance:', ["%#8.5g" % self.deviance]),
                     ('Pearson chi2:', ["%#6.3g" % self.pearson_chi2])
                     ]

        if hasattr(self, 'cov_type'):
            top_left.append(('Covariance Type:', [self.cov_type]))

        if title is None:
            title = "Generalized Linear Model Regression Results"

        # create summary tables
        from statsmodels.iolib.summary import Summary
        smry = Summary()
        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                             yname=yname, xname=xname, title=title)
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha,
                              use_t=self.use_t)

        if hasattr(self, 'constraints'):
            smry.add_extra_txt(['Model has been estimated subject to linear '
                                'equality constraints.'])
        return smry

    def summary2(self, yname=None, xname=None, title=None, alpha=.05,
                 float_format="%.4f"):
        """Experimental summary for regression Results

        Parameters
        ----------
        yname : str
            Name of the dependent variable (optional)
        xname : list[str], optional
            Names for the exogenous variables, default is `var_#` for ## in
            the number of regressors. Must match the number of parameters in
            the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals
        float_format : str
            print format for floats in parameters summary

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary : class to hold summary results
        """
        self.method = 'IRLS'
        from statsmodels.iolib import summary2
        smry = summary2.Summary()
        smry.add_base(results=self, alpha=alpha, float_format=float_format,
                      xname=xname, yname=yname, title=title)
        if hasattr(self, 'constraints'):
            smry.add_text('Model has been estimated subject to linear '
                          'equality constraints.')

        return smry


class GLMResultsWrapper(lm.RegressionResultsWrapper):
    _attrs = {
        'resid_anscombe': 'rows',
        'resid_deviance': 'rows',
        'resid_pearson': 'rows',
        'resid_response': 'rows',
        'resid_working': 'rows'
    }
    _wrap_attrs = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_attrs,
                                   _attrs)


wrap.populate_wrapper(GLMResultsWrapper, GLMResults)

if __name__ == "__main__":
    import statsmodels.api as sm
    data = sm.datasets.longley.load(as_pandas=False)
    # data.exog = add_constant(data.exog)
    GLMmod = GLM(data.endog, data.exog).fit()
    GLMT = GLMmod.summary(returns='tables')
    # GLMT[0].extend_right(GLMT[1])
    # print(GLMT[0])
    # print(GLMT[2])
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
