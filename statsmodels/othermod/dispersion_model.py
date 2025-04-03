# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 08:29:59 2021

Author: Josef Perktod
License: BSD-3
"""

import numpy as np
from scipy import special, stats

from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.genmod import families


FLOAT_EPS = np.finfo(float).eps

# define some shortcuts
np_log = np.log
np_pi = np.pi
sps_gamln = special.gammaln


class MultiLinkModel(GenericLikelihoodModel):
    '''Maximum Likelihood Estimation of Model with multiple sets of regressors.

    This class models location or mean, scale or dispersion and optional
    extra distribution parameters, where each can have explanatory variables
    with link functions.

    Parameters
    ----------
    endog : array_like
        1d array of endogenous response variable.
    exog : array_like
        Array of explanatory variables for first distribution parameter.
    exog_scale : array_like or None
        Array of explanatory variables for second distribution parameter.
    exog_extras : list or tuple or None
        List of array of explanatory variables for other distribution
        parameters.
    link : instance of link class
        Link for first distribution parameter.
        Default to identity link, ``families.links.identity()``.
    link_scale : instance of link class
        Link for second distribution parameter.
        Default is ``families.links.Log()``.
    link_extras : list or None
        List of link instances for extra parameters.
        Default to identity link for each extra parameter.
    k_extra : int
        Number of extra distribution parameters, in addition to the first two.
        Required argument in this class.

    '''
    def __init__(self, endog, exog, exog_scale=None,
                 exog_extras=None, link=None,
                 link_scale=families.links.Log(),
                 link_extras=None, k_extra=None, **kwds):

        # etmp = np.array(endog)
        if k_extra is None:
            raise ValueError("k_extra is required")
        self.k_extra = k_extra

        # base datahandling only adds names for one exog
        if exog_scale is None:
            extra_names = ['scale']
            exog_scale = np.ones((len(endog), 1), dtype='f')
        else:
            extra_names = ['scale-%s' % zc for zc in
                           (exog_scale.columns
                            if hasattr(exog_scale, 'columns')
                            else range(1, exog_scale.shape[1] + 1))]

        # handle extras
        self.nobs = endog.shape[0]  # no self.endog yet
        self.k_params_li = [exog.shape[1], exog_scale.shape[1]]
        if exog_extras is None:
            self.k_params_li.extend([1] * (k_extra))
        else:
            for i in range(k_extra):
                if exog_extras[i] is None:
                    self.k_params_li.append(1)
                    exog_extras[i] = np.ones((self.nobs, 1))
                else:
                    if exog_extras[i].ndim == 1:
                        exog_extras[i] = exog_extras[i][:, None]
                    if exog_extras[i].ndim > 2:
                        raise ValueError("exog_extras has more than 2 dim")
                    self.k_params_li.append(exog_extras[i].shape[1])

        self.k_params_cumli = np.cumsum(self.k_params_li).tolist()
        self.exog_extras = exog_extras

        for i in range(self.k_extra):
            extra_names.extend(['a%i-%s' % (i, zc)
                                for zc in range(self.k_params_li[2 + i])])
        kwds['extra_params_names'] = extra_names
        # 'extra_params_names' will be attached by super

        super().__init__(endog, exog, exog_scale=exog_scale,
                         **kwds)

        if link is None:
            link = families.links.identity()
        self.link = link
        self.link_scale = link_scale
        if link_extras is None and self.k_extra > 0:
            link_extras = [families.links.identity()
                           for _ in range(self.k_extra)]
        self.link_extras = link_extras
        # not needed, handled by super:
        # self.exog_scale = exog_scale

        self.df_model = self.nparams - 1
        self.df_resid = self.nobs - self.nparams
        # need to fix, used for start_params,
        # self.k_vars = self.exog.shape[1] + self.exog_scale.shape[1]
        assert len(self.exog_scale) == len(self.endog)
        self.hess_type = "oim"
        if 'exog_scale' not in self._init_keys:
            self._init_keys.extend(['exog_scale'])

        # todo: maybe not here
        self._set_start_params()
        self._init_keys.extend(['link', 'link_scale', 'link_extras'])
        self._null_drop_keys = ['exog_scale', 'exog_extras']
        # self.results_class = BetaResults
        # self.results_class_wrapper = BetaResultsWrapper

    def initialize(self):
        # TODO: here or in __init__
        self.k_vars = self.exog.shape[1]
        self.k_params = (self.exog.shape[1] + self.exog_scale.shape[1] +
                         self.k_extra)
        self.fixed_params = None
        super().initialize()

    def _split_params(self, params):
        return np.split(params, self.k_params_cumli[:-1])

    # todo use propertie for start_params
    def _set_start_params(self, start_params=None, use_kurtosis=False):
        if start_params is not None:
            self.start_params = start_params
        else:
            from statsmodels.regression.linear_model import OLS
            res_ols = OLS(self.endog, self.exog).fit()
            start_params = 0.1*np.ones(self.k_params)
            start_params[:self.k_vars] = res_ols.params

            # Here we only use constant, missing link
            # TODO use regression
            # using link makes convergence slower in the examples
            start_params[self.exog.shape[1]] = self.link_scale(res_ols.scale)

            self.start_params = start_params

    def _predict_dargs(self, params):
        """Predict distribution parameters (args) for sample given params.

        Parameters
        ----------
        params : ndarray
            The parameters of the model. The last 2 parameters are degrees of
            freedom and scale.

        Returns
        -------
        tuple : Parameters for distribution used in `_loglikeobs`.
        """
        k_mean = self.exog.shape[1]
        k_scale = self.exog_scale.shape[1]
        p_split = self._split_params(params)
        params_loc = p_split[0]
        linpred_loc = np.dot(self.exog, params_loc)
        loc = self.link.inverse(linpred_loc)

        params_scale = p_split[1]
        linpred_scale = np.dot(self.exog_scale, params_scale)
        scale = self.link_scale.inverse(linpred_scale)

        args = [loc, scale]

        if self.exog_extras is None:
            args.extend([i for i in params[k_mean + k_scale:]])
        else:
            for i in range(self.k_extra):
                linpred = self.exog_extras[i].dot(p_split[2 + i])
                args.append(self.link_extras[i].inverse(linpred))
                if linpred.ndim > 1:
                    raise

        return tuple(args)

    def loglike(self, params):
        return self.loglikeobs(params).sum(0)

    def _loglikeobs(self, mu, scale, *args, endog=None):
        raise NotImplementedError

    def loglikeobs(self, params):
        """
        Loglikelihood of linear model with t distributed errors.

        Parameters
        ----------
        params : ndarray
            The parameters of the model.

        Returns
        -------
        loglike : ndarray
            The log likelihood of the model evaluated at `params` for each
            observation defined by self.endog and self.exog.

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[... \\right]

        """
        # print len(params),
        # store_params.append(params)
        if self.fixed_params is not None:
            # print 'using fixed'
            params = self.expandparams(params)

        args = self._predict_dargs(params)
        # endog = self.endog
        ll_obs = self._loglikeobs(*args, endog=self.endog)
        return ll_obs

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        linpred = np.dot(exog, params[:self.exog.shape[1]])
        m = self.link.inverse(linpred)
        return m


class GaussianMultiLink(MultiLinkModel):

    def _loglikeobs(self, mu, scale, endog=None):
        ll_obs = -(endog - mu) ** 2 / scale
        ll_obs += -np.log(scale) - np.log(2 * np.pi)
        ll_obs /= 2
        return ll_obs


class Johnsonsu(MultiLinkModel):

    def _loglikeobs(self, mu, scale, *args, endog=None):
        if self.k_extra == 0:
            ll_obs = stats.johnsonsu.logpdf(endog, 0, 1, mu, scale)
        else:
            ll_obs = stats.johnsonsu.logpdf(endog, *args, mu, scale)
        return ll_obs


class Het2pModel(GenericLikelihoodModel):
    '''Maximum Likelihood Estimation of Linear Model with t-distributed errors

    This is an example for generic MLE.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    '''
    def __init__(self, endog, exog, exog_scale=None,
                 link_scale=families.links.Log(), **kwds):

        # etmp = np.array(endog)

        if exog_scale is None:
            extra_names = ['scale']
            exog_scale = np.ones((len(endog), 1), dtype='f')
        else:
            extra_names = ['scale-%s' % zc for zc in
                           (exog_scale.columns
                            if hasattr(exog_scale, 'columns')
                            else range(1, exog_scale.shape[1] + 1))]

        kwds['extra_params_names'] = extra_names

        super().__init__(endog, exog, exog_scale=exog_scale,
                         **kwds)
        # self.link = link
        self.link_scale = link_scale
        # not needed, handled by super:
        # self.exog_scale = exog_scale
        # inherited df do not account for precision params
        self.nobs = self.endog.shape[0]
        self.df_model = self.nparams - 1
        self.df_resid = self.nobs - self.nparams
        # need to fix, used for start_params,
        # self.k_vars = self.exog.shape[1] + self.exog_scale.shape[1]
        assert len(self.exog_scale) == len(self.endog)
        self.hess_type = "oim"
        if 'exog_scale' not in self._init_keys:
            self._init_keys.extend(['exog_scale'])

        # todo: maybe not here
        self._set_start_params()
        self._init_keys.extend(['link_scale'])
        self._null_drop_keys = ['exog_scale']
        # self.results_class = BetaResults
        # self.results_class_wrapper = BetaResultsWrapper

    def initialize(self):
        # TODO: here or in __init__
        self.k_vars = self.exog.shape[1]
        self.k_params = self.exog.shape[1] + self.exog_scale.shape[1]
        self.fixed_params = None
        super().initialize()

    # todo use propertie for start_params
    def _set_start_params(self, start_params=None, use_kurtosis=False):
        if start_params is not None:
            self.start_params = start_params
        else:
            from statsmodels.regression.linear_model import OLS
            res_ols = OLS(self.endog, self.exog).fit()
            start_params = 0.1*np.ones(self.k_params)
            start_params[:self.k_vars] = res_ols.params

            # Here we only use constant, missing link
            # TODO use regression
            # using link makes convergence slower in the examples
            start_params[self.exog.shape[1]] = self.link_scale(res_ols.scale)

            self.start_params = start_params

    def _predict_locscale(self, params):
        k_mean = self.exog.shape[1]
        k_scale = self.exog_scale.shape[1]
        beta = params[:k_mean]
        loc = np.dot(self.exog, beta)

        params_scale = params[k_mean : k_mean + k_scale]  # noqa: E203
        linpred_scale = np.dot(self.exog_scale, params_scale)
        scale = self.link_scale.inverse(linpred_scale)

        return loc, scale

    def loglike(self, params):
        return self.loglikeobs(params).sum(0)

    def _loglikeobs(self, mu, scale, endog=None):
        raise NotImplementedError

    def loglikeobs(self, params):
        """
        Loglikelihood of linear model with t distributed errors.

        Parameters
        ----------
        params : ndarray
            The parameters of the model. The last 2 parameters are degrees of
            freedom and scale.

        Returns
        -------
        loglike : ndarray
            The log likelihood of the model evaluated at `params` for each
            observation defined by self.endog and self.exog.

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}...

        """
        # print len(params),
        # store_params.append(params)
        if self.fixed_params is not None:
            # print 'using fixed'
            params = self.expandparams(params)

        loc, scale = self._predict_locscale(params)
        # endog = self.endog
        ll_obs = self._loglikeobs(loc, scale, endog=self.endog)
        return ll_obs

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return np.dot(exog, params[:self.exog.shape[1]])


class GaussianHet(Het2pModel):

    def _loglikeobs(self, mu, scale, endog=None):
        ll_obs = -(endog - mu) ** 2 / scale
        ll_obs += -np.log(scale) - np.log(2 * np.pi)
        ll_obs /= 2
        return ll_obs


class GammaHet(Het2pModel):

    def _clean(self, x):
        """
        Helper function to trim the data so that it is in (0,inf)

        Notes
        -----
        The need for this function was discovered through usage and its
        possible that other families might need a check for validity of the
        domain.
        """
        return np.clip(x, FLOAT_EPS, np.inf)

    def _loglikeobs(self, mu, scale, endog=None):
        endog_mu = self._clean(endog / mu)
        # weight_scale = var_weights / scale
        ll_obs = (np.log(endog_mu / scale) - endog_mu) / scale
        ll_obs -= special.gammaln(1 / scale) + np.log(endog)

        return ll_obs


class TLinearModelHet(GenericLikelihoodModel):
    '''Maximum Likelihood Estimation of Linear Model with t-distributed errors

    '''
    def __init__(self, endog, exog, exog_scale=None,
                 link_scale=families.links.Log(), fix_df=False, **kwds):

        if exog_scale is None:
            extra_names = ['scale']
            exog_scale = np.ones((len(endog), 1), dtype='f')
        else:
            extra_names = ['scale-%s' % zc for zc in
                           (exog_scale.columns
                            if hasattr(exog_scale, 'columns')
                            else range(1, exog_scale.shape[1] + 1))]

        if 'fix_df' not in kwds:
            extra_names.append('df')

        kwds['extra_params_names'] = extra_names

        super().__init__(endog, exog, exog_scale=exog_scale,
                         **kwds)
        # self.link = link
        self.link_scale = link_scale
        # not needed, handled by super:
        # self.exog_scale = exog_scale
        # inherited df do not account for precision params
        self.nobs = self.endog.shape[0]
        self.df_model = self.nparams - 1
        self.df_resid = self.nobs - self.nparams
        # need to fix, used for start_params,
        # self.k_vars = self.exog.shape[1] + self.exog_scale.shape[1]
        assert len(self.exog_scale) == len(self.endog)
        self.hess_type = "oim"
        if 'exog_scale' not in self._init_keys:
            self._init_keys.extend(['exog_scale'])

        # todo: maybe not here
        self._set_start_params()
        self._init_keys.extend(['link_scale'])
        self._null_drop_keys = ['exog_scale']
        # self.results_class = BetaResults
        # self.results_class_wrapper = BetaResultsWrapper

    def initialize(self):
        # TODO: here or in __init__
        self.k_vars = self.exog.shape[1]
        if not hasattr(self, 'fix_df'):
            self.fix_df = False

        if self.fix_df is False:
            # df will be estimated, no parameter restrictions
            self.fixed_params = None
            self.fixed_paramsmask = None
            self.k_params = self.exog.shape[1] + self.exog_scale.shape[1] + 1
            # extra_params_names = ['df', 'scale']
        else:
            # df fixed
            k_ex = self.exog.shape[1] + self.exog_scale.shape[1]
            self.k_params = k_ex
            fixdf = np.nan * np.zeros(k_ex + 1)
            fixdf[-1] = self.fix_df
            self.fixed_params = fixdf
            self.fixed_paramsmask = np.isnan(fixdf)
            # extra_params_names = ['scale']

        # self._set_extra_params_names(extra_params_names
        super().initialize()

    # todo use propertie for start_params
    def _set_start_params(self, start_params=None, use_kurtosis=False):
        if start_params is not None:
            self.start_params = start_params
        else:
            from statsmodels.regression.linear_model import OLS
            res_ols = OLS(self.endog, self.exog).fit()
            start_params = 0.1*np.ones(self.k_params)
            start_params[:self.k_vars] = res_ols.params

            if self.fix_df is False:

                if use_kurtosis:
                    kurt = stats.kurtosis(res_ols.resid)
                    df = 6. / kurt + 4
                else:
                    df = 5

                start_params[-1] = df
            # TODO adjust scale for df
            # Here we only use constant, missing link
            # TODO use regression
            # using link makes convergence slower in the examples
            start_params[self.exog.shape[1]] = self.link_scale(res_ols.scale)

            self.start_params = start_params

    def _predict_locscale(self, params):
        k_mean = self.exog.shape[1]
        k_scale = self.exog_scale.shape[1]
        beta = params[:k_mean]
        loc = np.dot(self.exog, beta)

        params_scale = params[k_mean : k_mean + k_scale]  # noqa: E203
        linpred_scale = np.dot(self.exog_scale, params_scale)
        scale = self.link_scale.inverse(linpred_scale)
        df = params[-1]
        return df, loc, scale

    def loglike(self, params):
        return self.loglikeobs(params).sum(0)

    def loglikeobs(self, params):
        """
        Loglikelihood of linear model with t distributed errors.

        Parameters
        ----------
        params : ndarray
            The parameters of the model. The last 2 parameters are degrees of
            freedom and scale.

        Returns
        -------
        loglike : ndarray
            The log likelihood of the model evaluated at `params` for each
            observation defined by self.endog and self.exog.

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+
                  y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]

        The t distribution is the standard t distribution and not a
        standardized t distribution, which means that the scale parameter is
        not equal to the standard deviation.

        """

        if self.fixed_params is not None:
            params = self.expandparams(params)

        df, loc, scale = self._predict_locscale(params)
        endog = self.endog
        scale_sqrt = np.sqrt(scale)
        x = (endog - loc) / scale_sqrt
        # next part is stats.t._logpdf
        lPx = sps_gamln((df+1)/2) - sps_gamln(df/2.)
        lPx -= 0.5*np_log(df*np_pi) + (df+1)/2.*np_log(1+(x**2)/df)
        lPx -= np_log(scale_sqrt)  # correction for scale
        return lPx

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return np.dot(exog, params[:self.exog.shape[1]])
