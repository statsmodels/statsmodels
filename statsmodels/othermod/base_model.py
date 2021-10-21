# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 08:29:59 2021

Author: Josef Perktod
License: BSD-3
"""

import numpy as np

from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.genmod import families


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
        self.k_params = self.k_params_cumli[-1]
        self.nparams = self.k_params  # TODO: old attribute
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

        self.df_null = 2 + self.k_extra
        self.df_model = self.k_params - self.df_null
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
        # self.k_params = (self.exog.shape[1] + self.exog_scale.shape[1] +
        #                  self.k_extra)
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


class _MultiLinkFamilyModel(MultiLinkModel):
    """Experimental class using DFamily
    """

    def __init__(self, endog, exog, exog_scale=None,
                 exog_extras=None, dfamily=None, link=None,
                 link_scale=families.links.Log(),
                 link_extras=None, k_extra=None, **kwds):

        if dfamily is None:
            raise ValueError("dfamily is required")
        if type(dfamily) is type:
            # Note, difficult to debug if class is used
            raise ValueError("dfamily needs to be an instance, not a class")

        self.dfamily = dfamily

        super().__init__(
            endog, exog,
            exog_scale=exog_scale,
            exog_extras=exog_extras,
            link=link,
            link_scale=link_scale,
            link_extras=link_extras,
            k_extra=dfamily.k_args - 2,
            **kwds)

    def _loglikeobs(self, *dargs, endog=None):
        # Todo: need `dkwargs` like n_trials in BetaBinomial
        ll_obs = self.dfamily.loglike_obs(endog, *dargs)
        return ll_obs
