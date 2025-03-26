# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:12:52 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
import pandas as pd
from scipy import sparse

from numpy.testing import assert_allclose

from statsmodels.tools.decorators import cache_writable, cache_readonly
from statsmodels.regression.linear_model import OLS
from statsmodels.base.model import LikelihoodModelResults


class DummyData(object):

    def __init__(self, **kwds):
        # **kwds because I don't know yet what's required
        self.__dict__.update(kwds)


class SparseOLS(object):
    """minimal class for OLS with sparse exog

    no data checking or conversion is performed yet

    """

    def __init__(self, endog, exog, data_kwds=None):
        self.endog = endog
        if not sparse.issparse(exog):
            raise ValueError('exog is not sparse')
        self.exog = exog   # should be sparse

        if data_kwds is None:
            data_kwds = {}
        self.data = DummyData(**data_kwds)


        # define alias for results compatibility
        # add whiten for WLS
        self.wexog = self.exog
        self.wendog = self.endog


    def fit(self, cov_type='nonrobust', cov_kwds=None, use_t=None):
        exog = self.exog

        res_sp = sparse.linalg.lsqr(exog, self.endog, iter_lim=100)
        params = res_sp[0]
        xtx = exog.T.dot(exog).toarray()
        normalized_cov_params = np.linalg.pinv(xtx)

        res = SparseOLSResults(self, params, normalized_cov_params,
                               cov_type=cov_type, cov_kwds=cov_kwds,
                               use_t=use_t)
        return res

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        predicted = exog.dot(params)
        if sparse.issparse(predicted):
            return predicted.toarray()
        else:
            return predicted

# could try to subclass RegressionResults instead
class SparseOLSResults(LikelihoodModelResults):

    _cache = {}

    def __init__(self, model, params, normalized_cov_params,
                 cov_type='nonrobust', cov_kwds=None, use_t=False):

        self.model = model
        # some housekeeping
        # df_model needs constant check/assumption
        self.df_resid = self.nobs - model.exog.shape[1]  # assume full rank
        if not hasattr(model.data, 'param_names'):
            model.data.param_names = ['x%d'% i for i in range(5)]


        # we need to cheat here for the moment because super doesn't call
        # self._get_robustcov_results

        if cov_kwds is None:
            cov_kwds = {}

        super(SparseOLSResults, self).__init__(model, params,
                                               normalized_cov_params,
                               cov_type='nonrobust', cov_kwds=cov_kwds)

        self._get_robustcov_results(cov_type=cov_type, use_self=True,
                                    use_t=use_t, **cov_kwds)



    ######## copied from regression
    @cache_readonly
    def nobs(self):
        return float(self.model.wexog.shape[0])

    @cache_readonly
    def fittedvalues(self):
        return self.model.predict(self.params, self.model.exog)

    @cache_readonly
    def wresid(self):
        return self.model.wendog - self.model.predict(self.params,
                self.model.wexog)

    @cache_readonly
    def resid(self):
        return self.model.endog - self.model.predict(self.params,
                self.model.exog)

    #TODO: fix writable example
    @cache_writable()
    def scale(self):
        wresid = self.wresid
        return np.dot(wresid, wresid) / self.df_resid

    ############ end copy

    def _get_robustcov_results(self, cov_type='nonrobust', use_self=True,
                                   use_t=None, **cov_kwds):


        if cov_kwds is None:
            cov_kwds = {}

        if cov_type == 'nonrobust':
            self.cov_type = 'nonrobust'
            self.cov_kwds = {'description' : 'Standard Errors assume that the ' +
                             'covariance matrix of the errors is correctly ' +
                             'specified.'}
            scale = self.resid.dot(self.resid) / self.df_resid
            self.cov_params_default = self.normalized_cov_params * scale
            self._cache = {}  #empty cache
            self.use_t = use_t if use_t is not None else True

        elif cov_type == 'HC0':
            exog = self.model.exog  # sparse
            resid = self.resid  # dense
            xtxi = self.normalized_cov_params
            n_rows = resid.shape[0]
            xu = exog.T.dot(sparse.dia_matrix((resid, 0), shape=(n_rows, n_rows))).T

            S = xu.T.dot(xu).toarray()

            cov_p = xtxi.dot(S).dot(xtxi)

            # attach
            self.cov_params_default = cov_p
            self.use_t = use_t if use_t is not None else False

        else:
            raise ValueError('cov_type not supported, only nonrobust and HC0')


    def predict(self, exog=None, *args, **kwargs):

        # super/inherited method uses asarray, breaks with sparse
        return self.model.predict(self.params, exog, *args, **kwargs)
