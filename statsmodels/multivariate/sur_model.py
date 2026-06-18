# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:58:40 2017

Author: Josef Perktold
"""

import numpy as np
from scipy import sparse, stats
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
from statsmodels.stats.moment_helpers import cov2corr


def cov_func_spherical(resid, ddof=None):
    k = resid.shape[1]
    if ddof is None:
        ddof = k
    # overall variance
    var = resid.var(ddof=ddof)
    cov = np.diag(var * np.ones(k))
    return cov


def cov_func_diagonal(resid, ddof=None):
    k = resid.shape[1]
    if ddof is None:
        ddof = k
    # overall variance
    var = resid.var(ddof=ddof, axis=0)
    cov = np.diag(var)
    return cov


class DataDummy(object):
    # dummy data class because it is required in base LikelihoodModelResults

    def __init__(self, param_names):
        self.param_names = param_names

def params2block(params, k_vars_list):
    bs = np.split(params, np.cumsum(k_vars_list[:-1]))
    B = sparse.block_diag(bs).tocsc()
    return B

def _default_names(k_vars_list):

    names = ['eq%d_var%d' % (eq, k) for eq in range(len(k_vars_list))
                                    for k in range(k_vars_list[eq])]
    return names

class SURCompact(object):
    """Seemingly Unrelated Regression - compact version

    Requires balanced data set, nobs for each equation is the same.

    This is an experimental version that uses nobs x sum(k_vars) matrix for
    exog, i.e. where the exog of all equations are column stacked.
    """

    def __init__(self, endog_list, exog_list, cov_func=None):
        # this requires balanced dataset, all nobs the same
        self.endog_cstacked = self.y_cstacked = np.column_stack(endog_list)
        self.exog_cstacked = self.x_cstacked = np.column_stack(exog_list)
        self.k_vars_list = [ex.shape[1] for ex in exog_list]
        if cov_func is None:
            self.cov_func = lambda x: np.cov(x, rowvar=0)
        else:
            self.cov_func = cov_func

        self.n_eq = len(self.k_vars_list)
        self.nobs, n_eq = self.endog_cstacked.shape
        assert self.n_eq == n_eq
        # TODO
        self.data = DataDummy(_default_names(self.k_vars_list))

    def _predict(self, params_block, x_cstacked=None):

        if x_cstacked is None:
            x_cstacked = self.x_cstacked

        fitted = params_block.dot(x_cstacked.T).T
        return fitted

    def _resid(self, params):
        B = params2block(params, self.k_vars_list)
        fitted = self._predict(B)
        resid = self.y_cstacked - fitted
        return resid

    def _cov_resid(self, params, inv=False):
        resid = self._resid(params)

        if inv is False:
            return self.cov_func(resid)
        else:
            # replace with a direct method,
            # e.g. for patterned, sparse or regularized inv cov
            return np.linalg.pinv(self.cov_func(resid))

    def chol_cov_resid(self, params, inv=False):
        c = self._cov_resid(params=params, inv=inv)
        chol = np.linalg.cholesky(c)
        return chol

    def _wscore(self,params):
        x_cstacked = self.model.exog_cstacked
        ks = self.model.k_vars_list

        # there is some computation duplication
        resid = self._resid(params)
        cov_chol = self.chol_cov_resid(params, inv=False)
        wresid = resid.dot(cov_chol)
        wresid_expanded = np.repeat(wresid, ks, axis=1)
        wxu = x_cstacked * wresid_expanded
        return(wxu)


    def _fit_once(self, covi):
        y_cstacked = self.endog_cstacked
        x_cstacked = self.exog_cstacked
        ks = self.k_vars_list
        # moment matrix stacked

        #covi = np.eye(3)
        covi_ex = np.repeat(np.repeat(covi, ks, axis=1), ks, axis=0)
        covi_y = np.repeat(covi, ks, axis=0)

        xx = x_cstacked.T.dot(x_cstacked)
        wxx = covi_ex * xx
        #xy = x_cstacked.T.dot(y_cstacked)
        #xy = np.concatenate([xi.T.dot(yi) for yi, xi in zip(ys, xs)])
        wxy = (covi_y * x_cstacked.T.dot(y_cstacked)).sum(1)

        b = np.linalg.solve(wxx, wxy)
        return b, wxx

    def fit(self, start_params=None, maxiter=10, atol=1e-8, rtol=1e-5):

        if start_params is None:
            # start with OLS
            covi = np.eye(self.n_eq)
            params, wxx = self._fit_once(covi)
            covi_old = covi
        else:
            params = start_params

        converged = False
        for it in range(maxiter):
            covi = self._cov_resid(params, inv=True)
            params, wxx = self._fit_once(covi)
            if np.allclose(covi, covi_old, atol=atol, rtol=rtol):
                converged = True
                break

            covi_old = covi

        wxx_inv = np.linalg.pinv(wxx)
        res = SURCompactResult(self, params, wxx_inv, converged=converged,
                        covi_fit=covi)
        return res


class SURCompactResult(LikelihoodModelResults):

    # no super calls yet

    def __init__(self, model, params, normalized_cov_params, **kwargs):
        self.model = model
        self.params = params
        self.normalized_cov_params = normalized_cov_params
        self.scale = 1
        self.params_block = params2block(params, model.k_vars_list)
        self.__dict__.update(kwargs)

        # just a guess
        self.df_resid = self.model.nobs - max(self.model.k_vars_list)


    @cache_readonly
    def fittedvalues(self):
        return self.predict()

    @cache_readonly
    def resid(self):
        return self.model.endog_cstacked - self.fittedvalues

    @cache_readonly
    def cov_resid(self):
        # attach with default options
        return self.get_cov_resid()

    def predict(self, exog=None):
        if exog is None:
            exog = self.model.exog_cstacked

        predicted = self.model._predict(self.params_block, exog)
        return predicted

    def get_cov_resid(self, method='cov', ddof=None):
        if method != 'cov':
            print('not yet')
        if ddof is not None:
            raise NotImplementedError('df not yet available, using mle ddof=0')

        return np.cov(self.resid, rowvar=0)

    def lmtest_uncorr(self):
        nobs = self.model.nobs
        n_eq = self.model.n_eq
        corr_resid = cov2corr(self.cov_resid)
        # LM test for uncorrelated residuals
        rs = corr_resid[np.tril_indices(n_eq, -1)]
        lmstat = nobs * (rs**2).sum()
        pvalue = stats.chi2.sf(lmstat, n_eq * (n_eq - 1) / 2)
        return lmstat, pvalue

    def _get_score_values(self, eq_scaling=False):
        """column stacked score x u, cross sectional correlation ignored

        This is just for trying out for HC cov_type.
        """
        x_cstacked = self.model.exog_cstacked
        ks = self.model.k_vars_list
        resid = self.resid
        if eq_scaling:
            # this is just a try to rescale resid to take se of cross-sectional
            # correlation into account, maybe cholesky?
            #resid /= (np.diag(self.cov_resid)) # why not np.sqrt ?
            resid *= (np.diag(self.covi_fit))
        resid_expanded = np.repeat(resid, ks, axis=1)
        xu = x_cstacked * resid_expanded
        return xu

    def _cov_hc0(self, eq_scaling=True):
        """experimental, just a try

        not completely correct
        does not use whithened residual, i.e. use x u not wx u, in cov(score)
        scaling should be wrong in inner part of sandwich, i.e. inconsistent
        with normalized_cov_params (outer part of sandwich)
        """
        h = self.normalized_cov_params
        xu = self._get_score_values(eq_scaling=eq_scaling)
        c = np.cov(xu, rowvar=0) * self.model.nobs
        cov_hc0 = h.dot(c.dot(h))
        return cov_hc0

    @cache_readonly
    def bse_hc0(self):
        return np.sqrt(np.diag(self._cov_hc0()))
