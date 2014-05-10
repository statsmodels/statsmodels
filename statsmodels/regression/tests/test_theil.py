# -*- coding: utf-8 -*-
"""
Created on Mon May 05 17:29:56 2014

Author: Josef Perktold
"""

import os
import numpy as np
from scipy import stats

from numpy.testing import assert_allclose, assert_equal, assert_warns

from statsmodels.regression.linear_model import OLS, WLS, GLS
from statsmodels.tools.tools import add_constant
from statsmodels.tools.sm_exceptions import InvalidTestWarning

from statsmodels.sandbox.regression.penalized import TheilGLS


class TestTheilTextile(object):

    @classmethod
    def setup_class(cls):

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(cur_dir, "results", "theil_textile_predict.csv")
        cls.res_predict = np.recfromtxt(filepath, delimiter=",")


        names = "year	lconsump	lincome	lprice".split()

        data = np.array('''\
        1923	1.99651	1.98543	2.00432
        1924	1.99564	1.99167	2.00043
        1925	2	2	2
        1926	2.04766	2.02078	1.95713
        1927	2.08707	2.02078	1.93702
        1928	2.07041	2.03941	1.95279
        1929	2.08314	2.04454	1.95713
        1930	2.13354	2.05038	1.91803
        1931	2.18808	2.03862	1.84572
        1932	2.18639	2.02243	1.81558
        1933	2.20003	2.00732	1.78746
        1934	2.14799	1.97955	1.79588
        1935	2.13418	1.98408	1.80346
        1936	2.22531	1.98945	1.72099
        1937	2.18837	2.0103	1.77597
        1938	2.17319	2.00689	1.77452
        1939	2.2188	2.0162	1.78746'''.split(), float).reshape(-1, 4)


        endog = data[:, 1]
        # constant at the end to match Stata
        exog = np.column_stack((data[:, 2:], np.ones(endog.shape[0])))

        #prior(lprice -0.7 0.15 lincome 1 0.15) cov(lprice lincome -0.01)
        r_matrix = np.array([[1, 0, 0], [0, 1, 0]])
        r_mean = [1, -0.7]

        cov_r = np.array([[0.15**2, -0.01], [-0.01, 0.15**2]])
        mod = TheilGLS(endog, exog, r_matrix, q_matrix=r_mean, sigma_prior=cov_r)
        cls.res1 = mod.fit(cov_type='data-prior')
        from .results import results_theil_textile as resmodule
        cls.res2 = resmodule.results_theil_textile


    def test_basic(self):
        pt = self.res2.params_table[:,:6].T
        params2, bse2, tvalues2, pvalues2, ci_low, ci_upp = pt
        assert_allclose(self.res1.params, params2, rtol=2e-6)

        #TODO tgmixed seems to use scale from initial OLS, not from final res
        # np.sqrt(res.scale / res_ols.scale)
        # see below mse_resid which is equal to scale
        corr_fact = 0.9836026210570028
        assert_allclose(self.res1.bse / corr_fact, bse2, rtol=2e-6)
        assert_allclose(self.res1.tvalues  * corr_fact, tvalues2, rtol=2e-6)
        # pvalues are very small
        assert_allclose(self.res1.pvalues, pvalues2, atol=2e-6)
        assert_allclose(self.res1.pvalues, pvalues2, rtol=0.7)
        ci = self.res1.conf_int()
        # not scale corrected
        assert_allclose(ci[:,0], ci_low, rtol=0.01)
        assert_allclose(ci[:,1], ci_upp, rtol=0.01)
        assert_allclose(self.res1.rsquared, self.res2.r2, rtol=2e-6)

        # Note: tgmixed is using k_exog for df_resid
        corr_fact = self.res1.df_resid / self.res2.df_r
        assert_allclose(np.sqrt(self.res1.mse_resid * corr_fact), self.res2.rmse, rtol=2e-6)


    def test_other(self):
        tc = self.res1.test_compatibility()
        assert_allclose(np.squeeze(tc[0]), self.res2.compat, rtol=2e-6)
        assert_allclose(np.squeeze(tc[1]), self.res2.pvalue, rtol=2e-6)

        frac = self.res1.share_data()
        assert_allclose(frac, self.res2.frac_sample, rtol=2e-6)


    def test_no_penalization(self):
        res_ols = OLS(self.res1.model.endog, self.res1.model.exog).fit()
        res_theil = self.res1.model.fit(lambd=0, cov_type='data-prior')
        assert_allclose(res_theil.params, res_ols.params, rtol=1e-11)
        assert_allclose(res_theil.bse, res_ols.bse, rtol=1e-11)


    def test_smoke(self):
        self.res1.summary()


class CheckEquivalenceMixin(object):

    tol = {'default': (1e-4, 1e-20)}


    @classmethod
    def get_sample(cls):
        np.random.seed(987456)
        nobs, k_vars = 200, 5
        beta = 0.5 * np.array([0.1, 1, 1, 0, 0])
        x = np.random.randn(nobs, k_vars)
        x[:, 0] = 1
        y = np.dot(x, beta) + 2 * np.random.randn(nobs)
        return y, x



    def test_attributes(self):

        attributes_fit = ['params', 'rsquared', 'df_resid', 'df_model',
                          'llf', 'aic', 'bic'
                          #'fittedvalues', 'resid'
                          ]
        attributes_inference = ['bse', 'tvalues', 'pvalues']
        import copy
        attributes = copy.copy(attributes_fit)

        if not getattr(self, 'skip_inference', False):
            attributes.extend(attributes_inference)

        for att in attributes:
            r1 = getattr(self.res1, att)
            r2 = getattr(self.res2, att)
            if not np.size(r1) == 1:
                r1 = r1[:len(r2)]

            # check if we have overwritten tolerance
            rtol, atol = self.tol.get(att, self.tol['default'])
            message = 'attribute: ' + att #+ '\n%r\n\%r' % (r1, r2)
            assert_allclose(r1, r2, rtol=rtol, atol=atol, err_msg=message)

        # models are not close enough for some attributes at high precision
        assert_allclose(self.res1.fittedvalues, self.res1.fittedvalues,
                        rtol=1e-3, atol=1e-4)
        assert_allclose(self.res1.resid, self.res1.resid,
                        rtol=1e-3, atol=1e-4)


class TestTheil1(CheckEquivalenceMixin):
    # penalize last two parameters to zero

    @classmethod
    def setup_class(cls):
        y, x = cls.get_sample()
        mod1 = TheilGLS(y, x, sigma_prior=[0, 0, 1., 1.])
        cls.res1 = mod1.fit(200000)
        cls.res2 = OLS(y, x[:, :3]).fit()


class TestTheil2(CheckEquivalenceMixin):
    # no penalization = same as OLS

    @classmethod
    def setup_class(cls):
        y, x = cls.get_sample()
        mod1 = TheilGLS(y, x, sigma_prior=[0, 0, 1., 1.])
        cls.res1 = mod1.fit(0)
        cls.res2 = OLS(y, x).fit()


class TestTheil3(CheckEquivalenceMixin):
    # perfect multicollinearity = same as OLS in terms of fit
    # inference: bse, ... is different

    @classmethod
    def setup_class(cls):
        cls.skip_inference = True
        y, x = cls.get_sample()
        xd = np.column_stack((x, x))
        #sp = np.zeros(5), np.ones(5)
        r_matrix = np.eye(5, 10, 5)
        mod1 = TheilGLS(y, xd, r_matrix=r_matrix) #sigma_prior=[0, 0, 1., 1.])
        cls.res1 = mod1.fit(0.001, cov_type='data-prior')
        cls.res2 = OLS(y, x).fit()


class TestTheilGLS(CheckEquivalenceMixin):
    # penalize last two parameters to zero

    @classmethod
    def setup_class(cls):
        y, x = cls.get_sample()
        nobs = len(y)
        weights = (np.arange(nobs) < (nobs // 2)) + 0.5
        mod1 = TheilGLS(y, x, sigma=weights, sigma_prior=[0, 0, 1., 1.])
        cls.res1 = mod1.fit(200000)
        cls.res2 = GLS(y, x[:, :3], sigma=weights).fit()


class TestTheilLinRestriction(CheckEquivalenceMixin):
    # impose linear restriction with small uncertainty - close to OLS

    @classmethod
    def setup_class(cls):
        y, x = cls.get_sample()
        #merge var1 and var2
        x2 = x[:, :2].copy()
        x2[:, 1] += x[:, 2]
        #mod1 = TheilGLS(y, x, r_matrix =[[0, 1, -1, 0, 0]])
        mod1 = TheilGLS(y, x[:, :3], r_matrix =[[0, 1, -1]])
        cls.res1 = mod1.fit(200000)
        cls.res2 = OLS(y, x2).fit()

        # adjust precision, careful: cls.tol is mutable
        tol = {'pvalues': (1e-4, 2e-7),
               'tvalues': (5e-4, 0)}
        tol.update(cls.tol)
        cls.tol = tol

class TestTheilLinRestrictionApprox(CheckEquivalenceMixin):
    # impose linear restriction with some uncertainty

    @classmethod
    def setup_class(cls):
        y, x = cls.get_sample()
        #merge var1 and var2
        x2 = x[:, :2].copy()
        x2[:, 1] += x[:, 2]
        #mod1 = TheilGLS(y, x, r_matrix =[[0, 1, -1, 0, 0]])
        mod1 = TheilGLS(y, x[:, :3], r_matrix =[[0, 1, -1]])
        cls.res1 = mod1.fit(100)
        cls.res2 = OLS(y, x2).fit()

        # adjust precision, careful: cls.tol is mutable
        import copy
        tol = copy.copy(cls.tol)
        tol2 = {'default': (0.15,  0),
                'params':  (0.05, 0),
                'pvalues': (0.02, 0.001),
                }
        tol.update(tol2)
        cls.tol = tol
