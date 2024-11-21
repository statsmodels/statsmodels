"""
Created on Jul. 3, 2024 5:00:12 a.m.

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import leastsq
from scipy import stats

# from statsmodels.regression.linear_model import OLS
# from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust import norms as rnorms

from statsmodels.tools.func_nonlinear import MentenNL as Menten
from statsmodels.tools.func_nonlinear import (
    func_menten,
    )
from statsmodels.miscmodels.robust_genericmle import (
    MEstimatorHD,
    )


class CheckRobustNonlinear():

    @classmethod
    def setup_class(cls):
        nobs2 = 50
        sig_e = 0.75
        np.random.seed(456753)
        x = np.random.uniform(0, 10, size=nobs2)
        x.sort()
        beta_m = [10, 1]
        y_true = func_menten(beta_m, x)
        y = y_true + sig_e * np.random.randn(nobs2)
        y[-8::2] += 5

        cls.params_true = np.concatenate((beta_m, [sig_e]))
        cls.y_true = y_true
        cls.y = y
        cls.x = x

    def test_leastsquares(self):

        y, x = self.y, self.x

        res_cf = leastsq(lambda p: y - func_menten(p,x), x0=[0.5,0.5])
        fitted_ls = func_menten(res_cf[0], x)
        scale_ls = np.sqrt(np.mean((y - fitted_ls)**2))

        mod = MEstimatorHD(
            y, x,
            norm=rnorms.LeastSquares(),
            predictor=Menten(exog=x)
            )

        start_params = np.array([10, 2, 1.])
        # norm = mod.norm
        #scale_fac = stats.norm.expect(lambda t: t*norm.psi(t) - norm.rho(t))
        mod.k_extra = 3
        mod.df_resid = len(y) - mod.k_extra
        res = mod.fit(start_params, method='bfgs')

        assert_allclose(res.params[:2], res_cf[0], rtol=1e-5)
        assert_allclose(res.params[-1], scale_ls, rtol=1e-5)

    def test_huber(self):

        y, x = self.y, self.x
        y_true = self.y_true
        params_true = self.params_true

        mod = MEstimatorHD(
            y, x,
            norm=rnorms.HuberT(),
            predictor=Menten(exog=x)
            )
        start_params = np.array([10, 2, 1.])
        norm = mod.norm
        scale_fac = stats.norm.expect(lambda t: t*norm.psi(t) - norm.rho(t))

        res = mod.fit(start_params, method='bfgs')
        fittedvalues = res.predict()

        assert_allclose(mod.scale_fac, scale_fac, rtol=1e-5)

        # Huber is affected by outliers
        assert_allclose(res.params[:2], params_true[:2], rtol=0.2)
        assert_allclose(res.params[-1], params_true[-1], rtol=0.5)

        assert_allclose(fittedvalues, y_true, rtol=0.1)

    def test_biweight(self):

        y, x = self.y, self.x
        y_true = self.y_true
        params_true = self.params_true

        mod = MEstimatorHD(
            y, x,
            norm=rnorms.TukeyBiweight(),
            predictor=Menten(exog=x)
            )
        # results are sensitive to start_params
        start_params = np.array([10, 2, 2.])  # params_true * 1.1
        norm = mod.norm
        scale_fac = stats.norm.expect(lambda t: t*norm.psi(t) - norm.rho(t))

        res = mod.fit(start_params, method='bfgs')
        fittedvalues = res.predict()

        assert_allclose(mod.scale_fac, scale_fac, rtol=1e-5)

        # Huber is affected by outliers
        assert_allclose(res.params[:2], params_true[:2], rtol=0.2)
        # sign of scale parameter is currently indeterminate
        assert_allclose(np.abs(res.params[-1]), params_true[-1], rtol=0.5)

        # prediction in this case is not much better than Huber
        assert_allclose(fittedvalues, y_true, rtol=0.1)
        assert ((fittedvalues - y_true)**2).mean() < 0.01
        # assert res.mle_retvals.converged


    def test_iterative(self):
        y, x = self.y, self.x
        y_true = self.y_true
        params_true = self.params_true

        mod = MEstimatorHD(
            y, x,
            norm=rnorms.TukeyBiweight(),
            predictor=Menten(exog=x)
            )
        # results are sensitive to start_params
        start_params = np.array([10, 2, 1.])  # params_true * 1.1
        norm = mod.norm
        scale_fac = stats.norm.expect(lambda t: t*norm.psi(t) - norm.rho(t))

        # todo: wrong exog_names
        mod.exog_names[:] = ["a", "b", "scale"]
        # fit_iterative kwds not supported yet
        res = mod.fit_iterative(start_params[:2]) #, method='bfgs')
        fittedvalues = res.predict()

        assert_allclose(mod.scale_fac, scale_fac, rtol=1e-5)

        # Huber is affected by outliers
        assert_allclose(res.params[:2], params_true[:2], rtol=0.1)
        # sign of scale parameter is currently indeterminate
        assert_allclose(np.abs(res.scale), params_true[-1], rtol=0.1)
        assert len(res.params) == 2

        # prediction in this case is not much better than Huber
        assert_allclose(fittedvalues, y_true, rtol=0.1)


class TestMentenHD(CheckRobustNonlinear):
    pass
