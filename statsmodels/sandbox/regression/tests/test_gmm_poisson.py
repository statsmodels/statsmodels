"""

TestGMMMultTwostepDefault() has lower precision

"""

from statsmodels.compat.python import lmap

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy import stats

from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.regression import gmm


def get_data():
    import os

    curdir = os.path.split(__file__)[0]
    dt = pd.read_csv(Path(curdir).joinpath("racd10data_with_transformed.csv"))
    mask = ~((np.asarray(dt["private"]) == 1) & (dt["medicaid"] == 1))
    mask = mask & (dt["docvis"] <= 70)
    dt3 = dt[mask]
    dt3["const"] = 1
    return dt3


DATA = get_data()


def moment_exponential_add(params, exog, exp=True):
    if not np.isfinite(params).all():
        print("invalid params", params)
    if exp:
        predicted = np.exp(np.dot(exog, params))
        predicted = np.clip(predicted, 0, 1e100)
    else:
        predicted = np.dot(exog, params)
    return predicted


def moment_exponential_mult(params, data, exp=True):
    endog = data[:, 0]
    exog = data[:, 1:]
    if not np.isfinite(params).all():
        print("invalid params", params)
    if exp:
        predicted = np.exp(np.dot(exog, params))
        predicted = np.clip(predicted, 0, 1e100)
        resid = endog / predicted - 1
        if not np.isfinite(resid).all():
            print("invalid resid", resid)
    else:
        resid = endog - np.dot(exog, params)
    return resid


class CheckGMM:
    params_tol = [5e-06, 5e-06]
    bse_tol = [5e-07, 5e-07]
    q_tol = [5e-06, 1e-09]
    j_tol = [5e-05, 1e-09]

    def test_basic(self):
        res1, res2 = (self.res1, self.res2)
        rtol, atol = self.params_tol
        assert_allclose(res1.params, res2.params, rtol=rtol, atol=0)
        assert_allclose(res1.params, res2.params, rtol=0, atol=atol)
        rtol, atol = self.bse_tol
        assert_allclose(res1.bse, res2.bse, rtol=rtol, atol=0)
        assert_allclose(res1.bse, res2.bse, rtol=0, atol=atol)

    def test_other(self):
        res1, res2 = (self.res1, self.res2)
        rtol, atol = self.q_tol
        assert_allclose(res1.q, res2.Q, rtol=atol, atol=rtol)
        rtol, atol = self.j_tol
        assert_allclose(res1.jval, res2.J, rtol=atol, atol=rtol)
        j, jpval, jdf = res1.jtest()
        assert_allclose(res1.jval, res2.J, rtol=13, atol=13)
        pval = stats.chi2.sf(res2.J, res2.J_df)
        assert_allclose(jpval, pval, rtol=rtol, atol=atol)
        assert_equal(jdf, res2.J_df)

    @pytest.mark.smoke
    def test_summary(self):
        res1 = self.res1
        summ = res1.summary()
        assert_equal(len(summ.tables[1]), len(res1.params) + 1)


class TestGMMAddOnestep(CheckGMM):

    @classmethod
    def setup_class(cls):
        XLISTEXOG2 = "aget aget2 educyr actlim totchr".split()
        endog_name = "docvis"
        exog_names = "private medicaid".split() + XLISTEXOG2 + ["const"]
        instrument_names = "income ssiratio".split() + XLISTEXOG2 + ["const"]
        endog = DATA[endog_name]
        exog = DATA[exog_names]
        instrument = DATA[instrument_names]

        def asarray(x):
            return np.asarray(x, float)

        endog, exog, instrument = lmap(asarray, [endog, exog, instrument])
        cls.bse_tol = [5e-06, 5e-07]
        start = OLS(np.log(endog + 1), exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.NonlinearIVGMM(endog, exog, instrument, moment_exponential_add)
        res0 = mod.fit(
            start,
            maxiter=0,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-08, "disp": 0},
            wargs={"centered": False},
        )
        cls.res1 = res0
        from .results_gmm_poisson import results_addonestep as results

        cls.res2 = results


class TestGMMAddTwostep(CheckGMM):

    @classmethod
    def setup_class(cls):
        XLISTEXOG2 = "aget aget2 educyr actlim totchr".split()
        endog_name = "docvis"
        exog_names = "private medicaid".split() + XLISTEXOG2 + ["const"]
        instrument_names = "income ssiratio".split() + XLISTEXOG2 + ["const"]
        endog = DATA[endog_name]
        exog = DATA[exog_names]
        instrument = DATA[instrument_names]

        def asarray(x):
            return np.asarray(x, float)

        endog, exog, instrument = lmap(asarray, [endog, exog, instrument])
        cls.bse_tol = [5e-06, 5e-07]
        start = OLS(np.log(endog + 1), exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.NonlinearIVGMM(endog, exog, instrument, moment_exponential_add)
        res0 = mod.fit(
            start,
            maxiter=2,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-08, "disp": 0},
            wargs={"centered": False},
            has_optimal_weights=False,
        )
        cls.res1 = res0
        from .results_gmm_poisson import results_addtwostep as results

        cls.res2 = results


class TestGMMMultOnestep(CheckGMM):

    @classmethod
    def setup_class(cls):
        XLISTEXOG2 = "aget aget2 educyr actlim totchr".split()
        endog_name = "docvis"
        exog_names = "private medicaid".split() + XLISTEXOG2 + ["const"]
        instrument_names = "income medicaid ssiratio".split() + XLISTEXOG2 + ["const"]
        endog = DATA[endog_name]
        exog = DATA[exog_names]
        instrument = DATA[instrument_names]

        def asarray(x):
            return np.asarray(x, float)

        endog, exog, instrument = lmap(asarray, [endog, exog, instrument])
        endog_ = np.zeros(len(endog))
        exog_ = np.column_stack((endog, exog))
        cls.bse_tol = [5e-06, 5e-07]
        cls.q_tol = [0.04, 0]
        cls.j_tol = [0.04, 0]
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.NonlinearIVGMM(endog_, exog_, instrument, moment_exponential_mult)
        res0 = mod.fit(
            start,
            maxiter=0,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-08, "disp": 0},
            wargs={"centered": False},
            has_optimal_weights=False,
        )
        cls.res1 = res0
        from .results_gmm_poisson import results_multonestep as results

        cls.res2 = results


class TestGMMMultTwostep(CheckGMM):

    @classmethod
    def setup_class(cls):
        XLISTEXOG2 = "aget aget2 educyr actlim totchr".split()
        endog_name = "docvis"
        exog_names = "private medicaid".split() + XLISTEXOG2 + ["const"]
        instrument_names = "income medicaid ssiratio".split() + XLISTEXOG2 + ["const"]
        endog = DATA[endog_name]
        exog = DATA[exog_names]
        instrument = DATA[instrument_names]

        def asarray(x):
            return np.asarray(x, float)

        endog, exog, instrument = lmap(asarray, [endog, exog, instrument])
        endog_ = np.zeros(len(endog))
        exog_ = np.column_stack((endog, exog))
        cls.bse_tol = [5e-06, 5e-07]
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.NonlinearIVGMM(endog_, exog_, instrument, moment_exponential_mult)
        res0 = mod.fit(
            start,
            maxiter=2,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-08, "disp": 0},
            wargs={"centered": False},
            has_optimal_weights=False,
        )
        cls.res1 = res0
        from .results_gmm_poisson import results_multtwostep as results

        cls.res2 = results


class TestGMMMultTwostepDefault(CheckGMM):

    @classmethod
    def setup_class(cls):
        XLISTEXOG2 = "aget aget2 educyr actlim totchr".split()
        endog_name = "docvis"
        exog_names = "private medicaid".split() + XLISTEXOG2 + ["const"]
        instrument_names = "income medicaid ssiratio".split() + XLISTEXOG2 + ["const"]
        endog = DATA[endog_name]
        exog = DATA[exog_names]
        instrument = DATA[instrument_names]

        def asarray(x):
            return np.asarray(x, float)

        endog, exog, instrument = lmap(asarray, [endog, exog, instrument])
        endog_ = np.zeros(len(endog))
        exog_ = np.column_stack((endog, exog))
        cls.bse_tol = [0.004, 0.0005]
        cls.params_tol = [5e-05, 5e-05]
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.NonlinearIVGMM(endog_, exog_, instrument, moment_exponential_mult)
        res0 = mod.fit(
            start,
            maxiter=2,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-08, "disp": 0},
        )
        cls.res1 = res0
        from .results_gmm_poisson import results_multtwostepdefault as results

        cls.res2 = results


class TestGMMMultTwostepCenter(CheckGMM):

    @classmethod
    def setup_class(cls):
        XLISTEXOG2 = "aget aget2 educyr actlim totchr".split()
        endog_name = "docvis"
        exog_names = "private medicaid".split() + XLISTEXOG2 + ["const"]
        instrument_names = "income medicaid ssiratio".split() + XLISTEXOG2 + ["const"]
        endog = DATA[endog_name]
        exog = DATA[exog_names]
        instrument = DATA[instrument_names]

        def asarray(x):
            return np.asarray(x, float)

        endog, exog, instrument = lmap(asarray, [endog, exog, instrument])
        endog_ = np.zeros(len(endog))
        exog_ = np.column_stack((endog, exog))
        cls.bse_tol = [0.0005, 5e-05]
        cls.params_tol = [5e-05, 5e-05]
        start = OLS(endog, exog).fit().params
        nobs, k_instr = instrument.shape
        w0inv = np.dot(instrument.T, instrument) / nobs
        mod = gmm.NonlinearIVGMM(endog_, exog_, instrument, moment_exponential_mult)
        res0 = mod.fit(
            start,
            maxiter=2,
            inv_weights=w0inv,
            optim_method="bfgs",
            optim_args={"gtol": 1e-08, "disp": 0},
            wargs={"centered": True},
            has_optimal_weights=False,
        )
        cls.res1 = res0
        from .results_gmm_poisson import results_multtwostepcenter as results

        cls.res2 = results

    def test_more(self):
        J_p = 0.332254330027383
        _, jpval, _ = self.res1.jtest()
        assert_allclose(jpval, J_p, rtol=5e-05, atol=0)


if __name__ == "__main__":
    tt = TestGMMAddOnestep()
    tt.setup_class()
    tt.test_basic()
    tt.test_other()
    tt = TestGMMAddTwostep()
    tt.setup_class()
    tt.test_basic()
    tt.test_other()
    tt = TestGMMMultOnestep()
    tt.setup_class()
    tt.test_basic()
    tt = TestGMMMultTwostep()
    tt.setup_class()
    tt.test_basic()
    tt.test_other()
    tt = TestGMMMultTwostepDefault()
    tt.setup_class()
    tt.test_basic()
    tt.test_other()
    tt = TestGMMMultTwostepCenter()
    tt.setup_class()
    tt.test_basic()
    tt.test_other()
