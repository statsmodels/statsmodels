"""
Tests for exponential smoothing models

Notes
-----

These tests are primarily against the `fpp` functions `ses`, `holt`, and `hw`
and against the `forecast` function `ets`. There are a couple of details about
how these packages work that are relevant for the tests:

Trend smoothing parameterization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that `fpp` and `ets` use
different parameterizations for the trend smoothing parameter. Our
implementation in `statespace.exponential_smoothing` uses the same
parameterization as `ets`.

The `fpp` package follows Holt's recursive equations directly, in which the
trend updating is:

.. math::

    b_t = \\beta^* (\\ell_t - \\ell_{t-1}) + (1 - \\beta^*) b_{t-1}

In our implementation, state updating is done by the Kalman filter, in which
the trend updating equation is:

.. math::

    b_{t|t} = b_{t|t-1} + \\beta (y_t - l_{t|t-1})

by rewriting the Kalman updating equation in the form of Holt's method, we
find that we must have :math:`\\beta = \\beta^* \\alpha`. This is the same
parameterization used by `ets`, which does not use the Kalman fitler but
instead uses an innovations state space framework.

Loglikelihood
^^^^^^^^^^^^^

The `ets` package has a `loglik` output value, but it does not compute the
loglikelihood itself, but rather a version without the constant parameters. It
appears to compute:

.. math::

    -\\frac{n}{2} \\log \\left (\\sum_{t=1}^n \\varepsilon_t^2 \\right)

while the loglikelihood is:

.. math::

    -\\frac{n}{2}
    \\log \\left (2 \\pi e \\frac{1}{n} \\sum_{t=1}^n \\varepsilon_t^2 \\right)

See Hyndman et al. (2008), pages 68-69. In particular, the former equation -
which is the value returned by `ets` - is -0.5 times equation (5.3), since for
these models we have :math:`r(x_{t-1}) = 1`. The latter equation is the log
of the likelihood formula given at the top of page 69.

Confidence intervals
^^^^^^^^^^^^^^^^^^^^

The range of the confidence intervals depends on the estimated variance,
sigma^2. In our default, we concentrate this variance out of the loglikelihood
function, meaning that the default is to use the maximum likelihood estimate
for forecasting purposes. forecast::ets uses a degree-of-freedom-corrected
estimate of sigma^2, and so our default confidence bands will differ. To
correct for this in the tests, we set `concentrate_scale=False` and use the
estimated variance from forecast::ets.

TODO: may want to add a parameter allowing specification of the variance
      estimator.

Author: Chad Fulton
License: BSD-3
"""

from pathlib import Path

import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

current_path = Path(__file__).resolve().parent
results_path = Path(current_path).joinpath("results")
params_path = Path(results_path).joinpath("exponential_smoothing_params.csv")
predict_path = Path(results_path).joinpath("exponential_smoothing_predict.csv")
states_path = Path(results_path).joinpath("exponential_smoothing_states.csv")
results_params = pd.read_csv(params_path, index_col=[0])
results_predict = pd.read_csv(predict_path, index_col=[0])
results_states = pd.read_csv(states_path, index_col=[0])
oildata = pd.Series(
    [
        446.6565229,
        454.4733065,
        455.662974,
        423.6322388,
        456.2713279,
        440.5880501,
        425.3325201,
        485.1494479,
        506.0481621,
        526.7919833,
        514.268889,
        494.2110193,
    ],
    index=pd.period_range(start="1996", end="2007", freq="Y"),
)
air = pd.Series(
    [
        17.5534,
        21.8601,
        23.8866,
        26.9293,
        26.8885,
        28.8314,
        30.0751,
        30.9535,
        30.1857,
        31.5797,
        32.577569,
        33.477398,
        39.021581,
        41.386432,
        41.596552,
    ],
    index=pd.period_range(start="1990", end="2004", freq="Y"),
)
aust = pd.Series(
    [
        41.727458,
        24.04185,
        32.328103,
        37.328708,
        46.213153,
        29.346326,
        36.48291,
        42.977719,
        48.901525,
        31.180221,
        37.717881,
        40.420211,
        51.206863,
        31.887228,
        40.978263,
        43.772491,
        55.558567,
        33.850915,
        42.076383,
        45.642292,
        59.76678,
        35.191877,
        44.319737,
        47.913736,
    ],
    index=pd.period_range(start="2005Q1", end="2010Q4", freq="Q-OCT"),
)


class CheckExponentialSmoothing:

    @classmethod
    def setup_class(cls, name, res):
        cls.name = name
        cls.res = res
        cls.nobs = res.nobs
        cls.nforecast = len(results_predict["%s_mean" % cls.name]) - cls.nobs
        cls.forecast = res.get_forecast(cls.nforecast)

    def test_fitted(self):
        predicted = results_predict["%s_mean" % self.name]
        assert_allclose(self.res.fittedvalues, predicted.iloc[: self.nobs])

    def test_output(self):
        has_llf = ~np.isnan(results_params[self.name]["llf"])
        if has_llf:
            assert_allclose(self.res.mse, results_params[self.name]["mse"])
            actual = -0.5 * self.nobs * np.log(np.sum(self.res.resid**2))
            assert_allclose(actual, results_params[self.name]["llf"])
        else:
            assert_allclose(self.res.sse, results_params[self.name]["sse"])

    def test_forecasts(self):
        predicted = results_predict["%s_mean" % self.name]
        assert_allclose(self.forecast.predicted_mean, predicted.iloc[self.nobs :])

    def test_conf_int(self):
        ci_95 = self.forecast.conf_int(alpha=0.05)
        lower = results_predict["%s_lower" % self.name]
        upper = results_predict["%s_upper" % self.name]
        assert_allclose(ci_95["lower y"], lower.iloc[self.nobs :])
        assert_allclose(ci_95["upper y"], upper.iloc[self.nobs :])

    def test_initial_states(self):
        mask = results_states.columns.str.startswith(self.name)
        desired = results_states.loc[:, mask].dropna().iloc[0]
        assert_allclose(self.res.initial_state.iloc[0], desired)

    def test_states(self):
        mask = results_states.columns.str.startswith(self.name)
        desired = results_states.loc[:, mask].dropna().iloc[1:]
        assert_allclose(self.res.filtered_state[1:].T, desired)

    def test_misc(self):
        mod = self.res.model
        assert_equal(mod.k_params, len(mod.start_params))
        assert_equal(mod.k_params, len(mod.param_names))
        self.res.summary()


class TestSESFPPFixed02(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(oildata, initialization_method="simple")
        res = mod.filter([results_params["oil_fpp1"]["alpha"]])
        super().setup_class("oil_fpp1", res)


class TestSESFPPFixed06(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(oildata, initialization_method="simple")
        res = mod.filter([results_params["oil_fpp2"]["alpha"]])
        super().setup_class("oil_fpp2", res)


class TestSESFPPEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(
            oildata, initialization_method="estimated", concentrate_scale=False
        )
        res = mod.filter(
            [
                results_params["oil_fpp3"]["alpha"],
                results_params["oil_fpp3"]["sigma2"],
                results_params["oil_fpp3"]["l0"],
            ]
        )
        super().setup_class("oil_fpp3", res)


class TestSESETSEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(
            oildata, initialization_method="estimated", concentrate_scale=False
        )
        res = mod.filter(
            [
                results_params["oil_ets"]["alpha"],
                results_params["oil_ets"]["sigma2"],
                results_params["oil_ets"]["l0"],
            ]
        )
        super().setup_class("oil_ets", res)

    @pytest.mark.thread_unsafe(reason="statespace cython code is not thread safe")
    def test_mle_estimates(self):
        mle_res = self.res.model.fit(disp=0)
        assert_(self.res.llf <= mle_res.llf)


class TestHoltFPPFixed(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(
            air, trend=True, concentrate_scale=False, initialization_method="simple"
        )
        params = [
            results_params["air_fpp1"]["alpha"],
            results_params["air_fpp1"]["beta_star"],
            results_params["air_fpp1"]["sigma2"],
        ]
        params[1] = params[0] * params[1]
        res = mod.filter(params)
        super().setup_class("air_fpp1", res)

    def test_conf_int(self):
        j = np.arange(1, 14)
        alpha, beta, sigma2 = self.res.params
        c = np.r_[0, alpha + beta * j]
        se = (sigma2 * (1 + np.cumsum(c**2))) ** 0.5
        assert_allclose(self.forecast.se_mean, se)


class TestHoltDampedFPPEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(
            air, trend=True, damped_trend=True, concentrate_scale=False
        )
        params = [
            results_params["air_fpp2"]["alpha"],
            results_params["air_fpp2"]["beta"],
            results_params["air_fpp2"]["phi"],
            results_params["air_fpp2"]["sigma2"],
            results_params["air_fpp2"]["l0"],
            results_params["air_fpp2"]["b0"],
        ]
        res = mod.filter(params)
        super().setup_class("air_fpp2", res)


class TestHoltDampedETSEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(
            air, trend=True, damped_trend=True, concentrate_scale=False
        )
        params = [
            results_params["air_ets"]["alpha"],
            results_params["air_ets"]["beta"],
            results_params["air_ets"]["phi"],
            results_params["air_ets"]["sigma2"],
            results_params["air_ets"]["l0"],
            results_params["air_ets"]["b0"],
        ]
        res = mod.filter(params)
        super().setup_class("air_ets", res)

    @pytest.mark.thread_unsafe(reason="statespace cython code is not thread safe")
    def test_mle_estimates(self):
        mle_res = self.res.model.fit(disp=0)
        assert_(self.res.llf <= mle_res.llf)


class TestHoltWintersFPPEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(
            aust, trend=True, seasonal=4, concentrate_scale=False
        )
        params = np.r_[
            results_params["aust_fpp1"]["alpha"],
            results_params["aust_fpp1"]["beta"],
            results_params["aust_fpp1"]["gamma"],
            results_params["aust_fpp1"]["sigma2"],
            results_params["aust_fpp1"]["l0"],
            results_params["aust_fpp1"]["b0"],
            results_params["aust_fpp1"]["s0_0"],
            results_params["aust_fpp1"]["s0_1"],
            results_params["aust_fpp1"]["s0_2"],
        ]
        res = mod.filter(params)
        super().setup_class("aust_fpp1", res)


class TestHoltWintersETSEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(
            aust, trend=True, seasonal=4, concentrate_scale=False
        )
        params = np.r_[
            results_params["aust_ets1"]["alpha"],
            results_params["aust_ets1"]["beta"],
            results_params["aust_ets1"]["gamma"],
            results_params["aust_ets1"]["sigma2"],
            results_params["aust_ets1"]["l0"],
            results_params["aust_ets1"]["b0"],
            results_params["aust_ets1"]["s0_0"],
            results_params["aust_ets1"]["s0_1"],
            results_params["aust_ets1"]["s0_2"],
        ]
        res = mod.filter(params)
        super().setup_class("aust_ets1", res)


class TestHoltWintersDampedETSEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(
            aust, trend=True, damped_trend=True, seasonal=4, concentrate_scale=False
        )
        params = np.r_[
            results_params["aust_ets2"]["alpha"],
            results_params["aust_ets2"]["beta"],
            results_params["aust_ets2"]["gamma"],
            results_params["aust_ets2"]["phi"],
            results_params["aust_ets2"]["sigma2"],
            results_params["aust_ets2"]["l0"],
            results_params["aust_ets2"]["b0"],
            results_params["aust_ets2"]["s0_0"],
            results_params["aust_ets2"]["s0_1"],
            results_params["aust_ets2"]["s0_2"],
        ]
        res = mod.filter(params)
        super().setup_class("aust_ets2", res)

    @pytest.mark.thread_unsafe(reason="statespace cython code is not thread safe")
    def test_mle_estimates(self):
        mle_res = self.res.model.fit(disp=0, maxiter=100)
        assert_(self.res.llf <= mle_res.llf)


class TestHoltWintersNoTrendETSEstimated(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, seasonal=4, concentrate_scale=False)
        params = np.r_[
            results_params["aust_ets3"]["alpha"],
            results_params["aust_ets3"]["gamma"],
            results_params["aust_ets3"]["sigma2"],
            results_params["aust_ets3"]["l0"],
            results_params["aust_ets3"]["s0_0"],
            results_params["aust_ets3"]["s0_1"],
            results_params["aust_ets3"]["s0_2"],
        ]
        res = mod.filter(params)
        super().setup_class("aust_ets3", res)

    def test_conf_int(self):
        j = np.arange(1, 5)
        alpha, gamma, sigma2 = self.res.params[:3]
        c = np.r_[0, alpha + gamma * (j % 4 == 0).astype(int)]
        se = (sigma2 * (1 + np.cumsum(c**2))) ** 0.5
        assert_allclose(self.forecast.se_mean, se)

    @pytest.mark.thread_unsafe(reason="statespace cython code is not thread safe")
    def test_mle_estimates(self):
        start_params = [0.5, 0.4, 4, 32, 2.3, -2, -9]
        mle_res = self.res.model.fit(start_params, disp=0, maxiter=100)
        assert_(self.res.llf <= mle_res.llf)


class CheckKnownInitialization:

    @classmethod
    def setup_class(cls, mod, start_params):
        cls.mod = mod
        cls.start_params = start_params
        endog = mod.data.orig_endog
        cls.res = cls.mod.fit(start_params, disp=0, maxiter=100)
        cls.initial_level = cls.res.params.get("initial_level", None)
        cls.initial_trend = cls.res.params.get("initial_trend", None)
        cls.initial_seasonal = None
        if cls.mod.seasonal:
            cls.initial_seasonal = [cls.res.params["initial_seasonal"]] + [
                cls.res.params["initial_seasonal.L%d" % i]
                for i in range(1, cls.mod.seasonal_periods - 1)
            ]
        cls.params = cls.res.params[:"initial_level"].drop("initial_level")
        cls.init_params = cls.res.params["initial_level":]
        cls.known_mod = cls.mod.clone(
            endog,
            initialization_method="known",
            initial_level=cls.initial_level,
            initial_trend=cls.initial_trend,
            initial_seasonal=cls.initial_seasonal,
        )

    @pytest.mark.thread_unsafe(reason="statespace cython code is not thread safe")
    def test_given_params(self):
        known_res = self.known_mod.filter(self.params)
        assert_allclose(known_res.llf, self.res.llf)
        assert_allclose(known_res.predicted_state, self.res.predicted_state)
        assert_allclose(known_res.predicted_state_cov, self.res.predicted_state_cov)
        assert_allclose(known_res.filtered_state, self.res.filtered_state)

    @pytest.mark.thread_unsafe(reason="statespace cython code is not thread safe")
    def test_estimated_params(self):
        fit_res1 = self.mod.fit_constrained(
            self.init_params.to_dict(),
            start_params=self.start_params,
            includes_fixed=True,
            disp=0,
        )
        fit_res2 = self.known_mod.fit(
            self.start_params[:"initial_level"].drop("initial_level"), disp=0
        )
        assert_allclose(
            fit_res1.params[:"initial_level"].drop("initial_level"), fit_res2.params
        )
        assert_allclose(fit_res1.llf, fit_res2.llf)
        assert_allclose(fit_res1.scale, fit_res2.scale)
        assert_allclose(fit_res1.predicted_state, fit_res2.predicted_state)
        assert_allclose(fit_res1.predicted_state_cov, fit_res2.predicted_state_cov)
        assert_allclose(fit_res1.filtered_state, fit_res2.filtered_state)


class TestSESKnownInitialization(CheckKnownInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(oildata)
        start_params = pd.Series([0.8, 440.0], index=mod.param_names)
        super().setup_class(mod, start_params)


class TestHoltKnownInitialization(CheckKnownInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True)
        start_params = pd.Series([0.95, 0.0005, 15.0, 1.5], index=mod.param_names)
        super().setup_class(mod, start_params)


class TestHoltDampedKnownInitialization(CheckKnownInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True, damped_trend=True)
        start_params = pd.Series([0.9, 0.0005, 0.9, 14.0, 2.0], index=mod.param_names)
        super().setup_class(mod, start_params)


class TestHoltWintersKnownInitialization(CheckKnownInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, trend=True, seasonal=4)
        start_params = pd.Series(
            [0.0005, 0.0004, 0.5, 33.0, 0.4, 2.5, -2.0, -9.0], index=mod.param_names
        )
        super().setup_class(mod, start_params)


class TestHoltWintersDampedKnownInitialization(CheckKnownInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True, damped_trend=True, seasonal=4)
        start_params = pd.Series(
            [0.0005, 0.0004, 0.0005, 0.95, 17.0, 1.5, -0.2, 0.1, 0.4],
            index=mod.param_names,
        )
        super().setup_class(mod, start_params)


class TestHoltWintersNoTrendKnownInitialization(CheckKnownInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, seasonal=4)
        start_params = pd.Series([0.5, 0.49, 30.0, 2.0, -2, -9], index=mod.param_names)
        super().setup_class(mod, start_params)


class CheckHeuristicInitialization:

    @classmethod
    def setup_class(cls, mod):
        cls.mod = mod
        cls.res = cls.mod.filter(cls.mod.start_params)
        init_heuristic = np.r_[cls.mod._initial_level]
        if cls.mod.trend:
            init_heuristic = np.r_[init_heuristic, cls.mod._initial_trend]
        if cls.mod.seasonal:
            init_heuristic = np.r_[init_heuristic, cls.mod._initial_seasonal]
        cls.init_heuristic = init_heuristic
        endog = cls.mod.data.orig_endog
        initial_seasonal = cls.mod._initial_seasonal
        cls.known_mod = cls.mod.clone(
            endog,
            initialization_method="known",
            initial_level=cls.mod._initial_level,
            initial_trend=cls.mod._initial_trend,
            initial_seasonal=initial_seasonal,
        )
        cls.known_res = cls.mod.filter(cls.mod.start_params)


class TestSESHeuristicInitialization(CheckHeuristicInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(oildata, initialization_method="heuristic")
        super().setup_class(mod)

    def test_heuristic(self):
        nobs = 10
        exog = np.c_[np.ones(nobs), np.arange(nobs) + 1]
        desired = np.linalg.pinv(exog).dot(oildata.values[:nobs])[0]
        assert_allclose(self.init_heuristic, desired)


class TestHoltHeuristicInitialization(CheckHeuristicInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True, initialization_method="heuristic")
        super().setup_class(mod)

    def test_heuristic(self):
        nobs = 10
        exog = np.c_[np.ones(nobs), np.arange(nobs) + 1]
        desired = np.linalg.pinv(exog).dot(air.values[:nobs])
        assert_allclose(self.init_heuristic, desired)


class TestHoltDampedHeuristicInitialization(CheckHeuristicInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(
            air, trend=True, damped_trend=True, initialization_method="heuristic"
        )
        super().setup_class(mod)

    def test_heuristic(self):
        TestHoltHeuristicInitialization.test_heuristic(self)


class TestHoltWintersHeuristicInitialization(CheckHeuristicInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(
            aust, trend=True, seasonal=4, initialization_method="heuristic"
        )
        super().setup_class(mod)

    def test_heuristic(self):
        trend = aust[:20].rolling(4).mean().rolling(2).mean().shift(-2).dropna()
        nobs = 10
        exog = np.c_[np.ones(nobs), np.arange(nobs) + 1]
        desired = np.linalg.pinv(exog).dot(trend[:nobs])
        if not self.mod.trend:
            desired = desired[:1]
        detrended = aust - trend
        initial_seasonal = np.nanmean(detrended.values.reshape(6, 4), axis=0)
        initial_seasonal = initial_seasonal[::-1]
        desired = np.r_[desired, initial_seasonal - np.mean(initial_seasonal)]
        assert_allclose(self.init_heuristic, desired)


class TestHoltWintersDampedHeuristicInitialization(CheckHeuristicInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(
            aust,
            trend=True,
            damped_trend=True,
            seasonal=4,
            initialization_method="heuristic",
        )
        super().setup_class(mod)

    def test_heuristic(self):
        TestHoltWintersHeuristicInitialization.test_heuristic(self)


class TestHoltWintersNoTrendHeuristicInitialization(CheckHeuristicInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, seasonal=4, initialization_method="heuristic")
        super().setup_class(mod)

    def test_heuristic(self):
        TestHoltWintersHeuristicInitialization.test_heuristic(self)


def test_concentrated_initialization():
    mod1 = ExponentialSmoothing(oildata, initialization_method="concentrated")
    mod2 = ExponentialSmoothing(oildata)
    res1 = mod1.filter([0.1])
    res2 = mod2.fit_constrained({"smoothing_level": 0.1}, disp=0)
    res1 = mod1.fit(disp=0)
    res2 = mod2.fit(disp=0)
    assert_allclose(res1.llf, res2.llf)
    assert_allclose(res1.initial_state, res2.initial_state, rtol=1e-05)


class CheckConcentratedInitialization:

    @classmethod
    def setup_class(cls, mod, start_params=None, atol=0, rtol=1e-07):
        cls.start_params = start_params
        cls.atol = atol
        cls.rtol = rtol
        cls.mod = mod
        cls.conc_mod = mod.clone(
            mod.data.orig_endog, initialization_method="concentrated"
        )
        cls.params = pd.Series(
            [0.5, 0.2, 0.2, 0.95],
            index=[
                "smoothing_level",
                "smoothing_trend",
                "smoothing_seasonal",
                "damping_trend",
            ],
        )
        drop = []
        if not cls.mod.trend:
            drop += ["smoothing_trend", "damping_trend"]
        elif not cls.mod.damped_trend:
            drop += ["damping_trend"]
        if not cls.mod.seasonal:
            drop += ["smoothing_seasonal"]
        cls.params.drop(drop, inplace=True)

    @pytest.mark.thread_unsafe(reason="statespace cython code is not thread safe")
    def test_given_params(self):
        res = self.mod.fit_constrained(self.params.to_dict(), disp=0)
        conc_res = self.conc_mod.filter(self.params.values)
        assert_allclose(conc_res.llf, res.llf, atol=self.atol, rtol=self.rtol)
        assert_allclose(
            conc_res.initial_state, res.initial_state, atol=self.atol, rtol=self.rtol
        )

    @pytest.mark.thread_unsafe(reason="statespace cython code is not thread safe")
    def test_estimated_params(self):
        res = self.mod.fit(self.start_params, disp=0, maxiter=100)
        np.set_printoptions(suppress=True)
        conc_res = self.conc_mod.fit(self.start_params[: len(self.params)], disp=0)
        assert_allclose(conc_res.llf, res.llf, atol=self.atol, rtol=self.rtol)
        assert_allclose(
            conc_res.initial_state, res.initial_state, atol=self.atol, rtol=self.rtol
        )


class TestSESConcentratedInitialization(CheckConcentratedInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(oildata)
        start_params = pd.Series([0.85, 447.0], index=mod.param_names)
        super().setup_class(mod, start_params=start_params, rtol=1e-05)


class TestHoltConcentratedInitialization(CheckConcentratedInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True)
        start_params = pd.Series([0.95, 0.0005, 15.0, 1.5], index=mod.param_names)
        super().setup_class(mod, start_params=start_params, rtol=0.0001)


class TestHoltDampedConcentratedInitialization(CheckConcentratedInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(air, trend=True, damped_trend=True)
        start_params = pd.Series([0.95, 0.0005, 0.9, 15.0, 2.5], index=mod.param_names)
        super().setup_class(mod, start_params=start_params, rtol=0.1)


class TestHoltWintersConcentratedInitialization(CheckConcentratedInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, trend=True, seasonal=4)
        start_params = pd.Series(
            [0.0005, 0.0004, 0.0002, 33.0, 0.4, 2.2, -2.0, -9.3], index=mod.param_names
        )
        super().setup_class(mod, start_params=start_params, rtol=0.001)


class TestHoltWintersDampedConcentratedInitialization(CheckConcentratedInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, trend=True, damped_trend=True, seasonal=4)
        start_params = pd.Series(
            [0.0005, 0.0004, 0.0005, 0.95, 17.0, 1.5, -0.2, 0.1, 0.4],
            index=mod.param_names,
        )
        super().setup_class(mod, start_params=start_params, rtol=0.1)


class TestHoltWintersNoTrendConcentratedInitialization(CheckConcentratedInitialization):

    @classmethod
    def setup_class(cls):
        mod = ExponentialSmoothing(aust, seasonal=4)
        start_params = pd.Series(
            [0.5, 0.49, 32.0, 2.3, -2.1, -9.3], index=mod.param_names
        )
        super().setup_class(mod, start_params=start_params, rtol=0.0001)


class TestMultiIndex(CheckExponentialSmoothing):

    @classmethod
    def setup_class(cls):
        oildata_copy = oildata.copy()
        oildata_copy.name = ("oil", "data")
        mod = ExponentialSmoothing(oildata_copy, initialization_method="simple")
        res = mod.filter([results_params["oil_fpp2"]["alpha"]])
        super().setup_class("oil_fpp2", res)

    def test_conf_int(self):
        ci_95 = self.forecast.conf_int(alpha=0.05)
        lower = results_predict["%s_lower" % self.name]
        upper = results_predict["%s_upper" % self.name]
        assert_allclose(ci_95["lower ('oil', 'data')"], lower.iloc[self.nobs :])
        assert_allclose(ci_95["upper ('oil', 'data')"], upper.iloc[self.nobs :])


def test_invalid():
    with pytest.raises(ValueError, match="Cannot have a seasonal period of 1."):
        ExponentialSmoothing(aust, seasonal=1)
    with pytest.raises(
        TypeError,
        match="seasonal must be integer_like \\(int or np.integer, but not bool or timedelta64\\) or None",
    ):
        ExponentialSmoothing(aust, seasonal=True)
    with pytest.raises(ValueError, match='Invalid initialization method "invalid".'):
        ExponentialSmoothing(aust, initialization_method="invalid")
    with pytest.raises(
        ValueError,
        match='`initial_level` argument must be provided when initialization method is set to "known".',
    ):
        ExponentialSmoothing(aust, initialization_method="known")
    with pytest.raises(
        ValueError,
        match='`initial_trend` argument must be provided for models with a trend component when initialization method is set to "known".',
    ):
        ExponentialSmoothing(
            aust, trend=True, initialization_method="known", initial_level=0
        )
    with pytest.raises(
        ValueError,
        match='`initial_seasonal` argument must be provided for models with a seasonal component when initialization method is set to "known".',
    ):
        ExponentialSmoothing(
            aust, seasonal=4, initialization_method="known", initial_level=0
        )
    for arg in ["initial_level", "initial_trend", "initial_seasonal"]:
        msg = 'Cannot give `%s` argument when initialization is "estimated"' % arg
        with pytest.raises(ValueError, match=msg):
            mod = ExponentialSmoothing(aust, **{arg: 0})
    with pytest.raises(
        ValueError,
        match="Invalid length of initial seasonal values. Must be one of s or s-1, where s is the number of seasonal periods.",
    ):
        ExponentialSmoothing(
            aust,
            seasonal=4,
            initialization_method="known",
            initial_level=0,
            initial_seasonal=0,
        )
    with pytest.raises(
        NotImplementedError, match="ExponentialSmoothing does not support `exog`."
    ):
        mod = ExponentialSmoothing(aust)
        mod.clone(aust, exog=air)


def test_parameterless_model():
    rs = np.random.RandomState(9991611)
    x = np.cumsum(rs.standard_normal(1000))
    ses = ExponentialSmoothing(x, initial_level=x[0], initialization_method="known")
    with ses.fix_params({"smoothing_level": 0.5}):
        res = ses.fit()
    assert np.isnan(res.bse).all()
    assert res.fixed_params == ["smoothing_level"]
