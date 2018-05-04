"""
Tests for SARIMAX models

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os

import warnings
from statsmodels.tsa.statespace import sarimax, tools
from statsmodels.tsa import arima_model as arima
from .results import results_sarimax
from statsmodels.tools import add_constant
from numpy.testing import assert_equal, assert_almost_equal, assert_raises, assert_allclose

current_path = os.path.dirname(os.path.abspath(__file__))

realgdp_path = 'results' + os.sep + 'results_realgdpar_stata.csv'
realgdp_results = pd.read_csv(current_path + os.sep + realgdp_path)

coverage_path = 'results' + os.sep + 'results_sarimax_coverage.csv'
coverage_results = pd.read_csv(current_path + os.sep + coverage_path)

try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except ImportError:
    have_matplotlib = False

IS_WINDOWS = os.name == 'nt'


class TestSARIMAXStatsmodels(object):
    """
    Test ARIMA model using SARIMAX class against statsmodels ARIMA class

    Notes
    -----

    Standard errors are quite good for the OPG case.
    """
    @classmethod
    def setup_class(cls):
        cls.true = results_sarimax.wpi1_stationary
        endog = cls.true['data']

        cls.model_a = arima.ARIMA(endog, order=(1, 1, 1))
        cls.result_a = cls.model_a.fit(disp=-1)

        cls.model_b = sarimax.SARIMAX(endog, order=(1, 1, 1), trend='c',
                                       simple_differencing=True,
                                       hamilton_representation=True)
        cls.result_b = cls.model_b.fit(disp=-1)

    def test_loglike(self):
        assert_allclose(self.result_b.llf, self.result_a.llf)

    def test_aic(self):
        assert_allclose(self.result_b.aic, self.result_a.aic)

    def test_bic(self):
        assert_allclose(self.result_b.bic, self.result_a.bic)

    def test_hqic(self):
        assert_allclose(self.result_b.hqic, self.result_a.hqic)

    def test_mle(self):
        # ARIMA estimates the mean of the process, whereas SARIMAX estimates
        # the intercept. Convert the mean to intercept to compare
        params_a = self.result_a.params.copy()
        params_a[0] = (1 - params_a[1]) * params_a[0]
        assert_allclose(self.result_b.params[:-1], params_a, atol=5e-5)

    def test_bse(self):
        # Test the complex step approximated BSE values
        bse = self.result_b._cov_params_approx(approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[1:-1], self.result_a.bse[1:], atol=1e-5)

    def test_t_test(self):
        import statsmodels.tools._testing as smt
        #self.result_b.pvalues
        #self.result_b._cache['pvalues'] += 1  # use to trigger failure
        smt.check_ttest_tvalues(self.result_b)
        smt.check_ftest_pvalues(self.result_b)


class TestRealGDPARStata(object):
    """
    Includes tests of filtered states and standardized forecast errors.

    Notes
    -----
    Could also test the usual things like standard errors, etc. but those are
    well-tested elsewhere.
    """
    @classmethod
    def setup_class(cls):
        dlgdp = np.log(realgdp_results['value']).diff()[1:].values
        cls.model = sarimax.SARIMAX(dlgdp, order=(12, 0, 0), trend='n',
                                     hamilton_representation=True)
        # Estimated by Stata
        params = [
            .40725515, .18782621, -.01514009, -.01027267, -.03642297,
            .11576416, .02573029, -.00766572, .13506498, .08649569, .06942822,
            -.10685783, .00007999607
        ]
        cls.results = cls.model.filter(params)

    def test_filtered_state(self):
        for i in range(12):
            assert_allclose(
                realgdp_results.iloc[1:]['u%d' % (i+1)],
                self.results.filter_results.filtered_state[i],
                atol=1e-6
            )

    def test_standardized_forecasts_error(self):
        assert_allclose(
            realgdp_results.iloc[1:]['rstd'],
            self.results.filter_results.standardized_forecasts_error[0],
            atol=1e-3
        )


class SARIMAXStataTests(object):
    def test_loglike(self):
        assert_almost_equal(
            self.result.llf,
            self.true['loglike'], 4
        )

    def test_aic(self):
        assert_almost_equal(
            self.result.aic,
            self.true['aic'], 3
        )

    def test_bic(self):
        assert_almost_equal(
            self.result.bic,
            self.true['bic'], 3
        )

    def test_hqic(self):
        hqic = (
            -2*self.result.llf +
            2*np.log(np.log(self.result.nobs_effective)) *
            self.result.params.shape[0]
        )
        assert_almost_equal(
            self.result.hqic,
            hqic, 3
        )

    def test_standardized_forecasts_error(self):
        cython_sfe = self.result.standardized_forecasts_error
        self.result._standardized_forecasts_error = None
        python_sfe = self.result.standardized_forecasts_error
        assert_allclose(cython_sfe, python_sfe)


class ARIMA(SARIMAXStataTests):
    """
    ARIMA model

    Stata arima documentation, Example 1
    """
    @classmethod
    def setup_class(cls, true, *args, **kwargs):
        cls.true = true
        endog = true['data']

        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)

        cls.model = sarimax.SARIMAX(endog, order=(1, 1, 1), trend='c',
                                     *args, **kwargs)

        # Stata estimates the mean of the process, whereas SARIMAX estimates
        # the intercept of the process. Get the intercept.
        intercept = (1 - true['params_ar'][0]) * true['params_mean'][0]
        params = np.r_[intercept, true['params_ar'], true['params_ma'],
                       true['params_variance']]

        cls.result = cls.model.filter(params)

    def test_mle(self):
        result = self.model.fit(disp=-1)
        assert_allclose(
            result.params, self.result.params,
            atol=1e-3
        )


class TestARIMAStationary(ARIMA):
    """
    Notes
    -----

    Standard errors are very good for the OPG and complex step approximation
    cases.
    """
    @classmethod
    def setup_class(cls):
        super(TestARIMAStationary, cls).setup_class(
            results_sarimax.wpi1_stationary
        )

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[1], self.true['se_ar_opg'], atol=1e-7)
        assert_allclose(self.result.bse[2], self.true['se_ma_opg'], atol=1e-7)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-7)
        assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-7)

        # The below tests pass irregularly; they give a sense of the precision
        # available with finite differencing
        # finite difference, non-centered
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-2)
        #     assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-1)

        #     # finite difference, centered
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True).diagonal()**0.5
        #     assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-3)
        #     assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-3)

    def test_bse_oim(self):
        # OIM covariance type
        oim_bse = self.result.cov_params_oim.diagonal()**0.5
        assert_allclose(oim_bse[1], self.true['se_ar_oim'], atol=1e-3)
        assert_allclose(oim_bse[2], self.true['se_ma_oim'], atol=1e-2)

    def test_bse_robust(self):
        robust_oim_bse = self.result.cov_params_robust_oim.diagonal()**0.5
        robust_approx_bse = self.result.cov_params_robust_approx.diagonal()**0.5
        true_robust_bse = np.r_[
            self.true['se_ar_robust'], self.true['se_ma_robust']
        ]

        assert_allclose(robust_oim_bse[1:3], true_robust_bse, atol=1e-2)
        assert_allclose(robust_approx_bse[1:3], true_robust_bse, atol=1e-3)


class TestARIMADiffuse(ARIMA):
    """
    Notes
    -----

    Standard errors are very good for the OPG and quite good for the complex
    step approximation cases.
    """
    @classmethod
    def setup_class(cls, **kwargs):
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = (
            results_sarimax.wpi1_diffuse['initial_variance']
        )
        super(TestARIMADiffuse, cls).setup_class(results_sarimax.wpi1_diffuse,
                                               **kwargs)

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[1], self.true['se_ar_opg'], atol=1e-7)
        assert_allclose(self.result.bse[2], self.true['se_ma_opg'], atol=1e-7)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-4)
        assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-4)

        # The below tests do not pass
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")

        #     # finite difference, non-centered : failure
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-4)
        #     assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-4)

        #     # finite difference, centered : failure
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True).diagonal()**0.5
        #     assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-4)
        #     assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-4)

    def test_bse_oim(self):
        # OIM covariance type
        bse = self.result._cov_params_oim().diagonal()**0.5
        assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-2)
        assert_allclose(bse[2], self.true['se_ma_oim'], atol=1e-1)


class AdditiveSeasonal(SARIMAXStataTests):
    """
    ARIMA model with additive seasonal effects

    Stata arima documentation, Example 2
    """
    @classmethod
    def setup_class(cls, true, *args, **kwargs):
        cls.true = true
        endog = np.log(true['data'])

        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)

        cls.model = sarimax.SARIMAX(
            endog, order=(1, 1, (1, 0, 0, 1)), trend='c', *args, **kwargs
        )

        # Stata estimates the mean of the process, whereas SARIMAX estimates
        # the intercept of the process. Get the intercept.
        intercept = (1 - true['params_ar'][0]) * true['params_mean'][0]
        params = np.r_[intercept, true['params_ar'], true['params_ma'],
                       true['params_variance']]

        cls.result = cls.model.filter(params)

    def test_mle(self):
        result = self.model.fit(disp=-1)
        assert_allclose(
            result.params, self.result.params,
            atol=1e-3
        )


class TestAdditiveSeasonal(AdditiveSeasonal):
    """
    Notes
    -----

    Standard errors are very good for the OPG and quite good for the complex
    step approximation cases.
    """
    @classmethod
    def setup_class(cls):
        super(TestAdditiveSeasonal, cls).setup_class(
            results_sarimax.wpi1_seasonal
        )

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[1], self.true['se_ar_opg'], atol=1e-6)
        assert_allclose(self.result.bse[2:4], self.true['se_ma_opg'], atol=1e-5)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-4)
        assert_allclose(bse[2:4], self.true['se_ma_oim'], atol=1e-4)

        # The below tests pass irregularly; they give a sense of the precision
        # available with finite differencing
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
            
        #     # finite difference, non-centered
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-2)
        #     assert_allclose(bse[2:4], self.true['se_ma_oim'], atol=1e-2)

        #     # finite difference, centered
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True).diagonal()**0.5
        #     assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-3)
        #     assert_allclose(bse[2:4], self.true['se_ma_oim'], atol=1e-3)

    def test_bse_oim(self):
        # OIM covariance type
        bse = self.result._cov_params_oim().diagonal()**0.5
        assert_allclose(bse[1], self.true['se_ar_oim'], atol=1e-2)
        assert_allclose(bse[2:4], self.true['se_ma_oim'], atol=1e-1)


class Airline(SARIMAXStataTests):
    """
    Multiplicative SARIMA model: "Airline" model

    Stata arima documentation, Example 3
    """
    @classmethod
    def setup_class(cls, true, *args, **kwargs):
        cls.true = true
        endog = np.log(true['data'])

        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)

        cls.model = sarimax.SARIMAX(
            endog, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12),
            trend='n', *args, **kwargs
        )

        params = np.r_[true['params_ma'], true['params_seasonal_ma'],
                       true['params_variance']]

        cls.result = cls.model.filter(params)

    def test_mle(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = self.model.fit(disp=-1)
            assert_allclose(
                result.params, self.result.params,
                atol=1e-4
            )


class TestAirlineHamilton(Airline):
    """
    Notes
    -----

    Standard errors are very good for the OPG and complex step approximation
    cases.
    """
    @classmethod
    def setup_class(cls):
        super(TestAirlineHamilton, cls).setup_class(
            results_sarimax.air2_stationary
        )

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[0], self.true['se_ma_opg'], atol=1e-6)
        assert_allclose(self.result.bse[1], self.true['se_seasonal_ma_opg'], atol=1e-6)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-6)
        assert_allclose(bse[1], self.true['se_seasonal_ma_oim'], atol=1e-6)

        # The below tests pass irregularly; they give a sense of the precision
        # available with finite differencing
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        
        #     # finite difference, non-centered
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-2)
        #     assert_allclose(bse[1], self.true['se_seasonal_ma_oim'], atol=1e-2)

        #     # finite difference, centered
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-4)
        #     assert_allclose(bse[1], self.true['se_seasonal_ma_oim'], atol=1e-4)

    def test_bse_oim(self):
        # OIM covariance type
        oim_bse = self.result.cov_params_oim.diagonal()**0.5
        assert_allclose(oim_bse[0], self.true['se_ma_oim'], atol=1e-1)
        assert_allclose(oim_bse[1], self.true['se_seasonal_ma_oim'], atol=1e-1)


class TestAirlineHarvey(Airline):
    """
    Notes
    -----

    Standard errors are very good for the OPG and complex step approximation
    cases.
    """
    @classmethod
    def setup_class(cls):
        super(TestAirlineHarvey, cls).setup_class(
            results_sarimax.air2_stationary, hamilton_representation=False
        )

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[0], self.true['se_ma_opg'], atol=1e-6)
        assert_allclose(self.result.bse[1], self.true['se_seasonal_ma_opg'], atol=1e-6)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-6)
        assert_allclose(bse[1], self.true['se_seasonal_ma_oim'], atol=1e-6)

        # The below tests pass irregularly; they give a sense of the precision
        # available with finite differencing
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        
        #     # finite difference, non-centered
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-2)
        #     assert_allclose(bse[1], self.true['se_seasonal_ma_oim'], atol=1e-2)

        #     # finite difference, centered
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-4)
        #     assert_allclose(bse[1], self.true['se_seasonal_ma_oim'], atol=1e-4)

    def test_bse_oim(self):
        # OIM covariance type
        oim_bse = self.result.cov_params_oim.diagonal()**0.5
        assert_allclose(oim_bse[0], self.true['se_ma_oim'], atol=1e-1)
        assert_allclose(oim_bse[1], self.true['se_seasonal_ma_oim'], atol=1e-1)


class TestAirlineStateDifferencing(Airline):
    """
    Notes
    -----

    Standard errors are very good for the OPG and quite good for the complex
    step approximation cases.
    """
    @classmethod
    def setup_class(cls):
        super(TestAirlineStateDifferencing, cls).setup_class(
            results_sarimax.air2_stationary, simple_differencing=False,
            hamilton_representation=False
        )

    def test_bic(self):
        # Due to diffuse component of the state (which technically changes the
        # BIC calculation - see Durbin and Koopman section 7.4), this is the
        # best we can do for BIC
        assert_almost_equal(
            self.result.bic,
            self.true['bic'], 0
        )

    def test_mle(self):
        result = self.model.fit(method='nm', maxiter=1000, disp=0)
        assert_allclose(
            result.params, self.result.params,
            atol=1e-3)

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[0], self.true['se_ma_opg'], atol=1e-6)
        assert_allclose(self.result.bse[1], self.true['se_seasonal_ma_opg'], atol=1e-6)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-4)
        assert_allclose(bse[1], self.true['se_seasonal_ma_oim'], atol=1e-4)

        # The below tests do not pass
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")

        #     # finite difference, non-centered : failure with NaNs
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-2)
        #     assert_allclose(bse[1], self.true['se_seasonal_ma_oim'], atol=1e-2)

        #     # finite difference, centered : failure with NaNs
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ma_oim'], atol=1e-4)
        #     assert_allclose(bse[1], self.true['se_seasonal_ma_oim'], atol=1e-4)

    def test_bse_oim(self):
        # OIM covariance type
        oim_bse = self.result.cov_params_oim.diagonal()**0.5
        assert_allclose(oim_bse[0], self.true['se_ma_oim'], atol=1e-1)
        assert_allclose(oim_bse[1], self.true['se_seasonal_ma_oim'], atol=1e-1)


class Friedman(SARIMAXStataTests):
    """
    ARMAX model: Friedman quantity theory of money

    Stata arima documentation, Example 4
    """
    @classmethod
    def setup_class(cls, true, exog=None, *args, **kwargs):
        cls.true = true
        endog = np.r_[true['data']['consump']]
        if exog is None:
            exog = add_constant(true['data']['m2'])

        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)

        cls.model = sarimax.SARIMAX(
            endog, exog=exog, order=(1, 0, 1), *args, **kwargs
        )

        params = np.r_[true['params_exog'], true['params_ar'],
                       true['params_ma'], true['params_variance']]

        cls.result = cls.model.filter(params)


class TestFriedmanMLERegression(Friedman):
    """
    Notes
    -----

    Standard errors are very good for the OPG and complex step approximation
    cases.
    """
    @classmethod
    def setup_class(cls):
        super(TestFriedmanMLERegression, cls).setup_class(
            results_sarimax.friedman2_mle
        )

    def test_mle(self):
        result = self.model.fit(disp=-1)
        assert_allclose(
            result.params, self.result.params,
            atol=1e-2, rtol=1e-3
        )

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[0:2], self.true['se_exog_opg'], atol=1e-4)
        assert_allclose(self.result.bse[2], self.true['se_ar_opg'], atol=1e-6)
        assert_allclose(self.result.bse[3], self.true['se_ma_opg'], atol=1e-6)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[0:2], self.true['se_exog_oim'], atol=1e-4)
        assert_allclose(bse[2], self.true['se_ar_oim'], atol=1e-6)
        assert_allclose(bse[3], self.true['se_ma_oim'], atol=1e-6)

        # The below tests pass irregularly; they give a sense of the precision
        # available with finite differencing
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        
        #     # finite difference, non-centered
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_exog_oim'][0], rtol=1)
        #     assert_allclose(bse[1], self.true['se_exog_oim'][1], atol=1e-2)
        #     assert_allclose(bse[2], self.true['se_ar_oim'], atol=1e-2)
        #     assert_allclose(bse[3], self.true['se_ma_oim'], atol=1e-2)

        #     # finite difference, centered
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_exog_oim'][0], rtol=1)
        #     assert_allclose(bse[1], self.true['se_exog_oim'][1], atol=1e-2)
        #     assert_allclose(bse[2], self.true['se_ar_oim'], atol=1e-2)
        #     assert_allclose(bse[3], self.true['se_ma_oim'], atol=1e-2)

    def test_bse_oim(self):
        # OIM covariance type
        bse = self.result.cov_params_oim.diagonal()**0.5
        assert_allclose(bse[0], self.true['se_exog_oim'][0], rtol=1)
        assert_allclose(bse[1], self.true['se_exog_oim'][1], atol=1e-2)
        assert_allclose(bse[2], self.true['se_ar_oim'], atol=1e-2)
        assert_allclose(bse[3], self.true['se_ma_oim'], atol=1e-2)


class TestFriedmanStateRegression(Friedman):
    """
    Notes
    -----

    MLE is not very close and standard errors are not very close for any set of
    parameters.

    This is likely because we're comparing against the model where the
    regression coefficients are also estimated by MLE. So this test should be
    considered just a very basic "sanity" test.
    """
    @classmethod
    def setup_class(cls):
        # Remove the regression coefficients from the parameters, since they
        # will be estimated as part of the state vector
        true = dict(results_sarimax.friedman2_mle)
        exog = add_constant(true['data']['m2']) / 10.

        true['mle_params_exog'] = true['params_exog'][:]
        true['mle_se_exog'] = true['se_exog_opg'][:]

        true['params_exog'] = []
        true['se_exog'] = []

        super(TestFriedmanStateRegression, cls).setup_class(
            true, exog=exog, mle_regression=False
        )

        cls.true_params = np.r_[true['params_exog'], true['params_ar'],
                                 true['params_ma'], true['params_variance']]

        cls.result = cls.model.filter(cls.true_params)


    def test_mle(self):
        result = self.model.fit(disp=-1)
        assert_allclose(
            result.params, self.result.params,
            atol=1e-1, rtol=2e-1
        )

    def test_regression_parameters(self):
        # The regression effects are integrated into the state vector as
        # the last two states (thus the index [-2:]). The filtered
        # estimates of the state vector produced by the Kalman filter and
        # stored in `filtered_state` for these state elements give the
        # recursive least squares estimates of the regression coefficients
        # at each time period. To get the estimates conditional on the
        # entire dataset, use the filtered states from the last time
        # period (thus the index [-1]).
        assert_almost_equal(
            self.result.filter_results.filtered_state[-2:, -1] / 10.,
            self.true['mle_params_exog'], 1
        )

    # Loglikelihood (and so aic, bic) is slightly different when states are
    # integrated into the state vector
    def test_loglike(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_bse(self):
        # test defaults
        assert_equal(self.result.cov_type, 'opg')
        assert_equal(self.result._cov_approx_complex_step, True)
        assert_equal(self.result._cov_approx_centered, False)
        # default covariance type (opg)
        assert_allclose(self.result.bse[0], self.true['se_ar_opg'], atol=1e-2)
        assert_allclose(self.result.bse[1], self.true['se_ma_opg'], atol=1e-2)

    def test_bse_approx(self):
        # complex step
        bse = self.result._cov_params_approx(
            approx_complex_step=True).diagonal()**0.5
        assert_allclose(bse[0], self.true['se_ar_oim'], atol=1e-1)
        assert_allclose(bse[1], self.true['se_ma_oim'], atol=1e-1)

        # The below tests do not pass
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")

        #     # finite difference, non-centered : failure (catastrophic cancellation)
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ar_oim'], atol=1e-3)
        #     assert_allclose(bse[1], self.true['se_ma_oim'], atol=1e-2)

        #     # finite difference, centered : failure (nan)
        #     bse = self.result._cov_params_approx(
        #         approx_complex_step=False, approx_centered=True).diagonal()**0.5
        #     assert_allclose(bse[0], self.true['se_ar_oim'], atol=1e-3)
        #     assert_allclose(bse[1], self.true['se_ma_oim'], atol=1e-3)

    def test_bse_oim(self):
        # OIM covariance type
        bse = self.result._cov_params_oim().diagonal()**0.5
        assert_allclose(bse[0], self.true['se_ar_oim'], atol=1e-1)
        assert_allclose(bse[1], self.true['se_ma_oim'], atol=1e-1)


class TestFriedmanPredict(Friedman):
    """
    ARMAX model: Friedman quantity theory of money, prediction

    Stata arima postestimation documentation, Example 1 - Dynamic forecasts

    This follows the given Stata example, although it is not truly forecasting
    because it compares using the actual data (which is available in the
    example but just not used in the parameter MLE estimation) against dynamic
    prediction of that data. Here `test_predict` matches the first case, and
    `test_dynamic_predict` matches the second.
    """
    @classmethod
    def setup_class(cls):
        super(TestFriedmanPredict, cls).setup_class(
            results_sarimax.friedman2_predict
        )

    # loglike, aic, bic are not the point of this test (they could pass, but we
    # would have to modify the data so that they were calculated to
    # exclude the last 15 observations)
    def test_loglike(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_predict(self):
        assert_almost_equal(
            self.result.predict(),
            self.true['predict'], 3
        )

    def test_dynamic_predict(self):
        dynamic = len(self.true['data']['consump'])-15-1
        assert_almost_equal(
            self.result.predict(dynamic=dynamic),
            self.true['dynamic_predict'], 3
        )


class TestFriedmanForecast(Friedman):
    """
    ARMAX model: Friedman quantity theory of money, forecasts

    Variation on:
    Stata arima postestimation documentation, Example 1 - Dynamic forecasts

    This is a variation of the Stata example, in which the endogenous data is
    actually made to be missing so that the predict command must forecast.

    As another unit test, we also compare against the case in State when
    predict is used against missing data (so forecasting) with the dynamic
    option also included. Note, however, that forecasting in State space models
    amounts to running the Kalman filter against missing datapoints, so it is
    not clear whether "dynamic" forecasting (where instead of missing
    datapoints for lags, we plug in previous forecasted endog values) is
    meaningful.
    """
    @classmethod
    def setup_class(cls):
        true = dict(results_sarimax.friedman2_predict)

        true['forecast_data'] = {
            'consump': true['data']['consump'][-15:],
            'm2': true['data']['m2'][-15:]
        }
        true['data'] = {
            'consump': true['data']['consump'][:-15],
            'm2': true['data']['m2'][:-15]
        }

        super(TestFriedmanForecast, cls).setup_class(true)

        cls.result = cls.model.filter(cls.result.params)

    # loglike, aic, bic are not the point of this test (they could pass, but we
    # would have to modify the data so that they were calculated to
    # exclude the last 15 observations)
    def test_loglike(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_forecast(self):
        end = len(self.true['data']['consump'])+15-1
        exog = add_constant(self.true['forecast_data']['m2'])
        assert_almost_equal(
            self.result.predict(end=end, exog=exog),
            self.true['forecast'], 3
        )

    def test_dynamic_forecast(self):
        end = len(self.true['data']['consump'])+15-1
        dynamic = len(self.true['data']['consump'])-1
        exog = add_constant(self.true['forecast_data']['m2'])
        assert_almost_equal(
            self.result.predict(end=end, dynamic=dynamic, exog=exog),
            self.true['dynamic_forecast'], 3
        )

class SARIMAXCoverageTest(object):
    @classmethod
    def setup_class(cls, i, decimal=4, endog=None, *args, **kwargs):
        # Dataset
        if endog is None:
            endog = results_sarimax.wpi1_data

        # Loglikelihood, parameters
        cls.true_loglike = coverage_results.loc[i]['llf']
        cls.true_params = np.array([float(x) for x in coverage_results.loc[i]['parameters'].split(',')])
        # Stata reports the standard deviation; make it the variance
        cls.true_params[-1] = cls.true_params[-1]**2

        # Test parameters
        cls.decimal = decimal

        # Compare using the Hamilton representation and simple differencing
        kwargs.setdefault('simple_differencing', True)
        kwargs.setdefault('hamilton_representation', True)

        cls.model = sarimax.SARIMAX(endog, *args, **kwargs)

    def test_loglike(self):
        self.result = self.model.filter(self.true_params)

        assert_allclose(
            self.result.llf,
            self.true_loglike,
            atol=0.7 * 10**(-self.decimal)
        )

    def test_start_params(self):
        # just a quick test that start_params isn't throwing an exception
        # (other than related to invertibility)
        stat, inv = self.model.enforce_stationarity, self.model.enforce_invertibility
        self.model.enforce_stationarity = False
        self.model.enforce_invertibility = False
        self.model.start_params
        self.model.enforce_stationarity, self.model.enforce_invertibility = stat, inv

    def test_transform_untransform(self):
        stat, inv = self.model.enforce_stationarity, self.model.enforce_invertibility
        true_constrained = self.true_params

        # Sometimes the parameters given by Stata are not stationary and / or
        # invertible, so we need to skip those transformations for those
        # parameter sets
        self.model.update(self.true_params)
        contracted_polynomial_seasonal_ar = self.model.polynomial_seasonal_ar[self.model.polynomial_seasonal_ar.nonzero()]
        self.model.enforce_stationarity = (
            (self.model.k_ar == 0 or tools.is_invertible(np.r_[1, -self.model.polynomial_ar[1:]])) and
            (len(contracted_polynomial_seasonal_ar) <= 1 or tools.is_invertible(np.r_[1, -contracted_polynomial_seasonal_ar[1:]]))
        )
        contracted_polynomial_seasonal_ma = self.model.polynomial_seasonal_ma[self.model.polynomial_seasonal_ma.nonzero()]
        self.model.enforce_invertibility = (
            (self.model.k_ma == 0 or tools.is_invertible(np.r_[1, self.model.polynomial_ma[1:]])) and
            (len(contracted_polynomial_seasonal_ma) <= 1 or tools.is_invertible(np.r_[1, contracted_polynomial_seasonal_ma[1:]]))
        )

        unconstrained = self.model.untransform_params(true_constrained)
        constrained = self.model.transform_params(unconstrained)

        assert_almost_equal(constrained, true_constrained, 4)
        self.model.enforce_stationarity, self.model.enforce_invertibility = stat, inv

    def test_results(self):
        self.result = self.model.filter(self.true_params)

        # Just make sure that no exceptions are thrown during summary
        self.result.summary()

        # Make sure that no exceptions are thrown during plot_diagnostics
        if have_matplotlib:
            fig = self.result.plot_diagnostics()
            plt.close(fig)

        # And make sure no expections are thrown calculating any of the
        # covariance matrix types
        self.result.cov_params_default
        self.result.cov_params_approx
        self.result.cov_params_oim
        self.result.cov_params_opg
        self.result.cov_params_robust_oim
        self.result.cov_params_robust_approx

    def test_predict(self):
        result = self.model.filter(self.true_params)
        # Test predict does not throw exceptions, and produces the right shaped
        # output
        predict = result.predict()
        assert_equal(predict.shape, (self.model.nobs,))

        predict = result.predict(start=10, end=20)
        assert_equal(predict.shape, (11,))

        predict = result.predict(start=10, end=20, dynamic=10)
        assert_equal(predict.shape, (11,))

        # Test forecasts
        if self.model.k_exog == 0:
            predict = result.predict(start=self.model.nobs,
                                 end=self.model.nobs+10, dynamic=-10)
            assert_equal(predict.shape, (11,))

            predict = result.predict(start=self.model.nobs,
                                     end=self.model.nobs+10, dynamic=-10)

            forecast = result.forecast()
            assert_equal(forecast.shape, (1,))

            forecast = result.forecast(10)
            assert_equal(forecast.shape, (10,))
        else:
            exog = np.r_[[0]*self.model.k_exog*11].reshape(11, self.model.k_exog)

            predict = result.predict(start=self.model.nobs,
                                     end=self.model.nobs+10, dynamic=-10,
                                     exog=exog)
            assert_equal(predict.shape, (11,))

            predict = result.predict(start=self.model.nobs,
                                     end=self.model.nobs+10, dynamic=-10,
                                     exog=exog)

            exog = np.r_[[0]*self.model.k_exog].reshape(1, self.model.k_exog)
            forecast = result.forecast(exog=exog)
            assert_equal(forecast.shape, (1,))

    def test_init_keys_replicate(self):
        mod1 = self.model

        kwargs = self.model._get_init_kwds()
        endog = mod1.data.orig_endog
        exog = mod1.data.orig_exog

        model2 = sarimax.SARIMAX(endog, exog, **kwargs)
        res1 = self.model.filter(self.true_params)
        res2 = model2.filter(self.true_params)
        rtol = 1e-6 if IS_WINDOWS else 1e-13
        assert_allclose(res2.llf, res1.llf, rtol=rtol)


class Test_ar(SARIMAXCoverageTest):
    # // AR: (p,0,0) x (0,0,0,0)
    # arima wpi, arima(3,0,0) noconstant vce(oim)
    # save_results 1
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,0)
        super(Test_ar, cls).setup_class(0, *args, **kwargs)

class Test_ar_as_polynomial(SARIMAXCoverageTest):
    # // AR: (p,0,0) x (0,0,0,0)
    # arima wpi, arima(3,0,0) noconstant vce(oim)
    # save_results 1
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = ([1,1,1],0,0)
        super(Test_ar_as_polynomial, cls).setup_class(0, *args, **kwargs)

class Test_ar_trend_c(SARIMAXCoverageTest):
    # // 'c'
    # arima wpi c, arima(3,0,0) noconstant vce(oim)
    # save_results 2
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,0)
        kwargs['trend'] = 'c'
        super(Test_ar_trend_c, cls).setup_class(1, *args, **kwargs)
        # Modify true params to convert from mean to intercept form
        cls.true_params[0] = (1 - cls.true_params[1:4].sum()) * cls.true_params[0]

class Test_ar_trend_ct(SARIMAXCoverageTest):
    # // 'ct'
    # arima wpi c t, arima(3,0,0) noconstant vce(oim)
    # save_results 3
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,0)
        kwargs['trend'] = 'ct'
        super(Test_ar_trend_ct, cls).setup_class(2, *args, **kwargs)
        # Modify true params to convert from mean to intercept form
        cls.true_params[:2] = (1 - cls.true_params[2:5].sum()) * cls.true_params[:2]

class Test_ar_trend_polynomial(SARIMAXCoverageTest):
    # // polynomial [1,0,0,1]
    # arima wpi c t3, arima(3,0,0) noconstant vce(oim)
    # save_results 4
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,0)
        kwargs['trend'] = [1,0,0,1]
        super(Test_ar_trend_polynomial, cls).setup_class(3, *args, **kwargs)
        # Modify true params to convert from mean to intercept form
        cls.true_params[:2] = (1 - cls.true_params[2:5].sum()) * cls.true_params[:2]

class Test_ar_diff(SARIMAXCoverageTest):
    # // AR and I(d): (p,d,0) x (0,0,0,0)
    # arima wpi, arima(3,2,0) noconstant vce(oim)
    # save_results 5
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,2,0)
        super(Test_ar_diff, cls).setup_class(4, *args, **kwargs)

class Test_ar_seasonal_diff(SARIMAXCoverageTest):
    # // AR and I(D): (p,0,0) x (0,D,0,s)
    # arima wpi, arima(3,0,0) sarima(0,2,0,4) noconstant vce(oim)
    # save_results 6
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,0)
        kwargs['seasonal_order'] = (0,2,0,4)
        super(Test_ar_seasonal_diff, cls).setup_class(5, *args, **kwargs)

class Test_ar_diffuse(SARIMAXCoverageTest):
    # // AR and diffuse initialization
    # arima wpi, arima(3,0,0) noconstant vce(oim) diffuse
    # save_results 7
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,0)
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1e9
        super(Test_ar_diffuse, cls).setup_class(6, *args, **kwargs)

class Test_ar_no_enforce(SARIMAXCoverageTest):
    # // AR: (p,0,0) x (0,0,0,0)
    # arima wpi, arima(3,0,0) noconstant vce(oim)
    # save_results 1
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,0)
        kwargs['enforce_stationarity'] = False
        kwargs['enforce_invertibility'] = False
        kwargs['initial_variance'] = 1e9
        kwargs['loglikelihood_burn'] = 0
        super(Test_ar_no_enforce, cls).setup_class(6, *args, **kwargs)
        # Reset loglikelihood burn, which gets automatically set to the number
        # of states if enforce_stationarity = False
        cls.model.ssm.loglikelihood_burn = 0

    def test_init_keys_replicate(self):
        mod1 = self.model

        kwargs = self.model._get_init_kwds()
        endog = mod1.data.orig_endog
        exog = mod1.data.orig_exog

        model2 = sarimax.SARIMAX(endog, exog, **kwargs)
        # Fixes needed for edge case model
        model2.ssm._initial_variance = mod1.ssm._initial_variance
        model2.ssm.initialization = mod1.ssm.initialization

        res1 = self.model.filter(self.true_params)
        res2 = model2.filter(self.true_params)
        rtol = 1e-6 if IS_WINDOWS else 1e-13
        assert_allclose(res2.llf, res1.llf, rtol=rtol)


class Test_ar_exogenous(SARIMAXCoverageTest):
    # // ARX
    # arima wpi x, arima(3,0,0) noconstant vce(oim)
    # save_results 8
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,0)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        super(Test_ar_exogenous, cls).setup_class(7, *args, **kwargs)

class Test_ar_exogenous_in_state(SARIMAXCoverageTest):
    # // ARX
    # arima wpi x, arima(3,0,0) noconstant vce(oim)
    # save_results 8
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,0)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        kwargs['mle_regression'] = False
        super(Test_ar_exogenous_in_state, cls).setup_class(7, *args, **kwargs)
        cls.true_regression_coefficient = cls.true_params[0]
        cls.true_params = cls.true_params[1:]

    def test_loglike(self):
        # Regression in the state vector gives a different loglikelihood, so
        # just check that it's approximately the same
        self.result = self.model.filter(self.true_params)

        assert_allclose(
            self.result.llf,
            self.true_loglike,
            atol=2
        )

    def test_regression_coefficient(self):
        # Test that the regression coefficient (estimated as the last filtered
        # state estimate for the regression state) is the same as the Stata
        # MLE state
        self.result = self.model.filter(self.true_params)

        assert_allclose(
            self.result.filter_results.filtered_state[3][-1],
            self.true_regression_coefficient,
            self.decimal
        )

class Test_ma(SARIMAXCoverageTest):
    # // MA: (0,0,q) x (0,0,0,0)
    # arima wpi, arima(0,0,3) noconstant vce(oim)
    # save_results 9
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,3)
        super(Test_ma, cls).setup_class(8, *args, **kwargs)

class Test_ma_as_polynomial(SARIMAXCoverageTest):
    # // MA: (0,0,q) x (0,0,0,0)
    # arima wpi, arima(0,0,3) noconstant vce(oim)
    # save_results 9
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,[1,1,1])
        super(Test_ma_as_polynomial, cls).setup_class(8, *args, **kwargs)

class Test_ma_trend_c(SARIMAXCoverageTest):
    # // 'c'
    # arima wpi c, arima(0,0,3) noconstant vce(oim)
    # save_results 10
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,3)
        kwargs['trend'] = 'c'
        super(Test_ma_trend_c, cls).setup_class(9, *args, **kwargs)

class Test_ma_trend_ct(SARIMAXCoverageTest):
    # // 'ct'
    # arima wpi c t, arima(0,0,3) noconstant vce(oim)
    # save_results 11
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,3)
        kwargs['trend'] = 'ct'
        super(Test_ma_trend_ct, cls).setup_class(10, *args, **kwargs)

class Test_ma_trend_polynomial(SARIMAXCoverageTest):
    # // polynomial [1,0,0,1]
    # arima wpi c t3, arima(0,0,3) noconstant vce(oim)
    # save_results 12
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,3)
        kwargs['trend'] = [1,0,0,1]
        super(Test_ma_trend_polynomial, cls).setup_class(11, *args, **kwargs)

class Test_ma_diff(SARIMAXCoverageTest):
    # // MA and I(d): (0,d,q) x (0,0,0,0)
    # arima wpi, arima(0,2,3) noconstant vce(oim)
    # save_results 13
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,2,3)
        super(Test_ma_diff, cls).setup_class(12, *args, **kwargs)

class Test_ma_seasonal_diff(SARIMAXCoverageTest):
    # // MA and I(D): (p,0,0) x (0,D,0,s)
    # arima wpi, arima(0,0,3) sarima(0,2,0,4) noconstant vce(oim)
    # save_results 14
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,3)
        kwargs['seasonal_order'] = (0,2,0,4)
        super(Test_ma_seasonal_diff, cls).setup_class(13, *args, **kwargs)

class Test_ma_diffuse(SARIMAXCoverageTest):
    # // MA and diffuse initialization
    # arima wpi, arima(0,0,3) noconstant vce(oim) diffuse
    # save_results 15
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,3)
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1e9
        super(Test_ma_diffuse, cls).setup_class(14, *args, **kwargs)

class Test_ma_exogenous(SARIMAXCoverageTest):
    # // MAX
    # arima wpi x, arima(0,0,3) noconstant vce(oim)
    # save_results 16
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,3)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        super(Test_ma_exogenous, cls).setup_class(15, *args, **kwargs)

class Test_arma(SARIMAXCoverageTest):
    # // ARMA: (p,0,q) x (0,0,0,0)
    # arima wpi, arima(3,0,3) noconstant vce(oim)
    # save_results 17
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,3)
        super(Test_arma, cls).setup_class(16, *args, **kwargs)

class Test_arma_trend_c(SARIMAXCoverageTest):
    # // 'c'
    # arima wpi c, arima(3,0,2) noconstant vce(oim)
    # save_results 18
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,2)
        kwargs['trend'] = 'c'
        super(Test_arma_trend_c, cls).setup_class(17, *args, **kwargs)
        # Modify true params to convert from mean to intercept form
        cls.true_params[:1] = (1 - cls.true_params[1:4].sum()) * cls.true_params[:1]

class Test_arma_trend_ct(SARIMAXCoverageTest):
    # // 'ct'
    # arima wpi c t, arima(3,0,2) noconstant vce(oim)
    # save_results 19
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,2)
        kwargs['trend'] = 'ct'
        super(Test_arma_trend_ct, cls).setup_class(18, *args, **kwargs)
        # Modify true params to convert from mean to intercept form
        cls.true_params[:2] = (1 - cls.true_params[2:5].sum()) * cls.true_params[:2]

class Test_arma_trend_polynomial(SARIMAXCoverageTest):
    # // polynomial [1,0,0,1]
    # arima wpi c t3, arima(3,0,2) noconstant vce(oim)
    # save_results 20
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,2)
        kwargs['trend'] = [1,0,0,1]
        super(Test_arma_trend_polynomial, cls).setup_class(19, *args, **kwargs)
        # Modify true params to convert from mean to intercept form
        cls.true_params[:2] = (1 - cls.true_params[2:5].sum()) * cls.true_params[:2]

class Test_arma_diff(SARIMAXCoverageTest):
    # // ARMA and I(d): (p,d,q) x (0,0,0,0)
    # arima wpi, arima(3,2,2) noconstant vce(oim)
    # save_results 21
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,2,2)
        super(Test_arma_diff, cls).setup_class(20, *args, **kwargs)

class Test_arma_seasonal_diff(SARIMAXCoverageTest):
    # // ARMA and I(D): (p,0,q) x (0,D,0,s)
    # arima wpi, arima(3,0,2) sarima(0,2,0,4) noconstant vce(oim)
    # save_results 22
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,2)
        kwargs['seasonal_order'] = (0,2,0,4)
        super(Test_arma_seasonal_diff, cls).setup_class(21, *args, **kwargs)

class Test_arma_diff_seasonal_diff(SARIMAXCoverageTest):
    # // ARMA and I(d) and I(D): (p,d,q) x (0,D,0,s)
    # arima wpi, arima(3,2,2) sarima(0,2,0,4) noconstant vce(oim)
    # save_results 23
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,2,2)
        kwargs['seasonal_order'] = (0,2,0,4)
        super(Test_arma_diff_seasonal_diff, cls).setup_class(22, *args, **kwargs)

class Test_arma_diffuse(SARIMAXCoverageTest):
    # // ARMA and diffuse initialization
    # arima wpi, arima(3,0,2) noconstant vce(oim) diffuse
    # save_results 24
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,2)
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1e9
        super(Test_arma_diffuse, cls).setup_class(23, *args, **kwargs)

class Test_arma_exogenous(SARIMAXCoverageTest):
    # // ARMAX
    # arima wpi x, arima(3,0,2) noconstant vce(oim)
    # save_results 25
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,0,2)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        super(Test_arma_exogenous, cls).setup_class(24, *args, **kwargs)

class Test_seasonal_ar(SARIMAXCoverageTest):
    # // SAR: (0,0,0) x (P,0,0,s)
    # arima wpi, sarima(3,0,0,4) noconstant vce(oim)
    # save_results 26
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (3,0,0,4)
        super(Test_seasonal_ar, cls).setup_class(25, *args, **kwargs)

class Test_seasonal_ar_as_polynomial(SARIMAXCoverageTest):
    # // SAR: (0,0,0) x (P,0,0,s)
    # arima wpi, sarima(3,0,0,4) noconstant vce(oim)
    # save_results 26
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = ([1,1,1],0,0,4)
        super(Test_seasonal_ar_as_polynomial, cls).setup_class(25, *args, **kwargs)

class Test_seasonal_ar_trend_c(SARIMAXCoverageTest):
    # // 'c'
    # arima wpi c, sarima(3,0,0,4) noconstant vce(oim)
    # save_results 27
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (3,0,0,4)
        kwargs['trend'] = 'c'
        super(Test_seasonal_ar_trend_c, cls).setup_class(26, *args, **kwargs)
        # Modify true params to convert from mean to intercept form
        cls.true_params[:1] = (1 - cls.true_params[1:4].sum()) * cls.true_params[:1]

class Test_seasonal_ar_trend_ct(SARIMAXCoverageTest):
    # // 'ct'
    # arima wpi c t, sarima(3,0,0,4) noconstant vce(oim)
    # save_results 28
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (3,0,0,4)
        kwargs['trend'] = 'ct'
        super(Test_seasonal_ar_trend_ct, cls).setup_class(27, *args, **kwargs)
        # Modify true params to convert from mean to intercept form
        cls.true_params[:2] = (1 - cls.true_params[2:5].sum()) * cls.true_params[:2]

class Test_seasonal_ar_trend_polynomial(SARIMAXCoverageTest):
    # // polynomial [1,0,0,1]
    # arima wpi c t3, sarima(3,0,0,4) noconstant vce(oim)
    # save_results 29
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (3,0,0,4)
        kwargs['trend'] = [1,0,0,1]
        super(Test_seasonal_ar_trend_polynomial, cls).setup_class(28, *args, **kwargs)
        # Modify true params to convert from mean to intercept form
        cls.true_params[:2] = (1 - cls.true_params[2:5].sum()) * cls.true_params[:2]

class Test_seasonal_ar_diff(SARIMAXCoverageTest):
    # // SAR and I(d): (0,d,0) x (P,0,0,s)
    # arima wpi, arima(0,2,0) sarima(3,0,0,4) noconstant vce(oim)
    # save_results 30
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,2,0)
        kwargs['seasonal_order'] = (3,0,0,4)
        super(Test_seasonal_ar_diff, cls).setup_class(29, *args, **kwargs)

class Test_seasonal_ar_seasonal_diff(SARIMAXCoverageTest):
    # // SAR and I(D): (0,0,0) x (P,D,0,s)
    # arima wpi, sarima(3,2,0,4) noconstant vce(oim)
    # save_results 31
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (3,2,0,4)
        super(Test_seasonal_ar_seasonal_diff, cls).setup_class(30, *args, **kwargs)

class Test_seasonal_ar_diffuse(SARIMAXCoverageTest):
    # // SAR and diffuse initialization
    # arima wpi, sarima(3,0,0,4) noconstant vce(oim) diffuse
    # save_results 32
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (3,0,0,4)
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1e9
        super(Test_seasonal_ar_diffuse, cls).setup_class(31, *args, **kwargs)

class Test_seasonal_ar_exogenous(SARIMAXCoverageTest):
    # // SARX
    # arima wpi x, sarima(3,0,0,4) noconstant vce(oim)
    # save_results 33
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (3,0,0,4)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        super(Test_seasonal_ar_exogenous, cls).setup_class(32, *args, **kwargs)

class Test_seasonal_ma(SARIMAXCoverageTest):
    # // SMA
    # arima wpi, sarima(0,0,3,4) noconstant vce(oim)
    # save_results 34
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (0,0,3,4)
        super(Test_seasonal_ma, cls).setup_class(33, *args, **kwargs)

class Test_seasonal_ma_as_polynomial(SARIMAXCoverageTest):
    # // SMA
    # arima wpi, sarima(0,0,3,4) noconstant vce(oim)
    # save_results 34
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (0,0,[1,1,1],4)
        super(Test_seasonal_ma_as_polynomial, cls).setup_class(33, *args, **kwargs)

class Test_seasonal_ma_trend_c(SARIMAXCoverageTest):
    # // 'c'
    # arima wpi c, sarima(0,0,3,4) noconstant vce(oim)
    # save_results 35
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (0,0,3,4)
        kwargs['trend'] = 'c'
        kwargs['decimal'] = 3
        super(Test_seasonal_ma_trend_c, cls).setup_class(34, *args, **kwargs)

class Test_seasonal_ma_trend_ct(SARIMAXCoverageTest):
    # // 'ct'
    # arima wpi c t, sarima(0,0,3,4) noconstant vce(oim)
    # save_results 36
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (0,0,3,4)
        kwargs['trend'] = 'ct'
        super(Test_seasonal_ma_trend_ct, cls).setup_class(35, *args, **kwargs)

class Test_seasonal_ma_trend_polynomial(SARIMAXCoverageTest):
    # // polynomial [1,0,0,1]
    # arima wpi c t3, sarima(0,0,3,4) noconstant vce(oim)
    # save_results 37
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (0,0,3,4)
        kwargs['trend'] = [1,0,0,1]
        kwargs['decimal'] = 3
        super(Test_seasonal_ma_trend_polynomial, cls).setup_class(36, *args, **kwargs)

class Test_seasonal_ma_diff(SARIMAXCoverageTest):
    # // SMA and I(d): (0,d,0) x (0,0,Q,s)
    # arima wpi, arima(0,2,0) sarima(0,0,3,4) noconstant vce(oim)
    # save_results 38
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,2,0)
        kwargs['seasonal_order'] = (0,0,3,4)
        super(Test_seasonal_ma_diff, cls).setup_class(37, *args, **kwargs)

class Test_seasonal_ma_seasonal_diff(SARIMAXCoverageTest):
    # // SMA and I(D): (0,0,0) x (0,D,Q,s)
    # arima wpi, sarima(0,2,3,4) noconstant vce(oim)
    # save_results 39
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (0,2,3,4)
        super(Test_seasonal_ma_seasonal_diff, cls).setup_class(38, *args, **kwargs)

class Test_seasonal_ma_diffuse(SARIMAXCoverageTest):
    # // SMA and diffuse initialization
    # arima wpi, sarima(0,0,3,4) noconstant vce(oim) diffuse
    # save_results 40
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (0,0,3,4)
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1e9
        super(Test_seasonal_ma_diffuse, cls).setup_class(39, *args, **kwargs)

class Test_seasonal_ma_exogenous(SARIMAXCoverageTest):
    # // SMAX
    # arima wpi x, sarima(0,0,3,4) noconstant vce(oim)
    # save_results 41
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (0,0,3,4)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        super(Test_seasonal_ma_exogenous, cls).setup_class(40, *args, **kwargs)

class Test_seasonal_arma(SARIMAXCoverageTest):
    # // SARMA: (0,0,0) x (P,0,Q,s)
    # arima wpi, sarima(3,0,2,4) noconstant vce(oim)
    # save_results 42
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (3,0,2,4)
        super(Test_seasonal_arma, cls).setup_class(41, *args, **kwargs)

class Test_seasonal_arma_trend_c(SARIMAXCoverageTest):
    # // 'c'
    # arima wpi c, sarima(3,0,2,4) noconstant vce(oim)
    # save_results 43
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (3,0,2,4)
        kwargs['trend'] = 'c'
        super(Test_seasonal_arma_trend_c, cls).setup_class(42, *args, **kwargs)
        # Modify true params to convert from mean to intercept form
        cls.true_params[:1] = (1 - cls.true_params[1:4].sum()) * cls.true_params[:1]

class Test_seasonal_arma_trend_ct(SARIMAXCoverageTest):
    # // 'ct'
    # arima wpi c t, sarima(3,0,2,4) noconstant vce(oim)
    # save_results 44
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (3,0,2,4)
        kwargs['trend'] = 'ct'
        super(Test_seasonal_arma_trend_ct, cls).setup_class(43, *args, **kwargs)
        # Modify true params to convert from mean to intercept form
        cls.true_params[:2] = (1 - cls.true_params[2:5].sum()) * cls.true_params[:2]

class Test_seasonal_arma_trend_polynomial(SARIMAXCoverageTest):
    # // polynomial [1,0,0,1]
    # arima wpi c t3, sarima(3,0,2,4) noconstant vce(oim)
    # save_results 45
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (3,0,2,4)
        kwargs['trend'] = [1,0,0,1]
        kwargs['decimal'] = 3
        super(Test_seasonal_arma_trend_polynomial, cls).setup_class(44, *args, **kwargs)
        # Modify true params to convert from mean to intercept form
        cls.true_params[:2] = (1 - cls.true_params[2:5].sum()) * cls.true_params[:2]

    def test_results(self):
        self.result = self.model.filter(self.true_params)

        # Just make sure that no exceptions are thrown during summary
        self.result.summary()

        # Make sure that no exceptions are thrown during plot_diagnostics
        if have_matplotlib:
            fig = self.result.plot_diagnostics()
            plt.close(fig)

        # And make sure no expections are thrown calculating any of the
        # covariance matrix types
        self.result.cov_params_default
        # Known failure due to the complex step inducing non-stationary
        # parameters, causing a failure in the solve_discrete_lyapunov call
        # self.result.cov_params_approx
        self.result.cov_params_oim
        self.result.cov_params_opg

class Test_seasonal_arma_diff(SARIMAXCoverageTest):
    # // SARMA and I(d): (0,d,0) x (P,0,Q,s)
    # arima wpi, arima(0,2,0) sarima(3,0,2,4) noconstant vce(oim)
    # save_results 46
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,2,0)
        kwargs['seasonal_order'] = (3,0,2,4)
        super(Test_seasonal_arma_diff, cls).setup_class(45, *args, **kwargs)

class Test_seasonal_arma_seasonal_diff(SARIMAXCoverageTest):
    # // SARMA and I(D): (0,0,0) x (P,D,Q,s)
    # arima wpi, sarima(3,2,2,4) noconstant vce(oim)
    # save_results 47
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (3,2,2,4)
        super(Test_seasonal_arma_seasonal_diff, cls).setup_class(46, *args, **kwargs)

class Test_seasonal_arma_diff_seasonal_diff(SARIMAXCoverageTest):
    # // SARMA and I(d) and I(D): (0,d,0) x (P,D,Q,s)
    # arima wpi, arima(0,2,0) sarima(3,2,2,4) noconstant vce(oim)
    # save_results 48
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,2,0)
        kwargs['seasonal_order'] = (3,2,2,4)
        super(Test_seasonal_arma_diff_seasonal_diff, cls).setup_class(47, *args, **kwargs)

    def test_results(self):
        self.result = self.model.filter(self.true_params)

        # Just make sure that no exceptions are thrown during summary
        self.result.summary()

        # Make sure that no exceptions are thrown during plot_diagnostics
        if have_matplotlib:
            fig = self.result.plot_diagnostics()
            plt.close(fig)

        # And make sure no expections are thrown calculating any of the
        # covariance matrix types
        self.result.cov_params_default
        # Known failure due to the complex step inducing non-stationary
        # parameters, causing a failure in the solve_discrete_lyapunov call
        # self.result.cov_params_approx
        self.result.cov_params_oim
        self.result.cov_params_opg

class Test_seasonal_arma_diffuse(SARIMAXCoverageTest):
    # // SARMA and diffuse initialization
    # arima wpi, sarima(3,0,2,4) noconstant vce(oim) diffuse
    # save_results 49
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (3,0,2,4)
        kwargs['decimal'] = 3
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1e9
        super(Test_seasonal_arma_diffuse, cls).setup_class(48, *args, **kwargs)

class Test_seasonal_arma_exogenous(SARIMAXCoverageTest):
    # // SARMAX
    # arima wpi x, sarima(3,0,2,4) noconstant vce(oim)
    # save_results 50
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (0,0,0)
        kwargs['seasonal_order'] = (3,0,2,4)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        super(Test_seasonal_arma_exogenous, cls).setup_class(49, *args, **kwargs)

class Test_sarimax_exogenous(SARIMAXCoverageTest):
    # // SARIMAX and exogenous
    # arima wpi x, arima(3,2,2) sarima(3,2,2,4) noconstant vce(oim)
    # save_results 51
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,2,2)
        kwargs['seasonal_order'] = (3,2,2,4)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        super(Test_sarimax_exogenous, cls).setup_class(50, *args, **kwargs)

    def test_results_params(self):
        result = self.model.filter(self.true_params)
        assert_allclose(self.true_params[1:4], result.arparams)
        assert_allclose(self.true_params[4:6], result.maparams)
        assert_allclose(self.true_params[6:9], result.seasonalarparams)
        assert_allclose(self.true_params[9:11], result.seasonalmaparams)

class Test_sarimax_exogenous_not_hamilton(SARIMAXCoverageTest):
    # // SARIMAX and exogenous
    # arima wpi x, arima(3,2,2) sarima(3,2,2,4) noconstant vce(oim)
    # save_results 51
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,2,2)
        kwargs['seasonal_order'] = (3,2,2,4)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        kwargs['hamilton_representation'] = False
        kwargs['simple_differencing'] = False
        super(Test_sarimax_exogenous_not_hamilton, cls).setup_class(50, *args, **kwargs)

class Test_sarimax_exogenous_diffuse(SARIMAXCoverageTest):
    # // SARIMAX and exogenous diffuse
    # arima wpi x, arima(3,2,2) sarima(3,2,2,4) noconstant vce(oim) diffuse
    # save_results 52
    @classmethod
    def setup_class(cls, *args, **kwargs):
        kwargs['order'] = (3,2,2)
        kwargs['seasonal_order'] = (3,2,2,4)
        endog = results_sarimax.wpi1_data
        kwargs['exog'] = (endog - np.floor(endog))**2
        kwargs['decimal'] = 2
        kwargs['initialization'] = 'approximate_diffuse'
        kwargs['initial_variance'] = 1e9
        super(Test_sarimax_exogenous_diffuse, cls).setup_class(51, *args, **kwargs)

class Test_arma_exog_trend_polynomial_missing(SARIMAXCoverageTest):
    # // ARMA and exogenous and trend polynomial and missing
    # gen wpi2 = wpi
    # replace wpi2 = . in 10/19
    # arima wpi2 x c t3, arima(3,0,2) noconstant vce(oim)
    # save_results 53
    @classmethod
    def setup_class(cls, *args, **kwargs):
        endog = np.r_[results_sarimax.wpi1_data]
        # Note we're using the non-missing exog data
        kwargs['exog'] = ((endog - np.floor(endog))**2)[1:]
        endog[9:19] = np.nan
        endog = endog[1:] - endog[:-1]
        endog[9] = np.nan
        kwargs['order'] = (3,0,2)
        kwargs['trend'] = [0,0,0,1]
        kwargs['decimal'] = 1
        super(Test_arma_exog_trend_polynomial_missing, cls).setup_class(52, endog=endog, *args, **kwargs)
        # Modify true params to convert from mean to intercept form
        cls.true_params[0] = (1 - cls.true_params[2:5].sum()) * cls.true_params[0]

# Miscellaneous coverage tests
def test_simple_time_varying():
    # This tests time-varying parameters regression when in fact the parameters
    # are not time-varying, and in fact the regression fit is perfect
    endog = np.arange(100)*1.0
    exog = 2*endog
    mod = sarimax.SARIMAX(endog, exog=exog, order=(0,0,0), time_varying_regression=True, mle_regression=False)

    # Ignore the warning that MLE doesn't converge
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(disp=-1)

    # Test that the estimated variances of the errors are essentially zero
    # 5 digits necessary to accommodate 32-bit numpy / scipy with OpenBLAS 0.2.18
    assert_almost_equal(res.params, [0, 0], 5)

    # Test that the time-varying coefficients are all 0.5 (except the first
    # one)
    assert_almost_equal(res.filter_results.filtered_state[0][1:], [0.5]*99, 9)

def test_invalid_time_varying():
    assert_raises(ValueError, sarimax.SARIMAX, endog=[1,2,3], mle_regression=True, time_varying_regression=True)

def test_manual_stationary_initialization():
    endog = results_sarimax.wpi1_data

    # Create the first model to compare against
    mod1 = sarimax.SARIMAX(endog, order=(3,0,0))
    res1 = mod1.filter([0.5,0.2,0.1,1])

    # Create a second model with "known" initialization
    mod2 = sarimax.SARIMAX(endog, order=(3,0,0))
    mod2.ssm.initialize_known(res1.filter_results.initial_state,
                              res1.filter_results.initial_state_cov)
    mod2.initialize_state()  # a noop in this case (include for coverage)
    res2 = mod2.filter([0.5,0.2,0.1,1])

    # Create a third model with "known" initialization, but specified in kwargs
    mod3 = sarimax.SARIMAX(endog, order=(3,0,0),
                           initialization='known',
                           initial_state=res1.filter_results.initial_state,
                           initial_state_cov=res1.filter_results.initial_state_cov)
    res3 = mod3.filter([0.5,0.2,0.1,1])

    # Create the forth model with stationary initialization specified in kwargs
    mod4 = sarimax.SARIMAX(endog, order=(3,0,0), initialization='stationary')
    res4 = mod4.filter([0.5,0.2,0.1,1])

    # Just test a couple of things to make sure the results are the same
    assert_almost_equal(res1.llf, res2.llf)
    assert_almost_equal(res1.filter_results.filtered_state,
                        res2.filter_results.filtered_state)

    assert_almost_equal(res1.llf, res3.llf)
    assert_almost_equal(res1.filter_results.filtered_state,
                        res3.filter_results.filtered_state)

    assert_almost_equal(res1.llf, res4.llf)
    assert_almost_equal(res1.filter_results.filtered_state,
                        res4.filter_results.filtered_state)

def test_manual_approximate_diffuse_initialization():
    endog = results_sarimax.wpi1_data

    # Create the first model to compare against
    mod1 = sarimax.SARIMAX(endog, order=(3,0,0))
    mod1.ssm.initialize_approximate_diffuse(1e9)
    res1 = mod1.filter([0.5,0.2,0.1,1])

    # Create a second model with "known" initialization
    mod2 = sarimax.SARIMAX(endog, order=(3,0,0))
    mod2.ssm.initialize_known(res1.filter_results.initial_state,
                              res1.filter_results.initial_state_cov)
    mod2.initialize_state()  # a noop in this case (include for coverage)
    res2 = mod2.filter([0.5,0.2,0.1,1])

    # Create a third model with "known" initialization, but specified in kwargs
    mod3 = sarimax.SARIMAX(endog, order=(3,0,0),
                           initialization='known',
                           initial_state=res1.filter_results.initial_state,
                           initial_state_cov=res1.filter_results.initial_state_cov)
    res3 = mod3.filter([0.5,0.2,0.1,1])

    # Create the forth model with approximate diffuse initialization specified
    # in kwargs
    mod4 = sarimax.SARIMAX(endog, order=(3,0,0),
                           initialization='approximate_diffuse',
                           initial_variance=1e9)
    res4 = mod4.filter([0.5,0.2,0.1,1])

    # Just test a couple of things to make sure the results are the same
    assert_almost_equal(res1.llf, res2.llf)
    assert_almost_equal(res1.filter_results.filtered_state,
                        res2.filter_results.filtered_state)

    assert_almost_equal(res1.llf, res3.llf)
    assert_almost_equal(res1.filter_results.filtered_state,
                        res3.filter_results.filtered_state)

    assert_almost_equal(res1.llf, res4.llf)
    assert_almost_equal(res1.filter_results.filtered_state,
                        res4.filter_results.filtered_state)

def test_results():
    endog = results_sarimax.wpi1_data

    mod = sarimax.SARIMAX(endog, order=(1,0,1))
    res = mod.filter([0.5,-0.5,1], cov_type='oim')

    assert_almost_equal(res.arroots, 2.)
    assert_almost_equal(res.maroots, 2.)

    assert_almost_equal(res.arfreq, np.arctan2(0, 2) / (2*np.pi))
    assert_almost_equal(res.mafreq, np.arctan2(0, 2) / (2*np.pi))

    assert_almost_equal(res.arparams, [0.5])
    assert_almost_equal(res.maparams, [-0.5])


def test_misc_exog():
    # Tests for missing data
    nobs = 20
    k_endog = 1
    np.random.seed(1208)
    endog = np.random.normal(size=(nobs, k_endog))
    endog[:4, 0] = np.nan
    exog1 = np.random.normal(size=(nobs, 1))
    exog2 = np.random.normal(size=(nobs, 2))

    index = pd.date_range('1970-01-01', freq='QS', periods=nobs)
    endog_pd = pd.DataFrame(endog, index=index)
    exog1_pd = pd.Series(exog1.squeeze(), index=index)
    exog2_pd = pd.DataFrame(exog2, index=index)

    models = [
        sarimax.SARIMAX(endog, exog=exog1, order=(1, 1, 0)),
        sarimax.SARIMAX(endog, exog=exog2, order=(1, 1, 0)),
        sarimax.SARIMAX(endog, exog=exog2, order=(1, 1, 0),
                        simple_differencing=False),
        sarimax.SARIMAX(endog_pd, exog=exog1_pd, order=(1, 1, 0)),
        sarimax.SARIMAX(endog_pd, exog=exog2_pd, order=(1, 1, 0)),
        sarimax.SARIMAX(endog_pd, exog=exog2_pd, order=(1, 1, 0),
                        simple_differencing=False),
    ]

    for mod in models:
        # Smoke tests
        mod.start_params
        res = mod.fit(disp=False)
        res.summary()
        res.predict()
        res.predict(dynamic=True)
        res.get_prediction()

        oos_exog = np.random.normal(size=(1, mod.k_exog))
        res.forecast(steps=1, exog=oos_exog)
        res.get_forecast(steps=1, exog=oos_exog)

        # Smoke tests for invalid exog
        oos_exog = np.random.normal(size=(1))
        assert_raises(ValueError, res.forecast, steps=1, exog=oos_exog)

        oos_exog = np.random.normal(size=(2, mod.k_exog))
        assert_raises(ValueError, res.forecast, steps=1, exog=oos_exog)

        oos_exog = np.random.normal(size=(1, mod.k_exog + 1))
        assert_raises(ValueError, res.forecast, steps=1, exog=oos_exog)

    # Test invalid model specifications
    assert_raises(ValueError, sarimax.SARIMAX, endog, exog=np.zeros((10, 4)),
                  order=(1, 1, 0))


def test_datasets():
    # Test that some unusual types of datasets work

    np.random.seed(232849)
    endog = np.random.binomial(1, 0.5, size=100)
    exog = np.random.binomial(1, 0.5, size=100)
    mod = sarimax.SARIMAX(endog, exog=exog, order=(1, 0, 0))
    res = mod.fit(disp=-1)


def test_predict_custom_index():
    np.random.seed(328423)
    endog = pd.DataFrame(np.random.normal(size=50))
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0))
    res = mod.smooth(mod.start_params)
    out = res.predict(start=1, end=1, index=['a'])
    assert_equal(out.index.equals(pd.Index(['a'])), True)


def test_arima000():
    from statsmodels.tsa.statespace.tools import compatibility_mode

    # Test an ARIMA(0,0,0) with measurement error model (i.e. just estimating
    # a variance term)
    np.random.seed(328423)
    nobs = 50
    endog = pd.DataFrame(np.random.normal(size=nobs))
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), measurement_error=False)
    res = mod.smooth(mod.start_params)
    assert_allclose(res.smoothed_state, endog.T)

    # ARIMA(0, 1, 0)
    mod = sarimax.SARIMAX(endog, order=(0, 1, 0), measurement_error=False)
    res = mod.smooth(mod.start_params)
    assert_allclose(res.smoothed_state[1:, 1:], endog.diff()[1:].T)

    # SARIMA(0, 1, 0)x(0, 1, 0, 1)
    mod = sarimax.SARIMAX(endog, order=(0, 1, 0), measurement_error=True,
                          seasonal_order=(0, 1, 0, 1))
    res = mod.smooth(mod.start_params)

    # Exogenous variables
    error = np.random.normal(size=nobs)
    endog = np.ones(nobs) * 10 + error
    exog = np.ones(nobs)

    # We need univariate filtering here, to guarantee we won't hit singular
    # forecast error covariance matrices.
    if compatibility_mode:
        return

    # OLS
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), exog=exog)
    mod.ssm.filter_univariate = True
    res = mod.smooth([10., 1.])
    assert_allclose(res.smoothed_state[0], error, atol=1e-10)

    # RLS
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), exog=exog,
                          mle_regression=False)
    mod.ssm.filter_univariate = True
    mod.initialize_known([0., 10.], np.diag([1., 0.]))
    res = mod.smooth([1.])
    assert_allclose(res.smoothed_state[0], error, atol=1e-10)
    assert_allclose(res.smoothed_state[1], 10, atol=1e-10)

    # RLS + TVP
    mod = sarimax.SARIMAX(endog, order=(0, 0, 0), exog=exog,
                          mle_regression=False, time_varying_regression=True)
    mod.ssm.filter_univariate = True
    mod.initialize_known([10.], np.diag([0.]))
    res = mod.smooth([0., 1.])
    assert_allclose(res.smoothed_state[0], 10, atol=1e-10)
