"""
Tests for VARMAX models

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function

import numpy as np
import pandas as pd
import os
import re

import warnings
from statsmodels.tsa.statespace import mlemodel, varmax
from .results import results_varmax
from numpy.testing import assert_equal, assert_almost_equal, assert_raises, assert_allclose
from nose.exc import SkipTest
from statsmodels.iolib.summary import forg

current_path = os.path.dirname(os.path.abspath(__file__))

var_path = 'results' + os.sep + 'results_var_stata.csv'
var_results = pd.read_csv(current_path + os.sep + var_path)

varmax_path = 'results' + os.sep + 'results_varmax_stata.csv'
varmax_results = pd.read_csv(current_path + os.sep + varmax_path)


class CheckVARMAX(object):
    """
    Test Vector Autoregression against Stata's `dfactor` code (Stata's
    `var` function uses OLS and not state space / MLE, so we can't get
    equivalent log-likelihoods)
    """

    def test_mle(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            # Fit with all transformations
            # results = self.model.fit(method='powell', disp=-1)
            results = self.model.fit(maxiter=100, disp=False)
            # Fit now without transformations
            self.model.enforce_stationarity = False
            self.model.enforce_invertibility = False
            results = self.model.fit(results.params, method='nm', maxiter=1000,
                                     disp=False)
            self.model.enforce_stationarity = True
            self.model.enforce_invertibility = True
            assert_allclose(results.llf, self.results.llf, rtol=1e-5)

    def test_params(self):
        # Smoke test to make sure the start_params are well-defined and
        # lead to a well-defined model
        self.model.filter(self.model.start_params)
        # Similarly a smoke test for param_names
        assert_equal(len(self.model.start_params), len(self.model.param_names))
        # Finally make sure the transform and untransform do their job
        actual = self.model.transform_params(self.model.untransform_params(self.model.start_params))
        assert_allclose(actual, self.model.start_params)
        # Also in the case of enforce invertibility and stationarity = False
        self.model.enforce_stationarity = False
        self.model.enforce_invertibility = False
        actual = self.model.transform_params(self.model.untransform_params(self.model.start_params))
        self.model.enforce_stationarity = True
        self.model.enforce_invertibility = True
        assert_allclose(actual, self.model.start_params)

    def test_results(self):
        # Smoke test for creating the summary
        self.results.summary()

        # Test cofficient matrix creation (via a different, more direct, method)
        if self.model.k_ar > 0:
            coefficients = np.array(self.results.params[self.model._params_ar]).reshape(self.model.k_endog, self.model.k_endog * self.model.k_ar)
            coefficient_matrices = np.array([
                coefficients[:self.model.k_endog, i*self.model.k_endog:(i+1)*self.model.k_endog]
                for i in range(self.model.k_ar)
            ])
            assert_equal(self.results.coefficient_matrices_var, coefficient_matrices)
        else:
            assert_equal(self.results.coefficient_matrices_var, None)
        if self.model.k_ma > 0:
            coefficients = np.array(self.results.params[self.model._params_ma]).reshape(self.model.k_endog, self.model.k_endog * self.model.k_ma)
            coefficient_matrices = np.array([
                coefficients[:self.model.k_endog, i*self.model.k_endog:(i+1)*self.model.k_endog]
                for i in range(self.model.k_ma)
            ])
            assert_equal(self.results.coefficient_matrices_vma, coefficient_matrices)
        else:
            assert_equal(self.results.coefficient_matrices_vma, None)

    def test_loglike(self):
        assert_allclose(self.results.llf, self.true['loglike'], rtol=1e-6)

    def test_aic(self):
        # We only get 3 digits from Stata
        assert_allclose(self.results.aic, self.true['aic'], atol=3)

    def test_bic(self):
        # We only get 3 digits from Stata
        assert_allclose(self.results.bic, self.true['bic'], atol=3)

    def test_predict(self, end, atol=1e-6, **kwargs):
        # Tests predict + forecast
        assert_allclose(
            self.results.predict(end=end, **kwargs),
            self.true['predict'],
            atol=atol)

    def test_dynamic_predict(self, end, dynamic, atol=1e-6, **kwargs):
        # Tests predict + dynamic predict + forecast
        assert_allclose(
            self.results.predict(end=end, dynamic=dynamic, **kwargs),
            self.true['dynamic_predict'],
            atol=atol)


class CheckLutkepohl(CheckVARMAX):
    @classmethod
    def setup_class(cls, true, order, trend, error_cov_type, cov_type='approx',
             included_vars=['dln_inv', 'dln_inc', 'dln_consump'],
             **kwargs):
        cls.true = true
        # 1960:Q1 - 1982:Q4
        dta = pd.DataFrame(
            results_varmax.lutkepohl_data, columns=['inv', 'inc', 'consump'],
            index=pd.date_range('1960-01-01', '1982-10-01', freq='QS'))

        dta['dln_inv'] = np.log(dta['inv']).diff()
        dta['dln_inc'] = np.log(dta['inc']).diff()
        dta['dln_consump'] = np.log(dta['consump']).diff()

        endog = dta.ix['1960-04-01':'1978-10-01', included_vars]

        cls.model = varmax.VARMAX(endog, order=order, trend=trend,
                                   error_cov_type=error_cov_type, **kwargs)

        cls.results = cls.model.smooth(true['params'], cov_type=cov_type)

    def test_predict(self, **kwargs):
        super(CheckLutkepohl, self).test_predict(end='1982-10-01', **kwargs)

    def test_dynamic_predict(self, **kwargs):
        super(CheckLutkepohl, self).test_dynamic_predict(end='1982-10-01', dynamic='1961-01-01', **kwargs)


class TestVAR(CheckLutkepohl):
    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1.copy()
        true['predict'] = var_results.ix[1:, ['predict_1', 'predict_2', 'predict_3']]
        true['dynamic_predict'] = var_results.ix[1:, ['dyn_predict_1', 'dyn_predict_2', 'dyn_predict_3']]
        super(TestVAR, cls).setup_class(
            true,  order=(1,0), trend='nc',
            error_cov_type="unstructured")

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()**0.5
        assert_allclose(bse**2, self.true['var_oim'], atol=1e-4)

    def test_bse_oim(self):
        bse = self.results._cov_params_oim().diagonal()**0.5
        assert_allclose(bse**2, self.true['var_oim'], atol=1e-2)

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Check the model overview table
        assert_equal(re.search(r'Model:.*VAR\(1\)', tables[0]) is None, False)

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset = i * self.model.k_endog
            table = tables[i+2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for equation %s' % name, table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 8)

            # -> Check that we have the right coefficients
            assert_equal(re.search('L1.dln_inv +%.4f' % params[offset + 0], table) is None, False)
            assert_equal(re.search('L1.dln_inc +%.4f' % params[offset + 1], table) is None, False)
            assert_equal(re.search('L1.dln_consump +%.4f' % params[offset + 2], table) is None, False)

        # Test the error covariance matrix table
        table = tables[-1]
        assert_equal(re.search('Error covariance matrix', table) is None, False)
        assert_equal(len(table.split('\n')), 11)

        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert_equal(re.search('%s +%.4f' % (names[i], params[i]), table) is None, False)

class TestVAR_diagonal(CheckLutkepohl):
    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_diag.copy()
        true['predict'] = var_results.ix[1:, ['predict_diag1', 'predict_diag2', 'predict_diag3']]
        true['dynamic_predict'] = var_results.ix[1:, ['dyn_predict_diag1', 'dyn_predict_diag2', 'dyn_predict_diag3']]
        super(TestVAR_diagonal, cls).setup_class(
            true,  order=(1,0), trend='nc',
            error_cov_type="diagonal")

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()**0.5
        assert_allclose(bse**2, self.true['var_oim'], atol=1e-5)

    def test_bse_oim(self):
        bse = self.results._cov_params_oim().diagonal()**0.5
        assert_allclose(bse**2, self.true['var_oim'], atol=1e-2)

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Check the model overview table
        assert_equal(re.search(r'Model:.*VAR\(1\)', tables[0]) is None, False)

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset = i * self.model.k_endog
            table = tables[i+2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for equation %s' % name, table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 8)

            # -> Check that we have the right coefficients
            assert_equal(re.search('L1.dln_inv +%.4f' % params[offset + 0], table) is None, False)
            assert_equal(re.search('L1.dln_inc +%.4f' % params[offset + 1], table) is None, False)
            assert_equal(re.search('L1.dln_consump +%.4f' % params[offset + 2], table) is None, False)

        # Test the error covariance matrix table
        table = tables[-1]
        assert_equal(re.search('Error covariance matrix', table) is None, False)
        assert_equal(len(table.split('\n')), 8)

        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert_equal(re.search('%s +%.4f' % (names[i], params[i]), table) is None, False)


class TestVAR_measurement_error(CheckLutkepohl):
    """
    Notes
    -----
    There does not appear to be a way to get Stata to estimate a VAR with
    measurement errors. Thus this test is mostly a smoke test that measurement
    errors are setup correctly: it uses the same params from TestVAR_diagonal
    and sets the measurement errors variance params to zero to check that the
    loglike and predict are the same.

    It also checks that the state-space representation with positive
    measurement errors is correct.
    """
    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_diag_meas.copy()
        true['predict'] = var_results.ix[1:, ['predict_diag1', 'predict_diag2', 'predict_diag3']]
        true['dynamic_predict'] = var_results.ix[1:, ['dyn_predict_diag1', 'dyn_predict_diag2', 'dyn_predict_diag3']]
        super(TestVAR_measurement_error, cls).setup_class(
            true,  order=(1,0), trend='nc',
            error_cov_type="diagonal", measurement_error=True)

        # Create another filter results with positive measurement errors
        cls.true_measurement_error_variances = [1., 2., 3.]
        params = np.r_[true['params'][:-3], cls.true_measurement_error_variances]
        cls.results2 = cls.model.smooth(params)

    def test_mle(self):
        # With the additional measurment error parameters, this wouldn't be
        # a meaningful test
        pass

    def test_bse_approx(self):
        # This would just test the same thing as TestVAR_diagonal.test_bse_approx
        pass

    def test_bse_oim(self):
        # This would just test the same thing as TestVAR_diagonal.test_bse_oim
        pass

    def test_aic(self):
        # Since the measurement error is added, the number
        # of parameters, and hence the aic and bic, will be off
        pass

    def test_bic(self):
        # Since the measurement error is added, the number
        # of parameters, and hence the aic and bic, will be off
        pass

    def test_representation(self):
        # Test that the state space representation in the measurement error
        # case is correct
        for name in self.model.ssm.shapes.keys():
            if name == 'obs':
                pass
            elif name == 'obs_cov':
                actual = self.results2.filter_results.obs_cov
                desired = np.diag(self.true_measurement_error_variances)[:,:,np.newaxis]
                assert_equal(actual, desired)
            else:
                assert_equal(getattr(self.results2.filter_results, name),
                             getattr(self.results.filter_results, name))

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Check the model overview table
        assert_equal(re.search(r'Model:.*VAR\(1\)', tables[0]) is None, False)

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset = i * self.model.k_endog
            table = tables[i+2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for equation %s' % name, table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 9)

            # -> Check that we have the right coefficients
            assert_equal(re.search('L1.dln_inv +%.4f' % params[offset + 0], table) is None, False)
            assert_equal(re.search('L1.dln_inc +%.4f' % params[offset + 1], table) is None, False)
            assert_equal(re.search('L1.dln_consump +%.4f' % params[offset + 2], table) is None, False)
            assert_equal(re.search('measurement_variance +%.4g' % params[-(i+1)], table) is None, False)

        # Test the error covariance matrix table
        table = tables[-1]
        assert_equal(re.search('Error covariance matrix', table) is None, False)
        assert_equal(len(table.split('\n')), 8)

        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert_equal(re.search('%s +%.4f' % (names[i], params[i]), table) is None, False)

class TestVAR_obs_intercept(CheckLutkepohl):
    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_obs_intercept.copy()
        true['predict'] = var_results.ix[1:, ['predict_int1', 'predict_int2', 'predict_int3']]
        true['dynamic_predict'] = var_results.ix[1:, ['dyn_predict_int1', 'dyn_predict_int2', 'dyn_predict_int3']]
        super(TestVAR_obs_intercept, cls).setup_class(
            true, order=(1,0), trend='nc',
            error_cov_type="diagonal", obs_intercept=true['obs_intercept'])

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()**0.5
        assert_allclose(bse**2, self.true['var_oim'], atol=1e-4)

    def test_bse_oim(self):
        bse = self.results._cov_params_oim().diagonal()**0.5
        assert_allclose(bse**2, self.true['var_oim'], atol=1e-2)

    def test_aic(self):
        # Since the obs_intercept is added in in an ad-hoc way here, the number
        # of parameters, and hence the aic and bic, will be off
        pass

    def test_bic(self):
        # Since the obs_intercept is added in in an ad-hoc way here, the number
        # of parameters, and hence the aic and bic, will be off
        pass


class TestVAR_exog(CheckLutkepohl):
    # Note: unlike the other tests in this file, this is against the Stata
    # var function rather than the Stata dfactor function
    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_exog.copy()
        true['predict'] = var_results.ix[1:75, ['predict_exog1_1', 'predict_exog1_2', 'predict_exog1_3']]
        true['predict'].iloc[0, :] = 0
        true['fcast'] = var_results.ix[76:, ['fcast_exog1_dln_inv', 'fcast_exog1_dln_inc', 'fcast_exog1_dln_consump']]
        exog = np.arange(75) + 3
        super(TestVAR_exog, cls).setup_class(
            true, order=(1,0), trend='nc', error_cov_type='unstructured',
            exog=exog, initialization='approximate_diffuse', loglikelihood_burn=1)

    def test_mle(self):
        pass

    def test_aic(self):
        # Stata's var calculates AIC differently
        pass

    def test_bic(self):
        # Stata's var calculates BIC differently
        pass

    def test_bse_approx(self):
        # Exclude the covariance cholesky terms
        bse = self.results._cov_params_approx().diagonal()**0.5
        assert_allclose(bse[:-6]**2, self.true['var_oim'], atol=1e-5)

    def test_bse_oim(self):
        # Exclude the covariance cholesky terms
        bse = self.results._cov_params_oim().diagonal()**0.5
        assert_allclose(bse[:-6]**2, self.true['var_oim'], atol=1e-5)

    def test_predict(self):
        super(CheckLutkepohl, self).test_predict(end='1978-10-01', atol=1e-3)

    def test_dynamic_predict(self):
        # Stata's var cannot subsequently use dynamic
        pass

    def test_forecast(self):
        # Tests forecast
        exog = (np.arange(75, 75+16) + 3)[:, np.newaxis]

        # Test it through the results class wrapper
        desired = self.results.forecast(steps=16, exog=exog)
        assert_allclose(desired, self.true['fcast'], atol=1e-6)

        # Test it directly (i.e. without the wrapping done in
        # VARMAXResults.get_prediction which converts exog to state_intercept)
        beta = self.results.params[-9:-6]
        state_intercept = np.concatenate([
            exog*beta[0], exog*beta[1], exog*beta[2]], axis=1).T
        desired = mlemodel.MLEResults.get_prediction(
            self.results._results, start=75, end=75+15,
            state_intercept=state_intercept).predicted_mean
        assert_allclose(desired, self.true['fcast'], atol=1e-6)

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Check the model overview table
        assert_equal(re.search(r'Model:.*VARX\(1\)', tables[0]) is None, False)

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset = i * self.model.k_endog
            table = tables[i+2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for equation %s' % name, table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 9)

            # -> Check that we have the right coefficients
            assert_equal(re.search('L1.dln_inv +%.4f' % params[offset + 0], table) is None, False)
            assert_equal(re.search('L1.dln_inc +%.4f' % params[offset + 1], table) is None, False)
            assert_equal(re.search('L1.dln_consump +%.4f' % params[offset + 2], table) is None, False)
            assert_equal(re.search('beta.x1 +' + forg(params[self.model._params_regression][i], prec=4), table) is None, False)

        # Test the error covariance matrix table
        table = tables[-1]
        assert_equal(re.search('Error covariance matrix', table) is None, False)
        assert_equal(len(table.split('\n')), 11)

        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert_equal(re.search('%s +%.4f' % (names[i], params[i]), table) is None, False)

class TestVAR_exog2(CheckLutkepohl):
    # This is a regression test, to make sure that the setup with multiple exog
    # works correctly. The params are from Stata, but the loglike is from
    # this model. Likely the small discrepancy (see the results file) is from
    # the approximate diffuse initialization.
    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var1_exog2.copy()
        true['predict'] = var_results.ix[1:75, ['predict_exog2_1', 'predict_exog2_2', 'predict_exog2_3']]
        true['predict'].iloc[0, :] = 0
        true['fcast'] = var_results.ix[76:, ['fcast_exog2_dln_inv', 'fcast_exog2_dln_inc', 'fcast_exog2_dln_consump']]
        exog = np.c_[np.ones((75,1)), (np.arange(75) + 3)[:, np.newaxis]]
        super(TestVAR_exog2, cls).setup_class(
            true, order=(1,0), trend='nc', error_cov_type='unstructured',
            exog=exog, initialization='approximate_diffuse', loglikelihood_burn=1)

    def test_mle(self):
        pass

    def test_aic(self):
        pass

    def test_bic(self):
        pass

    def test_bse_approx(self):
        pass

    def test_bse_oim(self):
        pass

    def test_predict(self):
        super(CheckLutkepohl, self).test_predict(end='1978-10-01', atol=1e-3)

    def test_dynamic_predict(self):
        # Stata's var cannot subsequently use dynamic
        pass

    def test_forecast(self):
        # Tests forecast
        exog = np.c_[np.ones((16, 1)), (np.arange(75, 75+16) + 3)[:, np.newaxis]]

        desired = self.results.forecast(steps=16, exog=exog)
        assert_allclose(desired, self.true['fcast'], atol=1e-6)


class TestVAR2(CheckLutkepohl):
    @classmethod
    def setup_class(cls):
        true = results_varmax.lutkepohl_var2.copy()
        true['predict'] = var_results.ix[1:, ['predict_var2_1', 'predict_var2_2']]
        true['dynamic_predict'] = var_results.ix[1:, ['dyn_predict_var2_1', 'dyn_predict_var2_2']]
        super(TestVAR2, cls).setup_class(
            true, order=(2,0), trend='nc', error_cov_type='unstructured',
            included_vars=['dln_inv', 'dln_inc'])

    def test_bse_approx(self):
        # Exclude the covariance cholesky terms
        bse = self.results._cov_params_approx().diagonal()**0.5
        assert_allclose(bse[:-3]**2, self.true['var_oim'][:-3], atol=1e-5)

    def test_bse_oim(self):
        # Exclude the covariance cholesky terms
        bse = self.results._cov_params_oim().diagonal()**0.5
        assert_allclose(bse[:-3]**2, self.true['var_oim'][:-3], atol=1e-2)

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Check the model overview table
        assert_equal(re.search(r'Model:.*VAR\(2\)', tables[0]) is None, False)

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset = i * self.model.k_endog * self.model.k_ar
            table = tables[i+2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for equation %s' % name, table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 9)

            # -> Check that we have the right coefficients
            assert_equal(re.search('L1.dln_inv +%.4f' % params[offset + 0], table) is None, False)
            assert_equal(re.search('L1.dln_inc +%.4f' % params[offset + 1], table) is None, False)
            assert_equal(re.search('L2.dln_inv +%.4f' % params[offset + 2], table) is None, False)
            assert_equal(re.search('L2.dln_inc +%.4f' % params[offset + 3], table) is None, False)

        # Test the error covariance matrix table
        table = tables[-1]
        assert_equal(re.search('Error covariance matrix', table) is None, False)
        assert_equal(len(table.split('\n')), 8)

        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert_equal(re.search('%s +%.4f' % (names[i], params[i]), table) is None, False)


class CheckFREDManufacturing(CheckVARMAX):
    @classmethod
    def setup_class(cls, true, order, trend, error_cov_type, cov_type='approx',
                 **kwargs):
        cls.true = true
        # 1960:Q1 - 1982:Q4
        with open(current_path + os.sep + 'results' + os.sep + 'manufac.dta', 'rb') as test_data:
            dta = pd.read_stata(test_data)
        dta.index = dta.month
        dta['dlncaputil'] = dta['lncaputil'].diff()
        dta['dlnhours'] = dta['lnhours'].diff()

        endog = dta.ix['1972-02-01':, ['dlncaputil', 'dlnhours']]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cls.model = varmax.VARMAX(endog, order=order, trend=trend,
                                       error_cov_type=error_cov_type, **kwargs)

        cls.results = cls.model.smooth(true['params'], cov_type=cov_type)


class TestVARMA(CheckFREDManufacturing):
    """
    Test against the sspace VARMA example with some params set to zeros.
    """

    @classmethod
    def setup_class(cls):
        true = results_varmax.fred_varma11.copy()
        true['predict'] = varmax_results.ix[1:, ['predict_varma11_1', 'predict_varma11_2']]
        true['dynamic_predict'] = varmax_results.ix[1:, ['dyn_predict_varma11_1', 'dyn_predict_varma11_2']]
        super(TestVARMA, cls).setup_class(
              true, order=(1,1), trend='nc', error_cov_type='diagonal')

    def test_mle(self):
        # Since the VARMA model here is generic (we're just forcing zeros
        # in some params) whereas Stata's is restricted, the MLE test isn't
        # meaninful
        pass

    def test_bse_approx(self):
        # Standard errors do not match Stata's
        raise SkipTest('Known failure: standard errors do not match.')

    def test_bse_oim(self):
        # Standard errors do not match Stata's
        raise SkipTest('Known failure: standard errors do not match.')

    def test_aic(self):
        # Since the VARMA model here is generic (we're just putting in zeros
        # for some params), Stata assumes a different estimated number of
        # parameters; hence the aic and bic, will be off
        pass

    def test_bic(self):
        # Since the VARMA model here is generic (we're just putting in zeros
        # for some params), Stata assumes a different estimated number of
        # parameters; hence the aic and bic, will be off
        pass

    def test_predict(self):
        super(TestVARMA, self).test_predict(end='2009-05-01', atol=1e-4)

    def test_dynamic_predict(self):
        super(TestVARMA, self).test_dynamic_predict(end='2009-05-01', dynamic='2000-01-01')

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Check the model overview table
        assert_equal(re.search(r'Model:.*VARMA\(1,1\)', tables[0]) is None, False)

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset_ar = i * self.model.k_endog
            offset_ma = self.model.k_endog**2 * self.model.k_ar + i * self.model.k_endog
            table = tables[i+2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for equation %s' % name, table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 9)

            # -> Check that we have the right coefficients
            assert_equal(re.search('L1.dlncaputil +' + forg(params[offset_ar + 0], prec=4), table) is None, False)
            assert_equal(re.search('L1.dlnhours +' + forg(params[offset_ar + 1], prec=4), table) is None, False)
            assert_equal(re.search(r'L1.e\(dlncaputil\) +' + forg(params[offset_ma + 0], prec=4), table) is None, False)
            assert_equal(re.search(r'L1.e\(dlnhours\) +' + forg(params[offset_ma + 1], prec=4), table) is None, False)

        # Test the error covariance matrix table
        table = tables[-1]
        assert_equal(re.search('Error covariance matrix', table) is None, False)
        assert_equal(len(table.split('\n')), 7)

        params = params[self.model._params_state_cov]
        names = self.model.param_names[self.model._params_state_cov]
        for i in range(len(names)):
            assert_equal(re.search('%s +%s' % (names[i], forg(params[i], prec=4)), table) is None, False)


class TestVMA1(CheckFREDManufacturing):
    """
    Test against the sspace VARMA example with some params set to zeros.
    """

    @classmethod
    def setup_class(cls):
        true = results_varmax.fred_vma1.copy()
        true['predict'] = varmax_results.ix[1:, ['predict_vma1_1', 'predict_vma1_2']]
        true['dynamic_predict'] = varmax_results.ix[1:, ['dyn_predict_vma1_1', 'dyn_predict_vma1_2']]
        super(TestVMA1, cls).setup_class(
              true, order=(0,1), trend='nc', error_cov_type='diagonal')

    def test_mle(self):
        # Since the VARMA model here is generic (we're just forcing zeros
        # in some params) whereas Stata's is restricted, the MLE test isn't
        # meaninful
        pass

    def test_bse_approx(self):
        # Standard errors do not match Stata's
        raise SkipTest('Known failure: standard errors do not match.')

    def test_bse_oim(self):
        # Standard errors do not match Stata's
        raise SkipTest('Known failure: standard errors do not match.')

    def test_aic(self):
        # Since the VARMA model here is generic (we're just putting in zeros
        # for some params), Stata assumes a different estimated number of
        # parameters; hence the aic and bic, will be off
        pass

    def test_bic(self):
        # Since the VARMA model here is generic (we're just putting in zeros
        # for some params), Stata assumes a different estimated number of
        # parameters; hence the aic and bic, will be off
        pass

    def test_predict(self):
        super(TestVMA1, self).test_predict(end='2009-05-01', atol=1e-4)

    def test_dynamic_predict(self):
        super(TestVMA1, self).test_dynamic_predict(end='2009-05-01', dynamic='2000-01-01')


def test_specifications():
    # Tests for model specification and state space creation
    endog = np.arange(20).reshape(10,2)
    exog = np.arange(10)
    exog2 = pd.Series(exog, index=pd.date_range('2000-01-01', '2009-01-01', freq='AS'))

    # Test successful model creation
    mod = varmax.VARMAX(endog, exog=exog, order=(1,0))

    # Test successful model creation with pandas exog
    mod = varmax.VARMAX(endog, exog=exog2, order=(1,0))


def test_misspecifications():
    # Clear warnings
    varmax.__warningregistry__ = {}

    # Tests for model specification and misspecification exceptions
    endog = np.arange(20).reshape(10,2)

    # Bad trend specification
    assert_raises(ValueError, varmax.VARMAX, endog, order=(1,0), trend='')

    # Bad error_cov_type specification
    assert_raises(ValueError, varmax.VARMAX, endog, order=(1,0), error_cov_type='')

    # Bad order specification
    assert_raises(ValueError, varmax.VARMAX, endog, order=(0,0))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        varmax.VARMAX(endog, order=(1,1))

    # Warning with VARMA specification
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        varmax.VARMAX(endog, order=(1,1))

        message = ('Estimation of VARMA(p,q) models is not generically robust,'
                   ' due especially to identification issues.')
        assert_equal(str(w[0].message), message)
    warnings.resetwarnings()


def test_misc_exog():
    # Tests for missing data
    nobs = 20
    k_endog = 2
    np.random.seed(1208)
    endog = np.random.normal(size=(nobs, k_endog))
    endog[:4, 0] = np.nan
    endog[2:6, 1] = np.nan
    exog1 = np.random.normal(size=(nobs, 1))
    exog2 = np.random.normal(size=(nobs, 2))

    index = pd.date_range('1970-01-01', freq='QS', periods=nobs)
    endog_pd = pd.DataFrame(endog, index=index)
    exog1_pd = pd.Series(exog1.squeeze(), index=index)
    exog2_pd = pd.DataFrame(exog2, index=index)

    models = [
        varmax.VARMAX(endog, exog=exog1, order=(1, 0)),
        varmax.VARMAX(endog, exog=exog2, order=(1, 0)),
        varmax.VARMAX(endog_pd, exog=exog1_pd, order=(1, 0)),
        varmax.VARMAX(endog_pd, exog=exog2_pd, order=(1, 0)),
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
    assert_raises(ValueError, varmax.VARMAX, endog, exog=np.zeros((10, 4)),
                  order=(1, 0))
