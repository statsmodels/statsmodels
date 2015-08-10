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
from statsmodels.datasets import webuse
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from numpy.testing import assert_equal, assert_almost_equal, assert_raises, assert_allclose
from nose.exc import SkipTest
from statsmodels.iolib.summary import forg

current_path = os.path.dirname(os.path.abspath(__file__))

output_path = 'results' + os.sep + 'results_dynamic_factor_stata.csv'
output_results = pd.read_csv(current_path + os.sep + output_path)


class CheckStaticFactor(object):
    def __init__(self, true, k_factors, factor_order, cov_type='oim',
                 included_vars=['dln_inv', 'dln_inc', 'dln_consump'],
                 **kwargs):
        self.true = true
        # 1960:Q1 - 1982:Q4
        dta = pd.DataFrame(
            results_varmax.lutkepohl_data, columns=['inv', 'inc', 'consump'],
            index=pd.date_range('1960-01-01', '1982-10-01', freq='QS'))

        dta['dln_inv'] = np.log(dta['inv']).diff()
        dta['dln_inc'] = np.log(dta['inc']).diff()
        dta['dln_consump'] = np.log(dta['consump']).diff()

        endog = dta.ix['1960-04-01':'1978-10-01', included_vars]

        self.model = dynamic_factor.StaticFactors(endog, k_factors=k_factors,
                                                  factor_order=factor_order,
                                                  **kwargs)

        self.results = self.model.filter(true['params'], cov_type=cov_type)

    def test_params(self):
        # Smoke test to make sure the start_params are well-defined and
        # lead to a well-defined model
        self.model.filter(self.model.start_params)
        # Similarly a smoke test for param_names
        assert_equal(len(self.model.start_params), len(self.model.param_names))
        # Finally make sure the transform and untransform do their job
        actual = self.model.transform_params(self.model.untransform_params(self.model.start_params))
        assert_allclose(actual, self.model.start_params)
        # Also in the case of enforce stationarity = False
        self.model.enforce_stationarity = False
        actual = self.model.transform_params(self.model.untransform_params(self.model.start_params))
        self.model.enforce_stationarity = True
        assert_allclose(actual, self.model.start_params)

    def test_results(self):
        # Smoke test for creating the summary
        self.results.summary()

        # Test cofficient matrix creation (via a different, more direct, method)
        if self.model.factor_order > 0:
            coefficients = np.array(self.results.params[self.model._params_transition]).reshape(self.model.k_factors, self.model.k_factors * self.model.factor_order)
            coefficient_matrices = np.array([
                coefficients[:self.model.k_factors, i*self.model.k_factors:(i+1)*self.model.k_factors]
                for i in range(self.model.factor_order)
            ])
            assert_equal(self.results.coefficient_matrices_var, coefficient_matrices)
        else:
            assert_equal(self.results.coefficient_matrices_var, None)

    def test_no_enforce(self):
        # Test that nothing goes wrong when we don't enforce stationarity
        params = self.model.untransform_params(self.true['params'])
        params[self.model._params_transition] = (
            self.true['params'][self.model._params_transition])
        self.model.enforce_stationarity = False
        results = self.model.filter(params, transformed=False)
        self.model.enforce_stationarity = True
        assert_allclose(results.llf, self.results.llf, rtol=1e-5)

    def test_mle(self):
        results = self.model.fit(maxiter=100, disp=False)
        results = self.model.fit(results.params, method='nm', maxiter=1000,
                                 disp=False)
        assert_allclose(results.llf, self.results.llf, rtol=1e-5)

    def test_loglike(self):
        assert_allclose(self.results.llf, self.true['loglike'], rtol=1e-6)

    def test_bse_oim(self):
        raise SkipTest('Known failure: standard errors do not match.')
        # assert_allclose(self.results.bse, self.true['bse_oim'], atol=1e-2)

    def test_aic(self):
        # We only get 3 digits from Stata
        assert_allclose(self.results.aic, self.true['aic'], atol=3)

    def test_bic(self):
        # We only get 3 digits from Stata
        assert_allclose(self.results.bic, self.true['bic'], atol=3)

    def test_predict(self):
        # Tests predict + forecast
        assert_allclose(
            self.results.predict(end='1982-10-01'),
            self.true['predict'].T,
            atol=1e-6)

    def test_dynamic_predict(self):
        # Tests predict + dynamic predict + forecast
        assert_allclose(
            self.results.predict(end='1982-10-01', dynamic='1961-01-01'),
            self.true['dynamic_predict'].T,
            atol=1e-6)

class TestStaticFactor(CheckStaticFactor):
    def __init__(self):
        true = results_dynamic_factor.lutkepohl_dfm.copy()
        true['predict'] = output_results.ix[1:, ['predict_dfm_1', 'predict_dfm_2', 'predict_dfm_3']]
        true['dynamic_predict'] = output_results.ix[1:, ['dyn_predict_dfm_1', 'dyn_predict_dfm_2', 'dyn_predict_dfm_3']]
        super(TestStaticFactor, self).__init__(true, k_factors=1, factor_order=2)

class TestStaticFactor2(CheckStaticFactor):
    def __init__(self):
        true = results_dynamic_factor.lutkepohl_dfm2.copy()
        true['predict'] = output_results.ix[1:, ['predict_dfm2_1', 'predict_dfm2_2', 'predict_dfm2_3']]
        true['dynamic_predict'] = output_results.ix[1:, ['dyn_predict_dfm2_1', 'dyn_predict_dfm2_2', 'dyn_predict_dfm2_3']]
        super(TestStaticFactor2, self).__init__(true, k_factors=2, factor_order=1)

    def test_mle(self):
        # Stata's MLE on this model doesn't converge, so no reason to check
        pass

    def test_bse(self):
        # Stata's MLE on this model doesn't converge, and four of their
        # params don't even have bse (possibly they are still at starting
        # values?), so no reason to check this
        pass

    def test_aic(self):
        # Stata uses 9 df (i.e. 9 params) here instead of 13, because since the
        # model didn't coverge, 4 of the parameters aren't fully estimated
        # (possibly they are still at starting values?) so the AIC is off
        pass

    def test_bic(self):
        # Stata uses 9 df (i.e. 9 params) here instead of 13, because since the
        # model didn't coverge, 4 of the parameters aren't fully estimated
        # (possibly they are still at starting values?) so the BIC is off
        pass

    def test_summary(self):
        summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Check the model overview table
        assert_equal(re.search(r'Model:.*StaticFactors\(factors=2, order=1\)', tables[0]) is None, False)

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset_loading = self.model.k_factors * i
            offset_var = self.model.k_factors * self.model.k_endog
            table = tables[i + 1]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for equation %s' % name, table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 8)

            # -> Check that we have the right coefficients
            assert_equal(re.search('loading.f1 +' + forg(params[offset_loading + 0], prec=4), table) is None, False)
            assert_equal(re.search('loading.f2 +' + forg(params[offset_loading + 1], prec=4), table) is None, False)
            assert_equal(re.search('sigma2 +' + forg(params[offset_var + i], prec=4), table) is None, False)

        # For each factor, check the output
        for i in range(self.model.k_factors):
            offset = self.model.k_endog * (self.model.k_factors + 1) + i * self.model.k_factors
            table = tables[self.model.k_endog + i + 1]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for factor equation f%d' % (i+1), table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 7)

            # -> Check that we have the right coefficients
            assert_equal(re.search('L1.f1 +' + forg(params[offset + 0], prec=4), table) is None, False)
            assert_equal(re.search('L1.f2 +' + forg(params[offset + 1], prec=4), table) is None, False)


def test_misspecification():
    # Tests for model specification and misspecification exceptions
    endog = np.arange(20).reshape(10,2)

    # Too many factors
    assert_raises(ValueError, dynamic_factor.StaticFactors, endog, k_factors=2, factor_order=1)
