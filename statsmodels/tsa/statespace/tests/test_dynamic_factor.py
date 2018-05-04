"""
Tests for VARMAX models

Author: Chad Fulton
License: Simplified-BSD
"""
from __future__ import division, absolute_import, print_function
from statsmodels.compat.testing import skip

import numpy as np
import pandas as pd
import os
import re

import warnings
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from numpy.testing import assert_equal, assert_almost_equal, assert_raises, assert_allclose
from statsmodels.iolib.summary import forg

try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except ImportError:
    have_matplotlib = False

current_path = os.path.dirname(os.path.abspath(__file__))

output_path = 'results' + os.sep + 'results_dynamic_factor_stata.csv'
output_results = pd.read_csv(current_path + os.sep + output_path)


class CheckDynamicFactor(object):
    @classmethod
    def setup_class(cls, true, k_factors, factor_order, cov_type='approx',
                 included_vars=['dln_inv', 'dln_inc', 'dln_consump'],
                 demean=False, filter=True, **kwargs):
        cls.true = true
        # 1960:Q1 - 1982:Q4
        dta = pd.DataFrame(
            results_varmax.lutkepohl_data, columns=['inv', 'inc', 'consump'],
            index=pd.date_range('1960-01-01', '1982-10-01', freq='QS'))

        dta['dln_inv'] = np.log(dta['inv']).diff()
        dta['dln_inc'] = np.log(dta['inc']).diff()
        dta['dln_consump'] = np.log(dta['consump']).diff()

        endog = dta.loc['1960-04-01':'1978-10-01', included_vars]

        if demean:
            endog -= dta.iloc[1:][included_vars].mean()

        cls.model = dynamic_factor.DynamicFactor(endog, k_factors=k_factors,
                                                  factor_order=factor_order,
                                                  **kwargs)

        if filter:
            cls.results = cls.model.smooth(true['params'], cov_type=cov_type)

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.results.summary()

        # Test cofficient matrix creation (via a different, more direct, method)
        if self.model.factor_order > 0:
            coefficients = np.array(self.results.params[self.model._params_factor_transition]).reshape(self.model.k_factors, self.model.k_factors * self.model.factor_order)
            coefficient_matrices = np.array([
                coefficients[:self.model.k_factors, i*self.model.k_factors:(i+1)*self.model.k_factors]
                for i in range(self.model.factor_order)
            ])
            assert_equal(self.results.coefficient_matrices_var, coefficient_matrices)
        else:
            assert_equal(self.results.coefficient_matrices_var, None)

        # Smoke test for plot_coefficients_of_determination
        if have_matplotlib:
            fig = self.results.plot_coefficients_of_determination();
            plt.close(fig)

    def test_no_enforce(self):
        return
        # Test that nothing goes wrong when we don't enforce stationarity
        params = self.model.untransform_params(self.true['params'])
        params[self.model._params_transition] = (
            self.true['params'][self.model._params_transition])
        self.model.enforce_stationarity = False
        results = self.model.filter(params, transformed=False)
        self.model.enforce_stationarity = True
        assert_allclose(results.llf, self.results.llf, rtol=1e-5)

    def test_mle(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            results = self.model.fit(method='powell', maxiter=100, disp=False)
            results = self.model.fit(results.params, maxiter=1000, disp=False)
            results = self.model.fit(results.params, method='nm', maxiter=1000,
                                     disp=False)
            if not results.llf > self.results.llf:
                assert_allclose(results.llf, self.results.llf, rtol=1e-5)

    def test_loglike(self):
        assert_allclose(self.results.llf, self.true['loglike'], rtol=1e-6)

    def test_aic(self):
        # We only get 3 digits from Stata
        assert_allclose(self.results.aic, self.true['aic'], atol=3)

    def test_bic(self):
        # We only get 3 digits from Stata
        assert_allclose(self.results.bic, self.true['bic'], atol=3)

    def test_predict(self, **kwargs):
        # Tests predict + forecast
        assert_allclose(
            self.results.predict(end='1982-10-01', **kwargs),
            self.true['predict'],
            atol=1e-6)

    def test_dynamic_predict(self, **kwargs):
        # Tests predict + dynamic predict + forecast
        assert_allclose(
            self.results.predict(end='1982-10-01', dynamic='1961-01-01', **kwargs),
            self.true['dynamic_predict'],
            atol=1e-6)

class TestDynamicFactor(CheckDynamicFactor):
    """
    Test for a dynamic factor model with 1 AR(2) factor
    """
    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_dfm.copy()
        true['predict'] = output_results.iloc[1:][['predict_dfm_1', 'predict_dfm_2', 'predict_dfm_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_dfm_1', 'dyn_predict_dfm_2', 'dyn_predict_dfm_3']]
        super(TestDynamicFactor, cls).setup_class(true, k_factors=1, factor_order=2)

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()**0.5
        assert_allclose(bse, self.true['bse_oim'], atol=1e-5)

class TestDynamicFactor2(CheckDynamicFactor):
    """
    Test for a dynamic factor model with two VAR(1) factors
    """
    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_dfm2.copy()
        true['predict'] = output_results.iloc[1:][['predict_dfm2_1', 'predict_dfm2_2', 'predict_dfm2_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_dfm2_1', 'dyn_predict_dfm2_2', 'dyn_predict_dfm2_3']]
        super(TestDynamicFactor2, cls).setup_class(true, k_factors=2, factor_order=1)

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Make sure we have the right number of tables
        assert_equal(len(tables), 2 + self.model.k_endog + self.model.k_factors + 1)

        # Check the model overview table
        assert_equal(re.search(r'Model:.*DynamicFactor\(factors=2, order=1\)', tables[0]) is None, False)

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset_loading = self.model.k_factors * i
            offset_var = self.model.k_factors * self.model.k_endog
            table = tables[i + 2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for equation %s' % name, table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 7)

            # -> Check that we have the right coefficients
            assert_equal(re.search('loading.f1 +' + forg(params[offset_loading + 0], prec=4), table) is None, False)
            assert_equal(re.search('loading.f2 +' + forg(params[offset_loading + 1], prec=4), table) is None, False)

        # For each factor, check the output
        for i in range(self.model.k_factors):
            offset = self.model.k_endog * (self.model.k_factors + 1) + i * self.model.k_factors
            table = tables[self.model.k_endog + i + 2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for factor equation f%d' % (i+1), table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 7)

            # -> Check that we have the right coefficients
            assert_equal(re.search('L1.f1 +' + forg(params[offset + 0], prec=4), table) is None, False)
            assert_equal(re.search('L1.f2 +' + forg(params[offset + 1], prec=4), table) is None, False)

        # Check the Error covariance matrix output
        table = tables[2 + self.model.k_endog + self.model.k_factors]

        # -> Make sure we have the right table / table name
        name = self.model.endog_names[i]
        assert_equal(re.search('Error covariance matrix', table) is None, False)

        # -> Make sure it's the right size
        assert_equal(len(table.split('\n')), 8)

        # -> Check that we have the right coefficients
        offset = self.model.k_endog * self.model.k_factors
        for i in range(self.model.k_endog):
            assert_equal(re.search('sigma2.%s +%s' % (self.model.endog_names[i], forg(params[offset + i], prec=4)), table) is None, False)



class TestDynamicFactor_exog1(CheckDynamicFactor):
    """
    Test for a dynamic factor model with 1 exogenous regressor: a constant
    """
    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_dfm_exog1.copy()
        true['predict'] = output_results.iloc[1:][['predict_dfm_exog1_1', 'predict_dfm_exog1_2', 'predict_dfm_exog1_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_dfm_exog1_1', 'dyn_predict_dfm_exog1_2', 'dyn_predict_dfm_exog1_3']]
        exog = np.ones((75,1))
        super(TestDynamicFactor_exog1, cls).setup_class(true, k_factors=1, factor_order=1, exog=exog)

    def test_predict(self):
        exog = np.ones((16, 1))
        super(TestDynamicFactor_exog1, self).test_predict(exog=exog)

    def test_dynamic_predict(self):
        exog = np.ones((16, 1))
        super(TestDynamicFactor_exog1, self).test_dynamic_predict(exog=exog)

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()**0.5
        assert_allclose(bse**2, self.true['var_oim'], atol=1e-5)

class TestDynamicFactor_exog2(CheckDynamicFactor):
    """
    Test for a dynamic factor model with 2 exogenous regressors: a constant
    and a time-trend
    """
    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_dfm_exog2.copy()
        true['predict'] = output_results.iloc[1:][['predict_dfm_exog2_1', 'predict_dfm_exog2_2', 'predict_dfm_exog2_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_dfm_exog2_1', 'dyn_predict_dfm_exog2_2', 'dyn_predict_dfm_exog2_3']]
        exog = np.c_[np.ones((75,1)), (np.arange(75) + 2)[:, np.newaxis]]
        super(TestDynamicFactor_exog2, cls).setup_class(true, k_factors=1, factor_order=1, exog=exog)

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()**0.5
        assert_allclose(bse**2, self.true['var_oim'], atol=1e-5)

    def test_predict(self):
        exog = np.c_[np.ones((16, 1)), (np.arange(75, 75+16) + 2)[:, np.newaxis]]
        super(TestDynamicFactor_exog2, self).test_predict(exog=exog)

    def test_dynamic_predict(self):
        exog = np.c_[np.ones((16, 1)), (np.arange(75, 75+16) + 2)[:, np.newaxis]]
        super(TestDynamicFactor_exog2, self).test_dynamic_predict(exog=exog)

    def test_summary(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Make sure we have the right number of tables
        assert_equal(len(tables), 2 + self.model.k_endog + self.model.k_factors + 1)

        # Check the model overview table
        assert_equal(re.search(r'Model:.*DynamicFactor\(factors=1, order=1\)', tables[0]) is None, False)
        assert_equal(re.search(r'.*2 regressors', tables[0]) is None, False)

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset_loading = self.model.k_factors * i
            offset_exog = self.model.k_factors * self.model.k_endog
            table = tables[i + 2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for equation %s' % name, table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 8)

            # -> Check that we have the right coefficients
            assert_equal(re.search('loading.f1 +' + forg(params[offset_loading + 0], prec=4), table) is None, False)
            assert_equal(re.search('beta.const +' + forg(params[offset_exog + i*2 + 0], prec=4), table) is None, False)
            assert_equal(re.search('beta.x1 +' + forg(params[offset_exog + i*2 + 1], prec=4), table) is None, False)

        # For each factor, check the output
        for i in range(self.model.k_factors):
            offset = self.model.k_endog * (self.model.k_factors + 3) + i * self.model.k_factors
            table = tables[self.model.k_endog + i + 2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for factor equation f%d' % (i+1), table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 6)

            # -> Check that we have the right coefficients
            assert_equal(re.search('L1.f1 +' + forg(params[offset + 0], prec=4), table) is None, False)

        # Check the Error covariance matrix output
        table = tables[2 + self.model.k_endog + self.model.k_factors]

        # -> Make sure we have the right table / table name
        name = self.model.endog_names[i]
        assert_equal(re.search('Error covariance matrix', table) is None, False)

        # -> Make sure it's the right size
        assert_equal(len(table.split('\n')), 8)

        # -> Check that we have the right coefficients
        offset = self.model.k_endog * (self.model.k_factors + 2)
        for i in range(self.model.k_endog):
            assert_equal(re.search('sigma2.%s +%s' % (self.model.endog_names[i], forg(params[offset + i], prec=4)), table) is None, False)

class TestDynamicFactor_general_errors(CheckDynamicFactor):
    """
    Test for a dynamic factor model where errors are as general as possible,
    meaning:

    - Errors are vector autocorrelated, VAR(1)
    - Innovations are correlated
    """
    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_dfm_gen.copy()
        true['predict'] = output_results.iloc[1:][['predict_dfm_gen_1', 'predict_dfm_gen_2', 'predict_dfm_gen_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_dfm_gen_1', 'dyn_predict_dfm_gen_2', 'dyn_predict_dfm_gen_3']]
        super(TestDynamicFactor_general_errors, cls).setup_class(true, k_factors=1, factor_order=1, error_var=True, error_order=1, error_cov_type='unstructured')

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()
        assert_allclose(bse[:3], self.true['var_oim'][:3], atol=1e-5)
        assert_allclose(bse[-10:], self.true['var_oim'][-10:], atol=2e-4)

    @skip("Known failure, no sequence of optimizers has been found which can achieve the maximum.")
    def test_mle(self):
        # The following gets us to llf=546.53, which is still not good enough
        # llf = 300.842477412
        # res = mod.fit(method='lbfgs', maxiter=10000)
        # llf = 460.26576722
        # res = mod.fit(res.params, method='nm', maxiter=10000, maxfev=10000)
        # llf = 542.245718508
        # res = mod.fit(res.params, method='lbfgs', maxiter=10000)
        # llf = 544.035160955
        # res = mod.fit(res.params, method='nm', maxiter=10000, maxfev=10000)
        # llf = 557.442240083
        # res = mod.fit(res.params, method='lbfgs', maxiter=10000)
        # llf = 558.199513262
        # res = mod.fit(res.params, method='nm', maxiter=10000, maxfev=10000)
        # llf = 559.049076604
        # res = mod.fit(res.params, method='nm', maxiter=10000, maxfev=10000)
        # llf = 559.049076604
        # ...
        pass

    def test_summary(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summary = self.results.summary()
        tables = [str(table) for table in summary.tables]
        params = self.true['params']

        # Make sure we have the right number of tables
        assert_equal(len(tables), 2 + self.model.k_endog + self.model.k_factors + self.model.k_endog + 1)

        # Check the model overview table
        assert_equal(re.search(r'Model:.*DynamicFactor\(factors=1, order=1\)', tables[0]) is None, False)
        assert_equal(re.search(r'.*VAR\(1\) errors', tables[0]) is None, False)

        # For each endogenous variable, check the output
        for i in range(self.model.k_endog):
            offset_loading = self.model.k_factors * i
            table = tables[i + 2]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for equation %s' % name, table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 6)

            # -> Check that we have the right coefficients
            assert_equal(re.search('loading.f1 +' + forg(params[offset_loading + 0], prec=4), table) is None, False)

        # For each factor, check the output
        for i in range(self.model.k_factors):
            offset = self.model.k_endog * self.model.k_factors + 6 + i * self.model.k_factors
            table = tables[2 + self.model.k_endog + i]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for factor equation f%d' % (i+1), table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 6)

            # -> Check that we have the right coefficients
            assert_equal(re.search('L1.f1 +' + forg(params[offset + 0], prec=4), table) is None, False)

        # For each error equation, check the output
        for i in range(self.model.k_endog):
            offset = self.model.k_endog * (self.model.k_factors + i) + 6 + self.model.k_factors
            table = tables[2 + self.model.k_endog + self.model.k_factors + i]

            # -> Make sure we have the right table / table name
            name = self.model.endog_names[i]
            assert_equal(re.search('Results for error equation e\(%s\)' % name, table) is None, False)

            # -> Make sure it's the right size
            assert_equal(len(table.split('\n')), 8)

            # -> Check that we have the right coefficients
            for j in range(self.model.k_endog):
                name = self.model.endog_names[j]
                assert_equal(re.search('L1.e\(%s\) +%s' % (name, forg(params[offset + j], prec=4)), table) is None, False)

        # Check the Error covariance matrix output
        table = tables[2 + self.model.k_endog + self.model.k_factors + self.model.k_endog]

        # -> Make sure we have the right table / table name
        name = self.model.endog_names[i]
        assert_equal(re.search('Error covariance matrix', table) is None, False)

        # -> Make sure it's the right size
        assert_equal(len(table.split('\n')), 11)

        # -> Check that we have the right coefficients
        offset = self.model.k_endog * self.model.k_factors
        assert_equal(re.search('sqrt.var.dln_inv +' + forg(params[offset + 0], prec=4), table) is None, False)
        assert_equal(re.search('sqrt.cov.dln_inv.dln_inc +' + forg(params[offset + 1], prec=4), table) is None, False)
        assert_equal(re.search('sqrt.var.dln_inc +' + forg(params[offset + 2], prec=4), table) is None, False)
        assert_equal(re.search('sqrt.cov.dln_inv.dln_consump +' + forg(params[offset + 3], prec=4), table) is None, False)
        assert_equal(re.search('sqrt.cov.dln_inc.dln_consump +' + forg(params[offset + 4], prec=4), table) is None, False)
        assert_equal(re.search('sqrt.var.dln_consump +' + forg(params[offset + 5], prec=4), table) is None, False)

class TestDynamicFactor_ar2_errors(CheckDynamicFactor):
    """
    Test for a dynamic factor model where errors are as general as possible,
    meaning:

    - Errors are vector autocorrelated, VAR(1)
    - Innovations are correlated
    """
    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_dfm_ar2.copy()
        true['predict'] = output_results.iloc[1:][['predict_dfm_ar2_1', 'predict_dfm_ar2_2', 'predict_dfm_ar2_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_dfm_ar2_1', 'dyn_predict_dfm_ar2_2', 'dyn_predict_dfm_ar2_3']]
        super(TestDynamicFactor_ar2_errors, cls).setup_class(true, k_factors=1, factor_order=1, error_order=2)

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()
        assert_allclose(bse, self.true['var_oim'], atol=1e-5)

    def test_mle(self):
        with warnings.catch_warnings(record=True) as w:
            # Depending on the system, this test can reach a greater precision,
            # but for cross-platform results keep it at 1e-2
            mod = self.model
            res1 = mod.fit(maxiter=100, optim_score='approx', disp=False)
            res = mod.fit(res1.params, method='nm', maxiter=10000, optim_score='approx', disp=False)
            assert_allclose(res.llf, self.results.llf, atol=1e-2)

class TestDynamicFactor_scalar_error(CheckDynamicFactor):
    """
    Test for a dynamic factor model where innovations are uncorrelated and
    are forced to have the same variance.
    """
    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_dfm_scalar.copy()
        true['predict'] = output_results.iloc[1:][['predict_dfm_scalar_1', 'predict_dfm_scalar_2', 'predict_dfm_scalar_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_dfm_scalar_1', 'dyn_predict_dfm_scalar_2', 'dyn_predict_dfm_scalar_3']]
        exog = np.ones((75,1))
        super(TestDynamicFactor_scalar_error, cls).setup_class(true, k_factors=1, factor_order=1, exog=exog, error_cov_type='scalar')

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()
        assert_allclose(bse, self.true['var_oim'], atol=1e-5)

    def test_predict(self):
        exog = np.ones((16, 1))
        super(TestDynamicFactor_scalar_error, self).test_predict(exog=exog)

    def test_dynamic_predict(self):
        exog = np.ones((16, 1))
        super(TestDynamicFactor_scalar_error, self).test_dynamic_predict(exog=exog)


class TestStaticFactor(CheckDynamicFactor):
    """
    Test for a static factor model (i.e. factors are not autocorrelated).
    """
    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_sfm.copy()
        true['predict'] = output_results.iloc[1:][['predict_sfm_1', 'predict_sfm_2', 'predict_sfm_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_sfm_1', 'dyn_predict_sfm_2', 'dyn_predict_sfm_3']]
        super(TestStaticFactor, cls).setup_class(true, k_factors=1, factor_order=0)

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()
        assert_allclose(bse, self.true['var_oim'], atol=1e-5)

    def test_bic(self):
        # Stata uses 5 df (i.e. 5 params) here instead of 6, because one param
        # is basically zero.
        pass


class TestSUR(CheckDynamicFactor):
    """
    Test for a seemingly unrelated regression model (i.e. no factors) with
    errors cross-sectionally, but not auto-, correlated
    """
    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_sur.copy()
        true['predict'] = output_results.iloc[1:][['predict_sur_1', 'predict_sur_2', 'predict_sur_3']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_sur_1', 'dyn_predict_sur_2', 'dyn_predict_sur_3']]
        exog = np.c_[np.ones((75,1)), (np.arange(75) + 2)[:, np.newaxis]]
        super(TestSUR, cls).setup_class(true, k_factors=0, factor_order=0, exog=exog, error_cov_type='unstructured')

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()
        assert_allclose(bse[:6], self.true['var_oim'][:6], atol=1e-5)

    def test_predict(self):
        exog = np.c_[np.ones((16, 1)), (np.arange(75, 75+16) + 2)[:, np.newaxis]]
        super(TestSUR, self).test_predict(exog=exog)

    def test_dynamic_predict(self):
        exog = np.c_[np.ones((16, 1)), (np.arange(75, 75+16) + 2)[:, np.newaxis]]
        super(TestSUR, self).test_dynamic_predict(exog=exog)


class TestSUR_autocorrelated_errors(CheckDynamicFactor):
    """
    Test for a seemingly unrelated regression model (i.e. no factors) where
    the errors are vector autocorrelated, but innovations are uncorrelated.

    """
    @classmethod
    def setup_class(cls):
        true = results_dynamic_factor.lutkepohl_sur_auto.copy()
        true['predict'] = output_results.iloc[1:][['predict_sur_auto_1', 'predict_sur_auto_2']]
        true['dynamic_predict'] = output_results.iloc[1:][['dyn_predict_sur_auto_1', 'dyn_predict_sur_auto_2']]
        exog = np.c_[np.ones((75,1)), (np.arange(75) + 2)[:, np.newaxis]]
        super(TestSUR_autocorrelated_errors, cls).setup_class(true, k_factors=0, factor_order=0, exog=exog, error_order=1, error_var=True, error_cov_type='diagonal', included_vars=['dln_inv', 'dln_inc'])

    def test_bse_approx(self):
        bse = self.results._cov_params_approx().diagonal()
        assert_allclose(bse, self.true['var_oim'], atol=1e-5)

    def test_predict(self):
        exog = np.c_[np.ones((16, 1)), (np.arange(75, 75+16) + 2)[:, np.newaxis]]
        super(TestSUR_autocorrelated_errors, self).test_predict(exog=exog)

    def test_dynamic_predict(self):
        exog = np.c_[np.ones((16, 1)), (np.arange(75, 75+16) + 2)[:, np.newaxis]]
        super(TestSUR_autocorrelated_errors, self).test_dynamic_predict(exog=exog)


def test_misspecification():
    # Tests for model specification and misspecification exceptions
    endog = np.arange(20).reshape(10,2)

    # Too few endog
    assert_raises(ValueError, dynamic_factor.DynamicFactor, endog[:,0], k_factors=0, factor_order=0)

    # Too many factors
    assert_raises(ValueError, dynamic_factor.DynamicFactor, endog, k_factors=2, factor_order=1)

    # Bad error_cov_type specification
    assert_raises(ValueError, dynamic_factor.DynamicFactor, endog, k_factors=1, factor_order=1, order=(1,0), error_cov_type='')

def test_miscellaneous():
    # Initialization with 1-dimensional exog array
    exog = np.arange(75)
    mod = CheckDynamicFactor()
    mod.setup_class(true=None, k_factors=1, factor_order=1, exog=exog, filter=False)
    exog = pd.Series(np.arange(75), index=pd.date_range(start='1960-04-01', end='1978-10-01', freq='QS'))
    mod = CheckDynamicFactor()
    mod.setup_class(true=None, k_factors=1, factor_order=1, exog=exog, filter=False)


def test_predict_custom_index():
    np.random.seed(328423)
    endog = pd.DataFrame(np.random.normal(size=(50, 2)))
    mod = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1)
    res = mod.smooth(mod.start_params)
    out = res.predict(start=1, end=1, index=['a'])
    assert_equal(out.index.equals(pd.Index(['a'])), True)
