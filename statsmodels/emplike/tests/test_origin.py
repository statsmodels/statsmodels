import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pytest

from statsmodels.datasets import cancer
from statsmodels.emplike.descriptive import EmpLikeTestResult
from statsmodels.emplike.originregress import ELOriginRegress
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from .results.el_results import OriginResults


class GenRes:
    """
    Loads data and creates class instance ot be tested.
    """
    @classmethod
    def setup_class(cls):
        data = cancer.load()
        cls.res1 = ELOriginRegress(data.endog, data.exog).fit()
        cls.res2 = OriginResults()


class TestOrigin(GenRes):
    """
    See OriginResults for details on how tests were computed
    """
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.test_params, 4)

    def test_predict(self):
        # Built independently of the class fixture's `res1.model` (an OLS
        # instance with an added constant, not an ELOriginRegress), so that
        # this exercises ELOriginRegress.predict itself.
        data = cancer.load()
        mod = ELOriginRegress(data.endog, data.exog)
        res = mod.fit()
        exog = np.asarray(mod.exog)

        # documented formula: predict = [1, exog] @ params, intercept fixed 0
        pred = mod.predict(res.params)
        expected = np.dot(add_constant(exog, prepend=True), res.params)
        assert_allclose(pred, expected, rtol=1e-12)

        # explicit in-sample exog matches the exog=None default
        assert_allclose(pred, mod.predict(res.params, exog=exog), rtol=1e-12)

        # out-of-sample exog
        new_exog = exog[:5] + 1.0
        pred_new = mod.predict(res.params, exog=new_exog)
        expected_new = np.dot(add_constant(new_exog, prepend=True), res.params)
        assert_allclose(pred_new, expected_new, rtol=1e-12)

        # sanity check against a plain origin-constrained OLS fit (no EL
        # weighting) on the same data: predictions should be close, since
        # both are consistent estimators of the same origin-restricted
        # linear relationship.
        ols_origin = OLS(np.asarray(mod.endog), exog).fit()
        pred_ols = ols_origin.predict(new_exog)
        assert_allclose(pred_new, pred_ols, rtol=0.05)

    def test_llf(self):
        assert_almost_equal(self.res1.llf_el, self.res2.test_llf_hat, 4)

    def test_hypothesis_beta1(self):
        res = self.res1.el_test([.0034], [1], result_object=True)
        assert_almost_equal(res.llr, self.res2.test_llf_hypoth, 4)

    def test_el_test_namedtuple(self):
        # return_weights=True already yields three values, so the NamedTuple
        # is adopted silently
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            full = self.res1.el_test([.0034], [1], return_weights=1)
        assert isinstance(full, EmpLikeTestResult)
        assert_equal(len(full), 3)

        # the two-value path still warns and still returns a plain tuple
        with pytest.warns(FutureWarning, match="el_test"):
            legacy = self.res1.el_test([.0034], [1])
        assert not isinstance(legacy, EmpLikeTestResult)
        assert_equal(len(legacy), 2)
        assert_almost_equal(legacy, full[:2], 10)

        # weights are not computed when they are not requested
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            opted_in = self.res1.el_test([.0034], [1], result_object=True)
        assert isinstance(opted_in, EmpLikeTestResult)
        assert opted_in.weights is None

    def test_ci_beta(self):
        ci = self.res1.conf_int_el(1)
        ll = ci[0]
        ul = ci[1]
        llf_low = np.sum(np.log(self.res1.el_test([ll], [1],
                                                  return_weights=1)[2]))
        llf_high = np.sum(np.log(self.res1.el_test([ul], [1],
                                                   return_weights=1)[2]))
        assert_almost_equal(llf_low, self.res2.test_llf_conf, 4)
        assert_almost_equal(llf_high, self.res2.test_llf_conf, 4)
