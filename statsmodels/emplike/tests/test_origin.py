import warnings

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest

from statsmodels.datasets import cancer
from statsmodels.emplike.descriptive import EmpLikeTestResult
from statsmodels.emplike.originregress import ELOriginRegress

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

    def test_llf(self):
        assert_almost_equal(self.res1.llf_el, self.res2.test_llf_hat, 4)

    def test_hypothesis_beta1(self):
        res = self.res1.el_test([.0034], [1], use_namedtuple=True)
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
            opted_in = self.res1.el_test([.0034], [1], use_namedtuple=True)
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
