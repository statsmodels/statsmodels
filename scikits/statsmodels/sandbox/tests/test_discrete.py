"""
Tests for discrete models
"""

import numpy as np
from numpy.testing import *
from scikits.statsmodels.sandbox.discretemod import *

DECIMAL = 4
DECIMAL_less = 3
DECIMAL_lesser = 2
DECIMAL_least = 1
DECIMAL_none = 0

class ChecKModelResults(object):
    """
    res2 should be the test results from RModelWrap
    or the results as defined in model_results_data
    """
    def test_params:
        pass

    def test_conf_int:
        pass

    def test_cov_params:
        pass

    def test_f_test:
        pass
    # does this even make sense?  should be lr_test

    def test_llf:
        pass

    def test_llr:
        pass

    def test_margeff:
        pass
    # this probably needs it's own test class?

    def test_normalized_cov_params:
        pass
    # keep this one?

    def test_params:
        pass

    def test_t:
        pass

    def test_t_test:
        pass

class TestProbit(CheckModelResults):
    pass

class TestLogit(CheckModelResults):
    pass

class TestPoisson(CheckModelResults):
    pass

class testMNLogit(CheckModelResults):
    pass



