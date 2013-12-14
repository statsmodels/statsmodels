"""
Test functions for tools.statstools
"""

import numpy as np
from numpy.testing import (assert_almost_equal, TestCase)

from statsmodels.stats import stattools


class TestStattools(TestCase):
    def test_medcouple_symmetric(self):
        mc = stattools.medcouple(np.arange(5.0))
        assert_almost_equal(mc, 0)


    def test_medcouple_nonzero(self):
        # Note: The R example is wrong here.  This is the correct value.
        mc = stattools.medcouple(np.array([1, 2, 7, 9, 10.0]))
        assert_almost_equal(mc, -0.38095238)

    def test_medcouple_symmetry(self):
        # Note: The R example is wrong here.  This is the correct value.
        x = np.random.standard_normal(100)
        mcp = stattools.medcouple(x)
        mcn = stattools.medcouple(-x)
        assert_almost_equal(mcp + mcn, 0)

