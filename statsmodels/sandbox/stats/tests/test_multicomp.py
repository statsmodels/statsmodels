"""
Tests corresponding to sandbox.stats.multicomp
"""

from statsmodels.compat.scipy import SP_LT_116

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from statsmodels.sandbox.stats.multicomp import tukey_pvalues


@pytest.mark.skipif(not SP_LT_116, reason="SciPy < 1.16.0 required")
def test_tukey_pvalues():
    # TODO: testcase with 3 is not good because all pairs
    #  has also 3*(3-1)/2=3 elements
    res = tukey_pvalues(3.649, 3, 16)
    assert_almost_equal(0.05, res[0], 3)
    assert_almost_equal(0.05 * np.ones(3), res[1], 3)
