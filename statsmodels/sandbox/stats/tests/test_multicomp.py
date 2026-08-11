"""
Tests corresponding to sandbox.stats.multicomp
"""
from statsmodels.compat.scipy import SP_LT_2, SP_LT_116

import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from scipy import stats

from statsmodels.sandbox.stats.multicomp import (
    get_tukeyQcrit,
    tiecorrect,
    tukey_pvalues,
)


@pytest.mark.skipif(not SP_LT_116, reason="mvndst removed in SciPy 1.16")
def test_tukey_pvalues():
    # TODO: testcase with 3 is not good because all pairs
    #  has also 3*(3-1)/2=3 elements
    res = tukey_pvalues(3.649, 3, 16)
    assert_almost_equal(0.05, res[0], 3)
    assert_almost_equal(0.05 * np.ones(3), res[1], 3)


def test_get_tukeyqcrit_matches_published_table():
    # reference values from the standard Tukey HSD critical value table
    # (k=8 treatments, df=8 error degrees of freedom)
    assert_almost_equal(get_tukeyQcrit(8, 8, alpha=0.05), 5.60, decimal=2)
    assert_almost_equal(get_tukeyQcrit(8, 8, alpha=0.01), 7.47, decimal=2)


def test_get_tukeyqcrit_invalid_alpha_raises():
    with pytest.raises(ValueError, match="only implemented"):
        get_tukeyQcrit(8, 8, alpha=0.1)


def test_tiecorrect_matches_scipy():
    xranks = stats.rankdata([7.68, 7.69, 7.70, 7.70, 7.72, 7.73, 7.73, 7.76])
    if SP_LT_2:
        expected_result = stats.tiecorrect(xranks)
    else:
        # Saved from SciPy's stats.tiecorrect before deprecation
        expected_result = 0.9761904761904762
    assert_almost_equal(tiecorrect(xranks), expected_result, decimal=8)
