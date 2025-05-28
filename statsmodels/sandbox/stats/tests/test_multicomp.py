"""
Tests corresponding to sandbox.stats.multicomp
"""

from statsmodels.compat.scipy import SP_LT_116

import numpy as np
from numpy.testing import assert_almost_equal
import pytest
import pandas as pd

from statsmodels.sandbox.stats.multicomp import tukey_pvalues
from .results import result_hsubsets_dataframe


@pytest.mark.skipif(not SP_LT_116, reason="mvndst removed in SciPy 1.16")
def test_tukey_pvalues():
    # TODO: testcase with 3 is not good because all pairs
    #  has also 3*(3-1)/2=3 elements
    res = tukey_pvalues(3.649, 3, 16)
    assert_almost_equal(0.05, res[0], 3)
    assert_almost_equal(0.05 * np.ones(3), res[1], 3)


@pytest.mark.parametrize(
    "alpha, expected_df",
    [
        (0.01, result_hsubsets_dataframe.expected_df_alpha001),
        (0.05, result_hsubsets_dataframe.expected_df_alpha005),
        (0.1, result_hsubsets_dataframe.expected_df_alpha01),
    ],
)
def test_create_homogeneous_subsets_dataframe(
    alpha,
    expected_df,
):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.sandbox.stats.multicomp import TukeyHSDResults

    data = result_hsubsets_dataframe.data
    groups = result_hsubsets_dataframe.groups

    tukey = pairwise_tukeyhsd(
        endog=data,
        groups=groups,
        alpha=alpha,
    )
    assert isinstance(tukey, TukeyHSDResults)
    hs_df = tukey.create_homogeneous_subsets_dataframe()
    assert isinstance(hs_df, pd.DataFrame)
    assert hs_df.equals(expected_df)
    assert hs_df.index.name == "Group"
