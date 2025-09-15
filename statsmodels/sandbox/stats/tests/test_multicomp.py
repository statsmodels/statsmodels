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

@pytest.fixture
def sample_data_balanced():
    df = pd.read_csv('./tests/results/sample_data_balanced.csv')
    return df

@pytest.fixture
def sample_data_unbalanced():
    df = pd.read_csv('./tests/results/sample_data_unbalanced.csv')
    return df

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
def test_create_homogeneous_subsets_dataframe_balanced(
    alpha,
    expected_df,
    sample_data_balanced
):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.sandbox.stats.multicomp import TukeyHSDResults

    data = sample_data_balanced["data"]
    groups = sample_data_balanced["groups"]

    tukey = pairwise_tukeyhsd(
        endog=data,
        groups=groups,
        alpha=alpha,
    )
    assert isinstance(tukey, TukeyHSDResults)
    hs_df = tukey.create_homogeneous_subsets_dataframe()
    assert isinstance(hs_df, pd.DataFrame)
    assert pd.testing.assert_frame_equal(hs_df, expected_df) is None
    assert hs_df.index.name == "Group"


@pytest.mark.parametrize(
    "alpha, expected_df",
    [
        (0.05, result_hsubsets_dataframe.expected_df_unbalanced_alpha005),
    ],
)
def test_create_homogeneous_subsets_dataframe_unbalanced(
    alpha,
    expected_df,
    sample_data_unbalanced
):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.sandbox.stats.multicomp import TukeyHSDResults

    data = sample_data_unbalanced["data"]
    groups = sample_data_unbalanced["groups"]

    tukey = pairwise_tukeyhsd(
        endog=data,
        groups=groups,
        alpha=alpha,
    )
    assert isinstance(tukey, TukeyHSDResults)
    hs_df = tukey.create_homogeneous_subsets_dataframe()
    assert isinstance(hs_df, pd.DataFrame)
    assert pd.testing.assert_frame_equal(hs_df, expected_df) is None
    assert hs_df.index.name == "Group"
