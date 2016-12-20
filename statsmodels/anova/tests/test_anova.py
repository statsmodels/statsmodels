import pandas as pd
import pytest
from ..anova import ANOVA
from numpy.testing import assert_array_almost_equal


DV = [7, 3, 6, 6, 5, 8, 6, 7,
      7, 11, 9, 11, 10, 10, 11, 11,
      8, 14, 10, 11, 12, 10, 11, 12,
      16, 7, 11, 9, 10, 11, 8, 8,
      16, 10, 13, 10, 10, 14, 11, 12,
      24, 29, 10, 22, 25, 28, 22, 24,
      1, 3, 5, 8, 3, 5, 6, 8,
      9, 18, 19, 1, 12, 15, 2, 3,
      3, 4, 13, 21, 2, 11, 18, 2,
      12, 7, 12, 3, 19, 1, 4, 13,
      13, 14, 3, 4, 8, 19, 21, 2,
      4, 9, 12, 2, 5, 8, 2, 4]

id = [1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8]

id = ['%d' % i for i in id]

A = ['a','a','a','a','a','a','a','a',
     'a','a','a','a','a','a','a','a',
     'a','a','a','a','a','a','a','a',
     'b','b','b','b','b','b','b','b',
     'b','b','b','b','b','b','b','b',
     'b','b','b','b','b','b','b','b',
     'a','a','a','a','a','a','a','a',
     'a','a','a','a','a','a','a','a',
     'a','a','a','a','a','a','a','a',
     'b','b','b','b','b','b','b','b',
     'b','b','b','b','b','b','b','b',
     'b','b','b','b','b','b','b','b']

B = ['a','a','a','a','a','a','a','a',
     'b','b','b','b','b','b','b','b',
     'c','c','c','c','c','c','c','c',
     'a','a','a','a','a','a','a','a',
     'b','b','b','b','b','b','b','b',
     'c','c','c','c','c','c','c','c',
     'a','a','a','a','a','a','a','a',
     'b','b','b','b','b','b','b','b',
     'c','c','c','c','c','c','c','c',
     'a','a','a','a','a','a','a','a',
     'b','b','b','b','b','b','b','b',
     'c','c','c','c','c','c','c','c']

D = ['a','a','a','a','a','a','a','a',
     'a','a','a','a','a','a','a','a',
     'a','a','a','a','a','a','a','a',
     'a','a','a','a','a','a','a','a',
     'a','a','a','a','a','a','a','a',
     'a','a','a','a','a','a','a','a',
     'b','b','b','b','b','b','b','b',
     'b','b','b','b','b','b','b','b',
     'b','b','b','b','b','b','b','b',
     'b','b','b','b','b','b','b','b',
     'b','b','b','b','b','b','b','b',
     'b','b','b','b','b','b','b','b']

data = pd.DataFrame([id, A, B, D, DV], index=['id', 'A', 'B', 'D', 'DV']).T


def test_single_factor_repeated_measures_anova():
    """
    Testing single factor repeated measures anova
    Results reproduces R `ezANOVA` function from library ez
    """
    df = ANOVA(data.iloc[:16, :], 'DV', within=['B'], subject='id').fit()
    a = [[1, 7, 22.4, 0.002125452]]
    assert_array_almost_equal(df.anova_table.iloc[:, [1, 2, 0, 3]].values,
                              a, decimal=5)


def test_two_factors_repeated_measures_anova():
    """
    Testing two factors repeated measures anova
    Results reproduces R `ezANOVA` function from library ez
    """
    df = ANOVA(data.iloc[:48, :], 'DV', within=['A', 'B'], subject='id').fit()
    a = [[1, 7, 40.14159, 3.905263e-04],
         [2, 14, 29.21739, 1.007549e-05],
         [2, 14, 17.10545, 1.741322e-04]]
    assert_array_almost_equal(df.anova_table.iloc[:,[1, 2, 0, 3]].values,
                              a, decimal=5)


def test_three_factors_repeated_measures_anova():
    """
    Testing three factors repeated measures anova
    Results reproduces R `ezANOVA` function from library ez
    """
    df = ANOVA(data, 'DV', within=['A', 'B', 'D'], subject='id').fit()
    a = [[1,  7,  8.7650709, 0.021087505],
         [2, 14,  8.4985785, 0.003833921],
         [1,  7, 20.5076546, 0.002704428],
         [2, 14,  0.8457797, 0.450021759],
         [1,  7, 21.7593382, 0.002301792],
         [2, 14,  6.2416695, 0.011536846],
         [2, 14,  5.4253359, 0.018010647]]
    assert_array_almost_equal(df.anova_table.iloc[:,[1, 2, 0, 3]].values,
                              a, decimal=5)


def test_invalid_factor_name():
    """
    Test with a factor name of 'C', which conflicts with patsy.
    """
    C = B

    with pytest.raises(ValueError):
        ANOVA(data.iloc[:48, :], 'DV', within=['A', 'C'], subject='id').fit()
