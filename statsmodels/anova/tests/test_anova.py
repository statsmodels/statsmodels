import pandas as pd
from ..anova import ANOVA
from numpy.testing import assert_array_almost_equal

DV = [7, 3, 6, 6, 5, 8, 6, 7,
      7, 11, 9, 11, 10, 10, 11, 11,
      8, 14, 10, 11, 12, 10, 11, 12,
      16, 7, 11, 9, 10, 11, 8, 8,
      16, 10, 13, 10, 10, 14, 11, 12,
      24, 29, 10, 22, 25, 28, 22, 24]
id = [1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8]
A =  ['a','a','a','a','a','a','a','a',
      'a','a','a','a','a','a','a','a',
      'a','a','a','a','a','a','a','a',
      'b','b','b','b','b','b','b','b',
      'b','b','b','b','b','b','b','b',
      'b','b','b','b','b','b','b','b']
B =  ['a','a','a','a','a','a','a','a',
      'b','b','b','b','b','b','b','b',
      'c','c','c','c','c','c','c','c',
      'a','a','a','a','a','a','a','a',
      'b','b','b','b','b','b','b','b',
      'c','c','c','c','c','c','c','c']

data = pd.DataFrame([id, A, B, DV], index=['id', 'A', 'B', 'DV']).T


def test_two_factors_repeated_measures_anova():
    """
    Results reproduces R `ezANOVA` function from library ez
    """
    dv = 'DV'
    within  = ['A', 'B']
    subject = 'id'
    df = ANOVA(data, 'DV', within=['A', 'B'], subject='id').fit()
    a = [[1, 7, 40.14159, 3.905263e-04],
         [2, 14, 29.21739, 1.007549e-05],
         [2, 14, 17.10545, 1.741322e-04]]
    assert_array_almost_equal(df.anova_table.iloc[:,[1, 2, 0, 3]].values,
                              a, decimal=5)
