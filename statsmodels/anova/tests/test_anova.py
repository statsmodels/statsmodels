import pandas as pd
from ..anova import anova_b1r1, anova_r1, anova_r2

X = pd.DataFrame([[1.17, 1.78, 1.29, 1.29],
                  [1.77, 1.98, 1.99, 1.99],
                  [1.49, 1.69, 1.79, 1.59],
                  [0.65, 0.99, 0.69, 1.09],
                  [1.58, 1.70, 1.89, 1.89],
                  [3.13, 3.15, 2.99, 3.09],
                  [2.09, 1.88, 2.09, 2.49],
                  [0.62, 0.65, 0.65, 0.69],
                  [5.89, 5.99, 5.99, 6.99],
                  [4.46, 4.84, 4.99, 5.15]])

def test_anova_r1():
    """
    .. [1] https://ww2.coastal.edu/kingw/statistics/R-tutorials/repeated.html
    """

    print(anova_r1(X))