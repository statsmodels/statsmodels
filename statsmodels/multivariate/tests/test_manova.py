# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA, Cancorr
from numpy.testing import assert_almost_equal, assert_raises_regex

# Example data
# https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/
#     viewer.htm#statug_introreg_sect012.htm
X = pd.DataFrame([['Minas Graes', 2.068, 2.070, 1.580],
                  ['Minas Graes', 2.068, 2.074, 1.602],
                  ['Minas Graes', 2.090, 2.090, 1.613],
                  ['Minas Graes', 2.097, 2.093, 1.613],
                  ['Minas Graes', 2.117, 2.125, 1.663],
                  ['Minas Graes', 2.140, 2.146, 1.681],
                  ['Matto Grosso', 2.045, 2.054, 1.580],
                  ['Matto Grosso', 2.076, 2.088, 1.602],
                  ['Matto Grosso', 2.090, 2.093, 1.643],
                  ['Matto Grosso', 2.111, 2.114, 1.643],
                  ['Santa Cruz', 2.093, 2.098, 1.653],
                  ['Santa Cruz', 2.100, 2.106, 1.623],
                  ['Santa Cruz', 2.104, 2.101, 1.653]],
                 columns=['Loc', 'Basal', 'Occ', 'Max'])

data_fit = pd.DataFrame([[191, 36, 50,  5, 162,  60],
                         [189, 37, 52,  2, 110,  60],
                         [193, 38, 58, 12, 101, 101],
                         [162, 35, 62, 12, 105,  37],
                         [189, 35, 46, 13, 155,  58],
                         [182, 36, 56,  4, 101,  42],
                         [211, 38, 56,  8, 101,  38],
                         [167, 34, 60,  6, 125,  40],
                         [176, 31, 74, 15, 200,  40],
                         [154, 33, 56, 17, 251, 250],
                         [169, 34, 50, 17, 120,  38],
                         [166, 33, 52, 13, 210, 115],
                         [154, 34, 64, 14, 215, 105],
                         [247, 46, 50,  1,  50,  50],
                         [193, 36, 46,  6,  70,  31],
                         [202, 37, 62, 12, 210, 120],
                         [176, 37, 54,  4,  60,  25],
                         [157, 32, 52, 11, 230,  80],
                         [156, 33, 54, 15, 225,  73],
                         [138, 33, 68,  2, 110,  43]])
def test_manova_sas_example():
    # Results should be the same as figure 4.5 of
    # https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/
    # viewer.htm#statug_introreg_sect012.htm
    mod = MANOVA.from_formula('Basal + Occ + Max ~ Loc', data=X)
    r = mod.test()
    assert_almost_equal(r[1][1].loc["Wilks’ lambda", 'Value'],
                        0.60143661, decimal=8)
    assert_almost_equal(r[1][1].loc["Pillai’s trace", 'Value'],
                        0.44702843, decimal=8)
    assert_almost_equal(r[1][1].loc["Hotelling-Lawley trace", 'Value'],
                        0.58210348, decimal=8)
    assert_almost_equal(r[1][1].loc["Roy’s greatest root", 'Value'],
                        0.35530890, decimal=8)
    assert_almost_equal(r[1][1].loc["Wilks’ lambda", 'F Value'],
                        0.77, decimal=2)
    assert_almost_equal(r[1][1].loc["Pillai’s trace", 'F Value'],
                        0.86, decimal=2)
    assert_almost_equal(r[1][1].loc["Hotelling-Lawley trace", 'F Value'],
                        0.75, decimal=2)
    assert_almost_equal(r[1][1].loc["Roy’s greatest root", 'F Value'],
                        1.07, decimal=2)
    assert_almost_equal(r[1][1].loc["Wilks’ lambda", 'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r[1][1].loc["Pillai’s trace", 'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r[1][1].loc["Hotelling-Lawley trace", 'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r[1][1].loc["Roy’s greatest root", 'Num DF'],
                        3, decimal=3)
    assert_almost_equal(r[1][1].loc["Wilks’ lambda", 'Den DF'],
                        16, decimal=3)
    assert_almost_equal(r[1][1].loc["Pillai’s trace", 'Den DF'],
                        18, decimal=3)
    assert_almost_equal(r[1][1].loc["Hotelling-Lawley trace", 'Den DF'],
                        9.0909, decimal=4)
    assert_almost_equal(r[1][1].loc["Roy’s greatest root", 'Den DF'],
                        9, decimal=3)
    assert_almost_equal(r[1][1].loc["Wilks’ lambda", 'Pr > F'],
                        0.6032, decimal=4)
    assert_almost_equal(r[1][1].loc["Pillai’s trace", 'Pr > F'],
                        0.5397, decimal=4)
    assert_almost_equal(r[1][1].loc["Hotelling-Lawley trace", 'Pr > F'],
                        0.6272, decimal=4)
    assert_almost_equal(r[1][1].loc["Roy’s greatest root", 'Pr > F'],
                        0.4109, decimal=4)


def test_manova_interaction_term():
    mod = MANOVA.from_formula('Basal + Occ ~ Loc * Max', data=X)
    r = mod.test()
    # H-L race R ouput is different compared to SAS
    assert_almost_equal(r[3][1].loc["Wilks’ lambda", 'Value'],
                        0.30923, decimal=4)
    assert_almost_equal(r[3][1].loc["Pillai’s trace", 'Value'],
                        0.84231, decimal=4)
    assert_almost_equal(r[3][1].loc["Roy’s greatest root", 'Value'],
                        1.3917, decimal=4)
    assert_almost_equal(r[3][1].loc["Wilks’ lambda", 'F Value'],
                        2.3949, decimal=4)
    assert_almost_equal(r[3][1].loc["Pillai’s trace", 'F Value'],
                        2.5465, decimal=4)
    assert_almost_equal(r[3][1].loc["Roy’s greatest root", 'F Value'],
                        4.8708, decimal=4)
    assert_almost_equal(r[3][1].loc["Wilks’ lambda", 'Num DF'],
                        4, decimal=7)
    assert_almost_equal(r[3][1].loc["Pillai’s trace", 'Num DF'],
                        4, decimal=7)
    assert_almost_equal(r[3][1].loc["Roy’s greatest root", 'Num DF'],
                        2, decimal=7)
    assert_almost_equal(r[3][1].loc["Wilks’ lambda", 'Den DF'],
                        12, decimal=7)
    assert_almost_equal(r[3][1].loc["Pillai’s trace", 'Den DF'],
                        14, decimal=7)
    assert_almost_equal(r[3][1].loc["Roy’s greatest root", 'Den DF'],
                        7, decimal=7)
    assert_almost_equal(r[3][1].loc["Wilks’ lambda", 'Pr > F'],
                        0.1083267, decimal=7)
    assert_almost_equal(r[3][1].loc["Pillai’s trace", 'Pr > F'],
                        0.0859654, decimal=7)
    assert_almost_equal(r[3][1].loc["Roy’s greatest root", 'Pr > F'],
                        0.0472659, decimal=7)


def test_manova_test_input_validation():
    mod = MANOVA.from_formula('Basal + Occ + Max ~ Loc', data=X)
    hypothesis = [('test', np.array([[1, 1, 1]]), None)]
    mod.test(hypothesis)
    hypothesis = [('test', np.array([[1, 1]]), None)]
    assert_raises_regex(ValueError,
                        ('Contrast matrix L should have the same number of '
                         'columns as exog! 2 != 3'),
                        mod.test, hypothesis)
    hypothesis = [('test', np.array([[1, 1, 1]]), np.array([[1], [1], [1]]))]
    mod.test(hypothesis)
    hypothesis = [('test', np.array([[1, 1, 1]]), np.array([[1], [1]]))]
    assert_raises_regex(ValueError,
                        ('Transform matrix M should have the same number of '
                         'rows as the number of columns of endog! 2 != 3'),
                        mod.test, hypothesis)


def test_cancorr():
    X1 = data_fit.iloc[:, :3]
    Y1 = data_fit.iloc[:, 3:]
    mod = Cancorr(Y1, X1).fit()
    assert_almost_equal(mod.multi_stats.loc["Wilks’ lambda", 'Value'],
                        0.35039053, decimal=8)
    assert_almost_equal(mod.multi_stats.loc["Pillai’s trace", 'Value'],
                        0.67848151, decimal=8)
    assert_almost_equal(mod.multi_stats.loc["Hotelling-Lawley trace", 'Value'],
                        1.77194146, decimal=8)
    assert_almost_equal(mod.multi_stats.loc["Roy’s greatest root", 'Value'],
                        1.72473874, decimal=8)
    assert_almost_equal(mod.multi_stats.loc["Wilks’ lambda", 'F Value'],
                        2.05, decimal=2)
    assert_almost_equal(mod.multi_stats.loc["Pillai’s trace", 'F Value'],
                        1.56, decimal=2)
    assert_almost_equal(mod.multi_stats.loc["Hotelling-Lawley trace",
                                            'F Value'],
                        2.64, decimal=2)
    assert_almost_equal(mod.multi_stats.loc["Roy’s greatest root", 'F Value'],
                        9.20, decimal=2)
    assert_almost_equal(mod.multi_stats.loc["Wilks’ lambda", 'Num DF'],
                        9, decimal=3)
    assert_almost_equal(mod.multi_stats.loc["Pillai’s trace", 'Num DF'],
                        9, decimal=3)
    assert_almost_equal(mod.multi_stats.loc["Hotelling-Lawley trace",
                                            'Num DF'],
                        9, decimal=3)
    assert_almost_equal(mod.multi_stats.loc["Roy’s greatest root", 'Num DF'],
                        3, decimal=3)
    assert_almost_equal(mod.multi_stats.loc["Wilks’ lambda", 'Den DF'],
                        34.223, decimal=3)
    assert_almost_equal(mod.multi_stats.loc["Pillai’s trace", 'Den DF'],
                        48, decimal=3)
    assert_almost_equal(mod.multi_stats.loc["Hotelling-Lawley trace",
                                            'Den DF'],
                        19.053, decimal=3)
    assert_almost_equal(mod.multi_stats.loc["Roy’s greatest root", 'Den DF'],
                        16, decimal=3)
    assert_almost_equal(mod.multi_stats.loc["Wilks’ lambda", 'Pr > F'],
                        0.0635, decimal=4)
    assert_almost_equal(mod.multi_stats.loc["Pillai’s trace", 'Pr > F'],
                        0.1551, decimal=4)
    assert_almost_equal(mod.multi_stats.loc["Hotelling-Lawley trace",
                                            'Pr > F'],
                        0.0357, decimal=4)
    assert_almost_equal(mod.multi_stats.loc["Roy’s greatest root", 'Pr > F'],
                        0.0009, decimal=4)
    assert_almost_equal(mod.stats.loc[0, "Wilks' lambda"],
                        0.35039053, decimal=8)
    assert_almost_equal(mod.stats.loc[1, "Wilks' lambda"],
                        0.95472266, decimal=8)
    assert_almost_equal(mod.stats.loc[2, "Wilks' lambda"],
                        0.99473355, decimal=8)
    assert_almost_equal(mod.stats.loc[0, 'F Value'],
                        2.05, decimal=2)
    assert_almost_equal(mod.stats.loc[1, 'F Value'],
                        0.18, decimal=2)
    assert_almost_equal(mod.stats.loc[2, 'F Value'],
                        0.08, decimal=2)
    assert_almost_equal(mod.stats.loc[0, 'Num DF'],
                        9, decimal=2)
    assert_almost_equal(mod.stats.loc[1, 'Num DF'],
                        4, decimal=2)
    assert_almost_equal(mod.stats.loc[2, 'Num DF'],
                        1, decimal=2)
    assert_almost_equal(mod.stats.loc[0, 'Den DF'],
                        34.223, decimal=3)
    assert_almost_equal(mod.stats.loc[1, 'Den DF'],
                        30, decimal=2)
    assert_almost_equal(mod.stats.loc[2, 'Den DF'],
                        16, decimal=2)
    assert_almost_equal(mod.stats.loc[0, 'Pr > F'],
                        0.0635, decimal=4)
    assert_almost_equal(mod.stats.loc[1, 'Pr > F'],
                        0.9491, decimal=4)
    assert_almost_equal(mod.stats.loc[2, 'Pr > F'],
                        0.7748, decimal=4)
