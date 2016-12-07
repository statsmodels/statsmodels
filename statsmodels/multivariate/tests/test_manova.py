# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from numpy.testing import assert_almost_equal, assert_raises_regex
from numpy.testing import assert_array_almost_equal

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


def test_manova_sas_example():
    # Results should be the same as figure 4.5 of
    # https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/
    # viewer.htm#statug_introreg_sect012.htm
    mod = MANOVA.from_formula('Basal + Occ + Max ~ Loc', data=X)
    r = mod.test()
    assert_almost_equal(r['Loc'].loc["Wilks’ lambda", 'Value'],
                        0.60143661, decimal=8)
    assert_almost_equal(r['Loc'].loc["Pillai’s trace", 'Value'],
                        0.44702843, decimal=8)
    assert_almost_equal(r['Loc'].loc["Hotelling-Lawley trace", 'Value'],
                        0.58210348, decimal=8)
    assert_almost_equal(r['Loc'].loc["Roy’s greatest root", 'Value'],
                        0.35530890, decimal=8)
    assert_almost_equal(r['Loc'].loc["Wilks’ lambda", 'F Value'],
                        0.77, decimal=2)
    assert_almost_equal(r['Loc'].loc["Pillai’s trace", 'F Value'],
                        0.86, decimal=2)
    assert_almost_equal(r['Loc'].loc["Hotelling-Lawley trace", 'F Value'],
                        0.75, decimal=2)
    assert_almost_equal(r['Loc'].loc["Roy’s greatest root", 'F Value'],
                        1.07, decimal=2)
    assert_almost_equal(r['Loc'].loc["Wilks’ lambda", 'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r['Loc'].loc["Pillai’s trace", 'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r['Loc'].loc["Hotelling-Lawley trace", 'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r['Loc'].loc["Roy’s greatest root", 'Num DF'],
                        3, decimal=3)
    assert_almost_equal(r['Loc'].loc["Wilks’ lambda", 'Den DF'],
                        16, decimal=3)
    assert_almost_equal(r['Loc'].loc["Pillai’s trace", 'Den DF'],
                        18, decimal=3)
    assert_almost_equal(r['Loc'].loc["Hotelling-Lawley trace", 'Den DF'],
                        9.0909, decimal=4)
    assert_almost_equal(r['Loc'].loc["Roy’s greatest root", 'Den DF'],
                        9, decimal=3)
    assert_almost_equal(r['Loc'].loc["Wilks’ lambda", 'Pr > F'],
                        0.6032, decimal=4)
    assert_almost_equal(r['Loc'].loc["Pillai’s trace", 'Pr > F'],
                        0.5397, decimal=4)
    assert_almost_equal(r['Loc'].loc["Hotelling-Lawley trace", 'Pr > F'],
                        0.6272, decimal=4)
    assert_almost_equal(r['Loc'].loc["Roy’s greatest root", 'Pr > F'],
                        0.4109, decimal=4)


def test_compare_r_lm_anova_output_dogs_data():
    ''' Testing  with 2 between-subject effect
    Compares with R lm anova output
    '''
    data = pd.DataFrame([['Morphine', 'N', .04, .20, .10, .08],
                         ['Morphine', 'N', .02, .06, .02, .02],
                         ['Morphine', 'N', .07, 1.40, .48, .24],
                         ['Morphine', 'N', .17, .57, .35, .24],
                         ['Morphine', 'Y', .10, .09, .13, .14],
                         ['placebo', 'Y', .07, .07, .06, .07],
                         ['placebo', 'Y', .05, .07, .06, .07],
                         ['placebo', 'N', .03, .62, .31, .22],
                         ['placebo', 'N', .03, 1.05, .73, .60],
                         ['placebo', 'N', .07, .83, 1.07, .80],
                         ['Trimethaphan', 'N', .09, 3.13, 2.06, 1.23],
                         ['Trimethaphan', 'Y', .10, .09, .09, .08],
                         ['Trimethaphan', 'Y', .08, .09, .09, .10],
                         ['Trimethaphan', 'Y', .13, .10, .12, .12],
                         ['Trimethaphan', 'Y', .06, .05, .05, .05]],
                        columns=['Drug', 'Depleted', 'Histamine0',
                                 'Histamine1',
                                 'Histamine3', 'Histamine5'])

    for i in range(2,6):
        data.iloc[:, i] = np.log(data.iloc[:, i])

    mod = MANOVA.from_formula(
        'Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted',
        data)
    r = mod.test()
    a = [[6.63472514e-03, 4, 6, 2.24583217e+02, 1.16241802e-06],
         [9.93365275e-01, 4, 6, 2.24583217e+02, 1.16241802e-06],
         [1.49722144e+02, 4, 6, 2.24583217e+02, 1.16241802e-06],
         [1.49722144e+02, 4, 6, 2.24583217e+02, 1.16241802e-06]]
    assert_array_almost_equal(r['Intercept'].values, a, decimal=6)
    a = [[0.09711919, 8., 12.,        3.31325352, 0.03054676],
         [1.32559483, 8., 14.,        3.43975859, 0.02101755],
         [4.94409805, 8., 6.63157895, 3.53952474, 0.06115465],
         [3.79813274, 4., 7.,         6.64673229, 0.01558302]]
    assert_array_almost_equal(r['Drug'].values, a, decimal=6)
    a = [[1.15958333e-01, 4, 6, 1.14356810e+01, 5.69444657e-03],
         [8.84041667e-01, 4, 6, 1.14356810e+01, 5.69444657e-03],
         [7.62378732e+00, 4, 6, 1.14356810e+01, 5.69444657e-03],
         [7.62378732e+00, 4, 6, 1.14356810e+01, 5.69444657e-03]]
    assert_array_almost_equal(r['Depleted'].values, a, decimal=6)
    a = [[0.15234366,  8., 12.,        2.34307678, 0.08894239],
         [1.13013353,  8., 14.,        2.27360606, 0.08553213],
         [3.70989596,  8., 6.63157895, 2.65594824, 0.11370285],
         [3.1145597,   4., 7.,         5.45047947, 0.02582767]]
    assert_array_almost_equal(r['Drug:Depleted'].values, a, decimal=6)


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



