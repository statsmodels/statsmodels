# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from numpy.testing import assert_almost_equal
from numpy.testing import assert_raises

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
    r = mod.mv_test()
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Value'],
                        0.60143661, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Value'],
                        0.44702843, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace", 'Value'],
                        0.58210348, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Value'],
                        0.35530890, decimal=8)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'F Value'],
                        0.77, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'F Value'],
                        0.86, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace", 'F Value'],
                        0.75, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'F Value'],
                        1.07, decimal=2)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace", 'Num DF'],
                        6, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Num DF'],
                        3, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Den DF'],
                        16, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Den DF'],
                        18, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace", 'Den DF'],
                        9.0909, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Den DF'],
                        9, decimal=3)
    assert_almost_equal(r['Loc']['stat'].loc["Wilks' lambda", 'Pr > F'],
                        0.6032, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Pillai's trace", 'Pr > F'],
                        0.5397, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Hotelling-Lawley trace", 'Pr > F'],
                        0.6272, decimal=4)
    assert_almost_equal(r['Loc']['stat'].loc["Roy's greatest root", 'Pr > F'],
                        0.4109, decimal=4)


def test_manova_test_input_validation():
    mod = MANOVA.from_formula('Basal + Occ + Max ~ Loc', data=X)
    hypothesis = [('test', np.array([[1, 1, 1]]), None)]
    mod.mv_test(hypothesis)
    hypothesis = [('test', np.array([[1, 1]]), None)]
    assert_raises(ValueError, mod.mv_test, hypothesis)
    # FIXME: don't docstring-out test
    """
    assert_raises_regex(ValueError,
                        ('Contrast matrix L should have the same number of '
                         'columns as exog! 2 != 3'),
                        mod.mv_test, hypothesis)
    """
    hypothesis = [('test', np.array([[1, 1, 1]]), np.array([[1], [1], [1]]))]
    mod.mv_test(hypothesis)
    hypothesis = [('test', np.array([[1, 1, 1]]), np.array([[1], [1]]))]
    assert_raises(ValueError, mod.mv_test, hypothesis)
    # FIXME: don't docstring-out test
    """
    assert_raises_regex(ValueError,
                        ('Transform matrix M should have the same number of '
                         'rows as the number of columns of endog! 2 != 3'),
                        mod.mv_test, hypothesis)
    """


def test_endog_1D_array():
    assert_raises(ValueError, MANOVA.from_formula, 'Basal ~ Loc', X)
