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
    mod = MANOVA.from_formula('Basal + Occ + Max ~ Loc', data=X, method='qr')
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


def test_compare_spss_output_dogs_data():
    ''' Testing within-subject effect interact with 2 between-subject effect
    Compares with SPSS MANOVA output
    '''
    data = pd.DataFrame([['Morphine',      'N',  .04,  .20,  .10,  .08],
                         ['Morphine',      'N',  .02,  .06,  .02,  .02],
                         ['Morphine',      'N',  .07, 1.40,  .48,  .24],
                         ['Morphine',      'N',  .17,  .57,  .35,  .24],
                         ['Morphine',      'Y',  .10,  .09,  .13,  .14],
                         ['Morphine',      'Y',  .07,  .07,  .06,  .07],
                         ['Morphine',      'Y',  .05,  .07,  .06,  .07],
                         ['Trimethaphan',  'N',  .03,  .62,  .31,  .22],
                         ['Trimethaphan',  'N',  .03, 1.05,  .73,  .60],
                         ['Trimethaphan',  'N',  .07,  .83, 1.07,  .80],
                         ['Trimethaphan',  'N',  .09, 3.13, 2.06, 1.23],
                         ['Trimethaphan',  'Y',  .10,  .09,  .09,  .08],
                         ['Trimethaphan',  'Y',  .08,  .09,  .09,  .10],
                         ['Trimethaphan',  'Y',  .13,  .10,  .12,  .12],
                         ['Trimethaphan',  'Y',  .06,  .05,  .05,  .05]],
                        columns = ['Drug', 'Depleted', 'Histamine0', 'Histamine1',
                                   'Histamine3', 'Histamine5'])

    for i in range(2,6):
        data.iloc[:, i] = np.log(data.iloc[:, i])

    # Repeated measures with orthogonal polynomial contrasts coding
    from patsy.contrasts import Poly, Sum
    contrast = Poly([0, 1, 3, 5]).code_without_intercept([0, 1, 3, 5])
    data['p1'] = 0
    data['p2'] = 0
    data['p3'] = 0
    data[['p1', 'p2', 'p3']] = data.iloc[:, 2:6].values.dot(contrast.matrix)
    mod = MANOVA.from_formula(
        'p1 + p2 + p3 ~ Drug * Depleted',
        data, method='qr')
    r = mod.test()
    a = [[1.00382414e-01, 3, 9, 2.68857128e+01, 7.97286681e-05],
         [8.99617586e-01, 3, 9, 2.68857128e+01, 7.97286681e-05],
         [8.96190427e+00, 3, 9, 2.68857128e+01, 7.97286681e-05],
         [8.96190427e+00, 3, 9, 2.68857128e+01, 7.97286681e-05]]
    assert_array_almost_equal(r[0][1].values, a, decimal=6)
    a = [[0.32804105, 3, 9, 6.14519685, 0.01466738],
         [0.67195895, 3, 9, 6.14519685, 0.01466738],
         [2.04839895, 3, 9, 6.14519685, 0.01466738],
         [2.04839895, 3, 9, 6.14519685, 0.01466738]]
    assert_array_almost_equal(r[1][1].values, a, decimal=6)
    a = [[1.15524129e-01, 3, 9, 2.29686009e+01, 1.49013694e-04],
         [8.84475871e-01, 3, 9, 2.29686009e+01, 1.49013694e-04],
         [7.65620029e+00, 3, 9, 2.29686009e+01, 1.49013694e-04],
         [7.65620029e+00, 3, 9, 2.29686009e+01, 1.49013694e-04]]
    assert_array_almost_equal(r[2][1].values, a, decimal=6)
    a = [[1.93830104e-01, 3, 9, 1.24774720e+01, 1.47439758e-03],
         [8.06169896e-01, 3, 9, 1.24774720e+01, 1.47439758e-03],
         [4.15915732e+00, 3, 9, 1.24774720e+01, 1.47439758e-03],
         [4.15915732e+00, 3, 9, 1.24774720e+01, 1.47439758e-03]]
    assert_array_almost_equal(r[3][1].values, a, decimal=6)


def test_manova_interaction_term():
    mod = MANOVA.from_formula('Basal + Occ ~ Loc * Max', data=X, method='qr')
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



