# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from statsmodels.multivariate.glm import GLM
from numpy.testing import assert_array_almost_equal


def compare_spss_output_dogs_data(method):
    ''' Testing within-subject effect interact with 2 between-subject effect
    Compares with R car library linearHypothesis output

    Note: The test statistis Phillai, Wilks, Hotelling-Lawley
          and Roy are the same as R output but the approximate F and degree
          of freedoms can be different. This is due to the fact that this
          implementation is based on SAS formula [1]

    .. [1] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
    '''
    data = pd.DataFrame([['Morphine',      'N',  .04,  .20,  .10,  .08],
                         ['Morphine',      'N',  .02,  .06,  .02,  .02],
                         ['Morphine',      'N',  .07, 1.40,  .48,  .24],
                         ['Morphine',      'N',  .17,  .57,  .35,  .24],
                         ['Morphine',       'Y',  .10,  .09,  .13,  .14],
                         ['placebo',       'Y',  .07,  .07,  .06,  .07],
                         ['placebo',       'Y',  .05,  .07,  .06,  .07],
                         ['placebo',       'N',  .03,  .62,  .31,  .22],
                         ['placebo',       'N',  .03, 1.05,  .73,  .60],
                         ['placebo',       'N',  .07,  .83, 1.07,  .80],
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
    mod = GLM.from_formula(
        'Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted',
        data, method=method)
    r = mod.test()
    a = [[2.68607660e-02, 4, 6, 5.43435304e+01, 7.59585610e-05],
         [9.73139234e-01, 4, 6, 5.43435304e+01, 7.59585610e-05],
         [3.62290202e+01, 4, 6, 5.43435304e+01, 7.59585610e-05],
         [3.62290202e+01, 4, 6, 5.43435304e+01, 7.59585610e-05]]
    assert_array_almost_equal(r[0][-1].values, a, decimal=6)
    a = [[8.39646619e-02, 8, 1.20000000e+01, 3.67658068e+00, 2.12614444e-02],
         [1.18605382e+00, 8, 1.40000000e+01, 2.55003861e+00, 6.01270701e-02],
         [7.69391362e+00, 8, 6.63157895e+00, 5.50814270e+00, 2.07392260e-02],
         [7.25036952e+00, 4, 7.00000000e+00, 1.26881467e+01, 2.52669877e-03]]
    assert_array_almost_equal(r[1][-1].values, a, decimal=6)
    a = [[0.32048892, 4., 6., 3.18034906, 0.10002373],
         [0.67951108, 4., 6., 3.18034906, 0.10002373],
         [2.12023271, 4., 6., 3.18034906, 0.10002373],
         [2.12023271, 4., 6., 3.18034906, 0.10002373]]
    assert_array_almost_equal(r[2][-1].values, a, decimal=6)
    a = [[0.15234366, 8., 12.,        2.34307678, 0.08894239],
         [1.13013353, 8., 14.,        2.27360606, 0.08553213],
         [3.70989596, 8., 6.63157895, 2.65594824, 0.11370285],
         [3.1145597,  4., 7.,         5.45047947, 0.02582767]]
    assert_array_almost_equal(r[3][-1].values, a, decimal=6)

def test_glm_dogs_example():
    compare_spss_output_dogs_data(method='svd')
    compare_spss_output_dogs_data(method='pinv')
