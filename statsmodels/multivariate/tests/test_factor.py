# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from statsmodels.multivariate.factor import Factor
from numpy.testing import assert_array_almost_equal, assert_raises_regex

# Example data
# https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/
#     viewer.htm#statug_introreg_sect012.htm
X = pd.DataFrame([['Minas Graes', 2.068, 2.070, 1.580, 1],
                  ['Minas Graes', 2.068, 2.074, 1.602, 2],
                  ['Minas Graes', 2.090, 2.090, 1.613, 3],
                  ['Minas Graes', 2.097, 2.093, 1.613, 4],
                  ['Minas Graes', 2.117, 2.125, 1.663, 5],
                  ['Minas Graes', 2.140, 2.146, 1.681, 6],
                  ['Matto Grosso', 2.045, 2.054, 1.580, 7],
                  ['Matto Grosso', 2.076, 2.088, 1.602, 8],
                  ['Matto Grosso', 2.090, 2.093, 1.643, 9],
                  ['Matto Grosso', 2.111, 2.114, 1.643, 10],
                  ['Santa Cruz', 2.093, 2.098, 1.653, 11],
                  ['Santa Cruz', 2.100, 2.106, 1.623, 12],
                  ['Santa Cruz', 2.104, 2.101, 1.653, 13]],
                 columns=['Loc', 'Basal', 'Occ', 'Max', 'id'])


def test_example_compare_to_R_output():
    mod = Factor(X.iloc[:, 1:], 2)
    mod.fit()
    a = np.array([0.9925, 0.9727, 0.9653, 0.3511])
    assert_array_almost_equal(mod.communality, a, decimal=4)
    a = np.array([[0.97541115, 0.20280987],
                  [0.97113975, 0.17207499],
                  [0.9618705, -0.2004196],
                  [0.37570708, -0.45821379]])
    assert_array_almost_equal(mod.loadings, a, decimal=8)


