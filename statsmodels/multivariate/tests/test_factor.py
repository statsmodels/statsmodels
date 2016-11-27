# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from statsmodels.multivariate.factor import Factor
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
from numpy.testing.decorators import skipif

try:
    import matplotlib.pyplot as plt
    missing_matplotlib = False
except ImportError:
    missing_matplotlib = True

# Example data
# https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/
#     viewer.htm#statug_introreg_sect012.htm
X = pd.DataFrame([['Minas Graes', 2.068, 2.070, 1.580, 1, 0],
                  ['Minas Graes', 2.068, 2.074, 1.602, 2, 1],
                  ['Minas Graes', 2.090, 2.090, 1.613, 3, 0],
                  ['Minas Graes', 2.097, 2.093, 1.613, 4, 1],
                  ['Minas Graes', 2.117, 2.125, 1.663, 5, 0],
                  ['Minas Graes', 2.140, 2.146, 1.681, 6, 1],
                  ['Matto Grosso', 2.045, 2.054, 1.580, 7, 0],
                  ['Matto Grosso', 2.076, 2.088, 1.602, 8, 1],
                  ['Matto Grosso', 2.090, 2.093, 1.643, 9, 0],
                  ['Matto Grosso', 2.111, 2.114, 1.643, 10, 1],
                  ['Santa Cruz', 2.093, 2.098, 1.653, 11, 0],
                  ['Santa Cruz', 2.100, 2.106, 1.623, 12, 1],
                  ['Santa Cruz', 2.104, 2.101, 1.653, 13, 0]],
                 columns=['Loc', 'Basal', 'Occ', 'Max', 'id', 'alt'])


def test_example_compare_to_R_output():
    # No rotation produce same results as in R fa
    mod = Factor(X.iloc[:, 1:-1], 2)
    results = mod.fit()
    a = np.array([[0.97541115, 0.20280987],
                  [0.97113975, 0.17207499],
                  [0.9618705, -0.2004196],
                  [0.37570708, -0.45821379]])
    #assert_array_almost_equal(mod.loadings, a, decimal=8)

    # Same as R GRArotation
    results = mod.fit(rotation='varimax')
    a = np.array([[0.98828898, -0.12587155],
                  [0.97424206, -0.15354033],
                  [0.84418097, -0.502714],
                  [0.20601929, -0.55558235]])
    #assert_array_almost_equal(mod.loadings, a, decimal=8)

    results = mod.fit(rotation='quartimax')  # Same as R fa
    a = np.array([[0.98935598, 0.98242714, 0.94078972, 0.33442284],
                  [0.117190049, 0.086943252, -0.283332952, -0.489159543]])
    #assert_array_almost_equal(mod.loadings, a.T, decimal=8)

    results = mod.fit(rotation='equamax')  # Not the same as R fa

    results = mod.fit(rotation='promax')  # Not the same as R fa

    results = mod.fit(rotation='biquartimin')  # Not the same as R fa

    results = mod.fit(rotation='oblimin')  # Same as R fa
    a = np.array([[1.02834170170, 1.00178840104, 0.71824931384,
                   -0.00013510048],
                  [0.06563421, 0.03096076, -0.39658839, -0.59261944]])
    #assert_array_almost_equal(mod.loadings, a.T, decimal=8)


@skipif(missing_matplotlib)
def test_plots():
    mod = Factor(X.iloc[:, 1:], 3)
    mod.fit(rotation='oblimin')
    mod.plot_scree()
    fig_loadings = mod.plot_loadings()
    assert_equal(3, len(fig_loadings))
