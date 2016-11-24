import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from numpy.testing import assert_almost_equal

# Example data
# https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/
#     viewer.htm#statug_introreg_sect012.htm
X = pd.DataFrame([[0, 0, 0, 2.068, 2.070, 1.580],
                  [0, 0, 0, 2.068, 2.074, 1.602],
                  [0, 0, 0, 2.090, 2.090, 1.613],
                  [0, 0, 0, 2.097, 2.093, 1.613],
                  [0, 0, 0, 2.117, 2.125, 1.663],
                  [0, 0, 0, 2.140, 2.146, 1.681],
                  [1, 0, 0, 2.045, 2.054, 1.580],
                  [1, 0, 0, 2.076, 2.088, 1.602],
                  [1, 0, 0, 2.090, 2.093, 1.643],
                  [1, 0, 0, 2.111, 2.114, 1.643],
                  [0, 1, 1, 2.093, 2.098, 1.653],
                  [0, 1, 1, 2.100, 2.106, 1.623],
                  [0, 1, 1, 2.104, 2.101, 1.653]])

def test_manova_sas_example():
    m = MANOVA(X=X.iloc[:, [0, 1]], Y=X.iloc[:, [3, 4, 5]],
               L=np.array([[0, 1, 0], [0, 0, 1]]))
    assert_almost_equal(m.results_.loc["Wilks’ lambda", 'Pr > F'],
                        0.6032, decimal=4)
    assert_almost_equal(m.results_.loc["Pillai’s trace", 'Pr > F'],
                        0.5397, decimal=4)
    assert_almost_equal(m.results_.loc["Hotelling-Lawley trace", 'Pr > F'],
                        0.6272, decimal=4)
    assert_almost_equal(m.results_.loc["Roy’s greatest root", 'Pr > F'],
                        0.4109, decimal=4)


