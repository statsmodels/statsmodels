import numpy as np
from numpy.testing import dec

import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot


try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except:
    have_matplotlib = False


@dec.skipif(not have_matplotlib)
def test_qqplot():
    #just test that it runs
    data = sm.datasets.longley.load()
    data.exog = sm.add_constant(data.exog)
    mod_fit = sm.OLS(data.endog, data.exog).fit()
    res = mod_fit.resid
    fig = sm.qqplot(res)

    plt.close(fig)
