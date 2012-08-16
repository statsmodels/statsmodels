import numpy as np
from numpy.testing import dec

import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot, qqline, ProbPlot
from scipy import stats


try:
    import matplotlib.pyplot as plt
    import matplotlib
    if matplotlib.__version__ < '1':
        raise
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

    plt.close('all')

@dec.skipif(not have_matplotlib)
def test_ProbPlot():
    #just test that it runs
    data = sm.datasets.longley.load()
    data.exog = sm.add_constant(data.exog)
    mod_fit = sm.OLS(data.endog, data.exog).fit()
    res = sm.ProbPlot(mod_fit.resid, stats.t, distargs=(4,))
    fig1 = res.qqplot()
    fig2 = res.ppplot()
    fig3 = res.probplot()

    plt.close('all')

@dec.skipif(not have_matplotlib)
def test_qqline():
    #just test that it runs
    data = sm.datasets.longley.load()
    data.exog = sm.add_constant(data.exog)
    mod_fit = sm.OLS(data.endog, data.exog).fit()
    res = mod_fit.resid
    for line in ['r', 'q', '45', 's']
        fig = sm.qqplot(res, line=line)

    plt.close('all')
