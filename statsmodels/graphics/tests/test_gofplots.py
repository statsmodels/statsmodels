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
    fig = sm.qqplot(res, line='r')

    plt.close('all')

@dec.skipif(not have_matplotlib)
def test_ProbPlot():
    #just test that it runs
    data = sm.datasets.longley.load()
    data.exog = sm.add_constant(data.exog)
    mod_fit = sm.OLS(data.endog, data.exog).fit()
    res = sm.ProbPlot(mod_fit.resid, stats.t, distargs=(4,))

    # basic tests modeled after example in docstring
    fig1 = res.qqplot(line='r')
    fig2 = res.ppplot(line='r')
    fig3 = res.probplot(line='r')

    plt.close('all')

@dec.skipif(not have_matplotlib)
def test_ProbPlot_comparison():
    # two fake samples for comparison
    x = np.random.normal(loc=8.25, scale=3.25, size=37)
    y = np.random.normal(loc=8.25, scale=3.25, size=37)
    pp_x = sm.ProbPlot(x)
    pp_y = sm.ProbPlot(y)

    # test `other` kwarg with `ProbPlot` instance
    fig4 = pp_x.qqplot(other=pp_y)
    fig5 = pp_x.ppplot(other=pp_y)

    plt.close('all')

@dec.skipif(not have_matplotlib)
def test_ProbPlot_comparison_arrays():
    # two fake samples for comparison
    x = np.random.normal(loc=8.25, scale=3.25, size=37)
    y = np.random.normal(loc=8.25, scale=3.25, size=37)
    pp_x = sm.ProbPlot(x)
    pp_y = sm.ProbPlot(y)

    # test `other` kwarg with array
    fig6 = pp_x.qqplot(other=y)
    fig7 = pp_x.ppplot(other=y)
    plt.close('all')

@dec.skipif(not have_matplotlib)
def test_qqplot_2samples():
    #just test that it runs
    x = np.random.normal(loc=8.25, scale=3.25, size=37)
    y = np.random.normal(loc=8.25, scale=3.25, size=37)

    pp_x = sm.ProbPlot(x)
    pp_y = sm.ProbPlot(y)

    # also tests all values for line
    for line in ['r', 'q', '45', 's']:
        # test with `ProbPlot` instances
        fig2 = sm.qqplot_2samples(pp_x, pp_y, line=line)

    plt.close('all')

@dec.skipif(not have_matplotlib)
def test_qqplot_2samples_arrays():
    #just test that it runs
    x = np.random.normal(loc=8.25, scale=3.25, size=37)
    y = np.random.normal(loc=8.25, scale=3.25, size=37)

    pp_x = sm.ProbPlot(x)
    pp_y = sm.ProbPlot(y)

    # also tests all values for line
    for line in ['r', 'q', '45', 's']:
        # test with arrays
        fig1 = sm.qqplot_2samples(x, y, line=line)

    plt.close('all')
