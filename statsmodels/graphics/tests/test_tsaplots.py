from statsmodels.compat.python import lmap, lzip, map
import numpy as np
import pandas as pd
from numpy.testing import dec

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, month_plot, quarter_plot
import statsmodels.tsa.arima_process as tsp


try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except:
    have_matplotlib = False


@dec.skipif(not have_matplotlib)
def test_plot_acf():
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1., -0.9]
    ma = np.r_[1.,  0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    acf = armaprocess.acf(20)[:20]
    plot_acf(acf, ax=ax)

    plt.close(fig)


@dec.skipif(not have_matplotlib)
def test_plot_month():
    dta = sm.datasets.elnino.load_pandas().data
    dta['YEAR'] = dta.YEAR.astype(int).apply(str)
    dta = dta.set_index('YEAR').T.unstack()
    dates = lmap(lambda x : pd.datetools.parse('1 '+' '.join(x)),
                                            dta.index.values)

    # test dates argument
    fig = month_plot(dta.values, dates=dates, ylabel='el nino')
    plt.close(fig)

    # test with a TimeSeries DatetimeIndex with no freq
    dta.index = pd.DatetimeIndex(dates)
    fig = month_plot(dta)
    plt.close(fig)

    # w freq
    dta.index = pd.DatetimeIndex(dates, freq='M')
    fig = month_plot(dta)
    plt.close(fig)

    # test with a TimeSeries PeriodIndex
    dta.index = pd.PeriodIndex(dates, freq='M')
    fig = month_plot(dta)
    plt.close(fig)

@dec.skipif(not have_matplotlib)
def test_plot_quarter():
    dta = sm.datasets.macrodata.load_pandas().data
    dates = lmap('Q'.join, zip(dta.year.astype(int).apply(str),
                              dta.quarter.astype(int).apply(str)))
    # test dates argument
    quarter_plot(dta.unemp.values, dates)

    # test with a DatetimeIndex with no freq
    parser = pd.datetools.parse_time_string
    dta.set_index(pd.DatetimeIndex((x[0] for x in map(parser, dates))),
                  inplace=True)
    quarter_plot(dta.unemp)

    # w freq
    # see pandas #6631
    dta.index = pd.DatetimeIndex((x[0] for x in map(parser, dates)),
                                   freq='QS-Oct')
    quarter_plot(dta.unemp)

    # w PeriodIndex
    dta.index = pd.PeriodIndex((x[0] for x in map(parser, dates)),
                                   freq='Q')
    quarter_plot(dta.unemp)
