from statsmodels.compat.python import lmap, map
from statsmodels.compat.pandas import datetools
import numpy as np
import pandas as pd
from numpy.testing import dec, assert_equal

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import (plot_acf, plot_pacf, month_plot,
                                           quarter_plot, seasonal_plot)
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
    ma = np.r_[1., 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    acf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    plot_acf(acf, ax=ax, lags=10)
    plot_acf(acf, ax=ax)
    plot_acf(acf, ax=ax, alpha=None)

    plt.close(fig)


@dec.skipif(not have_matplotlib)
def test_plot_acf_irregular():
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1., -0.9]
    ma = np.r_[1., 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    acf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    plot_acf(acf, ax=ax, lags=np.arange(1, 11))
    plot_acf(acf, ax=ax, lags=10, zero=False)
    plot_acf(acf, ax=ax, alpha=None, zero=False)

    plt.close(fig)


@dec.skipif(not have_matplotlib)
def test_plot_pacf():
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1., -0.9]
    ma = np.r_[1., 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    pacf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    plot_pacf(pacf, ax=ax)
    plot_pacf(pacf, ax=ax, alpha=None)

    plt.close(fig)


@dec.skipif(not have_matplotlib)
def test_plot_pacf_irregular():
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1., -0.9]
    ma = np.r_[1., 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    pacf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)
    plot_pacf(pacf, ax=ax, lags=np.arange(1, 11))
    plot_pacf(pacf, ax=ax, lags=10, zero=False)
    plot_pacf(pacf, ax=ax, alpha=None, zero=False)

    plt.close(fig)

@dec.skipif(not have_matplotlib)
def test_plot_month():
    dta = sm.datasets.elnino.load_pandas().data
    dta['YEAR'] = dta.YEAR.astype(int).apply(str)
    dta = dta.set_index('YEAR').T.unstack()
    dates = lmap(lambda x: datetools.parse_time_string('1 '+' '.join(x))[0],
                 dta.index.values)

    # test dates argument
    fig = month_plot(dta.values, dates=dates, ylabel='el nino')
    plt.close(fig)

    # test with a TimeSeries DatetimeIndex with no freq
    dta.index = pd.DatetimeIndex(dates)
    fig = month_plot(dta)
    plt.close(fig)

    # w freq
    dta.index = pd.DatetimeIndex(dates, freq='MS')
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
    plt.close('all')

    # test with a DatetimeIndex with no freq
    parser = datetools.parse_time_string
    dta.set_index(pd.DatetimeIndex((x[0] for x in map(parser, dates))),
                  inplace=True)
    quarter_plot(dta.unemp)
    plt.close('all')

    # w freq
    # see pandas #6631
    dta.index = pd.DatetimeIndex((x[0] for x in map(parser, dates)),
                                   freq='QS-Oct')
    quarter_plot(dta.unemp)
    plt.close('all')

    # w PeriodIndex
    dta.index = pd.PeriodIndex((x[0] for x in map(parser, dates)),
                                   freq='Q')
    quarter_plot(dta.unemp)
    plt.close('all')

@dec.skipif(not have_matplotlib)
def test_seasonal_plot():
    rs = np.random.RandomState(1234)
    data = rs.randn(20,12)
    data += 6*np.sin(np.arange(12.0)/11*np.pi)[None,:]
    data = data.ravel()
    months = np.tile(np.arange(1,13),(20,1))
    months = months.ravel()
    df = pd.DataFrame([data,months],index=['data','months']).T
    grouped = df.groupby('months')['data']
    labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig = seasonal_plot(grouped, labels)
    ax = fig.get_axes()[0]
    output = [tl.get_text() for tl in ax.get_xticklabels()]
    assert_equal(labels, output)
    plt.close('all')
