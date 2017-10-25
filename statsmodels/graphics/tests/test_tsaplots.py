from statsmodels.compat.python import lmap, BytesIO

from distutils.version import LooseVersion

import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_
from statsmodels.compat.testing import skipif

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import (plot_acf, plot_pacf, month_plot,
                                           quarter_plot, seasonal_plot)
import statsmodels.tsa.arima_process as tsp


try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except:
    have_matplotlib = False

pandas_lt_0_19_2 = LooseVersion(pd.__version__) < '0.19.1'


@skipif(not have_matplotlib, reason='matplotlib not available')
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


@skipif(not have_matplotlib, reason='matplotlib not available')
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


@skipif(not have_matplotlib, reason='matplotlib not available')
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


@skipif(not have_matplotlib, reason='matplotlib not available')
def test_plot_pacf_kwargs():
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1., -0.9]
    ma = np.r_[1., 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    pacf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)

    buff = BytesIO()
    plot_pacf(pacf, ax=ax)
    fig.savefig(buff, format='rgba')
    plt.close(fig)

    buff_linestyle = BytesIO()
    fig_linestyle = plt.figure()
    ax = fig_linestyle.add_subplot(111)
    plot_pacf(pacf, ax=ax, ls='-')
    fig_linestyle.savefig(buff_linestyle, format='rgba')
    plt.close(fig_linestyle)

    buff_with_vlines = BytesIO()
    fig_with_vlines = plt.figure()
    ax = fig_with_vlines.add_subplot(111)
    vlines_kwargs = {'linestyles': 'dashdot'}
    plot_pacf(pacf, ax=ax, vlines_kwargs=vlines_kwargs)
    fig_with_vlines.savefig(buff_with_vlines, format='rgba')
    plt.close(fig_with_vlines)

    buff.seek(0)
    buff_linestyle.seek(0)
    buff_with_vlines.seek(0)
    plain = buff.read()
    linestyle = buff_linestyle.read()
    with_vlines = buff_with_vlines.read()

    assert_(plain != linestyle)
    assert_(with_vlines != plain)
    assert_(linestyle != with_vlines)


@skipif(not have_matplotlib, reason='matplotlib not available')
def test_plot_acf_kwargs():
    # Just test that it runs.
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ar = np.r_[1., -0.9]
    ma = np.r_[1., 0.9]
    armaprocess = tsp.ArmaProcess(ar, ma)
    rs = np.random.RandomState(1234)
    acf = armaprocess.generate_sample(100, distrvs=rs.standard_normal)

    buff = BytesIO()
    plot_acf(acf, ax=ax)
    fig.savefig(buff, format='rgba')
    plt.close(fig)

    buff_with_vlines = BytesIO()
    fig_with_vlines = plt.figure()
    ax = fig_with_vlines.add_subplot(111)
    vlines_kwargs = {'linestyles': 'dashdot'}
    plot_acf(acf, ax=ax, vlines_kwargs=vlines_kwargs)
    fig_with_vlines.savefig(buff_with_vlines, format='rgba')
    plt.close(fig_with_vlines)

    buff.seek(0)
    buff_with_vlines.seek(0)
    plain = buff.read()
    with_vlines = buff_with_vlines.read()

    assert_(with_vlines != plain)


@skipif(not have_matplotlib, reason='matplotlib not available')
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


@skipif(not have_matplotlib or pandas_lt_0_19_2,
        reason='matplotlib not available or pandas too old')
def test_plot_month():
    dta = sm.datasets.elnino.load_pandas().data
    dta['YEAR'] = dta.YEAR.astype(int).apply(str)
    dta = dta.set_index('YEAR').T.unstack()
    dates = pd.to_datetime(['-'.join([x[1], x[0]]) for x in dta.index.values])

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


@skipif(not have_matplotlib or pandas_lt_0_19_2,
        reason='matplotlib not available or pandas too old')
def test_plot_quarter():
    dta = sm.datasets.macrodata.load_pandas().data
    dates = lmap('Q'.join, zip(dta.year.astype(int).apply(str),
                               dta.quarter.astype(int).apply(str)))
    # test dates argument
    quarter_plot(dta.unemp.values, dates)
    plt.close('all')

    # test with a DatetimeIndex with no freq
    dta.set_index(pd.to_datetime(dates), inplace=True)
    quarter_plot(dta.unemp)
    plt.close('all')

    # w freq
    # see pandas #6631
    dta.index = pd.DatetimeIndex(pd.to_datetime(dates), freq='QS-Oct')
    quarter_plot(dta.unemp)
    plt.close('all')

    # w PeriodIndex
    dta.index = pd.PeriodIndex(pd.to_datetime(dates), freq='Q')
    quarter_plot(dta.unemp)
    plt.close('all')

@skipif(not have_matplotlib, reason='matplotlib not available')
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
