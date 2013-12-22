import numpy as np

from statsmodels.tsa.api import VAR
from statsmodels.api import datasets as ds
from statsmodels.tsa.base.datetools import dates_from_str


import pandas
mdata = ds.macrodata.load_pandas().data

# prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype('S4')
quarterly = dates["year"] + "Q" + dates["quarter"]
quarterly = dates_from_str(quarterly)

mdata = mdata[['realgdp','realcons','realinv']]
mdata.index = pandas.DatetimeIndex(quarterly)
data = np.log(mdata).diff().dropna()

model = VAR(data)
est = model.fit(maxlags=2)

def plot_input():
    est.plot()

def plot_acorr():
    est.plot_acorr()

def plot_irf():
    est.irf().plot()

def plot_irf_cum():
    irf = est.irf()
    irf.plot_cum_effects()

def plot_forecast():
    est.plot_forecast(10)

def plot_fevd():
    est.fevd(20).plot()
