import numpy as np

from statsmodels.tsa.api import VAR
from statsmodels.api import datasets as ds

mdata = ds.macrodata.load().data[['realgdp', 'realcons', 'realinv']]
names = mdata.dtype.names
data = mdata.view((float,3))
data = np.diff(np.log(data), axis=0)

model = VAR(data, names=names)
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
