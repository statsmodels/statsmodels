# -*- coding: utf-8 -*-
"""Examples using Pandas

"""


from __future__ import print_function
from statsmodels.compat.pandas import frequencies
from statsmodels.compat.python import zip
from datetime import datetime

import numpy as np

from pandas import DataFrame, Series

import statsmodels.api as sm
import statsmodels.tsa.api as tsa


data = sm.datasets.stackloss.load()
X = DataFrame(data.exog, columns=data.exog_name)
X['intercept'] = 1.
Y = Series(data.endog)

#Example: OLS
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())

print(results.params)
print(results.cov_params())

infl = results.get_influence()
print(infl.summary_table())

#raise

#Example RLM
huber_t = sm.RLM(Y, X, M=sm.robust.norms.HuberT())
hub_results = huber_t.fit()
print(hub_results.params)
print(hub_results.bcov_scaled)
print(hub_results.summary())


import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

def plot_acf_multiple(ys, lags=20):
    """

    """
    from statsmodels.tsa.stattools import acf
    # hack
    old_size = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 8

    plt.figure(figsize=(10, 10))
    xs = np.arange(lags + 1)

    acorr = np.apply_along_axis(lambda x: acf(x, nlags=lags), 0, ys)

    k = acorr.shape[1]
    for i in range(k):
        ax = plt.subplot(k, 1, i + 1)
        ax.vlines(xs, [0], acorr[:, i])

        ax.axhline(0, color='k')
        ax.set_ylim([-1, 1])

        # hack?
        ax.set_xlim([-1, xs[-1] + 1])

    mpl.rcParams['font.size'] = old_size

#Example TSA descriptive

data = sm.datasets.macrodata.load()
mdata = data.data
df = DataFrame.from_records(mdata)
quarter_end = frequencies.BQuarterEnd()
df.index = [quarter_end.rollforward(datetime(int(y), int(q) * 3, 1))
for y, q in zip(df.pop('year'), df.pop('quarter'))]
logged = np.log(df.ix[:, ['m1', 'realgdp', 'cpi']])
logged.plot(subplots=True)

log_difference = logged.diff().dropna()
plot_acf_multiple(log_difference.values)

#Example TSA VAR

model = tsa.VAR(log_difference, freq='D')
print(model.select_order())

res = model.fit(2)
print(res.summary())
print(res.is_stable())

irf = res.irf(20)
irf.plot()

fevd = res.fevd()
fevd.plot()

#print res.test_whiteness()
print(res.test_causality('m1', 'realgdp'))
#print res.test_normality() # exception
'''
Traceback (most recent call last):
  File "E:\Josef\eclipsegworkspace\statsmodels-git\statsmodels-josef\scikits\statsmodels\examples\ex_pandas.py", line 100, in <module>
    print res.test_normality()
  File "e:\josef\eclipsegworkspace\statsmodels-git\statsmodels-all\scikits\statsmodels\tsa\vector_ar\var_model.py", line 1456, in test_normality
    summ = output.normality_summary(results)
  File "e:\josef\eclipsegworkspace\statsmodels-git\statsmodels-all\scikits\statsmodels\tsa\vector_ar\output.py", line 182, in normality_summary
    return hypothesis_test_table(results, title, null_hyp)
  File "e:\josef\eclipsegworkspace\statsmodels-git\statsmodels-all\scikits\statsmodels\tsa\vector_ar\output.py", line 190, in hypothesis_test_table
    results['crit_value'],
KeyError: 'crit_value'
'''


#Example TSA ARMA

import numpy as np
import statsmodels.api as sm


# Generate some data from an ARMA process
from statsmodels.tsa.arima_process import arma_generate_sample
arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])
# The conventions of the arma_generate function require that we specify a
# 1 for the zero-lag of the AR and MA parameters and that the AR parameters
# be negated.
arparams = np.r_[1, -arparams]
maparam = np.r_[1, maparams]
nobs = 250
y = arma_generate_sample(arparams, maparams, nobs)
plt.figure()
plt.plot(y)

#Now, optionally, we can add some dates information. For this example,
# we'll use a pandas time series.
dates = sm.tsa.datetools.dates_from_range('1980m1', length=nobs)
y = Series(y, index=dates)
arma_mod = sm.tsa.ARMA(y, freq='M')
#arma_res = arma_mod.fit(order=(2,2), trend='nc', disp=-1) #fails
#old pandas 0.4.0: AttributeError: 'TimeSeries' object has no attribute 'name'
#arma_res.params

plt.show()
