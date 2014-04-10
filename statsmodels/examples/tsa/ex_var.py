
from __future__ import print_function
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

# some example data
mdata = sm.datasets.macrodata.load().data
mdata = mdata[['realgdp','realcons','realinv']]
names = mdata.dtype.names
data = mdata.view((float,3))

use_growthrate = False #True #False
if use_growthrate:
    data = 100 * 4 * np.diff(np.log(data), axis=0)

model = VAR(data, names=names)
res = model.fit(4)

nobs_all = data.shape[0]

#in-sample 1-step ahead forecasts
fc_in = np.array([np.squeeze(res.forecast(model.y[t-20:t], 1))
                  for t in range(nobs_all-6,nobs_all)])

print(fc_in - res.fittedvalues[-6:])

#out-of-sample 1-step ahead forecasts
fc_out = np.array([np.squeeze(VAR(data[:t]).fit(2).forecast(data[t-20:t], 1))
                   for t in range(nobs_all-6,nobs_all)])

print(fc_out - data[nobs_all-6:nobs_all])
print(fc_out - res.fittedvalues[-6:])


#out-of-sample h-step ahead forecasts
h = 2
fc_out = np.array([VAR(data[:t]).fit(2).forecast(data[t-20:t], h)[-1]
                   for t in range(nobs_all-6-h+1,nobs_all-h+1)])

print(fc_out - data[nobs_all-6:nobs_all])  #out-of-sample forecast error
print(fc_out - res.fittedvalues[-6:])

import matplotlib.pyplot as plt
res.plot_forecast(20)
#plt.show()
