
import numpy as np
import scikits.statsmodels.api as sm
from scikits.statsmodels.tsa.api import VAR

# some example data
mdata = sm.datasets.macrodata.load().data
mdata = mdata[['realgdp','realcons','realinv']]
names = mdata.dtype.names
data = mdata.view((float,3))
data = np.diff(np.log(data), axis=0)

model = VAR(data, names=names)
res = model.fit(2)

nobs_all = data.shape[0]

#in-sample 1-step ahead forecasts
fc_in = np.array([np.squeeze(res.forecast(model.y[t-20:t], 1))
                  for t in range(nobs_all-6,nobs_all)])

print fc_in - res.fittedvalues[-6:]

#out-of-sample 1-step ahead forecasts
fc_out = np.array([np.squeeze(VAR(data[:t]).fit(2).forecast(data[t-20:t], 1))
                   for t in range(nobs_all-6,nobs_all)])

print fc_out - data[nobs_all-6:nobs_all]
