
## Autoregressive Moving Average (ARMA): Artificial data

from __future__ import print_function
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(12345)


# Generate some data from an ARMA process:

arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])


# The conventions of the arma_generate function require that we specify a 1 for the zero-lag of the AR and MA parameters and that the AR parameters be negated.

arparams = np.r_[1, -arparams]
maparam = np.r_[1, maparams]
nobs = 250
y = arma_generate_sample(arparams, maparams, nobs)


#  Now, optionally, we can add some dates information. For this example, we'll use a pandas time series.

import pandas as pd
dates = sm.tsa.datetools.dates_from_range('1980m1', length=nobs)
y = pd.Series(y, index=dates)
arma_mod = sm.tsa.ARMA(y, order=(2,2))
arma_res = arma_mod.fit(trend='nc', disp=-1)
