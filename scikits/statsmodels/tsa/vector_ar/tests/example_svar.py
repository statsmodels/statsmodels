import numpy as np
import scikits.statsmodels.api as sm
from scikits.statsmodels.tsa.api import VAR, SVAR
import matplotlib.pyplot as plt
import scikits.statsmodels.tsa.vector_ar.util as util
import pandas as px

mdatagen = sm.datasets.macrodata.load().data
mdata = mdatagen[['realgdp','realcons','realinv']]
names = mdata.dtype.names
start = px.datetime(1959, 3, 31)
end = px.datetime(2009, 9, 30)
qtr = px.DateRange(start, end, offset=px.datetools.BQuarterEnd())
data = px.DataFrame(mdata, index=qtr)
data = (np.log(data)).diff().dropna()

#define structural inputs
A = np.asarray([[1, 0, 0],['E', 1, 0],['E', 'E', 1]])
B = np.asarray([['E', 0, 0], [0, 'E', 0], [0, 0, 'E']])
A_guess = np.asarray([0.5, 0.25, -0.38])
B_guess = np.asarray([0.5, 0.1, 0.05])
mymodel = SVAR(data, svar_type='AB', A=A, B=B, freq='Q')
res = mymodel.fit(maxlags=3, maxiter=10000, maxfun=10000, solver='bfgs')
res.irf(periods=30).plot(impulse='realgdp', plot_stderr=True, stderr_type='mc', repl=100)
