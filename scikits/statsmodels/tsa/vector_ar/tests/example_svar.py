import numpy as np
import scikits.statsmodels.api as sm
from scikits.statsmodels.tsa.api import VAR, SVAR
import matplotlib.pyplot as plt

mdata = sm.datasets.macrodata.load().data
mdata = mdata[['realgdp','realcons','realinv']]
names = mdata.dtype.names
data = mdata.view((float,3))
data = np.diff(np.log(data), axis=0)

#define structural inputs
A = np.asarray([[1, 0, 0],['E', 1, 0],['E', 'E', 1]])
B = np.asarray([['E', 0, 0], [0, 'E', 0], [0, 0, 'E']])
A_guess = np.asarray([0.5, 0.25, -0.38])
B_guess = np.asarray([0.5, 0.1, 0.05])
mymodel = SVAR(data, svar_type='AB', A=A, B=B, names=names)
res = mymodel.fit(maxlags=3, maxiter=10000, maxfun=10000, solver='bfgs')
res.irf(periods=30).plot(impulse='realgdp', plot_stderr=False)
