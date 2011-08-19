import numpy as np
import scikits.statsmodels.api as sm
from scikits.statsmodels.tsa.api import SVAR
import matplotlib.pyplot as plt
import scikits.statsmodels.tsa.vector_ar.util as util
import numpy.linalg as npl
from scipy import optimize

np.seterr(all='warn')

mdata = sm.datasets.macrodata.load().data
mdata = mdata[['realgdp','realcons','realinv']]
names = mdata.dtype.names
data = mdata.view((float,3))
data = np.diff(np.log(data), axis=0)
#define structural inputs
A = np.asarray([[1, 0, 0],['E', 1, 0],['E', 'E', 1]])
B = np.asarray([['E', 0, 0], [0, 'E', 0], [0, 0, 'E']])
#A_guess = np.asarray([[1, -0.2, 0],[1, 0.5, 0.3],[0, 0, 1]])
#B_guess = np.asarray([[0.5, 0, 0], [0, 0.1, 0], [0, 0, 0.05]])
mymodel = SVAR(data, svar_type='AB', A=A, B=B, names=names)
res = mymodel.fit(maxlags=3, override=True)
