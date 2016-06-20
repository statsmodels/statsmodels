from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
import numpy as np
from distributed_estimation import *

p = 20
N = 4000
m = 4

def exogGen(exog, partitions):
    """partitions exog data"""

    n_exog = exog.shape[0]
    n_part = np.ceil(n_exog / partitions)

    ii = 0
    while ii < n_exog:
        jj = int(min(ii + n_part, n_exog))
        yield exog[ii:jj, :]
        ii += int(n_part)

def endogGen(endog, partitions):
    """partitions endog data"""

    n_endog = endog.shape[0]
    n_part = np.ceil(n_endog / partitions)

    ii = 0
    while ii < n_endog:
        jj = int(min(ii + n_part, n_endog))
        yield endog[ii:jj]
        ii += int(n_part)

# handle OLS case
beta = np.random.normal(size=p)
beta = beta * np.random.randint(0, 2, p)
X = np.random.normal(size=(N, p))
y = X.dot(beta) + np.random.normal(size=N)

fitOLS = distributed_estimation(endogGen(y, m), exogGen(X, m), m, model_class=OLS, fit_kwds={"alpha": 0.5})

# handle GLM (logistic regression) case
prob = 1 / (1 + np.exp(-X.dot(beta) + np.random.normal(size=N)))
y = 1. * (prob > 0.5)

fitGLM = distributed_estimation(endogGen(y, m), exogGen(X, m), m, model_class=GLM, init_kwds={"family": Binomial()}, fit_kwds={"alpha": 0.01})

olsd = np.sum(np.abs(fitOLS[0] - beta))
olsn = np.sum(np.abs(fitOLS[1] - beta))
print olsd < olsn

glmd = np.sum(np.abs(fitGLM[0] - beta))
glmn = np.sum(np.abs(fitGLM[1] - beta))
print glmd < glmn
