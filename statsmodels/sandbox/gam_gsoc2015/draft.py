''' This file contains draft of code. Do not look at it '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from smooth_basis import make_poly_basis, make_bsplines_basis
from gam import PenalizedMixin, GamPenalty
from statsmodels.api import GLM
from statsmodels.discrete.discrete_model import Logit


class LogitGam(PenalizedMixin, Logit):
    pass


class GLMGam(PenalizedMixin, GLM):
    pass
  

def gam_gcv_path(X, y, der2, cov_der2, alphas, gam):
    
    n_samples, n_features = X.shape
    
    
    params0 = np.random.normal(0, 1, n_features)
    gcv = np.array([0]*len(alphas))
    for i, alpha in enumerate(alphas):
            gp = GamPenalty(wts=1, alpha=alpha, cov_der2=cov_der2, 
                            der2=der2)
    
            g = gam(y, basis, penal=gp)
            g_fitted = g.fit()

            # TODO: S should be estimated somehow. 
            S = np.zeros(shape=(n_samples, n_samples))
            tr = S.trace()
            gcv[i] = 1 / (n_samples + tr**2 / n_samples - 2 * tr)
            gcv[i] = gcv[i] * np.linalg.norm(res_g.predict(X))**2
            

    return gcv





n = 100

# make the data
x = np.array([1, 2, 3, 4, 5])   
y = np.array([1, 0, 0, 0, 1])

## make the splines basis ## 
df = 8
degree = 4
x = x - x.mean()
basis, der_basis, der2_basis = make_bsplines_basis(x, df, degree)
cov_der2 = np.dot(der2_basis.T, der2_basis)


## train the gam logit model ##
alpha = 5

params0 = np.random.normal(0, 1, df)
gp = GamPenalty(wts=1, alpha=alpha, cov_der2=cov_der2, der2=der2_basis)    
g = LogitGam(y, basis, penal=gp)
res_g = g.fit()
plt.plot(x, np.dot(basis, res_g.params))
plt.title('alpha=' + str(alpha))
plt.show()

