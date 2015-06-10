''' This file contains draft of code. Do not look at it '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from smooth_basis import make_poly_basis, make_bsplines_basis
from gam import PenalizedMixin, GamPenalty, LogitGam, GLMGam, Penalty
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from smooth_basis import make_poly_basis, make_bsplines_basis
from gam import PenalizedMixin, GamPenalty
from statsmodels.api import GLM
from statsmodels.discrete.discrete_model import Logit
from patsy import dmatrix
from patsy.state import stateful_transform
from smooth_basis import BS

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










