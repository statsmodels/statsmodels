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




class MultivariateGamPenalty(Penalty):
    
    def __init__(self, wts=1, alpha=1, cov_der2=None, der2=None):
        '''
        - cov_der2 is a list of squared matrix of shape (size_base, size_base)
        - der2 is a list of matrix of shape (n_samples, size_base) 
        - 
        '''
        
        assert(len(cov_der2) == len(der2))
        
        # the total number of columns in der2 i.e. the len of the params vector
        self.n_columns = np.sum(d2.shape[1] for d2 in der2)

        # the number of variables in the GAM model
        self.n_variables = len(cov_der2) 

        # if wts and alpha are not a list then each function has the same penalty
        # TODO: Review this
        self.alpha = np.array([alpha] * self.n_variables)
        self.wts = np.array([wts] * self.n_variables)

        
        n_samples = der2[0].shape[0] 
        self.mask = [np.array([False]*self.n_columns) 
                     for i in range(self.n_variables)]
        param_count = 0
        for i, d2 in enumerate(der2):
            n, dim_base = d2.shape
            #check that all the basis have the same number of samples 
            assert(n_samples == n) 
            self.mask[i][param_count: param_count + dim_base] = True
            param_count += dim_base
            
        self.gp = []
        for i in range(self.n_variables):
            gp = GamPenalty(wts=self.wts[i], alpha=self.alpha[i], 
                            cov_der2=cov_der2[i], der2=der2[i])
            self.gp.append(gp)

        return


    def func(self, params):
        
        cost = 0
        for i in range(self.n_variables):
            params_i = params[self.mask[i]]
            cost += self.gp[i].func(params_i)
        
        return  cost


    def grad(self, params):
        grad = []
        for i in range(self.n_variables):
            params_i = params[self.mask[i]]
            grad.append(self.gp[i].grad(params_i))

        return np.concatenate(grad)

    
    def deriv2(self, params):
        deriv2 = []
        for i in range(self.n_variables):
            params_i = params[self.mask[i]]
            deriv2.append(self.gp[i].grad(params_i))
        return np.concatenate(deriv2)




df = 10
degree = 5
alpha = 0
sigmoid = np.vectorize(lambda x: 1.0/(1.0 + np.exp(-x)))

n = 100
x1 = np.linspace(-1, 1, n)
x2 = np.linspace(-1, 1, n)
poly = x1*x1 + x2*x2*x2 + np.random.normal(0, 0.01, n)
y = sigmoid(poly)
y[y > 0.5] = 1
y[y <= 0.5] = 0

basis1, der_basis1, der2_basis1 = make_poly_basis(x1, degree)
basis2, der_basis2, der2_basis2 = make_poly_basis(x2, degree)



basis = np.hstack([basis1, basis2])
der_basis1 = [der_basis1, der_basis2]
der2_basis = [der2_basis1, der2_basis2]
cov_der2 = [np.dot(der2_basis1.T, der2_basis1),
            np.dot(der2_basis2.T, der2_basis2)]




gp = MultivariateGamPenalty(wts=1, alpha=alpha, cov_der2=cov_der2, 
                            der2=der2_basis)    
g = LogitGam(y, basis, penal=gp)
res_g = g.fit()

param1 = res_g.params[gp.mask[0]]
param2 = res_g.params[gp.mask[1]]


#param1 = np.array([0, 0, 1, 0, 0])
#param2 = np.array([0, 0, 0, 1, 0, 0])

print(param1, param2)
plt.subplot(2, 1, 1)
plt.plot(x1, np.dot(basis1, param1), label='x1')
plt.plot(x1, x1 * x1, '.')
plt.subplot(2, 1, 2)
plt.plot(x2, np.dot(basis2, param2), label='x2')
plt.plot(x2, x2*x2*x2, '.')
plt.show()


