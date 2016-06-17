from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Binomial
import numpy as np
from distributed_estimation import _generator, _gen_dist_params, _gen_debiased_params 

p = 20
N = 1000
m = 4

beta = np.random.normal(size=p)
beta = beta * np.random.randint(0, 2, p)
X = np.random.normal(size=(N, p))
y = X.dot(beta) + np.random.normal(size=N)
prob = 1 / (1 + np.exp(-y))
y = 1. * (prob >= 0.5)
mod = GLM(y, X, family=Binomial())
params = mod.fit().params
#mod_gen = _generator(mod, m)
#params = _gen_dist_params(mod_gen, m, p, {"alpha": 0.1, "L1_wt": 1}, {"scale": 1}, {"scale": 1})
#beta_tilde = _gen_debiased_params(params[0], params[1], params[2], params[3], m, p, 0)
#hessian = mod.hessian(np.array([1] * 20))
