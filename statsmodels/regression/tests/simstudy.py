import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/statsmodels_fork/statsmodels")

import numpy as np
from statsmodels.regression.bayes_mixed_glm import (
    BayesMixedGLM, BinomialBayesMixedGLM)
import statsmodels.api as sm
from scipy import sparse

def gen_logit_crossed(nc, cs, s1, s2):

    a = np.kron(np.eye(nc), np.ones((cs, 1)))
    b = np.kron(np.ones((cs, 1)), np.eye(nc))
    exog_vc = np.concatenate((a, b), axis=1)

    exog_fe = np.random.normal(size=(nc*cs, 1))
    vc = np.zeros(2*nc)
    vc[:nc] = s1 * np.random.normal(size=nc)
    vc[nc:] *= s2 * np.random.normal(size=nc)
    lp = np.dot(exog_fe, np.r_[-0.5]) + np.dot(exog_vc, vc)
    pr = 1 / (1 + np.exp(-lp))
    y = 1*(np.random.uniform(size=nc*cs) < pr)
    ident = np.zeros(2*nc, dtype=np.int)
    ident[nc:] = 1

    return y, exog_fe, exog_vc, ident


mn, sd = [], []
for k in range(100):
    y, exog_fe, exog_vc, ident = gen_logit_crossed(20, 20, 0, 0)

#    glmm1 = BayesMixedGLM(y, exog_fe, exog_vc, ident,
#                          family=sm.families.Binomial(),
#                          vcp_p=3)
#    rslt1 = glmm1.fit_map()

    glmm2 = BinomialBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=3.)
    rslt2 = glmm2.fit_vb()
    mn.append(rslt2.vcp_mean)
    sd.append(rslt2.vcp_sd)

mn = np.asarray(mn)
sd = np.asarray(sd)
