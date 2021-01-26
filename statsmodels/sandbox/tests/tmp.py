import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/fork4/statsmodels")

import numpy as np
from statsmodels.sandbox.phreg import PHreg

"""
s(theta_hat) = 0

s(theta_0) + H(theta_0)*(theta_hat-theta_0) = 0

theta_hat - theta_0 = -H_inv(theta_0) * s(theta_0)

cov(theta_hat) = H_inv(theta_0) * cov(s(theta_0)) * H_inv(theta_0)
"""

n = 1000
p = 5

exog = np.random.normal(size=(n, p))
lin_pred = exog.sum(1)
endog = -np.exp(lin_pred)*np.log(np.random.uniform(size=n))

exog_rep = np.kron(exog, np.ones((4,1)))
endog_rep = np.kron(endog, np.ones(4))

groups = np.kron(np.arange(n), np.ones(4))

mod = PHreg(endog, exog)
rslt = mod.fit()

mod_rep = PHreg(endog_rep, exog_rep)
rslt_rep = mod_rep.fit()

mod_rep_a = PHreg(endog_rep, exog_rep, groups=groups)
rslt_rep_a = mod_rep_a.fit()

