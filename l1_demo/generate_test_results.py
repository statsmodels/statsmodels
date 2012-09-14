import statsmodels.api as sm
import scipy as sp
import numpy as np
import pdb  # pdb.set_trace()

anes_data = sm.datasets.anes96.load()
anes_exog = anes_data.exog
anes_exog = sm.add_constant(anes_exog, prepend=False)
mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)

alpha = 10 * np.ones((mlogit_mod.J - 1, mlogit_mod.K))
alpha[-1,:] = 0
mlogit_l1_res = mlogit_mod.fit_regularized(method='l1', alpha=alpha, trim_mode='auto', auto_trim_tol=0.02, acc=1e-10)
print mlogit_l1_res.summary()
