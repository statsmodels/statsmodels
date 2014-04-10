"""Example: statsmodels.discretemod
"""

from __future__ import print_function
from statsmodels.compat.python import lrange
import numpy as np
import statsmodels.api as sm

anes_data = sm.datasets.anes96.load()
anes_exog = anes_data.exog
anes_exog = sm.add_constant(anes_exog, prepend=False)
mlogit_mod = sm.MNLogit(anes_data.endog, anes_exog)
mlogit_res = mlogit_mod.fit()

# The default method for the fit is Newton-Raphson
# However, you can use other solvers
mlogit_res = mlogit_mod.fit(method='bfgs', maxiter=100)
# The below needs a lot of iterations to get it right?
#TODO: Add a technical note on algorithms
#mlogit_res = mlogit_mod.fit(method='ncg') # this takes forever


from statsmodels.iolib.summary import (
                        summary_params_2d, summary_params_2dflat)

exog_names = [anes_data.exog_name[i] for i in [0, 2]+lrange(5,8)] + ['const']
endog_names = [anes_data.endog_name+'_%d' % i for i in np.unique(mlogit_res.model.endog)[1:]]
print('\n\nMultinomial')
print(summary_params_2d(mlogit_res, extras=['bse','tvalues'],
                         endog_names=endog_names, exog_names=exog_names))
tables, table_all = summary_params_2dflat(mlogit_res,
                                          endog_names=endog_names,
                                          exog_names=exog_names,
                                          keep_headers=True)
tables, table_all = summary_params_2dflat(mlogit_res,
                                          endog_names=endog_names,
                                          exog_names=exog_names,
                                          keep_headers=False)
print('\n\n')
print(table_all)
print('\n\n')
print('\n'.join((str(t) for t in tables)))

from statsmodels.iolib.summary import table_extend
at = table_extend(tables)
print(at)

print('\n\n')
print(mlogit_res.summary())
print(mlogit_res.summary(yname='PID'))
#the following is supposed to raise ValueError
#mlogit_res.summary(yname=['PID'])

endog_names = [anes_data.endog_name+'=%d' % i for i in np.unique(mlogit_res.model.endog)[1:]]
print(mlogit_res.summary(yname='PID', yname_list=endog_names, xname=exog_names))


''' #trying cPickle
from statsmodels.compat.python import cPickle #, copy

#copy.deepcopy(mlogit_res)  #raises exception: AttributeError: 'ResettableCache' object has no attribute '_resetdict'
mnl_res = mlogit_mod.fit(method='bfgs', maxiter=100)
mnl_res.cov_params()
#mnl_res.model.endog = None
#mnl_res.model.exog = None
cPickle.dump(mnl_res, open('mnl_res.dump', 'w'))
mnl_res_l = cPickle.load(open('mnl_res.dump', 'r'))
'''
