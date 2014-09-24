# -*- coding: utf-8 -*-
"""

Created on Thu Jul 18 14:57:46 2013

Author: Josef Perktold
"""

from __future__ import print_function
import numpy as np

from statsmodels.genmod.generalized_estimating_equations import GEE, GEEMargins

from statsmodels.genmod.families import Gaussian, Binomial, Poisson
from statsmodels.genmod.cov_struct import (Exchangeable, Independence,
                                           GlobalOddsRatio, Autoregressive,
                                           Nested)

from statsmodels.genmod.tests import gee_gaussian_simulation_check as gees

da,va = gees.gen_gendat_ar0(0.6)()
ga = Gaussian()
lhs = np.array([[0., 1, 1, 0, 0],])
rhs = np.r_[0.,]

example = []
if 'constraint' in example:
    md = GEE(da.endog, da.exog, da.group, da.time, ga, va,
                     constraint=(lhs, rhs))
    mdf = md.fit()
    print(mdf.summary())


md2 = GEE(da.endog, da.exog, da.group, da.time, ga, va,
                 constraint=None)
mdf2 = md2.fit()
print('\n\n')
print(mdf2.summary())


mdf2.use_t = False
mdf2.df_resid = np.diff(mdf2.model.exog.shape)
tt2 = mdf2.t_test(np.eye(len(mdf2.params)))
# need master to get wald_test
#print mdf2.wald_test(np.eye(len(mdf2.params))[1:])

'''
>>> mdf2.predict(da.exog.mean(0))
Traceback (most recent call last):
  File "<pyshell#11>", line 1, in <module>
    mdf2.predict(da.exog.mean(0))
  File "e:\josef\eclipsegworkspace\statsmodels-git\statsmodels-all-new2_py27\statsmodels\statsmodels\base\model.py", line 963, in predict
    return self.model.predict(self.params, exog, *args, **kwargs)
  File "e:\josef\eclipsegworkspace\statsmodels-git\statsmodels-all-new2_py27\statsmodels\statsmodels\genmod\generalized_estimating_equations.py", line 621, in predict
    fitted = offset + np.dot(exog, params)
TypeError: unsupported operand type(s) for +: 'NoneType' and 'numpy.float64'
'''
mdf2.predict(da.exog.mean(0), offset=0)
# -0.10867809062890971

marg2 = GEEMargins(mdf2, ())
print(marg2.summary())


mdf_nc = md2.fit(cov_type='naive')
mdf_bc = md2.fit(cov_type='bias_reduced')

mdf_nc.use_t = False
mdf_nc.df_resid = np.diff(mdf2.model.exog.shape)
mdf_bc.use_t = False
mdf_bc.df_resid = np.diff(mdf2.model.exog.shape)

tt_nc = mdf_nc.t_test(np.eye(len(mdf2.params)))
tt_bc = mdf_bc.t_test(np.eye(len(mdf2.params)))

print('\nttest robust')
print(tt2)
print('\nttest naive')
print(tt_nc)
print('\nttest bias corrected')
print(tt_bc)

print("\nbse after fit option ")
bse = np.column_stack((mdf2.bse, mdf2.bse, mdf_nc.bse, mdf_bc.bse))
print(bse)

print("\nimplemented `standard_errors`")
bse2 = np.column_stack((mdf2.bse, mdf2.standard_errors(),
                                 mdf2.standard_errors(covariance_type='naive'),
                                 mdf2.standard_errors(covariance_type='bias_reduced')))
print(bse2)
print("bse and `standard_errors` agree:", np.allclose(bse, bse2))

print("\nimplied standard errors in t_test")
bse1 = np.column_stack((mdf2.bse, tt2.sd, tt_nc.sd, tt_bc.sd))
print(bse1)
print("t_test uses correct cov_params:", np.allclose(bse1, bse2))
