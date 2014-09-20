# -*- coding: utf-8 -*-
"""Cluster robust standard errors for OLS

Created on Fri Dec 16 12:52:13 2011
Author: Josef Perktold
"""

from statsmodels.compat.python import urlretrieve
import numpy as np
from numpy.testing import assert_almost_equal

import statsmodels.api as sm
import statsmodels.stats.sandwich_covariance as sw

#http://www.ats.ucla.edu/stat/stata/seminars/svy_stata_intro/srs.dta

import statsmodels.iolib.foreign as dta

try:
    srs = dta.genfromdta("srs.dta")
    print('using local file')
except IOError:
    urlretrieve('http://www.ats.ucla.edu/stat/stata/seminars/svy_stata_intro/srs.dta', 'srs.dta')
    print('downloading file')
    srs = dta.genfromdta("srs.dta")
#    from statsmodels.datasets import webuse
#    srs = webuse('srs', 'http://www.ats.ucla.edu/stat/stata/seminars/svy_stata_intro/')
#    #does currently not cache file

y = srs['api00']
#older numpy don't reorder
#x = srs[['growth', 'emer', 'yr_rnd']].view(float).reshape(len(y), -1)
#force sequence
x = np.column_stack([srs[ii] for ii in ['growth', 'emer', 'yr_rnd']])
group = srs['dnum']

#xx = sm.add_constant(x, prepend=True)
xx = sm.add_constant(x, prepend=False) #const at end for Stata compatibility

#remove nan observation
mask = (xx!=-999.0).all(1)   #nan code in dta file
mask.shape
y = y[mask]
xx = xx[mask]
group = group[mask]

#run OLS

res_srs = sm.OLS(y, xx).fit()
print('params    ', res_srs.params)
print('bse_OLS   ', res_srs.bse)

#get cluster robust standard errors and compare with STATA

cov_cr = sw.cov_cluster(res_srs, group.astype(int))
bse_cr = sw.se_cov(cov_cr)
print('bse_rob   ', bse_cr)

res_stata = np.rec.array(
     [ ('growth', '|', -0.1027121, 0.22917029999999999, -0.45000000000000001, 0.65500000000000003, -0.55483519999999997, 0.34941109999999997),
       ('emer', '|', -5.4449319999999997, 0.72939690000000001, -7.46, 0.0, -6.8839379999999997, -4.0059269999999998),
       ('yr_rnd', '|', -51.075690000000002, 22.83615, -2.2400000000000002, 0.027, -96.128439999999998, -6.0229350000000004),
       ('_cons', '|', 740.3981, 13.460760000000001, 55.0, 0.0, 713.84180000000003, 766.95439999999996)],
      dtype=[('exogname', '|S6'), ('del', '|S1'), ('params', 'float'),
             ('bse', 'float'), ('tvalues', 'float'), ('pvalues', 'float'),
             ('cilow', 'float'), ('ciupp', 'float')])

print('diff Stata', bse_cr - res_stata.bse)
assert_almost_equal(bse_cr, res_stata.bse, decimal=6)

#We see that in this case the robust standard errors of the parameter estimates
#are larger than those of OLS by 8 to 35 %
print('reldiff to OLS', bse_cr/res_srs.bse - 1)
