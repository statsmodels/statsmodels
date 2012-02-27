# -*- coding: utf-8 -*-
"""

Created on Fri Dec 16 12:52:13 2011

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_almost_equal

import scikits.statsmodels.api as sm

import scikits.statsmodels.sandbox.panel.sandwich_covariance as sw
import scikits.statsmodels.sandbox.panel.sandwich_covariance_generic as swg


#http://www.ats.ucla.edu/stat/stata/seminars/svy_stata_intro/srs.dta
import scikits.statsmodels.iolib.foreign as dta

srs = dta.genfromdta("srs.dta")
y = srs['api00']
#x = srs[['growth', 'emer', 'yr_rnd']].view(float).reshape(len(y), -1)
#force sequence
x = np.column_stack([srs[ii] for ii in ['growth', 'emer', 'yr_rnd']])
group = srs['dnum']

#xx = sm.add_constant(x, prepend=True)
xx = sm.add_constant(x, prepend=False) #for Stata compatibility

#remove nan observation
mask = (xx!=-999.0).all(1)   #nan code in dta file
mask.shape
y = y[mask]
xx = xx[mask]
group = group[mask]

res_srs = sm.OLS(y, xx).fit()
print res_srs.params
print res_srs.bse

bse_cr = sw.cov_cluster(res_srs, group.astype(int))[1]
print bse_cr

res_stata = np.rec.array(
     [ ('growth', '|', -0.1027121, 0.22917029999999999, -0.45000000000000001, 0.65500000000000003, -0.55483519999999997, 0.34941109999999997),
       ('emer', '|', -5.4449319999999997, 0.72939690000000001, -7.46, 0.0, -6.8839379999999997, -4.0059269999999998),
       ('yr_rnd', '|', -51.075690000000002, 22.83615, -2.2400000000000002, 0.027, -96.128439999999998, -6.0229350000000004),
       ('_cons', '|', 740.3981, 13.460760000000001, 55.0, 0.0, 713.84180000000003, 766.95439999999996)],
      dtype=[('exogname', '|S6'), ('del', '|S1'), ('params', '<f8'),
             ('bse', '<f8'), ('tvalues', '<f8'), ('pvalues', '<f8'),
             ('cilow', '<f8'), ('ciupp', '<f8')])

print bse_cr - res_stata.bse
assert_almost_equal(bse_cr, res_stata.bse, decimal=6)
