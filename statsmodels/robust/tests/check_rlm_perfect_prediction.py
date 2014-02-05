# -*- coding: utf-8 -*-
"""Script to check the behavior of all RLM version for perfect prediction
or almost perfect prediction

check is whether `fail` is empty


Created on Mon Jan 27 08:47:53 2014
Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from statsmodels.robust.robust_linear_model import RLM
import statsmodels.robust.scale as rscale
import statsmodels.robust.norms as rnorms

norm_names = ['AndrewWave', 'Hampel', 'HuberT', 'LeastSquares', 'RamsayE',
              'TrimmedMean', 'TukeyBiweight']
# 'RobustNorm' is super class
norms = [getattr(rnorms, ni) for ni in norm_names]

scale_names = ['mad', 'Gaussian', 'Huber', 'HuberScale']
# note: 'mad' is a string keyword in RLM.fit, not a scale instance
scales = ['mad', rscale.HuberScale(), rscale.HuberScale(d=1.5)]
# scales not usable rscale.Gaussian() incorrect interface,
#                   rscale.Huber estimates mean by default
scales2 = [rscale.mad, rscale.Huber(), rscale.HuberScale(), rscale.HuberScale(d=1.5)]

y1 = np.array([27.01, 27.01, 28.5, 27.01, 27.04])
y2 = np.array([ 0,  0,  0,  0,  0,  0, -1,  1])
y3 = 4 + np.array([ 0,  0,  0,  0,  0,  0, -1.5,  1])
y4 = 4. + np.zeros(10)

endogs = [y1 - 27 + 4, 4 + y2, y3, y4]#[:-1]


y = y1
rlm = RLM(y, np.ones(len(y)), M=rnorms.TukeyBiweight())
res = rlm.fit(scale_est=rscale.HuberScale())
print res.params, res.bse, res.scale

success = []
fail = []
for norm in norms:
    for scale in scales:
        for y in endogs:
            try:
                rlm = RLM(y, np.ones(len(y)), M=norm())
                res = rlm.fit(scale_est=scale)
                #print res.params, res.bse, res.scale
                success.append([norm, scale, res.params, res.bse, res.scale])
            except Exception as e:
                fail.append([y, norm, scale, res.params, res.bse, res.scale])
                print '   using  ', norm, scale
                print e

params_all = np.array([r[2] for r in success]).reshape(-1, len(endogs))
scale_all = np.array([r[4] for r in success]).reshape(-1, len(endogs))
bse_all = np.array([r[3] for r in success]).reshape(-1, len(endogs))

assert_allclose(params_all, 4, atol=0.32)
assert_equal(len(fail), 0)
assert_equal(len(np.isnan(scale_all)), 0)
# TODO: check bse has 3 nans and the last column (y4) is all nan

rlm = RLM(y, np.ones(len(y)), M=rnorms.HuberT())
res = rlm.fit(scale_est=rscale.HuberScale())


print 'params'
print(params_all)
print '\nscale'
print(scale_all)
print '\nbse'
print(bse_all)


success = []
fail = []
for scale in scales2:
    for y in (endogs + [np.arange(5)] + [yi - yi.mean() for yi in endogs]):
        try:
            if isinstance(scale, rscale.HuberScale):
                res = scale(len(y) - 1, len(y), y)
            elif isinstance(scale, rscale.Huber):
                res = scale(y, mu=0)[1]
            else:
                res = scale(y)
            #print res.params, res.bse, res.scale
            success.append([y, scale, res])
        except Exception as e:
            fail.append([y, scale])
            print '   using  ', scale, y
            print e

scale_estimates = np.array([r[2] for r in success]).reshape(-1, len(endogs))
# regression test, currently two cases with nan scale in HuberScale
assert_equal(np.isnan(scale_estimates).sum(), 2)
assert_equal(len(fail), 0)

print fail
print '\nscale'
scale_estimates = np.array([r[2] for r in success]).reshape(-1, len(endogs))
print(scale_estimates)
print '\n number of nan scales', np.isnan(scale_estimates).sum()

# values for regression test, printed so rows are datasets
scale_regression = np.array([
       [ 0.,  1.483,  0., -0., -0.   ,  4.312,  0.541,  4.884, 0.639],
       [ 0.,  0.   , -0., -0., -0.   ,  4.264,  0.686,  4.832, 0.023],
       [ 0.,  0.   , -0., -0.,  4.924,  2.77 , np.nan,  4.779, 0.157],
       [ 0.,  0.   , -0., -0.,  4.359,  0.671,  5.518,  3.104, np.nan]]).T

assert_allclose(scale_estimates, scale_regression, atol=0.0005)
