# -*- coding: utf-8 -*-
"""Script to check the behavior of all RLM version for perfect prediction
or almost perfect prediction

check is whether `fail` is empty


Created on Mon Jan 27 08:47:53 2014
Author: Josef Perktold
"""

import numpy as np

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


rlm = RLM(y, np.ones(len(y)), M=rnorms.HuberT())
res = rlm.fit(scale_est=rscale.HuberScale())

print 'params'
print(np.array([r[2] for r in success]).reshape(-1, len(endogs)))
print '\nscale'
print(np.array([r[4] for r in success]).reshape(-1, len(endogs)))
print '\nbse'
print(np.array([r[3] for r in success]).reshape(-1, len(endogs)))


success = []
fail = []
for scale in scales2:
    for y in (endogs + [np.arange(5)] + [yi - yi.mean() for yi in endogs]):
        try:
            if isinstance(scale, rscale.HuberScale):
                res = scale(len(y) - 1., len(y), y) # BUG requires float
            elif isinstance(scale, rscale.Huber):
                res = scale(y, mu=np.array(0))[1]
            else:
                res = scale(y)
            #print res.params, res.bse, res.scale
            success.append([y, scale, res])
        except Exception as e:
            fail.append([y, scale])
            print '   using  ', scale, y
            print e

print fail
print '\nscale'
scale_estimates = np.array([r[2] for r in success]).reshape(-1, len(endogs))
print(scale_estimates)
print '\n numbe of nan scales', np.isnan(scale_estimates).sum()
