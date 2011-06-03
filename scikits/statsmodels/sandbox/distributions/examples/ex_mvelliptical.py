# -*- coding: utf-8 -*-
"""examples for multivariate normal and t distributions


Created on Fri Jun 03 16:00:26 2011

@author: josef
"""

import numpy as np
import scikits.statsmodels.sandbox.distributions.mv_normal as mvd

from numpy.testing import assert_array_almost_equal

cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
                   [ 0.5 ,  1.5 ,  0.6 ],
                   [ 0.75,  0.6 ,  2.  ]])

mu = np.array([-1, 0.0, 2.0])

mvn3 = mvd.MVNormal(mu, cov3)

#compare with random sample
x = mvn3.rvs(size=1000000)

xli = [[2., 1., 1.5],
       [0., 2., 1.5],
       [1.5, 1., 2.5],
       [0., 1., 1.5]]

xliarr = np.asarray(xli).T[None,:, :]

for a in xli:
    print mvn3.cdf(a),
print
print (x<np.array(xli[0])).all(-1).mean(0)
print (x[...,None]<xliarr).all(1).mean(0)
print  mvn3.expect_mc(lambda x: (x<xli[0]).all(-1), size=100000)
print  mvn3.expect_mc(lambda x: (x[...,None]<xliarr).all(1), size=100000)

#other methods
mvn3n = mvn3.normalized()

assert_array_almost_equal(mvn3n.cov, mvn3n.corr, decimal=15)
assert_array_almost_equal(mvn3n.mean, np.zeros(3), decimal=15)

xn = mvn3.normalize(x)
xn_cov = np.cov(xn, rowvar=0)
assert_array_almost_equal(mvn3n.cov, xn_cov, decimal=2)
assert_array_almost_equal(np.zeros(3), xn.mean(0), decimal=2)

mvn3n2 = mvn3.normalized2()
#assert_array_almost_equal(mvn3n.cov, mvn3n2.cov, decimal=2)
#mistake: "normalized2" standardizes
assert_array_almost_equal(np.eye(3), mvn3n2.cov, decimal=2)

xs = mvn3.standardize(x)
xs_cov = np.cov(xn, rowvar=0)
#another mixup xs is normalized
#assert_array_almost_equal(np.eye(3), xs_cov, decimal=2)
assert_array_almost_equal(mvn3.corr, xs_cov, decimal=2)
assert_array_almost_equal(np.zeros(3), xs.mean(0), decimal=2)

mv2m = mvn3.marginal(np.array([0,1]))
print mv2m.mean
print mv2m.cov

mv2c = mvn3.conditional(np.array([0,1]), [0])
print mv2c.mean
print mv2c.cov

