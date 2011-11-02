'''example for checking how far GAM work, GAM is incompletely fixed

Note: uncomment plt.show() to display graphs
'''
#FIXME problems calling GLM, 3rd parameter missing


# convert to script for testing, so we get interactive variable access
example = 1  # 1,2 or 3

import numpy as np
import numpy.random as R
from scikits.statsmodels.sandbox.gam import AdditiveModel
from scikits.statsmodels.sandbox.gam import Model as GAM #?
from scikits.statsmodels.genmod.families import family
from scikits.statsmodels.genmod.generalized_linear_model import GLM

standardize = lambda x: (x - x.mean()) / x.std()
demean = lambda x: (x - x.mean())
x1 = R.standard_normal(500)
x1.sort()
x2 = R.standard_normal(500)
x2.sort()
y = R.standard_normal((500,))

f1 = lambda x1: (x1 + x1**2 - 3 - 1.5 * x1**3 + np.exp(-x1))
f2 = lambda x2: (x2 + x2**2 - np.exp(x2))
z = standardize(f1(x1)) + standardize(f2(x2))
z = standardize(z) * 0.1

y += z
d = np.array([x1,x2]).T


if example == 1:
    print "normal"
    m = AdditiveModel(d)
    m.fit(y)
    x = np.linspace(-2,2,50)

    print m

import scipy.stats, time

if example == 2:
    print "binomial"
    f = family.Binomial()
    b = np.asarray([scipy.stats.bernoulli.rvs(p) for p in f.link.inverse(y)])
    b.shape = y.shape
    m = GAM(b, d, family=f)
    toc = time.time()
    m.fit(b)
    tic = time.time()
    print tic-toc


if example == 3:
    print "Poisson"
    f = family.Poisson()
    p = np.asarray([scipy.stats.poisson.rvs(p) for p in f.link.inverse(y)])
    p.shape = y.shape
    m = GAM(p, d, family=f)
    toc = time.time()
    m.fit(p)
    tic = time.time()
    print tic-toc

import matplotlib.pyplot as plt
plt.figure(num=1)
plt.plot(x1, standardize(m.smoothers[0](x1)), 'r')
plt.plot(x1, standardize(f1(x1)), linewidth=2)
plt.figure(num=2)
plt.plot(x2, standardize(m.smoothers[1](x2)), 'r')
plt.plot(x2, standardize(f2(x2)), linewidth=2)

#plt.show()



##     pylab.figure(num=1)
##     pylab.plot(x1, standardize(m.smoothers[0](x1)), 'b')
##     pylab.plot(x1, standardize(f1(x1)), linewidth=2)
##     pylab.figure(num=2)
##     pylab.plot(x2, standardize(m.smoothers[1](x2)), 'b')
##     pylab.plot(x2, standardize(f2(x2)), linewidth=2)
##     pylab.show()

