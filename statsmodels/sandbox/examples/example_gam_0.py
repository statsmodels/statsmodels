'''first examples for gam and PolynomialSmoother used for debugging

This example was written as a test case.
The data generating process is chosen so the parameters are well identified
and estimated.


Note: uncomment plt.show() to display graphs
'''

example = 2 #3  # 1,2 or 3

import numpy as np
from statsmodels.compat.python import zip
import numpy.random as R
import matplotlib.pyplot as plt

from statsmodels.sandbox.gam import AdditiveModel
from statsmodels.sandbox.gam import Model as GAM #?
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM

#np.random.seed(987654)

standardize = lambda x: (x - x.mean()) / x.std()
demean = lambda x: (x - x.mean())
nobs = 500
lb, ub = -1., 1. #for Poisson
#lb, ub = -0.75, 2 #0.75 #for Binomial
x1 = R.uniform(lb, ub, nobs)   #R.standard_normal(nobs)
x1 = np.linspace(lb, ub, nobs)
x1.sort()
x2 = R.uniform(lb, ub, nobs)   #
#x2 = R.standard_normal(nobs)
x2.sort()
#x2 = np.cos(x2)
x2 = x2 + np.exp(x2/2.)
#x2 = np.log(x2-x2.min()+0.1)
y = 0.5 * R.uniform(lb, ub, nobs)   #R.standard_normal((nobs,))

f1 = lambda x1: (2*x1 - 0.5 * x1**2  - 0.75 * x1**3) # + 0.1 * np.exp(-x1/4.))
f2 = lambda x2: (x2 - 1* x2**2) # - 0.75 * np.exp(x2))
z = standardize(f1(x1)) + standardize(f2(x2))
z = standardize(z) + 1 # 0.1
#try this
z = f1(x1) + f2(x2)
#z = demean(z)
z -= np.median(z)
print('z.std()', z.std())
#z = standardize(z) + 0.2
# with standardize I get better values, but I don't know what the true params are
print(z.mean(), z.min(), z.max())

#y += z  #noise
y = z

d = np.array([x1,x2]).T


if example == 1:
    print("normal")
    m = AdditiveModel(d)
    m.fit(y)
    x = np.linspace(-2,2,50)

    print(m)

import scipy.stats, time

if example == 2:
    print("binomial")
    mod_name = 'Binomial'
    f = families.Binomial()
    #b = np.asarray([scipy.stats.bernoulli.rvs(p) for p in f.link.inverse(y)])
    b = np.asarray([scipy.stats.bernoulli.rvs(p) for p in f.link.inverse(z)])
    b.shape = y.shape
    m = GAM(b, d, family=f)
    toc = time.time()
    m.fit(b)
    tic = time.time()
    print(tic-toc)
    #for plotting
    yp = f.link.inverse(y)
    p = b


if example == 3:
    print("Poisson")
    f = families.Poisson()
    #y = y/y.max() * 3
    yp = f.link.inverse(z)
    #p = np.asarray([scipy.stats.poisson.rvs(p) for p in f.link.inverse(y)], float)
    p = np.asarray([scipy.stats.poisson.rvs(p) for p in f.link.inverse(z)], float)
    p.shape = y.shape
    m = GAM(p, d, family=f)
    toc = time.time()
    m.fit(p)
    tic = time.time()
    print(tic-toc)

if example > 1:
    y_pred = m.results.mu# + m.results.alpha#m.results.predict(d)
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(p, '.')
    plt.plot(yp, 'b-', label='true')
    plt.plot(y_pred, 'r-', label='GAM')
    plt.legend(loc='upper left')
    plt.title('gam.GAM ' + mod_name)

    counter = 2
    for ii, xx in zip(['z', 'x1', 'x2'], [z, x1, x2]):
        sortidx = np.argsort(xx)
        #plt.figure()
        plt.subplot(2, 2, counter)
        plt.plot(xx[sortidx], p[sortidx], '.')
        plt.plot(xx[sortidx], yp[sortidx], 'b.', label='true')
        plt.plot(xx[sortidx], y_pred[sortidx], 'r.', label='GAM')
        plt.legend(loc='upper left')
        plt.title('gam.GAM ' + mod_name + ' ' + ii)
        counter += 1

#    counter = 2
#    for ii, xx in zip(['z', 'x1', 'x2'], [z, x1, x2]):
#        #plt.figure()
#        plt.subplot(2, 2, counter)
#        plt.plot(xx, p, '.')
#        plt.plot(xx, yp, 'b-', label='true')
#        plt.plot(xx, y_pred, 'r-', label='GAM')
#        plt.legend(loc='upper left')
#        plt.title('gam.GAM Poisson ' + ii)
#        counter += 1

    plt.figure()
    plt.plot(z, 'b-', label='true' )
    plt.plot(np.log(m.results.mu), 'r-', label='GAM')
    plt.title('GAM Poisson, raw')


plt.figure()
plt.plot(x1, standardize(m.smoothers[0](x1)), 'r')
plt.plot(x1, standardize(f1(x1)), linewidth=2)
plt.figure()
plt.plot(x2, standardize(m.smoothers[1](x2)), 'r')
plt.plot(x2, standardize(f2(x2)), linewidth=2)

##y_pred = m.results.predict(d)
##plt.figure()
##plt.plot(z, p, '.')
##plt.plot(z, yp, 'b-', label='true')
##plt.plot(z, y_pred, 'r-', label='AdditiveModel')
##plt.legend()
##plt.title('gam.AdditiveModel')




#plt.show()



##     pylab.figure(num=1)
##     pylab.plot(x1, standardize(m.smoothers[0](x1)), 'b')
##     pylab.plot(x1, standardize(f1(x1)), linewidth=2)
##     pylab.figure(num=2)
##     pylab.plot(x2, standardize(m.smoothers[1](x2)), 'b')
##     pylab.plot(x2, standardize(f2(x2)), linewidth=2)
##     pylab.show()

