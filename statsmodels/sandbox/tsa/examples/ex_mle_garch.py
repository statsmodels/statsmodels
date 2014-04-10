# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 01:01:50 2010

Author: josef-pktd

latest result
-------------
all are very close
garch0 has different parameterization of constant
ordering of parameters is different


seed 2780185
h.shape (2000,)
Optimization terminated successfully.
         Current function value: 2093.813397
         Iterations: 387
         Function evaluations: 676
ggres.params [-0.6146253   0.1914537   0.01039355  0.78802188]
Optimization terminated successfully.
         Current function value: 2093.972953
         Iterations: 201
         Function evaluations: 372
ggres0.params [-0.61537527  0.19635128  4.00706058]
Warning: Desired error not necessarily achieveddue to precision loss
         Current function value: 2093.972953
         Iterations: 51
         Function evaluations: 551
         Gradient evaluations: 110
ggres0.params [-0.61537855  0.19635265  4.00694669]
Optimization terminated successfully.
         Current function value: 2093.751420
         Iterations: 103
         Function evaluations: 187
[ 0.78671519  0.19692222  0.61457171]
-2093.75141963

Final Estimate:
 LLH:  2093.750    norm LLH:  2.093750
    omega    alpha1     beta1
0.7867438 0.1970437 0.6145467

long run variance comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
R
>>> 0.7867438/(1- 0.1970437- 0.6145467)
4.1757097302897526
Garch (gjr) asymetric, longrun var ?
>>> 1/(1-0.6146253 - 0.1914537 - 0.01039355) * 0.78802188
4.2937548579245242
>>> 1/(1-0.6146253 - 0.1914537 + 0.01039355) * 0.78802188
3.8569053452140345
Garch0
>>> (1-0.61537855 - 0.19635265) * 4.00694669
0.7543830449902722
>>> errgjr4.var() #for different random seed
4.0924199964716106

todo: add code and verify, check for longer lagpolys

"""

from __future__ import print_function
import numpy as np
from numpy.testing import assert_almost_equal

import matplotlib.pyplot as plt
import numdifftools as ndt

import statsmodels.api as sm
from statsmodels.sandbox import tsa
from statsmodels.sandbox.tsa.garch import *  # local import


nobs = 1000
examples = ['garch', 'rpyfit']
if 'garch' in examples:
    err,h = generate_kindofgarch(nobs, [1.0, -0.95], [1.0,  0.1], mu=0.5)
    plt.figure()
    plt.subplot(211)
    plt.plot(err)
    plt.subplot(212)
    plt.plot(h)
    #plt.show()

    seed = 3842774 #91234  #8837708
    seed = np.random.randint(9999999)
    print('seed', seed)
    np.random.seed(seed)
    ar1 = -0.9
    err,h = generate_garch(nobs, [1.0, ar1], [1.0,  0.50], mu=0.0,scale=0.1)
#    plt.figure()
#    plt.subplot(211)
#    plt.plot(err)
#    plt.subplot(212)
#    plt.plot(h)
#    plt.figure()
#    plt.subplot(211)
#    plt.plot(err[-400:])
#    plt.subplot(212)
#    plt.plot(h[-400:])
    #plt.show()
    garchplot(err, h)
    garchplot(err[-400:], h[-400:])


    np.random.seed(seed)
    errgjr,hgjr, etax = generate_gjrgarch(nobs, [1.0, ar1],
                                [[1,0],[0.5,0]], mu=0.0,scale=0.1)
    garchplot(errgjr[:nobs], hgjr[:nobs], 'GJR-GARCH(1,1) Simulation - symmetric')
    garchplot(errgjr[-400:nobs], hgjr[-400:nobs], 'GJR-GARCH(1,1) Simulation - symmetric')

    np.random.seed(seed)
    errgjr2,hgjr2, etax = generate_gjrgarch(nobs, [1.0, ar1],
                                [[1,0],[0.1,0.9]], mu=0.0,scale=0.1)
    garchplot(errgjr2[:nobs], hgjr2[:nobs], 'GJR-GARCH(1,1) Simulation')
    garchplot(errgjr2[-400:nobs], hgjr2[-400:nobs], 'GJR-GARCH(1,1) Simulation')

    np.random.seed(seed)
    errgjr3,hgjr3, etax3 = generate_gjrgarch(nobs, [1.0, ar1],
                        [[1,0],[0.1,0.9],[0.1,0.9],[0.1,0.9]], mu=0.0,scale=0.1)
    garchplot(errgjr3[:nobs], hgjr3[:nobs], 'GJR-GARCH(1,3) Simulation')
    garchplot(errgjr3[-400:nobs], hgjr3[-400:nobs], 'GJR-GARCH(1,3) Simulation')

    np.random.seed(seed)
    errgjr4,hgjr4, etax4 = generate_gjrgarch(nobs, [1.0, ar1],
                        [[1., 1,0],[0, 0.1,0.9],[0, 0.1,0.9],[0, 0.1,0.9]],
                        mu=0.0,scale=0.1)
    garchplot(errgjr4[:nobs], hgjr4[:nobs], 'GJR-GARCH(1,3) Simulation')
    garchplot(errgjr4[-400:nobs], hgjr4[-400:nobs], 'GJR-GARCH(1,3) Simulation')

    varinno = np.zeros(100)
    varinno[0] = 1.
    errgjr5,hgjr5, etax5 = generate_gjrgarch(100, [1.0, -0.],
                        [[1., 1,0],[0, 0.1,0.8],[0, 0.05,0.7],[0, 0.01,0.6]],
                        mu=0.0,scale=0.1, varinnovation=varinno)
    garchplot(errgjr5[:20], hgjr5[:20], 'GJR-GARCH(1,3) Simulation')
    #garchplot(errgjr4[-400:nobs], hgjr4[-400:nobs], 'GJR-GARCH(1,3) Simulation')


#plt.show()
seed = np.random.randint(9999999)  # 9188410
print('seed', seed)

x = np.arange(20).reshape(10,2)
x3 = np.column_stack((np.ones((x.shape[0],1)),x))
y, inp = miso_lfilter([1., 0],np.array([[-2.0,3,1],[0.0,0.0,0]]),x3)

nobs = 1000
warmup = 1000
np.random.seed(seed)
ar = [1.0, -0.7]#7, -0.16, -0.1]
#ma = [[1., 1, 0],[0, 0.6,0.1],[0, 0.1,0.1],[0, 0.1,0.1]]
ma = [[1., 0, 0],[0, 0.8,0.0]] #,[0, 0.9,0.0]]
#    errgjr4,hgjr4, etax4 = generate_gjrgarch(warmup+nobs, [1.0, -0.99],
#                        [[1., 1, 0],[0, 0.6,0.1],[0, 0.1,0.1],[0, 0.1,0.1]],
#                        mu=0.2, scale=0.25)

errgjr4,hgjr4, etax4 = generate_gjrgarch(warmup+nobs, ar, ma,
                     mu=0.4, scale=1.01)
errgjr4,hgjr4, etax4 = errgjr4[warmup:], hgjr4[warmup:], etax4[warmup:]
garchplot(errgjr4[:nobs], hgjr4[:nobs], 'GJR-GARCH(1,3) Simulation - DGP')
ggmod = Garch(errgjr4-errgjr4.mean())#hgjr4[:nobs])#-hgjr4.mean()) #errgjr4)
ggmod.nar = 1
ggmod.nma = 1
ggmod._start_params = np.array([-0.6, 0.1, 0.2, 0.0])
ggres = ggmod.fit(start_params=np.array([-0.6, 0.1, 0.2, 0.0]), maxiter=1000)
print('ggres.params', ggres.params)
garchplot(ggmod.errorsest, ggmod.h, title='Garch estimated')

ggmod0 = Garch0(errgjr4-errgjr4.mean())#hgjr4[:nobs])#-hgjr4.mean()) #errgjr4)
ggmod0.nar = 1
ggmod.nma = 1
start_params = np.array([-0.6, 0.2, 0.1])
ggmod0._start_params = start_params #np.array([-0.6, 0.1, 0.2, 0.0])
ggres0 = ggmod0.fit(start_params=start_params, maxiter=2000)
print('ggres0.params', ggres0.params)

ggmod0 = Garch0(errgjr4-errgjr4.mean())#hgjr4[:nobs])#-hgjr4.mean()) #errgjr4)
ggmod0.nar = 1
ggmod.nma = 1
start_params = np.array([-0.6, 0.2, 0.1])
ggmod0._start_params = start_params #np.array([-0.6, 0.1, 0.2, 0.0])
ggres0 = ggmod0.fit(start_params=start_params, method='bfgs', maxiter=2000)
print('ggres0.params', ggres0.params)


g11res = optimize.fmin(lambda params: -loglike_GARCH11(params, errgjr4-errgjr4.mean())[0], [0.93, 0.9, 0.2])
print(g11res)
llf = loglike_GARCH11(g11res, errgjr4-errgjr4.mean())
print(llf[0])

if 'rpyfit' in examples:
    from rpy import r
    r.library('fGarch')
    f = r.formula('~garch(1, 1)')
    fit = r.garchFit(f, data = errgjr4-errgjr4.mean(), include_mean=False)

if 'rpysim' in examples:
    from rpy import r
    f = r.formula('~garch(1, 1)')
    #fit = r.garchFit(f, data = errgjr4)
    x = r.garchSim( n = 500)
    print('R acf', tsa.acf(np.power(x,2))[:15])
    arma3 = Arma(np.power(x,2))
    arma3res = arma3.fit(start_params=[-0.2,0.1,0.5],maxiter=5000)
    print(arma3res.params)
    arma3b = Arma(np.power(x,2))
    arma3bres = arma3b.fit(start_params=[-0.2,0.1,0.5],maxiter=5000, method='bfgs')
    print(arma3bres.params)

    xr = r.garchSim( n = 100)

    x = np.asarray(xr)
    ggmod = Garch(x-x.mean())
    ggmod.nar = 1
    ggmod.nma = 1
    ggmod._start_params = np.array([-0.6, 0.1, 0.2, 0.0])
    ggres = ggmod.fit(start_params=np.array([-0.6, 0.1, 0.2, 0.0]), maxiter=1000)
    print('ggres.params', ggres.params)

    g11res = optimize.fmin(lambda params: -loglike_GARCH11(params, x-x.mean())[0], [0.6, 0.6, 0.2])
    print(g11res)
    llf = loglike_GARCH11(g11res, x-x.mean())
    print(llf[0])

    garchplot(ggmod.errorsest, ggmod.h, title='Garch estimated')
    fit = r.garchFit(f, data = x-x.mean(), include_mean=False, trace=False)
    print(r.summary(fit))

'''based on R default simulation
model = list(omega = 1e-06, alpha = 0.1, beta = 0.8)
nobs = 1000
(with nobs=500, gjrgarch doesn't do well

>>> ggres = ggmod.fit(start_params=np.array([-0.6, 0.1, 0.2, 0.0]), maxiter=1000)
Optimization terminated successfully.
         Current function value: -448.861335
         Iterations: 385
         Function evaluations: 690
>>> print('ggres.params', ggres.params
ggres.params [ -7.75090330e-01   1.57714749e-01  -9.60223930e-02   8.76021411e-07]
rearranged
8.76021411e-07 1.57714749e-01(-9.60223930e-02) 7.75090330e-01

>>> print(g11res
[  2.97459808e-06   7.83128600e-01   2.41110860e-01]
>>> llf = loglike_GARCH11(g11res, x-x.mean())
>>> print(llf[0]
442.603541936

Log Likelihood:
 -448.9376    normalized:  -4.489376
      omega       alpha1        beta1
1.01632e-06  1.02802e-01  7.57537e-01
'''


''' the following is for errgjr4-errgjr4.mean()
ggres.params [-0.54510407  0.22723132  0.06482633  0.82325803]
Final Estimate:
 LLH:  2065.56    norm LLH:  2.06556
        mu      omega     alpha1      beta1
0.07229732 0.83069480 0.26313883 0.53986167

ggres.params [-0.50779163  0.2236606   0.00700036  1.154832
Final Estimate:
 LLH:  2116.084    norm LLH:  2.116084
           mu         omega        alpha1         beta1
-4.759227e-17  1.145404e+00  2.288348e-01  5.085949e-01

run3
DGP
0.4/??    0.8      0.7
gjrgarch:
ggres.params [-0.45196579  0.2569641   0.02201904  1.11942636]
rearranged
const/omega  ma1/alpha1             ar1/beta1
1.11942636 0.2569641(+0.02201904) 0.45196579
g11:
[ 1.10262688  0.26680468  0.45724957]
-2055.73912687
R:
Final Estimate:
 LLH:  2055.738    norm LLH:  2.055738
           mu         omega        alpha1         beta1
-1.665226e-17  1.102396e+00  2.668712e-01  4.573224e-01
fit = r.garchFit(f, data = errgjr4-errgjr4.mean())
rpy.RPy_RException: Error in solve.default(fit$hessian) :
  Lapack routine dgesv: system is exactly singular

run4
DGP:
mu=0.4, scale=1.01
ma = [[1., 0, 0],[0, 0.8,0.0]], ar = [1.0, -0.7]
maybe something wrong with simulation

gjrgarch
ggres.params [-0.50554663  0.24449867 -0.00521004  1.00796791]
rearranged
1.00796791   0.24449867(-0.00521004)   0.50554663
garch11:
[ 1.01258264  0.24149155  0.50479994]
-2056.3877404
R include_constant=False
Final Estimate:
 LLH:  2056.397    norm LLH:  2.056397
    omega    alpha1     beta1
1.0123560 0.2409589 0.5049154
'''


erro,ho, etaxo = generate_gjrgarch(20, ar, ma, mu=0.04, scale=0.01,
                  varinnovation = np.ones(20))

if 'sp500' in examples:
    import tabular as tb
    import scikits.timeseries as ts

    a = tb.loadSV(r'C:\Josef\work-oth\gspc_table.csv')

    s = ts.time_series(a[0]['Close'][::-1],
                dates=ts.date_array(a[0]['Date'][::-1],freq="D"))

    sp500 = a[0]['Close'][::-1]
    sp500r = np.diff(np.log(sp500))


#plt.show()
