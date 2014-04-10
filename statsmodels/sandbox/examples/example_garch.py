import numpy as np

import matplotlib.pyplot as plt
#import scikits.timeseries as ts
#import scikits.timeseries.lib.plotlib as tpl

import statsmodels.api as sm
#from statsmodels.sandbox import tsa
from statsmodels.sandbox.tsa.garch import *  # local import

#dta2 = ts.tsfromtxt(r'gspc_table.csv',
#        datecols=0, skiprows=0, delimiter=',',names=True, freq='D')

#print dta2

aa=np.genfromtxt(r'gspc_table.csv', skip_header=0, delimiter=',', names=True)

cl = aa['Close']
ret = np.diff(np.log(cl))[-2000:]*1000.

ggmod = Garch(ret - ret.mean())#hgjr4[:nobs])#-hgjr4.mean()) #errgjr4)
ggmod.nar = 1
ggmod.nma = 1
ggmod._start_params = np.array([-0.1, 0.1, 0.1, 0.1])
ggres = ggmod.fit(start_params=np.array([-0.1, 0.1, 0.1, 0.0]),
                  maxiter=1000,method='bfgs')
print('ggres.params', ggres.params)
garchplot(ggmod.errorsest, ggmod.h, title='Garch estimated')

use_rpy = False
if use_rpy:
    from rpy import r
    r.library('fGarch')
    f = r.formula('~garch(1, 1)')
    fit = r.garchFit(f, data = ret - ret.mean(), include_mean=False)
    f = r.formula('~arma(1,1) + ~garch(1, 1)')
    fit = r.garchFit(f, data = ret)


ggmod0 = Garch0(ret - ret.mean())#hgjr4[:nobs])#-hgjr4.mean()) #errgjr4)
ggmod0.nar = 1
ggmod.nma = 1
start_params = np.array([-0.1, 0.1, ret.var()])
ggmod0._start_params = start_params #np.array([-0.6, 0.1, 0.2, 0.0])
ggres0 = ggmod0.fit(start_params=start_params, maxiter=2000)
print('ggres0.params', ggres0.params)

g11res = optimize.fmin(lambda params: -loglike_GARCH11(params, ret - ret.mean())[0], [0.01, 0.1, 0.1])
print(g11res)
llf = loglike_GARCH11(g11res, ret - ret.mean())
print(llf[0])


ggmod0 = Garch0(ret - ret.mean())#hgjr4[:nobs])#-hgjr4.mean()) #errgjr4)
ggmod0.nar = 2
ggmod.nma = 2
start_params = np.array([-0.1,-0.1, 0.1, 0.1, ret.var()])
ggmod0._start_params = start_params #np.array([-0.6, 0.1, 0.2, 0.0])
ggres0 = ggmod0.fit(start_params=start_params, maxiter=2000)#, method='ncg')
print('ggres0.params', ggres0.params)

ggmod = Garch(ret - ret.mean())#hgjr4[:nobs])#-hgjr4.mean()) #errgjr4)
ggmod.nar = 2
ggmod.nma = 2
start_params = np.array([-0.1,-0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
ggmod._start_params = start_params
ggres = ggmod.fit(start_params=start_params, maxiter=1000)#,method='bfgs')
print('ggres.params', ggres.params)
