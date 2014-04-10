'''

doesn't seem to work so well anymore even with nobs=1000 ???
works ok if noise variance is large
'''

from __future__ import print_function
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arma_mle import Arma as Arma
from statsmodels.tsa.arima_process import ARIMA as ARIMA_old
from statsmodels.sandbox.tsa.garch import Arma as Armamle_old
from statsmodels.tsa.arima import ARMA as ARMA_kf




print("\nExample 1")
ar = [1.0,  -0.6, 0.1]
ma = [1.0,  0.5, 0.3]
nobs = 1000
y22 = arma_generate_sample(ar, ma, nobs+1000, 0.5)[-nobs:]
y22 -= y22.mean()
start_params = [0.1, 0.1, 0.1, 0.1]
start_params_lhs = [-0.1, -0.1, 0.1, 0.1]

print('truelhs', np.r_[ar[1:], ma[1:]])





###bug in current version, fixed in Skipper and 1 more
###arr[1:q,:] = params[p+k:p+k+q]  # p to p+q short params are MA coeffs
###ValueError: array dimensions are not compatible for copy
##arma22 = ARMA_kf(y22, constant=False, order=(2,2))
##res = arma22.fit(start_params=start_params)
##print res.params

print('\nARIMA new')
arest2 = Arma(y22)

naryw = 4  #= 30
resyw = sm.regression.yule_walker(y22, order=naryw, inv=True)
arest2.nar = naryw
arest2.nma = 0
e = arest2.geterrors(np.r_[1, -resyw[0]])
x=sm.tsa.tsatools.lagmat2ds(np.column_stack((y22,e)),3,dropex=1,
                            trim='both')
yt = x[:,0]
xt = x[:,1:]
res_ols = sm.OLS(yt, xt).fit()
print('hannan_rissannen')
print(res_ols.params)
start_params = res_ols.params
start_params_mle = np.r_[-res_ols.params[:2],
                          res_ols.params[2:],
                          #res_ols.scale]
                          #areste.var()]
                          np.sqrt(res_ols.scale)]
#need to iterate, ar1 too large ma terms too small
#fix large parameters, if hannan_rissannen are too large
start_params_mle[:-1] = (np.sign(start_params_mle[:-1])
                         * np.minimum(np.abs(start_params_mle[:-1]),0.75))


print('conditional least-squares')

#print rhohat2
print('with mle')
arest2.nar = 2
arest2.nma = 2
#
res = arest2.fit_mle(start_params=start_params_mle, method='nm') #no order in fit
print(res.params)
rhohat2, cov_x2a, infodict, mesg, ier = arest2.fit((2,2))
print('\nARIMA_old')
arest = ARIMA_old(y22)
rhohat1, cov_x1, infodict, mesg, ier = arest.fit((2,0,2))
print(rhohat1)
print(np.sqrt(np.diag(cov_x1)))
err1 = arest.errfn(x=y22)
print(np.var(err1))
print('bse ls, formula  not checked')
print(np.sqrt(np.diag(cov_x1))*err1.std())
print('bsejac for mle')
#print arest2.bsejac
#TODO:check bsejac raises singular matrix linalg error
#in model.py line620: return np.linalg.inv(np.dot(jacv.T, jacv))

print('\nyule-walker')
print(sm.regression.yule_walker(y22, order=2, inv=True))

print('\nArmamle_old')
arma1 = Armamle_old(y22)
arma1.nar = 2
arma1.nma = 2
#arma1res = arma1.fit(start_params=np.r_[-0.5, -0.1, 0.1, 0.1, 0.5], method='fmin')
                     #maxfun=1000)
arma1res = arma1.fit(start_params=res.params*0.7, method='fmin')
print(arma1res.params)
