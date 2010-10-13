
import numpy as np
import scikits.statsmodels as sm
from scikits.statsmodels.sandbox.tsa.arima import arma_generate_sample
from scikits.statsmodels.sandbox.tsa.arma_mle import Arma as Arma
from scikits.statsmodels.sandbox.tsa.arima import ARIMA as ARIMA_old
from scikits.statsmodels.sandbox.regression.mle import Arma as Armamle_old
from scikits.statsmodels.sandbox.tsa.kalmanf import ARMA as ARMA_kf




print "\nExample 1"
ar = [1.0,  -0.6, 0.1]
ma = [1.0,  0.3, 0.1]
nobs = 100
y22 = arma_generate_sample(ar, ma, nobs, 0.2)
y22 -= y22.mean()
start_params = [0.1, 0.1, 0.1, 0.1]
start_params_lhs = [-0.1, -0.1, 0.1, 0.1]

print 'truelhs', np.r_[ar[1:], ma[1:]]


###bug in current version, fixed in Skipper and 1 more
###arr[1:q,:] = params[p+k:p+k+q]  # p to p+q short params are MA coeffs
###ValueError: array dimensions are not compatible for copy
##arma22 = ARMA_kf(y22, constant=False, order=(2,2))
##res = arma22.fit(start_params=start_params)
##print res.params

print '\nARIMA new'
arest2 = Arma(y22)
rhohat2, cov_x2a, infodict, mesg, ier = arest2.fit((2,0,2))
print rhohat2
print 'with mle'
arest2.nar = 2
arest2.nma = 2
res = arest2.fit_mle(start_params=np.r_[-0.1, -0.1, 0.1, 0.1, 0.5], method='nm') #no order in fit
print res.params

print '\nARIMA_old'
arest = ARIMA_old(y22)
rhohat1, cov_x1, infodict, mesg, ier = arest.fit((2,0,2))
print rhohat1
print np.sqrt(np.diag(cov_x1))
err1 = arest.errfn(x=y22)
print np.var(err1)
print 'bse ls, formula  not checked'
print np.sqrt(np.diag(cov_x1))*err1.std()
print 'bsejac for mle'
print arest2.bsejac

print '\nyule-walker'
print sm.regression.yule_walker(y22, order=2, inv=True)

print '\nArmamle_old'
arma1 = Armamle_old(y22)
arma1.nar = 2
arma1.nma = 2
#arma1res = arma1.fit(start_params=np.r_[-0.5, -0.1, 0.1, 0.1, 0.5], method='fmin')
                     #maxfun=1000)
arma1res = arma1.fit(start_params=res.params*0.7, method='fmin')
print arma1res.params
