

from __future__ import print_function
import numpy as np
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt
import statsmodels.sandbox.tsa.fftarma as fa
from statsmodels.tsa.descriptivestats import TsaDescriptive
from statsmodels.tsa.arma_mle import Arma

x = fa.ArmaFft([1, -0.5], [1., 0.4], 40).generate_sample(size=200, burnin=1000)
d = TsaDescriptive(x)
d.plot4()

#d.fit(order=(1,1))
d.fit((1,1), trend='nc')
print(d.res.params)

modc = Arma(x)
resls = modc.fit(order=(1,1))
print(resls[0])
rescm = modc.fit_mle(order=(1,1), start_params=[-0.4,0.4, 1.])
print(rescm.params)

#decimal 1 corresponds to threshold of 5% difference
assert_almost_equal(resls[0] / d.res.params, 1, decimal=1)
assert_almost_equal(rescm.params[:-1] / d.res.params, 1, decimal=1)
#copied to tsa.tests

plt.figure()
plt.plot(x, 'b-o')
plt.plot(modc.predicted(), 'r-')
plt.figure()
plt.plot(modc.error_estimate)
#plt.show()

from statsmodels.miscmodels.tmodel import TArma

modct = TArma(x)
reslst = modc.fit(order=(1,1))
print(reslst[0])
rescmt = modct.fit_mle(order=(1,1), start_params=[-0.4,0.4, 10, 1.],maxiter=500,
                       maxfun=500)
print(rescmt.params)


from statsmodels.tsa.arima_model import ARMA
mkf = ARMA(x)
##rkf = mkf.fit((1,1))
##rkf.params
rkf = mkf.fit((1,1), trend='nc')
print(rkf.params)

from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(12345)
y_arma22 = arma_generate_sample([1.,-.85,.35, -0.1],[1,.25,-.7], nsample=1000)
##arma22 = ARMA(y_arma22)
##res22 = arma22.fit(trend = 'nc', order=(2,2))
##print 'kf ',res22.params
##res22css = arma22.fit(method='css',trend = 'nc', order=(2,2))
##print 'css', res22css.params
mod22 = Arma(y_arma22)
resls22 = mod22.fit(order=(2,2))
print('ls ', resls22[0])
resmle22 = mod22.fit_mle(order=(2,2), maxfun=2000)
print('mle', resmle22.params)

f = mod22.forecast()
f3 = mod22.forecast3(start=900)[-20:]

print(y_arma22[-10:])
print(f[-20:])
print(f3[-109:-90])
plt.show()