

import numpy as np
from numpy.testing import assert_almost_equal
import matplotlib.pyplot as plt
import scikits.statsmodels.sandbox.tsa.fftarma as fa
from scikits.statsmodels.tsa.descriptivestats import TsaDescriptive
from scikits.statsmodels.tsa.arma_mle import Arma

x = fa.ArmaFft([1, -0.5], [1., 0.4], 40).generate_sample(size=200, burnin=1000)
d = TsaDescriptive(x)
d.plot4()

#d.fit(order=(1,1))
d.fit((1,1), trend='nc')
print d.mod.params

modc = Arma(x)
resls = modc.fit(order=(1,1))
print resls[0]
rescm = modc.fit_mle(order=(1,1), start_params=[-0.4,0.4, 1.])
print rescm.params

#decimal 1 corresponds to threshold of 5% difference
assert_almost_equal(resls[0] / d.mod.params, 1, decimal=1)
assert_almost_equal(rescm.params[:-1] / d.mod.params, 1, decimal=1)
#copied to tsa.tests

plt.figure()
plt.plot(x, 'b-o')
plt.plot(modc.predicted(), 'r-')
plt.figure()
plt.plot(modc.error_estimate)
#plt.show()

from scikits.statsmodels.miscmodels.tmodel import TArma

modct = TArma(x)
reslst = modc.fit(order=(1,1))
print reslst[0]
rescmt = modct.fit_mle(order=(1,1), start_params=[-0.4,0.4, 10, 1.],maxiter=500,
                       maxfun=500)
print rescmt.params
