

import numpy as np
from numpy.testing import assert_almost_equal
import scikits.statsmodels.sandbox.tsa.fftarma as fa
#from scikits.statsmodels.tsa.descriptivestats import TsaDescriptive
from scikits.statsmodels.tsa.arma_mle import Arma
from scikits.statsmodels.tsa.arima import ARMA

def test_compare_arma():
    #this is a preliminary test to compare arma_kf, arma_cond_ls and arma_cond_mle
    #the results returned by the fit methods are incomplete
    #for now without random.seed

    #np.random.seed(9876565)
    x = fa.ArmaFft([1, -0.5], [1., 0.4], 40).generate_sample(size=200, burnin=1000)

    d = ARMA(x)
    d.fit((1,1), trend='nc')

    modc = Arma(x)
    resls = modc.fit(order=(1,1))
    rescm = modc.fit_mle(order=(1,1), start_params=[0.4,0.4, 1.])

    #decimal 1 corresponds to threshold of 5% difference
    #still different sign  corrcted
    #assert_almost_equal(np.abs(resls[0] / d.params), np.ones(d.params.shape), decimal=1)
    assert_almost_equal(resls[0] / d.params, np.ones(d.params.shape), decimal=1)
    #rescm also contains variance estimate as last element of params

    #assert_almost_equal(np.abs(rescm.params[:-1] / d.params), np.ones(d.params.shape), decimal=1)
    assert_almost_equal(rescm.params[:-1] / d.params, np.ones(d.params.shape), decimal=1)
    return resls[0], d.params, rescm.params
