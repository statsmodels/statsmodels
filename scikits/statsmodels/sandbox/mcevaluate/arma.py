
import numpy as np
from scikits.statsmodels.sandbox.tsa.arima import arma_generate_sample
from scikits.statsmodels.sandbox.tsa.arma_mle import Arma


#TODO: still refactoring problem with cov_x
#copied from sandbox.tsa.arima.py
def mcarma22(niter=10):
    '''run Monte Carlo for ARMA(2,2)

    DGP parameters currently hard coded
    also sample size `nsample`

    was not a self contained function, used instances from outer scope
      now corrected

    '''
    nsample = 1000
    #ar = [1.0, 0, 0]
    ar = [1.0, -0.55, -0.1]
    #ma = [1.0, 0, 0]
    ma = [1.0,  0.3,  0.2]
    results = []
    results_bse = []
    for _ in range(niter):
        y2 = arma_generate_sample(ar,ma,nsample, 0.1)
        arest2 = Arma(y2)
        rhohat2a, cov_x2a, infodict, mesg, ier = arest2.fit((2,0,2))
        results.append(rhohat2a)
        err2a = arest2.geterrors(rhohat2a)
        sige2a = np.sqrt(np.dot(err2a,err2a)/nsample)
        #results_bse.append(sige2a * np.sqrt(np.diag(cov_x2a)))
        #results_bse.append(sige2a * np.sqrt(np.diag(cov_x2a)))
    return np.r_[ar[1:], ma[1:]], np.array(results)#, np.array(results_bse)

true, est = mcarma22(niter=50)
print true
#print est
print est.mean(0)

''' niter 50, sample size=1000
[-0.55 -0.1   0.3   0.2 ]
[-0.542401   -0.09904305  0.30840599  0.2052473 ]

[-0.55 -0.1   0.3   0.2 ]
[-0.54681176 -0.09742921  0.2996297   0.20624258]

'''
