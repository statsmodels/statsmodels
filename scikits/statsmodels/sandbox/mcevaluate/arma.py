
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
    nsample = 100
    #ar = [1.0, 0, 0]
    ar = [1.0, -0.55, -0.1]
    #ma = [1.0, 0, 0]
    ma = [1.0,  0.3,  0.2]
    results = []
    results_bse = []
    for _ in range(niter):
        y2 = arma_generate_sample(ar,ma,nsample+1000, 0.1)[-nsample:]
        y2 -= y2.mean()
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

''' niter 50, sample size=1000, 2 runs
[-0.55 -0.1   0.3   0.2 ]
[-0.542401   -0.09904305  0.30840599  0.2052473 ]

[-0.55 -0.1   0.3   0.2 ]
[-0.54681176 -0.09742921  0.2996297   0.20624258]


niter=50, sample size=200, 3 runs
[-0.55 -0.1   0.3   0.2 ]
[-0.64669489 -0.01134491  0.19972259  0.20634019]

[-0.55 -0.1   0.3   0.2 ]
[-0.53141595 -0.10653234  0.32297968  0.20505973]

[-0.55 -0.1   0.3   0.2 ]
[-0.50244588 -0.125455    0.33867488  0.19498214]

niter=50, sample size=100, 5 runs  --> ar1 too low, ma1 too high
[-0.55 -0.1   0.3   0.2 ]
[-0.35715008 -0.23392766  0.48771794  0.21901059]

[-0.55 -0.1   0.3   0.2 ]
[-0.3554852  -0.21581914  0.51744748  0.24759245]

[-0.55 -0.1   0.3   0.2 ]
[-0.3737861  -0.24665911  0.48031939  0.17274438]

[-0.55 -0.1   0.3   0.2 ]
[-0.30015385 -0.27705506  0.56168199  0.21995759]

[-0.55 -0.1   0.3   0.2 ]
[-0.35879991 -0.22999604  0.4761953   0.19670835]

new version, with burnin 1000 in DGP and demean
[-0.55 -0.1   0.3   0.2 ]
[-0.56770228 -0.00076025  0.25621825  0.24492449]

[-0.55 -0.1   0.3   0.2 ]
[-0.27598305 -0.2312364   0.57599134  0.23582417]

[-0.55 -0.1   0.3   0.2 ]
[-0.38059051 -0.17413628  0.45147109  0.20046776]

[-0.55 -0.1   0.3   0.2 ]
[-0.47789765 -0.08650743  0.3554441   0.24196087]
'''
