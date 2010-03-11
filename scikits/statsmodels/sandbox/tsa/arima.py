'''ARMA process and estimation with scipy.signal.lfilter

2009-09-06: copied from try_signal.py
    reparameterized same as signal.lfilter (positive coefficients)


Notes
-----
* pretty fast
* checked with Monte Carlo and cross comparison with statsmodels yule_walker
  for AR numbers are close but not identical to yule_walker
  not compared to other statistics packages, no degrees of freedom correction
* ARMA(2,2) estimation (in Monte Carlo) requires longer time series to estimate parameters
  without large variance. There might be different ARMA parameters
  with similar impulse response function that cannot be well
  distinguished with small samples (e.g. 100 observations)
* good for one time calculations for entire time series, not for recursive
  prediction
* class structure not very clean yet
* many one-liners with scipy.signal, but takes time to figure out usage
* missing result statistics, e.g. t-values, but standard errors in examples
* no criteria for choice of number of lags
* no constant term in ARMA process
* no integration, differencing for ARIMA
* written without textbook, works but not sure about everything
  briefly checked and it looks to be standard least squares, see below

* theoretical autocorrelation function of general ARMA
  Done, relatively easy to guess solution, time consuming to get
  theoretical test cases,
  example file contains explicit formulas for acovf of MA(1), MA(2) and ARMA(1,1)

* two names for lag polynomials ar = rhoy, ma = rhoe ?


Properties:
Judge, ... (1985): The Theory and Practise of Econometrics

BigJudge p. 237ff:
If the time series process is a stationary ARMA(p,q), then
minimizing the sum of squares is asymptoticaly (as T-> inf)
equivalent to the exact Maximum Likelihood Estimator

Because Least Squares conditional on the initial information
does not use all information, in small samples exact MLE can
be better.

Without the normality assumption, the least squares estimator
is still consistent under suitable conditions, however not
efficient

Author: josefpktd
License: BSD
'''

import numpy as np
from scipy import signal, optimize

class ARIMA(object):
    '''currently ARMA only, no differencing used - no I

    parameterized as
         rhoy(L) y_t = rhoe(L) eta_t

    A instance of this class preserves state, so new class instances should
    be created for different examples
    '''
    def __init__(self):
        pass
    def fit(self,x,p,q, rhoy0=None, rhoe0=None):
        '''estimate lag coefficients of ARMA orocess by least squares

        Parameters
        ----------
            x : array, 1d
                time series data
            p : int
                number of AR lags to estimate
            q : int
                number of MA lags to estimate
            rhoy0, rhoe0 : array_like (optional)
                starting values for estimation

        Returns
        -------
            rh, cov_x, infodict, mesg, ier : output of scipy.optimize.leastsq
            rh :
                estimate of lag parameters, concatenated [rhoy, rhoe]
            cov_x :
                unscaled (!) covariance matrix of coefficient estimates


        '''
        def errfn( rho):
            #rhoy, rhoe = rho
            rhoy = np.concatenate(([1], rho[:p]))
            rhoe = np.concatenate(([1], rho[p:]))
            etahatr = signal.lfilter(rhoy, rhoe, x)
            #print rho,np.sum(etahatr*etahatr)
            return etahatr

        if rhoy0 is None:
            rhoy0 = 0.5 * np.ones(p)
        if rhoe0 is None:
            rhoe0 = 0.5 * np.ones(q)
        usels = True
        if usels:
            rh, cov_x, infodict, mesg, ier = \
               optimize.leastsq(errfn, np.r_[rhoy0, rhoe0],ftol=1e-10,full_output=True)
        else:
            # fmin_bfgs is slow or doesn't work yet
            errfnsum = lambda rho : np.sum(errfn(rho)**2)
            #xopt, {fopt, gopt, Hopt, func_calls, grad_calls
            rh,fopt, gopt, cov_x, _,_, ier = \
                optimize.fmin_bfgs(errfnsum, np.r_[rhoy0, rhoe0], maxiter=2, full_output=True)
            infodict, mesg = None, None
        self.rh = rh
        self.rhoy = np.concatenate(([1], rh[:p]))
        self.rhoe = np.concatenate(([1], rh[p:])) #rh[-q:])) doesnt work for q=0
        self.error_estimate = errfn(rh)
        return rh, cov_x, infodict, mesg, ier

    def errfn(self, rho=None, p=None, x=None):
        ''' duplicate -> remove one
        '''
        #rhoy, rhoe = rho
        if not rho is None:
            rhoy = np.concatenate(([1],  rho[:p]))
            rhoe = np.concatenate(([1],  rho[p:]))
        else:
            rhoy = self.rhoy
            rhoe = self.rhoe
        etahatr = signal.lfilter(rhoy, rhoe, x)
        #print rho,np.sum(etahatr*etahatr)
        return etahatr

    def predicted(self, rhoy=None, rhoe=None):
        '''past predicted values of time series
        just added, not checked yet
        '''
        if rhoy is None:
            rhoy = self.rhoy
        if rhoe is None:
            rhoe = self.rhoe
        return self.x + self.error_estimate

    def forecast(self, ar=None, ma=None, nperiod=10):
        eta = np.r_[self.error_estimate, np.zeros(nperiod)]
        if ar is None:
            ar = self.rhoy
        if ma is None:
            ma = self.rhoe
        return signal.lfilter(ma, ar, eta)

    def generate_sample(self, ar, ma, nsample, std=1):
        eta = std * np.random.randn(nsample)
        return signal.lfilter(ma, ar, eta)

def arma_generate_sample(ar, ma, nsample, sigma=1, distrvs=np.random.randn):
    '''generate an random sample of an ARMA process

    Parameters
    ----------
    ar : array_like, 1d
        coefficient for autoregressive lag polynomial, including zero lag
    ma : array_like, 1d
        coefficient for moving-average lag polynomial, including zero lag
    nsample : int
        length of simulated time series
    sigma : float
        standard deviation of noise
    distrvs : function, random number generator
        function that generates the random numbers, and takes sample size
        as argument
        default: np.random.randn
        TODO: change to size argument


    Returns
    -------
    acovf : array
        autocovariance of ARMA process given by ar, ma


    '''
    eta = sigma * distrvs(nsample)
    return signal.lfilter(ma, ar, eta)

def arma_acovf(ar, ma, nobs=10):
    '''theoretical autocovariance function of ARMA process

    Parameters
    ----------
    ar : array_like, 1d
        coefficient for autoregressive lag polynomial, including zero lag
    ma : array_like, 1d
        coefficient for moving-average lag polynomial, including zero lag

    Returns
    -------
    acovf : array
        autocovariance of ARMA process given by ar, ma

    See Also
    --------
    arma_acf
    acovf


    Notes
    -----
    tries to do some crude numerical speed improvements for cases
    with high persistance.
    '''
    #increase length of impulse response for AR closer to 1
    #maybe cheap/fast enough to always keep nobs for ir large
    if np.abs(np.sum(ar)-1) > 0.9:
        nobs_ir = 1000
    else:
        nobs_ir = 100
    ir = arma_impulse_response(ar, ma, nobs=nobs_ir)
    #better save than sorry (?), I have no idea about the required precision
    #only checked for AR(1)
    while ir[-1] > 5*1e-5:
        nobs *= 10
        ir = arma_impulse_response(ar, ma, nobs=nobs)
    #again no idea where the speed break points are:
    if nobs_ir > 50000 and nobs < 1001:
        acovf = np.array([np.dot(ir[:nobs-t], ir[t:nobs]) for t in range(10)])
    else:
        acovf = np.correlate(ir,ir,'full')[len(ir)-1:]
    return acovf[:nobs]

def arma_acf(ar, ma, nobs=10):
    '''theoretical autocovariance function of ARMA process

    Parameters
    ----------
    ar : array_like, 1d
        coefficient for autoregressive lag polynomial, including zero lag
    ma : array_like, 1d
        coefficient for moving-average lag polynomial, including zero lag

    Returns
    -------
    acovf : array
        autocovariance of ARMA process given by ar, ma


    See Also
    --------
    arma_acovf
    acf
    acovf

    '''
    acovf = arma_acovf(ar, ma, nobs)
    return acovf/acovf[0]

def arma_pacf(ar, ma, nobs=10):
    '''partial autocorrelation function of an ARMA process

    Notes
    -----
    solves yule-walker equation for each lag order up to nobs lags

    not tested/checked yet
    '''
    apacf = np.zeros(nobs)
    acov = tsa.arma_acf(ar,ma)

    apacf[0] = 1.
    for k in range(2,nobs+1):
        r = acov[:k];
        apacf[k] = np.linalg.solve(scipy.linalg.toeplitz(r[:-1]), r[1:])[-1]


def arma_impulse_response(ar, ma, nobs=100):
    '''get the impulse response function (MA representation) for ARMA process

    Parameters
    ----------
    ma : array_like, 1d
        moving average lag polynomial
    ar : array_like, 1d
        auto regressive lag polynomial
    nobs : int
        number of observations to calculate

    Returns
    -------
    ir : array, 1d
        impulse response function with nobs elements
    `

    Notes
    -----
    This is the same as finding the MA representation of an ARMA(p,q).
    By reversing the role of ar and ma in the function arguments, the
    returned result is the AR representation of an ARMA(p,q), i.e

    ma_representation = arma_impulse_response(ar, ma, nobs=100)
    ar_representation = arma_impulse_response(ma, ar, nobs=100)

    fully tested against matlab

    Examples
    --------
    AR(1)
    >>> arma_impulse_response([1.0, -0.8], [1.], nobs=10)
    array([ 1.        ,  0.8       ,  0.64      ,  0.512     ,  0.4096    ,
            0.32768   ,  0.262144  ,  0.2097152 ,  0.16777216,  0.13421773])

    this is the same as
    >>> 0.8**np.arange(10)
    array([ 1.        ,  0.8       ,  0.64      ,  0.512     ,  0.4096    ,
            0.32768   ,  0.262144  ,  0.2097152 ,  0.16777216,  0.13421773])

    MA(2)
    >>> arma_impulse_response([1.0], [1., 0.5, 0.2], nobs=10)
    array([ 1. ,  0.5,  0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ])

    ARMA(1,2)
    >>> arma_impulse_response([1.0, -0.8], [1., 0.5, 0.2], nobs=10)
    array([ 1.        ,  1.3       ,  1.24      ,  0.992     ,  0.7936    ,
            0.63488   ,  0.507904  ,  0.4063232 ,  0.32505856,  0.26004685])


    '''
    impulse = np.zeros(nobs)
    impulse[0] = 1.
    return signal.lfilter(ma, ar, impulse)

#alias, easier to remember
arma2ma = arma_impulse_response

#alias, easier to remember
def arma2ar(ar, ma, nobs=100):
    '''get the AR representation of an ARMA process

    Parameters
    ----------
    ar : array_like, 1d
        auto regressive lag polynomial
    ma : array_like, 1d
        moving average lag polynomial
    nobs : int
        number of observations to calculate

    Returns
    -------
    ar : array, 1d
        coefficients of AR lag polynomial with nobs elements
    `

    Notes
    -----
    This is just an alias for

    ``ar_representation = arma_impulse_response(ma, ar, nobs=100)``

    fully tested against matlab

    Examples
    --------

    '''
    return arma_impulse_response(ma, ar, nobs=100)

def lpol2index(ar):
    '''remove zeros from lagpolynomial, squeezed representation with index

    Parameters
    ----------
    ar : array_like
        coefficients of lag polynomial

    Returns
    -------
    coeffs : array
        non-zero coefficients of lag polynomial
    index : array
        index (lags) of lagpolynomial with non-zero elements
    '''

    index = np.nonzero(ar)[0]
    coeffs = ar[index]
    return coeffs, index

def index2lpol(coeffs, index):
    '''expand coefficients to lag poly

    Parameters
    ----------
    coeffs : array
        non-zero coefficients of lag polynomial
    index : array
        index (lags) of lagpolynomial with non-zero elements
    ar : array_like
        coefficients of lag polynomial

    Returns
    -------
    ar : array_like
        coefficients of lag polynomial

    '''
    n = max(index)
    ar = np.zeros(n)
    ar[index] = coeffs
    return ar


def deconvolve(num, den, n=None):
    """Deconvolves divisor out of signal, division of polynomials for n terms

    calculates den^{-1} * num

    Parameters
    ----------
    num : array_like
        signal or lag polynomial
    denom : array_like
        coefficients of lag polynomial (linear filter)
    n : None or int
        number of terms of quotient

    Returns
    -------
    quot : array
        quotient or filtered series
    rem : array
        remainder

    Notes
    -----
    If num is a time series, then this applies the linear filter den^{-1}.
    If both num and den are both lagpolynomials, then this calculates the
    quotient polynomial for n terms and also returns the remainder.

    This is copied from scipy.signal.signaltools and added n as optional
    parameter.

    """
    num = np.atleast_1d(num)
    den = np.atleast_1d(den)
    N = len(num)
    D = len(den)
    if D > N and n is None:
        quot = [];
        rem = num;
    else:
        if n is None:
            n = N-D+1
        input = np.zeros(n, float)
        input[0] = 1
        quot = signal.lfilter(num, den, input)
        num_approx = signal.convolve(den, quot, mode='full')
        if len(num) < len(num_approx):  # 1d only ?
            num = np.concatenate((num, np.zeros(len(num_approx)-len(num))))
        rem = num - num_approx
    return quot, rem



def mcarma22(niter=10):
    '''run Monte Carlo for ARMA(2,2)

    DGP parameters currently hard coded
    also sample size `nsample`

    was not a self contained function, used instances from outer scope
      now corrected

    '''
    nsample = 1000
    #ar = [1.0, 0, 0]
    ar = [1.0, -0.75, -0.1]
    #ma = [1.0, 0, 0]
    ma = [1.0,  0.3,  0.2]
    results = []
    results_bse = []
    arest = ARIMA()
    arest2 = ARIMA()
    for _ in range(niter):
        y2 = arma_generate_sample(ar,ma,nsample,0.1)
        rhohat2a, cov_x2a, infodict, mesg, ier = arest2.fit(y2,2,2)
        results.append(rhohat2a)
        err2a = arest2.errfn(x=y2)
        sige2a = np.sqrt(np.dot(err2a,err2a)/nsample)
        results_bse.append(sige2a * np.sqrt(np.diag(cov_x2a)))
    return np.r_[ar[1:], ma[1:]], np.array(results), np.array(results_bse)


__all__ = ['ARIMA', 'arma_acf', 'arma_acovf', 'arma_generate_sample',
           'arma_impulse_response', 'arma2ar', 'arma2ma', 'deconvolve',
           'lpol2index', 'index2lpol']


if __name__ == '__main__':
    print mcarma22()


    # Simulate AR(1)
    #--------------
    # ar * y = ma * eta
    ar = [1, -0.8]
    ma = [1.0]

    # generate AR data
    eta = 0.1 * np.random.randn(1000)
    yar1 = signal.lfilter(ar, ma, eta)

    print "\nExample 0"
    arest = ARIMA()
    rhohat, cov_x, infodict, mesg, ier = arest.fit(yar1,1,1)
    print rhohat
    print cov_x

    print "\nExample 1"
    ar = [1.0,  -0.8]
    ma = [1.0,  0.5]
    y1 = arest.generate_sample(ar,ma,1000,0.1)
    rhohat1, cov_x1, infodict, mesg, ier = arest.fit(y1,1,1)
    print rhohat1
    print cov_x1
    err1 = arest.errfn(x=y1)
    print np.var(err1)
    import scikits.statsmodels as sm
    print sm.regression.yule_walker(y1, order=2, inv=True)

    print "\nExample 2"
    arest2 = ARIMA()
    nsample = 1000
    ar = [1.0, -0.6, -0.1]
    ma = [1.0,  0.3,  0.2]
    y2 = arest2.generate_sample(ar,ma,nsample,0.1)
    rhohat2, cov_x2, infodict, mesg, ier = arest2.fit(y2,1,2)
    print rhohat2
    print cov_x2
    err2 = arest.errfn(x=y2)
    print np.var(err2)
    print arest2.rhoy
    print arest2.rhoe
    print "true"
    print ar
    print ma
    rhohat2a, cov_x2a, infodict, mesg, ier = arest2.fit(y2,2,2)
    print rhohat2a
    print cov_x2a
    err2a = arest.errfn(x=y2)
    print np.var(err2a)
    print arest2.rhoy
    print arest2.rhoe
    print "true"
    print ar
    print ma

    print sm.regression.yule_walker(y2, order=2, inv=True)

    print "\nExample 20"
    arest20 = ARIMA()
    nsample = 1000
    ar = [1.0]#, -0.8, -0.4]
    ma = [1.0,  0.5,  0.2]
    y3 = arest20.generate_sample(ar,ma,nsample,0.01)
    rhohat3, cov_x3, infodict, mesg, ier = arest20.fit(y3,2,0)
    print rhohat3
    print cov_x3
    err3 = arest20.errfn(x=y3)
    print np.var(err3)
    print np.sqrt(np.dot(err3,err3)/nsample)
    print arest20.rhoy
    print arest20.rhoe
    print "true"
    print ar
    print ma

    rhohat3a, cov_x3a, infodict, mesg, ier = arest20.fit(y3,0,2)
    print rhohat3a
    print cov_x3a
    err3a = arest20.errfn(x=y3)
    print np.var(err3a)
    print np.sqrt(np.dot(err3a,err3a)/nsample)
    print arest20.rhoy
    print arest20.rhoe
    print "true"
    print ar
    print ma

    print sm.regression.yule_walker(y3, order=2, inv=True)

    print "\nExample 02"
    arest02 = ARIMA()
    nsample = 1000
    ar = [1.0, -0.8, 0.4] #-0.8, -0.4]
    ma = [1.0]#,  0.8,  0.4]
    y4 = arest02.generate_sample(ar,ma,nsample)
    rhohat4, cov_x4, infodict, mesg, ier = arest02.fit(y4,2,0)
    print rhohat4
    print cov_x4
    err4 = arest02.errfn(x=y4)
    print np.var(err4)
    sige = np.sqrt(np.dot(err4,err4)/nsample)
    print sige
    print sige * np.sqrt(np.diag(cov_x4))
    print np.sqrt(np.diag(cov_x4))
    print arest02.rhoy
    print arest02.rhoe
    print "true"
    print ar
    print ma

    rhohat4a, cov_x4a, infodict, mesg, ier = arest02.fit(y4,0,2)
    print rhohat4a
    print cov_x4a
    err4a = arest02.errfn(x=y4)
    print np.var(err4a)
    sige = np.sqrt(np.dot(err4a,err4a)/nsample)
    print sige
    print sige * np.sqrt(np.diag(cov_x4a))
    print np.sqrt(np.diag(cov_x4a))
    print arest02.rhoy
    print arest02.rhoe
    print "true"
    print ar
    print ma
    import scikits.statsmodels as sm
    print sm.regression.yule_walker(y4, order=2, method='mle', inv=True)

    def mc_summary(res, rt=None):
        if rt is None:
            rt = np.zeros(res.shape[1])
        print 'RMSE'
        print np.sqrt(((res-rt)**2).mean(0))
        print 'mean bias'
        print (res-rt).mean(0)
        print 'median bias'
        print np.median((res-rt),0)
        print 'median bias percent'
        print np.median((res-rt)/rt*100,0)
        print 'median absolute error'
        print np.median(np.abs(res-rt),0)
        print 'positive error fraction'
        print (res > rt).mean(0)

    run_mc = True#False
    if run_mc:
        import time
        t0 = time.time()
        rt, res_rho, res_bse = mcarma22(niter=1000)
        print 'elapsed time for Monte Carlo', time.time()-t0
        # 20 seconds for ARMA(2,2), 1000 iterations with 1000 observations
        sige2a = np.sqrt(np.dot(err2a,err2a)/nsample)
        print '\nbse of one sample'
        print sige2a * np.sqrt(np.diag(cov_x2a))
        print '\nMC of rho versus true'
        mc_summary(res_rho, rt)
        print '\nMC of bse versus zero'
        mc_summary(res_bse)
        print '\nMC of bse versus std'
        mc_summary(res_bse, res_rho.std(0))

    import matplotlib.pyplot as plt
    plt.plot(arest2.forecast()[-100:])
    #plt.show()

    ar1, ar2 = ([1, -0.4], [1, 0.5])
    ar2 = [1, -1]
    lagpolyproduct = np.convolve(ar1, ar2)
    print deconvolve(lagpolyproduct, ar2, n=None)
    print signal.deconvolve(lagpolyproduct, ar2)
    print deconvolve(lagpolyproduct, ar2, n=10)

