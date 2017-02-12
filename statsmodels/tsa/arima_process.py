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
from __future__ import print_function
from statsmodels.compat.python import range
import numpy as np
from scipy import signal, optimize, linalg


def arma_generate_sample(ar, ma, nsample, sigma=1, distrvs=np.random.randn,
                         burnin=0):
    """
    Generate a random sample of an ARMA process

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
    burnin : integer (default: 0)
        to reduce the effect of initial conditions, burnin observations at the
        beginning of the sample are dropped

    Returns
    -------
    sample : array
        sample of ARMA process given by ar, ma of length nsample

    Notes
    -----
    As mentioned above, both the AR and MA components should include the
    coefficient on the zero-lag. This is typically 1. Further, due to the
    conventions used in signal processing used in signal.lfilter vs.
    conventions in statistics for ARMA processes, the AR paramters should
    have the opposite sign of what you might expect. See the examples below.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> ar = np.r_[1, -arparams] # add zero-lag and negate
    >>> ma = np.r_[1, maparams] # add zero-lag
    >>> y = sm.tsa.arma_generate_sample(ar, ma, 250)
    >>> model = sm.tsa.ARMA(y, (2, 2)).fit(trend='nc', disp=0)
    >>> model.params
    array([ 0.79044189, -0.23140636,  0.70072904,  0.40608028])
    """
    #TODO: unify with ArmaProcess method
    eta = sigma * distrvs(nsample+burnin)
    return signal.lfilter(ma, ar, eta)[burnin:]


def arma_acovf(ar, ma, nobs=10):
    '''theoretical autocovariance function of ARMA process

    Parameters
    ----------
    ar : array_like, 1d
        coefficient for autoregressive lag polynomial, including zero lag
    ma : array_like, 1d
        coefficient for moving-average lag polynomial, including zero lag
    nobs : int
        number of terms (lags plus zero lag) to include in returned acovf

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
    Tries to do some crude numerical speed improvements for cases
    with high persistance. However, this algorithm is slow if the process is
    highly persistent and only a few autocovariances are desired.
    '''
    #increase length of impulse response for AR closer to 1
    #maybe cheap/fast enough to always keep nobs for ir large
    if np.abs(np.sum(ar)-1) > 0.9:
        nobs_ir = max(1000, 2 * nobs)  # no idea right now how large is needed
    else:
        nobs_ir = max(100, 2 * nobs)   # no idea right now
    ir = arma_impulse_response(ar, ma, nobs=nobs_ir)
    #better save than sorry (?), I have no idea about the required precision
    #only checked for AR(1)
    while ir[-1] > 5*1e-5:
        nobs_ir *= 10
        ir = arma_impulse_response(ar, ma, nobs=nobs_ir)
    #again no idea where the speed break points are:
    if nobs_ir > 50000 and nobs < 1001:
        acovf = np.array([np.dot(ir[:nobs-t], ir[t:nobs])
                          for t in range(nobs)])
    else:
        acovf = np.correlate(ir, ir, 'full')[len(ir)-1:]
    return acovf[:nobs]


def arma_acf(ar, ma, nobs=10):
    '''theoretical autocorrelation function of an ARMA process

    Parameters
    ----------
    ar : array_like, 1d
        coefficient for autoregressive lag polynomial, including zero lag
    ma : array_like, 1d
        coefficient for moving-average lag polynomial, including zero lag
    nobs : int
        number of terms (lags plus zero lag) to include in returned acf

    Returns
    -------
    acf : array
        autocorrelation of ARMA process given by ar, ma


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

    Parameters
    ----------
    ar : array_like, 1d
        coefficient for autoregressive lag polynomial, including zero lag
    ma : array_like, 1d
        coefficient for moving-average lag polynomial, including zero lag
    nobs : int
        number of terms (lags plus zero lag) to include in returned pacf

    Returns
    -------
    pacf : array
        partial autocorrelation of ARMA process given by ar, ma

    Notes
    -----
    solves yule-walker equation for each lag order up to nobs lags

    not tested/checked yet
    '''
    apacf = np.zeros(nobs)
    acov = arma_acf(ar, ma, nobs=nobs+1)

    apacf[0] = 1.
    for k in range(2, nobs+1):
        r = acov[:k]
        apacf[k-1] = linalg.solve(linalg.toeplitz(r[:-1]), r[1:])[-1]
    return apacf


def arma_periodogram(ar, ma, worN=None, whole=0):
    '''periodogram for ARMA process given by lag-polynomials ar and ma

    Parameters
    ----------
    ar : array_like
        autoregressive lag-polynomial with leading 1 and lhs sign
    ma : array_like
        moving average lag-polynomial with leading 1
    worN : {None, int}, optional
        option for scipy.signal.freqz   (read "w or N")
        If None, then compute at 512 frequencies around the unit circle.
        If a single integer, the compute at that many frequencies.
        Otherwise, compute the response at frequencies given in worN
    whole : {0,1}, optional
        options for scipy.signal.freqz
        Normally, frequencies are computed from 0 to pi (upper-half of
        unit-circle.  If whole is non-zero compute frequencies from 0 to 2*pi.

    Returns
    -------
    w : array
        frequencies
    sd : array
        periodogram, spectral density

    Notes
    -----
    Normalization ?

    This uses signal.freqz, which does not use fft. There is a fft version
    somewhere.

    '''
    w, h = signal.freqz(ma, ar, worN=worN, whole=whole)
    sd = np.abs(h)**2/np.sqrt(2*np.pi)
    if np.sum(np.isnan(h)) > 0:
        # this happens with unit root or seasonal unit root'
        print('Warning: nan in frequency response h, maybe a unit root')
    return w, sd


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

    Notes
    -----
    This is just an alias for
    ``ar_representation = arma_impulse_response(ma, ar, nobs=100)`` which has
    been fully tested against MATLAB.

    Examples
    --------

    '''
    return arma_impulse_response(ma, ar, nobs=nobs)


#moved from sandbox.tsa.try_fi
def ar2arma(ar_des, p, q, n=20, mse='ar', start=None):
    '''find arma approximation to ar process

    This finds the ARMA(p,q) coefficients that minimize the integrated
    squared difference between the impulse_response functions
    (MA representation) of the AR and the ARMA process. This does
    currently not check whether the MA lagpolynomial of the ARMA
    process is invertible, neither does it check the roots of the AR
    lagpolynomial.

    Parameters
    ----------
    ar_des : array_like
        coefficients of original AR lag polynomial, including lag zero
    p, q : int
        length of desired ARMA lag polynomials
    n : int
        number of terms of the impuls_response function to include in the
        objective function for the approximation
    mse : string, 'ar'
        not used yet,

    Returns
    -------
    ar_app, ma_app : arrays
        coefficients of the AR and MA lag polynomials of the approximation
    res : tuple
        result of optimize.leastsq

    Notes
    -----
    Extension is possible if we want to match autocovariance instead
    of impulse response function.

    TODO: convert MA lag polynomial, ma_app, to be invertible, by mirroring
    roots outside the unit intervall to ones that are inside. How do we do
    this?

    '''
    #p,q = pq
    def msear_err(arma, ar_des):
        ar, ma = np.r_[1, arma[:p-1]], np.r_[1, arma[p-1:]]
        ar_approx = arma_impulse_response(ma, ar, n)
##        print(ar,ma)
##        print(ar_des.shape, ar_approx.shape)
##        print(ar_des)
##        print(ar_approx)
        return (ar_des - ar_approx)  # ((ar - ar_approx)**2).sum()
    if start is None:
        arma0 = np.r_[-0.9 * np.ones(p-1), np.zeros(q-1)]
    else:
        arma0 = start
    res = optimize.leastsq(msear_err, arma0, ar_des, maxfev=5000)
    #print(res)
    arma_app = np.atleast_1d(res[0])
    ar_app = np.r_[1, arma_app[:p-1]],
    ma_app = np.r_[1, arma_app[p-1:]]
    return ar_app, ma_app, res


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
    ar = np.asarray(ar)
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


#moved from sandbox.tsa.try_fi
def lpol_fima(d, n=20):
    '''MA representation of fractional integration

    .. math:: (1-L)^{-d} for |d|<0.5  or |d|<1 (?)

    Parameters
    ----------
    d : float
        fractional power
    n : int
        number of terms to calculate, including lag zero

    Returns
    -------
    ma : array
        coefficients of lag polynomial

    '''
    #hide import inside function until we use this heavily
    from scipy.special import gammaln
    j = np.arange(n)
    return np.exp(gammaln(d+j) - gammaln(j+1) - gammaln(d))


#moved from sandbox.tsa.try_fi
def lpol_fiar(d, n=20):
    '''AR representation of fractional integration

    .. math:: (1-L)^{d} for |d|<0.5  or |d|<1 (?)

    Parameters
    ----------
    d : float
        fractional power
    n : int
        number of terms to calculate, including lag zero

    Returns
    -------
    ar : array
        coefficients of lag polynomial

    Notes:
    first coefficient is 1, negative signs except for first term,
    ar(L)*x_t
    '''
    #hide import inside function until we use this heavily
    from scipy.special import gammaln
    j = np.arange(n)
    ar = - np.exp(gammaln(-d+j) - gammaln(j+1) - gammaln(-d))
    ar[0] = 1
    return ar


#moved from sandbox.tsa.try_fi
def lpol_sdiff(s):
    '''return coefficients for seasonal difference (1-L^s)

    just a trivial convenience function

    Parameters
    ----------
    s : int
        number of periods in season

    Returns
    -------
    sdiff : list, length s+1

    '''
    return [1] + [0]*(s-1) + [-1]


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
        quot = []
        rem = num
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


class ArmaProcess(object):
    """
    Represent an ARMA process for given lag-polynomials

    This is a class to bring together properties of the process.
    It does not do any estimation or statistical analysis.

    Parameters
    ----------
    ar : array_like, 1d
        Coefficient for autoregressive lag polynomial, including zero lag.
        See the notes for some information about the sign.
    ma : array_like, 1d
        Coefficient for moving-average lag polynomial, including zero lag
    nobs : int, optional
        Length of simulated time series. Used, for example, if a sample is
        generated. See example.

    Notes
    -----
    As mentioned above, both the AR and MA components should include the
    coefficient on the zero-lag. This is typically 1. Further, due to the
    conventions used in signal processing used in signal.lfilter vs.
    conventions in statistics for ARMA processes, the AR paramters should
    have the opposite sign of what you might expect. See the examples below.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> ar = np.r_[1, -ar] # add zero-lag and negate
    >>> ma = np.r_[1, ma] # add zero-lag
    >>> arma_process = sm.tsa.ArmaProcess(ar, ma)
    >>> arma_process.isstationary
    True
    >>> arma_process.isinvertible
    True
    >>> y = arma_process.generate_sample(250)
    >>> model = sm.tsa.ARMA(y, (2, 2)).fit(trend='nc', disp=0)
    >>> model.params
    array([ 0.79044189, -0.23140636,  0.70072904,  0.40608028])
    """
    # maybe needs special handling for unit roots
    def __init__(self, ar, ma, nobs=100):
        self.ar = np.asarray(ar)
        self.ma = np.asarray(ma)
        self.arcoefs = -self.ar[1:]
        self.macoefs = self.ma[1:]
        self.arpoly = np.polynomial.Polynomial(self.ar)
        self.mapoly = np.polynomial.Polynomial(self.ma)
        self.nobs = nobs

    @classmethod
    def from_coeffs(cls, arcoefs, macoefs, nobs=100):
        """
        Create ArmaProcess instance from coefficients of the lag-polynomials

        Parameters
        ----------
        arcoefs : array-like
            Coefficient for autoregressive lag polynomial, not including zero
            lag. The sign is inverted to conform to the usual time series
            representation of an ARMA process in statistics. See the class
            docstring for more information.
        macoefs : array-like
            Coefficient for moving-average lag polynomial, including zero lag
        nobs : int, optional
            Length of simulated time series. Used, for example, if a sample
            is generated.
        """
        return cls(np.r_[1, -arcoefs], np.r_[1, macoefs], nobs=nobs)

    @classmethod
    def from_estimation(cls, model_results, nobs=None):
        """
        Create ArmaProcess instance from ARMA estimation results

        Parameters
        ----------
        model_results : ARMAResults instance
            A fitted model
        nobs : int, optional
            If None, nobs is taken from the results
        """
        arcoefs = model_results.arparams
        macoefs = model_results.maparams
        nobs = nobs or model_results.nobs
        return cls(np.r_[1, -arcoefs], np.r_[1, macoefs], nobs=nobs)

    def __mul__(self, oth):
        if isinstance(oth, self.__class__):
            ar = (self.arpoly * oth.arpoly).coef
            ma = (self.mapoly * oth.mapoly).coef
        else:
            try:
                aroth, maoth = oth
                arpolyoth = np.polynomial.Polynomial(aroth)
                mapolyoth = np.polynomial.Polynomial(maoth)
                ar = (self.arpoly * arpolyoth).coef
                ma = (self.mapoly * mapolyoth).coef
            except:
                print('other is not a valid type')
                raise
        return self.__class__(ar, ma, nobs=self.nobs)

    def __repr__(self):
        return 'ArmaProcess(%r, %r, nobs=%d)' % (self.ar.tolist(),
                                                 self.ma.tolist(),
                                                 self.nobs)

    def __str__(self):
        return 'ArmaProcess\nAR: %r\nMA: %r' % (self.ar.tolist(),
                                                self.ma.tolist())

    def acovf(self, nobs=None):
        nobs = nobs or self.nobs
        return arma_acovf(self.ar, self.ma, nobs=nobs)

    acovf.__doc__ = arma_acovf.__doc__

    def acf(self, nobs=None):
        nobs = nobs or self.nobs
        return arma_acf(self.ar, self.ma, nobs=nobs)

    acf.__doc__ = arma_acf.__doc__

    def pacf(self, nobs=None):
        nobs = nobs or self.nobs
        return arma_pacf(self.ar, self.ma, nobs=nobs)

    pacf.__doc__ = arma_pacf.__doc__

    def periodogram(self, nobs=None):
        nobs = nobs or self.nobs
        return arma_periodogram(self.ar, self.ma, worN=nobs)

    periodogram.__doc__ = arma_periodogram.__doc__

    def impulse_response(self, nobs=None):
        nobs = nobs or self.nobs
        return arma_impulse_response(self.ar, self.ma, worN=nobs)

    impulse_response.__doc__ = arma_impulse_response.__doc__

    def arma2ma(self, nobs=None):
        nobs = nobs or self.nobs
        return arma2ma(self.ar, self.ma, nobs=nobs)

    arma2ma.__doc__ = arma2ma.__doc__

    def arma2ar(self, nobs=None):
        nobs = nobs or self.nobs
        return arma2ar(self.ar, self.ma, nobs=nobs)

    arma2ar.__doc__ = arma2ar.__doc__

    @property
    def arroots(self):
        """
        Roots of autoregressive lag-polynomial
        """
        return self.arpoly.roots()

    @property
    def maroots(self):
        """
        Roots of moving average lag-polynomial
        """
        return self.mapoly.roots()

    @property
    def isstationary(self):
        '''Arma process is stationary if AR roots are outside unit circle

        Returns
        -------
        isstationary : boolean
             True if autoregressive roots are outside unit circle
        '''
        if np.all(np.abs(self.arroots) > 1):
            return True
        else:
            return False

    @property
    def isinvertible(self):
        '''Arma process is invertible if MA roots are outside unit circle

        Returns
        -------
        isinvertible : boolean
             True if moving average roots are outside unit circle
        '''
        if np.all(np.abs(self.maroots) > 1):
            return True
        else:
            return False

    def invertroots(self, retnew=False):
        '''make MA polynomial invertible by inverting roots inside unit circle

        Parameters
        ----------
        retnew : boolean
            If False (default), then return the lag-polynomial as array.
            If True, then return a new instance with invertible MA-polynomial

        Returns
        -------
        manew : array
           new invertible MA lag-polynomial, returned if retnew is false.
        wasinvertible : boolean
           True if the MA lag-polynomial was already invertible, returned if
           retnew is false.
        armaprocess : new instance of class
           If retnew is true, then return a new instance with invertible
           MA-polynomial
        '''
        #TODO: variable returns like this?
        pr = self.ma_roots()
        insideroots = np.abs(pr) < 1
        if insideroots.any():
            pr[np.abs(pr) < 1] = 1./pr[np.abs(pr) < 1]
            pnew = np.polynomial.Polynomial.fromroots(pr)
            mainv = pnew.coef/pnew.coef[0]
            wasinvertible = False
        else:
            mainv = self.ma
            wasinvertible = True
        if retnew:
            return self.__class__(self.ar, mainv, nobs=self.nobs)
        else:
            return mainv, wasinvertible

    def generate_sample(self, nsample=100, scale=1., distrvs=None, axis=0,
                        burnin=0):
        '''generate ARMA samples

        Parameters
        ----------
        nsample : int or tuple of ints
            If nsample is an integer, then this creates a 1d timeseries of
            length size. If nsample is a tuple, then the timeseries is along
            axis. All other axis have independent arma samples.
        scale : float
            standard deviation of noise
        distrvs : function, random number generator
            function that generates the random numbers, and takes sample size
            as argument
            default: np.random.randn
            TODO: change to size argument
        burnin : integer (default: 0)
            to reduce the effect of initial conditions, burnin observations
            at the beginning of the sample are dropped
        axis : int
            See nsample.

        Returns
        -------
        rvs : ndarray
            random sample(s) of arma process

        Notes
        -----
        Should work for n-dimensional with time series along axis, but not
        tested yet. Processes are sampled independently.
        '''
        if distrvs is None:
            distrvs = np.random.normal
        if np.ndim(nsample) == 0:
                nsample = [nsample]
        if burnin:
            #handle burin time for nd arrays
            #maybe there is a better trick in scipy.fft code
            newsize = list(nsample)
            newsize[axis] += burnin
            newsize = tuple(newsize)
            fslice = [slice(None)]*len(newsize)
            fslice[axis] = slice(burnin, None, None)
            fslice = tuple(fslice)
        else:
            newsize = tuple(nsample)
            fslice = tuple([slice(None)]*np.ndim(newsize))

        eta = scale * distrvs(size=newsize)
        return signal.lfilter(self.ma, self.ar, eta, axis=axis)[fslice]


__all__ = ['arma_acf', 'arma_acovf', 'arma_generate_sample',
           'arma_impulse_response', 'arma2ar', 'arma2ma', 'deconvolve',
           'lpol2index', 'index2lpol']


if __name__ == '__main__':


    # Simulate AR(1)
    #--------------
    # ar * y = ma * eta
    ar = [1, -0.8]
    ma = [1.0]

    # generate AR data
    eta = 0.1 * np.random.randn(1000)
    yar1 = signal.lfilter(ar, ma, eta)

    print("\nExample 0")
    arest = ARIMAProcess(yar1)
    rhohat, cov_x, infodict, mesg, ier = arest.fit((1,0,1))
    print(rhohat)
    print(cov_x)

    print("\nExample 1")
    ar = [1.0,  -0.8]
    ma = [1.0,  0.5]
    y1 = arest.generate_sample(ar,ma,1000,0.1)
    arest = ARIMAProcess(y1)
    rhohat1, cov_x1, infodict, mesg, ier = arest.fit((1,0,1))
    print(rhohat1)
    print(cov_x1)
    err1 = arest.errfn(x=y1)
    print(np.var(err1))
    import statsmodels.api as sm
    print(sm.regression.yule_walker(y1, order=2, inv=True))

    print("\nExample 2")
    nsample = 1000
    ar = [1.0, -0.6, -0.1]
    ma = [1.0,  0.3,  0.2]
    y2 = ARIMA.generate_sample(ar,ma,nsample,0.1)
    arest2 = ARIMAProcess(y2)
    rhohat2, cov_x2, infodict, mesg, ier = arest2.fit((1,0,2))
    print(rhohat2)
    print(cov_x2)
    err2 = arest.errfn(x=y2)
    print(np.var(err2))
    print(arest2.rhoy)
    print(arest2.rhoe)
    print("true")
    print(ar)
    print(ma)
    rhohat2a, cov_x2a, infodict, mesg, ier = arest2.fit((2,0,2))
    print(rhohat2a)
    print(cov_x2a)
    err2a = arest.errfn(x=y2)
    print(np.var(err2a))
    print(arest2.rhoy)
    print(arest2.rhoe)
    print("true")
    print(ar)
    print(ma)

    print(sm.regression.yule_walker(y2, order=2, inv=True))

    print("\nExample 20")
    nsample = 1000
    ar = [1.0]#, -0.8, -0.4]
    ma = [1.0,  0.5,  0.2]
    y3 = ARIMA.generate_sample(ar,ma,nsample,0.01)
    arest20 = ARIMAProcess(y3)
    rhohat3, cov_x3, infodict, mesg, ier = arest20.fit((2,0,0))
    print(rhohat3)
    print(cov_x3)
    err3 = arest20.errfn(x=y3)
    print(np.var(err3))
    print(np.sqrt(np.dot(err3,err3)/nsample))
    print(arest20.rhoy)
    print(arest20.rhoe)
    print("true")
    print(ar)
    print(ma)

    rhohat3a, cov_x3a, infodict, mesg, ier = arest20.fit((0,0,2))
    print(rhohat3a)
    print(cov_x3a)
    err3a = arest20.errfn(x=y3)
    print(np.var(err3a))
    print(np.sqrt(np.dot(err3a,err3a)/nsample))
    print(arest20.rhoy)
    print(arest20.rhoe)
    print("true")
    print(ar)
    print(ma)

    print(sm.regression.yule_walker(y3, order=2, inv=True))

    print("\nExample 02")
    nsample = 1000
    ar = [1.0, -0.8, 0.4] #-0.8, -0.4]
    ma = [1.0]#,  0.8,  0.4]
    y4 = ARIMA.generate_sample(ar,ma,nsample)
    arest02 = ARIMAProcess(y4)
    rhohat4, cov_x4, infodict, mesg, ier = arest02.fit((2,0,0))
    print(rhohat4)
    print(cov_x4)
    err4 = arest02.errfn(x=y4)
    print(np.var(err4))
    sige = np.sqrt(np.dot(err4,err4)/nsample)
    print(sige)
    print(sige * np.sqrt(np.diag(cov_x4)))
    print(np.sqrt(np.diag(cov_x4)))
    print(arest02.rhoy)
    print(arest02.rhoe)
    print("true")
    print(ar)
    print(ma)

    rhohat4a, cov_x4a, infodict, mesg, ier = arest02.fit((0,0,2))
    print(rhohat4a)
    print(cov_x4a)
    err4a = arest02.errfn(x=y4)
    print(np.var(err4a))
    sige = np.sqrt(np.dot(err4a,err4a)/nsample)
    print(sige)
    print(sige * np.sqrt(np.diag(cov_x4a)))
    print(np.sqrt(np.diag(cov_x4a)))
    print(arest02.rhoy)
    print(arest02.rhoe)
    print("true")
    print(ar)
    print(ma)
    import statsmodels.api as sm
    print(sm.regression.yule_walker(y4, order=2, method='mle', inv=True))


    import matplotlib.pyplot as plt
    plt.plot(arest2.forecast()[-100:])
    #plt.show()

    ar1, ar2 = ([1, -0.4], [1, 0.5])
    ar2 = [1, -1]
    lagpolyproduct = np.convolve(ar1, ar2)
    print(deconvolve(lagpolyproduct, ar2, n=None))
    print(signal.deconvolve(lagpolyproduct, ar2))
    print(deconvolve(lagpolyproduct, ar2, n=10))

