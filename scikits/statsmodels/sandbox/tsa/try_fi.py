
'''
using lfilter to get fractional integration polynomial (1-L)^d, d<1
`ri` is (1-L)^(-d), d<1
'''

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.special import gamma, gammaln
from scipy import signal, optimize

from scikits.statsmodels.sandbox import tsa


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
    j = np.arange(n)
    return np.exp(gammaln(d+j) - gammaln(j+1) - gammaln(d))

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
    j = np.arange(n)
    ar = - np.exp(gammaln(-d+j) - gammaln(j+1) - gammaln(-d))
    ar[0] = 1
    return ar

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
        ar_approx = tsa.arma_impulse_response(ma, ar,  n)
##        print ar,ma
##        print ar_des.shape, ar_approx.shape
##        print ar_des
##        print ar_approx
        return (ar_des - ar_approx) #((ar - ar_approx)**2).sum()
    if start is None:
        arma0 = np.r_[-0.9* np.ones(p-1), np.zeros(q-1)]
    else:
        arma0 = start
    res = optimize.leastsq(msear_err, arma0, ar_des, maxfev=5000)#, full_output=True)
    #print res
    arma_app = np.atleast_1d(res[0])
    ar_app = np.r_[1, arma_app[:p-1]],
    ma_app = np.r_[1, arma_app[p-1:]]
    return ar_app, ma_app, res



def test_fi():
    #test identity of ma and ar representation of fi lag polynomial
    n = 100
    mafromar = tsa.arma_impulse_response(lpol_fiar(0.4, n=n), [1], n)
    assert_array_almost_equal(mafromar, lpol_fima(0.4, n=n), 13)




d = 0.4
n = 1000
j = np.arange(n*10)
ri0 = gamma(d+j)/(gamma(j+1)*gamma(d))
#ri = np.exp(gammaln(d+j) - gammaln(j+1) - gammaln(d))   (d not -d)
ri = lpol_fima(d, n=n)  # get_ficoefs(d, n=n) old naming?
riinv = signal.lfilter([1], ri, [1]+[0]*(n-1))#[[5,10,20,25]]
'''
array([-0.029952  , -0.01100641, -0.00410998, -0.00299859])
>>> d=0.4; j=np.arange(1000);ri=gamma(d+j)/(gamma(j+1)*gamma(d))
>>> # (1-L)^d, d<1 is
>>> lfilter([1], ri, [1]+[0]*30)
array([ 1.        , -0.4       , -0.12      , -0.064     , -0.0416    ,
      -0.029952  , -0.0229632 , -0.01837056, -0.01515571, -0.01279816,
      -0.01100641, -0.0096056 , -0.00848495, -0.00757118, -0.00681406,
      -0.00617808, -0.0056375 , -0.00517324, -0.00477087, -0.00441934,
      -0.00410998, -0.00383598, -0.00359188, -0.00337324, -0.00317647,
      -0.00299859, -0.00283712, -0.00269001, -0.00255551, -0.00243214,
      -0.00231864])
>>> # verified for points [[5,10,20,25]] at 4 decimals with Bhardwaj, Swanson, Journal of Eonometrics 2006
'''
print lpol_fiar(0.4, n=20)
print lpol_fima(-0.4, n=20)
print np.sum((lpol_fima(-0.4, n=n)[1:] + riinv[1:])**2) #different signs
print np.sum((lpol_fiar(0.4, n=n)[1:] - riinv[1:])**2) #corrected signs
test_fi()

ar_true = [1, -0.4]
ma_true = [1, 0.5]


ar_desired = tsa.arma_impulse_response(ma_true, ar_true)
ar_app, ma_app, res = ar2arma(ar_desired, 2,1, n=100, mse='ar', start = [0.1])
print ar_app, ma_app
ar_app, ma_app, res = ar2arma(ar_desired, 2,2, n=100, mse='ar', start = [-0.1, 0.1])
print ar_app, ma_app
ar_app, ma_app, res = ar2arma(ar_desired, 2,3, n=100, mse='ar')#, start = [-0.1, 0.1])
print ar_app, ma_app

slow = 0
if slow:
    ar_desired = lpol_fiar(0.4, n=100)
    ar_app, ma_app, res = ar2arma(ar_desired, (3,1), n=100, mse='ar')#, start = [-0.1, 0.1])
    print ar_app, ma_app
    ar_app, ma_app, res = ar2arma(ar_desired, (10,10), n=100, mse='ar')#, start = [-0.1, 0.1])
    print ar_app, ma_app



