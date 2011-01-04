'''Multivariate Distribution

Probability of a multivariate t distribution

Now also mvstnormcdf has tests against R mvtnorm

Still need non-central t, extra options, and convenience function for
location, scale version.

Author: Josef Perktold
License: BSD (3-clause)

Reference:
Genz and Bretz for formula

'''
import numpy as np
from scipy import integrate, stats, special
from scipy.stats import chi,chi2

from extras import mvnormcdf, mvstdnormcdf

from numpy import exp as np_exp
from numpy import log as np_log
from scipy.special import gamma as sps_gamma
from scipy.special import gammaln as sps_gammaln

def chi2_pdf(self, x, df):
    '''pdf of chi-square distribution'''
    #from scipy.stats.distributions
    Px = x**(df/2.0-1)*exp(-x/2.0)
    Px /= special.gamma(df/2.0)* 2**(df/2.0)
    return Px

def chi_pdf(x, df):
    tmp = (df-1.)*np_log(x) + (-x*x*0.5) - (df*0.5-1)*np_log(2.0) \
          - sps_gammaln(df*0.5)
    return np_exp(tmp)
    #return x**(df-1.)*np_exp(-x*x*0.5)/(2.0)**(df*0.5-1)/sps_gamma(df*0.5)

def chi_logpdf(x, df):
    tmp = (df-1.)*np_log(x) + (-x*x*0.5) - (df*0.5-1)*np_log(2.0) \
          - sps_gammaln(df*0.5)
    return tmp

def funbgh(s, a, b, R, df):
    sqrt_df = np.sqrt(df+0.5)
    return np.exp(chi_logpdf(s,df) + np.log(mvstdnormcdf(s*a/sqrt_df, s*b/sqrt_df, R,
                                         maxpts=1000000, abseps=1e-6)))

def funbgh2(s, a, b, R, df):
    n = len(a)
    sqrt_df = np.sqrt(df)
    #np.power(s, df-1) * np_exp(-s*s*0.5)
    return  np_exp((df-1)*np_log(s)-s*s*0.5) \
           * mvstdnormcdf(s*a/sqrt_df, s*b/sqrt_df, R[np.tril_indices(n, -1)],
                          maxpts=1000000, abseps=1e-4)

def bghfactor(df):
    return np.power(2.0, 1-df*0.5) / sps_gamma(df*0.5)


def mvstdtprob(a, b, R, df, quadkwds=None, mvstkwds=None):
    '''probability of rectangular area of standard t distribution

    assumes mean is zero and R is correlation matrix

    Notes
    -----
    This function does not calculate the estimate of the combined error
    between the underlying multivariate normal probability calculations
    and the integration.

    '''
    ieps = 1e-5
    res, err = integrate.quad(funbgh2, *chi.ppf([ieps,1-ieps], df),
                          **dict(args=(a,b,R,df), epsabs=1e-3,limit=75))
    prob = res * bghfactor(df)
    return prob

#written by Enzo Michelangeli, style changes by josef-pktd
# Student's T random variable
def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution

    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))

    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable


    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal


from numpy.testing import assert_almost_equal
def test_mvn_mvt():
    corr_equal = np.asarray([[1.0, 0.5, 0.5],[0.5,1,0.5],[0.5,0.5,1]])
    a = -1 * np.ones(3)
    b = 3 * np.ones(3)
    df = 4
    #result from R, mvtnorm with option
    #algorithm = GenzBretz(maxpts = 100000, abseps = 0.000001, releps = 0)
    #     or higher
    probmvt_R = 0.60414   #reported error approx. 7.5e-06
    probmvn_R = 0.673970  #reported error approx. 6.4e-07
    assert_almost_equal(probmvt_R, mvstdtprob(a, b, corr_equal, df), 4)
    assert_almost_equal(probmvn_R, mvstdnormcdf(a, b, corr_equal, abseps=1e-5), 4)

    mvn_high = mvstdnormcdf(a, b, corr_equal, abseps=1e-8, maxpts=10000000)
    assert_almost_equal(probmvn_R, mvn_high, 5)
    #this still barely fails sometimes at 6 why?? error is -7.2627419411830374e-007
    #>>> 0.67396999999999996 - 0.67397072627419408
    #-7.2627419411830374e-007
    #>>> assert_almost_equal(0.67396999999999996, 0.67397072627419408, 6)
    #Fail


    corr2 = corr_equal.copy()
    corr2[2,1] = -0.5
    R2 = corr2  #alias, partial refactoring
    probmvn_R = 0.6472497 #reported error approx. 7.7e-08
    probmvt_R = 0.5881863 #highest reported error up to approx. 1.99e-06
    assert_almost_equal(mvstdtprob(a, b, R2, df), probmvt_R, 4)
    assert_almost_equal(mvstdnormcdf(a, b, R2, abseps=1e-5), probmvn_R, 4)

    #from -inf
    #print 'from -inf'
    a2 = a.copy()
    a2[:] = -np.inf
    probmvn_R = 0.9961141 #using higher precision in R, error approx. 6.866163e-07
    probmvt_R = 0.9522146 #using higher precision in R, error approx. 1.6e-07
    assert_almost_equal(mvstdtprob(a2, b, R2, df), probmvt_R, 4)
    assert_almost_equal(mvstdnormcdf(a2, b, R2, maxpts=100000, abseps=1e-5), probmvn_R, 4)

    #from 0 to inf
    #print '0 inf'
    probmvn_R = 0.1666667 #error approx. 6.1e-08
    probmvt_R = 0.1666667 #error approx. 8.2e-08
    assert_almost_equal(mvstdtprob(np.zeros(3), -a2, R2, df), probmvt_R, 4)
    assert_almost_equal(mvstdnormcdf(np.zeros(3), -a2, R2, maxpts=100000,
                                     abseps=1e-5), probmvn_R, 4)

    #unequal integration bounds
    #print "ue"
    a3 = np.array([0.5, -0.5, 0.5])
    probmvn_R = 0.06910487 #using higher precision in R, error approx. 3.5e-08
    probmvt_R = 0.05797867 #using higher precision in R, error approx. 5.8e-08
    assert_almost_equal(mvstdtprob(a3, a3+1, R2, df), probmvt_R, 4)
    assert_almost_equal(mvstdnormcdf(a3, a3+1, R2, maxpts=100000, abseps=1e-5),
                        probmvn_R, 4)



if __name__ == '__main__':
    corr = np.asarray([[1.0, 0, 0.5],[0,1,0],[0.5,0,1]])
    corr_indep = np.asarray([[1.0, 0, 0],[0,1,0],[0,0,1]])
    corr_equal = np.asarray([[1.0, 0.5, 0.5],[0.5,1,0.5],[0.5,0.5,1]])
    R = corr_equal
    a = np.array([-np.inf,-np.inf,-100.0])
    a = np.array([-0.96,-0.96,-0.96])
    b = np.array([0.0,0.0,0.0])
    b = np.array([0.96,0.96, 0.96])
    a[:] = -1
    b[:] = 3
    df = 10.
    sqrt_df = np.sqrt(df)
    print mvstdnormcdf(a, b, corr, abseps=1e-6)

    #print integrate.quad(funbgh, 0, np.inf, args=(a,b,R,df))
    print (stats.t.cdf(b[0], df) - stats.t.cdf(a[0], df))**3

    s = 1
    print mvstdnormcdf(s*a/sqrt_df, s*b/sqrt_df, R)


    df=4
    print mvstdtprob(a, b, R, df)
    test_mvn_mvt()

    S = np.array([[1.,.5],[.5,1.]])
    print multivariate_t_rvs([10.,20.], S, 2, 5)

    nobs = 10000
    rvst = multivariate_t_rvs([10.,20.], S, 2, nobs)
    print np.sum((rvst<[10.,20.]).all(1),0) * 1. / nobs
    print mvstdtprob(-np.inf*np.ones(2), np.zeros(2), R[:2,:2], 2)


    '''
        > lower <- -1
        > upper <- 3
        > df <- 4
        > corr <- diag(3)
        > delta <- rep(0, 3)
        > pmvt(lower=lower, upper=upper, delta=delta, df=df, corr=corr)
        [1] 0.5300413
        attr(,"error")
        [1] 4.321136e-05
        attr(,"msg")
        [1] "Normal Completion"
        > (pt(upper, df) - pt(lower, df))**3
        [1] 0.4988254

    '''


