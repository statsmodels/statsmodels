# -*- coding: utf-8 -*-
"""
Implements Lilliefors corrected Kolmogorov-Smirnov tests for normal and
exponential distributions.

`kstest_fit` is provided as a top-level function to access both tests.
`kstest_normal` and `kstest_exponential` are provided as convenience functions
with the appropriate test as the default.
`lilliefors` is provided as an alias for `kstest_fit`.

Created on Sat Oct 01 13:16:49 2011

Author: Josef Perktold
License: BSD-3

pvalues for Lilliefors test are based on formula and table in

An Analytic Approximation to the Distribution of Lilliefors's Test Statistic for Normality
Author(s): Gerard E. Dallal and Leland WilkinsonSource: The American Statistician, Vol. 40, No. 4 (Nov., 1986), pp. 294-296Published by: American Statistical AssociationStable URL: http://www.jstor.org/stable/2684607 .

On the Kolmogorov-Smirnov Test for Normality with Mean and Variance
Unknown
Hubert W. Lilliefors
Journal of the American Statistical Association, Vol. 62, No. 318. (Jun., 1967), pp. 399-402.


---

Updated 2017-07-23
Jacob C. Kimmel

Ref:
Lilliefors, H.W.
On the Kolmogorov-Smirnov test for the exponential distribution with mean unknown.
Journal of the American Statistical Association, Vol 64, No. 325. (1969), pp. 387–389.
"""
from functools import partial

from statsmodels.compat.python import string_types
import numpy as np
from scipy import stats
from .tabledist import TableDist


def ksstat(x, cdf, alternative='two_sided', args=()):
    """
    Calculate statistic for the Kolmogorov-Smirnov test for goodness of fit

    This calculates the test statistic for a test of the distribution G(x) of an observed
    variable against a given distribution F(x). Under the null
    hypothesis the two distributions are identical, G(x)=F(x). The
    alternative hypothesis can be either 'two_sided' (default), 'less'
    or 'greater'. The KS test is only valid for continuous distributions.

    Parameters
    ----------
    x : array_like, 1d
        array of observations
    cdf : string or callable
        string: name of a distribution in scipy.stats
        callable: function to evaluate cdf
    alternative : 'two_sided' (default), 'less' or 'greater'
        defines the alternative hypothesis (see explanation)
    args : tuple, sequence
        distribution parameters for call to cdf


    Returns
    -------
    D : float
        KS test statistic, either D, D+ or D-

    See Also
    --------
    scipy.stats.kstest

    Notes
    -----

    In the one-sided test, the alternative is that the empirical
    cumulative distribution function of the random variable is "less"
    or "greater" than the cumulative distribution function F(x) of the
    hypothesis, G(x)<=F(x), resp. G(x)>=F(x).

    In contrast to scipy.stats.kstest, this function only calculates the
    statistic which can be used either as distance measure or to implement
    case specific p-values.

    """
    nobs = float(len(x))

    if isinstance(cdf, string_types):
        cdf = getattr(stats.distributions, cdf).cdf
    elif hasattr(cdf, 'cdf'):
        cdf = getattr(cdf, 'cdf')

    x = np.sort(x)
    cdfvals = cdf(x, *args)

    if alternative in ['two_sided', 'greater']:
        Dplus = (np.arange(1.0, nobs+1)/nobs - cdfvals).max()
        if alternative == 'greater':
            return Dplus

    if alternative in ['two_sided', 'less']:
        Dmin = (cdfvals - np.arange(0.0, nobs)/nobs).max()
        if alternative == 'less':
            return Dmin

    D = np.max([Dplus,Dmin])

    return D


# new version with tabledist
# --------------------------

def get_lilliefors_table(dist='norm'):
    '''
    Generates tables for significance levels of Lilliefors test statistics

    Tables for available normal and exponential distribution testing,
    as specified in Lilliefors references above

    Parameters
    ----------
    dist : string.
        distribution being tested in set {'norm', 'exp'}.

    Returns
    -------
    lf : TableDist object.
        table of critical values
    '''
    # function just to keep things together
    # for this test alpha is sf probability, i.e. right tail probability

    if dist == 'norm':
        alpha = np.array([ 0.2  ,  0.15 ,  0.1  ,  0.05 ,  0.01 ,  0.001])[::-1]
        size = np.array([ 4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
                     16,  17,  18,  19,  20,  25,  30,  40, 100, 400, 900], float)

        # critical values, rows are by sample size, columns are by alpha
        crit_lf = np.array(   [[303, 321, 346, 376, 413, 433],
                               [289, 303, 319, 343, 397, 439],
                               [269, 281, 297, 323, 371, 424],
                               [252, 264, 280, 304, 351, 402],
                               [239, 250, 265, 288, 333, 384],
                               [227, 238, 252, 274, 317, 365],
                               [217, 228, 241, 262, 304, 352],
                               [208, 218, 231, 251, 291, 338],
                               [200, 210, 222, 242, 281, 325],
                               [193, 202, 215, 234, 271, 314],
                               [187, 196, 208, 226, 262, 305],
                               [181, 190, 201, 219, 254, 296],
                               [176, 184, 195, 213, 247, 287],
                               [171, 179, 190, 207, 240, 279],
                               [167, 175, 185, 202, 234, 273],
                               [163, 170, 181, 197, 228, 266],
                               [159, 166, 176, 192, 223, 260],
                               [143, 150, 159, 173, 201, 236],
                               [131, 138, 146, 159, 185, 217],
                               [115, 120, 128, 139, 162, 189],
                               [ 74,  77,  82,  89, 104, 122],
                               [ 37,  39,  41,  45,  52,  61],
                               [ 25,  26,  28,  30,  35,  42]])[:,::-1] / 1000.

        # also build a table for larger sample sizes
        def f(n):
            return np.array([0.736, 0.768, 0.805, 0.886, 1.031]) / np.sqrt(n)

        higher_sizes = np.array([35, 40, 45, 50, 60, 70,
                                 80, 100, 200, 500, 1000,
                                 2000, 3000, 5000, 10000, 100000], float)
        higher_crit_lf = np.zeros([higher_sizes.shape[0], crit_lf.shape[1]-1])
        for i in range(len(higher_sizes)):
            higher_crit_lf[i, :] = f(higher_sizes[i])

        alpha_large = alpha[:-1]
        size_large = np.concatenate([size, higher_sizes])
        crit_lf_large = np.vstack([crit_lf[:-4,:-1], higher_crit_lf])
        lf = TableDist(alpha, size, crit_lf)

    elif dist == 'exp':
        alpha = np.array([0.2,  0.15,  0.1,  0.05, 0.01])[::-1]
        size = np.array([3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30],
                        float)

        crit_lf = np.array([   [451, 479, 511, 551, 600],
                                [396, 422, 499, 487, 548],
                                [359, 382, 406, 442, 504],
                                [331, 351, 375, 408, 470],
                                [309, 327, 350, 382, 442],
                                [291, 308, 329, 360, 419],
                                [277, 291, 311, 341, 399],
                                [263, 277, 295, 325, 380],
                                [251, 264, 283, 311, 365],
                                [241, 254, 271, 298, 351],
                                [232, 245, 261, 287, 338],
                                [224, 237, 252, 277, 326],
                                [217, 229, 244, 269, 315],
                                [211, 222, 236, 261, 306],
                                [204, 215, 229, 253, 297],
                                [199, 210, 223, 246, 289],
                                [193, 204, 218, 239, 283],
                                [188, 199, 212, 234, 278],
                                [170, 180, 191, 210, 247],
                                [155, 164, 174, 192, 226]])[:,::-1] / 1000.

        def f(n):
            return np.array([.86, .91, .96, 1.06, 1.25]) / np.sqrt(n)

        higher_sizes = np.array([35, 40, 45, 50, 60, 70,
                                80, 100, 200, 500, 1000,
                                2000, 3000, 5000, 10000, 100000], float)
        higher_crit_lf = np.zeros([higher_sizes.shape[0], crit_lf.shape[1]])
        for i in range(len(higher_sizes)):
            higher_crit_lf[i,:] = f(higher_sizes[i])

        size = np.concatenate([size, higher_sizes])
        crit_lf = np.vstack([crit_lf, higher_crit_lf])
        lf = TableDist(alpha, size, crit_lf)
    else:
        raise ValueError("Invalid dist parameter. dist must be 'norm' or 'exp'")

    return lf

lilliefors_table_norm = get_lilliefors_table(dist='norm')
lilliefors_table_expon = get_lilliefors_table(dist='exp')


def pval_lf(Dmax, n):
    '''approximate pvalues for Lilliefors test

    This is only valid for pvalues smaller than 0.1 which is not checked in
    this function.

    Parameters
    ----------
    Dmax : array_like
        two-sided Kolmogorov-Smirnov test statistic
    n : int or float
        sample size

    Returns
    -------
    p-value : float or ndarray
        pvalue according to approximation formula of Dallal and Wilkinson.

    Notes
    -----
    This is mainly a helper function where the calling code should dispatch
    on bound violations. Therefore it doesn't check whether the pvalue is in
    the valid range.

    Precision for the pvalues is around 2 to 3 decimals. This approximation is
    also used by other statistical packages (e.g. R:fBasics) but might not be
    the most precise available.

    References
    ----------
    DallalWilkinson1986

    '''

    #todo: check boundaries, valid range for n and Dmax
    if n > 100:
        Dmax *= (n / 100.)**0.49
        n = 100
    pval = np.exp(-7.01256 * Dmax**2 * (n + 2.78019)
                  + 2.99587 * Dmax * np.sqrt(n + 2.78019) - 0.122119
                  + 0.974598/np.sqrt(n) + 1.67997/n)
    return pval


def kstest_fit(x, dist='norm', pvalmethod='approx'):
    """
    Lilliefors test for normality or an exponential distribution.

    Kolmogorov Smirnov test with estimated mean and variance

    Parameters
    ----------
    x : array_like, 1d
        data series, sample
    dist : {'norm', 'exp'}, optional
        Distribution to test in set.
    pvalmethod : {'approx', 'table'}, optional
        'approx' is only valid for normality. if `dist = 'exp'`,
        `table` is returned.
        'approx' uses the approximation formula of Dalal and Wilkinson,
        valid for pvalues < 0.1. If the pvalue is larger than 0.1, then the
        result of `table` is returned

        For normality:
        'table' uses the table from Dalal and Wilkinson, which is available
        for pvalues between 0.001 and 0.2, and the formula of Lilliefors for
        large n (n>900). Values in the table are linearly interpolated.
        Values outside the range will be returned as bounds, 0.2 for large and
        0.001 for small pvalues.
        For exponential:
        'table' uses the table from Lilliefors 1967, available for pvalues
        between 0.01 and 0.2.
        Values outside the range will be returned as bounds, 0.2 for large and
        0.01 for small pvalues.

    Returns
    -------
    ksstat : float
        Kolmogorov-Smirnov test statistic with estimated mean and variance.
    pvalue : float
        If the pvalue is lower than some threshold, e.g. 0.05, then we can
        reject the Null hypothesis that the sample comes from a normal
        distribution

    Notes
    -----
    Reported power to distinguish normal from some other distributions is lower
    than with the Anderson-Darling test.

    could be vectorized
    """
    x = np.asarray(x)
    nobs = len(x)

    if dist == 'norm':
        z = (x - x.mean()) / x.std(ddof=1)
        test_d = stats.norm.cdf
        lilliefors_table = lilliefors_table_norm
    elif dist == 'exp':
        z = x / x.mean()
        test_d = stats.expon.cdf
        lilliefors_table = lilliefors_table_expon
        pvalmethod = 'table'
    else:
        raise ValueError("Invalid dist parameter. dist must be 'norm' or 'exp'")

    d_ks = ksstat(z, test_d, alternative='two_sided')

    if pvalmethod == 'approx':
        pval = pval_lf(d_ks, nobs)
        # check pval is in desired range
        if pval > 0.1:
            pval = lilliefors_table.prob(d_ks, nobs)
    elif pvalmethod == 'table':
        pval = lilliefors_table.prob(d_ks, nobs)

    return d_ks, pval


lilliefors = kstest_fit

# namespace aliases
kstest_normal = kstest_fit
kstest_exponential = partial(kstest_fit, dist='exp')

# old version:
# ------------
'''
tble = \
00 20 15 10 05 01 .1
4 .303 .321 .346 .376 .413 .433
5 .289 .303 .319 .343 .397 .439
6 .269 .281 .297 .323 .371 .424
7 .252 .264 .280 .304 .351 .402
8 .239 .250 .265 .288 .333 .384
9 .227 .238 .252 .274 .317 .365
10 .217 .228 .241 .262 .304 .352
11 .208 .218 .231 .251 .291 .338
12 .200 .210 .222 .242 .281 .325
13 .193 .202 .215 .234 .271 .314
14 .187 .196 .208 .226 .262 .305
15 .181 .190 .201 .219 .254 .296
16 .176 .184 .195 .213 .247 .287
17 .171 .179 .190 .207 .240 .279
18 .167 .175 .185 .202 .234 .273
19 .163 .170 .181 .197 .228 .266
20 .159 .166 .176 .192 .223 .260
25 .143 .150 .159 .173 .201 .236
30 .131 .138 .146 .159 .185 .217
40 .115 .120 .128 .139 .162 .189
100 .074 .077 .082 .089 .104 .122
400 .037 .039 .041 .045 .052 .061
900 .025 .026 .028 .030 .035 .042'''

'''
parr = np.array([line.split() for line in tble.split('\n')],float)
size = parr[1:,0]
alpha = parr[0,1:] / 100.
crit = parr[1:, 1:]

alpha = np.array([ 0.2  ,  0.15 ,  0.1  ,  0.05 ,  0.01 ,  0.001])
size = np.array([ 4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
                 16,  17,  18,  19,  20,  25,  30,  40, 100, 400, 900], float)

#critical values, rows are by sample size, columns are by alpha
crit_lf = np.array(   [[303, 321, 346, 376, 413, 433],
                       [289, 303, 319, 343, 397, 439],
                       [269, 281, 297, 323, 371, 424],
                       [252, 264, 280, 304, 351, 402],
                       [239, 250, 265, 288, 333, 384],
                       [227, 238, 252, 274, 317, 365],
                       [217, 228, 241, 262, 304, 352],
                       [208, 218, 231, 251, 291, 338],
                       [200, 210, 222, 242, 281, 325],
                       [193, 202, 215, 234, 271, 314],
                       [187, 196, 208, 226, 262, 305],
                       [181, 190, 201, 219, 254, 296],
                       [176, 184, 195, 213, 247, 287],
                       [171, 179, 190, 207, 240, 279],
                       [167, 175, 185, 202, 234, 273],
                       [163, 170, 181, 197, 228, 266],
                       [159, 166, 176, 192, 223, 260],
                       [143, 150, 159, 173, 201, 236],
                       [131, 138, 146, 159, 185, 217],
                       [115, 120, 128, 139, 162, 189],
                       [ 74,  77,  82,  89, 104, 122],
                       [ 37,  39,  41,  45,  52,  61],
                       [ 25,  26,  28,  30,  35,  42]]) / 1000.

#original Lilliefors paper
crit_greater30 = lambda n: np.array([0.736, 0.768, 0.805, 0.886, 1.031])/np.sqrt(n)
alpha_greater30 = np.array([ 0.2  ,  0.15 ,  0.1  ,  0.05 ,  0.01 ,  0.001])


from scipy.interpolate import interp1d
n_alpha = 6
polyn = [interp1d(size, crit[:,i]) for i in range(n_alpha)]

def critpolys(n):
    return np.array([p(n) for p in polyn])

def pval_lftable(x, n):
    #returns extrem probabilities, 0.001 and 0.2, for out of range
    critvals = critpolys(n)
    if x < critvals[0]:
        return alpha[0]
    elif x > critvals[-1]:
        return alpha[-1]
    else:
        return interp1d(critvals, alpha)(x)

for n in [19, 19.5, 20, 21, 25]:
    print critpolys(n)

print pval_lftable(0.166, 20)
print pval_lftable(0.166, 21)



print 'n=25:', '.103 .052 .010'
print [pval_lf(x, 25) for x in [.159, .173, .201, .236]]

print 'n=10', '.103 .050 .009'
print [pval_lf(x, 10) for x in [.241, .262, .304, .352]]


print 'n=400', '.104 .050 .011'
print [pval_lf(x, 400) for x in crit[-2,2:-1]]
print 'n=900', '.093 .054 .011'
print [pval_lf(x, 900) for x in crit[-1,2:-1]]
print [pval_lftable(x, 400) for x in crit[-2,:]]
print [pval_lftable(x, 300) for x in crit[-3,:]]

xx = np.random.randn(40)
print kstest_normal(xx)

xx2 = np.array([ 1.138, -0.325, -1.461, -0.441, -0.005, -0.957, -1.52 ,  0.481,
        0.713,  0.175, -1.764, -0.209, -0.681,  0.671,  0.204,  0.403,
       -0.165,  1.765,  0.127, -1.261, -0.101,  0.527,  1.114, -0.57 ,
       -1.172,  0.697,  0.146,  0.704,  0.422,  0.63 ,  0.661,  0.025,
        0.177,  0.578,  0.945,  0.211,  0.153,  0.279,  0.35 ,  0.396])

( 1.138, -0.325, -1.461, -0.441, -0.005, -0.957, -1.52 ,  0.481,
        0.713,  0.175, -1.764, -0.209, -0.681,  0.671,  0.204,  0.403,
       -0.165,  1.765,  0.127, -1.261, -0.101,  0.527,  1.114, -0.57 ,
       -1.172,  0.697,  0.146,  0.704,  0.422,  0.63 ,  0.661,  0.025,
        0.177,  0.578,  0.945,  0.211,  0.153,  0.279,  0.35 ,  0.396)
r_lillieTest = [0.15096827429598147, 0.02225473302348436]
print kstest_normal(xx2), np.array(kstest_normal(xx2)) - r_lillieTest
print kstest_normal(xx2, 'table')
'''
