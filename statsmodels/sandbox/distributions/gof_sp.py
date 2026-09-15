# -*- coding: utf-8 -*-
"""

Created on Fri Jun 08 16:13:47 2012

Author: Josef Perktold, Scipy Developers

"""


from scipy.special import kolmogorov as ksprob

#from scipy.stats unchanged
def ks_2samp(data1, data2):
    """
    Computes the Kolmogorov-Smirnof statistic on 2 samples.

    This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution.

    Parameters
    ----------
    a, b : sequence of 1-D ndarrays
        two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can be different


    Returns
    -------
    D : float
        KS statistic
    p-value : float
        two-tailed p-value


    Notes
    -----

    This tests whether 2 samples are drawn from the same distribution. Note
    that, like in the case of the one-sample K-S test, the distribution is
    assumed to be continuous.

    This is the two-sided test, one-sided tests are not implemented.
    The test uses the two-sided asymptotic Kolmogorov-Smirnov distribution.

    If the K-S statistic is small or the p-value is high, then we cannot
    reject the hypothesis that the distributions of the two samples
    are the same.

    Examples
    --------

    >>> from scipy import stats
    >>> import numpy as np
    >>> from scipy.stats import ks_2samp

    >>> #fix random seed to get the same result
    >>> np.random.seed(12345678);

    >>> n1 = 200  # size of first sample
    >>> n2 = 300  # size of second sample

    different distribution
    we can reject the null hypothesis since the pvalue is below 1%

    >>> rvs1 = stats.norm.rvs(size=n1,loc=0.,scale=1);
    >>> rvs2 = stats.norm.rvs(size=n2,loc=0.5,scale=1.5)
    >>> ks_2samp(rvs1,rvs2)
    (0.20833333333333337, 4.6674975515806989e-005)

    slightly different distribution
    we cannot reject the null hypothesis at a 10% or lower alpha since
    the pvalue at 0.144 is higher than 10%

    >>> rvs3 = stats.norm.rvs(size=n2,loc=0.01,scale=1.0)
    >>> ks_2samp(rvs1,rvs3)
    (0.10333333333333333, 0.14498781825751686)

    identical distribution
    we cannot reject the null hypothesis since the pvalue is high, 41%

    >>> rvs4 = stats.norm.rvs(size=n2,loc=0.0,scale=1.0)
    >>> ks_2samp(rvs1,rvs4)
    (0.07999999999999996, 0.41126949729859719)

    """
    data1, data2 = map(np.asarray, (data1, data2))
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    n1 = len(data1)
    n2 = len(data2)
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1,data2])
    #reminder: searchsorted inserts 2nd into 1st array
    cdf1 = np.searchsorted(data1,data_all,side='right')/(1.0*n1)
    cdf2 = (np.searchsorted(data2,data_all,side='right'))/(1.0*n2)
    d = np.max(np.absolute(cdf1-cdf2))
    #Note: d absolute not signed distance
    en = np.sqrt(n1*n2/float(n1+n2))
    try:
        prob = ksprob((en+0.12+0.11/en)*d)
    except:
        prob = 1.0
    return d, prob



#from scipy.stats unchanged
def kstest(rvs, cdf, args=(), N=20, alternative = 'two_sided', mode='approx',**kwds):
    """
    Perform the Kolmogorov-Smirnov test for goodness of fit

    This performs a test of the distribution G(x) of an observed
    random variable against a given distribution F(x). Under the null
    hypothesis the two distributions are identical, G(x)=F(x). The
    alternative hypothesis can be either 'two_sided' (default), 'less'
    or 'greater'. The KS test is only valid for continuous distributions.

    Parameters
    ----------
    rvs : string or array or callable
        string: name of a distribution in scipy.stats

        array: 1-D observations of random variables

        callable: function to generate random variables, requires keyword
        argument `size`

    cdf : string or callable
        string: name of a distribution in scipy.stats, if rvs is a string then
        cdf can evaluate to `False` or be the same as rvs
        callable: function to evaluate cdf

    args : tuple, sequence
        distribution parameters, used if rvs or cdf are strings
    N : int
        sample size if rvs is string or callable
    alternative : 'two_sided' (default), 'less' or 'greater'
        defines the alternative hypothesis (see explanation)

    mode : 'approx' (default) or 'asymp'
        defines the distribution used for calculating p-value

        'approx' : use approximation to exact distribution of test statistic

        'asymp' : use asymptotic distribution of test statistic


    Returns
    -------
    D : float
        KS test statistic, either D, D+ or D-
    p-value :  float
        one-tailed or two-tailed p-value

    Notes
    -----

    In the one-sided test, the alternative is that the empirical
    cumulative distribution function of the random variable is "less"
    or "greater" than the cumulative distribution function F(x) of the
    hypothesis, G(x)<=F(x), resp. G(x)>=F(x).

    Examples
    --------

    >>> from scipy import stats
    >>> import numpy as np
    >>> from scipy.stats import kstest

    >>> x = np.linspace(-15,15,9)
    >>> kstest(x,'norm')
    (0.44435602715924361, 0.038850142705171065)

    >>> np.random.seed(987654321) # set random seed to get the same result
    >>> kstest('norm','',N=100)
    (0.058352892479417884, 0.88531190944151261)

    is equivalent to this

    >>> np.random.seed(987654321)
    >>> kstest(stats.norm.rvs(size=100),'norm')
    (0.058352892479417884, 0.88531190944151261)

    Test against one-sided alternative hypothesis:

    >>> np.random.seed(987654321)

    Shift distribution to larger values, so that cdf_dgp(x)< norm.cdf(x):

    >>> x = stats.norm.rvs(loc=0.2, size=100)
    >>> kstest(x,'norm', alternative = 'less')
    (0.12464329735846891, 0.040989164077641749)

    Reject equal distribution against alternative hypothesis: less

    >>> kstest(x,'norm', alternative = 'greater')
    (0.0072115233216311081, 0.98531158590396395)

    Don't reject equal distribution against alternative hypothesis: greater

    >>> kstest(x,'norm', mode='asymp')
    (0.12464329735846891, 0.08944488871182088)


    Testing t distributed random variables against normal distribution:

    With 100 degrees of freedom the t distribution looks close to the normal
    distribution, and the kstest does not reject the hypothesis that the sample
    came from the normal distribution

    >>> np.random.seed(987654321)
    >>> stats.kstest(stats.t.rvs(100,size=100),'norm')
    (0.072018929165471257, 0.67630062862479168)

    With 3 degrees of freedom the t distribution looks sufficiently different
    from the normal distribution, that we can reject the hypothesis that the
    sample came from the normal distribution at a alpha=10% level

    >>> np.random.seed(987654321)
    >>> stats.kstest(stats.t.rvs(3,size=100),'norm')
    (0.131016895759829, 0.058826222555312224)

    """
    if isinstance(rvs, basestring):
        #cdf = getattr(stats, rvs).cdf
        if (not cdf) or (cdf == rvs):
            cdf = getattr(distributions, rvs).cdf
            rvs = getattr(distributions, rvs).rvs
        else:
            raise AttributeError('if rvs is string, cdf has to be the same distribution')


    if isinstance(cdf, basestring):
        cdf = getattr(distributions, cdf).cdf
    if callable(rvs):
        kwds = {'size':N}
        vals = np.sort(rvs(*args,**kwds))
    else:
        vals = np.sort(rvs)
        N = len(vals)
    cdfvals = cdf(vals, *args)

    if alternative in ['two_sided', 'greater']:
        Dplus = (np.arange(1.0, N+1)/N - cdfvals).max()
        if alternative == 'greater':
            return Dplus, distributions.ksone.sf(Dplus,N)

    if alternative in ['two_sided', 'less']:
        Dmin = (cdfvals - np.arange(0.0, N)/N).max()
        if alternative == 'less':
            return Dmin, distributions.ksone.sf(Dmin,N)

    if alternative == 'two_sided':
        D = np.max([Dplus,Dmin])
        if mode == 'asymp':
            return D, distributions.kstwobign.sf(D*np.sqrt(N))
        if mode == 'approx':
            pval_two = distributions.kstwobign.sf(D*np.sqrt(N))
            if N > 2666 or pval_two > 0.80 - N*0.3/1000.0 :
                return D, distributions.kstwobign.sf(D*np.sqrt(N))
            else:
                return D, distributions.ksone.sf(D,N)*2

