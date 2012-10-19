'''Ttests and descriptive statistics with weights


Created on 2010-09-18

Author: josef-pktd
License: BSD (3-clause)

This follows in large parts the SPSS manual, which is largely the same as
the SAS manual with different, simpler notation.

Freq, Weight in SAS seems redundant since they always show up as product, SPSS
has only weights.

References
----------
SPSS manual
SAS manual

This has potential problems with ddof, I started to follow numpy with ddof=0
by default and users can change it, but this might still mess up the t-tests,
since the estimates for the standard deviation will be based on the ddof that
the user chooses.
- fixed ddof for the meandiff ttest, now matches scipy.stats.ttest_ind

Note: scipy has now a separate, pooled variance option in ttest, but I haven't
compared yet.

'''


import numpy as np
from scipy import stats

from statsmodels.tools.decorators import OneTimeProperty


class DescrStatsW(object):
    '''descriptive statistics with weights for case weights

    assumes that the data is 1d or 2d with (nobs,nvars) observations in rows,
    variables in columns, and that the same weight apply to each column.

    If degrees of freedom correction is used, then weights should add up to the
    number of observations. ttest also assumes that the sum of weights
    corresponds to the sample size.

    This is essentially the same as replicating each observations by it's
    weight, if the weights are integers, often called case weights.

    Parameters
    ----------
    data : array_like, 1-D or 2-D
        dataset
    weights : None or 1-D ndarray
        weights for each observation, with same length as zero axis of data
    ddof : int
        default ddof=0, degrees of freedom correction used for second moments,
        var, std, cov, corrcoef


    Examples
    --------

    Note: I don't know the seed for the following, so the numbers will
    differ

    >>> x1_2d = 1.0 + np.random.randn(20, 3)
    >>> w1 = np.random.randint(1,4, 20)
    >>> d1 = DescrStatsW(x1_2d, weights=w1)
    >>> d1.mean
    array([ 1.42739844,  1.23174284,  1.083753  ])
    >>> d1.var
    array([ 0.94855633,  0.52074626,  1.12309325])
    >>> d1.std_mean
    array([ 0.14682676,  0.10878944,  0.15976497])

    >>> tstat, pval, df = d1.ttest_mean(0)
    >>> tstat; pval; df
    array([  9.72165021,  11.32226471,   6.78342055])
    array([  1.58414212e-12,   1.26536887e-14,   2.37623126e-08])
    44.0

    >>> tstat, pval, df = d1.ttest_mean([0, 1, 1])
    >>> tstat; pval; df
    array([ 9.72165021,  2.13019609,  0.52422632])
    array([  1.58414212e-12,   3.87842808e-02,   6.02752170e-01])
    44.0

    #if weiqhts are integers, then asrepeats can be used

    >>> x1r = d1.asrepeats()
    >>> x1r.shape
    ...
    >>> stats.ttest_1samp(x1r, [0, 1, 1])
    ...


    '''
    def __init__(self, data, weights=None, ddof=0):

        self.data = np.asarray(data)
        if weights is None:
            self.weights = np.ones(self.data.shape[0])
        else:
            #why squeeze?
            self.weights = np.asarray(weights).squeeze().astype(float)
        self.ddof = ddof


    @OneTimeProperty
    def sum_weights(self):
        return self.weights.sum(0)

    @OneTimeProperty
    def nobs(self):
        '''alias for number of observations/cases, equal to sum of weights
        '''
        return self.sum_weights

    @OneTimeProperty
    def sum(self):
        return np.dot(self.data.T, self.weights)

    @OneTimeProperty
    def mean(self):
        return self.sum / self.sum_weights

    @OneTimeProperty
    def demeaned(self):
        return self.data - self.mean

    @OneTimeProperty
    def sumsquares(self):
        return np.dot((self.demeaned**2).T, self.weights)

    #need memoize instead of cache decorator
    def var_ddof(self, ddof=0):
        return self.sumsquares / (self.sum_weights - ddof)

    def std_ddof(self, ddof=0):
        return np.sqrt(self.var_ddof(ddof=ddof))

    @OneTimeProperty
    def var(self):
        '''variance with default degrees of freedom correction
        '''
        return self.sumsquares / (self.sum_weights - self.ddof)

    @OneTimeProperty
    def std(self):
        return np.sqrt(self.var)

    @OneTimeProperty
    def cov(self):
        '''covariance
        '''
        cov_ = np.dot(self.weights * self.demeaned.T, self.demeaned)
        cov_ /= (self.sum_weights - self.ddof)
        return cov_

    @OneTimeProperty
    def corrcoef(self):
        '''correlation coefficient with default ddof for standard deviation
        '''
        return self.cov / self.std / self.std[:,None]

    @OneTimeProperty
    def std_mean(self):
        '''standard deviation of mean

        TODO: this might assume self.ddof=0

        '''
        std = self.std
        if self.ddof != 0:
            #ddof correction,   (need copy of std)
            std = std * np.sqrt((self.sum_weights - self.ddof)
                                            / self.sum_weights)

        return std / np.sqrt(self.sum_weights - 1)


    def std_var(self):
        pass

    def confint_mean(self, alpha=0.05):
        dof = self.sum_weights - 1
        tcrit = stats.t.ppf(1-alpha/2., dof)
        lower = self.mean - tcrit * self.std_mean
        upper = self.mean + tcrit * self.std_mean
        return lower, upper



    def ttest_mean(self, value, alternative='two-sided'):
        '''ttest of Null hypothesis that mean is equal to value.

        The alternative hypothesis H1 is defined by the following
        'two-sided': H1: mean different than value
        'larger' :   H1: mean larger than value
        'smaller' :  H1: mean smaller than value

        '''
        tstat = (self.mean - value) / self.std_mean
        dof = self.sum_weights - 1
        from scipy import stats
        if alternative == 'two-sided':
           pvalue = stats.t.sf(np.abs(tstat), dof)*2
        elif alternative == 'larger':
           pvalue = stats.t.sf(tstat, dof)
        elif alternative == 'smaller':
           pvalue = stats.t.cdf(tstat, dof)

        return tstat, pvalue, dof

    def get_compare(self, other, weights=None):
        '''return an instance of CompareMeans with self and other

        Parameters
        ----------
        other : array_like or instance of DescrStatsW
            If array_like then this creates an instance of DescrStatsW with
            the given weights.
        weights : None or array
            weights are only used if other is not an instance of DescrStatsW

        Returns
        -------
        cm : instance of CompareMeans
            the instance has self attached as d1 and other as d2.

        See Also
        --------
        CompareMeans

        '''
        if not isinstance(other, self.__class__):
            d2 = DescrStatsW(other, weights)
        else:
            d2 = other
        return CompareMeans(self, d2)

    def asrepeats(self):
        '''get array that has repeats given by floor(weights)

        observations with weight=0 are dropped

        '''
        w_int = np.floor(self.weights).astype(int)
        return np.repeat(self.data, w_int, axis=0)



def tstat_generic(value, value2, std_diff, dof, alternative, diff=0):
    '''generic ttest to save typing'''
    tstat = (value - value2 + diff) / std_diff
    from scipy import stats
    if alternative in ['two-sided', '2-sided', '2']:
       pvalue = stats.t.sf(np.abs(tstat), dof)*2
    elif alternative in ['larger', 'l']:
       pvalue = stats.t.sf(tstat, dof)
    elif alternative in ['smaller', 's']:
       pvalue = stats.t.cdf(tstat, dof)
    else:
       raise ValueError('invalid alternative')
    return tstat, pvalue

def tconfint_generic(mean, std_mean, dof, alpha, alternative, diff=0):
    '''generic t-confint to save typing'''

    from scipy import stats
    if alternative in ['two-sided', '2-sided', '2']:
        tcrit = stats.t.ppf(1 - alpha / 2., dof)
        lower = mean - tcrit * std_mean
        upper = mean + tcrit * std_mean
    elif alternative in ['larger', 'l']:
        tcrit = stats.t.ppf(alpha, dof)
        lower = mean + tcrit * std_mean
        upper = np.inf
    elif alternative in ['smaller', 's']:
        tcrit = stats.t.ppf(1 - alpha, dof)
        lower = -np.inf
        upper = mean + tcrit * std_mean
    else:
       raise ValueError('invalid alternative')

    return lower, upper

class CompareMeans(object):
    '''temporary just to hold formulas

    formulas should also be correct for unweighted means

    not sure what happens if we have several variables.
    everything should go through vectorized but not checked yet.


    extend to any number of groups or write a version that works in that
    case, like in SAS and SPSS.

    Parameters
    ----------

    '''

    def __init__(self, d1, d2):
        '''assume d1, d2 hold the relevant attributes

        '''
        self.d1 = d1
        self.d2 = d2
        #assume nobs is available
 ##   if not hasattr(self.d1, 'nobs'):
 ##       d1.nobs1 = d1.sum_weights.astype(float)  #float just to make sure
 ##   self.nobs2 = d2.sum_weights.astype(float)

    @OneTimeProperty
    def std_meandiff_separatevar(self):
        #note I have too little control so far over ddof since it's an option
        #formula assumes var has ddof=0, so we subtract ddof=1 now
        d1 = self.d1
        d2 = self.d2
        return np.sqrt(d1.var / (d1.nobs-1) + d2.var / (d2.nobs-1))

    @OneTimeProperty
    def std_meandiff_pooledvar(self):
        '''
        uses d1.ddof, d2.ddof which should be one for the ttest
        hardcoding ddof=1 for varpooled
        '''
        d1 = self.d1
        d2 = self.d2
        #could make var_pooled into attribute
        var_pooled = ((d1.sumsquares + d2.sumsquares) /
                          #(d1.nobs - d1.ddof + d2.nobs - d2.ddof))
                          (d1.nobs - 1 + d2.nobs - 1))
        return np.sqrt(var_pooled * (1. / d1.nobs + 1. /d2.nobs))

    def dof_satt(self):
        d1 = self.d1
        d2 = self.d2
        #this follows blindly the SPSS manual
        #except I assume var has ddof=0
        #I should check d1.ddof, d2.ddof
        sem1 = d1.var / (d1.nobs-1)
        sem2 = d2.var / (d2.nobs-1)
        semsum = sem1 + sem2
        z1 = (sem1 / semsum)**2 / (d1.nobs - 1)
        z2 = (sem2 / semsum)**2 / (d2.nobs - 1)
        dof = 1. / (z1 + z2)
        return dof

    def ttest_ind(self, alternative='two-sided', usevar='pooled', diff=0):
        '''ttest for the null hypothesis of identical means

        note: I was looking for `usevar` option for the multiple comparison
           tests correction

        this should also be the same as onewaygls, except for ddof differences
        '''
        d1 = self.d1
        d2 = self.d2

        if usevar == 'pooled':
            stdm = self.std_meandiff_pooledvar
            dof = (d1.nobs - 1 + d2.nobs - 1)
        elif usevar == 'separate':
            stdm = self.std_meandiff_separatevar
            dof = self.dof_satt()

        tstat, pval = tstat_generic(d1.mean, d2.mean, stdm, dof, alternative,
                                    diff=diff)

        return tstat, pval, dof

    def confint_diff(self, alpha=0.05, alternative='two-sided',
                     usevar='pooled'):
        d1 = self.d1
        d2 = self.d2
        diff = d1.mean - d2.mean
        if usevar == 'pooled':
            std_diff = self.std_meandiff_pooledvar
            dof = (d1.nobs - 1 + d2.nobs - 1)
        elif usevar == 'separate':
            std_diff = self.std_meandiff_separatevar
            dof = self.dof_satt()

        res = tconfint_generic(diff, std_diff, dof, alpha=alpha,
                               alternative=alternative)
        return res

    def tost(self, low, upp, usevar='pooled'):
        tt1 = self.ttest_ind(alternative='smaller', usevar=usevar, diff=low)
        tt2 = self.ttest_ind(alternative='larger', usevar=usevar, diff=upp)
        return max(tt1[1], tt2[1]), (tt1, tt2)

#doesn't work for 2d, doesn't take weights into account
##    def test_equal_var(self):
##        '''Levene test for independence
##
##        '''
##        d1 = self.d1
##        d2 = self.d2
##        #rewrite this, for now just use scipy.stats
##        return stats.levene(d1.data, d2.data)


def ttest_ind(x1, x2, alternative='two-sided', usevar='pooled',
                      weights=(None, None), diff=0):
    '''ttest independent sample

    convenience function that uses the classes and throws away the intermediate
    results,
    compared to scipy stats: drops axis option, adds alternative, usevar, and
    weights option
    '''
    cm = CompareMeans(DescrStatsW(x1, weights=weights[0], ddof=0),
                     DescrStatsW(x2, weights=weights[1], ddof=0))
    tstat, pval, dof = cm.ttest_ind(alternative=alternative, usevar=usevar, diff=diff)

    return tstat, pval, dof


def tost_ind(x1, x2, low, upp, usevar='pooled', weights=(None, None),
             transform=None):
    '''tost independent sample

    convenience function that uses the classes and throws away the intermediate
    results,
    compared to scipy stats: drops axis option, adds alternative, usevar, and
    weights option
    '''

    if transform:
        xx = transform(np.hstack((x1, x2)))
        x1 = xx[:len(x1)]
        x2 = xx[len(x1):]
        low = transform(low)
        upp = transform(upp)
    cm = CompareMeans(DescrStatsW(x1, weights=weights[0], ddof=0),
                     DescrStatsW(x2, weights=weights[1], ddof=0))
    pval, res = cm.tost(low, upp, usevar=usevar)
    return pval, res[0], res[1]

def tost_paired(x, y, low, upp, transform=None, weights=None):

    if transform:
        xy = transform(np.hstack((x,y)))
        x = xy[:len(x)]
        y = xy[len(x):]
        low = transform(low)
        upp = transform(upp)
    dd = DescrStatsW(x - y, weights=weights, ddof=0)
    t1, pv1, df1 = dd.ttest_mean(low, alternative='larger')
    t2, pv2, df2 = dd.ttest_mean(upp, alternative='smaller')
    return max(pv1, pv2), (t1, pv1), (t2, pv2)

