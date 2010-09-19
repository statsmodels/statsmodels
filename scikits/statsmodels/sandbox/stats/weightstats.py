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

'''


import numpy as np
from scipy import stats

class OneTimeProperty(object):


    """A descriptor to make special properties that become normal attributes.

    This is meant to be used mostly by the auto_attr decorator in this module.
    Author: Fernando Perez, copied from nitime
    """
    def __init__(self,func):

        """Create a OneTimeProperty instance.

         Parameters
         ----------
           func : method

             The method that will be called the first time to compute a value.
             Afterwards, the method's name will be a standard attribute holding
             the value of this computation.
             """
        self.getter = func
        self.name = func.func_name

    def __get__(self,obj,type=None):
        """This will be called on attribute access on the class or instance. """

        if obj is None:
            # Being called on the class, return the original function. This way,
            # introspection works on the class.
            #return func
            print 'class access'
            return self.getter

        val = self.getter(obj)
        #print "** auto_attr - loading '%s'" % self.name  # dbg
        setattr(obj, self.name, val)
        return val


class DescrStatsW(object):
    '''descriptive statistics with weights for simple case

    assumes that the data is 1d or 2d with (nobs,nvars) ovservations in rows,
    variables in columns, and that the same weight apply to each column.

    If degrees of freedom correction is used than weights should add up to the
    number of observations. ttest also assumes that the sum of weights
    corresponds to the sample size.


    '''
    def __init__(self, data, weights=None, ddof=0):

        self.data = np.asarray(data)
        if weights is None:
           self.weights = np.ones(self.data.shape[0])
        else:
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
        return np.dot(self.data, self.weights)

    @OneTimeProperty
    def mean(self):
        return self.sum / self.sum_weights

    @OneTimeProperty
    def demeaned(self):
        return self.data - self.mean

    @OneTimeProperty
    def sumsquares(self):
        return np.dot(self.demeaned**2, self.weights)

    #need memoize instead of cache decorator
    def var_ddof(self, ddof=0):
        return sumsquares(self) / (self.sum_weights - ddof)

    def std_ddof(self, ddof=0):
        return np.sqrt(self.var(ddof=ddof))

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
        return np.dot(self.demeaned.T, self.demeaned) / self.sum_weights

    @OneTimeProperty
    def corrcoef(self):
        '''correlation coefficient with default ddof for standard deviation
        '''
        return self.cov / self.std() / self.std()[:,None]

    @OneTimeProperty
    def std_mean(self):
        '''standard deviation of mean

        '''
        return self.std / np.sqrt(self.sum_weights)


    def std_var(self):
        pass

    def confint_mean(self, alpha=0.05):
        dof = self.sum_weights - 1
        tcrit = stats.t.ppf((1+alpha)/2, dof)
        lower = self.mean - tcrit * self.std_mean
        upper = self.mean + tcrit * self.std_mean
        return lower, upper



    def ttest_mean(self, value, sides='two-sided'):
        '''ttest of Null hypothesis that mean is equal to value.

        The alternative hypothesis H1 is defined by sides
        'two-sided': H1: mean different than value
        'larger' :   H1: mean larger than value
        'smaller' :  H1: mean smaller than value

        '''
        tstat = (self.mean - self.value) / self.std_mean
        dof = self.sum_weights - 1
        from scipy import stats
        if tail == 'two-sided':
           pvalue = stats.t.sf(np.abs(tstat, dof))*2
        elif tail == 'larger':
           pvalue = stats.t.sf(tstat, dof)
        elif tail == 'smaller':
           pvalue = stats.t.cdf(tstat, dof)

        return tstat, pvalue, dof

    def ttest_meandiff(self, other):
       pass


def tstat_generic(value, value2, std_diff, dof, tail):
    '''generic ttest to save typing'''
    tstat = (value - value2) / std_diff
    from scipy import stats
    if tail == 'two-sided':
       pvalue = stats.t.sf(np.abs(tstat), dof)*2
    elif tail == 'larger':
       pvalue = stats.t.sf(tstat, dof)
    elif tail == 'smaller':
       pvalue = stats.t.cdf(tstat, dof)
    return tstat, pvalue


class CompareMeans(object):
    '''temporary just to hold formulas

    formulas should also be correct for unweighted means

    not sure what happens if we have several variables.
    everything should go through vectorized but not checked yet.

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
        return np.sqrt(d1.var / (d1.nobs-1) + d2.var / (d1.nobs-1))

    @OneTimeProperty
    def std_meandiff_pooledvar(self):
        '''
        uses d1.ddof, d2.ddof which should be one for the ttest
        '''
        d1 = self.d1
        d2 = self.d2
        #could make var_pooled into attribute
        var_pooled = ((d1.sumsquares + d2.sumsquares) /
                          (d1.nobs - d1.ddof + d2.nobs - d2.ddof))
        return np.sqrt(var_pooled * (1. / d1.nobs + 1. /d1.nobs))

    def ttest_ind(self, tail='two-sided', usevar='pooled'):
        '''ttest for the null hypothesis of identical means

        note: I was looking for `usevar` option for the multiple comparison
           tests correction

        this should also be the same as onewaygls, except for ddof differences
        '''
        d1 = self.d1
        d2 = self.d2

        if usevar == 'pooled':
            stdm = self.std_meandiff_pooledvar
            dof = (d1.nobs - 1 + d1.nobs - 1)
        elif usevar == 'separate':
            stdm = self.std_meandiff_pooledvar
            #this follows blindly the SPSS manual
            #except I assume var has ddof=0
            #I should check d1.ddof, d2.ddof
            sem1 = d1.var / (d1.nobs-1)
            sem2 = d2.var / (d2.nobs-1)
            semsum = sem1 + sem2
            z1 = (sem1 / semsum)**2 / (d1.nobs - 1)
            z2 = (sem2 / semsum)**2 / (d2.nobs - 1)
            dof = 1. / (z1 + z2)

        tstat, pval = tstat_generic(d1.mean, d2.mean, stdm, dof, tail)

        return tstat, pval, dof


    def test_equal_var():
        d1 = self.d1
        d2 = self.d2
        #rewrite this now just use scipy.stats
        return stats.levene(d1.data, d2.data)


def ttest_ind(x1, x2, tail='two-sided',
                        usevar='pooled',
                        weights=(None, None)):
    '''ttest independent sample

    convenience function that uses the classes and throws away the intermediate
    results,
    compared to scipy stats: drops axis option, adds tail, usevar, and
    weights option
    '''
    cm = CompareMeans(DescrStatsW(x1, weights=weights[0]),
                     DescrStatsW(x2, weights=weights[1]))
    tstat, pval, dof = cm.ttest_ind(tail=tail,usevar=usevar)
    return tstat, pval, dof



if __name__ == '__main__':
     m1, m2 = 1, 2
     x1, x2 = ([m1, m2] + np.random.randn(20, 2)).T
     d1 = DescrStatsW(x1)
     print ttest_ind(x1, x2)
     print ttest_ind(x1, x2, usevar='separate')
     print stats.ttest_ind(x1, x2)






