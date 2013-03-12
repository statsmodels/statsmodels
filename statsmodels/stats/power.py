# -*- coding: utf-8 -*-
"""Statistical power, solving for nobs, ... - trial version

Created on Sat Jan 12 21:48:06 2013

Author: Josef Perktold

Example
roundtrip - root with respect to all variables

       calculated, desired
nobs   33.367204205 33.367204205
effect 0.5 0.5
alpha  0.05 0.05
power   0.8 0.8


TODO:
refactoring
 - rename beta -> power,    beta (type 2 error is beta = 1-power)  DONE
 - I think the current implementation can handle any kinds of extra keywords
   (except for maybe raising meaningful exceptions
 - streamline code, I think internally classes can be merged
   how to extend to k-sample tests?
   user interface for different tests that map to the same (internal) test class
 - sequence of arguments might be inconsistent,
   arg and/or kwds so python checks what's required and what can be None.
 - templating for docstrings ?


"""

import numpy as np
from scipy import stats, optimize


def ttest_power(effect_size, nobs, alpha, df=None, alternative='two-sided'):
    '''Calculate power of a ttest
    '''
    d = effect_size
    if df is None:
        df = nobs - 1

    if alternative in ['two-sided', '2s']:
        alpha_ = alpha / 2.  #no inplace changes, doesn't work
    elif alternative in ['smaller', 'larger']:
        alpha_ = alpha
    else:
        raise ValueError("alternative has to be 'two-sided', 'larger' " +
                         "or 'smaller'")

    pow_ = 0
    if alternative in ['two-sided', '2s', 'larger']:
        crit_upp = stats.t.isf(alpha_, df)
        # use private methods, generic methods return nan with negative d
        pow_ = stats.nct._sf(crit_upp, df, d*np.sqrt(nobs))
    if alternative in ['two-sided', '2s', 'smaller']:
        crit_low = stats.t.ppf(alpha_, df)
        pow_ += stats.nct._cdf(crit_low, df, d*np.sqrt(nobs))
    return pow_

def normal_power(effect_size, nobs, alpha, alternative='two-sided', sigma=1.):
    '''Calculate power of a normal distributed test statistic

    '''
    d = effect_size

    if alternative in ['two-sided', '2s']:
        alpha_ = alpha / 2.  #no inplace changes, doesn't work
    elif alternative in ['smaller', 'larger']:
        alpha_ = alpha
    else:
        raise ValueError("alternative has to be 'two-sided', 'larger' " +
                         "or 'smaller'")

    pow_ = 0
    if alternative in ['two-sided', '2s', 'larger']:
        crit = stats.norm.isf(alpha_)
        pow_ = stats.norm.sf(crit - d*np.sqrt(nobs)/sigma)
    if alternative in ['two-sided', '2s', 'smaller']:
        crit = stats.norm.ppf(alpha_)
        pow_ += stats.norm.cdf(crit - d*np.sqrt(nobs)/sigma)
    return pow_

def ftest_power_k(effect_size, nobs, alpha, k_groups=2, df=None):
    '''power for ftest for one way anova with k equal sized groups

    nobs total sample size, sum over all groups

    should be general nobs observations, k_groups restrictions ???
    '''
    df_numer = nobs - k_groups
    df_denom = k_groups - 1
    crit = stats.f.isf(alpha, df_denom, df_numer)
    pow_ = stats.ncf.sf(crit, df_denom, df_numer, effect_size**2 * nobs)
    return pow_#, crit

def ftest_power(effect_size, df_numer, df_denom, alpha, ncc=1):
    '''power for ftest

    sample size is given implicitly by df_numer

    set ncc=0 to match t-test, or f-test in LikelihoodModelResults
    ncc=1 matches the non-centrality parameter in R::pwr::pwr.f2.test

    ftest_power with ncc=0 should also be correct for f_test in regression
    models, with df_num and d_denom as defined there. (not verified yet)

    '''

    nc = effect_size**2 * (df_denom + df_numer + ncc)
    crit = stats.f.isf(alpha, df_denom, df_numer)
    pow_ = stats.ncf.sf(crit, df_denom, df_numer, nc)
    return pow_ #, crit, nc


#module global for now
start_ttp = dict(effect_size=0.01, nobs=10., alpha=0.15, power=0.6,
                 nobs1=10, ratio=1)
#TODO: nobs1 and ratio are for ttest_ind,
#      need start_ttp for each test/class separately, added default start_value
#possible rootfinding problem for effect_size, starting small seems to work


#class based implementation
#--------------------------

class Power(object):
    '''Statistical Power calculations, Base Class

    so far this could all be class methods
    '''

    def power(self, *args, **kwds):
        raise NotImplementedError

    def _power_identity(self, *args, **kwds):
        power_ = kwds.pop('power')
        return self.power(*args, **kwds) - power_

    def solve_power(self, **kwds):
        '''solve for any one of the parameters of a t-test

        for t-test the keywords are:
            effect_size, nobs, alpha, power

        exactly one needs to be ``None``, all others need numeric values

        '''
        #TODO: maybe use explicit kwds,
        #    nicer but requires inspect? and not generic across tests
        #    I'm duplicating this in the subclass to get informative docstring
        key = [k for k,v in kwds.iteritems() if v is None]
        #print kwds, key;
        if len(key) != 1:
            raise ValueError('need exactly one keyword that is None')
        key = key[0]

        if key == 'power':
            del kwds['power']
            return self.power(**kwds)

        def func(x):
            kwds[key] = x
            return self._power_identity(**kwds)

        #TODO: I'm using the following so I get a warning when start_ttp is not defined
        try:
            start_value = start_ttp[key]
        except KeyError:
            start_value = 0.9
            print 'Warning: using default start_value'

        #TODO: check more cases to make this robust
        #return optimize.newton(func, start_value).item() #scalar
        val, infodict, ier, msg = optimize.fsolve(func, start_value, full_output=True) #scalar
        if ier != 1:
            print infodict
            if key in ['alpha', 'power']:
                val, r = optimize.brentq(func, 1e-8, 1-1e-8, full_output=True) #scalar
                if not r.converged:
                    print r
        return val

class TTestPower(Power):
    '''Statistical Power calculations for one sample or paired sample t-test

    '''

    def power(self, effect_size, nobs, alpha, df=None, alternative='two-sided'):
        '''Calculate the power of a t-test for one sample or paired samples.

        Parameters
        ----------
        effect_size : float
            standardized effect size, mean divided by the standard deviation.
            effect size has to be positive.
        nobs : int or float
            sample size, number of observations.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        df : int or float
            degrees of freedom. By default this is None, and the df from the
            one sample or paired ttest is used, ``df = nobs1 - 1``
        alternative : string, 'two-sided' (default), 'larger', 'smaller'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test. The one-sided test can be
            either 'larger', 'smaller'.
            .

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

       '''
        return ttest_power(effect_size, nobs, alpha, df=df,
                           alternative=alternative)

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, nobs=None, alpha=None, power=None,
                    alternative='two-sided'):
        '''solve for any one parameter of the power of a one sample t-test

        for the one sample t-test the keywords are:
            effect_size, nobs, alpha, power

        Exactly one needs to be ``None``, all others need numeric values.

        This test can also be used for a paired t-test, where effect size is
        defined in terms of the mean difference, and nobs is the number of
        pairs.

        Parameters
        ----------
        effect_size : float
            standardized effect size, mean divided by the standard deviation.
            effect size has to be positive.
        nobs : int or float
            sample size, number of observations.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        alternative : string, 'two-sided' (default) or 'one-sided'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test.
            'one-sided' assumes we are in the relevant tail.

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remainding parameters.


        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``fsolve``. If it fails to find a
        root, then for alpha or power ``brentq`` is used.
        However, there can still be cases where this fails.
        If it becomes necessary, then we will add options to control the root
        finding in future.

        '''
        return super(TTestPower, self).solve_power(effect_size=effect_size,
                                                      nobs=nobs,
                                                      alpha=alpha,
                                                      power=power,
                                                      alternative=alternative)

class TTestIndPower(Power):
    '''Statistical Power calculations for t-test for two independent sample

    currently only uses pooled variance

    '''


    def power(self, effect_size, nobs1, alpha, ratio=1, df=None,
              alternative='two-sided'):
        '''Calculate the power of a t-test for two independent sample

        Parameters
        ----------
        effect_size : float
            standardized effect size, difference between the two means divided
            by the standard deviation. `effect_size` has to be positive.
        nobs1 : int or float
            number of observations of sample 1. The number of observations of
            sample two is ratio times the size of sample 1,
            i.e. ``nobs2 = nobs1 * ratio``
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        ratio : float
            ratio of the number of observations in sample 2 relative to
            sample 1. see description of nobs1
            The default for ratio is 1; to solve for ratio given the other
            arguments, it has to be explicitly set to None.
        df : int or float
            degrees of freedom. By default this is None, and the df from the
            ttest with pooled variance is used, ``df = (nobs1 - 1 + nobs2 - 1)``
        alternative : string, 'two-sided' (default), 'larger', 'smaller'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test. The one-sided test can be
            either 'larger', 'smaller'.

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        '''

        nobs2 = nobs1*ratio
        #pooled variance
        if df is None:
            df = (nobs1 - 1 + nobs2 - 1)

        nobs = 1./ (1. / nobs1 + 1. / nobs2)
        return ttest_power(effect_size, nobs, alpha, df=df,
                           alternative=alternative)

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, nobs1=None, alpha=None, power=None,
                    ratio=1., alternative='two-sided'):
        '''solve for any one parameter of the power of a two sample t-test

        for t-test the keywords are:
            effect_size, nobs1, alpha, power, ratio

        exactly one needs to be ``None``, all others need numeric values

        Parameters
        ----------
        effect_size : float
            standardized effect size, difference between the two means divided
            by the standard deviation. `effect_size` has to be positive.
        nobs1 : int or float
            number of observations of sample 1. The number of observations of
            sample two is ratio times the size of sample 1,
            i.e. ``nobs2 = nobs1 * ratio``
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        ratio : float
            ratio of the number of observations in sample 2 relative to
            sample 1. see description of nobs1
            The default for ratio is 1; to solve for ratio given the other
            arguments it has to be explicitly set to None.
        alternative : string, 'two-sided' (default), 'larger', 'smaller'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test. The one-sided test can be
            either 'larger', 'smaller'.

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.


        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``fsolve``. If it fails to find a
        root, then for alpha or power ``brentq`` is used.
        However, there can still be cases where this fails.
        If it becomes necessary, then we will add options to control the root
        finding in future.

        '''
        return super(TTestIndPower, self).solve_power(effect_size=effect_size,
                                                      nobs1=nobs1,
                                                      alpha=alpha,
                                                      power=power,
                                                      ratio=ratio,
                                                      alternative=alternative)

class NormalIndPower(Power):
    '''Statistical Power calculations for z-test for two independent samples.

    currently only uses pooled variance

    '''


    def power(self, effect_size, nobs1, alpha, ratio=1,
              alternative='two-sided'):
        '''Calculate the power of a t-test for two independent sample

        Parameters
        ----------
        effect_size : float
            standardized effect size, difference between the two means divided
            by the standard deviation. effect size has to be positive.
        nobs1 : int or float
            number of observations of sample 1. The number of observations of
            sample two is ratio times the size of sample 1,
            i.e. ``nobs2 = nobs1 * ratio``
            ``ratio`` can be set to zero in order to get the power for a
            one sample test.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        ratio : float
            ratio of the number of observations in sample 2 relative to
            sample 1. see description of nobs1
            The default for ratio is 1; to solve for ratio given the other
            arguments it has to be explicitly set to None.
        alternative : string, 'two-sided' (default), 'larger', 'smaller'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test. The one-sided test can be
            either 'larger', 'smaller'.

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        '''

        if ratio > 0:
            nobs2 = nobs1*ratio
            #equivalent to nobs = n1*n2/(n1+n2)=n1*ratio/(1+ratio)
            nobs = 1./ (1. / nobs1 + 1. / nobs2)
        else:
            nobs = nobs1
        return normal_power(effect_size, nobs, alpha, alternative=alternative)

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, nobs1=None, alpha=None, power=None,
                    ratio=1., alternative='two-sided'):
        '''solve for any one parameter of the power of a two sample z-test

        for z-test the keywords are:
            effect_size, nobs1, alpha, power, ratio

        exactly one needs to be ``None``, all others need numeric values

        Parameters
        ----------
        effect_size : float
            standardized effect size, difference between the two means divided
            by the standard deviation.
            If ratio=0, then this is the standardized mean in the one sample
            test.
        nobs1 : int or float
            number of observations of sample 1. The number of observations of
            sample two is ratio times the size of sample 1,
            i.e. ``nobs2 = nobs1 * ratio``
            ``ratio`` can be set to zero in order to get the power for a
            one sample test.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        ratio : float
            ratio of the number of observations in sample 2 relative to
            sample 1. see description of nobs1
            The default for ratio is 1; to solve for ration given the other
            arguments it has to be explicitly set to None.
        alternative : string, 'two-sided' (default), 'larger', 'smaller'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test. The one-sided test can be
            either 'larger', 'smaller'.

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.


        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``fsolve``. If it fails to find a
        root, then for alpha or power ``brentq`` is used.
        However, there can still be cases where this fails.
        If it becomes necessary, then we will add options to control the root
        finding in future.

        '''
        return super(NormalIndPower, self).solve_power(effect_size=effect_size,
                                                      nobs1=nobs1,
                                                      alpha=alpha,
                                                      power=power,
                                                      ratio=ratio,
                                                      alternative=alternative)

class GofChisquarePower(Power):
    '''Statistical Power calculations for one sample chisquare test

    '''

    def power(self, effect_size, nobs, n_bins, alpha, ddof=0):
              #alternative='two-sided'):
        '''Calculate the power of a chisquare test for one sample

        Only two-sided alternative is implemented

        Parameters
        ----------
        effect_size : float
            standardized effect size, according to Cohen's definition.
            see :func:`statsmodels.stats.gof.chisquare_effectsize`
        nobs : int or float
            sample size, number of observations.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        n_bins : int
            number of bins or cells in the distribution.

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

       '''
        from statsmodels.stats.gof import chisquare_power
        return chisquare_power(effect_size, nobs, n_bins, alpha, ddof=0)

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, nobs=None, alpha=None,
                    power=None, n_bins=2):
        '''solve for any one parameter of the power of a one sample chisquare-test

        for the one sample chisquare-test the keywords are:
            effect_size, nobs, alpha, power

        Exactly one needs to be ``None``, all others need numeric values.

        n_bins needs to be defined, a default=2 is used.


        Parameters
        ----------
        effect_size : float
            standardized effect size, according to Cohen's definition.
            see :func:`statsmodels.stats.gof.chisquare_effectsize`
        nobs : int or float
            sample size, number of observations.
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        n_bins : int
            number of bins or cells in the distribution

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remaining parameters.


        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``fsolve``. If it fails to find a
        root, then for alpha or power ``brentq`` is used.
        However, there can still be cases where this fails.
        If it becomes necessary, then we will add options to control the root
        finding in future.

        '''
        return super(GofChisquarePower, self).solve_power(effect_size=effect_size,
                                                      nobs=nobs,
                                                      n_bins=n_bins,
                                                      alpha=alpha,
                                                      power=power)

class _GofChisquareIndPower(Power):
    '''Statistical Power calculations for chisquare goodness-of-fit test

    TODO: this is not working yet
          for 2sample case need two nobs in function
          no one-sided chisquare test, is there one? use normal distribution?
          -> drop one-sided options?
    '''


    def power(self, effect_size, nobs1, alpha, ratio=1,
              alternative='two-sided'):
        '''Calculate the power of a chisquare for two independent sample

        Parameters
        ----------
        effect_size : float
            standardize effect size, difference between the two means divided
            by the standard deviation. effect size has to be positive.
        nobs1 : int or float
            number of observations of sample 1. The number of observations of
            sample two is ratio times the size of sample 1,
            i.e. ``nobs2 = nobs1 * ratio``
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        ratio : float
            ratio of the number of observations in sample 2 relative to
            sample 1. see description of nobs1
            The default for ratio is 1; to solve for ration given the other
            arguments it has to be explicitely set to None.
        alternative : string, 'two-sided' (default) or 'one-sided'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test.
            'one-sided' assumes we are in the relevant tail.

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

        '''

        nobs2 = nobs1*ratio
        #equivalent to nobs = n1*n2/(n1+n2)=n1*ratio/(1+ratio)
        nobs = 1./ (1. / nobs1 + 1. / nobs2)
        return normal_power(effect_size, nobs, alpha, alternative=alternative)

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, nobs1=None, alpha=None, power=None,
                    ratio=1., alternative='two-sided'):
        '''solve for any one parameter of the power of a two sample z-test

        for z-test the keywords are:
            effect_size, nobs1, alpha, power, ratio

        exactly one needs to be ``None``, all others need numeric values

        Parameters
        ----------
        effect_size : float
            standardize effect size, difference between the two means divided
            by the standard deviation.
        nobs1 : int or float
            number of observations of sample 1. The number of observations of
            sample two is ratio times the size of sample 1,
            i.e. ``nobs2 = nobs1 * ratio``
        alpha : float in interval (0,1)
            significance level, e.g. 0.05, is the probability of a type I
            error, that is wrong rejections if the Null Hypothesis is true.
        power : float in interval (0,1)
            power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.
        ratio : float
            ratio of the number of observations in sample 2 relative to
            sample 1. see description of nobs1
            The default for ratio is 1; to solve for ration given the other
            arguments it has to be explicitely set to None.
        alternative : string, 'two-sided' (default) or 'one-sided'
            extra argument to choose whether the power is calculated for a
            two-sided (default) or one sided test.
            'one-sided' assumes we are in the relevant tail.

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remainding parameters.


        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``fsolve``. If it fails to find a
        root, then for alpha or power ``brentq`` is used.
        However, there can still be cases where this fails.
        If it becomes necessary, then we will add options to control the root
        finding in future.

        '''
        return super(_GofChisquareIndPower, self).solve_power(effect_size=effect_size,
                                                      nobs1=nobs1,
                                                      alpha=alpha,
                                                      power=power,
                                                      ratio=ratio,
                                                      alternative=alternative)

#shortcut functions
tt_solve_power = TTestPower().solve_power
tt_ind_solve_power = TTestIndPower().solve_power
zt_ind_solve_power = NormalIndPower().solve_power

if __name__ == '__main__':
    effect_size, alpha, power = 0.5, 0.05, 0.8

    ttest_pow = TTestPower()
    print '\nroundtrip - root with respect to all variables'
    print '\n       calculated, desired'

    nobs_p = ttest_pow.solve_power(effect_size=effect_size, nobs=None, alpha=alpha, power=power)
    print 'nobs  ', nobs_p
    print 'effect', ttest_pow.solve_power(effect_size=None, nobs=nobs_p, alpha=alpha, power=power), effect_size

    print 'alpha ', ttest_pow.solve_power(effect_size=effect_size, nobs=nobs_p, alpha=None, power=power), alpha
    print 'power  ', ttest_pow.solve_power(effect_size=effect_size, nobs=nobs_p, alpha=alpha, power=None), power

    print '\nroundtrip - root with respect to all variables'
    print '\n       calculated, desired'

    print 'nobs  ', tt_solve_power(effect_size=effect_size, nobs=None, alpha=alpha, power=power), nobs_p
    print 'effect', tt_solve_power(effect_size=None, nobs=nobs_p, alpha=alpha, power=power), effect_size

    print 'alpha ', tt_solve_power(effect_size=effect_size, nobs=nobs_p, alpha=None, power=power), alpha
    print 'power  ', tt_solve_power(effect_size=effect_size, nobs=nobs_p, alpha=alpha, power=None), power

    print '\none sided'
    nobs_p1 = tt_solve_power(effect_size=effect_size, nobs=None, alpha=alpha, power=power, alternative='1s')
    print 'nobs  ', nobs_p1
    print 'effect', tt_solve_power(effect_size=None, nobs=nobs_p1, alpha=alpha, power=power, alternative='1s'), effect_size
    print 'alpha ', tt_solve_power(effect_size=effect_size, nobs=nobs_p1, alpha=None, power=power, alternative='1s'), alpha
    print 'power  ', tt_solve_power(effect_size=effect_size, nobs=nobs_p1, alpha=alpha, power=None, alternative='1s'), power

    #start_ttp = dict(effect_size=0.01, nobs1=10., alpha=0.15, power=0.6)

    ttind_solve_power = TTestIndPower().solve_power

    print '\nroundtrip - root with respect to all variables'
    print '\n       calculated, desired'

    nobs_p2 = ttind_solve_power(effect_size=effect_size, nobs1=None, alpha=alpha, power=power)
    print 'nobs  ', nobs_p2
    print 'effect', ttind_solve_power(effect_size=None, nobs1=nobs_p2, alpha=alpha, power=power), effect_size

    print 'alpha ', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=None, power=power), alpha
    print 'power  ', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=alpha, power=None), power
    print 'ratio  ', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=alpha, power=power, ratio=None), 1

    print '\ncheck ratio'
    print 'smaller power', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=alpha, power=0.7, ratio=None), '< 1'
    print 'larger power ', ttind_solve_power(effect_size=effect_size, nobs1=nobs_p2, alpha=alpha, power=0.9, ratio=None), '> 1'
