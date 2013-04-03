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
from statsmodels.tools.rootfinding import brentq_expanding

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

def ftest_anova_power(effect_size, nobs, alpha, k_groups=2, df=None):
    '''power for ftest for one way anova with k equal sized groups

    nobs total sample size, sum over all groups

    should be general nobs observations, k_groups restrictions ???
    '''
    df_num = nobs - k_groups
    df_denom = k_groups - 1
    crit = stats.f.isf(alpha, df_denom, df_num)
    pow_ = stats.ncf.sf(crit, df_denom, df_num, effect_size**2 * nobs)
    return pow_#, crit

def ftest_power(effect_size, df_num, df_denom, alpha, ncc=1):
    '''power for ftest

    sample size is given implicitly by df_num

    set ncc=0 to match t-test, or f-test in LikelihoodModelResults
    ncc=1 matches the non-centrality parameter in R::pwr::pwr.f2.test

    ftest_power with ncc=0 should also be correct for f_test in regression
    models, with df_num and d_denom as defined there. (not verified yet)

    '''
    nc = effect_size**2 * (df_denom + df_num + ncc)
    crit = stats.f.isf(alpha, df_denom, df_num)
    pow_ = stats.ncf.sf(crit, df_denom, df_num, nc)
    return pow_ #, crit, nc


#class based implementation
#--------------------------

class Power(object):
    '''Statistical Power calculations, Base Class

    so far this could all be class methods
    '''

    def __init__(self):
        # used only for instance level start values
        self.start_ttp = dict(effect_size=0.01, nobs=10., alpha=0.15,
                              power=0.6, nobs1=10., ratio=1,
                              df_num=10, df_denom=3   # for FTestPower
                              )
        # TODO: nobs1 and ratio are for ttest_ind,
        #      need start_ttp for each test/class separately,
        # possible rootfinding problem for effect_size, starting small seems to
        # work
        from collections import defaultdict
        self.start_bqexp = defaultdict(dict)
        for key in ['nobs', 'nobs1', 'df_num', 'df_denom']:
            self.start_bqexp[key] = dict(low=2., start_upp=50.)
        for key in ['df_denom']:
            self.start_bqexp[key] = dict(low=1., start_upp=50.)
        for key in ['ratio']:
            self.start_bqexp[key] = dict(low=1e-8, start_upp=2)

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

        *attaches*

        cache_fit_res : list
            Cache of the result of the root finding procedure for the latest
            call to ``solve_power``, mainly for debugging purposes.
            The first element is the success indicator, one if successful.
            The remaining elements contain the return information of the up to
            three solvers that have been tried.


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
            fval = self._power_identity(**kwds)
            if np.isnan(fval):
                return np.inf
            else:
                return fval

        #TODO: I'm using the following so I get a warning when start_ttp is not defined
        try:
            start_value = self.start_ttp[key]
        except KeyError:
            start_value = 0.9
            print 'Warning: using default start_value for', key

        fit_kwds = self.start_bqexp[key]
        fit_res = []
        try:
            val, res = brentq_expanding(func, full_output=True, **fit_kwds)
            failed = False
            fit_res.append(res)
        except ValueError:
            failed = True
            fit_res.append(None)

        success = None
        if (not failed) and res.converged:
            success = 1
        else:
            # try backup
            #TODO: check more cases to make this robust
            val, infodict, ier, msg = optimize.fsolve(func, start_value,
                                                      full_output=True) #scalar
            fval = infodict['fvec']
            fit_res.append(infodict)
            if ier == 1 and np.abs(fval) < 1e-4 :
                success = 1
            else:
                #print infodict
                if key in ['alpha', 'power', 'effect_size']:
                    val, r = optimize.brentq(func, 1e-8, 1-1e-8,
                                             full_output=True) #scalar
                    success = 1 if r.converged else 0
                    fit_res.append(r)
                else:
                    success = 0

        if not success == 1:
            import warnings
            from statsmodels.tools.sm_exceptions import ConvergenceWarning
            warnings.warn('finding solution failed', ConvergenceWarning)

        #attach fit_res, for reading only, should be needed only for debugging
        fit_res.insert(0, success)
        self.cache_fit_res = fit_res
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
        # for debugging
        #print 'calling ttest power with', (effect_size, nobs, alpha, df, alternative)
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
            value solves the power equation given the remaining parameters.

        *attaches*

        cache_fit_res : list
            Cache of the result of the root finding procedure for the latest
            call to ``solve_power``, mainly for debugging purposes.
            The first element is the success indicator, one if successful.
            The remaining elements contain the return information of the up to
            three solvers that have been tried.

        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

        '''
        # for debugging
        #print 'calling ttest solve with', (effect_size, nobs, alpha, power, alternative)
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
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

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
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

        '''
        return super(NormalIndPower, self).solve_power(effect_size=effect_size,
                                                      nobs1=nobs1,
                                                      alpha=alpha,
                                                      power=power,
                                                      ratio=ratio,
                                                      alternative=alternative)


class FTestPower(Power):
    '''Statistical Power calculations for generic F-test

    '''

    def power(self, effect_size, df_num, df_denom, alpha, ncc=1):
        '''Calculate the power of a F-test.

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

        pow_ = ftest_power(effect_size, df_num, df_denom, alpha, ncc=1)
        #print effect_size, df_num, df_denom, alpha, pow_
        return pow_

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, df_num=None, df_denom=None,
                    nobs=None, alpha=None, power=None, ncc=1):
        '''solve for any one parameter of the power of a F-test

        for the one sample F-test the keywords are:
            effect_size, df_num, df_denom, alpha, power

        Exactly one needs to be ``None``, all others need numeric values.


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
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

        '''
        return super(FTestPower, self).solve_power(effect_size=effect_size,
                                                      df_num=df_num,
                                                      df_denom=df_denom,
                                                      alpha=alpha,
                                                      power=power)

class FTestAnovaPower(Power):
    '''Statistical Power calculations F-test for one factor balanced ANOVA

    '''

    def power(self, effect_size, nobs, alpha, k_groups=2):
        '''Calculate the power of a F-test for one factor ANOVA.

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
        k_groups : int or float
            number of groups in the ANOVA or k-sample comparison. Default is 2.

        Returns
        -------
        power : float
            Power of the test, e.g. 0.8, is one minus the probability of a
            type II error. Power is the probability that the test correctly
            rejects the Null Hypothesis if the Alternative Hypothesis is true.

       '''
        return ftest_anova_power(effect_size, nobs, alpha, k_groups=k_groups)

    #method is only added to have explicit keywords and docstring
    def solve_power(self, effect_size=None, nobs=None, alpha=None, power=None,
                    k_groups=2):
        '''solve for any one parameter of the power of a F-test

        for the one sample F-test the keywords are:
            effect_size, nobs, alpha, power

        Exactly one needs to be ``None``, all others need numeric values.


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

        Returns
        -------
        value : float
            The value of the parameter that was set to None in the call. The
            value solves the power equation given the remainding parameters.


        Notes
        -----
        The function uses scipy.optimize for finding the value that satisfies
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

        '''
        # update start values for root finding
        if not k_groups is None:
            self.start_ttp['nobs'] = k_groups * 10
            self.start_bqexp['nobs'] = dict(low=k_groups * 2,
                                            start_upp=k_groups * 10)
        # first attempt at special casing
        if effect_size is None:
            return self._solve_effect_size(effect_size=effect_size,
                                           nobs=nobs,
                                           alpha=alpha,
                                           k_groups=k_groups,
                                           power=power)

        return super(FTestAnovaPower, self).solve_power(effect_size=effect_size,
                                                      nobs=nobs,
                                                      alpha=alpha,
                                                      k_groups=k_groups,
                                                      power=power)

    def _solve_effect_size(self, effect_size=None, nobs=None, alpha=None,
                           power=None, k_groups=2):
        '''experimental, test failure in solve_power for effect_size
        '''
        def func(x):
            effect_size = x
            return self._power_identity(effect_size=effect_size,
                                          nobs=nobs,
                                          alpha=alpha,
                                          k_groups=k_groups,
                                          power=power)

        val, r = optimize.brentq(func, 1e-8, 1-1e-8, full_output=True)
        if not r.converged:
            print r
        return val


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
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

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
        the power equation. It first uses ``brentq`` with a prior search for
        bounds. If this fails to find a root, ``fsolve`` is used. If ``fsolve``
        also fails, then, for ``alpha``, ``power`` and ``effect_size``,
        ``brentq`` with fixed bounds is used. However, there can still be cases
        where this fails.

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
