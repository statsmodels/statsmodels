import numpy as np
from scipy import stats


class Runs(object):
    """
    Class for runs in a binary sequence

    Parameters
    ----------
    x : array_like, 1d
        data array,

    Notes
    -----
    This was written as a more general class for runs. This has some redundant
    calculations when only the runs_test is used.

    TODO: make it lazy

    The runs test could be generalized to more than 1d if there is a use case
    for it.

    This should be extended once I figure out what the distribution of runs
    of any length k is.

    The exact distribution for the runs test is also available but not yet
    verified.
    """

    def __init__(self, x):
        self.x = np.asarray(x)

        runstart = np.nonzero(np.diff(np.r_[[-np.inf], x, [np.inf]]))[0]
        self.runstart = runstart
        self.runs = runs = np.diff(runstart)
        self.runs_sign = runs_sign = x[runstart[:-1]]
        self.runs_pos = runs[runs_sign == 1]
        self.runs_neg = runs[runs_sign == 0]
        self.runs_freqs = np.bincount(runs)
        self.n_runs = len(self.runs)
        self.n_pos = (x == 1).sum()

    def runs_test(self, correction=True):
        """
        Basic version of runs test

        Parameters
        ----------
        correction: bool
            Following the SAS manual, for samplesize below 50, the test
            statistic is corrected by 0.5. This can be turned off with
            correction=False, and was included to match R, tseries, which
            does not use any correction.

        pvalue based on normal distribution, with integer correction
        """
        self.npo = npo = (self.runs_pos).sum()
        self.nne = nne = (self.runs_neg).sum()

        n = npo + nne
        npn = npo * nne
        rmean = 2. * npn / n + 1
        rvar = 2. * npn * (2.*npn - n) / n**2. / (n-1.)
        rstd = np.sqrt(rvar)
        rdemean = self.n_runs - rmean
        if n >= 50 or not correction:
            z = rdemean
        else:
            if rdemean > 0.5:
                z = rdemean - 0.5
            elif rdemean < 0.5:
                z = rdemean + 0.5
            else:
                z = 0.

        z /= rstd
        pval = 2 * stats.norm.sf(np.abs(z))
        return z, pval


def runstest_1samp(x, cutoff='mean', correction=True):
    """use runs test on binary discretized data above/below cutoff

    Parameters
    ----------
    x : array_like
        data, numeric
    cutoff : {'mean', 'median'} or number
        This specifies the cutoff to split the data into large and small
        values.
    correction: bool
        Following the SAS manual, for samplesize below 50, the test
        statistic is corrected by 0.5. This can be turned off with
        correction=False, and was included to match R, tseries, which
        does not use any correction.

    Returns
    -------
    z_stat : float
        test statistic, asymptotically normally distributed
    p-value : float
        p-value, reject the null hypothesis if it is below an type 1 error
        level, alpha .
    """

    if cutoff == 'mean':
        cutoff = np.mean(x)
    elif cutoff == 'median':
        cutoff = np.median(x)

    xindicator = (x >= cutoff).astype(int)
    return Runs(xindicator).runs_test(correction=correction)


def runstest_2samp(x, y=None, groups=None, correction=True):
    """
    Wald-Wolfowitz runstest for two samples

    This tests whether two samples come from the same distribution.

    Parameters
    ----------
    x : array_like
        data, numeric, contains either one group, if y is also given, or
        both groups, if additionally a group indicator is provided
    y : array_like (optional)
        data, numeric
    groups : array_like
        group labels or indicator the data for both groups is given in a
        single 1-dimensional array, x. If group labels are not [0,1], then
    correction: bool
        Following the SAS manual, for samplesize below 50, the test
        statistic is corrected by 0.5. This can be turned off with
        correction=False, and was included to match R, tseries, which
        does not use any correction.

    Returns
    -------
    z_stat : float
        test statistic, asymptotically normally distributed
    p-value : float
        p-value, reject the null hypothesis if it is below an type 1 error
        level, alpha .

    Notes
    -----
    Wald-Wolfowitz runs test.

    If there are ties, then then the test statistic and p-value that is
    reported, is based on the higher p-value between sorting all tied
    observations of the same group

    This test is intended for continuous distributions
    SAS has treatment for ties, but not clear, and sounds more complicated
    (minimum and maximum possible runs prevent use of argsort)
    (maybe it's not so difficult, idea: add small positive noise to first
    one, run test, then to the other, run test, take max(?) p-value - DONE
    This gives not the minimum and maximum of the number of runs, but should
    be close. Not true, this is close to minimum but far away from maximum.
    maximum number of runs would use alternating groups in the ties.)
    Maybe adding random noise would be the better approach.

    SAS has exact distribution for sample size <=30, doesn't look standard
    but should be easy to add.

    currently two-sided test only

    This has not been verified against a reference implementation. In a short
    Monte Carlo simulation where both samples are normally distribute, the test
    seems to be correctly sized for larger number of observations (30 or
    larger), but conservative (i.e. reject less often than nominal) with a
    sample size of 10 in each group.

    See Also
    --------
    runs_test_1samp
    Runs
    RunsProb
    """
    x = np.asarray(x)
    if y is not None:
        y = np.asarray(y)
        groups = np.concatenate((np.zeros(len(x)), np.ones(len(y))))
        # note reassigning x
        x = np.concatenate((x, y))
        gruni = np.arange(2)
    elif groups is not None:
        gruni = np.unique(groups)
        if gruni.size != 2:
            raise ValueError('not exactly two groups specified')
        # TODO: require groups to be numeric ???
    else:
        raise ValueError('either y or groups is necessary')

    xargsort = np.argsort(x)
    # check for ties
    x_sorted = x[xargsort]
    x_diff = np.diff(x_sorted)  # used for detecting and handling ties

    if x_diff.min() == 0:
        # TODO: warn?
        x_mindiff = x_diff[x_diff > 0].min()
        eps = x_mindiff/2.
        xx = x.copy()  # don't change original, just in case

        xx[groups == gruni[0]] += eps
        xargsort = np.argsort(xx)
        xindicator = groups[xargsort]
        z0, p0 = Runs(xindicator).runs_test(correction=correction)

        xx[groups == gruni[0]] -= eps   # restore xx = x
        xx[groups == gruni[1]] += eps
        xargsort = np.argsort(xx)
        xindicator = groups[xargsort]
        z1, p1 = Runs(xindicator).runs_test(correction=correction)

        idx = np.argmax([p0, p1])
        return [z0, z1][idx], [p0, p1][idx]

    else:
        xindicator = groups[xargsort]
        return Runs(xindicator).runs_test(correction=correction)
