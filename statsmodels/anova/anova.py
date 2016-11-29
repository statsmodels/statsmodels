"""Analysis of variance (ANOVA)

Author: Yichuan Liu
"""

import numpy as np
from scipy import stats

def long_to_wide(x, factor1, factor2=None, subject=None):
    raise NotImplementedError

def anova_r1(x):
    """
    One-way repeated measures ANOVA

    .. [1] http://www.statisticshell.com/docs/onewayrmhand.pdf
    Parameters
    ----------
    x : 2-D array
        Each row is a subject and Each column is a time level
    factor
    subject

    Returns
    -------

    """
    ssw = x.var(axis=1).sum() * (x.shape[1] - 1)
    dfw = x.shape[0]  * (x.shape[1] - 1)

    ssm = x.mean(axis=0).var() * x.shape[0] * (x.shape[1] - 1)
    dfm = x.shape[1] - 1

    ssr = ssw - ssm
    dfr = dfw - dfm

    msm = ssm / dfm
    msr = ssr / dfr

    f = msm / msr
    pval = stats.f.sf(f, dfm, dfr)

    return (ssm, ssr, f, dfm, dfr, pval)

def anova_r2(x, factor1, factor2, subject=None):
    raise NotImplementedError

def anova_b1r1(x, between, within, subject=None):
    raise NotImplementedError
