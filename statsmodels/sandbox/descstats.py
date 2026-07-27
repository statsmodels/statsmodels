"""
Glue for returning descriptive statistics.
"""

import os
from pathlib import Path

import numpy as np
from scipy import stats

from statsmodels.stats.descriptivestats import sign_test


def descstats(data, cols=None, axis=0):
    """
    Prints descriptive statistics for one or multiple variables.

    Parameters
    ----------
    data: numpy array
        `x` is the data

    v: list, optional
        A list of the column number of variables.
        Default is all columns.

    axis: 1 or 0
        axis order of data.  Default is 0 for column-ordered data.

    Examples
    --------
    >>> descstats(data.exog,v=['x_1','x_2','x_3'])
    """
    x = np.array(data)
    if cols is None:
        x = x[:, None]
    if cols is None and x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        desc = (
            "\n            ---------------------------------------------\n            Univariate Descriptive Statistics\n            ---------------------------------------------\n\n            Var. Name   %(name)12s\n            ----------\n            Obs.          %(nobs)22i  Range                  %(range)22s\n            Sum of Wts.   %(sum)22s  Coeff. of Variation     %(coeffvar)22.4g\n            Mode          %(mode)22.4g  Skewness                %(skewness)22.4g\n            Repeats       %(nmode)22i  Kurtosis                %(kurtosis)22.4g\n            Mean          %(mean)22.4g  Uncorrected SS          %(uss)22.4g\n            Median        %(median)22.4g  Corrected SS            %(ss)22.4g\n            Variance      %(variance)22.4g  Sum Observations        %(sobs)22.4g\n            Std. Dev.     %(stddev)22.4g\n            "
            % {
                "name": cols,
                "sum": "N/A",
                "nobs": len(x),
                "mode": stats.mode(x)[0][0],
                "nmode": stats.mode(x)[1][0],
                "mean": x.mean(),
                "median": np.median(x),
                "range": "(" + str(x.min()) + ", " + str(x.max()) + ")",
                "variance": x.var(),
                "stddev": x.std(),
                "coeffvar": stats.variation(x),
                "skewness": stats.skew(x),
                "kurtosis": stats.kurtosis(x),
                "uss": np.sum(x**2, axis=0),
                "ss": np.sum((x - x.mean()) ** 2, axis=0),
                "sobs": np.sum(x),
            }
        )
        desc += (
            "\n\n        Percentiles\n        -------------\n        1  %%          %12.4g\n        5  %%          %12.4g\n        10 %%          %12.4g\n        25 %%          %12.4g\n\n        50 %%          %12.4g\n\n        75 %%          %12.4g\n        90 %%          %12.4g\n        95 %%          %12.4g\n        99 %%          %12.4g\n        "
            % tuple(
                [
                    stats.scoreatpercentile(x, per)
                    for per in (1, 5, 10, 25, 50, 75, 90, 95, 99)
                ]
            )
        )
        t, p_t = stats.ttest_1samp(x, 0)
        M, p_M = sign_test(x)
        S, p_S = stats.wilcoxon(np.squeeze(x))
        desc += "\n\n        Tests of Location (H0: Mu0=0)\n        -----------------------------\n        Test                Statistic       Two-tailed probability\n        -----------------+-----------------------------------------\n        Student's t      |  t {:7.5f}   Pr > |t|   <{:.4f}\n        Sign             |  M {:8.2f}   Pr >= |M|  <{:.4f}\n        Signed Rank      |  S {:8.2f}   Pr >= |S|  <{:.4f}\n\n        ".format(
            t, p_t, M, p_M, S, p_S
        )
    elif x.shape[1] > 1:
        desc = (
            "\n    Var. Name   |     Obs.        Mean    Std. Dev.           Range\n    ------------+--------------------------------------------------------"
            + os.linesep
        )
        for var in range(x.shape[1]):
            xv = x[:, var]
            kwargs = {
                "name": var,
                "obs": len(xv),
                "mean": xv.mean(),
                "stddev": xv.std(),
                "range": "(" + str(xv.min()) + ", " + str(xv.max()) + ")" + os.linesep,
            }
            desc += (
                "%(name)15s %(obs)9i %(mean)12.4g %(stddev)12.4g %(range)20s" % kwargs
            )
    else:
        raise ValueError("data not understood")
    return desc


if __name__ == "__main__":
    import statsmodels.api as sm

    data = sm.datasets.longley.load()
    data.exog = sm.add_constant(data.exog, prepend=False)
    sum1 = descstats(data.exog)
    sum1a = descstats(data.exog[:, :1])
    if Path("./Econ724_PS_I_Data.csv").is_file():
        data2 = np.genfromtxt("./Econ724_PS_I_Data.csv", delimiter=",")
        sum2 = descstats(data2.ahe)
        sum3 = descstats(np.column_stack((data2.ahe, data2.yrseduc)))
        sum4 = descstats(np.column_stack([data2[_] for _ in data2.dtype.names]))
