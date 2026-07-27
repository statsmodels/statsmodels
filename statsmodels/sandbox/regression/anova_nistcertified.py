"""calculating anova and verifying with NIST test data

compares my implementations, stats.f_oneway and anova using statsmodels.OLS
"""

from statsmodels.compat.python import lmap

from pathlib import Path

import numpy as np
from scipy import stats

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from .try_ols_anova import data2dummy

filenameli = [
    "SiRstv.dat",
    "SmLs01.dat",
    "SmLs02.dat",
    "SmLs03.dat",
    "AtmWtAg.dat",
    "SmLs04.dat",
    "SmLs05.dat",
    "SmLs06.dat",
    "SmLs07.dat",
    "SmLs08.dat",
    "SmLs09.dat",
]


def getnist(filename):
    here = Path(__file__).parent
    fname = Path(here).joinpath("data", filename).resolve()
    with Path(fname).open(encoding="utf-8") as fd:
        content = fd.read().split("\n")
    [line.split() for line in content[60:]]
    certified = [line.split() for line in content[40:48] if line]
    dataf = np.loadtxt(fname, skiprows=60)
    y, x = dataf.T
    y = y.astype(int)
    caty = np.unique(y)
    f = float(certified[0][-1])
    R2 = float(certified[2][-1])
    resstd = float(certified[4][-1])
    dfbn = int(certified[0][-4])
    dfwn = int(certified[1][-3])
    prob = stats.f.sf(f, dfbn, dfwn)
    return (y, x, np.array([f, prob, R2, resstd]), certified, caty)


def anova_oneway(y, x, seq=0):
    yrvs = y[:, np.newaxis]
    xrvs = x[:, np.newaxis] - x.mean()
    from .try_catdata import groupsstats_dummy

    meang, varg, xdevmeangr, countg = groupsstats_dummy(yrvs[:, :1], xrvs[:, :1])
    sswn = np.dot(xdevmeangr.T, xdevmeangr)
    ssbn = np.dot((meang - xrvs.mean()) ** 2, countg.T)
    nobs = yrvs.shape[0]
    ncat = meang.shape[1]
    dfbn = ncat - 1
    dfwn = nobs - ncat
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    f = msb / msw
    prob = stats.f.sf(f, dfbn, dfwn)
    R2 = ssbn / (sswn + ssbn)
    resstd = np.sqrt(msw)

    def _fix2scalar(z):
        if np.shape(z) == (1, 1):
            return z[0, 0]
        else:
            return z

    f, prob, R2, resstd = lmap(_fix2scalar, (f, prob, R2, resstd))
    return (f, prob, R2, resstd)


def anova_ols(y, x):
    X = add_constant(data2dummy(x), prepend=False)
    res = OLS(y, X).fit()
    return (res.fvalue, res.f_pvalue, res.rsquared, np.sqrt(res.mse_resid))


if __name__ == "__main__":
    print("\n using new ANOVA anova_oneway")
    print("f, prob, R2, resstd")
    for fn in filenameli:
        print(fn)
        y, x, cert, certified, caty = getnist(fn)
        res = anova_oneway(y, x)
        rtol = {"SmLs08.dat": 0.027, "SmLs07.dat": 0.0017, "SmLs09.dat": 0.0001}.get(
            fn, 1e-07
        )
        np.testing.assert_allclose(np.array(res), cert, rtol=rtol)
    print("\n using stats ANOVA f_oneway")
    for fn in filenameli:
        print(fn)
        y, x, cert, certified, caty = getnist(fn)
        xlist = [x[y == ii] for ii in caty]
        res = stats.f_oneway(*xlist)
        print(np.array(res) - cert[:2])
    print("\n using statsmodels.OLS")
    print("f, prob, R2, resstd")
    for fn in filenameli[:]:
        print(fn)
        y, x, cert, certified, caty = getnist(fn)
        res = anova_ols(x, y)
        print(np.array(res) - cert)
