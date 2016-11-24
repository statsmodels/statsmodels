"""Multivariate analysis of variance

author: Yichuan Liu
"""
from __future__ import print_function, division

import numpy as np
from numpy.linalg import eigvals, inv, det, pinv, matrix_rank
from scipy import stats
import pandas as pd

from statsmodels.compat.python import range
from statsmodels.compat.numpy import nanmean


class MANOVA(object):
    """
    Multivariate analysis of variance

    For Y = B * X
    Testing L * B * M = 0

    Parameters
    ----------
    X : array-like, n by m_x
        Independent variables (IV). Variables in columns, observations in rows
    Y : array-like, n by m_y
        Dependent variables (DV). Variables in columns, observations in rows
    L : array-like, at least 1 row (1 by m_x+1)
        Hypothesis to be tested. IV in columns, sub-hypothesis in rows. First
        column is intercept
    M : array-like
        Transform matrix. Default to be m_y by m_y identity matrix

    """
    def __init__(self, X, Y, L, M=None):
        n_sample, n_iv = X.shape
        n_sample1, n_dv = Y.shape
        if M is None:
            M = np.eye(n_dv)
        # Add intercept
        X = np.concatenate([np.ones([n_sample, 1]), X], axis=1)
        # H = M'(LB)'(L(X'X)^L')^(LB)M   (`^` denotes inverse)
        v = X.shape[0] - X.shape[1]
        B = pinv(X).dot(Y)

        # t1 = (LB)M
        t1 = L.dot(B).dot(M)

        # H = t1'L(X'X)^L't1
        t2 = L.dot(inv(X.T.dot(X))).dot(L.T)
        q = matrix_rank(t2)
        H = t1.T.dot(inv(t2)).dot(t1)

        # E = M'(Y'Y - B'(X'X)B)M
        E = X.dot(B)
        E = np.subtract(Y.T.dot(Y), E.T.dot(E))
        E = M.T.dot(E).dot(M)

        EH = np.add(E, H)
        p = matrix_rank(EH)

        # eigenvalues of (E + H)^H
        eigv2 = np.sort(eigvals(inv(EH).dot(H)))

        # eigenvalues of (E+H)^H
        eigv1 = np.array([i / (1 - i) for i in eigv2])


        s = np.min([p, q])
        m = (np.abs(p - q) - 1) / 2
        n = (v - p - 1) / 2

        results = {}
        results["Wilks’ lambda"] =  np.prod(1 - eigv2)

        results["Pillai’s trace"] = eigv2.sum()

        results["Hotelling-Lawley trace"] = eigv1.sum()

        results["Roy’s greatest root"] = eigv1.max()

        results = pd.DataFrame(pd.Series(results), columns=['value'])

        r = v - (p - q + 1)/2
        u = (p*q - 2) / 4
        df1 = p * q
        if p*p + q*q - 5 > 0:
            t = np.sqrt((p*p*q*q - 4) / (p*p + q*q - 5))
        else:
            t = 1
        df2 = r*t - 2*u
        lmd = results.loc["Wilks’ lambda", 'value']
        lmd = np.power(lmd, 1 / t)
        F = (1 - lmd) / lmd * df2 / df1
        results.loc["Wilks’ lambda", 'Num DF'] = df1
        results.loc["Wilks’ lambda", 'Den DF'] = df2
        results.loc["Wilks’ lambda", 'F Value'] = F
        pval = stats.f.sf(F, df1, df2)
        results.loc["Wilks’ lambda", 'Pr > F'] = pval


        V = results.loc["Pillai’s trace", 'value']
        df1 = s * (2*m + s + 1)
        df2 = s * (2*n + s + 1)
        F = df2 / df1 * V / (s - V)
        results.loc["Pillai’s trace", 'Num DF'] = df1
        results.loc["Pillai’s trace", 'Den DF'] = df2
        results.loc["Pillai’s trace", 'F Value'] = F
        pval = stats.f.sf(F, df1, df2)
        results.loc["Pillai’s trace", 'Pr > F'] = pval

        U = results.loc["Hotelling-Lawley trace", 'value']
        if n > 0:
            b = (p + 2*n) * (q + 2*n) / 2 / (2*n + 1) / (n - 1)
            df1 = p * q
            df2 = 4 + (p*q + 2) / (b - 1)
            c = (df2 - 2) / 2 / n
            F = df2 / df1 * U / c
        else:
            df1 = s * (2*m + s + 1)
            df2 = s * (s*n + 1)
            F = df2 / df1 / s * U
        results.loc["Hotelling-Lawley trace", 'Num DF'] = df1
        results.loc["Hotelling-Lawley trace", 'Den DF'] = df2
        results.loc["Hotelling-Lawley trace", 'F Value'] =  F
        pval = stats.f.sf(F, df1, df2)
        results.loc["Hotelling-Lawley trace", 'Pr > F'] =  pval

        sigma = results.loc["Roy’s greatest root", 'value']
        r = np.max([p, q])
        df1 = r
        df2 = v - r + q
        F = df2 / df1 * sigma
        results.loc["Roy’s greatest root", 'Num DF'] = df1
        results.loc["Roy’s greatest root", 'Den DF'] = df2
        results.loc["Roy’s greatest root", 'F Value'] = F
        pval = stats.f.sf(F, df1, df2)
        results.loc["Roy’s greatest root", 'Pr > F'] = pval
        self.results_ = results.iloc[:, [0, 3, 1, 2, 4]]

    @property
    def stats(self):
        print(self.results_)







