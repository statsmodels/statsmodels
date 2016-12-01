# -*- coding: utf-8 -*-

"""Canonical correlation analysis

author: Yichuan Liu
"""
from __future__ import print_function, division

from statsmodels.base.model import Model
import numpy as np
from numpy.linalg import svd
from scipy import stats
import pandas as pd
from statsmodels.iolib import summary2
from .glm import multivariate_stats

class Cancorr(Model):
    """
    Canonical correlation analysis using singluar value decomposition

    For matrices x and y, find projections x_cancoef and y_cancoef such that:
        x1 = x * x_cancoef, x1' * x1 is identity matrix
        y1 = y * y_cancoef, y1' * y1 is identity matrix
    and the correlation between x1 and y1 is maximized.

    Attributes
    -----------
    endog : array
        See Parameters.
    exog : array
        See Parameters.
    design_info : patsy.DesignInfo
        Contain design info for the independent variables if model is
        constructed using `from_formula`
    stats : DataFrame
        Contain statistical tests results for each canonical correlation
    multi_stats : DataFrame
        Contain the multivariate statistical tests results
    cancorr : array
        The canonical correlations
    x_cancoeff: array
        The canonical coefficients for exog
    y_cancoeff: array
        The canonical coeefficients for endog

    .. [1] http://numerical.recipes/whp/notes/CanonCorrBySVD.pdf
    .. [2] http://www.csun.edu/~ata20315/psy524/docs/Psy524%20Lecture%208%20CC.pdf
    .. [3] http://www.mathematica-journal.com/2014/06/canonical-correlation-analysis/
    """
    def __init__(self, endog, exog, design_info=None, **kwargs):
        self.design_info = design_info
        super(Cancorr, self).__init__(endog, exog)

    def fit(self):
        nobs, p = self.endog.shape
        nobs, q = self.exog.shape
        k = np.min([p, q])

        x = np.array(self.exog)
        x = x - x.mean(0)
        y = np.array(self.endog)
        y = y - y.mean(0)

        e = 1e-8  # eigenvalue tolerance, values smaller than e is considered 0
        ux, sx, vx = svd(x, 0)
        # vx_ds = vx.T divided by sx
        vx_ds = vx.T
        for i in range(len(sx)):
            if sx[i] > e:
                vx_ds[:, i] = vx_ds[:, i] / sx[i]
            else:
                break
        uy, sy, vy = svd(y, 0)
        # vy_ds = vy.T divided by sy
        vy_ds = vy.T
        for i in range(len(sy)):
            if sy[i] > e:
                vy_ds[:, i] = vy_ds[:, i] / sy[i]
            else:
                break
        u, s, v = svd(ux.T.dot(uy), 0)

        # Correct any roundoff
        self.cancorr = np.array([max(0, min(s[i], 1)) for i in range(len(s))])

        self.x_cancoef = vx_ds.dot(u[:, :k])
        self.y_cancoef = vy_ds.dot(v.T[:, :k])

        self.stats = pd.DataFrame()
        a = -(nobs - 1  - (p + q + 1)/2)
        eigenvals = np.power(self.cancorr, 2)
        prod = 1
        for i in range(len(eigenvals)-1, -1, -1):
            prod *= 1 - eigenvals[i]
            p1 = p - i
            q1 = q - i
            r = (nobs - q - 1) - (p1 - q1 + 1)/2
            u = (p1*q1 - 2) / 4
            df1 = p1 * q1
            if p1*p1 + q1*q1 - 5 > 0:
                t = np.sqrt((p1*p1*q1*q1 - 4) / (p1*p1 + q1*q1 - 5))
            else:
                t = 1
            df2 = r*t - 2*u
            lmd = np.power(prod,  1 / t)
            F = (1 - lmd) / lmd * df2 / df1
            self.stats.loc[i, "Wilks' lambda"] = prod
            self.stats.loc[i, 'Num DF'] = df1
            self.stats.loc[i, 'Den DF'] = df2
            self.stats.loc[i, 'F Value'] = F
            pval = stats.f.sf(F, df1, df2)
            self.stats.loc[i, 'Pr > F'] = pval
            '''
            # Wilk's Chi square test of each canonical correlation
            df = (p - i + 1) * (q - i + 1)
            chi2 = a * np.log(prod)
            pval = stats.chi2.sf(chi2, df)
            self.stats.loc[i, 'Canonical correlation'] = self.cancorr[i]
            self.stats.loc[i, 'Chi-square'] = chi2
            self.stats.loc[i, 'DF'] = df
            self.stats.loc[i, 'Pr > ChiSq'] = pval
            '''
        ind = self.stats.index.values[::-1]
        self.stats = self.stats.loc[ind, :]

        # Multivariate tests
        self.multi_stats = multivariate_stats(eigenvals,
                                              p, q, nobs - q - 1)

        return CancorrResults(self)

class CancorrResults(object):
    """
    Canonical correlation results class

    Parameters
    ----------

    cancorr_obj : Cancorr class

    """
    def __init__(self, cancorr_obj):
        self.cancorr = cancorr_obj.cancorr
        self.x_cancoef = cancorr_obj.x_cancoef
        self.y_cancoef = cancorr_obj.y_cancoef
        self.multi_stats = cancorr_obj.multi_stats
        self.stats = cancorr_obj.stats

    def __str__(self):
        return self.summary().__str__()

    def summary(self):
        summ = summary2.Summary()
        summ.add_title('Cancorr results')
        summ.add_df(self.stats)
        summ.add_dict({'': ''})
        summ.add_dict({'Multivariate Statistics and F Approximations': ''})
        summ.add_df(self.multi_stats)
        return summ
