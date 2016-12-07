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
from .multivariate_ols import multivariate_stats


class CanCorr(Model):
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
    corr_values : array
        The canonical correlation values
    x_cancoeff: array
        The canonical coefficients for exog
    y_cancoeff: array
        The canonical coeefficients for endog

    .. [1] http://numerical.recipes/whp/notes/CanonCorrBySVD.pdf
    .. [2] http://www.csun.edu/~ata20315/psy524/docs/Psy524%20Lecture%208%20CC.pdf
    .. [3] http://www.mathematica-journal.com/2014/06/canonical-correlation-analysis/
    """
    def __init__(self, endog, exog, missing='none', hasconst=None, **kwargs):
        super(CanCorr, self).__init__(endog, exog, **kwargs)

    def fit(self, tolerance=1e-8):
        """Fit the model

        Parameters
        ----------
        tolerance : float
            eigenvalue tolerance, values smaller than which is considered 0
        Returns
        -------

        """
        nobs, k_yvar = self.endog.shape
        nobs, k_xvar = self.exog.shape
        k = np.min([k_yvar, k_xvar])

        x = np.array(self.exog)
        x = x - x.mean(0)
        y = np.array(self.endog)
        y = y - y.mean(0)

        ux, sx, vx = svd(x, 0)
        # vx_ds = vx.T divided by sx
        vx_ds = vx.T
        mask = sx > tolerance
        vx_ds[:, mask] /= sx[mask]
        uy, sy, vy = svd(y, 0)
        # vy_ds = vy.T divided by sy
        vy_ds = vy.T
        mask = sy > tolerance
        vy_ds[:, mask] /= sy[mask]
        u, s, v = svd(ux.T.dot(uy), 0)

        # Correct any roundoff
        self.cor_values= np.array([max(0, min(s[i], 1))
                                   for i in range(len(s))])

        self.x_cancoef = vx_ds.dot(u[:, :k])
        self.y_cancoef = vy_ds.dot(v.T[:, :k])

        return CanCorrResults(self)


class CanCorrResults(object):
    """
    Canonical correlation results class

    Parameters
    ----------
    fitted_cancorr : fitted CanCorr object

    Attributes
    -----------
    stats : DataFrame
        Contain statistical tests results for each canonical correlation
    stats_mv : DataFrame
        Contain the multivariate statistical tests results
    """
    def __init__(self, fitted_cancorr):
        self.cor_values = fitted_cancorr.cor_values
        self.x_cancoef = fitted_cancorr.x_cancoef
        self.y_cancoef = fitted_cancorr.y_cancoef
        self.nobs, self.k_yvar = fitted_cancorr.endog.shape
        nobs, self.k_xvar = fitted_cancorr.exog.shape

    def f_test(self):
        # Approximate F test using Wilk's lambda and other statistics
        k_yvar = self.k_yvar
        k_xvar = self.k_xvar
        nobs = self.nobs
        self.stats = pd.DataFrame()
        eigenvals = np.power(self.cor_values, 2)
        prod = 1
        for i in range(len(eigenvals) - 1, -1, -1):
            prod *= 1 - eigenvals[i]
            p = k_yvar - i
            q = k_xvar - i
            r = (nobs - k_yvar - 1) - (p - q + 1) / 2
            u = (p * q - 2) / 4
            df1 = p * q
            if p**2 + q**2 - 5 > 0:
                t = np.sqrt(((p*q)**2 - 4) / (p**2 + q**2 - 5))
            else:
                t = 1
            df2 = r * t - 2 * u
            lmd = np.power(prod, 1 / t)
            F = (1 - lmd) / lmd * df2 / df1
            self.stats.loc[i, 'Canonical Correlation'] = self.cor_values[i]
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

        # Multivariate tests (remember x has mean removed)
        self.stats_mv = multivariate_stats(eigenvals,
                                           k_yvar, k_xvar, nobs - k_xvar - 1)
    def __str__(self):
        return self.summary().__str__()

    def summary(self):
        summ = summary2.Summary()
        summ.add_title('Cancorr results')
        summ.add_df(self.stats)
        summ.add_dict({'': ''})
        summ.add_dict({'Multivariate Statistics and F Approximations': ''})
        summ.add_df(self.stats_mv)
        return summ
