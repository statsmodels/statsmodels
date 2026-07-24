# -*- coding: utf-8 -*-
"""Control Charts and Statistical Process Control

Created on Sun Dec 31 23:47:52 2017

Author: Josef Perktold


Some tools for building control charts and process monitoring.
The core should be tools for the statistics, estimation and control limits.
Control charts and monitoring interface are just for a basic example use case.

"""

from __future__ import division
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def update_std(std, nobs, y_new):
    raise NotImplementedError

class Holder(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

def control_limits_mean(y0, alpha=0.05, crit=None, distr='t'):

        endog = y0
        nobs = endog.shape[0]
        #index = np.arange(nobs)  # not used nor returned

        if crit is not None:
            crit = crit
        else:
            if distr == 't':
                crit = stats.t.isf(alpha/2, nobs-1)
            elif distr == 'n':
                crit = stats.norm.isf(alpha/2)
            else:
                raise ValueError('only t and normal for now' )

        center = endog.mean(0)
        std = endog.std(0, ddof=1)
        diff = crit * std
        upp = center + diff
        low = center - diff
        res = Holder(center=center,
                     std=std,
                     low=low,
                     upp=upp,
                     diff=diff)
        return res


class ControlChartMean(object):

    def __init__(self, y0, alpha=0.05, crit=None, distr='t'):

        self.endog = y0
        self.nobs = self.endog.shape[0]
        self.index = np.arange(self.nobs)

        if crit is not None:
            self.crit = crit
        else:
            if distr == 't':
                self.crit = stats.t.isf(alpha/2, self.nobs-1)
            elif distr == 'n':
                self.crit = stats.norm.isf(alpha/2)
            else:
                raise ValueError('only t and normal for now' )

        self.center = self.endog.mean(0)
        self.std = self.endog.std(0, ddof=1)
        self.diff = self.crit * self.std
        self.upp = self.center + self.diff
        self.low = self.center - self.diff

    def plot_chart(self, y_new, include_y0=True, ax=None):
        if include_y0 is True:
            y = np.concatenate((self.endog, y_new), axis=0)
        n1 = y_new.shape[0]
        idx_new = np.arange(self.index[-1] + 1, self.index[-1] + 1 + n1)
        idx = np.concatenate((self.index, idx_new))
        self.idx = idx

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if np.isscalar(self.upp) or True:
            ax.hlines(self.upp, idx[0], idx[-1], colors='k')
            ax.hlines(self.low, idx[0], idx[-1], colors='k')
            ax.hlines(self.center, idx[0], idx[-1], lw=2, colors='k')
        else:
            ax.plot(idx, self.upp, color='k')
            ax.plot(idx, self.low, color='k')
            ax.plot(idx, self.center, lw=2, color='k')

        out = (y > self.upp) | (y < self.low)
        ax.plot(idx, y, 'o-') #, color='b')
        if y.ndim > 1:
            idx_out = np.repeat(np.arange(y.shape[0])[:,None], y.shape[1], axis=1)[out]
        else:
            idx_out = idx[out]
        ax.plot(idx_out, y[out], 'o', color='red')
        ax.set_xlim(idx[0], idx[-1])
        ax.vlines(idx[self.nobs], *ax.get_ylim())
        ax.set_title('Mean Chart vectorized')
        return fig

    def update(self, y_new):
        n1 = y_new.shape[0]
        n0 = self.nobs
        self.center = (n0 * self.center + y_new.sum(0)) / (n0+n1)
        self.endog = np.concatenate((self.endog, y_new), axis=0)
        idx_new = np.arange(self.index[-1] + 1, self.index[-1] + 1 + n1)
        self.index = np.concatenate((self.index, idx_new))
        # no updating formula yet
        self.std = self.endog.std(0)
        self.nobs += n1
        self.diff = self.crit * self.std
        self.upp = self.center + self.diff
        self.low = self.center - self.diff


class ControlChartMvMean(ControlChartMean):
    """T2 control chart for multivariate mean
    """

    def __init__(self, y0, alpha=0.05, crit=None, distr='F'):

        self.endog = y0
        self.nobs, self.k_endog = self.endog.shape
        self.index = np.arange(self.nobs)

        if crit is not None:
            self.crit = crit
        else:
            if distr == 'F':
                self.crit = stats.f.isf(alpha/2, 1, self.nobs-1)
            elif distr == 'chi2':
                self.crit = stats.chi2.isf(alpha/2, 1)
            else:
                raise ValueError('only t and normal for now' )


        self.center = self.endog.mean(0)

        self.cov = np.cov(self.endog, rowvar=0, ddof=1)
        self.cov_inv = np.linalg.inv(self.cov)
        diff = self.endog - self.center
        distance = (diff.dot(self.cov_inv) * diff).sum(1)
        self.c = (self.nobs - 1)**2 / self.nobs
        self.statistic = distance / self.c
        df1 = self.k_endog / 2
        df2 = (self.nobs - self.k_endog - 1) / 2
        #self.upp = self.c * stats.beta.isf(alpha, df1, df2)
        self.upp = stats.beta.isf(alpha, df1, df2)
        self.low = 0

    def plot_chart(self, y_new, include_y0=True, ax=None):
        diff_new = y_new - self.center
        stat_new = (diff_new.dot(self.cov_inv) * diff_new).sum(1) / self.c
        df1 = self.k_endog / 2
        if include_y0 is True:
            y = np.concatenate((self.statistic, stat_new), axis=0)
        n1 = y_new.shape[0]
        idx_new = np.arange(self.index[-1] + 1, self.index[-1] + 1 + n1)
        idx = np.concatenate((self.index, idx_new))
        self.idx = idx

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if np.isscalar(self.upp) or True:
            ax.hlines(self.upp, idx[0], idx[-1], colors='k')
            ax.hlines(self.low, idx[0], idx[-1], colors='k')
            #ax.hlines(y, idx[0], idx[-1], lw=2, colors='k')
        else:
            ax.plot(idx, self.upp, color='k')
            ax.plot(idx, self.low, color='k')
            ax.plot(idx, self.statistic, lw=2, color='k')

        out = (y > self.upp) | (y < self.low)
        ax.plot(idx, y, 'o-') #, color='b')
        if y.ndim > 1:
            idx_out = np.repeat(np.arange(y.shape[0])[:,None], y.shape[1], axis=1)[out]
        else:
            idx_out = idx[out]
        ax.plot(idx_out, y[out], 'o', color='red')
        ax.set_xlim(idx[0], idx[-1])
        ax.vlines(idx[self.nobs], *ax.get_ylim())
        ax.set_title('Multivariate Mean Chart T2')
        return fig

class ControlChartEWMAMvMean(ControlChartMean):
    """T2 control chart for multivariate mean
    """

    def __init__(self, y0, weights, alpha=0.05, crit=None, distr='F'):

        self.endog = y0
        self.nobs, self.k_endog = self.endog.shape
        self.index = np.arange(self.nobs)
        self.w = w = weights

        if crit is not None:
            self.crit = crit
        else:
            if distr == 'F':
                self.crit = stats.f.isf(alpha/2, 1, self.nobs-1)
            elif distr == 'chi2':
                self.crit = stats.chi2.isf(alpha/2, 1)
            else:
                raise ValueError('only t and normal for now' )


        self.center = self.endog.mean(0)

        self.cov = np.cov(self.endog, rowvar=0, ddof=1)

        self.cov = np.cov(self.endog, rowvar=0, ddof=1)
        wt = w / (2. - w)
        self.cov_z = wt * self.cov * wt[:, None]
        self.cov_z_inv = np.linalg.inv(self.cov_z)
        z0 = self.endog.mean(0)
        self.z = z = self._generate_mewma(self.endog, z0, w)
        diff = np.asarray(z)
        distance = (diff.dot(self.cov_z_inv) * diff).sum(1)
        self.c = (self.nobs - 1)**2 / self.nobs
        self.statistic = distance / self.c
        df1 = self.k_endog / 2
        df2 = (self.nobs - self.k_endog - 1) / 2
        #self.upp = self.c * stats.beta.isf(alpha, df1, df2)
        self.upp = 0.5 #stats.beta.isf(alpha, df1, df2)
        self.low = 0

    def _generate_mewma(self, endog, z0, w):
        """using explicit loop for now
        """
        if endog.ndim == 2:
            n, k_endog = endog.shape
        elif endog.ndim == 0:
            n = 1
        z = [z0]
        for t in range(n):
            z.append(z[-1] + w * (endog[t] - z[-1]))
        return z[1:]


    def plot_chart(self, y_new, include_y0=True, ax=None):
        z = self._generate_mewma(y_new, self.z[-1], self.w)
        diff_new = np.asarray(z)
        stat_new = (diff_new.dot(self.cov_z_inv) * diff_new).sum(1) / self.c
        df1 = self.k_endog / 2
        if include_y0 is True:
            y = np.concatenate((self.statistic, stat_new), axis=0)
        n1 = y_new.shape[0]
        idx_new = np.arange(self.index[-1] + 1, self.index[-1] + 1 + n1)
        idx = np.concatenate((self.index, idx_new))
        self.idx = idx

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if np.isscalar(self.upp) or True:
            ax.hlines(self.upp, idx[0], idx[-1], colors='k')
            ax.hlines(self.low, idx[0], idx[-1], colors='k')
            #ax.hlines(y, idx[0], idx[-1], lw=2, colors='k')
        else:
            ax.plot(idx, self.upp, color='k')
            ax.plot(idx, self.low, color='k')
            ax.plot(idx, self.statistic, lw=2, color='k')

        out = (y > self.upp) | (y < self.low)
        ax.plot(idx, y, 'o-', alpha=0.5) #, color='b')
        if y.ndim > 1:
            idx_out = np.repeat(np.arange(y.shape[0])[:,None], y.shape[1], axis=1)[out]
        else:
            idx_out = idx[out]
        ax.plot(idx_out, y[out], 'o', color='red')
        ax.set_xlim(idx[0], idx[-1])
        ax.vlines(idx[self.nobs], *ax.get_ylim())
        ax.set_title('MEMWA Chart')
        return fig
