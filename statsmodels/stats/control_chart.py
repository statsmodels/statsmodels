"""
Control Charts for Statistical Process Monitoring
---
All control charts will have Phase I and Phase II stages

Phase 1 is the 'base' period where process stability is assessed using
historical data, and phase 2 operates off of the parameters determined in
phase I for future monitoring.

It is possible to plot:
- Phase I only chart
- Phase II only chart
- Both phases

References
----------
- Multivariate Statistical Process Control Charts: An Overview
(Bersimis 2006): https://mpra.ub.uni-muenchen.de/6399/1/MPRA
- Multivariate EWMA Charts:
https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc343.htm

"""

from __future__ import division

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class _ControlChartDataHolder(object):
    def __init__(self, y, phase=1, index=None, upper=None, lower=None):
        self.nobs, self.k_endog = y.shape
        self.index = index if index is not None else np.arange(y.shape[0])
        self.phase = phase
        self.upper = upper
        self.lower = lower
        self.out_control_idx = None


class ControlChart(object):
    """
    Base class for plotting control charts.
    """
    def __init__(self,
                 historical,
                 future=None,
                 alpha=0.06,
                 crit=None,
                 distr='t'):
        """
        Creates a control chart with historical (phase I) data.
        :param historical: A numpy array with shape (number of observations,
        number of variables)
        :param future: An optional numpy array of future (phase II) data
        :param alpha:
        :param crit:
        :param distr:
        """
        self.endog = historical

        self.historical_info = _ControlChartDataHolder(y=self.endog, phase=1)
        self.future_data = future

        if self.future_data is not None:
            new_idx_start = self.historical_info.index[-1] + 1
            new_idx = np.arange(new_idx_start,
                                new_idx_start + self.future_data.shape[0])
            self.future_info = _ControlChartDataHolder(y=self.future_data,
                                                       phase=2,
                                                       index=new_idx)

        self.alpha = alpha
        self.crit = crit if crit is not None \
            else self._critical_values_from_distr(self.historical_info,
                                                  distr,
                                                  alpha)
        self.center = self.endog.mean(0)
        self.std = self.endog.std(0, ddof=1)

        # Tries to calculate all statistics & control limits if possible
        self._update_statistics()

    def _update_statistics(self):
        (self.statistic, self.stat_new) = self.generate_chart_statistic()
        self.determine_control_limits(phase=1)
        idx_out, out = self.find_out_control(self.historical_info,
                                             self.statistic,
                                             phase=1)
        self.historical_info.out_control_idx = (idx_out, out)
        if self.future_data is not None:
            self.determine_control_limits(phase=2)
            idx_out, out = self.find_out_control(self.future_info,
                                                 self.stat_new,
                                                 phase=2)
            self.future_info.out_control_idx = (idx_out, out)

    def generate_chart_statistic(self):
        """
        Calculates the statistic that this control chart is based off of.
        This could be a dynamic value that changes with new observations or it
        may remain static.
        :return: This function should return a tuple of (phase I statistic,
        phase II statistic)
        """
        raise NotImplementedError('Needs to have a calculate statistic method')

    def determine_control_limits(self, phase):
        raise NotImplementedError(
            'Needs upper and lower control limit definition')

    def _validate_data(self):
        if self.historical_info.nobs == 0:
            raise ValueError('Needs more than one data point in phase I data')
        if self.future_data is not None:
            if self.future_info.nobs == 0:
                raise ValueError(
                    'Needs more than one data point in phase II data')
            if self.future_info.k_endog != self.historical_info.k_endog:
                raise ValueError(
                    'The phase II data must contain the same number of ' +
                    'variables as the phase I data.')

    def _validate_phase(self, phase):
        """
        Verifies that we can actually plot the requested control chart.
        Raises a :exc:`ValueError` if this is not possible.
        """
        if phase != 1 and self.future_data is None:
            raise ValueError(
                'Cannot plot a phase II chart when there is no future data.')

    def _critical_values_from_distr(self, info, distr, alpha):
        if distr == 'F':
            return stats.f.isf(alpha / 2, 1, info.nobs - 1)
        elif distr == 't':
            return stats.t.isf(alpha / 2, info.nobs - 1)
        elif distr == 'chi2':
            return stats.chi2.isf(alpha / 2, 1)
        elif distr == 'n':
            return stats.norm.isf(alpha / 2)
        else:
            raise ValueError('Not a supported distribution')

    def _info_from_phase(self, phase):
        """Returns the appropriate statistic values, indices, and control
        limits depending on the phase of the control chart.
        :return A tuple of (ControlChartDataHolder, statistic ndarray)"""
        if phase == 1:
            return self.historical_info, self.statistic
        elif phase == 2:
            return self.future_info, self.stat_new
        else:
            concat_stat = np.concatenate((self.statistic, self.stat_new),
                                         axis=0)
            concat_info = _ControlChartDataHolder(y=concat_stat, phase=phase)
        info = self.historical_info if phase == 1 else self.future_info
        stat = self.statistic if phase == 1 else self.stat_new
        return info, stat

    def plot_control_limits(self, phase, ax):
        info, statistic = self._info_from_phase(phase)
        if np.isscalar(info.upper):
            ax.fill_between(info.index,
                            info.upper,
                            info.lower,
                            color='k',
                            alpha=.2)
            ax.hlines(info.upper,
                      info.index[0],
                      info.index[-1],
                      colors='k',
                      linestyle='dashed')
            ax.hlines(info.lower,
                      info.index[0],
                      info.index[-1],
                      colors='k',
                      linestyle='dashed')
            ax.hlines(self.center,
                      info.index[0],
                      info.index[-1],
                      lw=2,
                      colors='k')

    def plot_phase(self, phase, ax):
        """
        Plots the control chart for the specified phase on the axes.
        :param phase - either 0 (both), 1, or 2
        :param ax - the axes to plot on
        :return the axes
        """
        # Gets the statistic, index, and control limits depending on the phase
        info, statistic = self._info_from_phase(phase)

        # Plot the statistic and control limits
        ax.plot(info.index, statistic, 'o-')
        self.plot_control_limits(phase=phase, ax=ax)

        # Highlight the out-of-control points
        idx_out, out = info.out_control_idx
        ax.plot(idx_out, statistic[out], 'o', color='red')
        return ax

    def update(self, future=None):
        if future is not None:
            if self.future_data is not None:
                self.future_data, self.future_info.index = \
                    self.concat_data_and_index(self.future_data,
                                               future,
                                               self.future_info.index)
            else:
                self.future_data = future
                new_idx = np.arange(self.historical_info.index[-1] + 1,
                                    self.historical_info.index[-1] + 1 +
                                    self.future_data.shape[0])
                self.future_info = _ControlChartDataHolder(y=self.future_data,
                                                           phase=2,
                                                           index=new_idx)
        self._update_statistics()

    def plot_chart(self, future=None, phase=1, ax=None, update=True):
        """
        Plots the control chart.

        :param future: Optional new data that we want to include for phase II.
        If phase II data already exists, the new data will be concatenated to
        that and statistics will be recalculated.

        :param phase: The phase of the control chart to plot (either 1, 2, or
        0 for both phases)

        :param ax: The axes to plot the chart on

        :param update: boolean Whether or not to update the future data

        :return chart axes
        """
        if update:
            self.update(future)

        self._validate_phase(phase)

        # Create new plot if none exists
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if phase == 1 or phase == 2:
            self.plot_phase(phase, ax=ax)
        else:
            self.plot_phase(phase=1, ax=ax)
            self.plot_phase(phase=2, ax=ax)

        xmin = self.historical_info.index.min() \
            if phase != 2 else self.future_info.index.min()
        xmax = self.future_info.index.max() \
            if phase != 1 else self.historical_info.index.max()
        ax.set_xlim(xmin, xmax)
        ax.set_title('Mean Chart vectorized')
        return ax

    def find_out_control(self, info, statistic, phase):
        """
        Finds the out-of-control points depending on the phase.
        :param info: `ControlChartDataHolder`
        :return: A tuple of the out-of-control indexes and all indexes
        """
        out = (statistic > info.upper) | (statistic < info.lower)
        if statistic.ndim > 1:
            if phase == 2:
                idx_out = np.repeat(
                    np.arange(self.statistic.shape[0],
                              self.statistic.shape[0] +
                              statistic.shape[0])[:, None],
                    statistic.shape[1], axis=1)[out]
            else:
                idx_out = np.repeat(
                    np.arange(statistic.shape[0])[:, None],
                    statistic.shape[1], axis=1)[out]
        else:
            idx_out = info.index[out]
        return idx_out, out

    def concat_data_and_index(self, y_old, y_new, old_idx):
        """
        returns: y, full index
        """
        y = np.concatenate((y_old, y_new), axis=0)
        n1 = y_new.shape[0]
        idx_new = np.arange(old_idx[-1] + 1, old_idx[-1] + 1 + n1)
        idx = np.concatenate((old_idx, idx_new))
        return y, idx


class ControlChartMean(ControlChart):

    def __init__(self, historical,
                 future=None,
                 alpha=0.05,
                 crit=None,
                 distr='t'):
        super(ControlChartMean, self).__init__(historical=historical,
                                               future=future,
                                               alpha=alpha,
                                               crit=crit,
                                               distr=distr)

    def generate_chart_statistic(self):
        return self.endog, self.future_data

    def determine_control_limits(self, phase):
        self.diff = self.crit * self.std
        self.historical_info.upper = self.center + self.diff
        self.historical_info.lower = self.center - self.diff
        if phase != 1:
            self.future_info.upper = self.center + self.diff
            self.future_info.lower = self.center - self.diff

    def plot_control_limits(self, phase, ax, cmap=None):
        info, y = self._info_from_phase(phase)
        n = info.upper.shape[0]
        for i in range(n):
            ax.hlines(info.upper[i],
                      info.index[0],
                      info.index[-1],
                      colors=cmap[i] if cmap else 'k', linestyle='dashed')
            ax.hlines(info.lower[i],
                      info.index[0],
                      info.index[-1],
                      colors=cmap[i] if cmap else 'k', linestyle='dashed')
            ax.fill_between(info.index,
                            info.upper[i],
                            info.lower[i],
                            color=cmap[i] if cmap else 'k', alpha=.2)

    def plot_phase(self, phase, ax):
        """
        Plots the control chart for the specified phase on the axes.
        :param phase - either 0 (both), 1, or 2
        :param ax - the axes to plot on
        :return the axes
        """
        # Gets the statistic, index, and control limits depending on the phase
        info, statistic = self._info_from_phase(phase)

        # Plot the statistic and control limits
        sbcmap = ['#006BA4', '#FF800E', '#ABABAB',
                  '#595959', '#5F9ED1', '#C85200',
                  '#898989', '#A2C8EC', '#FFBC79',
                  '#CFCFCF']
        for i in range(info.k_endog):
            ax.plot(info.index, statistic[:, i], 'o-', color=sbcmap[i])

        self.plot_control_limits(phase=phase, ax=ax, cmap=sbcmap)

        # Highlight the out-of-control points
        idx_out, out = info.out_control_idx
        ax.plot(idx_out, statistic[out], 'o', color='red')
        return ax

    def plot_chart(self, future=None, phase=1, ax=None, update=True):
        ax = super(ControlChartMean, self).plot_chart(future=future,
                                                      phase=phase,
                                                      ax=ax,
                                                      update=update)
        phase_name = "Phase %s" % phase \
            if phase == 1 or phase == 2 \
            else "Phases I & II"
        ax.set_title('Univariate Control Chart (Mean) %s' % phase_name)
        return ax


class MultiVariateControlChart(ControlChart):
    """
    This generates a multivariate control chart based on the T2
    Hotelling statistic.

    References:
    https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/hotell.htm
    """

    def __init__(self,
                 historical,
                 future=None,
                 alpha=0.05,
                 crit=None,
                 distr='F'):
        self.inverse_covariance = None
        self.sample_size = 1
        super(MultiVariateControlChart, self).__init__(
            historical=historical,
            future=future,
            alpha=alpha,
            crit=crit,
            distr=distr)

    def determine_control_limits(self, phase):
        info, _ = self._info_from_phase(phase)
        df1 = info.k_endog / 2
        df2 = (info.nobs - info.k_endog - 1) / 2
        if phase == 1:
            info.upper = ((info.nobs - 1) ** 2) / info.nobs * \
                         stats.beta.isf(self.alpha, df1, df2)
        else:
            info.upper = (info.k_endog * (info.nobs + 1) * (info.nobs - 1) /
                          (info.nobs * (info.nobs - info.k_endog))) * \
                         stats.f.isf(self.alpha,
                                     info.k_endog,
                                     info.nobs - info.k_endog)
        info.lower = 0
        return info.upper, info.lower

    def generate_chart_statistic(self):
        statistic, inv_cov = self.stat_func(self.endog,
                                            self.endog - self.center)
        if self.future_data is not None:
            stat_new, _ = self.stat_func(self.future_data,
                                         self.future_data - self.center,
                                         inv_cov)
        else:
            stat_new = None
        return statistic, stat_new

    def stat_func(self, values, diff, inv_cov=None):
        if inv_cov is None:
            covariance = np.cov(values, rowvar=False, ddof=1)
            inv_cov = np.linalg.inv(covariance)
        stat = self.sample_size * (diff.dot(inv_cov) * diff).sum(1)
        return stat, inv_cov

    def plot_chart(self, future=None, phase=1, ax=None, update=True):
        ax = super(MultiVariateControlChart, self).plot_chart(
            future=future,
            phase=phase,
            ax=ax,
            update=update)
        phase_name = "Phase %s" % phase \
            if phase == 1 or phase == 2 \
            else "Phases I & II"
        ax.set_title('Multivariate Control Chart (T2): %s' % phase_name)
        return ax


class EWMAMultivariateControlChart(MultiVariateControlChart):
    """
    Multivariate EWMA: combines T2 and EWMA
    """

    def __init__(self,
                 historical,
                 weight,
                 future=None,
                 alpha=0.05,
                 crit=None,
                 distr='F'):
        self.weight = weight
        super(EWMAMultivariateControlChart, self).__init__(
            historical=historical,
            future=future,
            alpha=alpha,
            crit=crit,
            distr=distr)

    def generate_chart_statistic(self):
        t2_historical = self._generate_t2_ewma(self.endog)
        t2_future = self._generate_t2_ewma(self.future_data) \
            if self.future_data is not None else None
        return t2_historical, t2_future

    def _generate_t2_ewma(self, y):
        """
        Reference: https://mpra.ub.uni-muenchen.de/6399/1/MPRA
        :param y:
        :return:
        """
        ewma_0 = y.mean(0)
        ewma = [ewma_0]
        t2s = []
        covariance = np.cov(y, rowvar=False, ddof=1)
        cons = (self.weight / (2. - self.weight))
        for i in range(1, y.shape[0] + 1):
            exp_wt = cons * (1 - (1 - self.weight) ** (2 * i))
            cov_ith_ewma = np.linalg.inv(exp_wt * covariance)
            ewma_i = self.weight * y[i - 1] + (1 - self.weight) * ewma[-1]
            ewma.append(ewma_i)
            t2 = np.matmul(np.matmul(ewma_i - ewma_0, cov_ith_ewma),
                           (ewma_i - ewma_0).transpose())
            t2s.append(t2)
        return np.array(t2s)

    def plot_chart(self, future=None, phase=1, ax=None, update=True):
        ax = super(EWMAMultivariateControlChart, self).plot_chart(
            future=future,
            phase=phase,
            ax=ax,
            update=update)
        phase_name = "Phase %s" % phase \
            if phase == 1 or phase == 2 else "Phases I & II"
        ax.set_title('MEMWA Chart: %s' % phase_name)
        return ax
