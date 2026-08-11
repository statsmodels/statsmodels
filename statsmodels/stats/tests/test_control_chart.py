import os

import pytest
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_allclose,
                           assert_approx_equal)
import pandas as pd
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.stats.control_chart import ControlChartMean, \
    MultiVariateControlChart, EWMAMultivariateControlChart
from statsmodels.graphics import utils
import unittest

cur_dir = os.path.abspath(os.path.dirname(__file__))


class TestControlChart(unittest.TestCase):

    @pytest.mark.matplotlib
    def test_no_future_phase_two(self):
        """
        A control chart with no future data should not be able to plot a
        non-phase I chart
        """
        _, mpl_ax = utils.create_mpl_ax()
        div_point = TestControlChart.nobs // 2
        old_data = TestControlChart.data[:div_point]
        cc = MultiVariateControlChart(old_data, alpha=0.05, distr='F')
        with self.assertRaises(ValueError):
            cc.plot_chart(phase=2, ax=mpl_ax)
        with self.assertRaises(ValueError):
            cc.plot_chart(phase=0, ax=mpl_ax)

    @pytest.mark.matplotlib
    def test_chart_updates(self):
        """
        Verifies that future data gets appended when we perform updates.
        """
        div_point = TestControlChart.nobs // 2
        old_data = TestControlChart.data[:div_point]
        new_data = TestControlChart.data[div_point:]
        new_data_part1 = TestControlChart.data[div_point:div_point+10]
        new_data_part2 = TestControlChart.data[div_point+10:]

        _, mpl_ax = utils.create_mpl_ax()
        cc = ControlChartMean(old_data, alpha=0.05)
        cc.plot_chart(future=new_data_part1, update=True, ax=mpl_ax)
        assert_array_equal(cc.future_data, new_data_part1)

        # shouldn't update from plotting if we specify update=False
        cc.plot_chart(future=new_data_part2, update=False, ax=mpl_ax)
        assert_array_equal(cc.future_data, new_data_part1)

        # should concatenate the current future data with new data
        cc.plot_chart(future=new_data_part2, update=True, ax=mpl_ax)
        assert_array_equal(cc.future_data, new_data)

    def test_univariate_control_chart_mean(self):
        """
        Verifies that a vectorized univariate control chart has appropriate
        control limits.
        """
        div_point = TestControlChart.nobs // 2
        old_data = TestControlChart.data[:div_point]
        cc = ControlChartMean(old_data, alpha=0.05)
        assert cc.historical_info.k_endog == 2
        assert_array_almost_equal(cc.historical_info.upper,
                                  [0.048826, 0.111855])
        assert_array_almost_equal(cc.historical_info.lower,
                                  [-0.028561,  0.055768])
        assert_array_equal(cc.historical_info.index, range(div_point))
        assert_array_almost_equal(cc.center, [0.01013271, 0.08381132])

    def test_multivariate_control_chart_phase_1(self):
        """
        This plots the phase 1 multivariate control chart (based on the T2
        Hotelling statistic).
        All results were verified using the R package 'MSQC'.
        """
        div_point = TestControlChart.nobs // 2
        old_data = TestControlChart.data[:div_point]
        cc = MultiVariateControlChart(old_data, alpha=0.05, distr='F')

        assert_array_almost_equal(cc.center, [0.01013271, 0.08381132])
        assert_allclose(cc.statistic,
                        TestControlChart.results_r_df['t2'].values[:53])

        assert_approx_equal(cc.historical_info.upper, 5.761466)
        assert cc.historical_info.lower == 0

        # Verify that the out of control indices are correct
        assert_array_equal(cc.historical_info.out_control_idx[0], [])

    @pytest.mark.matplotlib
    def test_multivariate_control_chart_phase_2(self):
        """
        This plots the phase 2 multivariate T2 control chart - the control
        limits for the phase II
        chart are different from the ones in phase I.
        All results were verified using the R package 'MSQC'.
        """
        div_point = TestControlChart.nobs // 2
        old_data = TestControlChart.data[:div_point]
        new_data = TestControlChart.data[div_point:]
        _, mpl_ax = utils.create_mpl_ax()
        cc = MultiVariateControlChart(old_data, new_data,
                                      alpha=0.05, distr='F')
        cc.plot_chart(phase=2, ax=mpl_ax)
        assert_array_almost_equal(cc.center, [0.01013271, 0.08381132])
        assert_allclose(cc.stat_new,
                        TestControlChart.results_r_df['t2'].values[53:])

        # Verify the control limits
        assert_approx_equal(cc.future_info.upper, 6.604564, significant=1)
        assert cc.future_info.lower == 0

        # Verify that the out of control indices are correct
        assert_array_equal(cc.future_info.out_control_idx[0],
                           [63,  99, 103, 104, 105, 106])

    @pytest.mark.matplotlib
    def test_multivariate_ewma(self):
        t2_ewma_stat = [0.47327288, 0.01709351, 0.51630749, 0.44703781,
                        1.66208689, 2.29793468, 4.06751727, 4.94504601,
                        7.14446526, 8.26090948, 8.20000580, 2.27296929,
                        0.65763094, 0.38346061, 1.56293047, 0.54236340,
                        0.24878652, 0.07328171, 1.06818145, 1.92653491,
                        3.40862051, 5.73445960, 8.83188106, 9.49446322,
                        9.03554348, 7.64767327, 7.13471209, 4.53168090,
                        1.92444186, 1.15660379, 1.97203766, 0.37564564,
                        0.02754679, 0.02904697, 1.19336615, 3.19269055,
                        6.27812788, 9.13550567, 7.50847548, 4.59700172,
                        2.79211406, 1.03192230, 1.28174096, 0.80579079,
                        1.11534028, 0.04317039, 1.16232575, 1.30112369,
                        1.60569109, 0.54756592, 2.53078470, 2.41653500,
                        3.03452030]

        div_point = TestControlChart.nobs // 2
        old_data = TestControlChart.data[:div_point]
        new_data = TestControlChart.data[div_point:]
        _, mpl_ax = utils.create_mpl_ax()
        cc = EWMAMultivariateControlChart(old_data, future=new_data,
                                          weight=0.5, alpha=0.05, distr='F')
        cc.plot_chart(phase=2, ax=mpl_ax)
        assert_array_almost_equal(cc.statistic, t2_ewma_stat)
        assert_approx_equal(cc.historical_info.upper, 5.761466)
        assert_approx_equal(cc.future_info.upper, 6.604564, significant=1)
        assert cc.historical_info.lower == 0

    @classmethod
    def setup_class(cls):
        dataset = e6.load(as_pandas=True)
        cls.data = dataset.data[['Dp', 'R']].values
        cls.nobs = len(cls.data)

        file_path = os.path.join(cur_dir, 'results', 'results_mvcc.csv')
        cls.results_r_df = pd.read_csv(file_path, index_col=0)
