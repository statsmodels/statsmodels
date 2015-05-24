import sys
import nose.tools as ntools

import numpy as np
import numpy.testing as npt
from numpy.testing import dec

import pandas as pd

from statsmodels.stats.robustros import RobustROSEstimator
from statsmodels.compat.python import StringIO

try:
    import matplotlib.pyplot as plt
    have_matplotlib = True
except:
    have_matplotlib = False


class CheckROSMixin(object):
    def setup(self):
        self.main_setup()
        self.data = pd.DataFrame({'res': self.obs, 'cen': self.cen})
        self.ros = RobustROSEstimator(self.data, result='res',
                                      censorship='cen')
        self.ros.estimate()

    def main_setup(self):
        self.setup_knowns()
        self.known_debug_cols = [
            'Zprelim', 'cen', 'det_limit_index', 'modeled',
            'modeled_data', 'plot_pos', 'rank', 'res'
        ]

    def test_zero_data(self):
        data = self.data.copy()
        data.loc[0, 'res'] = 0.0
        npt.assert_raises(ValueError, RobustROSEstimator, data)

    def test_negative_data(self):
        data = self.data.copy()
        data.loc[0, 'res'] = -1.0
        npt.assert_raises(ValueError, RobustROSEstimator, data)

    def test_data(self):
        ntools.assert_true(hasattr(self.ros, 'data'))
        ntools.assert_true(isinstance(self.ros.data, pd.DataFrame))

    def test_data_cols(self):
        known_cols = ['modeled', 'res', 'cen']
        npt.assert_array_equal(self.ros.data.columns.tolist(), known_cols)

    def test_debug_attr(self):
        ntools.assert_true(hasattr(self.ros, 'debug'))
        ntools.assert_true(isinstance(self.ros.data, pd.DataFrame))

    def test_debug_cols(self):

        npt.assert_array_equal(
            sorted(self.ros.debug.columns.tolist()),
            sorted(self.known_debug_cols)
        )

    def test_plotting_positions(self):
        pp = np.round(np.array(self.ros.debug.plot_pos), 3)
        npt.assert_array_almost_equal(
            sorted(pp),
            sorted(self.known_plot_pos),
            decimal=3
        )

    def test_nobs(self):
        ntools.assert_true(hasattr(self.ros, 'nobs'))
        npt.assert_equal(self.ros.nobs, self.data.shape[0])

    def test_ncen(self):
        ntools.assert_true(hasattr(self.ros, 'ncen'))
        npt.assert_equal(self.ros.ncen, self.data[self.data.cen].shape[0])

    def test_cohn_attr(self):
        ntools.assert_true(hasattr(self.ros, 'cohn'))
        ntools.assert_true(isinstance(self.ros.cohn, pd.DataFrame))

    def test_cohn_nuncen_above(self):
        npt.assert_array_equal(self.known_cohn_nuncen_above,
                               self.ros.cohn['nuncen_above'].values)

    def test_cohn_nobs_below(self):
        npt.assert_array_equal(self.known_cohn_nobs_below,
                               self.ros.cohn['nobs_below'].values)

    def test_cohnncen_equal(self):
        npt.assert_array_equal(self.known_cohnncen_equal,
                                  self.ros.cohn['ncen_equal'].values)

    def test_cohn_prob_exceedance(self):
        npt.assert_array_almost_equal(
            self.known_cohn_prob_exceedance, self.ros.cohn['prob_exceedance'], decimal=4
        )

    def test_MR_rosEstimator(self):
        modeled = np.array(self.ros.data.modeled)
        modeled.sort()

        npt.assert_array_almost_equal(self.known_modeled, modeled, decimal=2)

    def test_dup_index_error(self):
        data = self.data.append(self.data)
        npt.assert_raises(ValueError, RobustROSEstimator, data)

    def test_non_dataframe_error(self):
        data = self.data.values
        npt.assert_raises(ValueError, RobustROSEstimator, data)

    @dec.skipif(not have_matplotlib)
    def test_plot_default(self):
        ax = self.ros.plot()
        ntools.assert_true(isinstance(ax, plt.Axes))

    @dec.skipif(not have_matplotlib)
    def test_plot_ylogFalse_withAx(self):
        fig, ax = plt.subplots()
        ax = self.ros.plot(ylog=False, ax=ax)

    @ntools.raises(ValueError)
    def test_nonnumeric(self):
        N = 20
        data = pd.DataFrame({
            'res': ['A'] * N,
            'cen': [False] * N
        })
        ros = RobustROSEstimator(data, result='res', censorship='cen')

    @ntools.raises(ValueError)
    def test_scalars(self):
        ros = RobustROSEstimator(data=None, result=1, censorship='A')

    def test_transform_in(self):
        ntools.assert_equal(self.ros.transform_in, np.log)

    def test_transform_out(self):
        ntools.assert_equal(self.ros.transform_out, np.exp)


class testROSHelselArsenic(CheckROSMixin):
    '''
    Oahu arsenic data from Nondetects and Data Analysis by
    Dennis R. Helsel (John Wiley, 2005)

    Plotting positions are fudged since relative to source data since
    modeled data is what matters and (source data plot positions are
    not uniformly spaced, which seems weird)
    '''
    def setup_knowns(self):

        self.obs = np.array([
            3.2, 2.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 1.7, 1.5, 1.0, 1.0, 1.0, 1.0,
            0.9, 0.9, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5
        ])

        self.cen = np.array([
            False, False, True, True, True, True, True,
            True, True, True, False, False, True, True,
            True, True, False, True, False, False, False,
            False, False, False
        ])

        self.known_cohn_nuncen_above = np.array([6.0, 1.0, 2.0, 2.0, np.nan])
        self.known_cohn_nobs_below = np.array([0.0, 7.0, 12.0, 22.0, np.nan])
        self.known_cohnncen_equal = np.array([0.0, 1.0, 4.0, 8.0, np.nan])
        self.known_cohn_prob_exceedance = np.array([1.0, 0.3125, 0.2143, 0.0833, 0.0])
        self.known_plot_pos = np.array([
            0.102,  0.157,  0.204,  0.306,  0.314,  0.344,  0.407,  0.471,
            0.509,  0.611,  0.629,  0.713,  0.815,  0.098,  0.196,  0.295,
            0.393,  0.491,  0.589,  0.737,  0.829,  0.873,  0.944,  0.972
        ])

        self.known_modeled = np.array([
            3.20, 2.80, 1.42, 1.14, 0.95, 0.81, 0.68, 0.57,
            0.46, 0.35, 1.70, 1.50, 0.98, 0.76, 0.58, 0.41,
            0.90, 0.61, 0.70, 0.70, 0.60, 0.50, 0.50, 0.50
        ])

        self.known_modeled.sort()


class testROSHelselAppendixB(CheckROSMixin):
    '''
    Appendix B dataset from "Estimation of Descriptive Statists for Multiply
    Censored Water Quality Data", Water Resources Research, Vol 24,
    No 12, pp 1997 - 2004. December 1988.
    '''
    def setup_knowns(self):

        self.obs = np.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10., 10., 10.,
            3.0, 7.0, 9.0, 12., 15., 20., 27., 33., 50.
        ])
        self.cen = np.array([
            True, True, True, True, True, True, True, True, True,
            False, False, False, False, False, False, False,
            False, False
        ])
        self.data = pd.DataFrame({'res': self.obs, 'cen': self.cen})

        self.known_cohn_nuncen_above = np.array([3.0, 6.0, np.nan])
        self.known_cohn_nobs_below = np.array([6.0, 12.0, np.nan])
        self.known_cohnncen_equal = np.array([6.0, 3.0, np.nan])
        self.known_cohn_prob_exceedance = np.array([0.5555, 0.3333, 0.0])
        self.known_plot_pos = np.array([
            0.063, 0.127, 0.167, 0.190, 0.254, 0.317, 0.333, 0.381, 0.500,
            0.500, 0.556, 0.611, 0.714, 0.762, 0.810, 0.857, 0.905, 0.952
        ])
        self.known_modeled = np.array([
            0.47,  0.85, 1.11, 1.27, 1.76, 2.34, 2.50, 3.00, 3.03,
            4.80, 7.00, 9.00, 12.0, 15.0, 20.0, 27.0, 33.0, 50.0
        ])
        self.known_modeled.sort()


class testROSHelselAppendixB_withArrays(testROSHelselAppendixB):
    def setup(self):
        self.main_setup()
        self.ros = RobustROSEstimator(data=None, result=self.obs,
                                      censorship=self.cen)
        self.ros.estimate()


class testRNADAdata(CheckROSMixin):
    '''
    Arsenic Dataset from the R-Ndata Package.

    Plotting positions are fudged since relative to source data since
    modeled data is what matters and (source data plot positions are
    not uniformly spaced, which seems weird)
    '''
    def setup_knowns(self):

        datastring = StringIO("""res cen
            0.090  True
            0.090  True
            0.090  True
            0.101 False
            0.136 False
            0.340 False
            0.457 False
            0.514 False
            0.629 False
            0.638 False
            0.774 False
            0.788 False
            0.900  True
            0.900  True
            0.900  True
            1.000  True
            1.000  True
            1.000  True
            1.000  True
            1.000  True
            1.000 False
            1.000  True
            1.000  True
            1.000  True
            1.000  True
            1.000  True
            1.000  True
            1.000  True
            1.000  True
            1.000  True
            1.000  True
            1.000  True
            1.000  True
            1.100 False
            2.000 False
            2.000 False
            2.404 False
            2.860 False
            3.000 False
            3.000 False
            3.705 False
            4.000 False
            5.000 False
            5.960 False
            6.000 False
            7.214 False
           16.000 False
           17.716 False
           25.000 False
           51.000 False""")
        data = pd.read_csv(datastring, sep='\s+')
        self.obs = data.res
        self.cen = data.cen

        self.known_cohn_nuncen_above = np.array([9., 0.0, 18., np.nan])
        self.known_cohn_nobs_below = np.array([3., 15., 32., np.nan])
        self.known_cohnncen_equal = np.array([3., 3., 17., np.nan])
        self.known_cohn_prob_exceedance = np.array([0.84, 0.36, 0.36, 0])
        self.known_plot_pos = np.array([
            0.036,  0.040,  0.071,  0.080,  0.107,  0.120,  0.142,  0.160,
            0.178,  0.213,  0.249,  0.284,  0.320,  0.320,  0.356,  0.391,
            0.427,  0.462,  0.480,  0.498,  0.533,  0.569,  0.604,  0.208,
            0.256,  0.304,  0.352,  0.400,  0.448,  0.496,  0.544,  0.592,
            0.659,  0.678,  0.697,  0.716,  0.735,  0.754,  0.773,  0.792,
            0.811,  0.829,  0.848,  0.867,  0.886,  0.905,  0.924,  0.943,
            0.962,  0.981
        ])
        self.known_modeled = np.array([
            0.01907990,  0.03826254,  0.06080717,  0.10100000,  0.13600000,
            0.34000000,  0.45700000,  0.51400000,  0.62900000,  0.63800000,
            0.77400000,  0.78800000,  0.08745914,  0.25257575,  0.58544205,
            0.01711153,  0.03373885,  0.05287083,  0.07506079,  0.10081573,
            1.00000000,  0.13070334,  0.16539309,  0.20569039,  0.25257575,
            0.30725491,  0.37122555,  0.44636843,  0.53507405,  0.64042242,
            0.76644378,  0.91850581,  1.10390531,  1.10000000,  2.00000000,
            2.00000000,  2.40400000,  2.86000000,  3.00000000,  3.00000000,
            3.70500000,  4.00000000,  5.00000000,  5.96000000,  6.00000000,
            7.21400000, 16.00000000, 17.71600000, 25.00000000, 51.00000000
        ])
        self.known_plot_pos.sort()
        self.known_modeled.sort()


class testROSNoNDs(CheckROSMixin):
    def setup_knowns(self):

        np.random.seed(0)
        N = 20
        self.obs = np.random.lognormal(size=N)
        self.cen = [False] * N

        self.known_cohn_nuncen_above = np.array([])
        self.known_cohn_nobs_below = np.array([])
        self.known_cohnncen_equal = np.array([])
        self.known_cohn_prob_exceedance = np.array([])
        self.known_modeled = np.array([
            0.38, 0.43, 0.81, 0.86, 0.90, 1.13, 1.15, 1.37, 1.40,
            1.49, 1.51, 1.56, 2.14, 2.59, 2.66, 4.28, 4.46, 5.84,
            6.47, 9.4 ])
        self.known_plot_pos = np.array([
            0.034,  0.083,  0.132,  0.181,  0.230,  0.279,  0.328,
            0.377,  0.426,  0.475,  0.525,  0.574,  0.623,  0.672,
            0.721,  0.770,  0.819,  0.868,  0.917,  0.966
            ])
        self.known_modeled.sort()

    def main_setup(self):
        self.setup_knowns()
        self.known_debug_cols = [
            'rank', 'cen', 'det_limit_index', 'modeled', 'plot_pos', 'res'
        ]


class testROSoneND(CheckROSMixin):
    def setup_knowns(self):

        self.obs = np.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10., 10., 10.,
            3.0, 7.0, 9.0, 12., 15., 20., 27., 33., 50.
        ])
        self.cen = np.array([
            True, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False,
            False, False
        ])

        self.known_cohn_nuncen_above = np.array([17.0, np.nan])
        self.known_cohn_nobs_below = np.array([1.0, np.nan])
        self.known_cohnncen_equal = np.array([1.0, np.nan])
        self.known_cohn_prob_exceedance = np.array([0.9444, 0.0])
        self.known_plot_pos = np.array([
            0.028,  0.108,  0.16 ,  0.213,  0.265,  0.318,  0.37 ,  0.423,
            0.475,  0.528,  0.58 ,  0.633,  0.685,  0.738,  0.79 ,  0.843,
            0.895,  0.948])
        self.known_modeled = np.array([
            0.24, 1.0, 1.0, 1.0, 1.0, 1.0, 10., 10., 10.,
            3.0 , 7.0, 9.0, 12., 15., 20., 27., 33., 50.
        ])
        self.known_modeled.sort()

    def main_setup(self):
        self.setup_knowns()
        self.known_debug_cols = [
            'Zprelim', 'cen', 'det_limit_index', 'modeled',
            'modeled_data', 'plot_pos', 'rank', 'res'
        ]


class testROShalfDLs80pctNDs(CheckROSMixin):
    def setup_knowns(self):

        self.obs = np.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10., 10., 10.,
            3.0, 7.0, 9.0, 12., 15., 20., 27., 33., 50.
        ])
        self.cen = np.array([
            True, True, True, True, True, True, True, True,
            True, True, True, True, True, True, True, False,
            False, False
        ])

        self.known_cohn_nuncen_above = np.array([0., 0., 0., 0., 0., 0., 0., 3., np.nan])
        self.known_cohn_nobs_below = np.array([6., 7., 8., 9., 12., 13., 14., 15., np.nan])
        self.known_cohnncen_equal = np.array([6., 1., 1., 1., 3., 1., 1., 1., np.nan])
        self.known_cohn_prob_exceedance = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.known_plot_pos = np.array([
            0.038,  0.092,  0.146,  0.201,  0.255,  0.309,  0.364,  0.418,
            0.473,  0.527,  0.582,  0.636,  0.691,  0.745,  0.799,  0.854,
            0.908,  0.962
        ])

        self.known_modeled = np.array([
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5.0, 5.0, 5.0,
            1.5, 3.5, 4.5, 6.0, 7.5, 10., 27., 33., 50.
        ])
        self.known_modeled.sort()

    def main_setup(self):
        self.setup_knowns()
        self.known_debug_cols = [
            'cen', 'det_limit_index', 'modeled', 'plot_pos', 'rank', 'res'
        ]


class testROShalfDLs1noncensored(CheckROSMixin):
    def setup_knowns(self):

        self.obs = np.array([
            1.0, 1.0, 12., 15.,
        ])
        self.cen = np.array([
            True, True, True, False
        ])

        self.known_cohn_nuncen_above = np.array([0., 1., np.nan])
        self.known_cohn_nobs_below = np.array([2., 3., np.nan])
        self.known_cohnncen_equal = np.array([2., 1., np.nan])
        self.known_cohn_prob_exceedance = np.array([0., 0., 0.])
        self.known_plot_pos = np.array([ 0.159,  0.385,  0.615,  0.841])

        self.known_modeled = np.array([0.5,   0.5,   6. ,  15.])
        self.known_modeled.sort()

    def main_setup(self):
        self.setup_knowns()
        self.known_debug_cols = [
            'cen', 'det_limit_index', 'modeled', 'plot_pos', 'rank', 'res'
        ]

