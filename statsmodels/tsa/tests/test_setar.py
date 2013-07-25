"""
Tests for Self-Exciting Threshold Autoregression

References
----------

Hansen, Bruce. 1999.
"Testing for Linearity."
Journal of Economic Surveys 13 (5): 551-576.

Notes
-----

The sunspot data Hansen uses is slightly different from that found in
statsmodels. To match his dataset, four values have to be altered. See
CheckSunspots.__init__() for details.

Tests so far indicate that although the R package tsDyn can replicate some of
these results (see TestSunspotsSETAR2Search(), TestSunspotsSETAR3Search()),
in general they do not produce the same estimates, of AR parameters, delay
order, thresholds, or standard errors.
"""

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.base.datetools import dates_from_range
from statsmodels.tsa.setar_model import SETAR
from statsmodels.datasets import sunspots
from numpy.testing import assert_almost_equal, assert_equal, assert_raises

DECIMAL_8 = 8
DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1


class CheckSETAR(object):
    """
    Test SETAR

    Test SETAR estimates: parameters, standard errors, delay, and thresholds
    """
    def test_beta(self):
        assert_almost_equal(self.est_beta, self.true_beta, DECIMAL_5)

    def test_se(self):
        assert_almost_equal(self.est_se, self.true_se, DECIMAL_5)

    def test_delay(self):
        assert_equal(self.model.delay, self.true_delay)

    def test_threshold(self):
        assert_almost_equal(self.model.thresholds, self.true_thresholds,
                            DECIMAL_5)


class CheckSunspots(CheckSETAR):
    """
    Sunspots dataset, 1700 - 1988
    """

    def __init__(self):
        self.dta = sunspots.load_pandas().data
        self.dta.index = pd.Index(dates_from_range('1700', '2008'))
        self.dta = self.dta[self.dta.YEAR <= 1988]
        del self.dta["YEAR"]
        self.dta.SUNACTIVITY = 2*(np.sqrt(1 + self.dta.SUNACTIVITY) - 1)
        # modifications to match Hansen's sunspot data
        self.dta.SUNACTIVITY.iloc[[262, 280, 281, 287]] = [
            10.40967365,
            22.95596121,
            21.79075451,
            8.99090533
        ]


class TestSunspotsSETAR2(CheckSunspots):
    """
    SETAR(2) estimates, from Hansen (1999)

    Notes
    -----

    Exact true values taken from the R script sunspot.R, found at
    http://www.ssc.wisc.edu/~bhansen/progs/joes_99.html

    This test replicates the results found in Hansen (199). However, selection
    of this threshold (7.4233751915117967) relies on Hansen's threshold grid
    size of 100, and the specific way he selects the 100 points on the grid.

    In particular, if the sunspot.R script is altered to search over all
    possible distinct threshold values, then a different threshold (and even a
    different delay order) is selected. See TestSunspotsSETAR2Search().

    Note that the R package tsDyn cannot replicate this result with its setar()
    function, even if the delay and threshold are specified:

    dat <- read.table("sunspot.dat")
    dat <- (sqrt(1+dat)-1)*2
    setar(dat, m=11, nthresh=1, thDelay=1, th=7.4233751915117967)

    does not produce the same AR parameter estimates.
    """

    def __init__(self):
        super(TestSunspotsSETAR2, self).__init__()

        self.model = SETAR(self.dta, order=2, ar_order=11,
                           delay=2, thresholds=[7.4233751915117967])
        self.results = self.model.fit()

        self.true_delay = 2
        self.true_thresholds = [7.4233751915117967]

        self.est_beta = self.results.params
        self.true_beta = np.array([
            # Lower Regime
            -0.58033, 1.51810, -0.97235, 0.48658, -0.19141, -0.13742, 0.12219,
            0.12763, -0.22020, 0.46415, -0.06655, -0.07187,
            # Upper Regime
            2.31657, 0.94687, -0.02729, -0.48476, 0.32033, -0.21405, -0.03833,
            0.18431, -0.21549, 0.19081, -0.01986, 0.12535
        ])

        self.est_se = self.results.HC0_se
        self.true_se = np.array([
            # Lower Regime
            0.89518, 0.10298, 0.26196, 0.29449, 0.25556, 0.28179, 0.25957,
            0.21385, 0.23455, 0.25830, 0.20378, 0.12162,
            # Upper Regime
            0.55167, 0.07511, 0.11011, 0.09701, 0.08525, 0.08378, 0.07913,
            0.08159, 0.08961, 0.09258, 0.09305, 0.06588
        ])


class TestSunspotsSETAR2Search(CheckSunspots):
    """
    SETAR(2) estimates with threshold and delay search

    Exact true values taken from the R script sunspot.R, found at
    http://www.ssc.wisc.edu/~bhansen/progs/joes_99.html, and altered to set
    qnum <- 0 rather than qnum <- 100 to make threshold search across all
    possible distinct values.

    The estimated AR parameters can also be retrieved from the R package tsDyn,
    using the following call:

    dat <- read.table("sunspot.dat")
    dat <- (sqrt(1+dat)-1)*2
    setar(dat, m=11, nthresh=2, thDelay=2, trace=TRUE)

    where sunspot.dat is the datafile provided by Hansen.

    Note that the standard errors produced by tsDyn are not the White corrected
    standard errors, but they are not OLS standard errors either. One
    possibility is that they are estimating each regime separately, which
    renders consistent standard errors.
    (see e.g. "A Nonlinear Approach to US GNP" Potter 1995, footnote 6)

    Note that tsDyn requires specifying thDelay=2; otherwise the setar()
    function will select a delay of zero, which is meaningless from a data
    generation persepective.
    """

    def __init__(self):
        super(TestSunspotsSETAR2Search, self).__init__()

        self.model = SETAR(self.dta, order=2, ar_order=11,
                           threshold_grid_size=300)
        self.results = self.model.fit()

        self.true_delay = 3
        self.true_thresholds = [10.40967]

        self.est_beta = self.results.params
        self.true_beta = np.array([
            # Lower Regime
            0.02074, 1.39353, -0.77427, 0.21783, 0.13374, -0.08193, -0.12268,
            0.20495, -0.06742, 0.20038, -0.02421, 0.01291,
            # Upper Regime
            1.24378, 0.82276, 0.02471, -0.19059, 0.15658, -0.24567, 0.01078,
            0.17140, -0.22398, 0.23060, -0.19589, 0.21867
        ])

        self.est_se = self.results.HC0_se
        self.true_se = np.array([
            # Lower Regime
            0.77018, 0.09641, 0.18913, 0.20373, 0.19753, 0.18206, 0.19039,
            0.16763, 0.18410, 0.16300, 0.15153, 0.10121,
            # Upper Regime
            0.68325, 0.07162, 0.10783, 0.09347, 0.08021, 0.07800, 0.07903,
            0.08676, 0.08433, 0.08846, 0.09589, 0.07291
        ])


class TestSunspotsSETAR3Search(CheckSunspots):
    """
    SETAR(3) estimates, from Hansen (1999)

    Exact true betas taken from the R script sunspot3.R, found at
    http://www.ssc.wisc.edu/~bhansen/progs/joes_99.html, and altered to set
    qnum <- 0 rather than qnum <- 100 to make threshold search across all
    possible distinct values.

    The estimated AR parameters can also be retrieved from the R package tsDyn,
    using the following call:

    dat <- read.table("sunspot.dat")
    dat <- (sqrt(1+dat)-1)*2
    setar(dat, m=11, nthresh=2, thDelay=2, trace=TRUE)

    where sunspot.dat is the datafile provided by Hansen.

    Note that the standard errors produced by tsDyn are not the White corrected
    standard errors, but they are not OLS standard errors either. One
    possibility is that they are estimating each regime separately, which
    renders consistent standard errors.
    (see e.g. "A Nonlinear Approach to US GNP" Potter 1995, footnote 6)

    Note that tsDyn requires specifying thDelay=2; otherwise the setar()
    function will select a delay of zero, which is meaningless from a data
    generation persepective.
    """

    def __init__(self):
        super(TestSunspotsSETAR3Search, self).__init__()

        self.model = SETAR(self.dta, order=3, ar_order=11,
                           threshold_grid_size=300)
        self.results = self.model.fit()

        self.true_delay = 3
        self.true_thresholds = [6, 10.40967]

        self.est_beta = self.results.params
        self.true_beta = np.array([
            # Lower Regime
            0.84812, 1.35607, -0.55246, 0.04575, -0.12188, 0.37577, -0.44380,
            0.38031, 0.01222, -0.33422, 0.49685, -0.19650,
            # Middle Regime
            -5.80224, 1.61091, -1.11708, 0.82737, 0.48356, -0.49390, 0.17691,
            0.10140, -0.38067, 0.83826, -0.44566, 0.13308,
            # Upper Regime
            1.24378, 0.82276, 0.02471, -0.19059, 0.15658, -0.24567, 0.01078,
            0.17140, -0.22398, 0.23060, -0.19589, 0.21867
        ])

        self.est_se = self.results.HC0_se
        self.true_se = np.array([
            # Lower Regime
            0.75718, 0.09815, 0.20131, 0.24100, 0.18620, 0.21248, 0.20791,
            0.20938, 0.16679, 0.14551, 0.15069, 0.09097,
            # Middle Regime
            2.58669, 0.16705, 0.25911, 0.39755, 0.29883, 0.24448, 0.27383,
            0.21756, 0.25325, 0.24604, 0.20291, 0.13284,
            # Upper Regime
            0.68325, 0.07162, 0.10783, 0.09347, 0.08021, 0.07800, 0.07903,
            0.08676, 0.08433, 0.08846, 0.09589, 0.07291
        ])
