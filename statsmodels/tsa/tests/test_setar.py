"""
Tests for Self-Exciting Threshold Autoregression

References
----------

Hansen, Bruce. 1999.
"Testing for Linearity."
Journal of Economic Surveys 13 (5): 551-576.
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
        # TODO this seems to be a pretty low level of almost equal
        assert_almost_equal(self.est_beta, self.true_beta, DECIMAL_2)

    def test_se(self):
        pass

    def test_delay(self):
        pass

    def test_threshold(self):
        pass


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


class TestSunspotsSETAR2(CheckSunspots):
    """
    SETAR(2) estimates, from Hansen (1999)

    Exact true betas taken from the R script sunspot.R, found at
    http://www.ssc.wisc.edu/~bhansen/progs/joes_99.html

    Notes
    -----

    TODO for now, the delay and thresholds are set, but when their selection is
    implemented this can be changed to also test them.
    """

    def __init__(self):
        super(TestSunspotsSETAR2, self).__init__()

        model = SETAR(self.dta, order=2, delay=2, ar_order=11,
                      thresholds=(7.43,))
        results = model.fit()

        self.est_beta = results.params
        self.true_beta = np.array([
            # Lower Regime
            -0.58033, 1.51810, -0.97235, 0.48658, -0.19141, -0.13742, 0.12219,
            0.12763, -0.22020, 0.46415, -0.06655, -0.07187,
            # Upper Regime
            2.31657, 0.94687, -0.02729, -0.48476, 0.32033, -0.21405, -0.03833,
            0.18431, -0.21549, 0.19081, -0.01986, 0.12535
        ])
