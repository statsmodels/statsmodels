"""
Tests for Markov Autoregression

References
----------

Kim, Chang-Jin, and Charles R. Nelson. 1999.
"State-Space Models with Regime Switching:
Classical and Gibbs-Sampling Approaches with Applications".
MIT Press Books. The MIT Press.
"""

import os
import numpy as np
import pandas as pd
from results import results_mar
from statsmodels.tsa.base.datetools import dates_from_range
from statsmodels.tsa.mar_model import MAR
from numpy.testing import assert_almost_equal, assert_equal, assert_raises

DECIMAL_8 = 8
DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1


class TestHamilton1989(object):
    """
    Hamilton's (1989) Markov Switching Model of GNP (as presented in Kim and
    Nelson (1999))

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `statsmodels.tsa.tests.results.results_mar` for more details.
    """

    def __init__(self):
        self.true = results_mar.htm4_kim

        # Hamilton's 1989 GNP dataset: Quarterly, 1947.1 - 1986.4
        data = pd.DataFrame(
            self.true['data'],
            index=pd.date_range('1947-01-01', '1995-07-01', freq='QS'),
            columns=['gnp']
        )
        data['dlgnp'] = np.log(data['gnp']).diff()*100
        data = data['1952-01-01':'1984-10-01']

        # Two-state Markov-switching process, where GNP is an AR(4)
        mod = MAR(data.dlgnp, order=4, nstates=2)

        # Parameters from Table 4.1, Kim and Nelson (1999)
        params = np.array([
            1.15590, -2.20657,
            0.08983, -0.01861, -0.17434, -0.08392,
            -np.log(0.79619), # necessary due to transformation
            -0.21320, 1.12828
        ])

        # Log Likelihood
        self.loglike = mod.loglike(params)

        # Filtered probabilities
        (
            marginal_densities, filtered_joint_probabilities,
            filtered_joint_probabilities_t1
        ) = mod.filter(params)
        filtered_marginal_probabilities = mod.marginalize_probabilities(
            filtered_joint_probabilities[1:]
        )
        self.filtered = filtered_marginal_probabilities

        # Smoothed probabilities
        transitions = mod.separate_params(params)[0]
        smoothed_marginal_probabilities = mod.smooth(
            filtered_joint_probabilities, filtered_joint_probabilities_t1,
            transitions
        )
        self.smoothed = smoothed_marginal_probabilities

    def test_loglike(self):
        assert_almost_equal(
            self.loglike, self.true['-1*fout'], DECIMAL_5
        )

    def test_filtered_recession_probabilities(self):
        assert_almost_equal(
            self.filtered[:, 0], self.true['pr_tt0'], DECIMAL_5
        )

    def test_smoothed_recession_probabilities(self):
        assert_almost_equal(
            self.smoothed[:, 0], self.true['smooth0'], DECIMAL_5
        )


class TestKimNelsonStartz1998(object):
    """
    Kim, Nelson, and Startz's (1998) "application of a three-state
    Markov-switching variance model to monthly stock returns for the period
    1926:1 - 1986:12"

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `statsmodels.tsa.tests.results.results_mar` for more details.
    """

    def __init__(self):
        self.true = results_mar.stck_v3

        # Equal-Weighted Excess Returns
        data = pd.DataFrame(
            self.true['data'],
            # Note: it's not clear that these are the correct dates, but it
            # doesn't matter for the test.
            index=pd.date_range('1926-01-01', '1995-12-01', freq='MS'),
            columns=['ewer']
        )
        data = data[0:732]
        data['dmewer'] = data['ewer'] - data['ewer'].mean()

        # Two-state Markov-switching process, where GNP is an AR(4)
        mod = MAR(data.dmewer, order=0, nstates=3,
                  switch_ar=False, switch_var=True, switch_mean=[0,0,0])

        # Parameters from stck_v3.opt
        # Also correspond to Kim and Nelson (1999) Table 4.3, after
        # transformations.
        params = np.array([
            16.399767, 12.791361, 0.522758, 4.417225, -5.845336, -3.028234,
            # Division by 2 because in stck_v3.opt the parameters are
            # variances, and here they are standard deviations
            6.704260/2,  5.520378/2,  3.473059/2
        ])

        # Log Likelihood
        self.loglike = mod.loglike(params)

        # Filtered probabilities
        (
            marginal_densities, filtered_joint_probabilities,
            filtered_joint_probabilities_t1
        ) = mod.filter(params)
        filtered_marginal_probabilities = mod.marginalize_probabilities(
            filtered_joint_probabilities[1:]
        )
        self.filtered = filtered_marginal_probabilities

        # Smoothed probabilities
        transitions = mod.separate_params(params)[0]
        smoothed_marginal_probabilities = mod.smooth(
            filtered_joint_probabilities, filtered_joint_probabilities_t1,
            transitions
        )
        self.smoothed = smoothed_marginal_probabilities

    def test_loglike(self):
        assert_almost_equal(
            self.loglike, self.true['fout'], DECIMAL_5
        )

    def test_filtered_recession_probabilities(self):
        assert_almost_equal(
            self.filtered, self.true['prtt'], DECIMAL_5
        )

    def test_smoothed_recession_probabilities(self):
        assert_almost_equal(
            self.smoothed, self.true['sm0'], DECIMAL_5
        )