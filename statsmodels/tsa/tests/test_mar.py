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


class CheckHamilton1989(object):
    """
    Hamilton's (1989) Markov Switching Model of GNP (as presented in Kim and
    Nelson (1999))

    Test data produced using GAUSS code described in Kim and Nelson (1999) and
    found at http://econ.korea.ac.kr/~cjkim/SSMARKOV.htm

    See `statsmodels.tsa.tests.results.results_mar` for more details.
    """

    filter_method = None

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
            0.79619,
            -0.21320, 1.12828
        ])

        # Log Likelihood
        self.loglike = mod.loglike(params)

        # Filtered probabilities
        (
            marginal_densities, filtered_joint_probabilities,
            filtered_joint_probabilities_t1
        ) = mod.filter(params, self.filter_method)
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


class TestHamilton1989C(CheckHamilton1989):
    """
    Tests Hamilton's (1989) Markov Switching Model of GNP (as presented in Kim
    and Nelson (1999)) using the filter written in Cython.
    """
    filter_method = 'c'


class TestHamilton1989Python(CheckHamilton1989):
    """
    Tests Hamilton's (1989) Markov Switching Model of GNP (as presented in Kim
    and Nelson (1999)) using the filter written in pure Python.
    """
    filter_method = 'python'
