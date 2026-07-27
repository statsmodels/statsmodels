"""
Tests for Markov Autoregression models

Author: Chad Fulton
License: BSD-3
"""

from pathlib import Path
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels.tools import add_constant
from statsmodels.tsa.regime_switching import markov_autoregression

current_path = Path(__file__).resolve().parent
rgnp = [
    2.59316421,
    2.20217133,
    0.45827562,
    0.9687438,
    -0.24130757,
    0.89647478,
    2.05393219,
    1.73353648,
    0.93871289,
    -0.46477833,
    -0.80983406,
    -1.39763689,
    -0.39886093,
    1.1918416,
    1.45620048,
    2.11808228,
    1.08957863,
    1.32390273,
    0.87296367,
    -0.19773273,
    0.45420215,
    0.07221876,
    1.1030364,
    0.82097489,
    -0.05795795,
    0.58447772,
    -1.56192672,
    -2.05041027,
    0.53637183,
    2.33676839,
    2.34014559,
    1.2339263,
    1.8869648,
    -0.45920792,
    0.84940469,
    1.70139849,
    -0.28756312,
    0.09594627,
    -0.86080289,
    1.03447127,
    1.23685944,
    1.42004502,
    2.22410631,
    1.30210173,
    1.03517699,
    0.9253425,
    -0.16559951,
    1.3444382,
    1.37500131,
    1.73222184,
    0.71605635,
    2.21032143,
    0.85333031,
    1.00238776,
    0.42725441,
    2.14368343,
    1.43789184,
    1.57959926,
    2.27469826,
    1.95962656,
    0.25992399,
    1.01946914,
    0.49016398,
    0.5636338,
    0.5959546,
    1.43082857,
    0.56230122,
    1.15388393,
    1.68722844,
    0.77438205,
    -0.09647045,
    1.39600146,
    0.13646798,
    0.55223715,
    -0.39944872,
    -0.61671102,
    -0.08722561,
    1.2101835,
    -0.90729755,
    2.64916158,
    -0.0080694,
    0.51111895,
    -0.00401437,
    2.16821432,
    1.92586732,
    1.03504717,
    1.85897219,
    2.32004929,
    0.25570789,
    -0.09855274,
    0.89073682,
    -0.55896485,
    0.28350255,
    -1.31155407,
    -0.88278776,
    -1.97454941,
    1.01275265,
    1.68264723,
    1.38271284,
    1.86073637,
    0.4447377,
    0.41449001,
    0.99202275,
    1.36283576,
    1.59970522,
    1.98845816,
    -0.25684232,
    0.87786949,
    3.1095655,
    0.85324478,
    1.23337317,
    0.00314302,
    -0.09433369,
    0.89883322,
    -0.19036628,
    0.99772376,
    -2.39120054,
    0.06649673,
    1.26136017,
    1.91637838,
    -0.3348029,
    0.44207108,
    -1.40664911,
    -1.52129889,
    0.29919869,
    -0.80197448,
    0.15204792,
    0.98585027,
    2.13034606,
    1.34397924,
    1.61550522,
    2.70930099,
    1.24461412,
    0.50835466,
    0.14802167,
]
rec = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]


def test_predict():
    endog = np.ones(10)
    markov_autoregression.MarkovAutoregression(endog, k_regimes=2, order=1, trend="n")
    mod = markov_autoregression.MarkovAutoregression(
        endog, k_regimes=2, order=1, trend="n"
    )
    assert_equal(mod.nobs, 9)
    assert_equal(mod.endog, np.ones(9))
    params = np.r_[0.5, 0.5, 1.0, 0.1, 0.5]
    mod_resid = mod._resid(params)
    resids = np.zeros((2, 2, mod.nobs))
    resids[0, :, :] = np.ones(9) - 0.1 * np.ones(9)
    assert_allclose(mod_resid[0, :, :], resids[0, :, :])
    resids[1, :, :] = np.ones(9) - 0.5 * np.ones(9)
    assert_allclose(mod_resid[1, :, :], resids[1, :, :])
    endog = np.arange(10)
    mod = markov_autoregression.MarkovAutoregression(endog, k_regimes=2, order=1)
    assert_equal(mod.nobs, 9)
    assert_equal(mod.endog, np.arange(1, 10))
    params = np.r_[0.5, 0.5, 2.0, 3.0, 1.0, 0.1, 0.5]
    mod_resid = mod._resid(params)
    resids = np.zeros((2, 2, mod.nobs))
    resids[0, 0, :] = np.arange(1, 10) - 2.0 - 0.1 * (np.arange(9) - 2.0)
    assert_allclose(mod_resid[0, 0, :], resids[0, 0, :])
    resids[0, 1, :] = np.arange(1, 10) - 2.0 - 0.1 * (np.arange(9) - 3.0)
    assert_allclose(mod_resid[0, 1, :], resids[0, 1, :])
    resids[1, 0, :] = np.arange(1, 10) - 3.0 - 0.5 * (np.arange(9) - 2.0)
    assert_allclose(mod_resid[1, 0, :], resids[1, 0, :])
    resids[1, 1, :] = np.arange(1, 10) - 3.0 - 0.5 * (np.arange(9) - 3.0)
    assert_allclose(mod_resid[1, 1, :], resids[1, 1, :])
    endog = np.arange(10)
    mod = markov_autoregression.MarkovAutoregression(endog, k_regimes=3, order=2)
    assert_equal(mod.nobs, 8)
    assert_equal(mod.endog, np.arange(2, 10))
    params = np.r_[[0.3] * 6, 2.0, 3.0, 4, 1.0, 0.1, 0.5, 0.8, -0.05, -0.25, -0.4]
    mod_resid = mod._resid(params)
    resids = np.zeros((3, 3, 3, mod.nobs))
    resids[0, 0, 0, :] = (
        np.arange(2, 10)
        - 2.0
        - 0.1 * (np.arange(1, 9) - 2.0)
        - -0.05 * (np.arange(8) - 2.0)
    )
    assert_allclose(mod_resid[0, 0, 0, :], resids[0, 0, 0, :])
    resids[1, 0, 0, :] = (
        np.arange(2, 10)
        - 3.0
        - 0.5 * (np.arange(1, 9) - 2.0)
        - -0.25 * (np.arange(8) - 2.0)
    )
    assert_allclose(mod_resid[1, 0, 0, :], resids[1, 0, 0, :])
    resids[0, 2, 1, :] = (
        np.arange(2, 10)
        - 2.0
        - 0.1 * (np.arange(1, 9) - 4.0)
        - -0.05 * (np.arange(8) - 3.0)
    )
    assert_allclose(mod_resid[0, 2, 1, :], resids[0, 2, 1, :])
    endog = np.arange(10)
    exog = np.r_[0.4, 5, 0.2, 1.2, -0.3, 2.5, 0.2, -0.7, 2.0, -1.1]
    mod = markov_autoregression.MarkovAutoregression(
        endog, k_regimes=2, order=1, exog=exog
    )
    assert_equal(mod.nobs, 9)
    assert_equal(mod.endog, np.arange(1, 10))
    params = np.r_[0.5, 0.5, 2.0, 3.0, 1.5, 1.0, 0.1, 0.5]
    mod_resid = mod._resid(params)
    resids = np.zeros((2, 2, mod.nobs))
    resids[0, 0, :] = (
        np.arange(1, 10)
        - 2.0
        - 1.5 * exog[1:]
        - 0.1 * (np.arange(9) - 2.0 - 1.5 * exog[:-1])
    )
    assert_allclose(mod_resid[0, 0, :], resids[0, 0, :])
    resids[0, 1, :] = (
        np.arange(1, 10)
        - 2.0
        - 1.5 * exog[1:]
        - 0.1 * (np.arange(9) - 3.0 - 1.5 * exog[:-1])
    )
    assert_allclose(mod_resid[0, 1, :], resids[0, 1, :])
    resids[1, 0, :] = (
        np.arange(1, 10)
        - 3.0
        - 1.5 * exog[1:]
        - 0.5 * (np.arange(9) - 2.0 - 1.5 * exog[:-1])
    )
    assert_allclose(mod_resid[1, 0, :], resids[1, 0, :])
    resids[1, 1, :] = (
        np.arange(1, 10)
        - 3.0
        - 1.5 * exog[1:]
        - 0.5 * (np.arange(9) - 3.0 - 1.5 * exog[:-1])
    )
    assert_allclose(mod_resid[1, 1, :], resids[1, 1, :])


def test_conditional_loglikelihoods():
    endog = np.ones(10)
    mod = markov_autoregression.MarkovAutoregression(endog, k_regimes=2, order=1)
    assert_equal(mod.nobs, 9)
    assert_equal(mod.endog, np.ones(9))
    params = np.r_[0.5, 0.5, 2.0, 3.0, 2.0, 0.1, 0.5]
    resid = mod._resid(params)
    conditional_likelihoods = np.exp(-0.5 * resid**2 / 2) / np.sqrt(2 * np.pi * 2)
    assert_allclose(
        mod._conditional_loglikelihoods(params), np.log(conditional_likelihoods)
    )
    endog = np.ones(10)
    mod = markov_autoregression.MarkovAutoregression(
        endog, k_regimes=3, order=1, switching_variance=True
    )
    assert_equal(mod.nobs, 9)
    assert_equal(mod.endog, np.ones(9))
    params = np.r_[[0.3] * 6, 2.0, 3.0, 4.0, 1.5, 3.0, 4.5, 0.1, 0.5, 0.8]
    mod_conditional_loglikelihoods = mod._conditional_loglikelihoods(params)
    conditional_likelihoods = mod._resid(params)
    conditional_likelihoods[0, :, :] = np.exp(
        -0.5 * conditional_likelihoods[0, :, :] ** 2 / 1.5
    ) / np.sqrt(2 * np.pi * 1.5)
    assert_allclose(
        mod_conditional_loglikelihoods[0, :, :],
        np.log(conditional_likelihoods[0, :, :]),
    )
    conditional_likelihoods[1, :, :] = np.exp(
        -0.5 * conditional_likelihoods[1, :, :] ** 2 / 3.0
    ) / np.sqrt(2 * np.pi * 3.0)
    assert_allclose(
        mod_conditional_loglikelihoods[1, :, :],
        np.log(conditional_likelihoods[1, :, :]),
    )
    conditional_likelihoods[2, :, :] = np.exp(
        -0.5 * conditional_likelihoods[2, :, :] ** 2 / 4.5
    ) / np.sqrt(2 * np.pi * 4.5)
    assert_allclose(
        mod_conditional_loglikelihoods[2, :, :],
        np.log(conditional_likelihoods[2, :, :]),
    )


class MarkovAutoregression:

    @classmethod
    def setup_class(cls, true, endog, atol=1e-05, rtol=1e-07, **kwargs):
        cls.model = markov_autoregression.MarkovAutoregression(endog, **kwargs)
        cls.true = true
        cls.result = cls.model.smooth(cls.true["params"])
        cls.atol = atol
        cls.rtol = rtol

    def test_llf(self):
        assert_allclose(
            self.result.llf, self.true["llf"], atol=self.atol, rtol=self.rtol
        )

    def test_fit(self, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self.model.fit(disp=False, **kwargs)
        assert_allclose(res.llf, self.true["llf_fit"], atol=self.atol, rtol=self.rtol)

    @pytest.mark.smoke
    def test_fit_em(self, **kwargs):
        res_em = self.model._fit_em(**kwargs)
        assert_allclose(
            res_em.llf, self.true["llf_fit_em"], atol=self.atol, rtol=self.rtol
        )


hamilton_ar2_short_filtered_joint_probabilities = np.array(
    [
        [
            [
                [
                    0.0499506987,
                    0.000644048275,
                    6.2222714e-05,
                    4.45756755e-06,
                    5.26645567e-07,
                    7.99846146e-07,
                    1.19425705e-05,
                    0.00687762063,
                ],
                [
                    0.0195930395,
                    0.000325884335,
                    0.000112955091,
                    0.000338537103,
                    9.81927968e-06,
                    2.7169675e-05,
                    0.0058382829,
                    0.0764261509,
                ],
            ],
            [
                [
                    0.00197113193,
                    9.50372207e-05,
                    0.000198390978,
                    1.88188953e-06,
                    4.834494e-07,
                    1.1487286e-05,
                    4.02918239e-06,
                    0.000435015431,
                ],
                [
                    0.0224870443,
                    0.00127331172,
                    0.00962155856,
                    0.00404178695,
                    0.000275516282,
                    0.0118179572,
                    0.0599778157,
                    0.148149567,
                ],
            ],
        ],
        [
            [
                [
                    0.0670912859,
                    0.0184223872,
                    0.000255621792,
                    4.48500688e-05,
                    7.80481515e-05,
                    2.73734559e-06,
                    7.59835896e-06,
                    0.00142930726,
                ],
                [
                    0.0210053328,
                    0.00744036383,
                    0.000370388879,
                    0.0027187837,
                    0.00116152088,
                    7.42182691e-05,
                    0.00296490192,
                    0.0126774695,
                ],
            ],
            [
                [
                    0.0809335679,
                    0.0831016518,
                    0.024914908,
                    0.000578825626,
                    0.00219019941,
                    0.0012017913,
                    7.8365943e-05,
                    0.00276363377,
                ],
                [
                    0.736967899,
                    0.888697316,
                    0.964463954,
                    0.992270877,
                    0.996283886,
                    0.986863839,
                    0.931117063,
                    0.751241236,
                ],
            ],
        ],
    ]
)
hamilton_ar2_short_predicted_joint_probabilities = np.array(
    [
        [
            [
                [
                    [
                        0.120809334,
                        0.0376964436,
                        0.000486045844,
                        4.69578023e-05,
                        3.36400588e-06,
                        3.9744519e-07,
                        6.0362229e-07,
                        9.01273552e-06,
                    ],
                    [
                        0.0392723623,
                        0.0147863379,
                        0.000245936108,
                        8.52441571e-05,
                        0.000255484811,
                        7.41034525e-06,
                        2.05042201e-05,
                        0.00440599447,
                    ],
                ],
                [
                    [
                        0.0049913123,
                        0.00148756005,
                        7.17220245e-05,
                        0.000149720314,
                        1.42021122e-06,
                        3.64846209e-07,
                        8.66914462e-06,
                        3.04071516e-06,
                    ],
                    [
                        0.0470476003,
                        0.0169703652,
                        0.000960933974,
                        0.00726113047,
                        0.00305022748,
                        0.000207924699,
                        0.00891869322,
                        0.0452636381,
                    ],
                ],
            ],
            [
                [
                    [
                        0.0049913123,
                        0.00643506069,
                        0.00176698327,
                        2.45179642e-05,
                        4.30179435e-06,
                        7.48598845e-06,
                        2.62552503e-07,
                        7.287966e-07,
                    ],
                    [
                        0.00162256192,
                        0.0020147265,
                        0.000713642497,
                        3.55258493e-05,
                        0.000260772139,
                        0.000111407276,
                        7.11864528e-06,
                        0.000284378568,
                    ],
                ],
                [
                    [
                        0.00597950448,
                        0.00776274317,
                        0.00797069493,
                        0.0023897134,
                        5.55180599e-05,
                        0.000210072977,
                        0.000115269812,
                        7.51646942e-06,
                    ],
                    [
                        0.0563621989,
                        0.070686276,
                        0.085239403,
                        0.0925065601,
                        0.0951736612,
                        0.0955585689,
                        0.0946550451,
                        0.0893080931,
                    ],
                ],
            ],
        ],
        [
            [
                [
                    [
                        0.0392723623,
                        0.0122542551,
                        0.000158002431,
                        1.52649118e-05,
                        1.09356167e-06,
                        1.29200377e-07,
                        1.96223855e-07,
                        2.929835e-06,
                    ],
                    [
                        0.0127665503,
                        0.00480670161,
                        7.99482261e-05,
                        2.77109335e-05,
                        8.30522919e-05,
                        2.40893443e-06,
                        6.66545485e-06,
                        0.00143228843,
                    ],
                ],
                [
                    [
                        0.00162256192,
                        0.000483571884,
                        2.33151963e-05,
                        4.86706634e-05,
                        4.61678312e-07,
                        1.18603191e-07,
                        2.81814142e-06,
                        9.88467229e-07,
                    ],
                    [
                        0.0152941031,
                        0.00551667911,
                        0.000312377744,
                        0.0023604281,
                        0.000991559466,
                        6.7591583e-05,
                        0.00289926399,
                        0.0147141776,
                    ],
                ],
            ],
            [
                [
                    [
                        0.0470476003,
                        0.0606562252,
                        0.016655404,
                        0.000231103828,
                        4.05482745e-05,
                        7.05621631e-05,
                        2.47479309e-06,
                        6.86956236e-06,
                    ],
                    [
                        0.0152941031,
                        0.0189906063,
                        0.00672672133,
                        0.000334863029,
                        0.00245801156,
                        0.00105011361,
                        6.70996238e-05,
                        0.00268052335,
                    ],
                ],
                [
                    [
                        0.0563621989,
                        0.0731708248,
                        0.0751309569,
                        0.0225251946,
                        0.000523307566,
                        0.00198012644,
                        0.00108652148,
                        7.08494735e-05,
                    ],
                    [
                        0.531264334,
                        0.666281623,
                        0.803457913,
                        0.871957394,
                        0.897097216,
                        0.900725317,
                        0.892208794,
                        0.84180897,
                    ],
                ],
            ],
        ],
    ]
)
hamilton_ar2_short_smoothed_joint_probabilities = np.array(
    [
        [
            [
                [
                    0.0129898189,
                    0.000166298475,
                    1.29822987e-05,
                    9.95268382e-07,
                    1.84473346e-07,
                    7.18761267e-07,
                    1.69576494e-05,
                    0.00687762063,
                ],
                [
                    0.00509522472,
                    8.41459714e-05,
                    2.35672254e-05,
                    7.55872505e-05,
                    3.43949612e-06,
                    2.4415333e-05,
                    0.00828997024,
                    0.0764261509,
                ],
            ],
            [
                [
                    0.000590021731,
                    2.55342733e-05,
                    4.50698224e-05,
                    5.30734135e-07,
                    1.80741761e-07,
                    1.11483792e-05,
                    5.98539007e-06,
                    0.000435015431,
                ],
                [
                    0.00673107901,
                    0.000342109009,
                    0.00218579464,
                    0.00113987259,
                    0.000103004157,
                    0.0114692946,
                    0.089097635,
                    0.148149567,
                ],
            ],
        ],
        [
            [
                [
                    0.0634648123,
                    0.0179187451,
                    0.000237462147,
                    3.55542558e-05,
                    7.63980455e-05,
                    2.9052082e-06,
                    8.17644492e-06,
                    0.00142930726,
                ],
                [
                    0.0198699352,
                    0.00723695477,
                    0.000344076057,
                    0.00215527721,
                    0.00113696383,
                    7.87695658e-05,
                    0.00319047276,
                    0.0126774695,
                ],
            ],
            [
                [
                    0.0881925054,
                    0.0833092133,
                    0.0251106301,
                    0.00058100747,
                    0.00219065072,
                    0.0012022135,
                    7.56893839e-05,
                    0.00276363377,
                ],
                [
                    0.803066603,
                    0.890916999,
                    0.972040418,
                    0.996011175,
                    0.996489179,
                    0.987210535,
                    0.899315113,
                    0.751241236,
                ],
            ],
        ],
    ]
)


class TestHamiltonAR2Short(MarkovAutoregression):

    @classmethod
    def setup_class(cls):
        true = {
            "params": np.r_[
                0.754673,
                0.095915,
                -0.358811,
                1.163516,
                np.exp(-0.262658) ** 2,
                0.013486,
                -0.057521,
            ],
            "llf": -10.14066,
            "llf_fit": -4.0523073,
            "llf_fit_em": -8.885836,
        }
        super().setup_class(true, rgnp[-10:], k_regimes=2, order=2, switching_ar=False)

    def test_fit_em(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super().test_fit_em()

    def test_filter_output(self, **kwargs):
        res = self.result
        assert_allclose(
            res.filtered_joint_probabilities,
            hamilton_ar2_short_filtered_joint_probabilities,
        )
        desired = hamilton_ar2_short_predicted_joint_probabilities
        if desired.ndim > res.predicted_joint_probabilities.ndim:
            desired = desired.sum(axis=-2)
        assert_allclose(res.predicted_joint_probabilities, desired)

    def test_smoother_output(self, **kwargs):
        res = self.result
        assert_allclose(
            res.filtered_joint_probabilities,
            hamilton_ar2_short_filtered_joint_probabilities,
        )
        desired = hamilton_ar2_short_predicted_joint_probabilities
        if desired.ndim > res.predicted_joint_probabilities.ndim:
            desired = desired.sum(axis=-2)
        assert_allclose(res.predicted_joint_probabilities, desired)
        assert_allclose(
            res.smoothed_joint_probabilities[..., -1],
            hamilton_ar2_short_smoothed_joint_probabilities[..., -1],
        )
        assert_allclose(
            res.smoothed_joint_probabilities[..., -2],
            hamilton_ar2_short_smoothed_joint_probabilities[..., -2],
        )
        assert_allclose(
            res.smoothed_joint_probabilities[..., -3],
            hamilton_ar2_short_smoothed_joint_probabilities[..., -3],
        )
        assert_allclose(
            res.smoothed_joint_probabilities[..., :-3],
            hamilton_ar2_short_smoothed_joint_probabilities[..., :-3],
        )


hamilton_ar4_filtered = [
    0.776712,
    0.949192,
    0.99632,
    0.990258,
    0.940111,
    0.537442,
    0.140001,
    0.008942,
    0.04848,
    0.614097,
    0.910889,
    0.995463,
    0.979465,
    0.992324,
    0.984561,
    0.751038,
    0.776268,
    0.522048,
    0.814956,
    0.821786,
    0.472729,
    0.673567,
    0.029031,
    0.001556,
    0.433276,
    0.985463,
    0.995025,
    0.966067,
    0.998445,
    0.801467,
    0.960997,
    0.996431,
    0.461365,
    0.199357,
    0.027398,
    0.703626,
    0.946388,
    0.985321,
    0.998244,
    0.989567,
    0.98451,
    0.986811,
    0.793788,
    0.973675,
    0.984848,
    0.990418,
    0.918427,
    0.998769,
    0.977647,
    0.978742,
    0.927635,
    0.998691,
    0.988934,
    0.991654,
    0.999288,
    0.999073,
    0.918636,
    0.98771,
    0.966876,
    0.910015,
    0.82615,
    0.969451,
    0.844049,
    0.941525,
    0.993363,
    0.949978,
    0.615206,
    0.970915,
    0.787585,
    0.707818,
    0.200476,
    0.050835,
    0.140723,
    0.80985,
    0.086422,
    0.990344,
    0.785963,
    0.817425,
    0.659152,
    0.996578,
    0.99286,
    0.948501,
    0.996883,
    0.999712,
    0.906694,
    0.725013,
    0.96369,
    0.38696,
    0.241302,
    0.009078,
    0.015789,
    0.000896,
    0.54153,
    0.928686,
    0.953704,
    0.992741,
    0.935877,
    0.918958,
    0.977316,
    0.987941,
    0.9873,
    0.996769,
    0.645469,
    0.921285,
    0.999917,
    0.949335,
    0.968914,
    0.886025,
    0.777141,
    0.904381,
    0.368277,
    0.607429,
    0.002491,
    0.22761,
    0.871284,
    0.987717,
    0.288705,
    0.512124,
    0.030329,
    0.005177,
    0.256183,
    0.020955,
    0.05162,
    0.549009,
    0.991715,
    0.987892,
    0.995377,
    0.999833,
    0.993756,
    0.956164,
    0.927714,
]
hamilton_ar4_smoothed = [
    0.968096,
    0.991071,
    0.998559,
    0.958534,
    0.540652,
    0.072784,
    0.010999,
    0.006228,
    0.172144,
    0.898574,
    0.989054,
    0.998293,
    0.986434,
    0.993248,
    0.976868,
    0.858521,
    0.847452,
    0.67567,
    0.596294,
    0.165407,
    0.03527,
    0.127967,
    0.007414,
    0.004944,
    0.815829,
    0.998128,
    0.998091,
    0.993227,
    0.999283,
    0.9211,
    0.977171,
    0.971757,
    0.12468,
    0.06371,
    0.11457,
    0.954701,
    0.994852,
    0.997302,
    0.999345,
    0.995817,
    0.996218,
    0.99458,
    0.93399,
    0.996054,
    0.998151,
    0.996976,
    0.971489,
    0.999786,
    0.997362,
    0.996755,
    0.993053,
    0.999947,
    0.998469,
    0.997987,
    0.99983,
    0.99936,
    0.953176,
    0.992673,
    0.975235,
    0.938121,
    0.946784,
    0.986897,
    0.905792,
    0.969755,
    0.995379,
    0.91448,
    0.772814,
    0.931385,
    0.541742,
    0.394596,
    0.063428,
    0.027829,
    0.124527,
    0.286105,
    0.069362,
    0.99595,
    0.961153,
    0.962449,
    0.945022,
    0.999855,
    0.998943,
    0.980041,
    0.999028,
    0.999838,
    0.863305,
    0.607421,
    0.575983,
    0.0133,
    0.007562,
    0.000635,
    0.001806,
    0.002196,
    0.80355,
    0.972056,
    0.984503,
    0.998059,
    0.985211,
    0.988486,
    0.994452,
    0.994498,
    0.998873,
    0.999192,
    0.870482,
    0.976282,
    0.999961,
    0.984283,
    0.973045,
    0.786176,
    0.403673,
    0.275418,
    0.115199,
    0.25756,
    0.004735,
    0.493936,
    0.90736,
    0.873199,
    0.052959,
    0.076008,
    0.001653,
    0.000847,
    0.062027,
    0.021257,
    0.219547,
    0.955654,
    0.999851,
    0.997685,
    0.998324,
    0.999939,
    0.996858,
    0.969209,
    0.927714,
]


class TestHamiltonAR4(MarkovAutoregression):

    @classmethod
    def setup_class(cls):
        true = {
            "params": np.r_[
                0.754673,
                0.095915,
                -0.358811,
                1.163516,
                np.exp(-0.262658) ** 2,
                0.013486,
                -0.057521,
                -0.246983,
                -0.212923,
            ],
            "llf": -181.26339,
            "llf_fit": -181.26339,
            "llf_fit_em": -183.85444,
            "bse_oim": np.r_[
                0.0965189,
                0.0377362,
                0.2645396,
                0.0745187,
                np.nan,
                0.1199942,
                0.137663,
                0.1069103,
                0.1105311,
            ],
        }
        super().setup_class(true, rgnp, k_regimes=2, order=4, switching_ar=False)

    def test_filtered_regimes(self):
        res = self.result
        assert_equal(len(res.filtered_marginal_probabilities[:, 1]), self.model.nobs)
        assert_allclose(
            res.filtered_marginal_probabilities[:, 1], hamilton_ar4_filtered, atol=1e-05
        )

    def test_smoothed_regimes(self):
        res = self.result
        assert_equal(len(res.smoothed_marginal_probabilities[:, 1]), self.model.nobs)
        assert_allclose(
            res.smoothed_marginal_probabilities[:, 1], hamilton_ar4_smoothed, atol=1e-05
        )

    def test_bse(self):
        bse = self.result.cov_params_approx.diagonal() ** 0.5
        assert_allclose(bse[:4], self.true["bse_oim"][:4], atol=1e-06)
        assert_allclose(bse[6:], self.true["bse_oim"][6:], atol=1e-06)


class TestHamiltonAR2Switch(MarkovAutoregression):

    @classmethod
    def setup_class(cls):
        path = Path(current_path).joinpath("results", "results_predict_rgnp.csv")
        results = pd.read_csv(path)
        true = {
            "params": np.r_[
                0.3812383,
                0.3564492,
                -0.0055216,
                1.195482,
                0.6677098**2,
                0.3710719,
                0.4621503,
                0.7002937,
                -0.3206652,
            ],
            "llf": -179.32354,
            "llf_fit": -179.38684,
            "llf_fit_em": -184.99606,
            "bse_oim": np.r_[
                0.1424841,
                0.0994742,
                0.2057086,
                0.1225987,
                np.nan,
                0.1754383,
                0.1652473,
                0.187409,
                0.1295937,
            ],
            "smoothed0": results.iloc[3:]["switchar2_sm1"],
            "smoothed1": results.iloc[3:]["switchar2_sm2"],
            "predict0": results.iloc[3:]["switchar2_yhat1"],
            "predict1": results.iloc[3:]["switchar2_yhat2"],
            "predict_predicted": results.iloc[3:]["switchar2_pyhat"],
            "predict_filtered": results.iloc[3:]["switchar2_fyhat"],
            "predict_smoothed": results.iloc[3:]["switchar2_syhat"],
        }
        super().setup_class(true, rgnp, k_regimes=2, order=2)

    def test_smoothed_marginal_probabilities(self):
        assert_allclose(
            self.result.smoothed_marginal_probabilities[:, 0],
            self.true["smoothed0"],
            atol=1e-06,
        )
        assert_allclose(
            self.result.smoothed_marginal_probabilities[:, 1],
            self.true["smoothed1"],
            atol=1e-06,
        )

    def test_predict(self):
        actual = self.model.predict(self.true["params"], probabilities="smoothed")
        assert_allclose(actual, self.true["predict_smoothed"], atol=1e-06)
        actual = self.model.predict(self.true["params"], probabilities=None)
        assert_allclose(actual, self.true["predict_smoothed"], atol=1e-06)
        actual = self.result.predict(probabilities="smoothed")
        assert_allclose(actual, self.true["predict_smoothed"], atol=1e-06)
        actual = self.result.predict(probabilities=None)
        assert_allclose(actual, self.true["predict_smoothed"], atol=1e-06)

    def test_bse(self):
        bse = self.result.cov_params_approx.diagonal() ** 0.5
        assert_allclose(bse[:4], self.true["bse_oim"][:4], atol=1e-07)
        assert_allclose(bse[6:], self.true["bse_oim"][6:], atol=1e-07)


hamilton_ar1_switch_filtered = [
    0.840288,
    0.730337,
    0.900234,
    0.596492,
    0.921618,
    0.983828,
    0.959039,
    0.898366,
    0.477335,
    0.251089,
    0.049367,
    0.386782,
    0.942868,
    0.965632,
    0.982857,
    0.897603,
    0.946986,
    0.916413,
    0.640912,
    0.849296,
    0.778371,
    0.95442,
    0.929906,
    0.72393,
    0.891196,
    0.061163,
    0.004806,
    0.977369,
    0.997871,
    0.97795,
    0.89658,
    0.963246,
    0.430539,
    0.906586,
    0.974589,
    0.514506,
    0.683457,
    0.276571,
    0.956475,
    0.966993,
    0.971618,
    0.987019,
    0.91667,
    0.921652,
    0.930265,
    0.655554,
    0.965858,
    0.964981,
    0.97679,
    0.868267,
    0.98324,
    0.852052,
    0.91915,
    0.854467,
    0.987868,
    0.93584,
    0.958138,
    0.979535,
    0.956541,
    0.716322,
    0.919035,
    0.866437,
    0.899609,
    0.914667,
    0.976448,
    0.867252,
    0.953075,
    0.97785,
    0.884242,
    0.688299,
    0.968461,
    0.737517,
    0.870674,
    0.559413,
    0.380339,
    0.582813,
    0.941311,
    0.24002,
    0.999349,
    0.619258,
    0.828343,
    0.729726,
    0.991009,
    0.966291,
    0.899148,
    0.970798,
    0.977684,
    0.695877,
    0.637555,
    0.915824,
    0.4346,
    0.771277,
    0.113756,
    0.144002,
    0.008466,
    0.99486,
    0.993173,
    0.961722,
    0.978555,
    0.789225,
    0.836283,
    0.940383,
    0.968368,
    0.974473,
    0.980248,
    0.518125,
    0.904086,
    0.993023,
    0.802936,
    0.920906,
    0.685445,
    0.666524,
    0.923285,
    0.643861,
    0.938184,
    0.008862,
    0.945406,
    0.990061,
    0.9915,
    0.486669,
    0.805039,
    0.089036,
    0.025067,
    0.863309,
    0.352784,
    0.733295,
    0.92871,
    0.984257,
    0.926597,
    0.959887,
    0.984051,
    0.872682,
    0.824375,
    0.780157,
]
hamilton_ar1_switch_smoothed = [
    0.900074,
    0.758232,
    0.914068,
    0.637248,
    0.901951,
    0.979905,
    0.958935,
    0.888641,
    0.261602,
    0.148761,
    0.056919,
    0.424396,
    0.932184,
    0.954962,
    0.983958,
    0.895595,
    0.949519,
    0.923473,
    0.678898,
    0.848793,
    0.807294,
    0.958868,
    0.942936,
    0.809137,
    0.960892,
    0.032947,
    0.007127,
    0.967967,
    0.996551,
    0.979278,
    0.896181,
    0.987462,
    0.498965,
    0.908803,
    0.986893,
    0.48872,
    0.640492,
    0.325552,
    0.951996,
    0.959703,
    0.960914,
    0.986989,
    0.916779,
    0.92457,
    0.935348,
    0.677118,
    0.960749,
    0.958966,
    0.976974,
    0.838045,
    0.986562,
    0.847774,
    0.908866,
    0.82111,
    0.984965,
    0.915302,
    0.938196,
    0.976518,
    0.97378,
    0.744159,
    0.922006,
    0.873292,
    0.904035,
    0.917547,
    0.978559,
    0.870915,
    0.94842,
    0.979747,
    0.884791,
    0.711085,
    0.973235,
    0.726311,
    0.828305,
    0.446642,
    0.411135,
    0.639357,
    0.973151,
    0.141707,
    0.999805,
    0.618207,
    0.783239,
    0.672193,
    0.987618,
    0.964655,
    0.87739,
    0.962437,
    0.989002,
    0.692689,
    0.69937,
    0.937934,
    0.522535,
    0.824567,
    0.058746,
    0.146549,
    0.009864,
    0.994072,
    0.992084,
    0.956945,
    0.984297,
    0.795926,
    0.845698,
    0.935364,
    0.963285,
    0.972767,
    0.992168,
    0.528278,
    0.826349,
    0.996574,
    0.811431,
    0.930873,
    0.680756,
    0.721072,
    0.937977,
    0.731879,
    0.996745,
    0.016121,
    0.951187,
    0.98982,
    0.996968,
    0.592477,
    0.889144,
    0.036015,
    0.040084,
    0.858128,
    0.418984,
    0.746265,
    0.90799,
    0.980984,
    0.900449,
    0.934741,
    0.986807,
    0.872818,
    0.81208,
    0.780157,
]


class TestHamiltonAR1Switch(MarkovAutoregression):

    @classmethod
    def setup_class(cls):
        true = {
            "params": np.r_[
                0.85472458,
                0.53662099,
                1.041419,
                -0.479157,
                np.exp(-0.231404) ** 2,
                0.243128,
                0.713029,
            ],
            "llf": -186.7575,
            "llf_fit": -186.7575,
            "llf_fit_em": -189.25446,
        }
        super().setup_class(true, rgnp, k_regimes=2, order=1)

    def test_filtered_regimes(self):
        assert_allclose(
            self.result.filtered_marginal_probabilities[:, 0],
            hamilton_ar1_switch_filtered,
            atol=1e-05,
        )

    def test_smoothed_regimes(self):
        assert_allclose(
            self.result.smoothed_marginal_probabilities[:, 0],
            hamilton_ar1_switch_smoothed,
            atol=1e-05,
        )

    def test_expected_durations(self):
        expected_durations = [6.883477, 1.863513]
        assert_allclose(self.result.expected_durations, expected_durations, atol=1e-05)


hamilton_ar1_switch_tvtp_filtered = [
    0.999996,
    0.999211,
    0.999849,
    0.996007,
    0.999825,
    0.999991,
    0.999981,
    0.999819,
    0.041745,
    0.001116,
    1.74e-05,
    0.000155,
    0.999976,
    0.999958,
    0.999993,
    0.999878,
    0.99994,
    0.999791,
    0.996553,
    0.999486,
    0.998485,
    0.999894,
    0.999765,
    0.997657,
    0.999619,
    0.002853,
    1.09e-05,
    0.999884,
    0.999996,
    0.999997,
    0.999919,
    0.999987,
    0.989762,
    0.999807,
    0.999978,
    0.050734,
    0.01066,
    0.000217,
    0.006174,
    0.999977,
    0.999954,
    0.999995,
    0.999934,
    0.999867,
    0.999824,
    0.996783,
    0.999941,
    0.999948,
    0.999981,
    0.999658,
    0.999994,
    0.999753,
    0.999859,
    0.99933,
    0.999993,
    0.999956,
    0.99997,
    0.999996,
    0.999991,
    0.998674,
    0.999869,
    0.999432,
    0.99957,
    0.9996,
    0.999954,
    0.999499,
    0.999906,
    0.999978,
    0.999712,
    0.997441,
    0.999948,
    0.998379,
    0.999578,
    0.994745,
    0.045936,
    0.006816,
    0.027384,
    0.000278,
    1.0,
    0.996382,
    0.999541,
    0.99813,
    0.999992,
    0.99999,
    0.99986,
    0.999986,
    0.999997,
    0.99852,
    0.997777,
    0.999821,
    0.033353,
    0.011629,
    6.95e-05,
    4.52e-05,
    2.04e-06,
    0.999963,
    0.999977,
    0.999949,
    0.999986,
    0.99924,
    0.999373,
    0.999858,
    0.999946,
    0.999972,
    0.999991,
    0.994039,
    0.999817,
    0.999999,
    0.999715,
    0.999924,
    0.997763,
    0.997944,
    0.999825,
    0.996592,
    0.695147,
    0.000161,
    0.999665,
    0.999928,
    0.999988,
    0.992742,
    0.374214,
    0.001569,
    2.16e-05,
    0.000941,
    4.32e-05,
    0.000556,
    0.999955,
    0.999993,
    0.999942,
    0.999973,
    0.999999,
    0.999919,
    0.999438,
    0.998738,
]
hamilton_ar1_switch_tvtp_smoothed = [
    0.999997,
    0.999246,
    0.999918,
    0.996118,
    0.99974,
    0.99999,
    0.999984,
    0.999783,
    0.035454,
    0.000958,
    1.53e-05,
    0.000139,
    0.999973,
    0.999939,
    0.999994,
    0.99987,
    0.999948,
    0.999884,
    0.997243,
    0.999668,
    0.998424,
    0.999909,
    0.99986,
    0.998037,
    0.999559,
    0.002533,
    1.16e-05,
    0.999801,
    0.999993,
    0.999997,
    0.999891,
    0.999994,
    0.990096,
    0.999753,
    0.999974,
    0.048495,
    0.009289,
    0.000542,
    0.005991,
    0.999974,
    0.999929,
    0.999995,
    0.999939,
    0.99988,
    0.999901,
    0.996221,
    0.999937,
    0.999935,
    0.999985,
    0.99945,
    0.999995,
    0.999768,
    0.999897,
    0.99893,
    0.999992,
    0.999949,
    0.999954,
    0.999995,
    0.999994,
    0.998687,
    0.999902,
    0.999547,
    0.999653,
    0.999538,
    0.999966,
    0.999485,
    0.999883,
    0.999982,
    0.999831,
    0.99694,
    0.999968,
    0.998678,
    0.99978,
    0.993895,
    0.055372,
    0.020421,
    0.022913,
    0.000127,
    1.0,
    0.997072,
    0.999715,
    0.996893,
    0.99999,
    0.999991,
    0.999811,
    0.999978,
    0.999998,
    0.9991,
    0.997866,
    0.999787,
    0.034912,
    0.009932,
    5.91e-05,
    3.99e-05,
    1.77e-06,
    0.999954,
    0.999976,
    0.999932,
    0.999991,
    0.999429,
    0.999393,
    0.999845,
    0.999936,
    0.999961,
    0.999995,
    0.994246,
    0.99957,
    1.0,
    0.999702,
    0.999955,
    0.998611,
    0.998019,
    0.999902,
    0.998486,
    0.673991,
    0.000205,
    0.999627,
    0.999902,
    0.999994,
    0.993707,
    0.338707,
    0.001359,
    2.36e-05,
    0.000792,
    4.47e-05,
    0.000565,
    0.999932,
    0.999993,
    0.999931,
    0.99995,
    0.999999,
    0.99994,
    0.999626,
    0.998738,
]
expected_durations = [
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [1.223309, 1864.084],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
    [710.7573, 1.000391],
]


class TestHamiltonAR1SwitchTVTP(MarkovAutoregression):

    @classmethod
    def setup_class(cls):
        true = {
            "params": np.r_[
                6.564923,
                7.846371,
                -8.064123,
                -15.37636,
                1.02719,
                -0.71976,
                np.exp(-0.217003) ** 2,
                0.161489,
                0.022536,
            ],
            "llf": -163.914049,
            "llf_fit": -161.786477,
            "llf_fit_em": -163.914049,
        }
        exog_tvtp = np.c_[np.ones(len(rgnp)), rec]
        super().setup_class(true, rgnp, k_regimes=2, order=1, exog_tvtp=exog_tvtp)

    @pytest.mark.skip
    def test_fit_em(self):
        pass

    def test_filtered_regimes(self):
        assert_allclose(
            self.result.filtered_marginal_probabilities[:, 0],
            hamilton_ar1_switch_tvtp_filtered,
            atol=1e-05,
        )

    def test_smoothed_regimes(self):
        assert_allclose(
            self.result.smoothed_marginal_probabilities[:, 0],
            hamilton_ar1_switch_tvtp_smoothed,
            atol=1e-05,
        )

    def test_expected_durations(self):
        assert_allclose(
            self.result.expected_durations, expected_durations, rtol=1e-05, atol=1e-07
        )


class TestFilardo(MarkovAutoregression):

    @classmethod
    def setup_class(cls):
        path = Path(current_path).joinpath("results", "mar_filardo.csv")
        cls.mar_filardo = pd.read_csv(path)
        true = {
            "params": np.r_[
                4.35941747,
                -1.6493936,
                1.7702123,
                0.9945672,
                0.517298,
                -0.865888,
                np.exp(-0.362469) ** 2,
                0.189474,
                0.079344,
                0.110944,
                0.122251,
            ],
            "llf": -586.5718,
            "llf_fit": -586.5718,
            "llf_fit_em": -586.5718,
        }
        endog = cls.mar_filardo["dlip"].iloc[1:].values
        exog_tvtp = add_constant(cls.mar_filardo["dmdlleading"].iloc[:-1].values)
        super().setup_class(
            true, endog, k_regimes=2, order=4, switching_ar=False, exog_tvtp=exog_tvtp
        )

    @pytest.mark.skip
    def test_fit(self, **kwargs):
        pass

    @pytest.mark.skip
    def test_fit_em(self):
        pass

    def test_filtered_regimes(self):
        assert_allclose(
            self.result.filtered_marginal_probabilities[:, 0],
            self.mar_filardo["filtered_0"].iloc[5:],
            atol=1e-05,
        )

    def test_smoothed_regimes(self):
        assert_allclose(
            self.result.smoothed_marginal_probabilities[:, 0],
            self.mar_filardo["smoothed_0"].iloc[5:],
            atol=1e-05,
        )

    def test_expected_durations(self):
        assert_allclose(
            self.result.expected_durations,
            self.mar_filardo[["duration0", "duration1"]].iloc[5:],
            rtol=1e-05,
            atol=1e-07,
        )


class TestFilardoPandas(MarkovAutoregression):

    @classmethod
    def setup_class(cls):
        path = Path(current_path).joinpath("results", "mar_filardo.csv")
        cls.mar_filardo = pd.read_csv(path)
        cls.mar_filardo.index = pd.date_range("1948-02-01", "1991-04-01", freq="MS")
        true = {
            "params": np.r_[
                4.35941747,
                -1.6493936,
                1.7702123,
                0.9945672,
                0.517298,
                -0.865888,
                np.exp(-0.362469) ** 2,
                0.189474,
                0.079344,
                0.110944,
                0.122251,
            ],
            "llf": -586.5718,
            "llf_fit": -586.5718,
            "llf_fit_em": -586.5718,
        }
        endog = cls.mar_filardo["dlip"].iloc[1:]
        exog_tvtp = add_constant(cls.mar_filardo["dmdlleading"].iloc[:-1])
        super().setup_class(
            true, endog, k_regimes=2, order=4, switching_ar=False, exog_tvtp=exog_tvtp
        )

    @pytest.mark.skip
    def test_fit(self, **kwargs):
        pass

    @pytest.mark.skip
    def test_fit_em(self):
        pass

    def test_filtered_regimes(self):
        assert_allclose(
            self.result.filtered_marginal_probabilities[0],
            self.mar_filardo["filtered_0"].iloc[5:],
            atol=1e-05,
        )

    def test_smoothed_regimes(self):
        assert_allclose(
            self.result.smoothed_marginal_probabilities[0],
            self.mar_filardo["smoothed_0"].iloc[5:],
            atol=1e-05,
        )

    def test_expected_durations(self):
        assert_allclose(
            self.result.expected_durations,
            self.mar_filardo[["duration0", "duration1"]].iloc[5:],
            rtol=1e-05,
            atol=1e-07,
        )
