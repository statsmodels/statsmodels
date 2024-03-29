"""
Created on Mar. 12, 2024 11:10:14 a.m.

Author: Josef Perktold
License: BSD-3
"""

import pytest

# import numpy as np
from numpy.testing import assert_allclose

# from scipy import stats

from statsmodels.robust.norms import (
    AndrewWave,
    TrimmedMean,
    TukeyBiweight,
    # TukeyQuartic,
    Hampel,
    )

from statsmodels.robust.tools import (
    _var_normal,
    _var_normal_jump,
    _get_tuning_param,
    tuning_s_estimator_mean,
    )


effs = [0.9, 0.95, 0.98, 0.99]

results_menenez = [
    (TukeyBiweight, [3.8827, 4.6851, 5.9207, 7.0414]),
    # (TukeyQuartic, [3.1576, 3.6175, 4.2103, 4.6664]),
    # (TukeyQuartic(k=2), [3.8827, 4.6851, 5.9207, 7.0414]),
    (AndrewWave, [1.1117, 1.338, 1.6930, 2.0170]),
    (TrimmedMean, [2.5003, 2.7955, 3.1365, 3.3682]),
    ]


@pytest.mark.parametrize("case", results_menenez)
def test_eff(case):
    Norm, res2 = case

    if Norm is not TrimmedMean:
        res_eff = [1 / _var_normal(Norm(c)) for c in res2]
    else:
        res_eff = [1 / _var_normal_jump(Norm(c)) for c in res2]

    assert_allclose(res_eff, effs, atol=0.0005)

    if Norm is TukeyBiweight:
        # threshold name is c andg norm max is rho(c)
        for c in res2:
            # bp = stats.norm.expect(lambda x : norm.rho(x)) / norm.rho(norm.c)
            eff = 1 / _var_normal(Norm(c))
            tune = _get_tuning_param(Norm, eff)
            assert_allclose(tune, c)


def test_hampel_eff():
    # we cannot solve for multiple tuning parameters
    eff = 0.95
    # tuning parameters from Menezes et al 2021
    res_eff = 1 / _var_normal_jump(Hampel(a=1.35, b=2.70, c=5.40))
    assert_allclose(res_eff, eff, atol=0.005)


def test_tuning_biweight():
    # regression numbers but verified at 5 decimals
    norm = TukeyBiweight
    res = tuning_s_estimator_mean(norm, breakdown=0.5)
    res1 = [0.28682611623149523, 1.5476449837305166, 0.1996004163055662]
    assert_allclose(res.all[1:], res1, rtol=1e-7)
