"""
Created on Mar. 12, 2024 11:10:14 a.m.

Author: Josef Perktold
License: BSD-3
"""

import pytest

import numpy as np
from numpy.testing import assert_allclose

# from scipy import stats

from statsmodels.robust.norms import (
    AndrewWave,
    TrimmedMean,
    TukeyBiweight,
    TukeyQuartic,
    Hampel,
    HuberT,
    StudentT,
    )

from statsmodels.robust.tools import (
    _var_normal,
    _var_normal_jump,
    _get_tuning_param,
    tuning_s_estimator_mean,
    )


effs = [0.9, 0.95, 0.98, 0.99]

results_menenez = [
    (HuberT(), [0.9818, 1.345, 1.7459, 2.0102]),
    (TukeyBiweight(), [3.8827, 4.6851, 5.9207, 7.0414]),
    (TukeyQuartic(), [3.1576, 3.6175, 4.2103, 4.6664]),
    (TukeyQuartic(k=2), [3.8827, 4.6851, 5.9207, 7.0414]),  # biweight
    (StudentT(df=1), [1.7249, 2.3849, 3.3962, 4.2904]),  # Cauchy
    (AndrewWave(), [1.1117, 1.338, 1.6930, 2.0170]),
    # (Hampel(), [4.4209, 5.4, 7.00609, 8.0456]),
    # rounding problem in Hampel, menenez use a as tuning parameter
    (Hampel(), [4.4208, 5.5275, 7.006, 8.0456]),
    (TrimmedMean(), [2.5003, 2.7955, 3.1365, 3.3682]),
    ]


@pytest.mark.parametrize("case", results_menenez)
def test_eff(case):
    norm, res2 = case

    if norm.continuous == 2:
        var_func = _var_normal
    else:
        var_func = _var_normal_jump

    res_eff = []
    for c in res2:
        norm._set_tuning_param(c, inplace=True)
        res_eff.append(1 / var_func(norm))

    assert_allclose(res_eff, effs, atol=0.0005)

    for c in res2:
        # bp = stats.norm.expect(lambda x : norm.rho(x)) / norm.rho(norm.c)
        norm._set_tuning_param(c, inplace=True)
        eff = 1 / _var_normal(norm)
        tune = _get_tuning_param(norm, eff)
        assert_allclose(tune, c, rtol=1e-6, atol=5e-4)


def test_hampel_eff():
    # we cannot solve for multiple tuning parameters
    eff = 0.95
    # tuning parameters from Menezes et al 2021
    res_eff = 1 / _var_normal_jump(Hampel(a=1.35, b=2.70, c=5.40))
    assert_allclose(res_eff, eff, atol=0.005)


def test_tuning_biweight():
    # regression numbers but verified at 5 decimals
    norm = TukeyBiweight()
    res = tuning_s_estimator_mean(norm, breakdown=0.5)
    res1 = [0.28682611623149523, 1.5476449837305166, 0.1996004163055662]
    assert_allclose(res.all[1:], res1, rtol=1e-7)


@pytest.mark.parametrize("case", results_menenez)
def test_tuning_smoke(case):
    # regression numbers but verified at 5 decimals
    norm, _ = case
    # norm = Norm()
    if np.isfinite(norm.max_rho()):
        res = tuning_s_estimator_mean(norm, breakdown=0.5)
        assert res is not None
