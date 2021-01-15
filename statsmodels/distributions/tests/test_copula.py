# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 23:32:57 2021

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

import statsmodels.sandbox.distributions.copula as cop


ev_list = [
    [cop.transform_bilogistic,0.5, 0.9, (0.25, 0.05), 0.5],
    [cop.transform_tawn, 0.5, 0.9, (0.5, 0.5, 0.5), 0.4724570876035117],
    # note evd has asymmetry reversed, interchange variables
    [cop.transform_tawn2, 0.9, 0.5, (0.25, 0.05), 0.464357480263932],
    [cop.transform_tawn2, 0.9, 0.5, (0.5, 0.25), 0.4916117128670654],
    [cop.transform_tawn2, 0.5, 0.9, (0.5, 0.25), 0.48340673415789],
    # note evd has parameter for hr 1/lmbda (inverse of our parameter)
    [cop.transform_hr, 0.5, 0.9, (2,), 0.4551235014298542],
    [cop.transform_joe, 0.5, 0.9, (0.5, 0.75, 1/0.25), 0.4543698299835434],
    [cop.transform_joe, 0.9, 0.5, (0.5, 0.75, 1/0.25), 0.4539773435983587],
    ]


@pytest.mark.parametrize("case", ev_list)
def test_ev_copula(case):
    # check ev copulas, cdf and transform
    ev_tr, v1, v2, args, res1 = case
    res = cop.copula_bv_ev(v1, v2, ev_tr, args=args)
    assert_allclose(res, res1, rtol=1e-13)
