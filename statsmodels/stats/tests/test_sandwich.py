"""Tests for sandwich robust covariance estimation

see also in regression for cov_hac compared to Gretl and
sandbox.panel test_random_panel for comparing cov_cluster, cov_hac_panel and
cov_white

Created on Sat Dec 17 08:39:16 2011

Author: Josef Perktold
"""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

from statsmodels.regression.linear_model import OLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import add_constant


def test_cov_cluster_2groups():

    cur_dir = Path(__file__).parent.resolve()
    fpath = Path(cur_dir).joinpath("test_data.txt")
    pet = np.genfromtxt(fpath)
    endog = pet[:, -1]
    group = pet[:, 0].astype(int)
    time = pet[:, 1].astype(int)
    exog = add_constant(pet[:, 2])
    res = OLS(endog, exog).fit()
    cov01, covg, covt = sw.cov_cluster_2groups(res, group, group2=time)
    bse_petw = [0.0284, 0.0284]
    bse_pet0 = [0.067, 0.0506]
    bse_pet1 = [0.0234, 0.0334]
    bse_pet01 = [0.0651, 0.0536]
    bse_0 = sw.se_cov(covg)
    bse_1 = sw.se_cov(covt)
    bse_01 = sw.se_cov(cov01)
    assert_almost_equal(bse_petw, res.HC0_se, decimal=4)
    assert_almost_equal(bse_0, bse_pet0, decimal=4)
    assert_almost_equal(bse_1, bse_pet1, decimal=4)
    assert_almost_equal(bse_01, bse_pet01, decimal=4)


def test_hac_simple():
    from statsmodels.datasets import macrodata

    d2 = macrodata.load_pandas().data
    g_gdp = 400 * np.diff(np.log(d2["realgdp"].values))
    g_inv = 400 * np.diff(np.log(d2["realinv"].values))
    exogg = add_constant(np.c_[g_gdp, d2["realint"][:-1].values])
    res_olsg = OLS(g_inv, exogg).fit()
    cov1_r = [
        [+1.406438998786788, -0.31803287070833297, -0.06062111121648861],
        [-0.3180328707083329, 0.10973083489998187, +0.000395311760301478],
        [-0.06062111121648865, 0.0003953117603014895, +0.08751152891247099],
    ]
    cov2_r = [
        [+1.3855512908840137, -0.3133096102522685, -0.05972079768357048],
        [-0.3133096102522685, +0.10810116903513062, +0.000389440793564339],
        [-0.0597207976835705, +0.000389440793564336, +0.08621185274050362],
    ]
    cov1 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=True)
    se1 = sw.se_cov(cov1)
    cov2 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=False)
    se2 = sw.se_cov(cov2)
    assert_allclose(cov1, cov1_r)
    assert_allclose(cov2, cov2_r)
    assert_allclose(np.sqrt(np.diag(cov1_r)), se1)
    assert_allclose(np.sqrt(np.diag(cov2_r)), se2)
    cov3 = sw.cov_hac_simple(res_olsg, use_correction=False)
    cov4 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=False)
    assert_allclose(cov3, cov4)
