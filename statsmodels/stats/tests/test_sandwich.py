"""Tests for sandwich robust covariance estimation

see also in regression for cov_hac compared to Gretl and
sandbox.panel test_random_panel for comparing cov_cluster, cov_hac_panel and
cov_white

Created on Sat Dec 17 08:39:16 2011

Author: Josef Perktold
"""
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest

from statsmodels.regression.linear_model import OLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import add_constant


def test_cov_cluster_2groups():
    # comparing cluster robust standard errors to Peterson
    # requires Petersen's test_data
    # http://www.kellogg.northwestern.edu/faculty/petersen
    #      .../htm/papers/se/test_data.txt
    cur_dir = Path(__file__).parent.resolve()
    fpath = Path(cur_dir).joinpath("test_data.txt")
    pet = pd.read_csv(fpath, delimiter=r"\s+", header=None).values
    endog = pet[:, -1]
    group = pet[:, 0].astype(int)
    time = pet[:, 1].astype(int)
    exog = add_constant(pet[:, 2])
    res = OLS(endog, exog).fit()

    cov01, covg, covt = sw.cov_cluster_2groups(res, group, group2=time)

    # Reference number from Petersen
    # http://www.kellogg.northwestern.edu/faculty/petersen/htm
    #       .../papers/se/test_data.htm

    bse_petw = [0.0284, 0.0284]
    bse_pet0 = [0.0670, 0.0506]
    bse_pet1 = [0.0234, 0.0334]  # year
    bse_pet01 = [0.0651, 0.0536]  # firm and year
    bse_0 = sw.se_cov(covg)
    bse_1 = sw.se_cov(covt)
    bse_01 = sw.se_cov(cov01)
    # print res.HC0_se, bse_petw - res.HC0_se
    # print bse_0, bse_0 - bse_pet0
    # print bse_1, bse_1 - bse_pet1
    # print bse_01, bse_01 - bse_pet01
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

    # > NeweyWest(fm, lag = 4, prewhite = FALSE, sandwich = TRUE,
    #             verbose=TRUE, adjust=TRUE)
    # Lag truncation parameter chosen: 4
    #             (Intercept)                   ggdp                  lint
    cov1_r = [
        [+1.40643899878678802, -0.3180328707083329709, -0.060621111216488610],
        [-0.31803287070833292, 0.1097308348999818661, +0.000395311760301478],
        [-0.06062111121648865, 0.0003953117603014895, +0.087511528912470993]
    ]

    # > NeweyWest(fm, lag = 4, prewhite = FALSE, sandwich = TRUE,
    #             verbose=TRUE, adjust=FALSE)
    # Lag truncation parameter chosen: 4
    #         (Intercept)                  ggdp                  lint
    cov2_r = [
        [+1.3855512908840137, -0.313309610252268500, -0.059720797683570477],
        [-0.3133096102522685, +0.108101169035130618, +0.000389440793564339],
        [-0.0597207976835705, +0.000389440793564336, +0.086211852740503622]
    ]

    cov1 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=True)
    se1 = sw.se_cov(cov1)
    cov2 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=False)
    se2 = sw.se_cov(cov2)
    # Relax precision requirements for this test due to failure in NumPy 1.23
    assert_allclose(cov1, cov1_r)
    assert_allclose(cov2, cov2_r)
    assert_allclose(np.sqrt(np.diag(cov1_r)), se1)
    assert_allclose(np.sqrt(np.diag(cov2_r)), se2)

    # compare default for nlags
    cov3 = sw.cov_hac_simple(res_olsg, use_correction=False)
    cov4 = sw.cov_hac_simple(res_olsg, nlags=4, use_correction=False)
    assert_allclose(cov3, cov4)


def test_cov_hc0_to_hc3_match_results_attributes():
    # cov_hc0..cov_hc3 are exported from statsmodels.stats.api but were only
    # ever exercised in examples/ex_sandwich.py, which pytest does not
    # collect. RegressionResults independently computes the same HCx
    # covariances for its HCx_se attributes, giving a second implementation
    # to check the standalone functions against.
    rs = np.random.RandomState(12345)
    nobs, kvars = 100, 4
    x = rs.standard_normal((nobs, kvars - 1))
    exog = add_constant(x)
    beta = np.ones(kvars)
    sigma = 0.1 + np.exp(exog[:, -1])
    endog = exog.dot(beta) + sigma * rs.standard_normal(nobs)

    res = OLS(endog, exog).fit()

    assert_almost_equal(sw.se_cov(sw.cov_hc0(res)), res.HC0_se, decimal=13)
    assert_almost_equal(sw.se_cov(sw.cov_hc1(res)), res.HC1_se, decimal=13)
    assert_almost_equal(sw.se_cov(sw.cov_hc2(res)), res.HC2_se, decimal=13)
    assert_almost_equal(sw.se_cov(sw.cov_hc3(res)), res.HC3_se, decimal=13)

    assert_allclose(sw.cov_hc0(res), res.cov_HC0)
    assert_allclose(sw.cov_hc1(res), res.cov_HC1)
    assert_allclose(sw.cov_hc2(res), res.cov_HC2)
    assert_allclose(sw.cov_hc3(res), res.cov_HC3)


def test_cov_hac_negative_nlags():
    # a negative nlags used to leak a bare IndexError from the kernel loop
    rs = np.random.RandomState(12345)
    endog = rs.standard_normal(60)
    exog = np.column_stack([np.ones(60), rs.standard_normal((60, 2))])
    res = OLS(endog, exog).fit()
    with pytest.raises(ValueError, match="non-negative"):
        sw.cov_hac(res, nlags=-1)
    # nlags=0 (White) stays valid
    assert sw.cov_hac(res, nlags=0).shape == (3, 3)


def test_get_sandwich_arrays_tuple_has_default_df_resid():
    # the tuple form has no results instance, so df_resid falls back to
    # nobs - k_params
    rs = np.random.default_rng(12345)
    xu = rs.standard_normal((20, 3))
    hessian_inv = np.eye(3)

    xu_out, hessian_inv_out, df_resid = sw._get_sandwich_arrays((xu, hessian_inv))

    assert_allclose(xu_out, xu)
    assert_allclose(hessian_inv_out, hessian_inv)
    assert df_resid == 17


def _make_panel_sample():
    # small panel-like sample with a group and a time index
    rs = np.random.default_rng(20240607)
    n_groups, n_periods = 10, 12
    nobs = n_groups * n_periods
    group = np.repeat(np.arange(n_groups), n_periods)
    time = np.tile(np.arange(n_periods), n_groups)
    exog = add_constant(rs.standard_normal((nobs, 2)))
    endog = exog @ np.array([1.0, 0.5, -0.25]) + rs.standard_normal(nobs)
    bounds = np.arange(0, nobs + 1, n_periods)
    groupidx = list(pairwise(bounds))
    return endog, exog, group, time, groupidx


@pytest.mark.parametrize(
    "cov_type",
    [
        "white",
        "hac",
        "nw-panel-hac",
        "nw-panel-cluster",
        "nw-groupsum-hac",
        "nw-groupsum-cluster",
        "cluster",
    ],
)
def test_correction_uses_adjusted_df_resid(cov_type):
    # GH 9253: the small sample correction factor used to be hard coded as
    # nobs - k_params, so changes to results.df_resid had no effect
    endog, exog, group, time, groupidx = _make_panel_sample()
    mod = OLS(endog, exog)
    res = mod.fit()

    if cov_type == "white":
        cov_func, kwds = sw.cov_white_simple, {}
    elif cov_type == "hac":
        cov_func, kwds = sw.cov_hac_simple, {"nlags": 2}
    elif cov_type == "nw-panel-hac":
        cov_func = sw.cov_nw_panel
        kwds = {"nlags": 2, "groupidx": groupidx, "use_correction": "hac"}
    elif cov_type == "nw-panel-cluster":
        cov_func = sw.cov_nw_panel
        kwds = {"nlags": 2, "groupidx": groupidx, "use_correction": "cluster"}
    elif cov_type == "nw-groupsum-hac":
        cov_func = sw.cov_nw_groupsum
        kwds = {"nlags": 2, "time": time, "use_correction": "hac"}
    elif cov_type == "nw-groupsum-cluster":
        cov_func = sw.cov_nw_groupsum
        kwds = {"nlags": 2, "time": time, "use_correction": "cluster"}
    else:
        cov_func, kwds = sw.cov_cluster, {"group": group}

    cov = cov_func(res, **kwds)

    df_resid = res.df_resid
    # adjust df_resid on the model, as in the issue
    mod.df_resid = df_resid - 5
    res_adj = mod.fit()
    assert res_adj.df_resid == df_resid - 5

    cov_adj = cov_func(res_adj, **kwds)

    # only the small sample correction factor changed, it is scaled by the
    # ratio of the old and the new residual degrees of freedom
    assert_allclose(cov_adj, cov * df_resid / res_adj.df_resid)


@pytest.mark.parametrize("cov_type", ["cluster", "hac-panel", "hac-groupsum"])
def test_robust_cov_type_uses_adjusted_model_df_resid(cov_type):
    # GH 9253: the df_resid adjusted on the model has to be used by the
    # robust covariance, which is computed when the model is fitted
    endog, exog, group, time, groupidx = _make_panel_sample()

    if cov_type == "cluster":
        cov_kwds = {"groups": group}
    elif cov_type == "hac-panel":
        cov_kwds = {"groups": group, "maxlags": 2}
    else:
        cov_kwds = {"time": time, "maxlags": 2}

    res = OLS(endog, exog).fit(cov_type=cov_type, cov_kwds=cov_kwds)

    n_absorbed = 5
    mod = OLS(endog, exog)
    mod.df_model += n_absorbed
    mod.df_resid -= n_absorbed
    res_adj = mod.fit(cov_type=cov_type, cov_kwds=cov_kwds)

    assert res_adj.df_resid == res.df_resid - n_absorbed

    ratio = res.df_resid / res_adj.df_resid
    assert_allclose(res_adj.cov_params(), res.cov_params() * ratio)
    assert_allclose(res_adj.bse, res.bse * np.sqrt(ratio))


@pytest.mark.parametrize("cov_type", ["white", "cluster"])
def test_correction_ignores_increased_df_resid(cov_type):
    # df_resid is larger than nobs - k_params for models with additional
    # parameters, e.g. the dispersion parameter of NegativeBinomial, and for
    # rank deficient designs. The small sample correction then uses the
    # number of moment conditions, GH 9253
    endog, exog, group, time, groupidx = _make_panel_sample()
    mod = OLS(endog, exog)
    res = mod.fit()

    if cov_type == "white":
        cov_func, kwds = sw.cov_white_simple, {}
    else:
        cov_func, kwds = sw.cov_cluster, {"group": group}

    cov = cov_func(res, **kwds)

    mod.df_resid = res.df_resid + 5
    res_adj = mod.fit()
    assert res_adj.df_resid == res.df_resid + 5

    assert_allclose(cov_func(res_adj, **kwds), cov)


def test_correction_without_results_df_resid():
    # not all results instances define df_resid, e.g. the plain
    # LikelihoodModelResults that discrete models use internally when the
    # covariance is computed by fit(cov_type=...). The correction factor then
    # falls back to nobs - k_params, GH 9253
    from statsmodels.discrete.discrete_model import Poisson

    endog, exog, group, time, groupidx = _make_panel_sample()
    endog = np.round(np.abs(endog)).astype(int)

    res = Poisson(endog, exog).fit(
        disp=False, cov_type="cluster", cov_kwds={"groups": group}, use_t=False
    )

    assert_allclose(res.cov_params(), sw.cov_cluster(res._results, group))
