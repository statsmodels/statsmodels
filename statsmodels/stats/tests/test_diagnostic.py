"""Tests for Regression Diagnostics and Specification Tests

Created on Thu Feb 09 13:19:47 2012

Author: Josef Perktold
License: BSD-3

currently all tests are against R

"""

import json
from pathlib import Path

import numpy as np
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from statsmodels.datasets import macrodata, sunspots
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

cur_dir = Path(__file__).parent.resolve()


@pytest.fixture(scope="module")
def diagnostic_data():
    rs = np.random.RandomState(93674328)
    e = rs.standard_normal(500)
    x = rs.standard_normal((500, 3))
    y = x.sum(1) + e
    c = np.ones_like(y)
    data = pd.DataFrame(np.c_[y, c, x], columns=["y", "c", "x1", "x2", "x3"])
    return data


def compare_to_reference(sp, sp_dict, decimal=(12, 12)):
    assert_allclose(
        sp[0], sp_dict["statistic"], atol=10 ** (-decimal[0]), rtol=10 ** (-decimal[0])
    )
    assert_allclose(
        sp[1], sp_dict["pvalue"], atol=10 ** (-decimal[1]), rtol=10 ** (-decimal[0])
    )


def test_gq():
    d = macrodata.load().data
    realinv = d["realinv"]
    realgdp = d["realgdp"]
    realint = d["realint"]
    endog = realinv
    exog = add_constant(np.c_[realgdp, realint])
    het_gq_greater = dict(
        statistic=13.20512768685082,
        df1=99,
        df2=98,
        pvalue=1.246141976112324e-30,
        distr="f",
    )
    gq = smsdia.het_goldfeldquandt(endog, exog, split=0.5)
    compare_to_reference(gq, het_gq_greater, decimal=(12, 12))
    assert_equal(gq[-1], "increasing")


class TestDiagnosticG:

    @classmethod
    def setup_class(cls):
        d = macrodata.load_pandas().data
        gs_l_realinv = 400 * np.diff(np.log(d["realinv"].values))
        gs_l_realgdp = 400 * np.diff(np.log(d["realgdp"].values))
        lint = d["realint"][:-1].values
        tbilrate = d["tbilrate"][:-1].values
        endogg = gs_l_realinv
        exogg = add_constant(np.c_[gs_l_realgdp, lint])
        exogg2 = add_constant(np.c_[gs_l_realgdp, tbilrate])
        exogg3 = add_constant(np.c_[gs_l_realgdp])
        res_ols = OLS(endogg, exogg).fit()
        res_ols2 = OLS(endogg, exogg2).fit()
        res_ols3 = OLS(endogg, exogg3).fit()
        cls.res = res_ols
        cls.res2 = res_ols2
        cls.res3 = res_ols3
        cls.endog = cls.res.model.endog
        cls.exog = cls.res.model.exog

    def test_basic(self):
        params = np.array([-9.48167277465485, 4.3742216647032, -0.613996969478989])
        assert_almost_equal(self.res.params, params, decimal=12)

    def test_hac(self):
        res = self.res
        cov_hac_4 = np.array(
            [
                1.385551290884014,
                -0.3133096102522685,
                -0.0597207976835705,
                -0.3133096102522685,
                0.1081011690351306,
                0.000389440793564336,
                -0.0597207976835705,
                0.000389440793564339,
                0.0862118527405036,
            ]
        ).reshape((3, 3), order="F")
        cov_hac_10 = np.array(
            [
                1.257386180080192,
                -0.2871560199899846,
                -0.03958300024627573,
                -0.2871560199899845,
                0.1049107028987101,
                0.0003896205316866944,
                -0.03958300024627578,
                0.0003896205316866961,
                0.0985539340694839,
            ]
        ).reshape((3, 3), order="F")
        cov = sw.cov_hac_simple(res, nlags=4, use_correction=False)
        bse_hac = sw.se_cov(cov)
        assert_almost_equal(cov, cov_hac_4, decimal=12)
        assert_almost_equal(bse_hac, np.sqrt(np.diag(cov)), decimal=12)
        cov = sw.cov_hac_simple(res, nlags=10, use_correction=False)
        bse_hac = sw.se_cov(cov)
        assert_almost_equal(cov, cov_hac_10, decimal=12)
        assert_almost_equal(bse_hac, np.sqrt(np.diag(cov)), decimal=12)

    def test_het_goldfeldquandt(self):
        het_gq_greater = dict(
            statistic=0.5313259064778423,
            pvalue=0.9990217851193723,
            parameters=(98, 98),
            distr="f",
        )
        het_gq_less = dict(
            statistic=0.5313259064778423,
            pvalue=0.000978214880627621,
            parameters=(98, 98),
            distr="f",
        )
        het_gq_two_sided = dict(
            statistic=0.5313259064778423,
            pvalue=0.001956429761255241,
            parameters=(98, 98),
            distr="f",
        )
        het_gq_two_sided_01 = dict(
            statistic=0.5006976835928314,
            pvalue=0.001387126702579789,
            parameters=(88, 87),
            distr="f",
        )
        endogg, exogg = (self.endog, self.exog)
        gq = smsdia.het_goldfeldquandt(endogg, exogg, split=0.5)
        compare_to_reference(gq, het_gq_greater, decimal=(12, 12))
        assert_equal(gq[-1], "increasing")
        gq = smsdia.het_goldfeldquandt(
            endogg, exogg, split=0.5, alternative="decreasing"
        )
        compare_to_reference(gq, het_gq_less, decimal=(12, 12))
        assert_equal(gq[-1], "decreasing")
        gq = smsdia.het_goldfeldquandt(
            endogg, exogg, split=0.5, alternative="two-sided"
        )
        compare_to_reference(gq, het_gq_two_sided, decimal=(12, 12))
        assert_equal(gq[-1], "two-sided")
        gq = smsdia.het_goldfeldquandt(
            endogg, exogg, split=90, drop=21, alternative="two-sided"
        )
        compare_to_reference(gq, het_gq_two_sided_01, decimal=(12, 12))
        assert_equal(gq[-1], "two-sided")

    def test_het_breusch_pagan(self):
        res = self.res
        bptest = dict(
            statistic=0.709924388395087,
            pvalue=0.701199952134347,
            parameters=(2,),
            distr="f",
        )
        bp = smsdia.het_breuschpagan(res.resid, res.model.exog)
        compare_to_reference(bp, bptest, decimal=(12, 12))

    def test_het_breusch_pagan_1d_err(self):
        res = self.res
        x = np.asarray(res.model.exog)[:, -1]
        with pytest.raises(ValueError, match="The Breusch-Pagan"):
            smsdia.het_breuschpagan(res.resid, x)
        x = np.ones_like(x)
        with pytest.raises(ValueError, match="The Breusch-Pagan"):
            smsdia.het_breuschpagan(res.resid, x)
        x = np.asarray(res.model.exog).copy()
        x[:, 0] = 0
        with pytest.raises(ValueError, match="The Breusch-Pagan"):
            smsdia.het_breuschpagan(res.resid, x)

    def test_het_breusch_pagan_nonrobust(self):
        res = self.res
        bptest = dict(
            statistic=1.302014063483341,
            pvalue=0.5215203247110649,
            parameters=(2,),
            distr="f",
        )
        bp = smsdia.het_breuschpagan(res.resid, res.model.exog, robust=False)
        compare_to_reference(bp, bptest, decimal=(12, 12))

    def test_het_white(self):
        res = self.res
        hw = smsdia.het_white(res.resid, res.model.exog)
        hw_values = (
            33.50372289653844,
            2.988796059783026e-06,
            7.794510122843095,
            1.0354575277704231e-06,
        )
        assert_almost_equal(hw, hw_values)

    def test_het_white_no_interaction_terms(self):
        res = self.res
        hw = smsdia.het_white(res.resid, res.model.exog, interaction_terms=False)
        hw_values = (
            13.25091965953952,
            0.001326170478134868,
            6.985287047470471,
            0.001169716842511783,
        )
        assert_almost_equal(hw, hw_values)

    def test_het_white_error(self):
        res = self.res
        with pytest.raises(ValueError, match="White's heteroskedasticity"):
            smsdia.het_white(res.resid, res.model.exog[:, :1])

    def test_het_arch(self):
        archtest_4 = dict(
            statistic=3.43473400836259,
            pvalue=0.487871315392619,
            parameters=(4,),
            distr="chi2",
        )
        archtest_12 = dict(
            statistic=8.648320999014171,
            pvalue=0.732638635007718,
            parameters=(12,),
            distr="chi2",
        )
        at4 = smsdia.het_arch(self.res.resid, nlags=4)
        at12 = smsdia.het_arch(self.res.resid, nlags=12)
        compare_to_reference(at4[:2], archtest_4, decimal=(12, 13))
        compare_to_reference(at12[:2], archtest_12, decimal=(12, 13))

    def test_het_arch2(self):
        resid = self.res.resid
        res1 = smsdia.het_arch(resid, nlags=5, store=True)
        rs1 = res1[-1]
        res2 = smsdia.het_arch(resid, nlags=5, store=True)
        rs2 = res2[-1]
        assert_almost_equal(rs2.resols.params, rs1.resols.params, decimal=12)
        assert_almost_equal(res2[:4], res1[:4], decimal=12)
        res3 = smsdia.het_arch(resid, nlags=5)
        assert_almost_equal(res3[:4], res1[:4], decimal=12)

    def test_acorr_breusch_godfrey(self):
        res = self.res
        breuschgodfrey_f = dict(
            statistic=1.179280833676792,
            pvalue=0.321197487261203,
            parameters=(4, 195),
            distr="f",
        )
        breuschgodfrey_c = dict(
            statistic=4.771042651230007,
            pvalue=0.3116067133066697,
            parameters=(4,),
            distr="chi2",
        )
        bg = smsdia.acorr_breusch_godfrey(res, nlags=4)
        bg_r = [
            breuschgodfrey_c["statistic"],
            breuschgodfrey_c["pvalue"],
            breuschgodfrey_f["statistic"],
            breuschgodfrey_f["pvalue"],
        ]
        assert_almost_equal(bg, bg_r, decimal=11)
        bg2 = smsdia.acorr_breusch_godfrey(res, nlags=None)
        bg3 = smsdia.acorr_breusch_godfrey(res, nlags=10)
        assert_almost_equal(bg2, bg3, decimal=12)

    def test_acorr_breusch_godfrey_multidim(self):
        res = Bunch(resid=np.empty((100, 2)))
        with pytest.raises(ValueError, match="Model resid must be a 1d array"):
            smsdia.acorr_breusch_godfrey(res)

    def test_acorr_breusch_godfrey_exogs(self):
        data = sunspots.load_pandas().data["SUNACTIVITY"]
        res = ARIMA(data, order=(1, 0, 0), trend="n").fit()
        smsdia.acorr_breusch_godfrey(res, nlags=1)

    def test_acorr_ljung_box(self):
        res = self.res
        ljung_box_4 = dict(
            statistic=5.23587172795227,
            pvalue=0.263940335284713,
            parameters=(4,),
            distr="chi2",
        )
        ljung_box_bp_4 = dict(
            statistic=5.12462932741681,
            pvalue=0.2747471266820692,
            parameters=(4,),
            distr="chi2",
        )
        df = smsdia.acorr_ljungbox(res.resid, 4, boxpierce=True)
        compare_to_reference(
            [df.loc[4, "lb_stat"], df.loc[4, "lb_pvalue"]],
            ljung_box_4,
            decimal=(12, 12),
        )
        compare_to_reference(
            [df.loc[4, "bp_stat"], df.loc[4, "bp_pvalue"]],
            ljung_box_bp_4,
            decimal=(12, 12),
        )

    def test_acorr_ljung_box_big_default(self):
        res = self.res
        ljung_box_none = dict(
            statistic=51.03724531797195, pvalue=0.1133474492339, distr="chi2"
        )
        ljung_box_bp_none = dict(
            statistic=45.12238537034, pvalue=0.26638168491464, distr="chi2"
        )
        lags = min(40, res.resid.shape[0] // 2 - 2)
        df = smsdia.acorr_ljungbox(res.resid, boxpierce=True, lags=lags)
        idx = df.index.max()
        compare_to_reference(
            [df.loc[idx, "lb_stat"], df.loc[idx, "lb_pvalue"]],
            ljung_box_none,
            decimal=(12, 12),
        )
        compare_to_reference(
            [df.loc[idx, "bp_stat"], df.loc[idx, "bp_pvalue"]],
            ljung_box_bp_none,
            decimal=(12, 12),
        )

    def test_acorr_ljung_box_small_default(self):
        res = self.res
        ljung_box_small = dict(
            statistic=9.61503968281915,
            pvalue=0.72507000996945,
            parameters=(0,),
            distr="chi2",
        )
        ljung_box_bp_small = dict(
            statistic=7.41692150864936,
            pvalue=0.87940785887006,
            parameters=(0,),
            distr="chi2",
        )
        if isinstance(res.resid, np.ndarray):
            resid = res.resid[:30]
        else:
            resid = res.resid.iloc[:30]
        df = smsdia.acorr_ljungbox(resid, boxpierce=True, lags=13)
        idx = df.index.max()
        compare_to_reference(
            [df.loc[idx, "lb_stat"], df.loc[idx, "lb_pvalue"]],
            ljung_box_small,
            decimal=(12, 12),
        )
        compare_to_reference(
            [df.loc[idx, "bp_stat"], df.loc[idx, "bp_pvalue"]],
            ljung_box_bp_small,
            decimal=(12, 12),
        )

    def test_acorr_ljung_box_against_r(self):
        rs = np.random.RandomState(9876543)
        y1 = rs.standard_normal(100)
        e = rs.standard_normal(201)
        y2 = np.zeros_like(e)
        y2[0] = e[0]
        for i in range(1, 201):
            y2[i] = 0.5 * y2[i - 1] - 0.4 * e[i - 1] + e[i]
        y2 = y2[-100:]
        r_results_y1_lb = [
            [0.15685, 1, 0.6921],
            [5.4737, 5, 0.3608],
            [10.508, 10, 0.3971],
        ]
        r_results_y2_lb = [
            [2.8764, 1, 0.08989],
            [3.8104, 5, 0.577],
            [8.4779, 10, 0.5823],
        ]
        res_y1 = smsdia.acorr_ljungbox(y1, 10)
        res_y2 = smsdia.acorr_ljungbox(y2, 10)
        for i, loc in enumerate((1, 5, 10)):
            row = res_y1.loc[loc]
            assert_allclose(r_results_y1_lb[i][0], row.loc["lb_stat"], rtol=0.001)
            assert_allclose(r_results_y1_lb[i][2], row.loc["lb_pvalue"], rtol=0.001)
            row = res_y2.loc[loc]
            assert_allclose(r_results_y2_lb[i][0], row.loc["lb_stat"], rtol=0.001)
            assert_allclose(r_results_y2_lb[i][2], row.loc["lb_pvalue"], rtol=0.001)
        res = smsdia.acorr_ljungbox(y2, 10, boxpierce=True)
        assert_allclose(res.loc[10, "bp_stat"], 7.8935, rtol=0.001)
        assert_allclose(res.loc[10, "bp_pvalue"], 0.639, rtol=0.001)
        res = smsdia.acorr_ljungbox(y2, 10, boxpierce=True, model_df=1)
        assert_allclose(res.loc[10, "bp_pvalue"], 0.5449, rtol=0.001)

    def test_harvey_collier(self):
        harvey_collier = dict(
            statistic=0.494432160939874,
            pvalue=0.6215491310408242,
            parameters=198,
            distr="t",
        )
        hc = smsdia.linear_harvey_collier(self.res)
        compare_to_reference(hc, harvey_collier, decimal=(12, 12))
        hc_skip = smsdia.linear_harvey_collier(self.res, skip=20)
        assert not np.allclose(hc[0], hc_skip[0])

    def test_rainbow(self):
        raintest = dict(
            statistic=0.6809600116739604,
            pvalue=0.971832843583418,
            parameters=(101, 98),
            distr="f",
        )
        raintest_fraction_04 = dict(
            statistic=0.565551237772662,
            pvalue=0.997592305968473,
            parameters=(122, 77),
            distr="f",
        )
        rb = smsdia.linear_rainbow(self.res)
        compare_to_reference(rb, raintest, decimal=(12, 12))
        rb = smsdia.linear_rainbow(self.res, frac=0.4)
        compare_to_reference(rb, raintest_fraction_04, decimal=(12, 12))

    def test_compare_lr(self):
        res = self.res
        res3 = self.res3
        lrtest = dict(
            loglike1=-763.9752181602237,
            loglike2=-766.3091902020184,
            chi2value=4.66794408358942,
            pvalue=0.03073069384028677,
            df=(4, 3, 1),
        )
        lrt = res.compare_lr_test(res3)
        assert_almost_equal(lrt[0], lrtest["chi2value"], decimal=11)
        assert_almost_equal(lrt[1], lrtest["pvalue"], decimal=11)
        waldtest = dict(
            fvalue=4.65216373312492, pvalue=0.03221346195239025, df=(199, 200, 1)
        )
        wt = res.compare_f_test(res3)
        assert_almost_equal(wt[0], waldtest["fvalue"], decimal=11)
        assert_almost_equal(wt[1], waldtest["pvalue"], decimal=11)

    def test_compare_j(self):
        res = self.res
        res2 = self.res2
        jtest = [
            (
                "M1 + fitted(M2)",
                1.591505670785873,
                0.7384552861695823,
                2.15518217635237,
                0.03235457252531445,
                "*",
            ),
            (
                "M2 + fitted(M1)",
                1.305687653016899,
                0.4808385176653064,
                2.715438978051544,
                0.007203854534057954,
                "**",
            ),
        ]
        jt1 = smsdia.compare_j(res2, res)
        assert_almost_equal(jt1, jtest[0][3:5], decimal=12)
        jt2 = smsdia.compare_j(res, res2)
        assert_almost_equal(jt2, jtest[1][3:5], decimal=12)

    @pytest.mark.parametrize("comp", [smsdia.compare_cox, smsdia.compare_j])
    def test_compare_error(self, comp, diagnostic_data):
        data = diagnostic_data
        res1 = OLS(data.y, data[["c", "x1"]]).fit()
        res2 = OLS(data.x3, data[["c", "x2"]]).fit()
        with pytest.raises(ValueError, match="endogenous variables"):
            comp(res1, res2)

    @pytest.mark.parametrize("comp", [smsdia.compare_cox, smsdia.compare_j])
    def test_compare_nested(self, comp, diagnostic_data):
        data = diagnostic_data
        res1 = OLS(data.y, data[["c", "x1"]]).fit()
        res2 = OLS(data.y, data[["c", "x1", "x2"]]).fit()
        with pytest.raises(ValueError, match="The exog in results_x"):
            comp(res1, res2)
        with pytest.raises(ValueError, match="The exog in results_x"):
            comp(res2, res1)

    def test_compare_cox(self):
        res = self.res
        res2 = self.res2
        coxtest = [
            (
                "fitted(M1) ~ M2",
                -0.782030488930356,
                0.599696502782265,
                -1.304043770977755,
                0.1922186587840554,
                " ",
            ),
            (
                "fitted(M2) ~ M1",
                -2.248817107408537,
                0.392656854330139,
                -5.727181590258883,
                1.021128495098556e-08,
                "***",
            ),
        ]
        ct1 = smsdia.compare_cox(res, res2)
        assert_almost_equal(ct1, coxtest[0][3:5], decimal=12)
        ct2 = smsdia.compare_cox(res2, res)
        assert_almost_equal(ct2, coxtest[1][3:5], decimal=12)
        _, _, store = smsdia.compare_cox(res, res2, store=True)
        assert isinstance(store, smsdia.ResultsStore)

    def test_cusum_ols(self):
        cusum_ols = dict(
            statistic=1.055750610401214,
            pvalue=0.2149567397376543,
            parameters=(),
            distr="BB",
        )
        k_vars = 3
        cs_ols = smsdia.breaks_cusumolsresid(self.res.resid, ddof=k_vars)
        compare_to_reference(cs_ols, cusum_ols, decimal=(12, 12))

    def test_breaks_hansen(self):
        breaks_nyblom_hansen = dict(
            statistic=1.0300792740544484,
            pvalue=0.1136087530212015,
            parameters=(),
            distr="BB",
        )
        bh = smsdia.breaks_hansen(self.res)
        assert_almost_equal(bh[0], breaks_nyblom_hansen["statistic"], decimal=12)

    def test_recursive_residuals(self):
        reccumres_standardize = np.array(
            [
                -2.151,
                -3.748,
                -3.114,
                -3.096,
                -1.865,
                -2.23,
                -1.194,
                -3.5,
                -3.638,
                -4.447,
                -4.602,
                -4.631,
                -3.999,
                -4.83,
                -5.429,
                -5.435,
                -6.554,
                -8.093,
                -8.567,
                -7.532,
                -7.079,
                -8.468,
                -9.32,
                -12.256,
                -11.932,
                -11.454,
                -11.69,
                -11.318,
                -12.665,
                -12.842,
                -11.693,
                -10.803,
                -12.113,
                -12.109,
                -13.002,
                -11.897,
                -10.787,
                -10.159,
                -9.038,
                -9.007,
                -8.634,
                -7.552,
                -7.153,
                -6.447,
                -5.183,
                -3.794,
                -3.511,
                -3.979,
                -3.236,
                -3.793,
                -3.699,
                -5.056,
                -5.724,
                -4.888,
                -4.309,
                -3.688,
                -3.918,
                -3.735,
                -3.452,
                -2.086,
                -6.52,
                -7.959,
                -6.76,
                -6.855,
                -6.032,
                -4.405,
                -4.123,
                -4.075,
                -3.235,
                -3.115,
                -3.131,
                -2.986,
                -1.813,
                -4.824,
                -4.424,
                -4.796,
                -4.0,
                -3.39,
                -4.485,
                -4.669,
                -4.56,
                -3.834,
                -5.507,
                -3.792,
                -2.427,
                -1.756,
                -0.354,
                1.15,
                0.586,
                0.643,
                1.773,
                -0.83,
                -0.388,
                0.517,
                0.819,
                2.24,
                3.791,
                3.187,
                3.409,
                2.431,
                0.668,
                0.957,
                -0.928,
                0.327,
                -0.285,
                -0.625,
                -2.316,
                -1.986,
                -0.744,
                -1.396,
                -1.728,
                -0.646,
                -2.602,
                -2.741,
                -2.289,
                -2.897,
                -1.934,
                -2.532,
                -3.175,
                -2.806,
                -3.099,
                -2.658,
                -2.487,
                -2.515,
                -2.224,
                -2.416,
                -1.141,
                0.65,
                -0.947,
                0.725,
                0.439,
                0.885,
                2.419,
                2.642,
                2.745,
                3.506,
                4.491,
                5.377,
                4.624,
                5.523,
                6.488,
                6.097,
                5.39,
                6.299,
                6.656,
                6.735,
                8.151,
                7.26,
                7.846,
                8.771,
                8.4,
                8.717,
                9.916,
                9.008,
                8.91,
                8.294,
                8.982,
                8.54,
                8.395,
                7.782,
                7.794,
                8.142,
                8.362,
                8.4,
                7.85,
                7.643,
                8.228,
                6.408,
                7.218,
                7.699,
                7.895,
                8.725,
                8.938,
                8.781,
                8.35,
                9.136,
                9.056,
                10.365,
                10.495,
                10.704,
                10.784,
                10.275,
                10.389,
                11.586,
                11.033,
                11.335,
                11.661,
                10.522,
                10.392,
                10.521,
                10.126,
                9.428,
                9.734,
                8.954,
                9.949,
                10.595,
                8.016,
                6.636,
                6.975,
            ]
        )
        rr = smsdia.recursive_olsresiduals(self.res, skip=3, alpha=0.95)
        assert_equal(np.round(rr[5][1:], 3), reccumres_standardize)
        assert_almost_equal(rr[3][4:], np.diff(reccumres_standardize), 3)
        assert_almost_equal(rr[4][3:].std(ddof=1), 10.7242, decimal=4)
        order = np.arange(self.res.model.endog.shape[0])
        rr_order = smsdia.recursive_olsresiduals(
            self.res, skip=3, alpha=0.95, order_by=order
        )
        assert_allclose(rr[0], rr_order[0], atol=1e-06)
        ub0 = np.array(
            [13.37318571, 13.50758959, 13.64199346, 13.77639734, 13.91080121]
        )
        ub1 = np.array([39.44753774, 39.58194162, 39.7163455, 39.85074937, 39.98515325])
        lb, ub = rr[6]
        assert_almost_equal(ub[:5], ub0, decimal=7)
        assert_almost_equal(lb[:5], -ub0, decimal=7)
        assert_almost_equal(ub[-5:], ub1, decimal=7)
        assert_almost_equal(lb[-5:], -ub1, decimal=7)
        endog = self.res.model.endog
        exog = self.res.model.exog
        params = []
        ypred = []
        for i in range(3, 10):
            resi = OLS(endog[:i], exog[:i]).fit()
            ypred.append(resi.model.predict(resi.params, exog[i]))
            params.append(resi.params)
        assert_almost_equal(rr[2][3:10], ypred, decimal=12)
        assert_almost_equal(rr[0][3:10], endog[3:10] - ypred, decimal=12)
        assert_almost_equal(rr[1][2:9], params, decimal=12)

    def test_normality(self):
        res = self.res
        lilliefors1 = dict(
            statistic=0.0723390908786589,
            pvalue=0.01204113540102896,
            parameters=(),
            distr="-",
        )
        lilliefors2 = dict(
            statistic=0.301311621898024,
            pvalue=1.004305736618051e-51,
            parameters=(),
            distr="-",
        )
        lilliefors3 = dict(
            statistic=0.1333956004203103, pvalue=0.455683, parameters=(), distr="-"
        )
        lf1 = smsdia.lilliefors(res.resid, pvalmethod="approx")
        lf2 = smsdia.lilliefors(res.resid**2, pvalmethod="approx")
        if isinstance(res.resid, np.ndarray):
            resid = res.resid[:20]
        else:
            resid = res.resid.iloc[:20]
        lf3 = smsdia.lilliefors(resid, pvalmethod="approx")
        compare_to_reference(lf1, lilliefors1, decimal=(12, 12))
        compare_to_reference(lf2, lilliefors2, decimal=(12, 12))
        assert_allclose(lf2[1], lilliefors2["pvalue"], rtol=1e-10)
        compare_to_reference(lf3, lilliefors3, decimal=(12, 1))
        adr1 = dict(
            statistic=1.602209621518313,
            pvalue=0.0003937979149362316,
            parameters=(),
            distr="-",
        )
        adr3 = dict(
            statistic=0.3017073732210775,
            pvalue=0.5443499281265933,
            parameters=(),
            distr="-",
        )
        ad1 = smsdia.normal_ad(res.resid)
        compare_to_reference(ad1, adr1, decimal=(11, 13))
        ad2 = smsdia.normal_ad(res.resid**2)
        assert_(np.isinf(ad2[0]))
        ad3 = smsdia.normal_ad(resid)
        compare_to_reference(ad3, adr3, decimal=(11, 12))

    def test_influence(self):
        res = self.res
        infl = oi.OLSInfluence(res)
        path = Path(cur_dir).joinpath("results", "influence_lsdiag_R.json")
        with Path(path).open(encoding="utf-8") as fp:
            lsdiag = json.load(fp)
        assert_almost_equal(
            np.array(lsdiag["cov.scaled"]).reshape(3, 3), res.cov_params(), decimal=12
        )
        assert_almost_equal(
            np.array(lsdiag["cov.unscaled"]).reshape(3, 3),
            res.normalized_cov_params,
            decimal=12,
        )
        c0, c1 = infl.cooks_distance
        assert_almost_equal(c0, lsdiag["cooks"], decimal=12)
        assert_almost_equal(infl.hat_matrix_diag, lsdiag["hat"], decimal=12)
        assert_almost_equal(
            infl.resid_studentized_internal, lsdiag["std.res"], decimal=12
        )
        dffits, dffth = infl.dffits
        assert_almost_equal(dffits, lsdiag["dfits"], decimal=12)
        assert_almost_equal(
            infl.resid_studentized_external, lsdiag["stud.res"], decimal=12
        )
        fn = Path(cur_dir).joinpath("results/influence_measures_R.csv")
        infl_r = pd.read_csv(fn, index_col=0)
        infl_r2 = np.asarray(infl_r)
        assert_almost_equal(infl.dfbetas, infl_r2[:, :3], decimal=12)
        assert_almost_equal(infl.cov_ratio, infl_r2[:, 4], decimal=12)
        assert_almost_equal(dffits, infl_r2[:, 3], decimal=12)
        assert_almost_equal(c0, infl_r2[:, 5], decimal=12)
        assert_almost_equal(infl.hat_matrix_diag, infl_r2[:, 6], decimal=12)


class TestDiagnosticGPandas(TestDiagnosticG):

    @classmethod
    def setup_class(cls):
        d = macrodata.load_pandas().data
        d["gs_l_realinv"] = 400 * np.log(d["realinv"]).diff()
        d["gs_l_realgdp"] = 400 * np.log(d["realgdp"]).diff()
        d["lint"] = d["realint"].shift(1)
        d["tbilrate"] = d["tbilrate"].shift(1)
        d = d.dropna()
        cls.d = d
        endogg = d["gs_l_realinv"]
        exogg = add_constant(d[["gs_l_realgdp", "lint"]])
        exogg2 = add_constant(d[["gs_l_realgdp", "tbilrate"]])
        exogg3 = add_constant(d[["gs_l_realgdp"]])
        res_ols = OLS(endogg, exogg).fit()
        res_ols2 = OLS(endogg, exogg2).fit()
        res_ols3 = OLS(endogg, exogg3).fit()
        cls.res = res_ols
        cls.res2 = res_ols2
        cls.res3 = res_ols3
        cls.endog = cls.res.model.endog
        cls.exog = cls.res.model.exog


def test_spec_white():
    resdir = Path(cur_dir).joinpath("results")
    wsfiles = ["wspec1.csv", "wspec2.csv", "wspec3.csv", "wspec4.csv"]
    for file in wsfiles:
        mdlfile = Path(resdir).joinpath(file)
        mdl = np.asarray(pd.read_csv(mdlfile))
        lastcol = mdl.shape[1] - 1
        dv = mdl[:, lastcol]
        design = np.concatenate(
            (np.ones((mdl.shape[0], 1)), np.delete(mdl, lastcol, 1)), axis=1
        )
        resids = dv - np.dot(design, np.linalg.lstsq(design, dv, rcond=-1)[0])
        wsres = smsdia.spec_white(resids, design)
        if file == "wspec1.csv":
            assert_almost_equal(wsres, [3.251, 0.661, 5], decimal=3)
        elif file == "wspec2.csv":
            assert_almost_equal(wsres, [6.07, 0.733, 9], decimal=3)
        elif file == "wspec3.csv":
            assert_almost_equal(wsres, [6.767, 0.454, 7], decimal=3)
        else:
            assert_almost_equal(wsres, [8.462, 0.671, 11], decimal=3)


def test_spec_white_error():
    rs = np.random.RandomState(38342003)
    with pytest.raises(ValueError, match="White's specification test "):
        smsdia.spec_white(rs.standard_normal(100), rs.standard_normal((100, 1)))
    with pytest.raises(ValueError, match="White's specification test "):
        smsdia.spec_white(rs.standard_normal(100), rs.standard_normal((100, 2)))


def test_linear_lm_direct():
    rs = np.random.RandomState(38342002)
    endog = rs.standard_normal(500)
    exog = add_constant(rs.standard_normal((500, 3)))
    res = OLS(endog, exog).fit()
    lm_res = smsdia.linear_lm(res.resid, exog)
    aug = np.hstack([exog, exog[:, 1:] ** 2])
    res_aug = OLS(res.resid, aug).fit()
    stat = res_aug.rsquared * aug.shape[0]
    assert_allclose(lm_res[0], stat)


def grangertest():
    dict(fvalue=1.589672703015157, pvalue=0.178717196987075, df=(198, 193))


@pytest.mark.smoke
def test_outlier_influence_funcs():
    rs = np.random.RandomState(38342002)
    x = add_constant(rs.randn(10, 2))
    y = x.sum(1) + rs.randn(10)
    res = OLS(y, x).fit()
    out_05 = oi.summary_table(res)
    out_01 = oi.summary_table(res, alpha=0.01)
    assert_(np.all(out_01[1][:, 6] <= out_05[1][:, 6]))
    assert_(np.all(out_01[1][:, 7] >= out_05[1][:, 7]))
    res2 = OLS(y, x[:, 0]).fit()
    oi.summary_table(res2, alpha=0.05)
    infl = res2.get_influence()
    infl.summary_table()


def test_influence_wrapped():
    from pandas import DataFrame

    d = macrodata.load_pandas().data
    gs_l_realinv = 400 * np.log(d["realinv"]).diff().dropna()
    gs_l_realgdp = 400 * np.log(d["realgdp"]).diff().dropna()
    lint = d["realint"][:-1]
    gs_l_realgdp.index = lint.index
    gs_l_realinv.index = lint.index
    data = dict(const=np.ones_like(lint), lint=lint, lrealgdp=gs_l_realgdp)
    exog = DataFrame(data, columns=["const", "lrealgdp", "lint"])
    res = OLS(gs_l_realinv, exog).fit()
    infl = oi.OLSInfluence(res)
    df = infl.summary_frame()
    assert_(isinstance(df, DataFrame))
    path = Path(cur_dir).joinpath("results", "influence_lsdiag_R.json")
    with Path(path).open(encoding="utf-8") as fp:
        lsdiag = json.load(fp)
    c0, c1 = infl.cooks_distance
    assert_almost_equal(c0, lsdiag["cooks"], 12)
    assert_almost_equal(infl.hat_matrix_diag, lsdiag["hat"], 12)
    assert_almost_equal(infl.resid_studentized_internal, lsdiag["std.res"], 12)
    dffits, dffth = infl.dffits
    assert_almost_equal(dffits, lsdiag["dfits"], 12)
    assert_almost_equal(infl.resid_studentized_external, lsdiag["stud.res"], 12)
    fn = Path(cur_dir).joinpath("results/influence_measures_R.csv")
    infl_r = pd.read_csv(fn, index_col=0)
    infl_r2 = np.asarray(infl_r)
    assert_almost_equal(infl.dfbetas, infl_r2[:, :3], decimal=12)
    assert_almost_equal(infl.cov_ratio, infl_r2[:, 4], decimal=12)


def test_influence_dtype():
    y = np.ones(20)
    rs = np.random.RandomState(123)
    x = rs.randn(20, 3)
    res1 = OLS(y, x).fit()
    res2 = OLS(y * 1.0, x).fit()
    cr1 = res1.get_influence().cov_ratio
    cr2 = res2.get_influence().cov_ratio
    assert_allclose(cr1, cr2, rtol=1e-14)
    cr3 = np.array(
        [
            1.22239215,
            1.31551021,
            1.52671069,
            1.05003921,
            0.89099323,
            1.57405066,
            1.03230092,
            0.95844196,
            1.15531836,
            1.21963623,
            0.87699564,
            1.16707748,
            1.10481391,
            0.98839447,
            1.08999334,
            1.35680102,
            1.46227715,
            1.45966708,
            1.13659521,
            1.22799038,
        ]
    )
    assert_almost_equal(cr1, cr3, decimal=8)


def get_duncan_data():
    labels = [
        "accountant",
        "pilot",
        "architect",
        "author",
        "chemist",
        "minister",
        "professor",
        "dentist",
        "reporter",
        "engineer",
        "undertaker",
        "lawyer",
        "physician",
        "welfare.worker",
        "teacher",
        "conductor",
        "contractor",
        "factory.owner",
        "store.manager",
        "banker",
        "bookkeeper",
        "mail.carrier",
        "insurance.agent",
        "store.clerk",
        "carpenter",
        "electrician",
        "RR.engineer",
        "machinist",
        "auto.repairman",
        "plumber",
        "gas.stn.attendant",
        "coal.miner",
        "streetcar.motorman",
        "taxi.driver",
        "truck.driver",
        "machine.operator",
        "barber",
        "bartender",
        "shoe.shiner",
        "cook",
        "soda.clerk",
        "watchman",
        "janitor",
        "policeman",
        "waiter",
    ]
    exog = [
        [1.0, 62.0, 86.0],
        [1.0, 72.0, 76.0],
        [1.0, 75.0, 92.0],
        [1.0, 55.0, 90.0],
        [1.0, 64.0, 86.0],
        [1.0, 21.0, 84.0],
        [1.0, 64.0, 93.0],
        [1.0, 80.0, 100.0],
        [1.0, 67.0, 87.0],
        [1.0, 72.0, 86.0],
        [1.0, 42.0, 74.0],
        [1.0, 76.0, 98.0],
        [1.0, 76.0, 97.0],
        [1.0, 41.0, 84.0],
        [1.0, 48.0, 91.0],
        [1.0, 76.0, 34.0],
        [1.0, 53.0, 45.0],
        [1.0, 60.0, 56.0],
        [1.0, 42.0, 44.0],
        [1.0, 78.0, 82.0],
        [1.0, 29.0, 72.0],
        [1.0, 48.0, 55.0],
        [1.0, 55.0, 71.0],
        [1.0, 29.0, 50.0],
        [1.0, 21.0, 23.0],
        [1.0, 47.0, 39.0],
        [1.0, 81.0, 28.0],
        [1.0, 36.0, 32.0],
        [1.0, 22.0, 22.0],
        [1.0, 44.0, 25.0],
        [1.0, 15.0, 29.0],
        [1.0, 7.0, 7.0],
        [1.0, 42.0, 26.0],
        [1.0, 9.0, 19.0],
        [1.0, 21.0, 15.0],
        [1.0, 21.0, 20.0],
        [1.0, 16.0, 26.0],
        [1.0, 16.0, 28.0],
        [1.0, 9.0, 17.0],
        [1.0, 14.0, 22.0],
        [1.0, 12.0, 30.0],
        [1.0, 17.0, 25.0],
        [1.0, 7.0, 20.0],
        [1.0, 34.0, 47.0],
        [1.0, 8.0, 32.0],
    ]
    endog = [
        82.0,
        83.0,
        90.0,
        76.0,
        90.0,
        87.0,
        93.0,
        90.0,
        52.0,
        88.0,
        57.0,
        89.0,
        97.0,
        59.0,
        73.0,
        38.0,
        76.0,
        81.0,
        45.0,
        92.0,
        39.0,
        34.0,
        41.0,
        16.0,
        33.0,
        53.0,
        67.0,
        57.0,
        26.0,
        29.0,
        10.0,
        15.0,
        19.0,
        10.0,
        13.0,
        24.0,
        20.0,
        7.0,
        3.0,
        16.0,
        6.0,
        11.0,
        8.0,
        41.0,
        10.0,
    ]
    return (endog, exog, labels)


def test_outlier_test():
    endog, exog, labels = get_duncan_data()
    ndarray_mod = OLS(endog, exog).fit()
    rstudent = [
        3.1345185839,
        -2.397022399,
        2.0438046359,
        -1.9309187757,
        1.8870465798,
        -1.76049053,
        -1.7040324156,
        1.6024285876,
        -1.4332485037,
        -1.1044851583,
        1.0688582315,
        1.018527184,
        -0.9024219332,
        -0.9023876471,
        -0.8830953936,
        0.8265782334,
        0.8089220547,
        0.7682770197,
        0.7319491074,
        -0.6665962829,
        0.5227352794,
        -0.5135016547,
        0.5083881518,
        0.4999224372,
        -0.4980818221,
        -0.4759717075,
        -0.429356582,
        -0.4114056499,
        -0.3779540862,
        0.355687403,
        0.3409200462,
        0.3062248646,
        0.3038999429,
        -0.3030815773,
        -0.1873387893,
        0.1738050251,
        0.1424246593,
        -0.1292266025,
        0.1272066463,
        -0.0798902878,
        0.0788467222,
        0.0722556991,
        0.050509828,
        0.0233215136,
        0.0007112055,
    ]
    unadj_p = [
        0.003177202,
        0.021170298,
        0.047432955,
        0.060427645,
        0.06624812,
        0.085783008,
        0.095943909,
        0.116738318,
        0.15936889,
        0.275822623,
        0.291386358,
        0.314400295,
        0.372104049,
        0.37212204,
        0.382333561,
        0.413260793,
        0.423229432,
        0.44672537,
        0.468363101,
        0.508764039,
        0.60397199,
        0.610356737,
        0.613905871,
        0.619802317,
        0.621087703,
        0.636621083,
        0.669911674,
        0.682917818,
        0.707414459,
        0.723898263,
        0.734904667,
        0.760983108,
        0.762741124,
        0.763360242,
        0.852319039,
        0.862874018,
        0.887442197,
        0.897810225,
        0.899398691,
        0.936713197,
        0.937538115,
        0.942749758,
        0.959961394,
        0.981506948,
        0.999435989,
    ]
    bonf_p = [
        0.1429741,
        0.9526634,
        2.134483,
        2.719244,
        2.9811654,
        3.8602354,
        4.3174759,
        5.2532243,
        7.1716001,
        12.412018,
        13.1123861,
        14.1480133,
        16.7446822,
        16.7454918,
        17.2050103,
        18.5967357,
        19.0453245,
        20.1026416,
        21.0763395,
        22.8943818,
        27.1787396,
        27.4660532,
        27.6257642,
        27.8911043,
        27.9489466,
        28.6479487,
        30.1460253,
        30.7313018,
        31.8336506,
        32.5754218,
        33.07071,
        34.2442399,
        34.3233506,
        34.3512109,
        38.3543568,
        38.8293308,
        39.9348989,
        40.4014601,
        40.4729411,
        42.1520939,
        42.1892152,
        42.4237391,
        43.1982627,
        44.1678127,
        44.9746195,
    ]
    bonf_p = np.array(bonf_p)
    bonf_p[bonf_p > 1] = 1
    sorted_labels = [
        "minister",
        "reporter",
        "contractor",
        "insurance.agent",
        "machinist",
        "store.clerk",
        "conductor",
        "factory.owner",
        "mail.carrier",
        "streetcar.motorman",
        "carpenter",
        "coal.miner",
        "bartender",
        "bookkeeper",
        "soda.clerk",
        "chemist",
        "RR.engineer",
        "professor",
        "electrician",
        "gas.stn.attendant",
        "auto.repairman",
        "watchman",
        "banker",
        "machine.operator",
        "dentist",
        "waiter",
        "shoe.shiner",
        "welfare.worker",
        "plumber",
        "physician",
        "pilot",
        "engineer",
        "accountant",
        "lawyer",
        "undertaker",
        "barber",
        "store.manager",
        "truck.driver",
        "cook",
        "janitor",
        "policeman",
        "architect",
        "teacher",
        "taxi.driver",
        "author",
    ]
    res2 = np.c_[rstudent, unadj_p, bonf_p]
    res = oi.outlier_test(ndarray_mod, method="b", labels=labels, order=True)
    np.testing.assert_almost_equal(res.values, res2, 7)
    np.testing.assert_equal(res.index.tolist(), sorted_labels)
    data = pd.DataFrame(
        np.column_stack((endog, exog)),
        columns="y const var1 var2".split(),
        index=labels,
    )
    res_pd = OLS.from_formula("y ~ const + var1 + var2 - 0", data).fit()
    res_outl2 = oi.outlier_test(res_pd, method="b", order=True)
    assert_almost_equal(res_outl2.values, res2, 7)
    assert_equal(res_outl2.index.tolist(), sorted_labels)
    res_outl1 = res_pd.outlier_test(method="b")
    res_outl1 = res_outl1.sort_values(["unadj_p"], ascending=True)
    assert_almost_equal(res_outl1.values, res2, 7)
    assert_equal(res_outl1.index.tolist(), sorted_labels)
    assert_array_equal(res_outl2.index, res_outl1.index)
    res_outl3 = res_pd.outlier_test(method="b", order=True)
    assert_equal(res_outl3.index.tolist(), sorted_labels)
    res_outl4 = res_pd.outlier_test(method="b", order=True, cutoff=0.15)
    assert_equal(res_outl4.index.tolist(), sorted_labels[:1])


def test_ljungbox_dof_adj():
    data = sunspots.load_pandas().data["SUNACTIVITY"]
    res = AutoReg(data, 4).fit()
    resid = res.resid
    res1 = smsdia.acorr_ljungbox(resid, lags=10)
    res2 = smsdia.acorr_ljungbox(resid, lags=10, model_df=4)
    assert_allclose(res1.iloc[:, 0], res2.iloc[:, 0])
    assert np.all(np.isnan(res2.iloc[:4, 1]))
    assert np.all(res2.iloc[4:, 1] <= res1.iloc[4:, 1])


def test_ljungbox_auto_lag_selection():
    data = sunspots.load_pandas().data["SUNACTIVITY"]
    res = AutoReg(data, 4).fit()
    resid = res.resid
    res1 = smsdia.acorr_ljungbox(resid, auto_lag=True)
    res2 = smsdia.acorr_ljungbox(resid, model_df=4, auto_lag=True)
    assert_allclose(res1.iloc[:, 0], res2.iloc[:, 0])
    assert res1.shape[0] >= 1
    assert res2.shape[0] >= 1
    assert np.all(np.isnan(res2.iloc[:4, 1]))
    assert np.all(res2.iloc[4:, 1] <= res1.iloc[4:, 1])


def test_ljungbox_auto_lag_whitenoise():
    rs = np.random.RandomState(38342003)
    data = rs.randn(1000)
    res = smsdia.acorr_ljungbox(data, auto_lag=True)
    assert res.shape[0] >= 1


def test_ljungbox_errors_warnings():
    data = sunspots.load_pandas().data["SUNACTIVITY"]
    with pytest.raises(ValueError, match="model_df must"):
        smsdia.acorr_ljungbox(data, model_df=-1)
    with pytest.raises(ValueError, match="period must"):
        smsdia.acorr_ljungbox(data, model_df=-1, period=1)
    with pytest.raises(ValueError, match="period must"):
        smsdia.acorr_ljungbox(data, model_df=-1, period=-2)
    smsdia.acorr_ljungbox(data)
    ret = smsdia.acorr_ljungbox(data, lags=10)
    assert isinstance(ret, pd.DataFrame)


def test_ljungbox_period():
    data = sunspots.load_pandas().data["SUNACTIVITY"]
    ar_res = AutoReg(data, 4).fit()
    res = smsdia.acorr_ljungbox(ar_res.resid, period=13)
    res2 = smsdia.acorr_ljungbox(ar_res.resid, lags=26)
    assert_frame_equal(res, res2)


@pytest.mark.parametrize("cov_type", ["nonrobust", "HC0"])
def test_encompasing_direct(cov_type):
    rs = np.random.RandomState(38342002)
    x = rs.standard_normal((500, 2))
    e = rs.standard_normal((500, 1))
    x_extra = rs.standard_normal((500, 2))
    z_extra = rs.standard_normal((500, 3))
    y = x @ np.ones((2, 1)) + e
    x1 = np.hstack([x[:, :1], x_extra])
    z1 = np.hstack([x, z_extra])
    res1 = OLS(y, x1).fit()
    res2 = OLS(y, z1).fit()
    df = smsdia.compare_encompassing(res1, res2, cov_type=cov_type)
    direct1 = OLS(y, np.hstack([x1, x[:, 1:], z_extra])).fit(cov_type=cov_type)
    r1 = np.zeros((4, 3 + 1 + 3))
    r1[:, -4:] = np.eye(4)
    direct_test_1 = direct1.wald_test(r1, use_f=True, scalar=True)
    expected = (
        float(np.squeeze(direct_test_1.statistic)),
        float(np.squeeze(direct_test_1.pvalue)),
        int(direct_test_1.df_num),
        int(direct_test_1.df_denom),
    )
    assert_allclose(np.asarray(df.loc["x"]), expected, atol=1e-08)
    direct2 = OLS(y, np.hstack([z1, x_extra])).fit(cov_type=cov_type)
    r2 = np.zeros((2, 2 + 3 + 2))
    r2[:, -2:] = np.eye(2)
    direct_test_2 = direct2.wald_test(r2, use_f=True, scalar=True)
    expected = (
        float(np.squeeze(direct_test_2.statistic)),
        float(np.squeeze(direct_test_2.pvalue)),
        int(direct_test_2.df_num),
        int(direct_test_2.df_denom),
    )
    assert_allclose(np.asarray(df.loc["z"]), expected, atol=1e-08)


def test_encompasing_error():
    rs = np.random.RandomState(38342001)
    x = rs.standard_normal((500, 2))
    e = rs.standard_normal((500, 1))
    z_extra = rs.standard_normal((500, 3))
    y = x @ np.ones((2, 1)) + e
    z = np.hstack([x, z_extra])
    res1 = OLS(y, x).fit()
    res2 = OLS(y, z).fit()
    with pytest.raises(ValueError, match="The exog in results_x"):
        smsdia.compare_encompassing(res1, res2)
    with pytest.raises(TypeError, match="results_z must come from a linear"):
        smsdia.compare_encompassing(res1, 2)
    with pytest.raises(TypeError, match="results_x must come from a linear"):
        smsdia.compare_encompassing(4, 2)


@pytest.mark.smoke
@pytest.mark.parametrize("power", [2, 3])
@pytest.mark.parametrize("test_type", ["fitted", "exog", "princomp"])
@pytest.mark.parametrize("use_f", [True, False])
@pytest.mark.parametrize(
    "cov", [dict(cov_type="nonrobust", cov_kwds={}), dict(cov_type="HC0", cov_kwds={})]
)
def test_reset_smoke(power, test_type, use_f, cov):
    rs = np.random.RandomState(32320967 + power + int(use_f))
    x = add_constant(rs.standard_normal((1000, 3)))
    e = rs.standard_normal((1000, 1))
    x = np.hstack([x, x[:, 1:] ** 2])
    y = x @ np.ones((7, 1)) + e
    res = OLS(y, x[:, :4]).fit()
    smsdia.linear_reset(res, power=power, test_type=test_type, use_f=use_f, **cov)


@pytest.mark.smoke
@pytest.mark.parametrize("store", [True, False])
@pytest.mark.parametrize("ddof", [0, 2])
@pytest.mark.parametrize(
    "cov", [dict(cov_type="nonrobust", cov_kwds={}), dict(cov_type="HC0", cov_kwds={})]
)
def test_acorr_lm_smoke(store, ddof, cov):
    rs = np.random.RandomState(38342099)
    e = rs.standard_normal(250)
    smsdia.acorr_lm(e, nlags=6, store=store, ddof=ddof, **cov)
    smsdia.acorr_lm(e, nlags=None, store=store, period=12, ddof=ddof, **cov)


def test_acorr_lm_smoke_no_autolag():
    rs = np.random.RandomState(38342098)
    e = rs.standard_normal(250)
    smsdia.acorr_lm(e, nlags=6, store=False, ddof=0)


RS = np.random.RandomState(38342431)
RANDOM_ARRAY = RS.choice(500, size=500, replace=False)


@pytest.mark.parametrize("frac", [0.25, 0.5, 0.75])
@pytest.mark.parametrize(
    "order_by", [None, np.arange(500), RANDOM_ARRAY, "x0", ["x0", "x2"]]
)
def test_rainbow_smoke_order_by(frac, order_by):
    rs = np.random.RandomState(38342097)
    e = pd.DataFrame(rs.standard_normal((500, 1)))
    x = pd.DataFrame(rs.standard_normal((500, 3)), columns=[f"x{i}" for i in range(3)])
    y = x @ np.ones((3, 1)) + e
    res = OLS(y, x).fit()
    smsdia.linear_rainbow(res, frac=frac, order_by=order_by)


@pytest.mark.parametrize("center", [None, 0.33, 300])
def test_rainbow_smoke_centered(center):
    rs = np.random.RandomState(38342096)
    e = pd.DataFrame(rs.standard_normal((500, 1)))
    x = pd.DataFrame(rs.standard_normal((500, 3)), columns=[f"x{i}" for i in range(3)])
    y = x @ np.ones((3, 1)) + e
    res = OLS(y, x).fit()
    smsdia.linear_rainbow(res, use_distance=True, center=center)


def test_rainbow_exception():
    rs = np.random.RandomState(38342095)
    e = pd.DataFrame(rs.standard_normal((500, 1)))
    x = pd.DataFrame(rs.standard_normal((500, 3)), columns=[f"x{i}" for i in range(3)])
    y = x @ np.ones((3, 1)) + e
    res = OLS(y, x).fit()
    with pytest.raises(TypeError, match="order_by must contain"):
        smsdia.linear_rainbow(res, order_by="x5")
    res = OLS(np.asarray(y), np.asarray(x)).fit()
    with pytest.raises(TypeError, match="order_by must contain"):
        smsdia.linear_rainbow(res, order_by=("x0",))


def test_small_skip():
    rs = np.random.RandomState(38342094)
    y = rs.standard_normal(10)
    x = rs.standard_normal((10, 3))
    x[:3] = x[:1]
    with pytest.raises(ValueError, match="The initial regressor matrix,"):
        smsdia.recursive_olsresiduals(OLS(y, x).fit())


@pytest.mark.smoke
def test_diagnostics_pandas():
    n = 100
    rs = np.random.RandomState(38342093)
    df = pd.DataFrame({"y": rs.rand(n), "x": rs.rand(n), "z": rs.rand(n)})
    y, x = (df["y"], add_constant(df["x"]))
    res = OLS(df["y"], add_constant(df[["x"]])).fit()
    res_large = OLS(df["y"], add_constant(df[["x", "z"]])).fit()
    res_other = OLS(df["y"], add_constant(df[["z"]])).fit()
    smsdia.linear_reset(res_large)
    smsdia.linear_reset(res_large, test_type="fitted")
    smsdia.linear_reset(res_large, test_type="exog")
    smsdia.linear_reset(res_large, test_type="princomp")
    smsdia.het_goldfeldquandt(y, x)
    smsdia.het_breuschpagan(res.resid, x)
    smsdia.het_white(res.resid, x)
    smsdia.het_arch(res.resid)
    smsdia.acorr_breusch_godfrey(res)
    smsdia.acorr_ljungbox(y)
    smsdia.linear_rainbow(res)
    smsdia.linear_lm(res.resid, x)
    smsdia.linear_harvey_collier(res)
    smsdia.acorr_lm(res.resid)
    smsdia.breaks_cusumolsresid(res.resid)
    smsdia.breaks_hansen(res)
    smsdia.compare_cox(res, res_other)
    smsdia.compare_encompassing(res, res_other)
    smsdia.compare_j(res, res_other)
    smsdia.recursive_olsresiduals(res)
    smsdia.recursive_olsresiduals(res, order_by=np.arange(y.shape[0] - 1, 0 - 1, -1))
    smsdia.spec_white(res.resid, x)


def test_deprecated_argument():
    rs = np.random.RandomState(38342092)
    x = rs.randn(100)
    y = 2 * x + rs.randn(100)
    result = OLS(y, add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    with pytest.warns(FutureWarning, match="the "):
        smsdia.linear_reset(
            result,
            power=2,
            test_type="fitted",
            cov_type="HAC",
            cov_kwargs={"maxlags": 2},
        )


def test_diagnostics_hac():
    rs = np.random.RandomState(38342091)
    x = rs.randn(100)
    y = 2 * x + rs.randn(100)
    result = OLS(y, add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    reset_test = smsdia.linear_reset(
        result, power=2, test_type="fitted", cov_type="HAC", cov_kwds={"maxlags": 2}
    )
    assert reset_test.statistic > 0
    assert 0 <= reset_test.pvalue <= 1
