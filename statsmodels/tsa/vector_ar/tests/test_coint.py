"""Test Johansen's Cointegration test against jplv, Spatial Econometrics Toolbox

Created on Thu Aug 30 21:51:08 2012
Author: Josef Perktold

"""
from statsmodels.compat.numpy import ComplexWarning

from pathlib import Path
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from scipy import linalg, stats

from statsmodels.datasets import danish_data, macrodata
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_coint_rank

current_path = Path(__file__).resolve().parent
dta_path = Path(current_path).joinpath("Matlab_results", "test_coint.csv")
dta = pd.read_csv(dta_path, header=None, delimiter=r"\s+").values


class CheckCointJoh:

    def test_basic(self):
        assert_equal(self.res.ind, np.arange(len(self.res.ind), dtype=int))
        assert_equal(self.res.r0t.shape, (self.nobs_r, 8))

    def test_table_trace(self):
        table1 = np.column_stack((self.res.lr1, self.res.cvt))
        assert_almost_equal(table1, self.res1_m.reshape(table1.shape, order="F"))

    def test_table_maxeval(self):
        table2 = np.column_stack((self.res.lr2, self.res.cvm))
        assert_almost_equal(table2, self.res2_m.reshape(table2.shape, order="F"))

    def test_normalization(self):
        # GH 5517
        evec = self.res.evec
        non_zero = evec.flat != 0
        assert evec.flat[non_zero][0] > 0


class TestCointJoh12(CheckCointJoh):

    @classmethod
    def setup_class(cls):
        cls.res = coint_johansen(dta, 1, 2)
        cls.nobs_r = 173 - 1 - 2

        cls.res1_m = np.array(
            [
                241.985452556075,
                166.4781461662553,
                110.3298006342814,
                70.79801574443575,
                44.90887371527634,
                27.22385073668511,
                11.74205493173769,
                3.295435325623445,
                169.0618,
                133.7852,
                102.4674,
                75.1027,
                51.6492,
                32.0645,
                16.1619,
                2.7055,
                175.1584,
                139.278,
                107.3429,
                79.34220000000001,
                55.2459,
                35.0116,
                18.3985,
                3.8415,
                187.1891,
                150.0778,
                116.9829,
                87.7748,
                62.5202,
                41.0815,
                23.1485,
                6.6349,
            ]
        )
        cls.res2_m = np.array(
            [
                75.50730638981975,
                56.14834553197396,
                39.5317848898456,
                25.8891420291594,
                17.68502297859124,
                15.48179580494741,
                8.446619606114249,
                3.295435325623445,
                52.5858,
                46.5583,
                40.5244,
                34.4202,
                28.2398,
                21.8731,
                15.0006,
                2.7055,
                55.7302,
                49.5875,
                43.4183,
                37.1646,
                30.8151,
                24.2522,
                17.1481,
                3.8415,
                62.1741,
                55.8171,
                49.4095,
                42.8612,
                36.193,
                29.2631,
                21.7465,
                6.6349,
            ]
        )

        evec = np.array(
            [
                0.01102517075074406,
                -0.2185481584930077,
                0.04565819524210763,
                -0.06556394587400775,
                0.04711496306104131,
                -0.1500111976629196,
                0.03775327003706507,
                0.03479475877437702,
                0.007517888890275335,
                -0.2014629352546497,
                0.01526001455616041,
                0.0707900418057458,
                -0.002388919695513273,
                0.04486516694838273,
                -0.02936314422571188,
                0.009900554050392113,
                0.02846074144367176,
                0.02021385478834498,
                -0.04276914888645468,
                0.1738024290422287,
                0.07821155002012749,
                -0.1066523077111768,
                -0.3011042488399306,
                0.04965189679477353,
                0.07141291326159237,
                -0.01406702689857725,
                -0.07842109866080313,
                -0.04773566072362181,
                -0.04768640728128824,
                -0.04428737926285261,
                0.4143225656833862,
                0.04512787132114879,
                -0.06817130121837202,
                0.2246249779872569,
                -0.009356548567565763,
                0.006685350535849125,
                -0.02040894506833539,
                0.008131690308487425,
                -0.2503209797396666,
                0.01560186979508953,
                0.03327070126502506,
                -0.263036624535624,
                -0.04669882107497259,
                0.0146457545413255,
                0.01408691619062709,
                0.1004753600191269,
                -0.02239205763487946,
                -0.02169291468272568,
                0.08782313160608619,
                -0.07696508791577318,
                0.008925177304198475,
                -0.06230900392092828,
                -0.01548907461158638,
                0.04574831652028973,
                -0.2972228156126774,
                0.003469819004961912,
                -0.001868995544352928,
                0.05993345996347871,
                0.01213394328069316,
                0.02096614212178651,
                -0.08624395993789938,
                0.02108183181049973,
                -0.08470307289295617,
                -5.135072530480897e-005,
            ]
        )
        cls.evec_m = evec.reshape(cls.res.evec.shape, order="F")

        cls.eig_m = np.array(
            [
                0.3586376068088151,
                0.2812806889719111,
                0.2074818815675726,
                0.141259991767926,
                0.09880133062878599,
                0.08704563854307619,
                0.048471840356709,
                0.01919823444066367,
            ]
        )

    def test_evec(self):
        for col in range(self.evec_m.shape[1]):
            try:
                assert_almost_equal(self.res.evec[:, col], self.evec_m[:, col])
            except AssertionError:
                assert_almost_equal(self.res.evec[:, col], -self.evec_m[:, col])

    def test_evals(self):
        assert_almost_equal(self.res.eig, self.eig_m)


class TestCointJoh09(CheckCointJoh):

    @classmethod
    def setup_class(cls):
        cls.res = coint_johansen(dta, 0, 9)
        cls.nobs_r = 173 - 1 - 9
        # fprintf(1, '%18.16g, ', r1)
        cls.res1_m = np.array(
            [
                307.6888935095814,
                205.3839229398245,
                129.1330243009336,
                83.3101865760208,
                52.51955460357912,
                30.20027050520502,
                13.84158157562689,
                0.4117390188204866,
                153.6341,
                120.3673,
                91.109,
                65.8202,
                44.4929,
                27.0669,
                13.4294,
                2.7055,
                159.529,
                125.6185,
                95.7542,
                69.8189,
                47.8545,
                29.7961,
                15.4943,
                3.8415,
                171.0905,
                135.9825,
                104.9637,
                77.8202,
                54.6815,
                35.4628,
                19.9349,
                6.6349,
            ]
        )
        # r2 = [res.lr2 res.cvm]
        cls.res2_m = np.array(
            [
                102.3049705697569,
                76.25089863889085,
                45.82283772491284,
                30.7906319724417,
                22.31928409837409,
                16.35868892957814,
                13.4298425568064,
                0.4117390188204866,
                49.2855,
                43.2947,
                37.2786,
                31.2379,
                25.1236,
                18.8928,
                12.2971,
                2.7055,
                52.3622,
                46.2299,
                40.0763,
                33.8777,
                27.5858,
                21.1314,
                14.2639,
                3.8415,
                58.6634,
                52.3069,
                45.8662,
                39.3693,
                32.7172,
                25.865,
                18.52,
                6.6349,
            ]
        )


class TestCointJohMin18(CheckCointJoh):

    @classmethod
    def setup_class(cls):
        cls.res = coint_johansen(dta, -1, 8)
        cls.nobs_r = 173 - 1 - 8

        cls.res1_m = np.array(
            [
                260.6786029744658,
                162.7966072512681,
                105.8253545950566,
                71.16133060790817,
                47.68490211260372,
                28.11843682526138,
                13.03968537077271,
                2.25398078597622,
                137.9954,
                106.7351,
                79.5329,
                56.2839,
                37.0339,
                21.7781,
                10.4741,
                2.9762,
                143.6691,
                111.7797,
                83.9383,
                60.0627,
                40.1749,
                24.2761,
                12.3212,
                4.1296,
                154.7977,
                121.7375,
                92.7136,
                67.63670000000001,
                46.5716,
                29.5147,
                16.364,
                6.9406,
            ]
        )
        cls.res2_m = np.array(
            [
                97.88199572319769,
                56.97125265621156,
                34.66402398714837,
                23.47642849530445,
                19.56646528734234,
                15.07875145448866,
                10.7857045847965,
                2.25398078597622,
                45.893,
                39.9085,
                33.9271,
                27.916,
                21.837,
                15.7175,
                9.4748,
                2.9762,
                48.8795,
                42.7679,
                36.6301,
                30.4428,
                24.1592,
                17.7961,
                11.2246,
                4.1296,
                55.0335,
                48.6606,
                42.2333,
                35.7359,
                29.0609,
                22.2519,
                15.0923,
                6.9406,
            ]
        )


class TestCointJoh25(CheckCointJoh):

    @classmethod
    def setup_class(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=HypothesisTestWarning)
            cls.res = coint_johansen(dta, 2, 5)
        cls.nobs_r = 173 - 1 - 5

        # Note: critical values not available if trend>1
        cls.res1_m = np.array(
            [
                270.1887263915158,
                171.6870096307863,
                107.8613367358704,
                70.82424032233558,
                44.62551818267534,
                25.74352073857572,
                14.17882426926978,
                4.288656185006764,
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
            ]
        )
        cls.res1_m[cls.res1_m == 0] = np.nan
        cls.res2_m = np.array(
            [
                98.50171676072955,
                63.82567289491584,
                37.03709641353485,
                26.19872213966024,
                18.88199744409963,
                11.56469646930594,
                9.890168084263012,
                4.288656185006764,
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
            ]
        )
        cls.res2_m[cls.res2_m == 0] = np.nan


@pytest.mark.smoke
def test_coint_johansen_0lag():
    # GH 5731
    rs = np.random.RandomState(1927451)
    x_diff = rs.normal(0, 1, 1000)
    x = pd.Series(np.cumsum(x_diff))
    e1 = rs.normal(0, 1, 1000)
    y = x + 5 + e1
    data = pd.concat([x, y], axis=1)
    result = coint_johansen(data, det_order=-1, k_ar_diff=0)
    assert result.eig.shape == (2,)


@pytest.mark.parametrize("det_order, deterministic", [(-1, "n"), (0, "co")])
@pytest.mark.parametrize("k_ar_diff", [0, 1, 2])
def test_coint_johansen_trace_equals_vecm_lr(det_order, deterministic, k_ar_diff):
    # GH 7459: the trace statistic is the likelihood ratio statistic of the
    # VECM with rank r against the VECM with full rank. With k_ar_diff=0 the
    # levels were not lagged.
    data = np.log(macrodata.load_pandas().data[["realgdp", "realcons", "realinv"]])
    neqs = data.shape[1]
    res = coint_johansen(data, det_order, k_ar_diff)
    llf = [
        VECM(data, k_ar_diff=k_ar_diff, coint_rank=r, deterministic=deterministic)
        .fit()
        .llf
        for r in range(neqs + 1)
    ]
    lr_trace = [2 * (llf[neqs] - llf[r]) for r in range(neqs)]
    lr_max = [2 * (llf[r + 1] - llf[r]) for r in range(neqs)]
    assert_allclose(res.lr1, lr_trace, rtol=1e-6)
    assert_allclose(res.lr2, lr_max, rtol=1e-6)


# Reference values for k_ar_diff=0 on the log of realgdp, realcons and realinv
# of macrodata (203 observations), as (lr1, lr2). det_order -1 and 0: R 4.6.1
# with tsDyn 11.0.6, rank.test(VECM(data, lag=0, r=1, estim="ML", include="none"
# or "const"), type="trace" or "eigen"). urca::ca.jo needs K >= 2 and cannot
# fit a model without lagged differences. det_order 1 (the sample is detrended
# with a linear trend and the constant is partialled out afterwards) is not the
# model of any R package: the values are those of a base R implementation of
# the same steps (the moment matrices of the detrended differences and lagged
# levels, the eigenvalues of S11^-1 S10 S00^-1 S01).
ZERO_LAG_REFERENCE = {
    -1: (
        [224.65936991867954, 9.6618003484821973, 0.1274814825683236],
        [214.99756957019736, 9.5343188659138729, 0.1274814825683236],
    ),
    0: (
        [29.559752448160449, 12.996142176882785, 1.976442940717079],
        [16.563610271277664, 11.019699236165707, 1.976442940717079],
    ),
    1: (
        [25.018734502955127, 9.699926952440137, 0.253804581240836],
        [15.318807550514990, 9.446122371199301, 0.253804581240836],
    ),
}


@pytest.mark.parametrize("det_order", [-1, 0, 1])
def test_coint_johansen_zero_lag_reference(det_order):
    # GH 7459: without lagged differences the levels have to be lagged once
    data = np.log(macrodata.load_pandas().data[["realgdp", "realcons", "realinv"]])
    lr1, lr2 = ZERO_LAG_REFERENCE[det_order]
    res = coint_johansen(data, det_order, 0)
    assert_allclose(res.lr1, lr1, rtol=1e-8)
    assert_allclose(res.lr2, lr2, rtol=1e-8)
    # the eigenvalues follow from the maximum eigenvalue statistics
    nobs = data.shape[0] - 1
    assert_allclose(res.eig, 1 - np.exp(-np.asarray(lr2) / nobs), rtol=1e-8)


@pytest.mark.parametrize("method", ["trace", "maxeig"])
@pytest.mark.parametrize("det_order, rank", [(-1, 1), (0, 0), (1, 0)])
def test_select_coint_rank_zero_lag(det_order, rank, method):
    # GH 7459: with the unlagged levels the rank was 2 for det_order=-1 and 1
    # for det_order=0 (the trace statistic of rank 0 was 39.4 and not 29.6, the
    # 5% critical value is 29.8)
    data = np.log(macrodata.load_pandas().data[["realgdp", "realcons", "realinv"]])
    res = select_coint_rank(data, det_order, 0, method=method)
    assert res.rank == rank


def test_johansen_result_aliases_and_rkt_meth():
    # trace_stat/max_eig_stat/*_crit_vals are documented aliases for
    # lr1/lr2/cvt/cvm; rkt and meth have no alias and were never touched
    # by any existing test.
    res = coint_johansen(dta, 1, 2)

    assert res.trace_stat is res.lr1
    assert res.max_eig_stat is res.lr2
    assert res.trace_stat_crit_vals is res.cvt
    assert res.max_eig_stat_crit_vals is res.cvm
    assert res.meth == "johansen"
    assert res.rkt.shape == res.r0t.shape


def test_johansen_real_eigenvalues():
    # NumPy 2.5 and later return complex arrays from eig for real eigenvalues
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComplexWarning)
        res = coint_johansen(dta, 1, 2)
    assert not np.iscomplexobj(res.eig)
    assert not np.iscomplexobj(res.evec)


def test_coint_johansen_real_eigenvalues():
    # np.linalg.eig of the non-symmetric moment matrix product returned complex
    # eigenvalues for this random walk, cast to real with a ComplexWarning
    data = np.cumsum(np.random.default_rng(0).standard_normal((1000, 2)), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = coint_johansen(data, 0, 2)
    assert np.isrealobj(res.eig)
    assert np.all((res.eig >= 0) & (res.eig < 1))


@pytest.mark.parametrize("k_ar_diff", [1, 2])
def test_johansen_urca_danish(k_ar_diff):
    # urca 1.3.4, R 4.6.1: ca.jo(y, ecdet="none", K=k_ar_diff + 1), see
    # results/results_johansen_urca.R
    # ca.jo's statistics are reversed here to run from rank 0 to rank 3
    expected = {
        1: (
            [
                0.44821425566893597,
                0.17421468245810978,
                0.11690133941344229,
                0.01043602625491728,
            ],
            [
                48.8037309574590878,
                17.2901719813015191,
                7.1448883768914024,
                0.5560157618986894,
            ],
            [
                31.5135589761575687,
                10.1452836044101158,
                6.5888726149927139,
                0.5560157618986894,
            ],
        ),
        2: (
            [
                0.4274996665409753,
                0.2295183786296113,
                0.1089666787762107,
                0.0221312848345808,
            ],
            [
                49.724206955473313,
                20.721624978651171,
                7.163172155529512,
                1.163752513571682,
            ],
            [
                29.002581976822139,
                13.558452823121661,
                5.999419641957831,
                1.163752513571682,
            ],
        ),
    }
    eig, trace, max_eig = expected[k_ar_diff]
    y = danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]]
    res = coint_johansen(y, 0, k_ar_diff)
    assert_allclose(res.eig, eig, rtol=1e-8)
    assert_allclose(res.lr1, trace, rtol=1e-8)
    assert_allclose(res.lr2, max_eig, rtol=1e-8)


def _near_cointegrated(noise):
    rng = np.random.default_rng(20261028)
    y1 = np.cumsum(rng.standard_normal(200))
    y2 = np.cumsum(rng.standard_normal(200))
    z = rng.standard_normal(200)
    return np.column_stack((y1, y2, y1 - 0.5 * y2 + noise * z))


@pytest.mark.parametrize(
    "k_ar_diff, eig, trace, max_eig",
    [
        (
            1,
            [0.36540554504241405, 0.040365029653077172, 0.013235714614387145],
            [100.84051516026646, 10.796225890590578, 2.638169292398739],
            [90.044289269675878, 8.1580565981918388, 2.638169292398739],
        ),
        (
            2,
            [0.30057885169185051, 0.033658305112658369, 0.012847935629496819],
            [79.720223857332741, 9.2922871037768755, 2.5474432817087609],
            [70.427936753555865, 6.7448438220681147, 2.5474432817087609],
        ),
    ],
)
def test_johansen_near_cointegration(k_ar_diff, eig, trace, max_eig):
    # Reference values are the squared canonical correlations between the
    # residuals of the differences and of the levels, computed with mpmath
    # using 60 digits for the data of _near_cointegrated(1e-6). The system is
    # almost exactly cointegrated, which makes the moment matrices
    # ill-conditioned: the eigenvalues of the moment matrix product are
    # wrong (some are larger than 1 or negative) if they are computed with eig.
    res = coint_johansen(_near_cointegrated(1e-6), 0, k_ar_diff)
    assert_allclose(res.eig, eig, rtol=1e-6)
    assert_allclose(res.lr1, trace, rtol=1e-6)
    assert_allclose(res.lr2, max_eig, rtol=1e-6)


def test_johansen_eigenvalue_near_one():
    # The first eigenvalue differs from 1 by about 1e-18, so that
    # 1 - eig rounds to 0 in double precision and the statistics must not be
    # computed from log(1 - eig). Reference values from mpmath with 80 digits.
    rng = np.random.default_rng(7)
    y1 = np.resize([1.0, 0.5, -1.5], 100) + 1e-9 * rng.standard_normal(100)
    y2 = np.cumsum(rng.standard_normal(100))
    res = coint_johansen(np.column_stack((y1, y2)), -1, 1)
    assert_allclose(res.eig, [1.0, 0.00035259704457330609], rtol=1e-6)
    assert_allclose(res.lr1, [4062.8865249453624, 0.034560603709671807], rtol=1e-6)
    assert_allclose(res.lr2, [4062.8519643416528, 0.034560603709671807], rtol=1e-6)


def _singular_endog(case):
    rng = np.random.default_rng(12345)
    y = np.cumsum(rng.standard_normal((200, 2)), axis=0)
    if case == "duplicated":
        return np.column_stack((y, y[:, 0]))
    elif case == "collinear":
        return np.column_stack((y, y[:, 0] + y[:, 1]))
    elif case == "scaled collinear":
        return np.column_stack((y, 1e6 * y[:, 0] - 3 * y[:, 1]))
    # Differences that repeat with period 2, so the lagged difference is
    # proportional to the difference
    return np.column_stack((np.cumsum(np.tile([1.0, -1.0], 100)), y[:, 0]))


@pytest.mark.parametrize(
    "case, det_order, k_ar_diff",
    [
        ("duplicated", 0, 1),
        ("collinear", -1, 2),
        ("scaled collinear", 0, 0),
        ("periodic", 0, 1),
    ],
)
def test_johansen_linearly_dependent(case, det_order, k_ar_diff):
    with pytest.raises(np.linalg.LinAlgError, match="linearly dependent"):
        coint_johansen(_singular_endog(case), det_order, k_ar_diff)


@pytest.mark.parametrize(
    "shape, det_order, k_ar_diff", [((6, 2), 0, 2), ((5, 3), -1, 0)]
)
def test_johansen_too_few_observations(shape, det_order, k_ar_diff):
    endog = np.cumsum(np.random.default_rng(0).standard_normal(shape), axis=0)
    with pytest.raises(ValueError, match="too few observations"):
        coint_johansen(endog, det_order, k_ar_diff)


@pytest.mark.parametrize("det_order", [-1, 0, 1])
def test_johansen_more_than_12_variables(det_order):
    # The statistics are computed for any number of variables, but the tables
    # of critical values end at 12 variables: the critical values of the
    # first test (13 variables) are NaN and the others are tabulated
    rng = np.random.default_rng(20261122)
    endog = np.cumsum(rng.standard_normal((200, 13)), axis=0)
    with pytest.warns(HypothesisTestWarning, match="12 variables at most"):
        res = coint_johansen(endog, det_order, 1)
    assert np.all(np.isnan(res.cvt[0]))
    assert np.all(np.isnan(res.cvm[0]))
    for i in range(1, 13):
        assert_allclose(res.cvt[i], c_sjt(13 - i, det_order), rtol=1e-15)
        assert_allclose(res.cvm[i], c_sja(13 - i, det_order), rtol=1e-15)
    # The eigenvalues of the generalized symmetric eigenvalue problem of the
    # residual moment matrices (Luetkepohl 2005, p. 295) and the statistics
    # from them
    nobs = res.r0t.shape[0]
    s00 = res.r0t.T @ res.r0t / nobs
    s0k = res.r0t.T @ res.rkt / nobs
    skk = res.rkt.T @ res.rkt / nobs
    eig = linalg.eigh(s0k.T @ np.linalg.solve(s00, s0k), skk, eigvals_only=True)
    eig = eig[::-1]
    assert_allclose(res.eig, eig, rtol=1e-10)
    log_1m_eig = np.log(1 - eig)
    assert_allclose(res.lr1, -nobs * np.cumsum(log_1m_eig[::-1])[::-1], rtol=1e-10)
    assert_allclose(res.lr2, -nobs * log_1m_eig, rtol=1e-10)


@pytest.mark.parametrize("p", [-1, 0, 1])
def test_coint_tables_one_variable(p):
    # With one variable, the trace and the maximum eigenvalue statistics are
    # the same statistic
    assert_allclose(c_sja(1, p), c_sjt(1, p), rtol=1e-15)
    if p >= 0:
        # The tabulated values equal the 90%, 95% and 99% quantiles of
        # chi2(1) to the 4 decimals given
        expected = stats.chi2.ppf([0.90, 0.95, 0.99], 1)
        assert_allclose(c_sja(1, p), expected, rtol=1e-4)


@pytest.mark.parametrize("func", [c_sja, c_sjt])
@pytest.mark.parametrize("p", [-1, 0, 1])
def test_coint_tables_ordered(func, p):
    # The critical values increase with the level and with the number of
    # variables, and n can be any integer type
    values = np.array([func(n, p) for n in range(1, 13)])
    assert np.all(np.diff(values, axis=1) > 0)
    assert np.all(np.diff(values, axis=0) > 0)
    assert_allclose(func(np.int64(3), p), values[2], rtol=1e-15)


@pytest.mark.parametrize("func", [c_sja, c_sjt])
@pytest.mark.parametrize("n, p", [(0, 0), (13, 0), (3, -2), (3, 2)])
def test_coint_tables_out_of_range(func, n, p):
    assert np.all(np.isnan(func(n, p)))


@pytest.mark.parametrize("func", [c_sja, c_sjt])
@pytest.mark.parametrize("p", [0.5, -0.5, 1.5, -1.5])
def test_coint_tables_invalid_p(func, p):
    # p is not an integer. Between -1 and 1, c_sja raised an
    # UnboundLocalError and c_sjt "invalid p"; outside, both returned NaN.
    with pytest.raises(ValueError, match="p must be an integer, got"):
        func(3, p)


# Rows n = 3 of the tables, MacKinnon, Haug and Michelis (1996)
COINT_TABLES_N3 = {
    (c_sja, -1): [15.7175, 17.7961, 22.2519],
    (c_sja, 0): [18.8928, 21.1314, 25.8650],
    (c_sja, 1): [21.8731, 24.2522, 29.2631],
    (c_sjt, -1): [21.7781, 24.2761, 29.5147],
    (c_sjt, 0): [27.0669, 29.7961, 35.4628],
    (c_sjt, 1): [32.0645, 35.0116, 41.0815],
}


@pytest.mark.parametrize("func", [c_sja, c_sjt])
@pytest.mark.parametrize("p", [-1, 0, 1])
def test_coint_tables_integral_p(func, p):
    expected = COINT_TABLES_N3[func, p]
    for value in (p, float(p), np.float64(p), np.int64(p), np.array(p)):
        assert_allclose(func(3, value), expected, rtol=1e-15)


@pytest.mark.parametrize("func", [c_sja, c_sjt])
@pytest.mark.parametrize("p", [True, False, np.True_, np.False_])
def test_coint_tables_bool_p_deprecated(func, p):
    # A bool selected the table of 0 or 1. It is deprecated.
    with pytest.warns(FutureWarning, match="^p must be an integer. Passing a bool"):
        res = func(3, p)
    assert_allclose(res, COINT_TABLES_N3[func, int(p)], rtol=1e-15)


def test_coint_johansen_bool_det_order_deprecated():
    # det_order=True was used as 1 by coint_johansen and the tables
    endog = np.cumsum(np.random.default_rng(0).standard_normal((100, 2)), axis=0)
    expected = coint_johansen(endog, 1, 1)
    with pytest.warns(FutureWarning, match="^p must be an integer. Passing a bool"):
        res = coint_johansen(endog, True, 1)
    assert_allclose(res.lr1, expected.lr1, rtol=1e-12)
    assert_allclose(res.cvt, expected.cvt, rtol=1e-15)
    assert_allclose(res.cvm, expected.cvm, rtol=1e-15)
    # The critical values of 2 and 1 variables with a trend
    assert_allclose(res.cvt[:, 1], [18.3985, 3.8415], rtol=1e-15)


@pytest.mark.parametrize("func", [c_sja, c_sjt])
@pytest.mark.parametrize("p", [-1, 0, 1])
def test_coint_tables_return_copy(func, p):
    # The result was a view of the table, so changing it changed the
    # values returned by later calls
    expected = COINT_TABLES_N3[func, p]
    res = func(3, p)
    try:
        res[:] = 0.0
        assert_allclose(func(3, p), expected, rtol=1e-15)
    finally:
        # Restore the table if the result is a view
        res[:] = expected


@pytest.mark.parametrize("form", ["list", "tuple"])
def test_johansen_list_endog(form):
    # endog is array_like, but coint_johansen and select_coint_rank read
    # endog.shape before converting it, so a list raised an AttributeError.
    # urca 1.3.4, R 4.6.1: ca.jo(y, ecdet="none", K=2), as in
    # test_johansen_urca_danish
    trace = [48.8037309574590878, 17.2901719813015191, 7.1448883768914024]
    trace += [0.5560157618986894]
    y = danish_data.load_pandas().data[["lrm", "lry", "ibo", "ide"]].to_numpy()
    endog = y.tolist() if form == "list" else tuple(map(tuple, y))
    res = coint_johansen(endog, 0, 1)
    assert_allclose(res.lr1, trace, rtol=1e-8)
    rank = select_coint_rank(endog, 0, 1, method="trace", signif=0.05)
    # the first statistic is above its 5% critical value, the second is not
    crit = [c_sjt(4, 0)[1], c_sjt(3, 0)[1]]
    assert trace[0] > crit[0]
    assert trace[1] < crit[1]
    assert rank.rank == 1
    assert rank.neqs == 4
    assert_allclose(rank.test_stats, trace[:2], rtol=1e-8)
    assert_allclose(rank.crit_vals, crit, rtol=1e-15)
