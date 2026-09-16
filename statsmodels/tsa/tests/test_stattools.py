from statsmodels.compat.pandas import MONTH_END, YEAR_END, assert_index_equal
from statsmodels.compat.platform import PLATFORM_WIN
from statsmodels.compat.python import PYTHON_IMPL_WASM, lrange

from pathlib import Path
import warnings

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_equal,
)
import pandas as pd
from pandas import DataFrame, Series, date_range
import pytest
from scipy import stats
from scipy.interpolate import interp1d

from statsmodels.datasets import macrodata, modechoice, nile, randhie, sunspots
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.sm_exceptions import (
    CollinearityWarning,
    InfeasibleTestError,
    InterpolationWarning,
    MissingDataError,
    SingularMatrixWarning,
)

# Remove imports when range unit root test gets an R implementation
from statsmodels.tools.validation import array_like, bool_like
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import (
    AcfResult,
    ADFullerResult,
    BreakvarHeteroskedasticityResult,
    CcfResult,
    CointResult,
    JackknifeResult,
    KPSSResult,
    LevinsonDurbinPacfResult,
    LevinsonDurbinResult,
    PacfBurgResult,
    PacfResult,
    PccfResult,
    QStatResult,
    RURResult,
    acf,
    acovf,
    adfuller,
    arma_order_select_ic,
    block_jackknife,
    breakvar_heteroskedasticity_test,
    ccf,
    ccovf,
    coint,
    diebold_mariano_test,
    grangercausalitytests,
    innovations_algo,
    innovations_filter,
    kpss,
    levinson_durbin,
    levinson_durbin_pacf,
    leybourne,
    pacf,
    pacf_burg,
    pacf_ols,
    pacf_yw,
    pccf,
    q_stat,
    range_unit_root_test,
    zivot_andrews,
)

DECIMAL_8 = 8
DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1

CURR_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def acovf_data():
    rs = np.random.RandomState(12345)
    return rs.randn(250)


@pytest.fixture(scope="module")
def gc_data():
    mdata = macrodata.load_pandas().data
    mdata = mdata[["realgdp", "realcons"]].values
    data = mdata.astype(float)
    return np.diff(np.log(data), axis=0)


class CheckADF:
    """
    Test Augmented Dickey-Fuller

    Test values taken from Stata.
    """

    levels = ["1%", "5%", "10%"]
    data = macrodata.load_pandas()
    x = data.data["realgdp"].values
    y = data.data["infl"].values

    def test_teststat(self):
        assert_almost_equal(self.res1[0], self.teststat, DECIMAL_5)

    def test_pvalue(self):
        assert_almost_equal(self.res1[1], self.pvalue, DECIMAL_5)

    def test_critvalues(self):
        critvalues = [self.res1[4][lev] for lev in self.levels]
        assert_almost_equal(critvalues, self.critvalues, DECIMAL_2)


class TestADFConstant(CheckADF):
    """
    Dickey-Fuller test for unit root
    """

    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(
            cls.x, regression="c", autolag=None, maxlag=4, result_object=False
        )
        cls.teststat = 0.97505319
        cls.pvalue = 0.99399563
        cls.critvalues = [-3.476, -2.883, -2.573]


class TestADFConstantTrend(CheckADF):
    """"""

    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(
            cls.x, regression="ct", autolag=None, maxlag=4, result_object=False
        )
        cls.teststat = -1.8566374
        cls.pvalue = 0.67682968
        cls.critvalues = [-4.007, -3.437, -3.137]


# FIXME: do not leave commented-out
# class TestADFConstantTrendSquared(CheckADF):
#    """
#    """
#    pass
# TODO: get test values from R?


class TestADFNoConstant(CheckADF):
    """"""

    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(
            cls.x, regression="n", autolag=None, maxlag=4, result_object=False
        )
        cls.teststat = 3.5227498

        cls.pvalue = 0.99999
        # Stata does not return a p-value for noconstant.
        # Tau^max in MacKinnon (1994) is missing, so it is
        # assumed that its right-tail is well-behaved

        cls.critvalues = [-2.587, -1.950, -1.617]


# No Unit Root


class TestADFConstant2(CheckADF):
    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(
            cls.y, regression="c", autolag=None, maxlag=1, result_object=False
        )
        cls.teststat = -4.3346988
        cls.pvalue = 0.00038661
        cls.critvalues = [-3.476, -2.883, -2.573]


class TestADFConstantTrend2(CheckADF):
    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(
            cls.y, regression="ct", autolag=None, maxlag=1, result_object=False
        )
        cls.teststat = -4.425093
        cls.pvalue = 0.00199633
        cls.critvalues = [-4.006, -3.437, -3.137]


class TestADFNoConstant2(CheckADF):
    @classmethod
    def setup_class(cls):
        cls.res1 = adfuller(
            cls.y, regression="n", autolag=None, maxlag=1, result_object=False
        )
        cls.teststat = -2.4511596
        cls.pvalue = 0.013747
        # Stata does not return a p-value for noconstant
        # this value is just taken from our results
        cls.critvalues = [-2.587, -1.950, -1.617]
        _, _1, _2, cls.store = adfuller(
            cls.y,
            regression="n",
            autolag=None,
            maxlag=1,
            store=True,
            result_object=False,
        )

    def test_store_str(self):
        assert_equal(self.store.__str__(), "Augmented Dickey-Fuller Test Results")


@pytest.mark.parametrize("x", [np.full(8, 5.0)])
def test_adfuller_resid_variance_zero(x):
    with pytest.raises(ValueError):
        adfuller(x)


class CheckCorrGram:
    """
    Set up for ACF, PACF tests.
    """

    data = macrodata.load_pandas()
    x = data.data["realgdp"]
    filename = Path(CURR_DIR).joinpath("results", "results_corrgram.csv")
    results = pd.read_csv(filename, delimiter=",")


class TestACF(CheckCorrGram):
    """
    Test Autocorrelation Function
    """

    @classmethod
    def setup_class(cls):
        cls.acf = cls.results["acvar"]
        # cls.acf = np.concatenate(([1.], cls.acf))
        cls.qstat = cls.results["Q1"]
        cls.res1 = acf(
            cls.x, nlags=40, qstat=True, alpha=0.05, fft=False, result_object=False
        )
        cls.confint_res = cls.results[["acvar_lb", "acvar_ub"]].values

    def test_acf(self):
        assert_almost_equal(self.res1.acf[1:41], self.acf, DECIMAL_8)

    def test_confint(self):
        centered = self.res1.confint - self.res1.confint.mean(1)[:, None]
        assert_almost_equal(centered[1:41], self.confint_res, DECIMAL_8)

    def test_qstat(self):
        assert_almost_equal(self.res1.qstat[:40], self.qstat, DECIMAL_3)
        # 3 decimal places because of stata rounding

    # FIXME: enable/xfail/skip or delete
    # def pvalue(self):
    #    pass
    # NOTE: should not need testing if Q stat is correct

    def test_result_object_true(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = acf(
                self.x,
                nlags=40,
                qstat=True,
                alpha=0.05,
                fft=False,
                result_object=True,
            )
        assert isinstance(res, AcfResult)
        assert res[0] is res.acf
        assert res[1] is res.confint
        assert res[2] is res.qstat
        assert res[3] is res.pvalues
        assert_almost_equal(res.acf[1:41], self.acf, DECIMAL_8)
        assert_almost_equal(res.qstat[:40], self.qstat, DECIMAL_3)

    @pytest.mark.parametrize("qstat", [True, False])
    @pytest.mark.parametrize("alpha", [None, 0.05])
    def test_result_object_true_always_four_fields(self, qstat, alpha):
        # AcfResult always has 4 fields regardless of which of qstat/alpha
        # were requested; unrequested fields are None rather than omitted.
        if not (qstat or alpha):
            pytest.skip("not a variable-arity case; acf always returns bare array")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = acf(
                self.x,
                nlags=10,
                qstat=qstat,
                alpha=alpha,
                fft=False,
                result_object=True,
            )
        assert isinstance(res, AcfResult)
        assert isinstance(res.acf, np.ndarray)
        assert (res.confint is None) == (alpha is None)
        assert (res.qstat is None) == (not qstat)
        assert (res.pvalues is None) == (not qstat)

    def test_default_warns(self):
        with pytest.warns(FutureWarning, match="result_object"):
            res = acf(self.x, nlags=40, qstat=True, fft=False)
        assert isinstance(res, tuple)
        assert not isinstance(res, AcfResult)

    def test_no_qstat_no_alpha_never_warns(self):
        # The single-output path is unchanged and must stay silent, but an
        # explicit result_object=True still returns the result object with
        # the unrequested fields left as None.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = acf(self.x, nlags=40, fft=False)
            res_nt = acf(self.x, nlags=40, fft=False, result_object=True)
        assert isinstance(res, np.ndarray)
        assert isinstance(res_nt, AcfResult)
        assert_allclose(res_nt.acf, res)
        assert res_nt.confint is None
        assert res_nt.qstat is None
        assert res_nt.pvalues is None

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"alpha": 0.05}, 2),
            ({"qstat": True}, 3),
            ({"qstat": True, "alpha": 0.05}, 4),
        ],
    )
    def test_legacy_unpacking_preserved(self, kwargs, expected):
        # AcfResult always carries four fields, so it may only be adopted by
        # default where that matches the legacy arity.  Returning it for
        # alpha-only or qstat-only would raise "too many values to unpack"
        # in existing user code.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = acf(self.x, nlags=40, fft=False, **kwargs)
            opted_out = acf(self.x, nlags=40, fft=False, result_object=False, **kwargs)
        assert len(res) == expected
        assert len(opted_out) == expected

        # unpacking with exactly `expected` names is a stable, non-deprecated
        # part of the API, whether res is a bare tuple or an AcfResult.
        _ = [*res]
        assert len([*res]) == expected

    def test_only_full_request_adopts_result_object_silently(self):
        # Both qstat and alpha -> the four-field result object matches the
        # legacy 4-tuple exactly, so it is adopted with no warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = acf(self.x, nlags=40, fft=False, qstat=True, alpha=0.05)
        assert isinstance(res, AcfResult)
        # Requesting only one of them still warns and keeps the short tuple.
        for kwargs in ({"alpha": 0.05}, {"qstat": True}):
            with pytest.warns(FutureWarning, match="result_object"):
                short = acf(self.x, nlags=40, fft=False, **kwargs)
            assert not isinstance(short, AcfResult)


class TestACF_FFT(CheckCorrGram):
    # Test Autocorrelation Function using FFT
    @classmethod
    def setup_class(cls):
        cls.acf = cls.results["acvarfft"]
        cls.qstat = cls.results["Q1"]
        cls.res1 = acf(cls.x, nlags=40, qstat=True, fft=True, result_object=False)

    def test_acf(self):
        assert_almost_equal(self.res1[0][1:], self.acf, DECIMAL_8)

    def test_qstat(self):
        # todo why is res1/qstat 1 short
        assert_almost_equal(self.res1[1], self.qstat, DECIMAL_3)


class TestACFMissing(CheckCorrGram):
    # Test Autocorrelation Function using Missing
    @classmethod
    def setup_class(cls):
        cls.x = np.concatenate((np.array([np.nan]), cls.x))
        cls.acf = cls.results["acvar"]  # drop and conservative
        cls.qstat = cls.results["Q1"]
        cls.confint_res = cls.results[["acvar_lb", "acvar_ub"]].values
        cls.res_drop = acf(
            cls.x,
            nlags=40,
            qstat=True,
            alpha=0.05,
            missing="drop",
            fft=False,
            result_object=False,
        )
        cls.res_conservative = acf(
            cls.x,
            nlags=40,
            qstat=True,
            alpha=0.05,
            fft=False,
            missing="conservative",
            result_object=False,
        )
        cls.acf_none = np.empty(40) * np.nan  # lags 1 to 40 inclusive
        cls.qstat_none = np.empty(40) * np.nan
        cls.res_none = acf(
            cls.x,
            nlags=40,
            qstat=True,
            alpha=0.05,
            missing="none",
            fft=False,
            result_object=False,
        )

    def test_raise(self):
        with pytest.raises(MissingDataError):
            acf(
                self.x,
                nlags=40,
                qstat=True,
                fft=False,
                alpha=0.05,
                missing="raise",
            )

    def test_acf_none(self):
        assert_almost_equal(self.res_none.acf[1:41], self.acf_none, DECIMAL_8)

    def test_acf_drop(self):
        assert_almost_equal(self.res_drop.acf[1:41], self.acf, DECIMAL_8)

    def test_acf_conservative(self):
        assert_almost_equal(self.res_conservative.acf[1:41], self.acf, DECIMAL_8)

    def test_qstat_none(self):
        # todo why is res1/qstat 1 short
        assert_almost_equal(self.res_none.qstat, self.qstat_none, DECIMAL_3)

    def test_qstat_drop(self):
        assert_almost_equal(self.res_drop.qstat[:40], self.qstat, DECIMAL_3)

    def test_qstat_conservative(self):
        assert_almost_equal(self.res_conservative.qstat[:40], self.qstat, DECIMAL_3)

    def test_confint_drop(self):
        centered = self.res_drop.confint - self.res_drop.confint.mean(1)[:, None]
        assert_almost_equal(centered[1:41], self.confint_res, DECIMAL_8)

    def test_confint_conservative(self):
        centered = (
            self.res_conservative.confint
            - self.res_conservative.confint.mean(1)[:, None]
        )
        assert_almost_equal(centered[1:41], self.confint_res, DECIMAL_8)

    @pytest.mark.parametrize("missing", ["drop", "conservative"])
    def test_drop_all(self, missing):
        all_missing = np.full_like(self.x, np.nan)
        with pytest.raises(ValueError, match="All observations are missing"):
            acf(all_missing, nlags=40, missing=missing)


class TestPACF(CheckCorrGram):
    @classmethod
    def setup_class(cls):
        cls.pacfols = cls.results["PACOLS"]
        cls.pacfyw = cls.results["PACYW"]

    def test_ols(self):
        _result = pacf(
            self.x, nlags=40, alpha=0.05, method="ols", result_object=False
        )
        pacfols, confint = _result.pacf, _result.confint
        assert_almost_equal(pacfols[1:], self.pacfols, DECIMAL_6)
        centered = confint - confint.mean(1)[:, None]
        # from edited Stata ado file
        res = [[-0.1375625, 0.1375625]] * 40
        assert_almost_equal(centered[1:41], res, DECIMAL_6)
        # check lag 0
        assert_equal(centered[0], [0.0, 0.0])
        assert_equal(confint[0], [1, 1])
        assert_equal(pacfols[0], 1)

    def test_alpha_default_returns_result_object(self):
        # PacfResult has the same length and contents as the legacy
        # (pacf, confint) tuple, so it is adopted without a warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = pacf(self.x, nlags=40, alpha=0.05, method="ols")
        assert isinstance(res, PacfResult)
        assert len(res) == 2
        # The result object is used whenever it matches the legacy tuple's
        # contents, so result_object=False cannot opt out of it here.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            opted_out = pacf(
                self.x, nlags=40, alpha=0.05, method="ols", result_object=False
            )
        assert isinstance(opted_out, PacfResult)
        # ...and unpacking is a stable, non-deprecated part of the API.
        vals, confint = res
        assert_allclose(vals, res.pacf)
        assert_allclose(confint, res.confint)

    def test_alpha_result_object_true(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = pacf(self.x, nlags=40, alpha=0.05, method="ols", result_object=True)
        assert isinstance(res, PacfResult)
        assert res[0] is res.pacf
        assert res[1] is res.confint

    def test_no_alpha_never_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = pacf(self.x, nlags=40, method="ols")
            res_nt = pacf(self.x, nlags=40, method="ols", result_object=True)
        assert isinstance(res, np.ndarray)
        assert isinstance(res_nt, PacfResult)
        assert_allclose(res_nt.pacf, res)
        assert res_nt.confint is None

    def test_ols_inefficient(self):
        lag_len = 5
        pacfols = pacf_ols(self.x, nlags=lag_len, efficient=False)
        x = self.x.copy()
        x -= x.mean()
        n = x.shape[0]
        lags = np.zeros((n - 5, 5))
        lead = x[5:]
        direct = np.empty(lag_len + 1)
        direct[0] = 1.0
        for i in range(lag_len):
            lags[:, i] = x[5 - (i + 1) : -(i + 1)]
            direct[i + 1] = np.linalg.lstsq(lags[:, : (i + 1)], lead, rcond=None)[0][-1]
        assert_allclose(pacfols, direct, atol=1e-8)

    def test_yw(self):
        pacfyw = pacf_yw(self.x, nlags=40, method="mle")
        assert_almost_equal(pacfyw[1:], self.pacfyw, DECIMAL_8)

    @pytest.mark.skipif(PYTHON_IMPL_WASM, reason="No fp exception support in WASM")
    def test_yw_singular(self):
        with pytest.warns(SingularMatrixWarning):
            pacf(np.ones(30), nlags=6)

    def test_ld(self):
        pacfyw = pacf_yw(self.x, nlags=40, method="mle")
        pacfld = pacf(self.x, nlags=40, method="ldb")
        assert_almost_equal(pacfyw, pacfld, DECIMAL_8)

        pacfyw = pacf(self.x, nlags=40, method="yw")
        pacfld = pacf(self.x, nlags=40, method="lda")
        assert_almost_equal(pacfyw, pacfld, DECIMAL_8)

    def test_burg(self):
        pacfburg_ = pacf_burg(self.x, nlags=40).pacf
        pacfburg = pacf(self.x, nlags=40, method="burg")
        assert_almost_equal(pacfburg_, pacfburg, DECIMAL_8)


class TestCCF:
    """
    Test cross-correlation function
    """

    data = macrodata.load_pandas()
    x = data.data["unemp"].diff().dropna()
    y = data.data["infl"].diff().dropna()
    filename = Path(CURR_DIR).joinpath("results", "results_ccf.csv")
    results = pd.read_csv(filename, delimiter=",")
    nlags = 20

    @classmethod
    def setup_class(cls):
        cls.ccf = cls.results["ccf"]
        cls.res1 = ccf(cls.x, cls.y, nlags=cls.nlags, adjusted=False, fft=False)

    def test_ccf(self):
        assert_almost_equal(self.res1, self.ccf, DECIMAL_8)

    def test_confint(self):
        alpha = 0.05
        _result = ccf(
            self.x,
            self.y,
            nlags=self.nlags,
            adjusted=False,
            fft=False,
            alpha=alpha,
            result_object=False,
        )
        res2, confint = _result.ccf, _result.confint
        assert_equal(res2, self.res1)
        assert_almost_equal(res2 - confint[:, 0], confint[:, 1] - res2, DECIMAL_8)
        alpha1 = stats.norm.cdf(confint[:, 1] - res2, scale=1.0 / np.sqrt(len(self.x)))
        assert_almost_equal(alpha1, np.repeat(1 - alpha / 2.0, self.nlags), DECIMAL_8)

    def test_alpha_default_returns_result_object(self):
        # CcfResult has the same length and contents as the legacy
        # (ccf, confint) tuple, so it is adopted without a warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = ccf(self.x, self.y, nlags=self.nlags, alpha=0.05)
        assert isinstance(res, CcfResult)
        assert len(res) == 2
        # The result object is used whenever it matches the legacy tuple's
        # contents, so result_object=False cannot opt out of it here.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            opted_out = ccf(
                self.x, self.y, nlags=self.nlags, alpha=0.05, result_object=False
            )
        assert isinstance(opted_out, CcfResult)
        # ...and unpacking is a stable, non-deprecated part of the API.
        vals, confint = res
        assert_allclose(vals, res.ccf)
        assert_allclose(confint, res.confint)

    def test_alpha_result_object_true(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = ccf(self.x, self.y, nlags=self.nlags, alpha=0.05, result_object=True)
        assert isinstance(res, CcfResult)
        assert res[0] is res.ccf
        assert res[1] is res.confint

    def test_no_alpha_never_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = ccf(self.x, self.y, nlags=self.nlags)
            res_nt = ccf(self.x, self.y, nlags=self.nlags, result_object=True)
        assert isinstance(res, np.ndarray)
        assert isinstance(res_nt, CcfResult)
        assert_allclose(res_nt.ccf, res)
        assert res_nt.confint is None


class TestPCCF:
    data = macrodata.load_pandas()
    x = data.data["realgdp"]
    y = data.data["realcons"]
    filename = Path(CURR_DIR).joinpath("results", "results_pccf.csv")
    results = pd.read_csv(filename, delimiter=",")
    nlags = 20

    @classmethod
    def setup_class(cls):
        cls.pccf = cls.results["pccf"]
        cls.res1 = pccf(cls.x, cls.y, nlags=cls.nlags, method="ols")

    def test_pccf(self):
        assert_almost_equal(self.res1, self.pccf, DECIMAL_8)

    def test_pccf_hand_computed(self):
        x = np.array(
            [2.1, 4.5, 1.3, 6.8, 3.2, 5.7, 0.9, 7.4, 2.8, 4.1, 6.3, 1.7, 5.5, 3.9, 7.1]
        )
        y = np.array(
            [3.4, 2.7, 5.1, 1.8, 4.6, 3.3, 6.2, 2.5, 4.8, 3.1, 5.5, 2.2, 4.3, 3.7, 5.9]
        )
        result = pccf(x, y, nlags=3, method="ols")
        expected = np.array(
            [
                0.46195683919821806,
                0.11931602624087348,
                0.5204421499138578,
            ]
        )
        assert_almost_equal(result, expected, DECIMAL_8)

    def test_confint(self):
        alpha = 0.05
        _result = pccf(
            self.x,
            self.y,
            nlags=self.nlags,
            method="ols",
            alpha=alpha,
            result_object=False,
        )
        res2, confint = _result.pccf, _result.confint
        assert_equal(res2, self.res1)
        assert_almost_equal(res2 - confint[:, 0], confint[:, 1] - res2, DECIMAL_8)
        alpha1 = stats.norm.cdf(
            confint[:, 1] - res2,
            scale=1.0 / np.sqrt(len(self.x)),
        )
        assert_almost_equal(alpha1, np.repeat(1 - alpha / 2.0, self.nlags), DECIMAL_8)

    def test_alpha_default_returns_result_object(self):
        # PccfResult has the same length and contents as the legacy
        # (pccf, confint) tuple, so it is adopted without a warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = pccf(self.x, self.y, nlags=self.nlags, method="ols", alpha=0.05)
        assert isinstance(res, PccfResult)
        assert len(res) == 2
        # The result object is used whenever it matches the legacy tuple's
        # contents, so result_object=False cannot opt out of it here.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            opted_out = pccf(
                self.x,
                self.y,
                nlags=self.nlags,
                method="ols",
                alpha=0.05,
                result_object=False,
            )
        assert isinstance(opted_out, PccfResult)
        # ...and unpacking is a stable, non-deprecated part of the API.
        vals, confint = res
        assert_allclose(vals, res.pccf)
        assert_allclose(confint, res.confint)

    def test_alpha_result_object_true(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = pccf(
                self.x,
                self.y,
                nlags=self.nlags,
                method="ols",
                alpha=0.05,
                result_object=True,
            )
        assert isinstance(res, PccfResult)
        assert res[0] is res.pccf
        assert res[1] is res.confint

    def test_no_alpha_never_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = pccf(self.x, self.y, nlags=self.nlags, method="ols")
            res_nt = pccf(
                self.x, self.y, nlags=self.nlags, method="ols", result_object=True
            )
        assert isinstance(res, np.ndarray)
        assert isinstance(res_nt, PccfResult)
        assert_allclose(res_nt.pccf, res)
        assert res_nt.confint is None

    def test_confint_widths(self):
        alphas = [0.01, 0.05, 0.10]
        widths = {}
        for a in alphas:
            confint = pccf(
                self.x, self.y, nlags=5, method="ols", alpha=a, result_object=False
            ).confint
            widths[a] = confint[:, 1] - confint[:, 0]
        assert np.all(widths[0.01] > widths[0.05])
        assert np.all(widths[0.05] > widths[0.10])

    def test_pccf_edge_cases(self):
        x_small = np.array([1.0, 2.0, 3.0])
        y_small = np.array([4.0, 5.0, 6.0])
        result_small = pccf(x_small, y_small, nlags=1, method="ols")
        assert len(result_small) == 1
        assert not np.isnan(result_small[0])

        with pytest.raises(ValueError):
            pccf(self.x[:10], self.y[:15], nlags=5)

    def test_pccf_statistical_properties(self):
        result = pccf(self.x, self.y, nlags=10)
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= -1.0)
        assert np.all(valid_values <= 1.0)

        result_lag1 = pccf(self.x, self.y, nlags=1, method="ols")
        ccf_lag1 = np.corrcoef(self.x[:-1], self.y[1:])[0, 1]
        assert_almost_equal(result_lag1[0], ccf_lag1, DECIMAL_8)

    def test_pccf_parameter_validation(self):
        with pytest.raises(ValueError):
            pccf(self.x, self.y, nlags=0)
        with pytest.raises(ValueError):
            pccf(self.x, self.y, nlags=-1)
        with pytest.raises(ValueError):
            pccf(self.x[:10], self.y[:10], nlags=6)

    @pytest.mark.parametrize("method", ["ols", "yw", "ywm"])
    def test_constant_series(self, method):
        x_const = np.ones(50)
        y_const = np.ones(50) * 2.0
        result = pccf(x_const, y_const, nlags=5, method=method)
        assert len(result) == 5
        assert np.all(np.isnan(result))

    def test_yw_singular_intermediate_recursion_returns_nan(self):
        x = np.array([0.0, 0.0, -1.0, 2.0, -1.0])
        y = np.array([-1.0, 0.0, -2.0, -2.0, 0.0])
        result = pccf(x, y, nlags=2, method="yw")
        assert len(result) == 2
        assert np.all(np.isfinite(result) | np.isnan(result))

    def test_return_consistency(self):
        result_no_alpha = pccf(self.x, self.y, nlags=5)
        _result = pccf(
            self.x, self.y, nlags=5, alpha=0.05, result_object=False
        )
        result_with_alpha, confint = _result.pccf, _result.confint
        assert_almost_equal(result_no_alpha, result_with_alpha, DECIMAL_8)
        assert confint.shape == (5, 2)
        assert np.all(confint[:, 0] <= result_with_alpha)
        assert np.all(result_with_alpha <= confint[:, 1])

    def test_default_nlags(self):
        result = pccf(self.x, self.y)
        nobs = len(self.x)
        expected_nlags = min(int(10 * np.log10(nobs)), nobs // 2 - 1)
        assert len(result) == expected_nlags

    def test_nan_fallback_large_lag(self):
        x_short = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_short = np.array([2.0, 3.0, 1.0, 4.0, 2.0])
        result = pccf(x_short, y_short, nlags=2, method="ols")
        assert len(result) == 2
        assert np.isnan(result[1])

    def test_var1_pccf_cutoff(self):
        rng = np.random.default_rng(98765)
        n = 500
        x = np.zeros(n)
        y = np.zeros(n)
        x[0] = rng.standard_normal()
        y[0] = rng.standard_normal()
        for t in range(1, n):
            e = rng.standard_normal(2)
            x[t] = 0.6 * x[t - 1] + 0.3 * y[t - 1] + e[0]
            y[t] = 0.2 * x[t - 1] + 0.5 * y[t - 1] + e[1]
        result = pccf(x, y, nlags=8)
        threshold = 2.0 / np.sqrt(n)
        assert np.abs(result[0]) > threshold
        assert np.all(np.abs(result[3:]) < 3 * threshold)

    def test_independent_series(self):
        rng = np.random.default_rng(54321)
        x = rng.standard_normal(200)
        y = rng.standard_normal(200)
        result = pccf(x, y, nlags=10)
        threshold = 2.0 / np.sqrt(200)
        assert np.all(np.abs(result) < 3 * threshold)

    def test_yw_ols_agreement_stationary(self):
        rng = np.random.default_rng(11111)
        n = 2000
        x = np.zeros(n)
        y = np.zeros(n)
        x[0] = rng.standard_normal()
        y[0] = rng.standard_normal()
        for t in range(1, n):
            e = rng.standard_normal(2)
            x[t] = 0.5 * x[t - 1] + 0.2 * y[t - 1] + e[0]
            y[t] = 0.3 * x[t - 1] + 0.4 * y[t - 1] + e[1]
        yw = pccf(x, y, nlags=5, method="ywm")
        ols = pccf(x, y, nlags=5, method="ols")
        assert_almost_equal(yw, ols, decimal=2)

    def test_method_parameter_validation(self):
        with pytest.raises(ValueError):
            pccf(self.x, self.y, nlags=5, method="invalid")

    def test_yw_method_aliases(self):
        ywm = pccf(self.x, self.y, nlags=5, method="ywm")
        ywmle = pccf(self.x, self.y, nlags=5, method="ywmle")
        yw_mle = pccf(self.x, self.y, nlags=5, method="yw_mle")
        assert_almost_equal(ywm, ywmle, DECIMAL_8)
        assert_almost_equal(ywm, yw_mle, DECIMAL_8)

        yw = pccf(self.x, self.y, nlags=5, method="yw")
        ywa = pccf(self.x, self.y, nlags=5, method="ywa")
        ywadjusted = pccf(self.x, self.y, nlags=5, method="ywadjusted")
        yw_adjusted = pccf(self.x, self.y, nlags=5, method="yw_adjusted")
        assert_almost_equal(yw, ywa, DECIMAL_8)
        assert_almost_equal(yw, ywadjusted, DECIMAL_8)
        assert_almost_equal(yw, yw_adjusted, DECIMAL_8)

    def test_yw_analytical_var1(self):
        from scipy.linalg import solve_discrete_lyapunov

        A = np.array([[0.6, 0.3], [0.2, 0.5]])
        Q = np.eye(2)
        G0 = solve_discrete_lyapunov(A, Q)

        nlags = 3
        gamma = [G0]
        for h in range(1, nlags + 1):
            gamma.append(G0 @ np.linalg.matrix_power(A.T, h))

        sig_f = G0.copy()
        sig_b = G0.copy()
        phi_prev = [None] * (nlags + 1)
        psi_prev = [None] * (nlags + 1)
        expected = np.empty(nlags)

        for s in range(1, nlags + 1):
            delta_f = gamma[s].copy()
            delta_b = gamma[s].T.copy()
            for j in range(1, s):
                delta_f -= phi_prev[j] @ gamma[s - j]
                delta_b -= psi_prev[j] @ gamma[s - j].T
            d_f = np.sqrt(np.diag(sig_f))
            d_b = np.sqrt(np.diag(sig_b))
            expected[s - 1] = delta_f[0, 1] / (d_f[0] * d_b[1])
            phi_ss = delta_f @ np.linalg.inv(sig_b)
            psi_ss = delta_b @ np.linalg.inv(sig_f)
            phi_new = [None] * (nlags + 1)
            psi_new = [None] * (nlags + 1)
            phi_new[s] = phi_ss
            psi_new[s] = psi_ss
            for j in range(1, s):
                phi_new[j] = phi_prev[j] - phi_ss @ psi_prev[s - j]
                psi_new[j] = psi_prev[j] - psi_ss @ phi_prev[s - j]
            sig_f = sig_f - phi_ss @ delta_b
            sig_b = sig_b - psi_ss @ delta_f
            phi_prev = phi_new
            psi_prev = psi_new

        assert_almost_equal(expected[1:], np.zeros(nlags - 1), DECIMAL_8)

        rng = np.random.default_rng(77777)
        n = 5000
        z = np.zeros((n, 2))
        z[0] = rng.standard_normal(2)
        for t in range(1, n):
            z[t] = A @ z[t - 1] + rng.standard_normal(2)
        result = pccf(z[:, 0], z[:, 1], nlags=nlags, method="ywm")
        assert_almost_equal(result[0], expected[0], decimal=1)
        assert_almost_equal(result[1:], expected[1:], decimal=1)

    def test_asymmetry(self):
        rng = np.random.default_rng(42)
        n = 500
        x = np.zeros(n)
        y = np.zeros(n)
        for t in range(1, n):
            e = rng.standard_normal(2)
            x[t] = 0.6 * x[t - 1] + 0.3 * y[t - 1] + e[0]
            y[t] = 0.2 * x[t - 1] + 0.5 * y[t - 1] + e[1]
        fwd = pccf(x, y, nlags=3)
        rev = pccf(y, x, nlags=3)
        assert not np.allclose(fwd, rev)

    def test_nan_input_raises(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = np.arange(8, dtype=float)
        with pytest.raises(MissingDataError):
            pccf(x, y, nlags=3)
        with pytest.raises(MissingDataError):
            pccf(y, x, nlags=3, method="ols")

    def test_inf_input_raises(self):
        x = np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = np.arange(8, dtype=float)
        with pytest.raises(MissingDataError):
            pccf(x, y, nlags=3)

    def test_ols_underdetermined_returns_nan(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(30)
        y = rng.standard_normal(30)
        result = pccf(x, y, nlags=15, method="ols")
        for h in range(1, 16):
            n_obs = 30 - h
            n_cols = 2 * (h - 1) + 1
            if n_obs <= n_cols and h > 1:
                assert np.isnan(
                    result[h - 1]
                ), f"lag {h}: {n_obs} obs, {n_cols} cols should be NaN"


class TestBlockJackknife:
    """
    Test block (delete-k) jackknife estimator
    """

    def test_mean_closed_form(self):
        # Leave-one-out jackknife applied to the sample mean should exactly
        # reproduce the standard unbiased variance formula: Var(xbar) = s^2/n
        rng = np.random.default_rng(42)
        x = rng.normal(loc=10, scale=2, size=100)

        result = block_jackknife(x, np.mean, n_blocks=-1)
        theta_jack, se = result.theta_jack, result.se

        expected_se = np.sqrt(np.var(x, ddof=1) / len(x))

        assert_allclose(se, expected_se, rtol=1e-10)
        assert_almost_equal(theta_jack, np.mean(x), DECIMAL_6)

    def test_hand_computed(self):
        # Matches the manually-verified 3-block case:
        # x = [0..9], n_blocks=3
        x = np.arange(10)

        result = block_jackknife(x, np.mean, n_blocks=3)
        theta_jack, se = result.theta_jack, result.se

        assert_almost_equal(theta_jack, 4.309523809523809, DECIMAL_8)
        assert_almost_equal(se, 2.044294088920541, DECIMAL_8)

    def test_acf_runs(self):
        # Reproduces the motivating AR(1)/ACF use case from the original
        # feature request; checks shape and finiteness, not exact values,
        # since there is no independent closed-form to check ACF jackknife
        # SEs against.
        rng = np.random.default_rng(0)
        n = 5000
        phi = 0.8
        noise = rng.normal(scale=1.0, size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + noise[t]

        max_lag = 50
        n_blocks = n // 20

        def est(arr):
            return acf(arr, nlags=max_lag)

        result = block_jackknife(x, est, n_blocks=n_blocks)
        rho, rho_se = result.theta_jack, result.se

        assert rho.shape == (max_lag + 1,)
        assert rho_se.shape == (max_lag + 1,)
        assert np.all(np.isfinite(rho))
        assert np.all(np.isfinite(rho_se))
        assert np.all(rho_se >= 0)

    def test_invalid_n_blocks(self):
        x = np.arange(10)
        with pytest.raises(ValueError, match="n_blocks must be greater than 1"):
            block_jackknife(x, np.mean, n_blocks=1)
        with pytest.raises(ValueError, match="n_blocks cannot exceed"):
            block_jackknife(x, np.mean, n_blocks=20)

    def test_non_callable_statistic(self):
        x = np.arange(10)
        with pytest.raises(ValueError, match="must be callable"):
            block_jackknife(x, "not_a_function", n_blocks=2)


class TestBreakvarHeteroskedasticityTest:
    from scipy.stats import chi2, f

    def test_1d_input(self):

        input_residuals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        expected_statistic = (4.0**2 + 5.0**2) / (0.0**2 + 1.0**2)
        # ~ F(2, 2), two-sided test
        expected_pvalue = 2 * min(
            self.f.cdf(expected_statistic, 2, 2),
            self.f.sf(expected_statistic, 2, 2),
        )
        _result = breakvar_heteroskedasticity_test(input_residuals)
        actual_statistic, actual_pvalue = _result.statistic, _result.pvalue

        assert actual_statistic == expected_statistic
        assert actual_pvalue == expected_pvalue

    def test_2d_input_with_missing_values(self):

        input_residuals = np.array(
            [
                [0.0, 0.0, np.nan],
                [1.0, np.nan, 1.0],
                [2.0, 2.0, np.nan],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0],
                [7.0, 7.0, 7.0],
                [8.0, 8.0, 8.0],
            ]
        )
        expected_statistic = np.array(
            [
                (8.0**2 + 7.0**2 + 6.0**2) / (0.0**2 + 1.0**2 + 2.0**2),
                (8.0**2 + 7.0**2 + 6.0**2) / (0.0**2 + 2.0**2),
                np.nan,
            ]
        )
        # H(h) is a ratio of sums of squares, so it is F(dfn, dfd) only after
        # rescaling by dfd / dfn.  Column 1 has an unbalanced (3, 2) split.
        expected_pvalue = np.array(
            [
                2
                * min(
                    self.f.cdf(expected_statistic[0] * 3 / 3, 3, 3),
                    self.f.sf(expected_statistic[0] * 3 / 3, 3, 3),
                ),
                2
                * min(
                    self.f.cdf(expected_statistic[1] * 2 / 3, 3, 2),
                    self.f.sf(expected_statistic[1] * 2 / 3, 3, 2),
                ),
                np.nan,
            ]
        )
        _result = breakvar_heteroskedasticity_test(input_residuals)
        actual_statistic, actual_pvalue = _result.statistic, _result.pvalue

        assert_equal(actual_statistic, expected_statistic)
        assert_equal(actual_pvalue, expected_pvalue)

    @pytest.mark.parametrize("use_f", [True, False])
    @pytest.mark.parametrize(
        "alternative", ["increasing", "decreasing", "two-sided"]
    )
    def test_unbalanced_degrees_of_freedom(self, alternative, use_f):
        # When missing values leave the two subsets with different numbers of
        # usable residuals, H(h) -- a ratio of *sums* of squares -- must be
        # rescaled by denom_dof / numer_dof before it is F(dfn, dfd), and
        # inverting it for "decreasing" must swap the two degrees of freedom.
        resid = np.array([np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        # h = 3; numerator {7, 8, 9} -> dfn = 3, denominator {2, 3} -> dfd = 2
        stat = (7.0**2 + 8.0**2 + 9.0**2) / (2.0**2 + 3.0**2)
        dfn, dfd = 3, 2
        if alternative == "decreasing":
            stat = 1.0 / stat
            dfn, dfd = dfd, dfn
        if use_f:
            scaled, dist, args = stat * dfd / dfn, self.f, (dfn, dfd)
        else:
            scaled, dist, args = stat * dfd, self.chi2, (dfn,)
        if alternative == "two-sided":
            expected = 2 * min(dist.cdf(scaled, *args), dist.sf(scaled, *args))
        else:
            expected = dist.sf(scaled, *args)

        result = breakvar_heteroskedasticity_test(
            resid, alternative=alternative, use_f=use_f
        )
        assert_allclose(result.statistic, stat)
        assert_allclose(result.pvalue, expected)

    @pytest.mark.parametrize(
        "resid",
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ],
        ids=["balanced", "unbalanced"],
    )
    def test_one_sided_alternatives_are_complementary(self, resid):
        # The exact (F) versions of the two one-sided tests are opposite tails
        # of the same statistic, so their p-values sum to one -- including
        # when missing values make the degrees of freedom unbalanced, because
        # 1 / F(dfn, dfd) is F(dfd, dfn).  The unscaled ratio of sums does not
        # have this property.  It does not hold for use_f=False: the two chi2
        # approximations are different limits (dfd -> oo and dfn -> oo).
        resid = np.asarray(resid)
        up = breakvar_heteroskedasticity_test(
            resid, alternative="increasing"
        ).pvalue
        down = breakvar_heteroskedasticity_test(
            resid, alternative="decreasing"
        ).pvalue
        assert_allclose(up + down, 1.0)

    @pytest.mark.parametrize(
        "subset_length,expected_statistic,expected_pvalue",
        [
            (2, 41, 2 * min(f.cdf(41, 2, 2), f.sf(41, 2, 2))),
            (0.5, 10, 2 * min(f.cdf(10, 3, 3), f.sf(10, 3, 3))),
        ],
    )
    def test_subset_length(self, subset_length, expected_statistic, expected_pvalue):

        input_residuals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        _result = breakvar_heteroskedasticity_test(
            input_residuals,
            subset_length=subset_length,
        )
        actual_statistic, actual_pvalue = _result.statistic, _result.pvalue

        assert actual_statistic == expected_statistic
        assert actual_pvalue == expected_pvalue

    @pytest.mark.parametrize(
        "alternative,expected_statistic,expected_pvalue",
        [
            ("two-sided", 41, 2 * min(f.cdf(41, 2, 2), f.sf(41, 2, 2))),
            ("decreasing", 1 / 41, f.sf(1 / 41, 2, 2)),
            ("increasing", 41, f.sf(41, 2, 2)),
        ],
    )
    def test_alternative(self, alternative, expected_statistic, expected_pvalue):

        input_residuals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        _result = breakvar_heteroskedasticity_test(
            input_residuals,
            alternative=alternative,
        )
        actual_statistic, actual_pvalue = _result.statistic, _result.pvalue
        assert actual_statistic == expected_statistic
        assert actual_pvalue == expected_pvalue

    @pytest.mark.parametrize(
        "alias,canonical",
        [("2", "two-sided"), ("d", "decreasing"), ("i", "increasing")],
    )
    def test_alternative_deprecated_alias(self, alias, canonical):
        # undocumented short forms still work but warn, and are equivalent
        # to spelling out the documented alternative (case-insensitively)
        input_residuals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        with pytest.warns(FutureWarning, match="is a deprecated alias"):
            alias_result = breakvar_heteroskedasticity_test(
                input_residuals, alternative=alias.upper()
            )
        canonical_result = breakvar_heteroskedasticity_test(
            input_residuals, alternative=canonical
        )
        assert_allclose(alias_result, canonical_result)

        with pytest.raises(ValueError, match="alternative must be one of"):
            breakvar_heteroskedasticity_test(input_residuals, alternative="bogus")

    def test_use_chi2(self):

        input_residuals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        expected_statistic = (4.0**2 + 5.0**2) / (0.0**2 + 1.0**2)
        expected_pvalue = 2 * min(
            self.chi2.cdf(2 * expected_statistic, 2),
            self.chi2.sf(2 * expected_statistic, 2),
        )
        _result = breakvar_heteroskedasticity_test(
            input_residuals,
            use_f=False,
        )
        actual_statistic, actual_pvalue = _result.statistic, _result.pvalue
        assert actual_statistic == expected_statistic
        assert actual_pvalue == expected_pvalue


class CheckCoint:
    """
    Test Cointegration Test Results for 2-variable system

    Test values taken from Stata
    """

    levels = ["1%", "5%", "10%"]
    data = macrodata.load_pandas()
    y1 = data.data["realcons"].values
    y2 = data.data["realgdp"].values

    def test_tstat(self):
        assert_almost_equal(self.coint_t, self.teststat, DECIMAL_4)


# this does not produce the old results anymore
class TestCoint_t(CheckCoint):
    """
    Get AR(1) parameter on residuals
    """

    @classmethod
    def setup_class(cls):
        # cls.coint_t = coint(cls.y1, cls.y2, trend="c")[0]
        cls.coint_t = coint(cls.y1, cls.y2, trend="c", maxlag=0, autolag=None).coint_t
        cls.teststat = -1.8208817
        cls.teststat = -1.830170986148


def test_coint():
    nobs = 200
    scale_e = 1
    const = [1, 0, 0.5, 0]
    rs = np.random.RandomState(123)
    unit = rs.randn(nobs).cumsum()
    y = scale_e * rs.randn(nobs, 4)
    y[:, :2] += unit[:, None]
    y += const
    y = np.round(y, 4)

    # FIXME: enable/xfail/skip or delete
    for trend in []:  # ['c', 'ct', 'ctt', 'n']:
        print("\n", trend)
        print(coint(y[:, 0], y[:, 1], trend=trend, maxlag=4, autolag=None))
        print(coint(y[:, 0], y[:, 1:3], trend=trend, maxlag=4, autolag=None))
        print(coint(y[:, 0], y[:, 2:], trend=trend, maxlag=4, autolag=None))
        print(coint(y[:, 0], y[:, 1:], trend=trend, maxlag=4, autolag=None))

    # results from Stata egranger
    res_egranger = {}
    # trend = 'ct'
    res = res_egranger["ct"] = {}
    res[0] = [
        -5.615251442239,
        -4.406102369132,
        -3.82866685109,
        -3.532082997903,
    ]
    res[1] = [
        -5.63591313706,
        -4.758609717199,
        -4.179130554708,
        -3.880909696863,
    ]
    res[2] = [
        -2.892029275027,
        -4.758609717199,
        -4.179130554708,
        -3.880909696863,
    ]
    res[3] = [-5.626932544079, -5.08363327039, -4.502469783057, -4.2031051091]

    # trend = 'c'
    res = res_egranger["c"] = {}
    # first critical value res[0][1] has a discrepancy starting at 4th decimal
    res[0] = [
        -5.760696844656,
        -3.952043522638,
        -3.367006313729,
        -3.065831247948,
    ]
    # manually adjusted to have higher precision as in other cases
    res[0][1] = -3.952321293401682
    res[1] = [
        -5.781087068772,
        -4.367111915942,
        -3.783961136005,
        -3.483501524709,
    ]
    res[2] = [
        -2.477444137366,
        -4.367111915942,
        -3.783961136005,
        -3.483501524709,
    ]
    res[3] = [
        -5.778205811661,
        -4.735249216434,
        -4.152738973763,
        -3.852480848968,
    ]

    # trend = 'ctt'
    res = res_egranger["ctt"] = {}
    res[0] = [
        -5.644431269946,
        -4.796038299708,
        -4.221469431008,
        -3.926472577178,
    ]
    res[1] = [-5.665691609506, -5.111158174219, -4.53317278104, -4.23601008516]
    res[2] = [-3.161462374828, -5.111158174219, -4.53317278104, -4.23601008516]
    res[3] = [
        -5.657904558563,
        -5.406880189412,
        -4.826111619543,
        -4.527090164875,
    ]

    # The following for 'n' are only regression test numbers
    # trend = 'n' not allowed in egranger
    # trend = 'n'
    res = res_egranger["n"] = {}
    nan = np.nan  # shortcut for table
    res[0] = [-3.7146175989071137, nan, nan, nan]
    res[1] = [-3.8199323012888384, nan, nan, nan]
    res[2] = [-1.6865000791270679, nan, nan, nan]
    res[3] = [-3.7991270451873675, nan, nan, nan]

    for trend in ["c", "ct", "ctt", "n"]:
        res1 = {}
        res1[0] = coint(y[:, 0], y[:, 1], trend=trend, maxlag=4, autolag=None)
        res1[1] = coint(y[:, 0], y[:, 1:3], trend=trend, maxlag=4, autolag=None)
        res1[2] = coint(y[:, 0], y[:, 2:], trend=trend, maxlag=4, autolag=None)
        res1[3] = coint(y[:, 0], y[:, 1:], trend=trend, maxlag=4, autolag=None)

        for i in range(4):
            res = res_egranger[trend]

            assert_allclose(res1[i].coint_t, res[i][0], rtol=1e-11)
            r2 = res[i][1:]
            r1 = res1[i].critical_values
            assert_allclose(r1, r2, rtol=0, atol=6e-7)

    # use default autolag #4490
    res1_0 = coint(y[:, 0], y[:, 1], trend="ct", maxlag=4)
    assert_allclose(res1_0.critical_values, res_egranger["ct"][0][1:], rtol=0, atol=6e-7)
    # the following is just a regression test
    assert_allclose(
        [res1_0.coint_t, res1_0.pvalue],
        [-13.992946638547112, 2.270898990540678e-27],
        rtol=1e-10,
        atol=1e-27,
    )


def test_coint_identical_series():
    nobs = 200
    scale_e = 1
    rs = np.random.RandomState(123)
    y = scale_e * rs.randn(nobs)
    warnings.simplefilter("always", CollinearityWarning)
    with pytest.warns(CollinearityWarning):
        c = coint(y, y, trend="c", maxlag=0, autolag=None)
    assert_equal(c.pvalue, 0.0)
    assert np.isneginf(c.coint_t)


def test_coint_perfect_collinearity():
    # test uses nearly perfect collinearity
    nobs = 200
    scale_e = 1
    rs = np.random.RandomState(123)
    x = scale_e * rs.randn(nobs, 2)
    y = 1 + x.sum(axis=1) + 1e-7 * rs.randn(nobs)
    warnings.simplefilter("always", CollinearityWarning)
    with warnings.catch_warnings(record=True):
        c = coint(y, x, trend="c", maxlag=0, autolag=None)
    assert_equal(c.pvalue, 0.0)
    assert np.isneginf(c.coint_t)


class TestGrangerCausality:
    def test_grangercausality(self):
        # some example data
        mdata = macrodata.load_pandas().data
        mdata = mdata[["realgdp", "realcons"]].values
        data = mdata.astype(float)
        data = np.diff(np.log(data), axis=0)

        # R: lmtest:grangertest
        r_result = [0.243097, 0.7844328, 195, 2]  # f_test
        gr = grangercausalitytests(data[:, 1::-1], 2)
        assert_almost_equal(r_result, gr[2][0]["ssr_ftest"], decimal=7)
        assert_almost_equal(gr[2][0]["params_ftest"], gr[2][0]["ssr_ftest"], decimal=7)

    def test_grangercausality_single(self):
        mdata = macrodata.load_pandas().data
        mdata = mdata[["realgdp", "realcons"]].values
        data = mdata.astype(float)
        data = np.diff(np.log(data), axis=0)
        gr = grangercausalitytests(data[:, 1::-1], 2)
        gr2 = grangercausalitytests(data[:, 1::-1], [2])
        assert 1 in gr
        assert 1 not in gr2
        assert_almost_equal(gr[2][0]["ssr_ftest"], gr2[2][0]["ssr_ftest"], decimal=7)
        assert_almost_equal(gr[2][0]["params_ftest"], gr2[2][0]["ssr_ftest"], decimal=7)

    def test_granger_fails_on_nobs_check(self):
        # Test that if maxlag is too large, Granger Test raises a clear error.
        rs = np.random.RandomState(3239291)
        x = rs.rand(10, 2)
        grangercausalitytests(x, 2)  # This should pass.

    def test_granger_fails_on_finite_check(self):
        rs = np.random.RandomState(1234)
        x = rs.rand(1000, 2)
        x[500, 0] = np.nan
        x[750, 1] = np.inf
        with pytest.raises(ValueError, match="x contains NaN"):
            grangercausalitytests(x, 2)

    def test_granger_fails_on_zero_lag(self):
        rs = np.random.RandomState(388776)
        x = rs.rand(1000, 2)
        with pytest.raises(
            ValueError,
            match="maxlag must be a non-empty list containing only positive integers",
        ):
            grangercausalitytests(x, [0, 1, 2])


class TestKPSS:
    """
    R-code
    ------
    library(tseries)
    kpss.stat(x, "Level")
    kpss.stat(x, "Trend")

    In this context, x is the vector containing the
    macrodata['realgdp'] series.
    """

    def setup_method(self):
        self.data = macrodata.load_pandas()
        self.x = self.data.data["realgdp"].values

    def test_fail_nonvector_input(self):
        # should be fine
        with pytest.warns(InterpolationWarning):
            kpss(self.x, nlags="legacy", result_object=False)

        x = rs.rand(20, 2)
        with pytest.raises(ValueError):
            kpss(x)

    def test_fail_invalid_nlags_string(self):
        with pytest.raises(ValueError, match="nlags"):
            kpss(self.x, nlags="invalid", result_object=False)

    def test_fail_unclear_hypothesis(self):
        # these should be fine,
        with pytest.warns(InterpolationWarning):
            kpss(self.x, "c", nlags="legacy", result_object=False)
        with pytest.warns(InterpolationWarning):
            kpss(self.x, "C", nlags="legacy", result_object=False)
        with pytest.warns(InterpolationWarning):
            kpss(self.x, "ct", nlags="legacy", result_object=False)
        with pytest.warns(InterpolationWarning):
            kpss(self.x, "CT", nlags="legacy", result_object=False)

        with pytest.raises(ValueError):
            kpss(self.x, "unclear hypothesis", nlags="legacy")

    def test_teststat(self):
        with pytest.warns(InterpolationWarning):
            kpss_stat, _, _, _ = kpss(self.x, "c", 3, result_object=False)
        assert_almost_equal(kpss_stat, 5.0169, DECIMAL_3)

        with pytest.warns(InterpolationWarning):
            kpss_stat, _, _, _ = kpss(self.x, "ct", 3, result_object=False)
        assert_almost_equal(kpss_stat, 1.1828, DECIMAL_3)

    def test_pval(self):
        with pytest.warns(InterpolationWarning):
            _, pval, _, _ = kpss(self.x, "c", 3, result_object=False)
        assert_equal(pval, 0.01)

        with pytest.warns(InterpolationWarning):
            _, pval, _, _ = kpss(self.x, "ct", 3, result_object=False)
        assert_equal(pval, 0.01)

    def test_store(self):
        with pytest.warns(InterpolationWarning):
            _, _, _, store = kpss(self.x, "c", 3, True, result_object=False)

        # assert attributes, and make sure they're correct
        assert_equal(store.nobs, len(self.x))
        assert_equal(store.lags, 3)

    # test autolag function _kpss_autolag against SAS 9.3
    def test_lags(self):
        # real GDP from macrodata data set
        with pytest.warns(InterpolationWarning):
            res = kpss(self.x, "c", nlags="auto", result_object=False)
        assert_equal(res[2], 9)
        # real interest rates from macrodata data set
        res = kpss(
            sunspots.load().data["SUNACTIVITY"],
            "c",
            nlags="auto",
            result_object=False,
        )
        assert_equal(res[2], 7)
        # volumes from nile data set
        with pytest.warns(InterpolationWarning):
            res = kpss(
                nile.load().data["volume"], "c", nlags="auto", result_object=False
            )
        assert_equal(res[2], 5)
        # log-coinsurance from randhie data set
        with pytest.warns(InterpolationWarning):
            res = kpss(
                randhie.load().data["lncoins"],
                "ct",
                nlags="auto",
                result_object=False,
            )
        assert_equal(res[2], 75)
        # in-vehicle time from modechoice data set
        with pytest.warns(InterpolationWarning):
            res = kpss(
                modechoice.load().data["invt"],
                "ct",
                nlags="auto",
                result_object=False,
            )
        assert_equal(res[2], 18)

    def test_kpss_fails_on_nobs_check(self):
        # Test that if lags exceeds number of observations KPSS raises a
        # clear error
        # GH5925
        nobs = len(self.x)
        msg = rf"lags \({nobs}\) must be < number of observations \({nobs}\)"
        with pytest.raises(ValueError, match=msg):
            kpss(self.x, "c", nlags=nobs)

    def test_kpss_autolags_does_not_assign_lags_equal_to_nobs(self):
        # Test that if *autolags* exceeds number of observations, we set
        # suitable lags
        # GH5925
        base = np.array([0, 0, 0, 0, 0, 1, 1.0])
        data_which_breaks_autolag = np.r_[np.tile(base, 297 // 7), [0, 0, 0]]
        kpss(data_which_breaks_autolag, nlags="auto", result_object=False)

    def test_legacy_lags(self):
        # Test legacy lags are the same
        with pytest.warns(InterpolationWarning):
            res = kpss(self.x, "c", nlags="legacy", result_object=False)
        assert_equal(res[2], 15)

    def test_unknown_lags(self):
        # Test legacy lags are the same
        with pytest.raises(ValueError):
            kpss(self.x, "c", nlags="unknown")

    def test_none(self):
        with pytest.raises(ValueError, match="None is not a valid value"):
            kpss(self.x, nlags=None)

    def test_result_object_default_warns(self):
        with pytest.warns(InterpolationWarning, match="The test statistic is"):
            with pytest.warns(FutureWarning, match="result_object"):
                res = kpss(self.x, "c", nlags=3)
        assert not isinstance(res, KPSSResult)

    def test_result_object_false_silences_warning(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=FutureWarning)
            with pytest.warns(InterpolationWarning, match="The test statistic is"):
                res = kpss(self.x, "c", nlags=3, result_object=False)
        assert not isinstance(res, KPSSResult)

    def test_result_object_true_returns_result_object(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=FutureWarning)
            with pytest.warns(InterpolationWarning, match="The test statistic is"):
                res = kpss(self.x, "c", nlags=3, result_object=True)
        assert isinstance(res, KPSSResult)
        assert res.resstore is None
        assert res[0] == res.statistic
        assert res[1] == res.pvalue
        assert res.lags == 3
        assert isinstance(res.critical_values, dict)

    def test_result_object_true_with_store(self):
        with pytest.warns(InterpolationWarning, match="The test statistic is"):
            res = kpss(self.x, "c", nlags=3, store=True, result_object=True)
        assert isinstance(res, KPSSResult)
        assert res.lags == 3
        assert res.resstore is not None
        assert res.resstore.nobs == len(self.x)
        assert res.resstore.lags == 3


class TestRUR:
    """
    Simple implementation
    ------
    Since an R implementation of the test cannot be found, the method is tested against
    a simple implementation using a for loop.
    In this context, x is the vector containing the
    macrodata['realgdp'] series.
    """

    def setup_method(self):
        self.data = macrodata.load_pandas()
        self.x = self.data.data["realgdp"].values

    # To be removed when range unit test gets an R implementation
    def simple_rur(self, x, store=False):
        x = array_like(x, "x")
        store = bool_like(store, "store")

        nobs = x.shape[0]

        # if m is not one, n != m * n
        if nobs != x.size:
            raise ValueError(f"x of shape {x.shape} not understood")

        # Table from [1] has been replicated using 200,000 samples
        # Critical values for new n_obs values have been identified
        pvals = [0.01, 0.025, 0.05, 0.10, 0.90, 0.95]
        n = np.array([25, 50, 100, 150, 200, 250, 500, 1000, 2000, 3000, 4000, 5000])
        crit = np.array(
            [
                [0.6626, 0.8126, 0.9192, 1.0712, 2.4863, 2.7312],
                [0.7977, 0.9274, 1.0478, 1.1964, 2.6821, 2.9613],
                [0.907, 1.0243, 1.1412, 1.2888, 2.8317, 3.1393],
                [0.9543, 1.0768, 1.1869, 1.3294, 2.8915, 3.2049],
                [0.9833, 1.0984, 1.2101, 1.3494, 2.9308, 3.2482],
                [0.9982, 1.1137, 1.2242, 1.3632, 2.9571, 3.2482],
                [1.0494, 1.1643, 1.2712, 1.4076, 3.0207, 3.3584],
                [1.0846, 1.1959, 1.2988, 1.4344, 3.0653, 3.4073],
                [1.1121, 1.2200, 1.3230, 1.4556, 3.0948, 3.4439],
                [1.1204, 1.2295, 1.3318, 1.4656, 3.1054, 3.4632],
                [1.1309, 1.2347, 1.3318, 1.4693, 3.1165, 3.4717],
                [1.1377, 1.2402, 1.3408, 1.4729, 3.1252, 3.4807],
            ]
        )

        # Interpolation for nobs
        inter_crit = np.zeros((1, crit.shape[1]))
        for i in range(crit.shape[1]):
            f = interp1d(n, crit[:, i])
            inter_crit[0, i] = f(nobs)

        # Calculate RUR stat
        count = 0

        max_p = x[0]
        min_p = x[0]

        for v in x[1:]:
            if v > max_p:
                max_p = v
                count = count + 1
            if v < min_p:
                min_p = v
                count = count + 1

        rur_stat = count / np.sqrt(len(x))

        k = len(pvals) - 1
        for i in range(len(pvals) - 1, -1, -1):
            if rur_stat < inter_crit[0, i]:
                k = i
            else:
                break

        p_value = pvals[k]

        warn_msg = """\
        The test statistic is outside of the range of p-values available in the
        look-up table. The actual p-value is {direction} than the p-value returned.
        """
        direction = ""
        if p_value == pvals[-1]:
            direction = "smaller"
        elif p_value == pvals[0]:
            direction = "larger"

        if direction:
            warnings.warn(
                warn_msg.format(direction=direction), InterpolationWarning, stacklevel=2
            )

        crit_dict = {
            "10%": inter_crit[0, 3],
            "5%": inter_crit[0, 2],
            "2.5%": inter_crit[0, 1],
            "1%": inter_crit[0, 0],
        }

        if store:
            from statsmodels.stats.diagnostic import ResultsStore

            rstore = ResultsStore()
            rstore.nobs = nobs

            rstore.H0 = "The series is not stationary"
            rstore.HA = "The series is stationary"

            return rur_stat, p_value, crit_dict, rstore
        else:
            return rur_stat, p_value, crit_dict

    def test_fail_nonvector_input(self):
        with pytest.warns(InterpolationWarning):
            range_unit_root_test(self.x, result_object=False)

        rs = np.random.RandomState(8474768)
        x = rs.rand(20, 2)
        with pytest.raises(ValueError):
            range_unit_root_test(x, result_object=False)

    def test_teststat(self):
        with pytest.warns(InterpolationWarning):
            rur_stat, _, _ = range_unit_root_test(self.x, result_object=False)
        simple_rur_stat, _, _ = self.simple_rur(self.x)
        assert_almost_equal(rur_stat, simple_rur_stat, DECIMAL_3)

    def test_pval(self):
        with pytest.warns(InterpolationWarning):
            _, pval, _ = range_unit_root_test(self.x, result_object=False)
        _, simple_pval, _ = self.simple_rur(self.x)
        assert_equal(pval, simple_pval)

    def test_store(self):
        with pytest.warns(InterpolationWarning):
            result = range_unit_root_test(self.x, True, result_object=False)
        _, _, _, store = result

        # assert attributes, and make sure they're correct
        assert_equal(store.nobs, len(self.x))

    def test_result_object_true_returns_result_object(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=FutureWarning)
            warnings.filterwarnings("ignore", category=InterpolationWarning)
            res = range_unit_root_test(self.x, result_object=True)
        assert isinstance(res, RURResult)
        assert res.resstore is None
        assert res[0] == res.statistic
        assert res[1] == res.pvalue
        assert isinstance(res.critical_values, dict)

    def test_result_object_true_with_store(self):
        with pytest.warns(InterpolationWarning, match="The test statistic is"):
            res = range_unit_root_test(self.x, store=True, result_object=True)
        assert isinstance(res, RURResult)
        assert res.resstore is not None
        assert res.resstore.nobs == len(self.x)


def test_pandasacovf():
    s = Series(lrange(1, 11))
    assert_almost_equal(acovf(s, fft=False), acovf(s.values, fft=False))


def test_acovf2d():
    dta = sunspots.load_pandas().data
    dta.index = date_range(start="1700", end="2009", freq=YEAR_END)[:309]
    del dta["YEAR"]
    res = acovf(dta, fft=False)
    assert_equal(res, acovf(dta.values, fft=False))
    rs = np.random.RandomState(992333)
    x = rs.random((10, 2))
    with pytest.raises(ValueError):
        acovf(x, fft=False)


@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("adjusted", [True, False])
def test_acovf_fft_vs_convolution(demean, adjusted):
    rs = np.random.RandomState(83747)
    q = rs.normal(size=100)

    F1 = acovf(q, demean=demean, adjusted=adjusted, fft=True)
    F2 = acovf(q, demean=demean, adjusted=adjusted, fft=False)
    assert_almost_equal(F1, F2, decimal=7)


@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("adjusted", [True, False])
def test_ccovf_fft_vs_convolution(demean, adjusted):
    rs = np.random.RandomState(3843983)
    x = rs.normal(size=128)
    y = rs.normal(size=128)

    F1 = ccovf(x, y, demean=demean, adjusted=adjusted, fft=False)
    F2 = ccovf(x, y, demean=demean, adjusted=adjusted, fft=True)
    assert_almost_equal(F1, F2, decimal=7)


@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("adjusted", [True, False])
@pytest.mark.parametrize("fft", [True, False])
def test_compare_acovf_vs_ccovf(demean, adjusted, fft):
    rs = np.random.RandomState(14523)
    x = rs.normal(size=128)

    F1 = acovf(x, demean=demean, adjusted=adjusted, fft=fft)
    F2 = ccovf(x, x, demean=demean, adjusted=adjusted, fft=fft)
    assert_almost_equal(F1, F2, decimal=7)


@pytest.mark.parametrize("adjusted", [True, False])
@pytest.mark.parametrize("fft", [True, False])
def test_ccovf_different_lengths(adjusted, fft):
    # Regression test for GH#9565
    # ccovf crashed when len(x) != len(y) and adjusted=True
    rs = np.random.RandomState(98765)
    x = rs.normal(size=200)
    y = rs.normal(size=150)

    result = ccovf(x, y, adjusted=adjusted, fft=fft)
    # Output should always have length len(x)
    assert result.shape == (200,)
    assert np.all(np.isfinite(result))

    # Also test when len(x) < len(y)
    result2 = ccovf(y, x, adjusted=adjusted, fft=fft)
    assert result2.shape == (150,)
    assert np.all(np.isfinite(result2))


def test_ccovf_different_lengths_known_lag():
    # Verify that ccovf correctly identifies a known lag
    # when the arrays have different lengths (GH#9565)
    rs = np.random.RandomState(54321)
    x = rs.normal(size=200)
    # y is x shifted by 5 positions, but shorter
    y = x[5:180] + rs.normal(size=175) * 0.05

    result = ccovf(x, y, adjusted=False)
    # Peak should be at lag 5
    assert result.shape == (200,)
    assert np.argmax(result[:50]) == 5


@pytest.mark.parametrize("fft", [True, False])
def test_ccovf_adjusted_shorter_y(fft):
    # GH#9565 follow-up: with len(y) < len(x), adjusted=True must divide by
    # the number of overlapping observations, min(len(y), len(x) - k), not
    # by len(x) - k.
    x = np.ones(6)
    y = np.ones(2)

    # Every product is exactly 1.0, so every adjusted average must be 1.0.
    result = ccovf(x, y, adjusted=True, demean=False, fft=fft)
    assert_allclose(result, np.ones(6))

    # The unadjusted path divides by len(x) throughout, by definition, so the
    # overlap counts min(2, 6 - k) show through unscaled.
    unadjusted = ccovf(x, y, adjusted=False, demean=False, fft=fft)
    assert_allclose(unadjusted, np.array([2, 2, 2, 2, 2, 1]) / 6)


def test_ccf_different_lengths():
    # Regression test for GH#9565 (ccf calls ccovf)
    rs = np.random.RandomState(11111)
    x = rs.normal(size=100)
    y = rs.normal(size=80)

    result = ccf(x, y, adjusted=True, nlags=30)
    assert result.shape == (30,)
    assert np.all(np.isfinite(result))


@pytest.mark.smoke
@pytest.mark.slow
def test_arma_order_select_ic():
    # smoke test, assumes info-criteria are right
    from statsmodels.tsa.arima_process import arma_generate_sample

    arparams = np.array([0.75, -0.25])
    maparams = np.array([0.65, 0.35])
    arparams = np.r_[1, -arparams]
    nobs = 250
    rs = np.random.RandomState(2014)
    y = arma_generate_sample(arparams, maparams, nobs, distrvs=rs.standard_normal)
    res = arma_order_select_ic(y, ic=["aic", "bic"], trend="n")
    # regression tests in case we change algorithm to minic in sas
    aic_x = np.array(
        [
            [764.36517643, 552.7342255, 484.29687843],
            [562.10924262, 485.5197969, 480.32858497],
            [507.04581344, 482.91065829, 481.91926034],
            [484.03995962, 482.14868032, 483.86378955],
            [481.8849479, 483.8377379, 485.83756612],
        ]
    )
    bic_x = np.array(
        [
            [767.88663735, 559.77714733, 494.86126118],
            [569.15216446, 496.08417966, 494.41442864],
            [517.61019619, 496.99650196, 499.52656493],
            [498.12580329, 499.75598491, 504.99255506],
            [499.49225249, 504.96650341, 510.48779255],
        ]
    )
    aic = DataFrame(aic_x, index=lrange(5), columns=lrange(3))
    bic = DataFrame(bic_x, index=lrange(5), columns=lrange(3))
    assert_almost_equal(res.aic.values, aic.values, 5)
    assert_almost_equal(res.bic.values, bic.values, 5)
    assert_equal(res.aic_min_order, (1, 2))
    assert_equal(res.bic_min_order, (1, 2))
    assert res.aic.index.equals(aic.index)
    assert res.aic.columns.equals(aic.columns)
    assert res.bic.index.equals(bic.index)
    assert res.bic.columns.equals(bic.columns)

    index = pd.date_range("2000-1-1", freq=MONTH_END, periods=len(y))
    y_series = pd.Series(y, index=index)
    res_pd = arma_order_select_ic(
        y_series, max_ar=2, max_ma=1, ic=["aic", "bic"], trend="n"
    )
    assert_almost_equal(res_pd.aic.values, aic.values[:3, :2], 5)
    assert_almost_equal(res_pd.bic.values, bic.values[:3, :2], 5)
    assert_equal(res_pd.aic_min_order, (2, 1))
    assert_equal(res_pd.bic_min_order, (1, 1))

    res = arma_order_select_ic(y, ic="aic", trend="n")
    assert_almost_equal(res.aic.values, aic.values, 5)
    assert res.aic.index.equals(aic.index)
    assert res.aic.columns.equals(aic.columns)
    assert_equal(res.aic_min_order, (1, 2))


def test_diebold_mariano_test():
    y = np.array(
        [
            0.7905291971644244,
            1.3165423213858396,
            2.629194824493898,
            2.274163381390831,
            2.3350897900078773,
            4.309556160533848,
            3.2008616600117854,
            3.0279620894948263,
            2.567645423157318,
            0.8342527751292501,
            2.0985671460216464,
            3.408247061676925,
            2.9699882967078457,
            3.6732582708103863,
            3.4609292040029644,
        ]
    )
    f1 = np.array(
        [
            0.9824387126880363,
            3.347555830856235,
            1.8132440948000608,
            4.501884323116638,
            0.993073450965922,
            0.988116816888287,
            3.7834883570834847,
            2.498189378576747,
            4.8566413245270965,
            1.7412497391486617,
            3.4637323882747384,
            3.101483832900048,
            3.0411384793663037,
            3.2500575506493394,
            3.199491151189982,
        ]
    )
    f2 = np.array(
        [
            0.8002551776387801,
            2.35554283908294,
            2.5017107565163754,
            2.9709840260800897,
            2.031947528256514,
            2.044069317891979,
            1.8690271476061573,
            2.541558191277514,
            3.0762879697739653,
            3.420942734113572,
            3.9529070460967226,
            3.036772850400313,
            3.898343203105889,
            4.397338510819033,
            4.699775884928464,
        ]
    )

    res1 = diebold_mariano_test(y, f1, f2, criterion="mse")
    res2 = diebold_mariano_test(y, f1, f2, lags=2, criterion="mse")
    res3 = diebold_mariano_test(y, f1, f2, criterion="mad")

    d = (y - f1) ** 2 - (y - f2) ** 2
    maxlags = int(np.ceil(len(d) ** (1 / 3)))
    res = OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    assert_almost_equal(res1.statistic, res.tvalues[0], DECIMAL_3)
    assert_almost_equal(res1.pvalue, res.pvalues[0], DECIMAL_3)

    res = OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})

    assert_almost_equal(res2.statistic, res.tvalues[0], DECIMAL_3)
    assert_almost_equal(res2.pvalue, res.pvalues[0], DECIMAL_3)

    d = np.abs(y - f1) - np.abs(y - f2)
    res = OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    assert_almost_equal(res3.statistic, res.tvalues[0], DECIMAL_3)
    assert_almost_equal(res3.pvalue, res.pvalues[0], DECIMAL_3)

    y += 10
    f1 += 10
    f2 += 10
    res4 = diebold_mariano_test(y, f1, f2, criterion="mape")
    d = np.abs((y - f1) / y) - np.abs((y - f2) / y)
    res = OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    assert_almost_equal(res4.statistic, res.tvalues[0], DECIMAL_3)
    assert_almost_equal(res4.pvalue, res.pvalues[0], DECIMAL_3)


def test_diebold_mariano_harvey_adj():
    rs = np.random.default_rng()
    y, f1, f2 = rs.standard_normal((3, 100))
    res_no = diebold_mariano_test(y, f1, f2, harvey_adj=False)
    res_yes = diebold_mariano_test(y, f1, f2, harvey_adj=True, horizon=3)
    assert res_no.harvey_adj_factor is None
    assert isinstance(res_yes.harvey_adj_factor, float)
    assert 0 <= res_yes.harvey_adj_factor < 1
    direct_factor = np.sqrt((100 + 1 - (2 * 3) + ((3 * 2) / 100)) / 100)
    assert_almost_equal(res_yes.harvey_adj_factor, direct_factor)
    assert res_yes.pvalue > res_no.pvalue
    assert np.abs(res_no.statistic) > np.abs(res_yes.statistic)
    assert len(res_no) == 2


def test_diebold_mariano_equiv():
    rs = np.random.default_rng()
    y, f1, f2 = rs.standard_normal((3, 100))
    res = diebold_mariano_test(y, f1, f2)
    res_poly = diebold_mariano_test(y, f1, f2, criterion="poly", power=2)
    assert_almost_equal(res.statistic, res_poly.statistic)

    res = diebold_mariano_test(y, f1, f2, criterion="mad")
    res_mae = diebold_mariano_test(y, f1, f2, criterion="mae")
    res_poly = diebold_mariano_test(y, f1, f2, criterion="poly", power=1)
    assert_almost_equal(res.statistic, res_poly.statistic)
    assert_almost_equal(res_mae.statistic, res_poly.statistic)


@pytest.mark.smoke
def test_diebold_mariano_callable_smoke():

    rng = np.random.default_rng(0)
    y = rng.standard_normal(200) ** 2
    scale = rng.chisquare(5, size=y.shape) / 5
    forecast_a = 0.9 * scale * y
    forecast_b = scale * y

    def qlike(y, forecast):
        ratio = y / forecast
        return ratio - np.log(ratio) - 1

    res = diebold_mariano_test(y, forecast_a, forecast_b, criterion=qlike)
    assert np.isfinite(res.statistic)
    assert 0 <= res.pvalue <= 1


def test_diebold_mariano_exceptions():
    rs = np.random.default_rng()
    y, f1, f2 = rs.standard_normal((3, 100))
    with pytest.raises(ValueError, match="lags must be a non-negative integer"):
        diebold_mariano_test(y, f1, f2, lags=-1)
    with pytest.raises(ValueError, match="y, forecast_a and forecast_b must all"):
        diebold_mariano_test(y, f1, f2[::2])
    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        diebold_mariano_test(y, f1, f2, horizon=0)


def test_arma_order_select_ic_failure():
    # this should trigger an SVD convergence failure, smoke test that it
    # returns, likely platform dependent failure...
    # looks like AR roots may be cancelling out for 4, 1?
    y = np.array(
        [
            0.86074377817203640006,
            0.85316549067906921611,
            0.87104653774363305363,
            0.60692382068987393851,
            0.69225941967301307667,
            0.73336177248909339976,
            0.03661329261479619179,
            0.15693067239962379955,
            0.12777403512447857437,
            -0.27531446294481976,
            -0.24198139631653581283,
            -0.23903317951236391359,
            -0.26000241325906497947,
            -0.21282920015519238288,
            -0.15943768324388354896,
            0.25169301564268781179,
            0.1762305709151877342,
            0.12678133368791388857,
            0.89755829086753169399,
            0.82667068795350151511,
        ]
    )
    import warnings

    with warnings.catch_warnings():
        # catch a hessian inversion and convergence failure warning
        warnings.simplefilter("ignore")
        arma_order_select_ic(y)


def test_acf_fft_dataframe():
    # regression test #322

    result = acf(sunspots.load_pandas().data[["SUNACTIVITY"]], fft=True, nlags=20)
    assert_equal(result.ndim, 1)


def test_levinson_durbin_acov():
    rho = 0.9
    m = 20
    acov = rho ** np.arange(200)
    _ld_result = levinson_durbin(acov, m, isacov=True)
    sigma2_eps, ar, pacf = _ld_result.sigma_v, _ld_result.arcoefs, _ld_result.pacf
    assert_allclose(sigma2_eps, 1 - rho**2)
    assert_allclose(ar, np.array([rho] + [0] * (m - 1)), atol=1e-8)
    assert_allclose(pacf, np.array([1, rho] + [0] * (m - 1)), atol=1e-8)


@pytest.mark.parametrize("missing", ["conservative", "drop", "raise", "none"])
@pytest.mark.parametrize("fft", [False, True])
@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("adjusted", [True, False])
def test_acovf_nlags(acovf_data, adjusted, demean, fft, missing):
    full = acovf(acovf_data, adjusted=adjusted, demean=demean, fft=fft, missing=missing)
    limited = acovf(
        acovf_data,
        adjusted=adjusted,
        demean=demean,
        fft=fft,
        missing=missing,
        nlag=10,
    )
    assert_allclose(full[:11], limited)


@pytest.mark.parametrize("missing", ["conservative", "drop"])
@pytest.mark.parametrize("fft", [False, True])
@pytest.mark.parametrize("demean", [True, False])
@pytest.mark.parametrize("adjusted", [True, False])
def test_acovf_nlags_missing(acovf_data, adjusted, demean, fft, missing):
    acovf_data = acovf_data.copy()
    acovf_data[1:3] = np.nan
    full = acovf(acovf_data, adjusted=adjusted, demean=demean, fft=fft, missing=missing)
    limited = acovf(
        acovf_data,
        adjusted=adjusted,
        demean=demean,
        fft=fft,
        missing=missing,
        nlag=10,
    )
    assert_allclose(full[:11], limited)


def test_acovf_error(acovf_data):
    with pytest.raises(ValueError):
        acovf(acovf_data, nlag=250, fft=False)


def test_pacf2acf_ar():
    pacf = np.zeros(10)
    pacf[0] = 1
    pacf[1] = 0.9
    _ldp_result = levinson_durbin_pacf(pacf)
    ar, acf = _ldp_result.arcoefs, _ldp_result.acf
    assert_allclose(acf, 0.9 ** np.arange(10.0))
    assert_allclose(ar, pacf[1:], atol=1e-8)

    _ldp_result = levinson_durbin_pacf(pacf, nlags=5)
    ar, acf = _ldp_result.arcoefs, _ldp_result.acf
    assert_allclose(acf, 0.9 ** np.arange(6.0))
    assert_allclose(ar, pacf[1:6], atol=1e-8)


def test_pacf2acf_levinson_durbin():
    pacf = -(0.9 ** np.arange(11.0))
    pacf[0] = 1
    _ldp_result = levinson_durbin_pacf(pacf)
    ar, acf = _ldp_result.arcoefs, _ldp_result.acf
    _ld_result = levinson_durbin(acf, 10, isacov=True)
    ar_ld, pacf_ld = _ld_result.arcoefs, _ld_result.pacf
    assert_allclose(ar, ar_ld, atol=1e-8)
    assert_allclose(pacf, pacf_ld, atol=1e-8)

    # From R, FitAR, PacfToAR
    ar_from_r = [
        -4.1609,
        -9.2549,
        -14.4826,
        -17.6505,
        -17.5012,
        -14.2969,
        -9.5020,
        -4.9184,
        -1.7911,
        -0.3486,
    ]
    assert_allclose(ar, ar_from_r, atol=1e-4)


def test_pacf2acf_errors():
    pacf = -(0.9 ** np.arange(11.0))
    pacf[0] = 1
    with pytest.raises(ValueError):
        levinson_durbin_pacf(pacf, nlags=20)
    with pytest.raises(ValueError):
        levinson_durbin_pacf(pacf[1:])
    with pytest.raises(ValueError):
        levinson_durbin_pacf(np.zeros(10))
    with pytest.raises(ValueError):
        levinson_durbin_pacf(np.zeros((10, 2)))


def test_pacf_burg():
    rnd = np.random.RandomState(12345)
    e = rnd.randn(10001)
    y = e[1:] + 0.5 * e[:-1]
    _pb_result = pacf_burg(y, 10)
    pacf, sigma2 = _pb_result.pacf, _pb_result.sigma2
    yw_pacf = pacf_yw(y, 10)
    assert_allclose(pacf, yw_pacf, atol=5e-4)
    # Internal consistency check between pacf and sigma2
    ye = y - y.mean()
    s2y = ye.dot(ye) / 10000
    pacf[0] = 0
    sigma2_direct = s2y * np.cumprod(1 - pacf**2)
    assert_allclose(sigma2, sigma2_direct, atol=1e-3)


def test_pacf_burg_error():
    with pytest.raises(ValueError):
        pacf_burg(np.empty((20, 2)), 10)
    with pytest.raises(ValueError):
        pacf_burg(np.empty(100), 101)


def test_innovations_algo_brockwell_davis():
    ma = -0.9
    acovf = np.array([1 + ma**2, ma])
    theta, sigma2 = innovations_algo(acovf, nobs=4)
    exp_theta = np.array([[0], [-0.4972], [-0.6606], [-0.7404]])
    assert_allclose(theta, exp_theta, rtol=1e-4)
    assert_allclose(sigma2, [1.81, 1.3625, 1.2155, 1.1436], rtol=1e-4)

    theta, sigma2 = innovations_algo(acovf, nobs=500)
    assert_allclose(theta[-1, 0], ma)
    assert_allclose(sigma2[-1], 1.0)


def test_innovations_algo_rtol():
    ma = np.array([-0.9, 0.5])
    acovf = np.array([1 + (ma**2).sum(), ma[0] + ma[1] * ma[0], ma[1]])
    theta, sigma2 = innovations_algo(acovf, nobs=500)
    theta_2, sigma2_2 = innovations_algo(acovf, nobs=500, rtol=1e-8)
    assert_allclose(theta, theta_2)
    assert_allclose(sigma2, sigma2_2)


def test_innovations_errors():
    ma = -0.9
    acovf = np.array([1 + ma**2, ma])
    with pytest.raises(TypeError):
        innovations_algo(acovf, nobs=2.2)
    with pytest.raises(ValueError):
        innovations_algo(acovf, nobs=-1)
    with pytest.raises(ValueError):
        innovations_algo(np.empty((2, 2)))
    with pytest.raises(TypeError):
        innovations_algo(acovf, rtol="none")


def test_innovations_filter_brockwell_davis():
    ma = -0.9
    acovf = np.array([1 + ma**2, ma])
    theta, _ = innovations_algo(acovf, nobs=4)
    rs = np.random.RandomState(12345)
    e = rs.randn(5)
    endog = e[1:] + ma * e[:-1]
    resid = innovations_filter(endog, theta)
    expected = [endog[0]]
    for i in range(1, 4):
        expected.append(endog[i] - theta[i, 0] * expected[-1])
    expected = np.array(expected)
    assert_allclose(resid, expected)


def test_innovations_filter_pandas():
    ma = np.array([-0.9, 0.5])
    acovf = np.array([1 + (ma**2).sum(), ma[0] + ma[1] * ma[0], ma[1]])
    theta, _ = innovations_algo(acovf, nobs=10)
    rs = np.random.RandomState(12345)
    endog = rs.randn(10)
    endog_pd = pd.Series(endog, index=pd.date_range("2000-01-01", periods=10))
    resid = innovations_filter(endog, theta)
    resid_pd = innovations_filter(endog_pd, theta)
    assert_allclose(resid, resid_pd.values)
    assert_index_equal(endog_pd.index, resid_pd.index)


def test_innovations_filter_errors():
    ma = -0.9
    acovf = np.array([1 + ma**2, ma])
    theta, _ = innovations_algo(acovf, nobs=4)
    with pytest.raises(ValueError):
        innovations_filter(np.empty((2, 2)), theta)
    with pytest.raises(ValueError):
        innovations_filter(np.empty(4), theta[:-1])
    with pytest.raises(ValueError):
        innovations_filter(pd.DataFrame(np.empty((1, 4))), theta)


def test_innovations_algo_filter_kalman_filter():
    # Test the innovations algorithm and filter against the Kalman filter
    # for exact likelihood evaluation of an ARMA process
    ar_params = np.array([0.5])
    ma_params = np.array([0.2])
    # TODO could generalize to sigma2 != 1, if desired, after #5324 is merged
    # and there is a sigma2 argument to arma_acovf
    # (but maybe this is not really necessary for the point of this test)
    sigma2 = 1
    rs = np.random.RandomState(123456)
    endog = rs.normal(size=10)

    # Innovations algorithm approach
    acovf = arma_acovf(np.r_[1, -ar_params], np.r_[1, ma_params], nobs=len(endog))

    theta, v = innovations_algo(acovf)
    u = innovations_filter(endog, theta)
    llf_obs = -0.5 * u**2 / (sigma2 * v) - 0.5 * np.log(2 * np.pi * v)

    # Kalman filter apparoach
    mod = SARIMAX(endog, order=(len(ar_params), 0, len(ma_params)))
    res = mod.filter(np.r_[ar_params, ma_params, sigma2])

    # Test that the two approaches are identical
    atol = 1e-6 if PLATFORM_WIN else 0.0
    assert_allclose(u, res.forecasts_error[0], rtol=1e-6, atol=atol)
    assert_allclose(theta[1:, 0], res.filter_results.kalman_gain[0, 0, :-1], atol=atol)
    assert_allclose(llf_obs, res.llf_obs, atol=atol)


def test_adfuller_short_series():
    rs = np.random.RandomState(9374)
    y = rs.standard_normal(7)
    res = adfuller(y, store=True, result_object=False)
    assert res[-1].maxlag == 1
    y = rs.standard_normal(2)
    with pytest.raises(ValueError, match="sample size is too short"):
        adfuller(y)
    y = rs.standard_normal(3)
    with pytest.raises(ValueError, match="sample size is too short"):
        adfuller(y, regression="ct")


def test_adfuller_maxlag_too_large():
    rs = np.random.RandomState(3723)
    y = rs.standard_normal(100)
    with pytest.raises(ValueError, match="maxlag must be less than"):
        adfuller(y, maxlag=51)


@pytest.fixture
def adfuller_data():
    rs = np.random.RandomState(0)
    return np.cumsum(rs.standard_normal(200))


def test_adfuller_result_object_default_warns(adfuller_data):
    with pytest.warns(FutureWarning, match="result_object"):
        res = adfuller(adfuller_data)
    assert isinstance(res, tuple)
    assert not isinstance(res, ADFullerResult)
    assert len(res) == 6


def test_adfuller_result_object_false_silences_warning(adfuller_data):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = adfuller(adfuller_data, result_object=False)
    assert isinstance(res, tuple)
    assert not isinstance(res, ADFullerResult)
    assert len(res) == 6


def test_adfuller_result_object_true_returns_result_object(adfuller_data):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = adfuller(adfuller_data, result_object=True)
    assert isinstance(res, ADFullerResult)
    # still positionally indexable like the legacy tuple
    assert res[0] == res.statistic
    assert res[1] == res.pvalue
    assert res.resstore is None


def test_adfuller_result_object_true_with_store(adfuller_data):
    res = adfuller(adfuller_data, store=True, result_object=True)
    assert isinstance(res, ADFullerResult)
    assert res.resstore is not None
    assert res.resstore.__str__() == "Augmented Dickey-Fuller Test Results"


def test_adfuller_result_object_true_without_autolag(adfuller_data):
    res = adfuller(adfuller_data, autolag=None, maxlag=4, result_object=True)
    assert isinstance(res, ADFullerResult)
    assert res.icbest is None


def test_adfuller_result_object_matches_legacy_values(adfuller_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy = adfuller(adfuller_data)
        nt = adfuller(adfuller_data, result_object=True)
    assert_almost_equal(nt.statistic, legacy[0])
    assert_almost_equal(nt.pvalue, legacy[1])
    assert nt.lags == legacy[2]
    assert nt.nobs == legacy[3]
    assert nt.critical_values == legacy[4]
    assert_almost_equal(nt.icbest, legacy[5])


class SetupZivotAndrews:
    # test directory
    cur_dir = CURR_DIR
    run_dir = Path(cur_dir).joinpath("results")
    # use same file for testing failure modes
    fail_file = Path(run_dir).joinpath("rgnp.csv")
    fail_mdl = np.asarray(pd.read_csv(fail_file))


class TestZivotAndrews(SetupZivotAndrews):
    # failure mode tests
    def test_fail_regression_type(self):
        with pytest.raises(ValueError):
            zivot_andrews(self.fail_mdl, regression="x")

    def test_fail_trim_value(self):
        with pytest.raises(ValueError):
            zivot_andrews(self.fail_mdl, trim=0.5)

    def test_fail_array_shape(self):
        with pytest.raises(ValueError):
            rs = np.random.RandomState(37283)
            zivot_andrews(rs.rand(50, 2))

    def test_fail_autolag_type(self):
        with pytest.raises(ValueError):
            zivot_andrews(self.fail_mdl, autolag="None")

    @pytest.mark.parametrize("autolag", ["AIC", "aic", "Aic"])
    def test_autolag_case_sensitivity(self, autolag):
        res = zivot_andrews(self.fail_mdl, autolag=autolag)
        assert res[3] == 1

    # following tests compare results to R package urca.ur.za (1.13-0)
    def test_rgnp_case(self):
        res = zivot_andrews(self.fail_mdl, maxlag=8, regression="c", autolag=None)
        assert_allclose([res[0], res[1], res[4]], [-5.57615, 0.00312, 20], rtol=1e-3)

    def test_gnpdef_case(self):
        mdlfile = Path(self.run_dir).joinpath("gnpdef.csv")
        mdl = np.asarray(pd.read_csv(mdlfile))
        res = zivot_andrews(mdl, maxlag=8, regression="c", autolag="t-stat")
        assert_allclose(
            [res[0], res[1], res[3], res[4]],
            [-4.12155, 0.28024, 5, 40],
            rtol=1e-3,
        )

    def test_stkprc_case(self):
        mdlfile = Path(self.run_dir).joinpath("stkprc.csv")
        mdl = np.asarray(pd.read_csv(mdlfile))
        res = zivot_andrews(mdl, maxlag=8, regression="ct", autolag="t-stat")
        assert_allclose(
            [res[0], res[1], res[3], res[4]],
            [-5.60689, 0.00894, 1, 65],
            rtol=1e-3,
        )

    def test_rgnpq_case(self):
        mdlfile = Path(self.run_dir).joinpath("rgnpq.csv")
        mdl = np.asarray(pd.read_csv(mdlfile))
        res = zivot_andrews(mdl, maxlag=12, regression="t", autolag="t-stat")
        assert_allclose(
            [res[0], res[1], res[3], res[4]],
            [-3.02761, 0.63993, 12, 102],
            rtol=1e-3,
        )

    def test_rand10000_case(self):
        mdlfile = Path(self.run_dir).joinpath("rand10000.csv")
        mdl = np.asarray(pd.read_csv(mdlfile))
        res = zivot_andrews(mdl, regression="c", autolag="t-stat")
        assert_allclose(
            [res[0], res[1], res[3], res[4]],
            [-3.48223, 0.69111, 25, 7071],
            rtol=1e-3,
        )


def test_acf_conservate_nanops():
    # GH 6729
    rs = np.random.RandomState(32738493)
    e = rs.standard_normal(100)
    for i in range(1, e.shape[0]):
        e[i] += 0.9 * e[i - 1]
    e[::7] = np.nan
    result = acf(e, missing="conservative", nlags=10, fft=False)
    resid = e - np.nanmean(e)
    expected = np.ones(11)
    nobs = e.shape[0]
    gamma0 = np.nansum(resid * resid)
    for i in range(1, 10 + 1):
        expected[i] = np.nansum(resid[i:] * resid[: nobs - i]) / gamma0
    assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_pacf_nlags_error():
    rs = np.random.RandomState(12487)
    e = rs.standard_normal(99)
    with pytest.raises(ValueError, match="Can only compute partial"):
        pacf(e, 50)


def test_coint_auto_tstat():
    rs = np.random.RandomState(3733696641)
    x = np.cumsum(rs.standard_normal(100))
    y = np.cumsum(rs.standard_normal(100))
    res = coint(
        x,
        y,
        trend="c",
        method="aeg",
        maxlag=0,
        autolag="t-stat",
        return_results=False,
    )
    assert np.abs(res.coint_t) < 1.65


rs = np.random.RandomState(1)
a = rs.random_sample(120)
b = np.zeros_like(a)
df1 = pd.DataFrame({"b": b, "a": a})
df2 = pd.DataFrame({"a": a, "b": b})

b = np.ones_like(a)
df3 = pd.DataFrame({"b": b, "a": a})
df4 = pd.DataFrame({"a": a, "b": b})

gc_data_sets = [df1, df2, df3, df4]


@pytest.mark.parametrize("dataset", gc_data_sets)
def test_granger_causality_exceptions(dataset):
    with pytest.raises(InfeasibleTestError):
        grangercausalitytests(dataset, 4)


def test_granger_causality_exception_maxlag(gc_data):
    with pytest.raises(ValueError, match="maxlag must be"):
        grangercausalitytests(gc_data, maxlag=-1)
    with pytest.raises(NotImplementedError):
        grangercausalitytests(gc_data, 3, addconst=False)


@pytest.mark.parametrize("size", [3, 5, 7, 9])
def test_pacf_small_sample(size):
    rs = np.random.RandomState(490203)
    y = rs.standard_normal(size)
    a = pacf(y)
    assert isinstance(a, np.ndarray)
    _pb_result = pacf_burg(y)
    a, b = _pb_result.pacf, _pb_result.sigma2
    assert isinstance(a, np.ndarray)
    assert isinstance(b, np.ndarray)
    a = pacf_ols(y)
    assert isinstance(a, np.ndarray)
    a = pacf_yw(y)
    assert isinstance(a, np.ndarray)


def test_pacf_1_obs():
    rs = np.random.RandomState(34857549)
    y = rs.standard_normal(1)
    with pytest.raises(ValueError):
        pacf(y)
    with pytest.raises(ValueError):
        pacf_burg(y)
    with pytest.raises(ValueError):
        pacf_ols(y)
    pacf_yw(y)


def test_zivot_andrews_change_data():
    # GH9307
    years = pd.date_range(start="1990-01-01", end="2023-12-31", freq="YS")
    df = pd.DataFrame(index=years)
    df["variable1"] = np.where(df.index.year <= 2002, 10, 20)
    df["variable2"] = np.where(df.index.year <= 2002, 10, 20)
    df.iloc[-1] = 30

    # Zivot-Andrews test with data with type float64
    df = df.astype(float)
    df_original = df.copy()
    zivot_andrews(df["variable1"])
    zivot_andrews(df["variable1"], regression="c")
    pd.testing.assert_frame_equal(df, df_original)


class TestLeybourneMcCabe:
    cur_dir = CURR_DIR
    run_dir = Path(cur_dir).joinpath("results")

    # failure mode tests
    def test_fail_inputs(self):
        # use results/BAA.csv file for testing failure modes
        fail_file = Path(self.run_dir).joinpath("BAA.csv")
        fail_mdl = np.asarray(pd.read_csv(fail_file))
        with pytest.raises(ValueError):
            leybourne(fail_mdl, regression="nc")
        with pytest.raises(ValueError):
            leybourne(fail_mdl, method="gls")
        with pytest.raises(ValueError):
            leybourne(fail_mdl, varest="var98")
        with pytest.raises(ValueError):
            leybourne([[1, 1], [2, 2]], regression="c")
        with pytest.raises(ValueError):
            leybourne(fail_mdl, arlags=250)
        with pytest.raises(ValueError):
            leybourne(fail_mdl, arlags="error")

    # the following tests use data sets from Schwert (1987)
    # and were verified against Matlab 9.13
    def test_baa_results(self):
        mdl_file = Path(self.run_dir).joinpath("BAA.csv")
        mdl = np.asarray(pd.read_csv(mdl_file))
        res = leybourne(mdl, regression="ct", method="mle")
        assert_allclose(res[0:3], [5.4438, 0.0000, 3], rtol=1e-4, atol=1e-4)
        res = leybourne(mdl, regression="ct")
        assert_allclose(res[0:3], [5.4757, 0.0000, 3], rtol=1e-4, atol=1e-4)

    def test_dbaa_results(self):
        mdl_file = Path(self.run_dir).joinpath("DBAA.csv")
        mdl = np.asarray(pd.read_csv(mdl_file))
        res = leybourne(mdl, method="mle")
        assert_allclose(res[0:3], [0.096534, 0.602535, 2], rtol=1e-4, atol=1e-4)
        res = leybourne(mdl, regression="ct", method="mle")
        assert_allclose(res[0:3], [0.047924, 0.601817, 2], rtol=1e-4, atol=1e-4)

    def test_dsp500_results(self):
        mdl_file = Path(self.run_dir).joinpath("DSP500.csv")
        mdl = np.asarray(pd.read_csv(mdl_file))
        res = leybourne(mdl, method="mle")
        assert_allclose(res[0:3], [0.3118, 0.1256, 0], rtol=1e-4, atol=1e-4)
        res = leybourne(mdl, varest="var99", method="mle")
        assert_allclose(res[0:3], [0.306886, 0.129934, 0], rtol=1e-4, atol=1e-4)

    def test_dun_results(self):
        mdl_file = Path(self.run_dir).joinpath("DUN.csv")
        mdl = np.asarray(pd.read_csv(mdl_file))
        res = leybourne(mdl, regression="ct", method="ols")
        assert_allclose(res[0:3], [0.0938, 0.1890, 3], rtol=1e-4, atol=1e-4)

    @pytest.mark.xfail(reason="Fails due to numerical issues", strict=False)
    def test_dun_results_arima(self):
        mdl_file = Path(self.run_dir).joinpath("DUN.csv")
        mdl = np.asarray(pd.read_csv(mdl_file))
        res = leybourne(mdl, regression="ct")
        assert_allclose(res[0], 0.024083, rtol=1e-4, atol=1e-4)
        assert_allclose(res[1], 0.943151, rtol=1e-4, atol=1e-4)
        assert res[2] == 3

    def test_sp500_results(self):
        mdl_file = Path(self.run_dir).joinpath("SP500.csv")
        mdl = np.asarray(pd.read_csv(mdl_file))
        res = leybourne(mdl, arlags=4, regression="ct", method="mle")
        assert_allclose(res[0:2], [1.8761, 0.0000], rtol=1e-4, atol=1e-4)
        res = leybourne(mdl, arlags=4, regression="ct")
        assert_allclose(res[0:2], [1.9053, 0.0000], rtol=1e-4, atol=1e-4)

    def test_un_results(self):
        mdl_file = Path(self.run_dir).joinpath("UN.csv")
        mdl = np.asarray(pd.read_csv(mdl_file))
        res = leybourne(mdl, method="ols", varest="var99")
        assert_allclose(res[0:3], [556.0444, 0.0000, 4], rtol=1e-4, atol=1e-4)

    @pytest.mark.xfail(reason="Fails due to numerical issues", strict=False)
    def test_un_results_arima(self):
        mdl_file = Path(self.run_dir).joinpath("UN.csv")
        mdl = np.asarray(pd.read_csv(mdl_file))
        res = leybourne(mdl, varest="var99")
        assert_allclose(res[0], 285.5181, rtol=1e-4, atol=1e-4)
        assert_allclose(res[1], 0.0000, rtol=1e-4, atol=1e-4)
        assert res[2] == 4

    def test_lm_whitenoise(self):
        rg = np.random.RandomState(0)
        y = rg.standard_normal(250)
        res = leybourne(y, method="ols", varest="var99")
        assert res[2] == 0


def test_acovf_all_missing():
    x = np.full(100, np.nan)
    with pytest.raises(
        ValueError, match=r"All observations are missing after dropping."
    ):
        acovf(x, missing="drop")

    assert np.all(np.isnan(acovf(x, missing="conservative")))


def test_stattools_fixed_arity_result_objects():
    # These functions always return the same number of values, so unlike
    # the result_object=True/False/None functions, they unconditionally
    # return a result object with no deprecation cycle for *whether* one
    # is returned. Indexing/unpacking the result object itself is a
    # stable, non-deprecated part of the API.
    rs = np.random.RandomState(0)
    x = rs.standard_normal(200).cumsum()

    res = block_jackknife(rs.standard_normal(100), np.mean, n_blocks=10)
    assert isinstance(res, JackknifeResult)
    assert res[0] is res.theta_jack
    assert res[1] is res.se

    a = acf(x, nlags=10, fft=False)
    res = q_stat(a[1:], nobs=len(x))
    assert isinstance(res, QStatResult)
    assert res[0] is res.statistic
    assert res[1] is res.pvalue

    res = pacf_burg(x, nlags=5)
    assert isinstance(res, PacfBurgResult)
    assert res[0] is res.pacf
    assert res[1] is res.sigma2

    res = levinson_durbin(x, nlags=5, isacov=False)
    assert isinstance(res, LevinsonDurbinResult)
    assert res[0] == res.sigma_v
    assert res[1] is res.arcoefs
    assert res[2] is res.pacf
    assert res[3] is res.sigma
    assert res[4] is res.phi

    res2 = levinson_durbin_pacf(res.pacf, nlags=3)
    assert isinstance(res2, LevinsonDurbinPacfResult)
    assert res2[0] is res2.arcoefs
    assert res2[1] is res2.acf

    res = breakvar_heteroskedasticity_test(x)
    assert isinstance(res, BreakvarHeteroskedasticityResult)
    assert res[0] == res.statistic
    assert res[1] == res.pvalue

    y1 = (x + rs.standard_normal(200))[:, None]
    res = coint(x, y1, trend="c", maxlag=0, autolag=None)
    assert isinstance(res, CointResult)
    assert res[0] == res.coint_t
    assert res[1] == res.pvalue
    assert res[2] is res.critical_values
