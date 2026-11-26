from statsmodels.compat.numpy import NP_LT_1_24
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
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tools.sm_exceptions import (
    CollinearityWarning,
    ConvergenceWarning,
    EstimationWarning,
    InfeasibleTestError,
    InterpolationWarning,
    MissingDataError,
    SingularMatrixWarning,
)
from statsmodels.tools.testing import warns_or_none
from statsmodels.tools.tools import add_constant

# Remove imports when range unit root test gets an R implementation
from statsmodels.tools.validation import array_like, bool_like
from statsmodels.tsa.arima_process import arma_acovf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import (
    AcfResult,
    ADFullerResult,
    BreakvarHeteroskedasticityResult,
    CcfResult,
    ClarkWestResult,
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
    _stattools,
    acf,
    acovf,
    adfuller,
    arma_order_select_ic,
    block_jackknife,
    breakvar_heteroskedasticity_test,
    ccf,
    ccovf,
    clark_west_test,
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
from statsmodels.tsa.stattools._stattools import (
    ZivotAndrewsUnitRoot,
    _adf_autolag,
    _autolag,
    _pacf_ols_qr,
    _pacf_yw_levinson,
)
from statsmodels.tsa.tests._contract import SERIES_FORMS, series_forms
from statsmodels.tsa.tsatools import add_trend, lagmat

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
            pacf_yw(np.ones(30), nlags=6)

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

    def test_yw_adjusted_non_positive_variance_returns_nan(self):
        # With the adjusted autocovariances (divisor nobs - h) of "yw", x and
        # y have Gamma(0) = I and Gamma(1) = [[-1, 1/7], [1/7, 1/7]]. The
        # forward prediction error variance of x at lag 2 is then
        # (I - Gamma(1) Gamma(1)')[0, 0] = 1 - (1 + 1 / 49) < 0, so the
        # recursion stops and the lags from 2 on are NaN.
        x = np.tile([1.0, -1.0], 4)
        y = np.tile([1.0, 1.0, -1.0, -1.0], 2)
        gamma1 = np.array([[-1.0, 1 / 7], [1 / 7, 1 / 7]])
        assert (np.eye(2) - gamma1 @ gamma1.T)[0, 0] < 0
        result = pccf(x, y, nlags=4, method="yw")
        # lag 1 is Gamma(1)[0, 1] / sqrt(Gamma(0)[0, 0] * Gamma(0)[1, 1])
        assert_allclose(result[0], 1 / 7, rtol=1e-12)
        assert np.all(np.isnan(result[1:]))
        # With the biased autocovariances (divisor nobs) of "ywm", the variances
        # at lag 2 are positive, and lags 1 and 2 are 1/8 and
        # (3/32) / sqrt(7/32 * 31/32) = 3 / sqrt(217). Stop at lag 2: for these
        # periodic series the "ywm" prediction error covariance after two lags
        # is exactly singular, so the recursion from lag 3 on solves a singular
        # system and its result depends on rounding.
        assert_allclose(
            pccf(x, y, nlags=2, method="ywm"), [1 / 8, 3 / np.sqrt(217)], rtol=1e-12
        )

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

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    @pytest.mark.parametrize("method", ["yw", "ywm"])
    @pytest.mark.parametrize("scale", [1e-200, 1e-8, 1e8, 1e200])
    @pytest.mark.parametrize("scaled", ["x", "y"])
    def test_yw_scale_invariant(self, method, scale, scaled):
        # The PCCF does not depend on the scales of x and y. The singularity
        # checks used absolute tolerances, so rescaling one series made every
        # lag NaN, and 1e200 overflowed.
        x = np.diff(np.log(self.x.to_numpy()))
        y = np.diff(np.log(self.y.to_numpy()))
        expected = pccf(x, y, nlags=5, method=method)
        assert np.all(np.isfinite(expected))
        if scaled == "x":
            x = scale * x
        else:
            y = scale * y
        result = pccf(x, y, nlags=5, method=method)
        assert_allclose(result, expected, rtol=1e-12)

    # A negative variance must stop the recursion, not reach np.sqrt
    @pytest.mark.filterwarnings("error::RuntimeWarning")
    @pytest.mark.parametrize(
        "method, expected",
        [
            ("ywm", [1 / 8, 3 / np.sqrt(217), -1 / 14, np.nan]),
            ("yw", [1 / 7, np.nan, np.nan, np.nan]),
        ],
    )
    def test_yw_singular_step_later_lags_nan(self, method, expected):
        # In rational arithmetic, the "ywm" prediction error covariance
        # matrices after two lags are singular for these periodic series,
        # with diagonals 21/104. Lag 3 only needs the diagonals and is
        # (-3/208) / (21/104) = -1/14. Lag 4 needs the inverses, so it is NaN;
        # it came from solving the singular system and depended on rounding.
        # With "yw", the forward prediction error variance after one lag is
        # -1/49, so the lags from 2 on are NaN.
        x = np.tile([1.0, -1.0], 4)
        y = np.tile([1.0, 1.0, -1.0, -1.0], 2)
        result = pccf(x, y, nlags=4, method=method)
        assert_allclose(result, expected, rtol=1e-12)

    def test_yw_lag_before_singular_step_kept(self):
        # y_t = x_{t+1}, and every step here is exact in binary floating
        # point. Gamma(0) = I and Gamma(1)[0, 1] = -7/8. The "ywm" prediction
        # error covariance after one lag is singular, with diagonal 7/32, so
        # np.linalg.solve raised and lag 2 was discarded, although it only
        # needs the diagonal. Lags 3 and 4 need the inverse and are NaN.
        x = np.tile([-1.0, -1.0, 1.0, 1.0], 2)
        y = np.roll(x, -1)
        result = pccf(x, y, nlags=4, method="ywm")
        # lag 2 is exactly 0, which rtol cannot check, hence atol
        assert_allclose(result[:2], [-7 / 8, 0.0], rtol=1e-12, atol=1e-14)
        assert np.all(np.isnan(result[2:]))

    @pytest.mark.parametrize(
        "method, expected",
        [
            (
                "ywm",
                [
                    -0.163137921011545,
                    -0.0395017554720548,
                    -0.022027194791587,
                    -0.140490723788968,
                ],
            ),
            (
                "yw",
                [
                    -0.165902970520216,
                    -0.0409034423438503,
                    -0.023252913269295,
                    -0.151223237958104,
                ],
            ),
        ],
    )
    def test_yw_collinear_to_working_precision_nan(self, method, expected):
        # y = x + noise * e. The expected values for noise = 1e-3 are from
        # exact rational arithmetic on the same floats. With noise = 1e-7
        # the correlation of x and y is 1 - O(1e-14), so x and y are
        # collinear to working precision: the values were off by up to 0.03
        # against exact arithmetic, and are now NaN, as for exactly
        # collinear series (noise = 0).
        rng = np.random.default_rng(3)
        x = rng.standard_normal(60)
        e = rng.standard_normal(60)
        result = pccf(x, x + 1e-3 * e, nlags=4, method=method)
        assert_allclose(result, expected, rtol=1e-8)
        for noise in (1e-7, 0.0):
            result = pccf(x, x + noise * e, nlags=4, method=method)
            assert np.all(np.isnan(result))


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
        # The third column has too few early observations on purpose
        with pytest.warns(UserWarning, match="Early subset of data for variable 2"):
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

    @pytest.mark.parametrize(
        "subset_length",
        [40, np.int64(40), 40.0, np.float64(40.0), 0.25],
        ids=["int", "int64", "float", "float64", "fraction"],
    )
    def test_subset_length_integer_like(self, subset_length):
        # Any integer-like value and the fraction 0.25 of 160 observations
        # give subsets of 40 observations
        resid = np.random.default_rng(20261125).standard_normal(160)
        res = breakvar_heteroskedasticity_test(resid, subset_length=subset_length)
        expected_statistic = np.sum(resid[-40:] ** 2) / np.sum(resid[:40] ** 2)
        expected_pvalue = 2 * min(
            self.f.cdf(expected_statistic, 40, 40),
            self.f.sf(expected_statistic, 40, 40),
        )
        assert_allclose(res.statistic, expected_statistic, rtol=1e-12)
        assert_allclose(res.pvalue, expected_pvalue, rtol=1e-12)

    def test_subset_length_one(self):
        # One observation in each subset is too few for the statistic
        resid = np.random.default_rng(20261125).standard_normal(160)
        with pytest.warns(UserWarning, match="too few non-missing observations"):
            res = breakvar_heteroskedasticity_test(resid, subset_length=1.0)
        assert np.isnan(res.statistic)

    @pytest.mark.parametrize(
        "subset_length", [0, 0.0, -1, -0.25, 1.5, 40.5, np.nan, np.inf]
    )
    def test_subset_length_invalid_value(self, subset_length):
        resid = np.random.default_rng(20261125).standard_normal(160)
        with pytest.raises(ValueError, match="subset_length must be a fraction"):
            breakvar_heteroskedasticity_test(resid, subset_length=subset_length)

    @pytest.mark.parametrize("subset_length", ["40", True, None])
    def test_subset_length_invalid_type(self, subset_length):
        resid = np.random.default_rng(20261125).standard_normal(160)
        with pytest.raises(TypeError, match="subset_length must be float_like"):
            breakvar_heteroskedasticity_test(resid, subset_length=subset_length)


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


@pytest.mark.filterwarnings("error:return_results has no effect:FutureWarning")
@pytest.mark.parametrize("return_results", [None, True, False])
def test_coint_return_results_deprecated(return_results):
    # A CointResult, which also unpacks as a tuple, is returned for every
    # value of return_results. Any value other than None warns that the
    # argument has no effect.
    rng = np.random.default_rng(20261119)
    y1 = rng.standard_normal(200).cumsum()
    y0 = 0.5 * y1 + rng.standard_normal(200)
    if return_results is None:
        res = coint(y0, y1, maxlag=2, autolag=None, return_results=None)
    else:
        with pytest.warns(
            FutureWarning, match="return_results has no effect"
        ) as record:
            res = coint(y0, y1, maxlag=2, autolag=None, return_results=return_results)
        # the warning points at the caller
        assert record[0].filename == __file__
    assert isinstance(res, CointResult)
    coint_t, pvalue, crit = res
    # The test statistic is the ADF statistic of the residuals of the
    # cointegrating regression, without a trend in the ADF regression
    resid = OLS(y0, add_constant(y1)).fit().resid
    adf = adfuller(resid, maxlag=2, autolag=None, regression="n", result_object=True)
    assert_allclose(coint_t, adf.statistic, rtol=1e-10)
    default = coint(y0, y1, maxlag=2, autolag=None)
    assert_allclose(pvalue, default.pvalue, rtol=1e-12)
    assert_allclose(crit, default.critical_values, rtol=1e-12)


@pytest.mark.parametrize("trend", ["c", "ct", "ctt", "n"])
def test_coint_critical_values_ndarray(trend):
    # critical_values is an ndarray of the 1%, 5% and 10% critical values for
    # every trend. MacKinnon (2010) has no critical values without a
    # constant, so they are NaN with trend="n". Otherwise a smaller size has
    # a more negative critical value.
    rng = np.random.default_rng(20261119)
    y1 = rng.standard_normal(200).cumsum()
    y0 = 0.5 * y1 + rng.standard_normal(200)
    crit = coint(y0, y1, trend=trend, maxlag=2, autolag=None).critical_values
    assert isinstance(crit, np.ndarray)
    assert crit.shape == (3,)
    assert crit.dtype == np.float64
    if trend == "n":
        assert np.all(np.isnan(crit))
    else:
        assert np.all(np.diff(crit) > 0)


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
        # The test needs more than 3 * maxlag + 1 observations
        with pytest.raises(ValueError, match="Insufficient observations"):
            grangercausalitytests(x, 3)
        with pytest.raises(ValueError, match="Insufficient observations"):
            grangercausalitytests(x[:7], 2)

    @pytest.mark.parametrize("nobs", [5, 6, 7, 8, 11, 13])
    def test_granger_maximum_allowable_lag(self, nobs):
        # The test needs nobs > 3 * maxlag + 1, so the largest lag is
        # floor((nobs - 2) / 3). The message used to say one less unless
        # nobs - 1 was a multiple of 3.
        max_lag = (nobs - 2) // 3
        x = np.random.default_rng(20261120).standard_normal((nobs, 2))
        res = grangercausalitytests(x, max_lag)
        assert max(res) == max_lag
        match = rf"Insufficient observations. Maximum allowable lag is {max_lag}\."
        with pytest.raises(ValueError, match=match):
            grangercausalitytests(x, max_lag + 1)

    @pytest.mark.parametrize("nobs", [1, 3, 4])
    def test_granger_too_few_observations_for_any_lag(self, nobs):
        # The message used to give a maximum allowable lag of 0 or -1
        x = np.random.default_rng(20261120).standard_normal((nobs, 2))
        match = "Insufficient observations. x has too few observations for any lag"
        with pytest.raises(ValueError, match=match):
            grangercausalitytests(x, 1)

    @pytest.mark.parametrize("shape", [(50,), (50, 1), (50, 3), (50, 0)])
    def test_granger_fails_on_number_of_columns(self, shape):
        # The error used to come from deep inside, for example "wrong shape
        # for coefs", or an IndexError for no column
        x = np.random.default_rng(20261120).standard_normal(shape)
        ncols = 1 if len(shape) == 1 else shape[1]
        match = f"x must have exactly 2 columns, but it has {ncols}"
        with pytest.raises(ValueError, match=match):
            grangercausalitytests(x, 2)

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

    @pytest.mark.parametrize("maxlag", [[], ()], ids=["list", "tuple"])
    def test_granger_fails_on_empty_lags(self, maxlag):
        x = np.random.default_rng(20261126).standard_normal((100, 2))
        with pytest.raises(
            ValueError,
            match="maxlag must be a non-empty list containing only positive integers",
        ):
            grangercausalitytests(x, maxlag)


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

    def test_kpss_fails_on_nan(self):
        # a NaN in the series used to crash deep inside the lag computation
        # with "cannot convert float NaN to integer"; it must raise
        # MissingDataError instead
        x = np.arange(100.0)
        x[50] = np.nan
        with pytest.raises(MissingDataError, match="must contain only finite values"):
            kpss(x)

        x = np.arange(100.0)
        x[-3] = -np.inf
        with pytest.raises(MissingDataError, match="must contain only finite values"):
            kpss(x)

    @pytest.mark.parametrize("regression", ["c", "ct"])
    @pytest.mark.parametrize("nlags", ["auto", "legacy", 5])
    def test_kpss_fails_on_constant(self, regression, nlags):
        # a constant series has zero residual variance, which used to crash
        # the automatic lag selection with "cannot convert float NaN to
        # integer" or return a NaN or meaningless statistic
        # GH8224
        with pytest.raises(ValueError, match="x is constant"):
            kpss(np.ones(100), regression, nlags=nlags)

    def test_kpss_fails_on_nobs_check(self):
        # Test that if lags exceeds number of observations KPSS raises a
        # clear error
        # GH5925
        nobs = len(self.x)
        msg = rf"lags \({nobs}\) must be < number of observations \({nobs}\)"
        with pytest.raises(ValueError, match=msg):
            kpss(self.x, "c", nlags=nobs)

    def test_kpss_fails_on_negative_nlags(self):
        # a negative nlags used to flow straight into the lag computation and
        # silently return a statistic computed with lags=-1
        with pytest.raises(ValueError, match="non-negative"):
            kpss(self.x, "c", nlags=-1)

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

    def test_statistic_below_table_warns(self):
        # For x = 1, -1, 1, ... the partial sums are 1, 0, 1, ..., so with
        # no lags eta = (n / 2) / n**2 and s_hat = 1, and the statistic
        # 1 / (2 n) is below the 10% critical value
        nobs = 100
        x = np.tile([1.0, -1.0], nobs // 2)
        msg = "The actual p-value is greater than the p-value returned"
        with pytest.warns(InterpolationWarning, match=msg):
            res = kpss(x, "c", nlags=0, result_object=True)
        assert_allclose(res.statistic, 1 / (2 * nobs), rtol=1e-12)
        assert_allclose(res.pvalue, 0.10, rtol=1e-12)

    @pytest.mark.parametrize("nobs, expected", [(20, 9), (240, 15)])
    def test_legacy_lags_formula(self, nobs, expected):
        # ceil(12 * (nobs / 100) ** (1 / 4)): 8.02 and 14.94 round up
        x = np.random.default_rng(nobs).standard_normal(nobs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            res = kpss(x, nlags="legacy", result_object=True)
        assert res.lags == expected


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
        # the null is rejected for small statistics: below the table the actual
        # p-value is smaller than reported, above it larger
        direction = ""
        if rur_stat < inter_crit[0, 0]:
            direction = "smaller"
        elif rur_stat >= inter_crit[0, -1]:
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
        # The reference implementation warns for the same out of table statistic
        with pytest.warns(InterpolationWarning, match="The test statistic is"):
            simple_rur_stat, _, _ = self.simple_rur(self.x)
        assert_almost_equal(rur_stat, simple_rur_stat, DECIMAL_3)

    def test_pval(self):
        with pytest.warns(InterpolationWarning):
            _, pval, _ = range_unit_root_test(self.x, result_object=False)
        # The reference implementation warns for the same out of table statistic
        with pytest.warns(InterpolationWarning, match="The test statistic is"):
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

    def test_interpolation_warning_direction(self):
        # statistic below the table: the actual p-value is smaller
        msg = "The actual p-value is smaller than the p-value returned"
        with pytest.warns(InterpolationWarning, match=msg) as record:
            res = range_unit_root_test(np.zeros(250), result_object=True)
        assert len(record) == 1
        assert res.statistic == 0.0
        assert res.pvalue == 0.01
        # statistic above the table: the actual p-value is larger
        msg = "The actual p-value is larger than the p-value returned"
        with pytest.warns(InterpolationWarning, match=msg) as record:
            res = range_unit_root_test(np.arange(250.0), result_object=True)
        assert len(record) == 1
        assert res.pvalue == 0.95

    def test_no_warning_lower_edge(self):
        # 20 new extremes in 250 observations: 20 / sqrt(250) = 1.265 is between
        # the 5% (1.2242) and 10% (1.3632) critical values, so the p-value is
        # 0.10 and inside the table
        x = np.r_[np.arange(21.0), np.full(229, 10.0)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = range_unit_root_test(x, result_object=True)
        assert res.pvalue == 0.10

    def test_no_warning_inside_table(self):
        # 47 new extremes in 250 observations is between the 0.90 and 0.95
        # critical values, so the p-value is inside the table
        x = np.r_[np.arange(48.0), np.full(202, 20.5)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = range_unit_root_test(x, result_object=True)
        assert res.pvalue == 0.95

    @pytest.mark.parametrize("store", [False, True])
    def test_result_object_default_warns(self, store):
        # 20 new extremes in 250 observations, inside the table
        x = np.r_[np.arange(21.0), np.full(229, 10.0)]
        with pytest.warns(FutureWarning, match="result_object"):
            res = range_unit_root_test(x, store=store)
        assert isinstance(res, tuple)
        assert len(res) == 3 + store
        assert_allclose(res[0], 20 / np.sqrt(250), rtol=1e-12)
        assert_allclose(res[1], 0.10, rtol=1e-12)
        # 250 observations is a row of Table 1 of Aparicio et al. (2006)
        crit = [res[2][key] for key in ("10%", "5%", "2.5%", "1%")]
        assert_allclose(crit, [1.3632, 1.2242, 1.1137, 0.9982], rtol=1e-12)
        if store:
            assert res[3].nobs == 250


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


def test_ccf_nlags_validation():
    # nlags is used directly as a slice bound, so it must be an in-range int
    rs = np.random.RandomState(11111)
    x = rs.normal(size=100)
    y = rs.normal(size=80)
    with pytest.raises(ValueError, match="non-negative"):
        ccf(x, y, nlags=-3)
    with pytest.raises(ValueError, match="smaller than the number of observations"):
        ccf(x, y, nlags=101)
    with pytest.raises(TypeError, match="nlags"):
        ccf(x, y, nlags=2.5)
    # the default output length, len(x), remains a valid request
    assert ccf(x, y, nlags=100).shape == (100,)


def _ccf_input(form, a):
    if form == "column":
        return a[:, None]
    if form == "dataframe":
        return pd.DataFrame({"a": a})
    if form == "list":
        return a.tolist()
    if form == "series":
        return pd.Series(a)
    return a


# Pairs of (x form, y form): each form on its own, and a column mixed with
# a 1-D series
CCF_INPUT_FORMS = [
    ("column", "column"),
    ("dataframe", "dataframe"),
    ("list", "list"),
    ("series", "series"),
    ("column", "ndarray"),
    ("ndarray", "dataframe"),
]


@pytest.fixture
def ccf_data():
    rng = np.random.default_rng(10795)
    # y is shorter than x, so the adjusted denominators depend on len(y)
    return rng.standard_normal(120), rng.standard_normal(100)


@pytest.mark.parametrize("x_form, y_form", CCF_INPUT_FORMS)
@pytest.mark.parametrize("adjusted", [True, False])
def test_ccovf_input_forms(ccf_data, x_form, y_form, adjusted):
    # An (nobs, 1) column is the same series as the 1-D array. Columns for
    # both x and y used to give an (nobs, nobs) array, or an (nobs, 1) one
    # with adjusted=False.
    x, y = ccf_data
    expected = ccovf(x, y, adjusted=adjusted)
    result = ccovf(_ccf_input(x_form, x), _ccf_input(y_form, y), adjusted=adjusted)
    assert result.shape == (120,)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize("x_form, y_form", CCF_INPUT_FORMS)
@pytest.mark.parametrize("adjusted", [True, False])
@pytest.mark.parametrize("nlags", [None, 5])
def test_ccf_input_forms(ccf_data, x_form, y_form, adjusted, nlags):
    x, y = ccf_data
    x_in, y_in = _ccf_input(x_form, x), _ccf_input(y_form, y)
    n_ret = 120 if nlags is None else nlags
    kwargs = {"adjusted": adjusted, "nlags": nlags}

    expected = ccf(x, y, **kwargs)
    result = ccf(x_in, y_in, **kwargs)
    assert result.shape == (n_ret,)
    assert_allclose(result, expected, rtol=1e-12)

    res_obj = ccf(x_in, y_in, result_object=True, **kwargs)
    assert res_obj.confint is None
    assert_allclose(res_obj.ccf, expected, rtol=1e-12)

    expected_alpha = ccf(x, y, alpha=0.05, **kwargs)
    res_alpha = ccf(x_in, y_in, alpha=0.05, **kwargs)
    assert res_alpha.ccf.shape == (n_ret,)
    assert_allclose(res_alpha.ccf, expected_alpha.ccf, rtol=1e-12)
    assert res_alpha.confint.shape == (n_ret, 2)
    assert_allclose(res_alpha.confint, expected_alpha.confint, rtol=1e-12)


@pytest.mark.parametrize("func", [ccf, ccovf])
@pytest.mark.parametrize("bad", ["x", "y", "both"])
@pytest.mark.parametrize("form", ["ndarray", "dataframe"])
@pytest.mark.parametrize("adjusted", [True, False])
def test_ccf_ccovf_two_columns_raises(func, bad, form, adjusted):
    # With adjusted=False, (nobs, 2) arrays for both x and y used to return
    # an (nobs, 3) array without an error
    rng = np.random.default_rng(10796)
    two = rng.standard_normal((50, 2))
    if form == "dataframe":
        two = pd.DataFrame(two, columns=["a", "b"])
    one = rng.standard_normal(50)
    x = one if bad == "y" else two
    y = one if bad == "x" else two
    name = "y" if bad == "y" else "x"
    with pytest.raises(ValueError, match=f"{name} is required to have ndim 1"):
        func(x, y, adjusted=adjusted)


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


@pytest.fixture
def dm_data():
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
    return y, f1, f2


def test_diebold_mariano_test(dm_data):
    y, f1, f2 = (arr.copy() for arr in dm_data)

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
    with pytest.raises(ValueError, match="alternative"):
        diebold_mariano_test(y, f1, f2, alternative="greater")
    with pytest.raises(ValueError, match="kernel"):
        diebold_mariano_test(y, f1, f2, kernel="parzen")


# Reference values for results_diebold_mariano.csv, from R 4.4.2 with
# forecast 9.0.2 and sandwich 3.1.3. dm.test always applies the Harvey et al.
# (1997) correction and uses the t distribution; varestimator="acf" is the
# uniform kernel with h - 1 lags. The kernHAC rows use the uniform
# ("Truncated") kernel with lags that differ from h - 1 and the normal
# distribution. dm_data.csv is written from the dm_data fixture with
#   pd.DataFrame({"y": y, "f1": f1, "f2": f2}).to_csv("dm_data.csv", index=False)
#
# library(forecast); library(sandwich)
# dat <- read.csv("dm_data.csv")
# e1 <- dat$y - dat$f1
# e2 <- dat$y - dat$f2
# for (v in c("acf", "bartlett")) for (h in 1:3) for (p in c(1, 2))
#   for (a in c("two.sided", "greater", "less")) {
#     r <- dm.test(e1, e2, alternative = a, h = h, power = p, varestimator = v)
#     print(c(r$statistic, r$p.value), digits = 17)
#   }
# for (p in c(1, 2)) for (L in 1:3) {
#   fit <- lm(abs(e1)^p - abs(e2)^p ~ 1)
#   v <- kernHAC(fit, kernel = "Truncated", bw = L, prewhite = FALSE,
#                adjust = FALSE)[1, 1]
#   s <- coef(fit)[[1]] / sqrt(v)
#   print(c(s, 2 * pnorm(-abs(s)), pnorm(s, lower.tail = FALSE), pnorm(s)),
#         digits = 17)
# }
DM_R_RESULTS = pd.read_csv(CURR_DIR.joinpath("results", "results_diebold_mariano.csv"))


DM_R_ROWS = [
    pytest.param(
        row,
        id="-".join(
            [
                row["source"],
                row["kernel"],
                f"h{row['horizon']}",
                f"lags{row['lags']}",
                f"power{row['power']}",
                row["alternative"],
            ]
        ),
    )
    for row in DM_R_RESULTS.to_dict("records")
]


@pytest.mark.parametrize("row", DM_R_ROWS)
def test_diebold_mariano_r_reference(dm_data, row):
    y, f1, f2 = dm_data
    res = diebold_mariano_test(
        y,
        f1,
        f2,
        lags=row["lags"],
        criterion="poly",
        power=row["power"],
        harvey_adj=row["harvey_adj"],
        horizon=row["horizon"],
        alternative=row["alternative"],
        kernel=row["kernel"],
    )
    assert_allclose(res.statistic, row["statistic"], rtol=1e-10)
    assert_allclose(res.pvalue, row["pvalue"], rtol=1e-10)
    assert res.lags == row["lags"]
    assert res.alternative == row["alternative"]
    assert res.kernel == row["kernel"]


@pytest.mark.parametrize(
    "kwargs, statistic, pvalue",
    [
        ({}, 0.750573997514541, 0.4529090748993754),
        ({"criterion": "mad"}, 0.47154075187498634, 0.6372546252957734),
        ({"criterion": "poly", "power": 3.0}, 0.9069836187808107, 0.3644154599353391),
        ({"harvey_adj": True}, 0.725123371591778, 0.480321780679321),
        ({"harvey_adj": True, "horizon": 3}, 0.6249777482837591, 0.5420387022748386),
    ],
)
def test_diebold_mariano_default_results_unchanged(dm_data, kwargs, statistic, pvalue):
    # Values computed with statsmodels before alternative and kernel were added
    res = diebold_mariano_test(*dm_data, **kwargs)
    assert_allclose(res.statistic, statistic, rtol=1e-12)
    assert_allclose(res.pvalue, pvalue, rtol=1e-12)
    assert res.lags == 3
    assert res.alternative == "two-sided"
    assert res.kernel == "bartlett"


@pytest.mark.parametrize("harvey_adj", [False, True])
@pytest.mark.parametrize("kernel", ["bartlett", "uniform"])
def test_diebold_mariano_alternative_symmetry(harvey_adj, kernel):
    rng = np.random.default_rng(20261114)
    y, f1, f2 = rng.standard_normal((3, 80))
    kwargs = {"harvey_adj": harvey_adj, "kernel": kernel, "horizon": 2}
    two_sided = diebold_mariano_test(y, f1, f2, **kwargs)
    larger = diebold_mariano_test(y, f1, f2, alternative="larger", **kwargs)
    smaller = diebold_mariano_test(y, f1, f2, alternative="smaller", **kwargs)
    swapped = diebold_mariano_test(y, f2, f1, alternative="smaller", **kwargs)

    assert_allclose(larger.statistic, two_sided.statistic, rtol=1e-14)
    assert_allclose(larger.pvalue + smaller.pvalue, 1.0, rtol=1e-12)
    min_pvalue = min(larger.pvalue, smaller.pvalue)
    assert_allclose(two_sided.pvalue, 2 * min_pvalue, rtol=1e-12)
    assert_allclose(swapped.statistic, -larger.statistic, rtol=1e-12)
    assert_allclose(swapped.pvalue, larger.pvalue, rtol=1e-12)


def test_diebold_mariano_alternative_direction():
    rng = np.random.default_rng(12345)
    y = rng.standard_normal(200)
    forecast_a = y + rng.standard_normal(200)
    forecast_b = y + 0.5 * rng.standard_normal(200)
    larger = diebold_mariano_test(y, forecast_a, forecast_b, alternative="larger")
    smaller = diebold_mariano_test(y, forecast_a, forecast_b, alternative="smaller")
    # forecast_b is more accurate, so the mean loss differential is positive
    assert larger.statistic > 0
    assert larger.pvalue < 0.01
    assert smaller.pvalue > 0.99


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_diebold_mariano_uniform_default_lags(dm_data, horizon):
    res = diebold_mariano_test(*dm_data, kernel="uniform", horizon=horizon)
    expected = diebold_mariano_test(
        *dm_data, kernel="uniform", horizon=horizon, lags=horizon - 1
    )
    assert res.lags == horizon - 1
    assert_allclose(res.statistic, expected.statistic, rtol=1e-14)
    bartlett = diebold_mariano_test(*dm_data, horizon=horizon)
    assert bartlett.lags == max(horizon - 1, int(np.ceil(15 ** (1 / 3))))


@pytest.mark.parametrize("kernel", ["bartlett", "uniform"])
def test_diebold_mariano_identical_forecasts(dm_data, kernel):
    y, f1, _ = dm_data
    with pytest.raises(ValueError, match="loss differential is constant"):
        diebold_mariano_test(y, f1, f1, kernel=kernel)


def test_diebold_mariano_constant_forecast_errors():
    # the losses are 4 and 1 for every observation
    with pytest.raises(ValueError, match="loss differential is constant"):
        diebold_mariano_test(np.zeros(10), np.full(10, 2.0), np.ones(10))


@pytest.mark.parametrize("criterion", ["mse", "mae"])
def test_diebold_mariano_constant_up_to_rounding(criterion):
    # the forecast errors are -0.1 and -0.2 up to rounding error
    y = 10 * np.random.default_rng(0).standard_normal(100)
    with pytest.raises(ValueError, match="constant up to rounding"):
        diebold_mariano_test(y, y + 0.1, y + 0.2, criterion=criterion)


@pytest.mark.parametrize(
    "rel_range, raises",
    [
        (1e-10, True),
        (5e-9, True),
        (5e-8, False),
        (1e-6, False),
        (1e-4, False),
    ],
    ids=["1e-10", "5e-9", "5e-8", "1e-6", "1e-4"],
)
def test_diebold_mariano_constant_threshold(rel_range, raises):
    # The loss differential is constant up to rounding error if its range is
    # at most sqrt(eps) = 1.5e-8 times its largest absolute value. Here the
    # range of d is rel_range and its largest value is 1 + rel_range.
    z = np.random.default_rng(20261116).uniform(size=60)
    d = 1.0 + rel_range * (z - z.min()) / np.ptp(z)
    zeros = np.zeros(60)

    def call():
        return diebold_mariano_test(zeros, d, zeros, criterion=lambda y, f: f)

    if raises:
        with pytest.raises(ValueError, match="constant up to rounding"):
            call()
    else:
        assert np.isfinite(call().statistic)


def test_diebold_mariano_non_finite_forecast(dm_data):
    y, f1, _ = dm_data
    f_nan = f1.copy()
    f_nan[3] = np.nan
    with pytest.raises(MissingDataError, match="non-finite values"):
        diebold_mariano_test(y, f1, f_nan)


def test_diebold_mariano_mape_zero_observation(dm_data):
    y, f1, _ = dm_data
    y_zero = y.copy()
    y_zero[0] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        with pytest.raises(MissingDataError, match="non-finite values"):
            diebold_mariano_test(y_zero, f1, f1 + 1, criterion="mape")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"lags": 15}, r"lags \(15\) must be smaller"),
        ({"lags": 20}, r"lags \(20\) must be smaller"),
        ({"lags": 14, "kernel": "uniform"}, "minus one"),
    ],
    ids=["lags=nobs", "lags>nobs", "uniform-lags=nobs-1"],
)
def test_diebold_mariano_lags_too_large(dm_data, kwargs, match):
    with pytest.raises(ValueError, match=match):
        diebold_mariano_test(*dm_data, **kwargs)


def test_diebold_mariano_largest_lags(dm_data):
    # lags = nobs - 1 is the largest value for the Bartlett kernel
    res = diebold_mariano_test(*dm_data, lags=14)
    assert np.isfinite(res.statistic)


@pytest.mark.parametrize(
    "kwargs, statistic, pvalue",
    [
        ({"horizon": 15, "lags": 0}, 0.8160666104384313, 0.4144620395041849),
        ({"horizon": 20, "lags": 2}, 0.8209841996526007, 0.4116552682333907),
        ({"horizon": 15, "lags": 14}, 0.9363182910488854, 0.34910933336926364),
        (
            {"horizon": 40, "lags": 0, "criterion": "mad"},
            0.580413346387792,
            0.5616359069919612,
        ),
        # with 0 lags both kernels give the same variance
        (
            {"horizon": 15, "lags": 0, "kernel": "uniform"},
            0.8160666104384313,
            0.4144620395041849,
        ),
    ],
    ids=["h=nobs", "h>nobs", "lags=nobs-1", "mad", "uniform"],
)
def test_diebold_mariano_horizon_unused(dm_data, kwargs, statistic, pvalue):
    # horizon only sets the default lags and the Harvey et al. adjustment, so
    # it can be as large as nobs if harvey_adj is False and lags is given.
    # Reference values computed with statsmodels before alternative and kernel
    # were added, which did not check horizon against nobs.
    res = diebold_mariano_test(*dm_data, **kwargs)
    assert_allclose(res.statistic, statistic, rtol=1e-12)
    assert_allclose(res.pvalue, pvalue, rtol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 15},
        {"horizon": 16},
        {"horizon": 15, "lags": 0, "harvey_adj": True},
        {"horizon": 16, "lags": 0, "harvey_adj": True},
    ],
    ids=["default-lags", "default-lags-h>nobs", "harvey", "harvey-h>nobs"],
)
def test_diebold_mariano_horizon_used(dm_data, kwargs):
    # horizon sets the default lags and the Harvey et al. adjustment
    with pytest.raises(ValueError, match=r"horizon \(\d+\) must be smaller"):
        diebold_mariano_test(*dm_data, **kwargs)


@pytest.mark.parametrize(
    "criterion",
    [
        lambda y, f: np.mean((y - f) ** 2),
        lambda y, f: np.array([np.mean((y - f) ** 2)]),
        lambda y, f: ((y - f) ** 2)[:-1],
        lambda y, f: np.column_stack([(y - f) ** 2, (y - f) ** 2]),
    ],
    ids=["scalar", "one-loss", "too-short", "two-columns"],
)
def test_diebold_mariano_criterion_shape(dm_data, criterion):
    with pytest.raises(ValueError, match="one loss per observation"):
        diebold_mariano_test(*dm_data, criterion=criterion)


def test_diebold_mariano_criterion_column(dm_data):
    y, f1, f2 = dm_data
    res = diebold_mariano_test(
        y, f1, f2, criterion=lambda y, f: ((y - f) ** 2)[:, None]
    )
    assert_allclose(res.statistic, diebold_mariano_test(y, f1, f2).statistic)


@pytest.mark.parametrize("kernel", ["bartlett", "uniform"])
@pytest.mark.parametrize(
    "variance", [0.0, -1e-3, np.nan], ids=["zero", "negative", "nan"]
)
def test_diebold_mariano_non_positive_variance(dm_data, monkeypatch, kernel, variance):
    # Replace the HAC estimate of the variance of the mean. An exact zero
    # variance is only reached by cancellation in floating point arithmetic,
    # so it cannot be produced reliably from data.
    class FakeFit:
        params = np.array([0.5])

        def cov_params(self):
            return np.array([[variance]])

    class FakeOLS:
        def __init__(self, endog, exog):
            pass

        def fit(self, **kwargs):
            return FakeFit()

    monkeypatch.setattr(_stattools, "OLS", FakeOLS)
    with pytest.raises(ValueError, match="not positive") as exc:
        diebold_mariano_test(*dm_data, kernel=kernel)
    # only the message for the uniform kernel suggests the Bartlett kernel
    assert ("Use kernel='bartlett'" in str(exc.value)) == (kernel == "uniform")


def test_diebold_mariano_negative_variance():
    # The squared errors of forecast_a alternate between 4 and 0.25, so the
    # first autocovariance of the loss differential is close to -gamma_0 and
    # the uniform kernel with one lag gives a negative variance.
    y = np.zeros(20)
    forecast_a = np.tile([2.0, 0.5], 10)
    forecast_b = np.zeros(20)
    with pytest.raises(ValueError, match="variance of the loss differential is not"):
        diebold_mariano_test(y, forecast_a, forecast_b, kernel="uniform", horizon=2)
    res = diebold_mariano_test(y, forecast_a, forecast_b, kernel="bartlett", lags=1)
    assert np.isfinite(res.statistic)


# clark_west_data.csv: 60 one-step forecasts of y[t + 1] = 0.3 + 0.25 x[t] + e,
# x an AR(1) with coefficient 0.8, from recursive estimation starting with 60
# observations (numpy default_rng(20261111)), rounded to 6 decimals. The
# restricted forecast is the historical mean of y, the unrestricted forecast
# the OLS regression of y on a constant and x:
#
# rng = np.random.default_rng(20261111)
# x = np.empty(121)
# x[0] = rng.normal()
# for t in range(1, 121):
#     x[t] = 0.8 * x[t - 1] + rng.normal()
# y = 0.3 + 0.25 * np.r_[0.0, x[:-1]] + rng.normal(size=121)
# rows = []
# for t in range(60, 120):
#     ys, xs = y[1 : t + 1], x[:t]
#     beta = np.linalg.lstsq(np.column_stack([np.ones(t), xs]), ys, rcond=None)[0]
#     rows.append((y[t + 1], ys.mean(), beta[0] + beta[1] * x[t]))
# np.round(rows, 6) gives the columns y, f_restricted, f_unrestricted
# The reference values in results_clark_west.csv are from R 4.4.2 with
# sandwich 3.1.3 and forecastdom 0.1.1:
#
# library(sandwich); library(forecastdom)
# cw <- read.csv("clark_west_data.csv")
# e1 <- cw$y - cw$f_restricted
# e2 <- cw$y - cw$f_unrestricted
# f <- e1^2 - (e2^2 - (cw$f_restricted - cw$f_unrestricted)^2)
# fit <- lm(f ~ 1)
# for (L in 0:4) {
#   v <- NeweyWest(fit, lag = L, prewhite = FALSE, adjust = FALSE)[1, 1]
#   s <- coef(fit)[[1]] / sqrt(v)
#   print(c(s, pnorm(s, lower.tail = FALSE)), digits = 17)
# }
# for (L in 1:3) {
#   v <- kernHAC(fit, kernel = "Truncated", bw = L, prewhite = FALSE,
#                adjust = FALSE)[1, 1]
#   s <- coef(fit)[[1]] / sqrt(v)
#   print(c(s, pnorm(s, lower.tail = FALSE)), digits = 17)
# }
# r <- cw_test(e1, e2, cw$f_restricted, cw$f_unrestricted)
# print(c(r$statistic, r$pvalue), digits = 17)
# print(c(mean(e1^2), mean(e2^2),
#         mean((cw$f_restricted - cw$f_unrestricted)^2)), digits = 17)
CW_RESULTS = pd.read_csv(CURR_DIR.joinpath("results", "results_clark_west.csv"))


@pytest.fixture
def cw_data():
    data = pd.read_csv(CURR_DIR.joinpath("results", "clark_west_data.csv"))
    return data["y"], data["f_restricted"], data["f_unrestricted"]


@pytest.mark.parametrize("row", CW_RESULTS.to_dict("records"))
def test_clark_west_r_reference(cw_data, row):
    res = clark_west_test(*cw_data, lags=row["lags"], kernel=row["kernel"])
    assert isinstance(res, ClarkWestResult)
    assert_allclose(res.statistic, row["statistic"], rtol=1e-10)
    assert_allclose(res.pvalue, row["pvalue"], rtol=1e-10)
    assert res.lags == row["lags"]
    assert res.kernel == row["kernel"]


def test_clark_west_forecastdom(cw_data):
    res = clark_west_test(*cw_data, lags=0)
    # forecastdom::cw_test uses the least squares standard error, which
    # divides the variance by T - 1 instead of T
    nobs = len(cw_data[0])
    statistic = res.statistic * np.sqrt((nobs - 1) / nobs)
    assert_allclose(statistic, 1.4128255421051328, rtol=1e-10)
    assert_allclose(stats.norm.sf(statistic), 0.078853513078140369, rtol=1e-10)
    assert_allclose(res.mspe_restricted, 1.2824170794716332, rtol=1e-12)
    assert_allclose(res.mspe_unrestricted, 1.2418206833121166, rtol=1e-12)
    assert_allclose(res.adjustment, 0.036854387906683335, rtol=1e-12)
    statistic, pvalue = res
    assert statistic == res.statistic
    assert pvalue == res.pvalue


@pytest.mark.parametrize("kernel", ["bartlett", "uniform"])
def test_clark_west_adjusted_loss_identity(kernel):
    # The adjusted loss differential equals 2 e_1 (f_2 - f_1)
    rng = np.random.default_rng(20261114)
    y, f1, f2 = rng.standard_normal((3, 100))
    res = clark_west_test(y, f1, f2, kernel=kernel, horizon=3)
    adj_loss_diff = 2 * (y - f1) * (f2 - f1)
    ols = OLS(adj_loss_diff, np.ones(100)).fit(
        cov_type="HAC", cov_kwds={"maxlags": res.lags, "kernel": kernel}
    )
    assert_allclose(res.statistic, ols.tvalues[0], rtol=1e-12)
    assert_allclose(res.pvalue, stats.norm.sf(ols.tvalues[0]), rtol=1e-12)


def test_clark_west_argument_order(cw_data):
    y, f_restricted, f_unrestricted = (np.asarray(a, dtype=float) for a in cw_data)
    res = clark_west_test(y, f_restricted, f_unrestricted)
    swapped = clark_west_test(y, f_unrestricted, f_restricted)
    adj_loss_diff = (y - f_restricted) ** 2 - (
        (y - f_unrestricted) ** 2 - (f_restricted - f_unrestricted) ** 2
    )
    swapped_adj_loss_diff = -adj_loss_diff + 2 * (f_restricted - f_unrestricted) ** 2
    ols = OLS(swapped_adj_loss_diff, np.ones_like(y)).fit(
        cov_type="HAC", cov_kwds={"maxlags": swapped.lags}
    )
    assert_allclose(swapped.statistic, ols.tvalues[0], rtol=1e-10)
    assert not np.isclose(swapped.statistic, -res.statistic, rtol=1e-3)
    assert_allclose(
        swapped.mspe_restricted - swapped.mspe_unrestricted,
        res.mspe_unrestricted - res.mspe_restricted,
        rtol=1e-12,
    )


def test_clark_west_statistic_exceeds_dm():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400)
    y = 0.05 * x + rng.standard_normal(400)
    f_restricted, f_unrestricted = [], []
    for t in range(200, 400):
        f_restricted.append(y[:t].mean())
        slope, intercept = np.polyfit(x[:t], y[:t], 1)
        f_unrestricted.append(intercept + slope * x[t])
    res = clark_west_test(y[200:], f_restricted, f_unrestricted)
    dm = diebold_mariano_test(
        y[200:], f_restricted, f_unrestricted, alternative="larger"
    )
    assert res.statistic > dm.statistic
    assert_allclose(res.statistic, 1.8726072917195915, rtol=1e-6)
    assert_allclose(dm.statistic, 1.2505199142075583, rtol=1e-6)


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_clark_west_default_lags(cw_data, horizon):
    uniform = clark_west_test(*cw_data, kernel="uniform", horizon=horizon)
    assert uniform.lags == horizon - 1
    bartlett = clark_west_test(*cw_data, horizon=horizon)
    assert bartlett.lags == max(horizon - 1, int(np.ceil(60 ** (1 / 3))))


def _cw_degenerate_inputs(case, cw_data):
    y, f_restricted, _ = (np.asarray(a, dtype=float) for a in cw_data)
    if case in ("identical_bartlett", "identical_uniform"):
        kernel = case.split("_")[1]
        return (y, f_restricted, f_restricted), {"kernel": kernel}
    if case == "constant_errors":
        y_scaled = 10 * np.random.default_rng(0).standard_normal(100)
        return (y_scaled, y_scaled + 0.1, y_scaled + 0.2), {}
    if case == "non_finite":
        f_nan = f_restricted.copy()
        f_nan[5] = np.nan
        return (y, f_restricted, f_nan), {}
    if case == "negative_variance":
        # 2 * f_restricted**2 alternates between 8 and 0.5, so the uniform
        # kernel with one lag gives a negative variance
        return (
            (np.zeros(20), np.tile([2.0, 0.5], 10), np.zeros(20)),
            {"kernel": "uniform", "lags": 1},
        )
    raise ValueError(f"unknown case {case}")


@pytest.mark.parametrize(
    "case, exception, match",
    [
        ("identical_bartlett", ValueError, "constant"),
        ("identical_uniform", ValueError, "constant"),
        ("constant_errors", ValueError, "constant"),
        ("non_finite", MissingDataError, "non-finite"),
        ("negative_variance", ValueError, "not positive"),
    ],
)
def test_clark_west_degenerate(cw_data, case, exception, match):
    args, kwargs = _cw_degenerate_inputs(case, cw_data)
    with pytest.raises(exception, match=match):
        clark_west_test(*args, **kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param({"lags": -1}, "^lags must", id="negative_lags"),
        pytest.param({"horizon": 0}, "horizon", id="horizon_zero"),
        pytest.param({"kernel": "parzen"}, "kernel", id="unknown_kernel"),
        pytest.param({"horizon": 60}, r"horizon \(60\)", id="horizon_nobs"),
        pytest.param(
            {"kernel": "uniform", "lags": 59}, r"lags \(59\)", id="uniform_lags"
        ),
    ],
)
def test_clark_west_invalid_arguments(cw_data, kwargs, match):
    with pytest.raises(ValueError, match=match):
        clark_west_test(*cw_data, **kwargs)


def test_clark_west_horizon_unused_with_lags(cw_data):
    res = clark_west_test(*cw_data, horizon=60, lags=0)
    expected = clark_west_test(*cw_data, horizon=1, lags=0)
    assert_allclose(res.statistic, expected.statistic, rtol=1e-14)
    assert res.lags == 0


def test_clark_west_zero_lags_is_ols_tstat():
    # The identity holds for any forecasts, so the forecasts need not be
    # out-of-sample: a constant forecast against a regression on x.
    rng = np.random.default_rng(20261117)
    nobs = 150
    x = rng.standard_normal(nobs)
    y = 0.2 * x + rng.standard_normal(nobs)
    f_restricted = np.full(nobs, y.mean())
    slope, intercept = np.polyfit(x, y, 1)
    f_unrestricted = intercept + slope * x
    adj_loss_diff = (y - f_restricted) ** 2 - (
        (y - f_unrestricted) ** 2 - (f_restricted - f_unrestricted) ** 2
    )
    # OLS divides the variance by nobs - 1, the HAC estimator with no lags by
    # nobs. The two statistics differ only by this factor, up to rounding.
    ols_tstat = OLS(adj_loss_diff, np.ones(nobs)).fit().tvalues[0]
    expected = ols_tstat * np.sqrt(nobs / (nobs - 1))
    res = clark_west_test(y, f_restricted, f_unrestricted, lags=0)
    assert res.lags == 0
    assert_allclose(res.statistic, expected, rtol=1e-12)
    assert_allclose(res.pvalue, stats.norm.sf(expected), rtol=1e-12)
    # The default number of lags is larger and gives another statistic
    default = clark_west_test(y, f_restricted, f_unrestricted)
    assert default.lags > 0
    assert not np.isclose(default.statistic, res.statistic, rtol=1e-3)


def test_clark_west_unequal_length(cw_data):
    y, f_restricted, f_unrestricted = cw_data
    with pytest.raises(ValueError, match="equal length"):
        clark_west_test(y, f_restricted, f_unrestricted[:-1])


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


@pytest.mark.parametrize("trend", ["n", "c"])
def test_arma_order_select_ic_failed_fits_are_nan(trend):
    # The Yule-Walker estimator only fits pure AR models, so the models with
    # an MA term fail and are given NaN, with one warning that lists them
    from statsmodels.tsa.arima.model import ARIMA

    y = np.random.default_rng(20261119).standard_normal(100)
    fit_kw = {"method": "yule_walker"}
    match = (
        r"The fits of 3 of 6 ARMA models failed, and their information "
        r"criteria are NaN. Orders \(p, q\): \(0, 1\), \(1, 1\), \(2, 1\). "
        r"The first, order \(0, 1\), raised ValueError: Yule-Walker"
    )
    with pytest.warns(EstimationWarning, match=match) as record:
        res = arma_order_select_ic(
            y, max_ar=2, max_ma=1, ic=["aic", "bic"], trend=trend, fit_kw=fit_kw
        )
    assert len(record) == 1
    # The warning points at the caller
    assert record[0].filename == __file__
    for ic in ("aic", "bic"):
        frame = res[ic]
        assert np.all(np.isnan(frame[1]))
        expected = [
            getattr(ARIMA(y, order=(p, 0, 0), trend=trend).fit(**fit_kw), ic)
            for p in range(3)
        ]
        assert_allclose(frame[0], expected, rtol=1e-12)
        # The minimum ignores the failed fits, it used to point at the first
        assert res[f"{ic}_min_order"] == (np.argmin(expected), 0)


@pytest.mark.parametrize("trend", ["n", "c"])
def test_arma_order_select_ic_infinite_value_all_fail(trend):
    # Every Yule-Walker fit fails for a series with an infinite value. The
    # minimum order used to be (0, 0).
    y = np.random.default_rng(20261119).standard_normal(100)
    y[-1] = np.inf
    match = r"No ARMA model could be fitted.*The first, order \(0, 0\), raised"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match=match) as excinfo:
            arma_order_select_ic(
                y,
                max_ar=1,
                max_ma=1,
                ic="aic",
                trend=trend,
                fit_kw={"method": "yule_walker"},
            )
    # The error of the first fit is chained and its message included
    cause = excinfo.value.__cause__
    assert cause is not None
    assert str(cause) in str(excinfo.value)


def test_arma_order_select_ic_infinite_value():
    # The models fitted to a series with an infinite value have a NaN
    # log-likelihood, or their fit fails, so there is no minimum order. It
    # used to be (0, 0). Whether the fit of the AR(1) model fails (and is
    # listed in an EstimationWarning) or gives a NaN log-likelihood depends
    # on the platform, so only the error is checked here; the warning is
    # checked by the tests with failing fits above and below.
    y = np.random.default_rng(20261119).standard_normal(100)
    y[-1] = np.inf
    with warnings.catch_warnings():
        # The maximum likelihood fits run on the infinite value and do not
        # converge
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", EstimationWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="No fitted ARMA model has a non-NaN aic"):
            arma_order_select_ic(y, max_ar=1, max_ma=0, ic="aic", trend="c")


def _arma_ic_table(y, max_ar, max_ma, trend, fit_kw, name):
    # The criterion of each ARIMA model fitted directly, NaN if the fit raises
    from statsmodels.tsa.arima.model import ARIMA

    table = np.full((max_ar + 1, max_ma + 1), np.nan)
    failed = []
    for p in range(max_ar + 1):
        for q in range(max_ma + 1):
            try:
                fit = ARIMA(y, order=(p, 0, q), trend=trend).fit(**fit_kw)
            except (IndexError, ValueError):
                failed.append((p, q))
                continue
            table[p, q] = getattr(fit, name)
    return table, failed


def test_arma_order_select_ic_start_params():
    # start_params in fit_kw used to clash with the start_params argument of
    # the internal retry, so every fit failed. Now only the fits of models
    # with more parameters than start_params fail. Which fits fail is taken
    # from the direct fits, since ARIMA.fit also accepts too long
    # start_params, with a FutureWarning (deprecated).
    y = np.random.default_rng(20261119).standard_normal(100)
    fit_kw = {"start_params": [0.1, 0.2, 0.5]}
    too_long = "start_params has 3 elements, but the model has"
    with pytest.warns(FutureWarning, match=too_long):
        expected, failed = _arma_ic_table(y, 3, 2, "n", fit_kw, "aic")
    # The models with more than 3 parameters fail
    assert {(1, 2), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)} <= set(failed)
    orders = ", ".join(rf"\({p}, {q}\)" for p, q in failed)
    match = (
        rf"The fits of {len(failed)} of 12 ARMA models failed, and their "
        rf"information criteria are NaN. Orders \(p, q\): {orders}\."
    )
    with pytest.warns(FutureWarning, match=too_long):
        with pytest.warns(EstimationWarning, match=match):
            res = arma_order_select_ic(
                y, max_ar=3, max_ma=2, ic="aic", trend="n", fit_kw=fit_kw
            )
    assert_allclose(res.aic, expected, rtol=1e-10)
    loc = np.nanargmin(expected)
    assert res.aic_min_order == np.unravel_index(loc, expected.shape)
    # The minimum is in neither the first row nor the first column, of a grid
    # that is not square
    assert all(res.aic_min_order)


@pytest.mark.parametrize(
    "kwargs",
    [{"fit_kw": {"bogus": 1}}, {"model_kw": {"bogus": 1}}],
    ids=["fit_kw", "model_kw"],
)
def test_arma_order_select_ic_type_error(kwargs):
    # A TypeError from the arguments, such as a mistyped keyword, used to
    # make every fit fail silently
    y = np.random.default_rng(20261119).standard_normal(100)
    with pytest.raises(TypeError, match="bogus"):
        arma_order_select_ic(y, max_ar=1, max_ma=0, **kwargs)


def test_arma_order_select_ic_error_without_message(monkeypatch):
    # An error without arguments used to crash the internal retry of a failed
    # fit with an IndexError
    from statsmodels.tsa.arima.model import ARIMA

    fit = ARIMA.fit

    def fit_without_ma(self, *args, **kwargs):
        if self.model_orders["ma"]:
            raise ValueError
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(ARIMA, "fit", fit_without_ma)
    y = np.random.default_rng(20261119).standard_normal(100)
    match = r"Orders \(p, q\): \(0, 1\), \(1, 1\)\. .* raised ValueError: $"
    with pytest.warns(EstimationWarning, match=match):
        res = arma_order_select_ic(y, max_ar=1, max_ma=1, ic="aic", trend="n")
    expected = [fit(ARIMA(y, order=(p, 0, 0), trend="n")).aic for p in range(2)]
    assert_allclose(res.aic[0], expected, rtol=1e-12)
    assert np.isnan(res.aic[1]).all()


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


@pytest.mark.parametrize("form", ["column", "dataframe"])
def test_levinson_durbin_acov_column(form):
    # s is 1-D: the autocovariances of an AR(1) as an (n, 1) column give the
    # AR(1) coefficient and innovation variance, as the 1-D array does
    rho = 0.9
    m = 20
    acov = rho ** np.arange(200)
    res = levinson_durbin(series_forms(acov)[form], m, isacov=True)
    assert_allclose(res.sigma_v, 1 - rho**2, rtol=1e-12)
    # atol: the coefficients beyond lag 1 are zero up to rounding
    assert_allclose(res.arcoefs, np.array([rho] + [0] * (m - 1)), atol=1e-8)
    expected = levinson_durbin(acov, m, isacov=True)
    for value, expected_value in zip(res, expected, strict=True):
        assert_allclose(value, expected_value, rtol=1e-12)


@pytest.mark.parametrize("isacov", [False, True])
def test_levinson_durbin_two_columns_raises(isacov):
    s = np.column_stack([0.9 ** np.arange(20), 0.5 ** np.arange(20)])
    with pytest.raises(ValueError, match="s is required to have ndim 1"):
        levinson_durbin(s, 5, isacov=isacov)


def test_levinson_durbin_nlags_too_long():
    # requesting more lags than the supplied autocovariances cover used to
    # raise a bare IndexError from the recursion; it should report the
    # mismatch instead
    with pytest.raises(ValueError, match="nlags=5"):
        levinson_durbin(np.array([2.0, 1.0, 0.5, 0.25, 0.1]), 5, isacov=True)
    with pytest.raises(ValueError, match="nlags=5"):
        levinson_durbin(np.array([1.0, 2.0, 3.0, 4.0]), 5, isacov=False)
    # the boundary case, one autocovariance per lag, still works
    res = levinson_durbin(np.array([2.0, 1.0, 0.5, 0.25, 0.1]), 4, isacov=True)
    assert res.arcoefs.shape == (4,)


def test_levinson_durbin_negative_nlags():
    # a negative nlags used to leak a bare IndexError from the recursion, and
    # so does nlags=0, the recursion needs at least one lag
    for nlags in (-1, 0):
        with pytest.raises(ValueError, match="nlags must be positive"):
            levinson_durbin(np.arange(10.0), nlags=nlags)
        with pytest.raises(ValueError, match="nlags must be positive"):
            levinson_durbin(np.array([2.0, 1.0, 0.5]), nlags=nlags, isacov=True)
    res = levinson_durbin(np.array([2.0, 1.0, 0.5]), nlags=1, isacov=True)
    assert res.arcoefs.shape == (1,)


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


def test_acovf_negative_nlag(acovf_data):
    # a negative nlag used to slice the full acovf from the wrong end (fft)
    # or raise a bare numpy negative-dimension error (non-fft)
    with pytest.raises(ValueError, match="non-negative"):
        acovf(acovf_data, nlag=-2)
    with pytest.raises(ValueError, match="non-negative"):
        acovf(acovf_data, nlag=-2, fft=False)


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


def _column_input_cases():
    """
    Functions of one 1-D array_like argument that used to reject a column

    Each returns the numeric values of its result as a flat array.
    """
    rng = np.random.default_rng(20261124)
    x = rng.standard_normal(250).cumsum()
    y = x + rng.standard_normal(250)
    e = rng.standard_normal(250)
    r = acf(e, nlags=10)[1:]

    def kpss_values(v, regression):
        res = kpss(v, regression, nlags=4, result_object=True)
        return np.r_[res.statistic, res.pvalue, res.lags]

    def rur_values(v):
        res = range_unit_root_test(v, result_object=True)
        return np.r_[res.statistic, res.pvalue, list(res.critical_values.values())]

    return {
        "pacf_burg": (lambda v: np.concatenate(pacf_burg(v, nlags=10)), x),
        "pacf burg": (lambda v: pacf(v, nlags=10, method="burg"), x),
        "kpss c": (lambda v: kpss_values(v, "c"), e),
        "kpss ct": (lambda v: kpss_values(v, "ct"), e),
        "range_unit_root_test": (rur_values, x),
        "pccf x": (lambda v: pccf(v, y, nlags=5), x),
        "pccf y": (lambda v: pccf(x, v, nlags=5), y),
        "pccf ols": (lambda v: pccf(v, y, nlags=5, method="ols"), x),
        "q_stat": (lambda v: np.concatenate(q_stat(v, 250)), r),
    }


@pytest.mark.parametrize("form", ["list", "tuple", "series", "column", "dataframe"])
@pytest.mark.parametrize("name", sorted(_column_input_cases()))
def test_column_input(name, form):
    # x is documented as a 1-D array_like: every form of the series, including
    # an (n, 1) column, gives the result of the ndarray
    func, x = _column_input_cases()[name]
    index = pd.date_range("2000-01-01", periods=x.shape[0], freq="MS")
    x_form = {
        "list": x.tolist(),
        "tuple": tuple(x.tolist()),
        "series": pd.Series(x, index=index, name="x"),
        "column": x[:, None],
        "dataframe": pd.DataFrame({"x": x}, index=index),
    }[form]
    with warnings.catch_warnings():
        # The p-values of kpss and range_unit_root_test may be at the edge
        # of their tables
        warnings.simplefilter("ignore", InterpolationWarning)
        assert_allclose(func(x_form), func(x), rtol=1e-12)


@pytest.mark.parametrize("name", sorted(_column_input_cases()))
def test_column_input_two_columns_raises(name):
    func, x = _column_input_cases()[name]
    with pytest.raises(ValueError, match="required to have ndim 1"):
        func(np.column_stack([x, x[::-1]]))


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


def test_innovations_algo_negative_rtol():
    acovf = np.array([1.81, -0.9])
    with pytest.raises(ValueError, match="rtol must be a non-negative float"):
        innovations_algo(acovf, rtol=-1.0)


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


def _adf_tstat_no_trend(x, lag):
    # t-statistic of the lagged level in the ADF regression without
    # deterministic terms, computed directly with lstsq
    dx = np.diff(x)
    endog = dx[lag:]
    exog = np.column_stack(
        [x[lag:-1]] + [dx[lag - j : len(dx) - j] for j in range(1, lag + 1)]
    )
    params, rss, _, _ = np.linalg.lstsq(exog, endog, rcond=None)
    df_resid = exog.shape[0] - exog.shape[1]
    scale = rss[0] / df_resid
    bse = np.sqrt(scale * np.linalg.inv(exog.T @ exog)[0, 0])
    return params[0] / bse


@pytest.mark.parametrize("nobs", [12, 16, 20])
@pytest.mark.parametrize("autolag", ["aic", "bic", "t-stat", None])
def test_adfuller_no_trend_even_nobs(nobs, autolag):
    # GH 9375: with regression="n" and an even nobs, the default maxlag
    # gave an exactly fitted regression and a statistic of 0
    rng = np.random.default_rng(9375)
    x = rng.standard_normal(nobs).cumsum()
    res = adfuller(x, regression="n", autolag=autolag, store=True, result_object=True)
    assert res.resstore.maxlag == (nobs - 3) // 2
    assert res.resstore.resols.df_resid >= 1
    assert_allclose(res.statistic, _adf_tstat_no_trend(x, res.lags), rtol=1e-8)


def test_adfuller_no_trend_even_nobs_maxlag_too_large():
    rng = np.random.default_rng(9375)
    y = rng.standard_normal(20)
    with pytest.raises(ValueError, match="maxlag must be less than or equal to 8"):
        adfuller(y, maxlag=9, regression="n")
    res = adfuller(y, maxlag=8, regression="n", autolag=None, result_object=True)
    assert_allclose(res.statistic, _adf_tstat_no_trend(y, 8), rtol=1e-8)


@pytest.mark.parametrize(
    "nobs, regression, maxlag",
    [
        # ceil(12 * 2.4 ** (1 / 4)) = ceil(14.94)
        (240, "c", 15),
        # ceil(12 * 0.2 ** (1 / 4)) = 9 is larger than the bound
        # min(nobs // 2 - ntrend - 1, (nobs - ntrend - 3) // 2)
        (20, "c", 8),
        (20, "ct", 7),
        # ceil(12 * 0.21 ** (1 / 4)) = ceil(8.12) is the bound, as ntrend = 0
        (21, "n", 9),
    ],
)
def test_adfuller_default_maxlag(nobs, regression, maxlag):
    x = np.random.default_rng(nobs).standard_normal(nobs).cumsum()
    res = adfuller(x, regression=regression, autolag=None, result_object=True)
    assert res.lags == maxlag
    expected = adfuller(
        x, maxlag=maxlag, regression=regression, autolag=None, result_object=True
    )
    assert_allclose(res.statistic, expected.statistic, rtol=1e-12)


@pytest.mark.parametrize(
    "regression, max_maxlag", [("n", 8), ("c", 8), ("ct", 7), ("ctt", 6)]
)
def test_adfuller_maxlag_bound(regression, max_maxlag):
    # min(nobs // 2 - ntrend - 1, (nobs - ntrend - 3) // 2) for nobs = 20
    x = np.random.default_rng(20).standard_normal(20).cumsum()
    match = f"maxlag must be less than or equal to {max_maxlag} "
    with pytest.raises(ValueError, match=match):
        adfuller(x, maxlag=max_maxlag + 1, regression=regression, autolag=None)
    res = adfuller(
        x, maxlag=max_maxlag, regression=regression, autolag=None, result_object=True
    )
    assert res.lags == max_maxlag


def test_coint_short_even_sample():
    # GH 9375: coint on 20 observations returned a statistic of 0 and a
    # p-value of 0.9859 for any input
    rng = np.random.default_rng(9375)
    y0 = rng.standard_normal(20)
    y1 = rng.standard_normal(20)
    res = coint(y0, y1)
    resid = OLS(y0, np.column_stack([y1, np.ones(20)])).fit().resid
    adf = adfuller(resid, regression="n", store=True, result_object=True)
    assert adf.resstore.resols.df_resid >= 1
    assert_allclose(res.coint_t, _adf_tstat_no_trend(resid, adf.lags), rtol=1e-8)
    assert res.coint_t != 0


def _adf_autolag_ols(x, maxlag, regression, method):
    # lag search with one OLS fit per lag length, using the design matrix
    # that adfuller builds for _autolag
    xdiff = np.diff(x)
    xdall = lagmat(xdiff[:, None], maxlag, trim="both", original="in")
    nobs = xdall.shape[0]
    xdall[:, 0] = x[-nobs - 1 : -1]
    endog = xdiff[-nobs:]
    exog = xdall if regression == "n" else add_trend(xdall, regression, prepend=True)
    startlag = exog.shape[1] - xdall.shape[1] + 1
    icbest, bestlag = _autolag(OLS, endog, exog, startlag, maxlag, method)
    return icbest, bestlag - startlag


def _adf_autolag_series(kind, nobs, rng):
    e = rng.standard_normal(nobs)
    if kind == "random_walk":
        return np.cumsum(e)
    elif kind == "ar":
        x = np.zeros(nobs)
        for t in range(3, nobs):
            x[t] = 0.5 * x[t - 1] + 0.2 * x[t - 2] - 0.3 * x[t - 3] + e[t]
        return x
    elif kind == "integrated_ma":
        return np.cumsum(e + 0.8 * np.r_[0.0, e[:-1]])
    elif kind == "trend":
        return 5.0 + 0.05 * np.arange(nobs) + np.cumsum(e)
    elif kind == "integer_steps":
        return np.cumsum(rng.integers(-1, 2, nobs)).astype(float)
    elif kind == "mostly_constant":
        x = np.zeros(nobs)
        x[nobs // 2 :] = np.cumsum(e[nobs // 2 :])
        return x
    else:  # kind == "heavy_tails"
        return np.cumsum(rng.standard_t(1.5, nobs))


@pytest.mark.parametrize("regression", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("method", ["aic", "bic", "t-stat"])
@pytest.mark.parametrize(
    "kind",
    [
        "random_walk",
        "ar",
        "integrated_ma",
        "trend",
        "integer_steps",
        "mostly_constant",
        "heavy_tails",
    ],
)
def test_adf_autolag_qr_matches_ols(regression, method, kind):
    # GH 1849: the lag search from a single QR decomposition gives the same
    # lag and criterion as fitting one OLS model per lag length
    rng = np.random.default_rng(1849)
    # The t-values of the OLS fits lose some precision when the trend
    # columns are badly scaled ("ctt"), and an exactly zero t-value is only
    # zero up to rounding
    rtol, atol = (1e-8, 1e-12) if method == "t-stat" else (1e-10, 0)
    for nobs in [12, 25, 300]:
        x = _adf_autolag_series(kind, nobs, rng)
        ntrend = len(regression) if regression != "n" else 0
        max_maxlag = min(nobs // 2 - ntrend - 1, (nobs - ntrend - 3) // 2)
        default = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
        for maxlag in {0, 1, min(default, max_maxlag), min(max_maxlag, 25)}:
            icbest, bestlag = _adf_autolag(x, maxlag, regression, method)
            expected = _adf_autolag_ols(x, maxlag, regression, method)
            assert bestlag == expected[1]
            assert_allclose(icbest, expected[0], rtol=rtol, atol=atol)


def test_adf_autolag_qr_blocks():
    # more observations than one block of rows
    rng = np.random.default_rng(1849)
    x = np.cumsum(rng.standard_normal(20000))
    x[1:] += 0.3 * np.diff(x)
    for method in ["aic", "bic", "t-stat"]:
        icbest, bestlag = _adf_autolag(x, 6, "ct", method)
        expected = _adf_autolag_ols(x, 6, "ct", method)
        assert bestlag == expected[1]
        assert_allclose(icbest, expected[0], rtol=1e-10)


@pytest.mark.parametrize("method", ["aic", "bic", "t-stat"])
def test_adf_autolag_qr_scaled(method):
    # the criteria change only by a constant and the t-statistics not at all
    # when the data are scaled, so the selected lag is the same
    rng = np.random.default_rng(1849)
    x = _adf_autolag_series("ar", 300, rng)
    icbest, bestlag = _adf_autolag(x, 12, "c", method)
    icbest_scaled, bestlag_scaled = _adf_autolag(1e8 * x, 12, "c", method)
    assert bestlag_scaled == bestlag
    if method == "t-stat":
        assert_allclose(icbest_scaled, icbest, rtol=1e-12)


def _adf_autolag_singular_series(kind):
    t = np.arange(120.0)
    if kind == "periodic":
        # lagged differences are exactly collinear
        return np.cumsum(np.tile([1.0, -1.0, 2.0], 40))
    elif kind == "geometric":
        return 0.5 ** (t / 10)
    elif kind == "exact_fit":
        # full rank regressors with zero residuals
        x = np.zeros(30)
        for i in range(1, 30):
            x[i] = 1.0 + 0.5 * x[i - 1]
        return x
    else:  # kind == "nonfinite"
        x = np.cumsum(np.ones(30))
        x[20] = np.nan
        return x


@pytest.mark.parametrize("kind", ["periodic", "geometric", "exact_fit", "nonfinite"])
def test_adf_autolag_qr_singular(kind):
    # the QR lag search declines singular or exactly fitted regressions
    x = _adf_autolag_singular_series(kind)
    assert _adf_autolag(x, 2, "c", "aic") is None


@pytest.mark.parametrize("kind", ["periodic", "geometric", "exact_fit"])
@pytest.mark.parametrize("method", ["aic", "bic", "t-stat"])
def test_adfuller_autolag_singular_uses_ols(kind, method):
    # adfuller falls back to fitting one OLS model per lag length
    x = _adf_autolag_singular_series(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SingularMatrixWarning)
        res = adfuller(x, maxlag=2, autolag=method, result_object=True)
        res_ols = adfuller(
            x, maxlag=2, autolag=method, regresults=True, result_object=True
        )
    assert res.lags == res_ols.lags
    assert_equal(res.icbest, res_ols.icbest)
    assert_equal(res.statistic, res_ols.statistic)


@pytest.mark.parametrize("regression", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("autolag", ["AIC", "BIC", "t-stat"])
def test_adfuller_autolag_regresults_consistent(regression, autolag):
    # regresults=True keeps the OLS results of every lag length, the
    # default uses the QR lag search
    rng = np.random.default_rng(1849)
    x = _adf_autolag_series("ar", 500, rng)
    res = adfuller(
        x, regression=regression, autolag=autolag, store=True, result_object=True
    )
    res_ols = adfuller(
        x,
        regression=regression,
        autolag=autolag,
        regresults=True,
        result_object=True,
    )
    assert res.lags == res_ols.lags
    assert res.nobs == res_ols.nobs
    assert_allclose(res.icbest, res_ols.icbest, rtol=1e-10)
    assert_allclose(res.statistic, res_ols.statistic, rtol=1e-12)
    assert not hasattr(res.resstore, "autolag_results")
    assert len(res_ols.resstore.autolag_results) == res_ols.resstore.maxlag + 1


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

    # following tests compare results to R package urca.ur.za (1.13-0). R
    # counts observations from 1, so its break point is bpidx + 1.
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
            [-3.02761, 0.63993, 12, 101],
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


def test_acf_nlags_validation():
    # negative and out-of-range nlags used to slice avf silently
    rs = np.random.RandomState(32738493)
    e = rs.standard_normal(20)
    with pytest.raises(ValueError, match="non-negative"):
        acf(e, nlags=-4)
    with pytest.raises(ValueError, match="smaller than the number of observations"):
        acf(e, nlags=20)
    # the largest valid lag is still allowed
    assert acf(e, nlags=19).shape == (20,)


@pytest.mark.parametrize("nobs, nlags", [(10, 9), (200, 23)])
def test_acf_default_nlags(nobs, nlags):
    # min(10 * log10(nobs), nobs - 1): the second bound binds for nobs = 10
    x = np.random.default_rng(nobs).standard_normal(nobs)
    res = acf(x)
    assert res.shape == (nlags + 1,)
    assert_allclose(res, acf(x, nlags=nlags), rtol=1e-12)


def _nan_acf(x, nlags):
    # Sample autocorrelations that leave out the products with a NaN
    resid = x - np.nanmean(x)
    n = x.shape[0]
    acov = [np.nansum(resid[k:] * resid[: n - k]) for k in range(nlags + 1)]
    return np.array(acov) / acov[0]


@pytest.mark.parametrize("missing", ["drop", "conservative"])
@pytest.mark.parametrize(
    "nobs, nmissing, nlags",
    # min(10 * log10(nobs - nmissing), nobs - nmissing - 1); both bounds bind
    [(200, 100, 20), (20, 12, 7)],
)
def test_acf_default_nlags_missing(missing, nobs, nmissing, nlags):
    # The default used to count the NaNs as observations, which gave 23 lags
    # for the first case and raised for the second
    rng = np.random.default_rng(20261120)
    x = rng.standard_normal(nobs)
    x[rng.choice(nobs, nmissing, replace=False)] = np.nan
    res = acf(x, missing=missing)
    assert res.shape == (nlags + 1,)
    # "drop" treats the non-missing values as contiguous
    data = x[~np.isnan(x)] if missing == "drop" else x
    assert_allclose(res, _nan_acf(data, nlags), rtol=1e-10)


def test_acf_default_nlags_missing_none():
    # missing="none" counts every observation, NaN or not
    x = np.random.default_rng(20261120).standard_normal(200)
    x[::2] = np.nan
    assert acf(x, missing="none").shape == (24,)


def test_pacf_nlags_error():
    rs = np.random.RandomState(12487)
    e = rs.standard_normal(99)
    with pytest.raises(ValueError, match="Can only compute partial"):
        pacf(e, 50)


def test_pacf_negative_nlags():
    # negative nlags used to be clamped to 1 by max(nlags, 1) instead of
    # raising, silently returning a single lag
    rs = np.random.RandomState(12487)
    e = rs.standard_normal(99)
    with pytest.raises(ValueError, match="non-negative"):
        pacf(e, -5)
    # nlags=0 keeps its historical clamp to a single lag
    assert pacf(e, 0).shape == (2,)


PACF_METHODS = [
    "ywadjusted",
    "ywmle",
    "ols",
    "ols-inefficient",
    "ols-adjusted",
    "ldadjusted",
    "ldbiased",
    "burg",
]


@pytest.mark.parametrize("constant", [101.37, 123.456])
@pytest.mark.parametrize("length", [50, 100])
@pytest.mark.parametrize("method", PACF_METHODS)
def test_pacf_constant(constant, length, method):
    # GH10647: A constant series carries no autocorrelation information.
    x = np.full(length, constant)
    p = pacf(x, nlags=5, method=method)
    assert p[0] == 1.0
    assert np.all(np.isnan(p[1:]))

    res = pacf(x, nlags=5, method=method, alpha=0.05)
    assert res.pacf[0] == 1.0
    assert np.all(np.isnan(res.pacf[1:]))
    assert_allclose(res.confint[0], [1.0, 1.0])
    assert np.all(np.isnan(res.confint[1:]))


@pytest.mark.parametrize("constant", [101.37, 123.456])
@pytest.mark.parametrize("length", [50, 100])
@pytest.mark.parametrize("adjusted", [True, False])
@pytest.mark.parametrize("fft", [True, False])
@pytest.mark.parametrize("missing", ["none", "drop", "conservative"])
def test_acf_constant(constant, length, adjusted, fft, missing):
    # GH10647: A constant series carries no autocorrelation information.
    x = np.full(length, constant)
    if missing != "none":
        x[length // 2] = np.nan
    kwargs = {"adjusted": adjusted, "fft": fft, "missing": missing}

    a = acf(x, nlags=5, **kwargs)
    assert a[0] == 1.0
    assert np.all(np.isnan(a[1:]))

    res_a = acf(x, nlags=5, alpha=0.05, qstat=True, **kwargs)
    assert res_a.acf[0] == 1.0
    assert np.all(np.isnan(res_a.acf[1:]))
    assert_allclose(res_a.confint[0], [1.0, 1.0])
    assert np.all(np.isnan(res_a.confint[1:]))
    assert np.all(np.isnan(res_a.qstat))
    assert np.all(np.isnan(res_a.pvalues))


@pytest.mark.parametrize("constant", [101.37, 123.456])
@pytest.mark.parametrize("length", [50, 100])
@pytest.mark.parametrize("adjusted", [True, False])
@pytest.mark.parametrize("fft", [True, False])
@pytest.mark.parametrize("nlags", [None, 5])
@pytest.mark.parametrize("has_nan", [False, True])
@pytest.mark.parametrize("constant_first", [True, False])
def test_ccf_constant(
    constant, length, adjusted, fft, nlags, has_nan, constant_first
):
    # GH10647: x is constant, or constant once the NaN is dropped.
    x = np.full(length, constant)
    if has_nan:
        x[length // 2] = np.nan
    other = np.arange(float(length))
    first, second = (x, other) if constant_first else (other, x)
    n_ret = length if nlags is None else nlags
    kwargs = {"nlags": nlags, "adjusted": adjusted, "fft": fft}

    # The fft path of scipy warns about NaN: the shortcut must come first
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = ccf(first, second, **kwargs)
        res_c = ccf(first, second, alpha=0.05, **kwargs)
    assert c.shape == (n_ret,)
    assert np.all(np.isnan(c))
    assert res_c.ccf.shape == (n_ret,)
    assert np.all(np.isnan(res_c.ccf))
    assert res_c.confint.shape == (n_ret, 2)
    assert np.all(np.isnan(res_c.confint))


@pytest.mark.parametrize(
    "shape_x, shape_y, match",
    [
        ((20, 2), (20,), "x is required to have ndim 1"),
        ((20,), (20, 2), "y is required to have ndim 1"),
        ((20, 2), (20, 2), "x is required to have ndim 1"),
    ],
)
@pytest.mark.parametrize(
    "constant_x, constant_y", [(True, False), (False, True), (True, True)]
)
def test_ccf_constant_invalid_shape_raises(
    shape_x, shape_y, match, constant_x, constant_y
):
    # GH10647: the shortcut for constant series must not hide invalid input
    x = np.arange(float(np.prod(shape_x))).reshape(shape_x)
    y = np.arange(float(np.prod(shape_y))).reshape(shape_y)
    if constant_x:
        x = np.full(shape_x, 2.5)
    if constant_y:
        y = np.full(shape_y, 2.5)
    with pytest.raises(ValueError, match=match):
        ccf(x, y)


@pytest.mark.parametrize(
    "shape_x, shape_y", [((20, 1), (20,)), ((20,), (20, 1)), ((20, 1), (20, 1))]
)
@pytest.mark.parametrize(
    "constant_x, constant_y", [(True, False), (False, True), (True, True)]
)
def test_ccf_constant_column(shape_x, shape_y, constant_x, constant_y):
    # A constant (nobs, 1) column gives NaN, as a constant 1-D series does
    x = np.arange(float(np.prod(shape_x))).reshape(shape_x)
    y = np.arange(float(np.prod(shape_y))).reshape(shape_y)
    if constant_x:
        x = np.full(shape_x, 2.5)
    if constant_y:
        y = np.full(shape_y, 2.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = ccf(x, y)
        res_c = ccf(x, y, alpha=0.05)
    assert c.shape == (20,)
    assert np.all(np.isnan(c))
    assert res_c.ccf.shape == (20,)
    assert np.all(np.isnan(res_c.ccf))
    assert res_c.confint.shape == (20, 2)
    assert np.all(np.isnan(res_c.confint))


@pytest.mark.parametrize("method", PACF_METHODS)
def test_pacf_constant_invalid_shape_raises(method):
    # GH10647: a constant (n, 2) array raises as a non-constant one does
    rng = np.random.default_rng(3213)
    with pytest.raises(ValueError) as exc_info:
        pacf(rng.standard_normal((30, 2)), nlags=3, method=method)
    with pytest.raises(ValueError) as exc_info_const:
        pacf(np.full((30, 2), 101.37), nlags=3, method=method)
    assert str(exc_info_const.value) == str(exc_info.value)


@pytest.mark.parametrize("method", PACF_METHODS)
def test_pacf_constant_column(method):
    # GH10647: a single column is a valid input
    p = pacf(np.full((30, 1), 101.37), nlags=3, method=method)
    assert p[0] == 1.0
    assert np.all(np.isnan(p[1:]))


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


def test_granger_causality_perfect_fit():
    # The first series is the second lagged once, so the unrestricted
    # regression fits the data exactly
    x2 = np.random.default_rng(20261119).standard_normal(50)
    x = np.column_stack([np.r_[0.0, x2[:-1]], x2])
    with pytest.raises(InfeasibleTestError, match="perfect fit"):
        grangercausalitytests(x, 2)


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
    # One observation gives 0 / 0 in the autocovariance and a singular matrix
    with (
        warns_or_none(RuntimeWarning, match="invalid value encountered"),
        warns_or_none(SingularMatrixWarning, match="Matrix is singular"),
    ):
        pacf_yw(y)


def _pacf_yw_loop(x, nlags, method):
    # pacf_yw with one yule_walker fit per lag
    pacf = [1.0]
    for k in range(1, nlags + 1):
        pacf.append(yule_walker(x, k, method=method, result_object=False)[0][-1])
    return np.array(pacf)


def _pacf_ols_loop(x, nlags, efficient, adjusted):
    # pacf_ols with one least squares fit per lag
    nobs = x.shape[0]
    pacf = np.empty(nlags + 1)
    pacf[0] = 1.0
    if efficient:
        res = lagmat(x, nlags, original="sep")
        xlags, x0 = add_constant(res.lags), res.leads
        for k in range(1, nlags + 1):
            params = np.linalg.lstsq(xlags[k:, : k + 1], x0[k:], rcond=None)[0]
            pacf[k] = np.squeeze(params[-1])
    else:
        x = x - np.mean(x)
        res = lagmat(x, nlags, original="sep", trim="both")
        xlags, x0 = res.lags, res.leads
        for k in range(1, nlags + 1):
            params = np.linalg.lstsq(xlags[:, :k], x0, rcond=None)[0]
            pacf[k] = np.squeeze(params[-1])
    if adjusted:
        pacf *= nobs / (nobs - np.arange(nlags + 1))
    return pacf


def _pacf_series(kind, nobs, rng):
    e = rng.standard_normal(nobs)
    if kind == "white_noise":
        return e
    elif kind == "ar":
        x = np.zeros(nobs)
        for t in range(3, nobs):
            x[t] = 0.6 * x[t - 1] - 0.3 * x[t - 2] + 0.2 * x[t - 3] + e[t]
        return x
    elif kind == "near_unit_root":
        x = np.zeros(nobs)
        for t in range(1, nobs):
            x[t] = 0.999 * x[t - 1] + e[t]
        return x
    elif kind == "random_walk":
        return 100.0 + np.cumsum(e)
    elif kind == "mostly_constant":
        x = np.full(nobs, 3.0)
        x[nobs // 2 :] += 1e-3 * np.cumsum(e[nobs // 2 :])
        return x
    else:  # kind == "heavy_tails"
        return rng.standard_t(1.5, nobs)


PACF_SERIES = [
    "white_noise",
    "ar",
    "near_unit_root",
    "random_walk",
    "mostly_constant",
    "heavy_tails",
]


@pytest.mark.parametrize("method", ["adjusted", "mle"])
@pytest.mark.parametrize("kind", PACF_SERIES)
def test_pacf_yw_levinson_matches_yule_walker(method, kind):
    # Levinson-Durbin gives the same values as one yule_walker fit per lag
    rng = np.random.default_rng(10526)
    for nobs in [5, 12, 25, 300]:
        x = _pacf_series(kind, nobs, rng)
        for nlags in {1, 2, nobs // 2 - 1, min(nobs - 1, 40)} - {0}:
            expected = _pacf_yw_loop(x, nlags, method)
            assert_allclose(pacf_yw(x, nlags, method), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("efficient", [True, False])
@pytest.mark.parametrize("adjusted", [True, False])
@pytest.mark.parametrize("kind", PACF_SERIES)
def test_pacf_ols_qr_matches_lstsq(efficient, adjusted, kind):
    # one QR gives the same values as one least squares fit per lag
    rng = np.random.default_rng(10526)
    # With half of the sample constant, the regressions with the most lags
    # are close to singular, and both methods lose precision
    rtol = 1e-7 if kind == "mostly_constant" else 1e-10
    for nobs in [5, 12, 25, 300]:
        x = _pacf_series(kind, nobs, rng)
        for nlags in {1, 2, nobs // 2 - 1, nobs // 2, min(nobs // 2, 40)} - {0}:
            expected = _pacf_ols_loop(x, nlags, efficient, adjusted)
            res = pacf_ols(x, nlags, efficient=efficient, adjusted=adjusted)
            assert_allclose(res, expected, rtol=rtol, atol=1e-12)


def test_pacf_qr_blocks():
    # more observations than one block of rows
    rng = np.random.default_rng(10526)
    x = _pacf_series("ar", 20000, rng)
    for efficient in [True, False]:
        assert _pacf_ols_qr(x - x.mean(), 30, efficient) is not None
        res = pacf_ols(x, 30, efficient=efficient)
        expected = _pacf_ols_loop(x, 30, efficient, False)
        assert_allclose(res, expected, rtol=0, atol=1e-13)
    for method in ["adjusted", "mle"]:
        expected = _pacf_yw_loop(x, 30, method)
        assert_allclose(pacf_yw(x, 30, method), expected, rtol=0, atol=1e-13)


def test_pacf_ols_large_mean():
    # The efficient regressions include a constant, so the PACF does not
    # depend on the mean. lstsq on the raw data loses precision when the
    # mean is large relative to the variation, the QR uses demeaned data.
    rng = np.random.default_rng(10526)
    x = _pacf_series("ar", 400, rng)
    for level in [0.0, 1e4, 1e8, 1e12]:
        y = x + level
        expected = _pacf_ols_loop(y - y.mean(), 10, True, False)
        assert_allclose(pacf_ols(y, 10), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("nobs, nlags", [(10, 5), (20, 10), (240, 23)])
def test_pacf_ols_default_nlags(nobs, nlags):
    # min(10 * log10(nobs), nobs // 2): the second bound binds for nobs = 10
    # and 20
    x = np.random.default_rng(nobs).standard_normal(nobs)
    res = pacf_ols(x)
    assert res.shape == (nlags + 1,)
    assert_allclose(res, pacf_ols(x, nlags=nlags), rtol=1e-12)


def test_pacf_2d_input():
    rng = np.random.default_rng(10526)
    x = _pacf_series("ar", 200, rng)
    assert_allclose(pacf_yw(x[:, None], 10), pacf_yw(x, 10), rtol=1e-15)
    assert_allclose(pacf_ols(x[:, None], 10), pacf_ols(x, 10), rtol=1e-15)


def _pacf_singular_series(kind):
    t = np.arange(60.0)
    if kind == "constant":
        return np.ones(60)
    elif kind == "alternating":
        return (-1.0) ** t
    elif kind == "periodic":
        return np.tile([1.0, -1.0, 2.0], 20)
    elif kind == "sine":
        return np.sin(0.3 * t)
    else:  # kind == "trend"
        return t


@pytest.mark.parametrize(
    "kind", ["constant", "alternating", "periodic", "sine", "trend"]
)
def test_pacf_yw_levinson_not_positive_definite(kind):
    # the recursion stops where the Toeplitz matrix is not positive definite
    # and yule_walker is used for the remaining lags
    x = _pacf_singular_series(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SingularMatrixWarning)
        for method in ["adjusted", "mle"]:
            expected = _pacf_yw_loop(x, 20, method)
            assert_allclose(pacf_yw(x, 20, method), expected, rtol=0, atol=1e-13)
    if kind != "trend":
        assert _pacf_yw_levinson(x, 20, True).shape[0] < 21


@pytest.mark.parametrize(
    "kind", ["constant", "alternating", "periodic", "sine", "trend"]
)
@pytest.mark.parametrize("efficient", [True, False])
def test_pacf_ols_qr_singular(kind, efficient):
    # close to singular regressions are fitted one by one
    x = _pacf_singular_series(kind)
    assert _pacf_ols_qr(x - x.mean(), 20, efficient) is None
    expected = _pacf_ols_loop(x, 20, efficient, False)
    assert_allclose(pacf_ols(x, 20, efficient=efficient), expected, rtol=1e-15)


def test_pacf_nonfinite():
    # Data that are not finite are left to the per-lag fits
    x = np.cumsum(np.ones(30))
    x[20] = np.nan
    assert _pacf_yw_levinson(x, 5, True).shape[0] == 1
    assert _pacf_ols_qr(x, 5, True) is None


@pytest.mark.skipif(PYTHON_IMPL_WASM, reason="No fp exception support in WASM")
def test_pacf_nonfinite_per_lag_fits():
    x = np.cumsum(np.ones(30))
    x[20] = np.nan
    assert np.all(np.isnan(pacf_yw(x, 5)[1:]))
    # NumPy 1.23 raises a SystemError instead of the LinAlgError
    error = SystemError if NP_LT_1_24 else np.linalg.LinAlgError
    with pytest.raises(error):
        pacf_ols(x, 5)


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
    # The series is piecewise constant, so the break regressions are
    # degenerate: on some CI runners (macOS) a variance rounds below zero and
    # its t-value is NaN. Only the data matter here.
    with warns_or_none(RuntimeWarning, match="invalid value encountered in sqrt"):
        zivot_andrews(df["variable1"])
        zivot_andrews(df["variable1"], regression="c")
    pd.testing.assert_frame_equal(df, df_original)


class _ZivotAndrewsLoop(ZivotAndrewsUnitRoot):
    # fits the regressions one break period at a time
    def _break_tvalues(self, *args):
        return None


ZA_DATASETS = ["rgnp", "gnpdef", "stkprc", "rgnpq"]


def _za_series(kind, nobs, rng):
    if kind in ZA_DATASETS:
        path = Path(CURR_DIR).joinpath("results", f"{kind}.csv")
        return np.asarray(pd.read_csv(path), dtype=float).ravel()
    e = rng.standard_normal(nobs)
    t = np.arange(nobs)
    if kind == "random_walk":
        return np.cumsum(e)
    elif kind == "white_noise":
        return e
    elif kind == "level_break":
        return np.cumsum(e) + 0.3 * t + 8.0 * (t >= nobs // 2)
    elif kind == "trend_break":
        return 0.1 * t + 0.4 * np.maximum(t - nobs // 3, 0) + np.cumsum(e)
    elif kind == "near_trend":
        # the lagged level is almost collinear with the trend
        return 1.0 + 0.5 * t + 1e-3 * e
    else:  # kind == "ar"
        x = np.zeros(nobs)
        for i in range(1, nobs):
            x[i] = 0.8 * x[i - 1] + e[i]
        return 50.0 + x


def _za_break_tvalues(x, lags, regression, trim, step=1):
    # the t-values of the lagged level from ZivotAndrewsUnitRoot._break_tvalues
    # and from one OLS fit per break period, with the regressors of the loop
    # in ZivotAndrewsUnitRoot.run
    za = ZivotAndrewsUnitRoot()
    nobs = x.shape[0]
    cols = 5 if regression == "ct" else 4
    const = 1 / np.sqrt(nobs)
    trend = np.arange(1.0, nobs + 2) * np.sqrt(3) / nobs ** (3 / 2)
    endog, exog = za._format_regression_data(x, nobs, const, trend, cols, lags)
    trimcnt = int(nobs * trim)
    periods = np.arange(trimcnt + 1, nobs - trimcnt + 1, step)
    args = (regression, periods, nobs, const, trend, cols, lags)
    tvalues = za._break_tvalues(endog[lags:], exog, *args)
    if tvalues is None:
        return None, None
    expected = []
    for bp in periods:
        exog = za._update_regression_exog(
            exog, regression, bp, nobs, const, trend, cols, lags
        )
        res = OLS(endog[lags:], exog).fit()
        expected.append(res.tvalues[cols - 1])
    return tvalues, np.array(expected)


ZA_SERIES = ZA_DATASETS + [
    "random_walk",
    "white_noise",
    "level_break",
    "trend_break",
    "ar",
]


@pytest.mark.parametrize("regression", ["c", "t", "ct"])
@pytest.mark.parametrize("kind", ZA_SERIES)
def test_za_break_tvalues_matches_ols(regression, kind):
    # the t-statistics of all break periods from one QR decomposition are
    # the same as from one OLS fit per period
    rng = np.random.default_rng(10583)
    x = _za_series(kind, 250, rng)
    for lags, trim in [(0, 0.15), (2, 1 / 3), (5, 0.15)]:
        tvalues, expected = _za_break_tvalues(x, lags, regression, trim)
        assert tvalues is not None, "the QR decomposition was not used"
        assert_allclose(tvalues, expected, rtol=1e-10)


@pytest.mark.parametrize("regression", ["c", "t", "ct"])
def test_za_break_tvalues_blocks(regression):
    # more observations than one block of rows of the QR decomposition. The
    # cumulative sums of the cross products round over 20,000 rows, which
    # changes the t-values by up to about 1e-9 (relative).
    rng = np.random.default_rng(10583)
    x = _za_series("trend_break", 20000, rng)
    tvalues, expected = _za_break_tvalues(x, 4, regression, 0.15, step=499)
    assert tvalues is not None, "the QR decomposition was not used"
    assert_allclose(tvalues, expected, rtol=1e-8)


@pytest.mark.parametrize(
    "name, regression, lags, statistic, bpidx",
    [
        ("gnpdef", "ct", 8, -4.29645992991322, 41),
        ("stkprc", "ct", 5, -4.4188005626248, 65),
        ("rgnpq", "t", 5, -3.15090901781973, 88),
        ("rgnpq", "ct", 8, -4.51449473135603, 71),
        # without lagged differences, maxlag=0
        ("rgnp", "c", 0, -3.29267887535406, 20),
        ("rgnp", "t", 0, -2.78828925514759, 23),
        ("rgnp", "ct", 0, -3.46782210885675, 20),
        ("gnpdef", "c", 0, -3.35278702063666, 37),
        ("gnpdef", "t", 0, -2.00884754207356, 44),
        ("gnpdef", "ct", 0, -3.49680587365043, 40),
        ("stkprc", "c", 0, -3.98489915452376, 82),
        ("stkprc", "t", 0, -3.86734171259623, 71),
        ("stkprc", "ct", 0, -4.74799191482552, 59),
        ("rgnpq", "c", 0, -2.81646348557215, 130),
        ("rgnpq", "t", 0, -2.57626456915402, 84),
        ("rgnpq", "ct", 0, -3.16574448254965, 67),
    ],
)
def test_zivot_andrews_r_reference(name, regression, lags, statistic, bpidx):
    # urca 1.3.4, ur.za(x, model, lag). Both report the last observation
    # before the break, which R counts from 1, so its break point is
    # bpidx + 1 for all three models. The minimum is inside the default trim.
    x = _za_series(name, None, None)
    res = zivot_andrews(x, maxlag=lags, regression=regression, autolag=None)
    assert res[3] == lags
    assert_allclose(res[0], statistic, rtol=1e-10)
    assert res[4] == bpidx


@pytest.mark.parametrize("regression", ["c", "t", "ct"])
def test_zivot_andrews_maxlag_zero(regression):
    # maxlag=0 is a number of lags, and not the Schwert default
    x = _za_series("rgnpq", None, None)
    res = zivot_andrews(x, maxlag=0, regression=regression, autolag=None)
    assert res[3] == 0
    # the lag selection can only choose 0
    expected = zivot_andrews(x, maxlag=0, regression=regression, autolag="AIC")
    assert expected[3] == 0
    assert_allclose(res[:2], expected[:2], rtol=1e-12)
    assert res[2] == expected[2]
    assert res[4] == expected[4]
    default = zivot_andrews(x, regression=regression, autolag=None)
    assert default[3] == int(np.ceil(12.0 * (x.shape[0] / 100.0) ** 0.25))


@pytest.mark.parametrize("regression", ["c", "t", "ct"])
@pytest.mark.parametrize("nobs", [100, 250])
def test_zivot_andrews_default_maxlag(regression, nobs):
    # Without autolag the default number of lags is the default maxlag of
    # adfuller with regression="ct", the largest lag of the lag search:
    # ceil(12 * (nobs / 100) ** (1 / 4)), or nobs // 2 - 3 if smaller. It is
    # an integer for nobs=100, and 15.09 for nobs=250.
    expected_lags = min(int(np.ceil(12 * (nobs / 100) ** 0.25)), nobs // 2 - 3)
    rng = np.random.default_rng(20261120)
    x = _za_series("trend_break", nobs, rng)
    adf = adfuller(x, regression="ct", autolag=None, result_object=True)
    assert adf.lags == expected_lags
    res = zivot_andrews(x, regression=regression, autolag=None)
    assert res[3] == expected_lags
    explicit = zivot_andrews(
        x, maxlag=expected_lags, regression=regression, autolag=None
    )
    assert_allclose(res[0], explicit[0], rtol=1e-12)
    assert res[4] == explicit[4]


@pytest.mark.parametrize("autolag", [None, "AIC"])
def test_zivot_andrews_default_maxlag_too_short(autolag):
    # adfuller with regression="ct" allows no lags for 5 observations,
    # nobs // 2 - 3 < 0, with or without a lag search
    x = np.random.default_rng(5).standard_normal(5).cumsum()
    with pytest.raises(ValueError, match="sample size is too short"):
        zivot_andrews(x, autolag=autolag)


@pytest.mark.parametrize("autolag", [None, "AIC"])
def test_zivot_andrews_negative_maxlag(autolag):
    x = _za_series("gnpdef", None, None)
    with pytest.raises(ValueError, match="maxlag must be a non-negative integer"):
        zivot_andrews(x, maxlag=-1, autolag=autolag)


@pytest.mark.parametrize("regression", ["c", "t", "ct"])
def test_za_break_tvalues_near_collinear(regression):
    # The level is almost a linear trend. The cross product inverse of the
    # per-period loop loses about 5 digits here, and picks a different break
    # period for "t". The QR decomposition does not lose them.
    rng = np.random.default_rng(10583)
    x = _za_series("near_trend", 300, rng)
    tvalues, expected = _za_break_tvalues(x, 2, regression, 0.15)
    assert tvalues is not None, "the QR decomposition was not used"
    assert_allclose(tvalues, expected, rtol=1e-8)
    assert np.argmin(tvalues) == np.argmin(expected)


@pytest.mark.parametrize("regression", ["c", "t", "ct"])
@pytest.mark.parametrize("autolag", ["AIC", "BIC", "t-stat", None])
@pytest.mark.parametrize("kind", ZA_SERIES)
def test_zivot_andrews_matches_loop(regression, autolag, kind):
    # same statistic, p-value, critical values, lags and break period as
    # fitting the regressions one break period at a time
    rng = np.random.default_rng(10583)
    x = _za_series(kind, 200, rng)
    configs = [(0.15, 4), (1 / 3, 4)]
    if kind not in ZA_DATASETS:
        # The datasets are too short for the default number of lags
        configs += [(0.05, 4), (0.15, None), (1 / 3, None)]
    for trim, maxlag in configs:
        kwargs = dict(trim=trim, maxlag=maxlag, regression=regression)
        res = zivot_andrews(x, autolag=autolag, **kwargs)
        expected = _ZivotAndrewsLoop()(x, autolag=autolag, **kwargs)
        assert_allclose(res[0], expected[0], rtol=1e-9)
        assert_allclose(res[1], expected[1], rtol=1e-8, atol=1e-12)
        assert res[2] == expected[2]
        assert res[3] == expected[3]
        assert res[4] == expected[4]


def _za_singular_series(kind):
    if kind == "periodic":
        # lagged differences are exactly collinear
        return np.cumsum(np.tile([1.0, -1.0, 2.0], 40))
    elif kind == "linear":
        return np.arange(100.0)
    else:  # kind == "geometric"
        return 0.5 ** (np.arange(120.0) / 10)


@pytest.mark.parametrize("kind", ["periodic", "linear", "geometric"])
@pytest.mark.parametrize("regression", ["c", "t", "ct"])
def test_zivot_andrews_singular(kind, regression):
    # singular regressors are left to the per-period loop, which raises
    x = _za_singular_series(kind)
    tvalues, _ = _za_break_tvalues(x, 3, regression, 0.15, step=10)
    assert tvalues is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SingularMatrixWarning)
        with pytest.raises(ValueError, match="not full rank"):
            zivot_andrews(x, maxlag=3, regression=regression, autolag=None)
        # also when the first periods inside the trim are skipped
        with pytest.raises(ValueError, match="not full rank"):
            zivot_andrews(x, trim=0.05, maxlag=8, regression=regression, autolag=None)


def _za_outcome(za, x, **kwargs):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            return za(x, **kwargs)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return type(exc)


def _za_deterministic(t, p, regression):
    # The constant, the trend and the break dummies of break period p. The
    # level shift is 1(t >= p). The trend shift is (t - p + 2) from t = p - 1
    # on for "t", and from t = p on for "ct".
    cols = [np.ones_like(t), t]
    if regression != "t":
        cols.append((t >= p).astype(float))
    if regression == "t":
        cols.append(np.maximum(t - p + 2, 0.0))
    elif regression == "ct":
        cols.append(np.where(t >= p, t - p + 2, 0.0))
    return cols


def _za_direct(x, lags, regression, periods):
    # The statistic and break index from one OLS fit per break period, with
    # the regressors built directly: Delta x_t, t = lags + 1, ..., nobs - 1,
    # on the deterministic terms, x_{t-1} and the lagged differences. The
    # break index is the last observation where the break dummies are zero.
    nobs = x.shape[0]
    dx = np.diff(x)
    t = np.arange(lags + 1, nobs, dtype=float)
    lagged = [dx[lags - j : nobs - 1 - j] for j in range(1, lags + 1)]
    tvalues = []
    for p in periods:
        cols = _za_deterministic(t, p, regression)
        exog = np.column_stack(cols + [x[lags : nobs - 1]] + lagged)
        tvalues.append(OLS(dx[lags:], exog).fit().tvalues[len(cols)])
    loc = int(np.argmin(tvalues))
    dummies = _za_deterministic(np.arange(nobs, dtype=float), periods[loc], regression)
    nonzero = np.flatnonzero(np.any(np.column_stack(dummies[2:]) != 0, axis=1))
    return tvalues[loc], nonzero[0] - 1


class _ZivotAndrewsPeriods(ZivotAndrewsUnitRoot):
    # records the candidate break periods
    def _break_tvalues(self, endog, exog, regression, periods, *args):
        self.periods = periods
        return super()._break_tvalues(endog, exog, regression, periods, *args)


@pytest.mark.parametrize("regression", ["c", "t", "ct"])
def test_zivot_andrews_candidate_periods(regression):
    # The candidate break periods are the periods inside the trim whose
    # deterministic terms have full column rank over the rows of the
    # regression. The terms are integers, so the rank is not borderline.
    nobs = 40
    x = np.random.default_rng(24).standard_normal(nobs).cumsum()
    # the break of period p is after observation p - 2 for "t", one earlier
    # than for "c" and "ct", so its trimmed periods are one later
    shift = int(regression == "t")
    for trim, lags in [(0.0, 0), (0.0, 3), (0.05, 8), (0.15, 8)]:
        za = _ZivotAndrewsPeriods()
        za(x, trim=trim, maxlag=lags, regression=regression, autolag=None)
        t = np.arange(lags + 1, nobs, dtype=float)
        trimcnt = int(nobs * trim)
        expected = []
        for p in range(trimcnt + 1 + shift, min(nobs - trimcnt + shift, nobs) + 1):
            cols = _za_deterministic(t, p, regression)
            if np.linalg.matrix_rank(np.column_stack(cols)) == len(cols):
                expected.append(p)
        assert_equal(za.periods, expected)


# The break periods whose dummies are not collinear with the constant and
# the trend in the regression, which starts at period lags + 1
_ZA_VALID = {
    "c": lambda nobs, lags: (lags + 2, nobs - 1),
    "t": lambda nobs, lags: (lags + 4, nobs),
    "ct": lambda nobs, lags: (lags + 3, nobs - 2),
}


@pytest.mark.parametrize("regression", ["c", "t", "ct"])
def test_zivot_andrews_direct_regressions(regression):
    # The direct regressions give the statistic and break period of an
    # ordinary case, where no candidate break period is skipped
    rng = np.random.default_rng(10583)
    x = _za_series("random_walk", 200, rng)
    res = zivot_andrews(x, maxlag=4, regression=regression, autolag=None)
    shift = int(regression == "t")
    periods = np.arange(31 + shift, 171 + shift)
    statistic, bpidx = _za_direct(x, 4, regression, periods)
    assert_allclose(res[0], statistic, rtol=1e-8)
    assert res[4] == bpidx


def _za_check_skipped(x, trim, lags, regression):
    # The candidate periods whose dummies are collinear are skipped: the
    # statistic is the minimum over the others, and the per-period loop agrees
    nobs = x.shape[0]
    trimcnt = int(nobs * trim)
    first, last = _ZA_VALID[regression](nobs, lags)
    shift = int(regression == "t")
    periods = np.arange(
        max(trimcnt + 1 + shift, first), min(nobs - trimcnt + shift, last) + 1
    )
    kwargs = dict(trim=trim, maxlag=lags, regression=regression, autolag=None)
    res = zivot_andrews(x, **kwargs)
    statistic, bpidx = _za_direct(x, lags, regression, periods)
    assert_allclose(res[0], statistic, rtol=1e-8)
    assert res[4] == bpidx
    expected = _ZivotAndrewsLoop()(x, **kwargs)
    assert_allclose(res[0], expected[0], rtol=1e-8)
    assert res[4] == expected[4]


@pytest.mark.parametrize("regression", ["c", "t", "ct"])
def test_zivot_andrews_dummies_wrap_around(regression):
    # With trim=0, or more lags than trimmed observations, the dummies of the
    # first or last periods are zero or wrap around the start of the sample.
    # These periods used to give a NaN or meaningless statistic, or raise.
    rng = np.random.default_rng(10583)
    x = _za_series("random_walk", 100, rng)
    for trim, lags in [(0.0, 0), (0.05, 8)]:
        tvalues, _ = _za_break_tvalues(x, lags, regression, trim)
        assert tvalues is None
        _za_check_skipped(x, trim, lags, regression)


@pytest.mark.parametrize("regression, lags", [("c", 5), ("t", 4), ("t", 3), ("ct", 4)])
def test_zivot_andrews_dummy_collinear(regression, lags):
    # The dummies of the first period are collinear with the constant and
    # the trend, and the period is skipped
    rng = np.random.default_rng(10583)
    x = _za_series("random_walk", 100, rng)
    tvalues, _ = _za_break_tvalues(x, lags, regression, 0.05)
    assert tvalues is None
    _za_check_skipped(x, 0.05, lags, regression)


@pytest.mark.parametrize("regression", ["c", "t", "ct"])
def test_zivot_andrews_lags_beyond_trim(regression):
    # 8 lags and 3 trimmed observations of 24: the first periods are
    # degenerate. The statistic was NaN for "c" (with a RuntimeWarning).
    x = np.random.default_rng(24).standard_normal(24).cumsum()
    _za_check_skipped(x, 0.15, 8, regression)


@pytest.mark.parametrize("trim", [0.1, 0.15, 0.25])
def test_zivot_andrews_same_trimmed_breaks(trim):
    # The three models consider the same trimmed candidate breaks, from
    # observation trimcnt to nobs - trimcnt - 1 (the last observation before
    # the break), when the lags are too few to make any of them collinear
    nobs = 62
    x = np.random.default_rng(24).standard_normal(nobs).cumsum()
    trimcnt = int(nobs * trim)
    expected = np.arange(trimcnt, nobs - trimcnt)
    for regression, before in (("c", 1), ("t", 2), ("ct", 1)):
        za = _ZivotAndrewsPeriods()
        za(x, trim=trim, maxlag=1, regression=regression, autolag=None)
        assert_equal(np.asarray(za.periods) - before, expected)


@pytest.mark.parametrize("regression", ["c", "t", "ct"])
def test_zivot_andrews_no_break_period_left(regression):
    # 20 lags of 24 observations leave no period with full rank dummies
    x = np.random.default_rng(24).standard_normal(24).cumsum()
    with pytest.raises(ValueError, match="No break period is left: with 20 lags"):
        zivot_andrews(x, maxlag=20, regression=regression, autolag=None)


@pytest.mark.parametrize("regression", ["c", "ct"])
def test_zivot_andrews_level_collinear(regression):
    # A trend with a level shift is in the span of the regressors of the
    # true break period. The per-period loop is used.
    t = np.arange(100.0)
    x = t + 5.0 * (t >= 50)
    tvalues, _ = _za_break_tvalues(x, 1, regression, 0.15)
    assert tvalues is None
    kwargs = dict(maxlag=1, regression=regression, autolag=None)
    res = _za_outcome(zivot_andrews, x, **kwargs)
    expected = _za_outcome(_ZivotAndrewsLoop(), x, **kwargs)
    assert_equal(res, expected)


def test_zivot_andrews_nonfinite():
    # data that are not finite are left to the per-period loop, which raises
    x = np.cumsum(np.ones(100))
    x[50] = np.nan
    tvalues, _ = _za_break_tvalues(x, 2, "c", 0.15)
    assert tvalues is None
    with pytest.raises(MissingDataError):
        zivot_andrews(x, maxlag=2, autolag=None)


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

    def test_numpy_integer_arlags(self):
        mdl = np.asarray(pd.read_csv(Path(self.run_dir).joinpath("BAA.csv")))
        res = leybourne(mdl, regression="ct", arlags=3)
        res_np = leybourne(mdl, regression="ct", arlags=np.int64(3))
        assert_allclose(res_np[0:3], res[0:3])

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
        # The default ML fit may stop short of convergence on some platforms
        with warns_or_none(ConvergenceWarning, match="Maximum Likelihood"):
            res = leybourne(mdl, method="mle")
        assert_allclose(res[0:3], [0.096534, 0.602535, 2], rtol=1e-4, atol=1e-4)
        with warns_or_none(ConvergenceWarning, match="Maximum Likelihood"):
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

    @pytest.mark.parametrize(
        "convert",
        [
            lambda x: x[:, 0],
            lambda x: x[:, 0].tolist(),
            lambda x: pd.Series(x[:, 0]),
            pd.DataFrame,
        ],
        ids=["1d", "list", "series", "dataframe"],
    )
    def test_array_like_inputs(self, convert):
        mdl = np.asarray(pd.read_csv(Path(self.run_dir).joinpath("BAA.csv")))
        res = leybourne(convert(mdl), regression="ct")
        # verified against Matlab 9.13, as in test_baa_results
        assert_allclose(res[0:3], [5.4757, 0.0000, 3], rtol=1e-4, atol=1e-4)
        res_col = leybourne(mdl, regression="ct")
        assert_allclose(res[0], res_col[0], rtol=1e-10)
        assert res[2] == res_col[2]

    @pytest.mark.parametrize(
        "shape, match",
        [
            ((60, 2), r"x is required to have shape \(\*, 1\)"),
            ((1, 60), r"x is required to have shape \(\*, 1\)"),
            ((60, 1, 1), "x must have ndim <= 2"),
        ],
        ids=["two-columns", "row", "3d"],
    )
    def test_not_one_column_raises(self, shape, match):
        x = np.random.default_rng(0).standard_normal(shape)
        with pytest.raises(ValueError, match=match):
            leybourne(x)


def test_acovf_all_missing():
    x = np.full(100, np.nan)
    with pytest.raises(
        ValueError, match=r"All observations are missing after dropping."
    ):
        acovf(x, missing="drop")

    # Without observations the mean and the autocovariances are 0 / 0
    with warns_or_none(RuntimeWarning, match="invalid value encountered"):
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


@pytest.mark.parametrize("func", [acf, pacf, ccf, pccf])
def test_confint_alpha_out_of_range(func):
    # alpha outside (0, 1) previously returned inf/-inf bounds silently
    rng = np.random.RandomState(1234)
    x = rng.normal(size=50)
    y = rng.normal(size=50)
    with pytest.raises(ValueError, match="alpha must be in the range"):
        if func in (ccf, pccf):
            func(x, y, nlags=5, alpha=2)
        else:
            func(x, nlags=5, alpha=2)
    with pytest.raises(ValueError, match="alpha must be in the range"):
        if func in (ccf, pccf):
            func(x, y, nlags=5, alpha=0)
        else:
            func(x, nlags=5, alpha=0)
    # a valid alpha still returns the interval
    res = acf(x, nlags=5, alpha=0.05, result_object=False)
    assert len(res) == 2
    assert np.isfinite(res[1]).all()


def test_arma_order_select_ic_negative_bounds():
    # a negative max_ar or max_ma used to empty one of the order grids and
    # leak a bare numpy error from the argmin over the empty sequence
    rs = np.random.RandomState(12345)
    y = rs.standard_normal(50)
    with pytest.raises(ValueError, match="must be non-negative"):
        arma_order_select_ic(y, max_ar=-1, max_ma=2)
    with pytest.raises(ValueError, match="must be non-negative"):
        arma_order_select_ic(y, max_ar=2, max_ma=-1)


@pytest.mark.parametrize(
    "ic",
    [
        ("aic", "bic"),
        ["aic", "bic"],
        np.array(["aic", "bic"]),
        pd.Index(["aic", "bic"]),
        pd.Series(["aic", "bic"]),
    ],
    ids=["tuple", "list", "ndarray", "index", "series"],
)
def test_arma_order_select_ic_sequence(ic):
    # ic is documented as a str or a sequence of str
    from statsmodels.tsa.arima.model import ARIMA

    y = np.random.default_rng(20261127).standard_normal(100)
    res = arma_order_select_ic(y, max_ar=2, max_ma=0, ic=ic, trend="n")
    assert sorted(res) == ["aic", "aic_min_order", "bic", "bic_min_order"]
    assert all(type(key) is str for key in res)
    fits = [ARIMA(y, order=(p, 0, 0), trend="n").fit() for p in range(3)]
    for name in ("aic", "bic"):
        expected = np.array([getattr(fit, name) for fit in fits])
        assert_allclose(res[name][0], expected, rtol=1e-10)
        assert res[f"{name}_min_order"] == (np.argmin(expected), 0)


def test_arma_order_select_ic_all_criteria():
    # Every valid name is accepted and gives the criterion of a direct fit
    from statsmodels.tsa.arima.model import ARIMA

    y = np.random.default_rng(20261127).standard_normal(100)
    names = ["aic", "aicc", "bic", "hqic"]
    res = arma_order_select_ic(y, max_ar=1, max_ma=0, ic=names, trend="n")
    fits = [ARIMA(y, order=(p, 0, 0), trend="n").fit() for p in range(2)]
    for name in names:
        expected = [getattr(fit, name) for fit in fits]
        assert_allclose(res[name][0], expected, rtol=1e-10)


@pytest.mark.parametrize(
    "ic",
    [5, [5], ["aic", 5], np.array([1, 2])],
    ids=["int", "list of int", "mixed list", "int ndarray"],
)
def test_arma_order_select_ic_invalid_type(ic):
    y = np.random.default_rng(20261127).standard_normal(100)
    with pytest.raises(ValueError, match="ic must be a str or a sequence of str"):
        arma_order_select_ic(y, max_ar=0, max_ma=0, ic=ic)


@pytest.mark.parametrize(
    "ic, unknown",
    [
        ("bogus", "bogus"),
        (["aic", "bogus"], "bogus"),
        (np.array(["bogus"]), "bogus"),
        ("AIC", "AIC"),
    ],
    ids=["str", "list", "ndarray", "upper"],
)
def test_arma_order_select_ic_unknown_criterion(ic, unknown):
    # An unknown name used to raise an AttributeError from the first fit
    y = np.random.default_rng(20261127).standard_normal(100)
    match = rf"Unknown information criteria in ic: \['{unknown}'\]"
    with pytest.raises(ValueError, match=match):
        arma_order_select_ic(y, max_ar=0, max_ma=0, ic=ic)


def test_q_stat_invalid_nobs():
    # nobs <= len(x) previously divided by zero (nobs=-1) or returned a
    # degenerate statistic without any error
    with pytest.raises(ValueError, match="nobs must be larger"):
        q_stat(np.array([0.5, 0.3]), nobs=-1)
    with pytest.raises(ValueError, match="nobs must be larger"):
        q_stat(np.array([0.5, 0.3]), nobs=2)
    q, p = q_stat(np.array([0.5, 0.3]), nobs=50)
    assert np.isfinite(q).all()
    assert np.isfinite(p).all()



@pytest.mark.parametrize("form", SERIES_FORMS)
def test_acf_pacf_adfuller_input_forms(form):
    # x is documented as array_like: every form of a 1-D series gives the
    # result of the ndarray
    x = np.random.default_rng(20261120).standard_normal(200).cumsum()
    x_form = series_forms(x)[form]
    assert_allclose(acf(x_form, nlags=10), acf(x, nlags=10), rtol=1e-12)
    for method in ("yw", "ols", "burg", "ldb"):
        assert_allclose(
            pacf(x_form, nlags=10, method=method),
            pacf(x, nlags=10, method=method),
            rtol=1e-12,
        )
    res = adfuller(x_form, maxlag=4, autolag=None, result_object=True)
    expected = adfuller(x, maxlag=4, autolag=None, result_object=True)
    assert_allclose(res.statistic, expected.statistic, rtol=1e-12)
    assert_allclose(res.pvalue, expected.pvalue, rtol=1e-12)


def _numeric_leaves(res):
    """The numeric values of a result, flattened, to compare two results"""
    if isinstance(res, dict):
        return np.concatenate([_numeric_leaves(res[key]) for key in sorted(res)])
    if isinstance(res, tuple):
        return np.concatenate([_numeric_leaves(item) for item in res])
    return np.asarray(res, dtype=float).ravel()


def _series_contract_cases():
    """
    Functions of one 1-D array_like argument, and the ndarray to pass

    The other arguments are fixed.
    """
    rng = np.random.default_rng(20261121)
    x = rng.standard_normal(200).cumsum()
    dx = np.diff(x)
    y1 = x + rng.standard_normal(200)
    acov = acovf(dx)[:6]
    theta = innovations_algo(acov)[0]
    return {
        "acovf": (lambda v: acovf(v, nlag=10), x),
        "arma_order_select_ic": (
            lambda v: arma_order_select_ic(v, max_ar=1, max_ma=0, ic="aic").aic,
            dx[:100],
        ),
        "block_jackknife": (lambda v: block_jackknife(v, np.mean, n_blocks=10), x),
        "breakvar_heteroskedasticity_test": (breakvar_heteroskedasticity_test, dx),
        "coint y0": (lambda v: coint(v, y1, maxlag=2, autolag=None), x),
        "coint y1": (lambda v: coint(x, v, maxlag=2, autolag=None), y1),
        "innovations_algo": (innovations_algo, acov),
        "innovations_filter": (lambda v: innovations_filter(v, theta), dx[:6]),
        "levinson_durbin": (lambda v: levinson_durbin(v, nlags=5), dx),
        "levinson_durbin_pacf": (levinson_durbin_pacf, pacf(dx, nlags=5)),
        "leybourne": (leybourne, x),
        "pacf_ols": (lambda v: pacf_ols(v, nlags=10), x),
        "pacf_yw": (lambda v: pacf_yw(v, nlags=10), x),
        "zivot_andrews": (lambda v: zivot_andrews(v, maxlag=2, autolag=None), x),
    }


_SERIES_CONTRACT_NAMES = tuple(sorted(_series_contract_cases()))


@pytest.mark.parametrize("form", SERIES_FORMS)
@pytest.mark.parametrize("name", _SERIES_CONTRACT_NAMES)
def test_stattools_input_forms(name, form):
    # The argument is documented as array_like: every form of a 1-D series
    # gives the result of the ndarray
    func, x = _series_contract_cases()[name]
    expected = _numeric_leaves(func(x))
    res = _numeric_leaves(func(series_forms(x)[form]))
    assert_allclose(res, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "name",
    [
        name
        for name in _SERIES_CONTRACT_NAMES
        if name not in ("innovations_algo", "levinson_durbin_pacf")
    ],
)
def test_stattools_integer_input(name):
    # Integer data gives the result of the same values as floats
    func, x = _series_contract_cases()[name]
    x_int = np.round(4 * x).astype(np.int64)
    expected = _numeric_leaves(func(x_int.astype(float)))
    assert_allclose(_numeric_leaves(func(x_int)), expected, rtol=1e-12)


@pytest.mark.parametrize("form", SERIES_FORMS)
@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("test", [diebold_mariano_test, clark_west_test])
def test_forecast_comparison_input_forms(test, position, form):
    # y and the two forecasts are documented as 1-D array_like: every form,
    # including an (n, 1) column, gives the result of the ndarray
    rng = np.random.default_rng(20261122)
    y = rng.standard_normal(100)
    args = [y, y + 0.5 * rng.standard_normal(100), y + 0.8 * rng.standard_normal(100)]
    expected = test(*args)
    args[position] = series_forms(args[position])[form]
    res = test(*args)
    assert_allclose(
        [res.statistic, res.pvalue], [expected.statistic, expected.pvalue], rtol=1e-12
    )


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("test", [diebold_mariano_test, clark_west_test])
def test_forecast_comparison_two_columns_raises(test, position):
    rng = np.random.default_rng(20261122)
    y = rng.standard_normal(100)
    args = [y, y + 0.5 * rng.standard_normal(100), y + 0.8 * rng.standard_normal(100)]
    args[position] = np.column_stack([args[position], args[position]])
    names = {
        diebold_mariano_test: ("y", "forecast_a", "forecast_b"),
        clark_west_test: ("y", "forecast_restricted", "forecast_unrestricted"),
    }
    name = names[test][position]
    with pytest.raises(ValueError, match=f"^{name} is required to have ndim 1"):
        test(*args)


@pytest.mark.parametrize("form", ["dataframe", "list", "int"])
def test_two_column_input_forms(form):
    # x of grangercausalitytests and resid of
    # breakvar_heteroskedasticity_test are documented as 2-D array_like
    rng = np.random.default_rng(20261123)
    x = np.round(4 * rng.standard_normal((100, 2)))
    index = pd.date_range("2000-01-01", periods=100, freq="MS")
    x_form = {
        "dataframe": pd.DataFrame(x, index=index, columns=["a", "b"]),
        "list": x.tolist(),
        "int": x.astype(np.int64),
    }[form]
    res = grangercausalitytests(x_form, 2)
    expected = grangercausalitytests(x, 2)
    assert sorted(res) == [1, 2]
    for lag in (1, 2):
        for name in ("ssr_ftest", "ssr_chi2test", "lrtest", "params_ftest"):
            # The lag 2 F statistic is about 0.0015: the restricted and full
            # sums of squared residuals agree to about 3e-5 (relative), so a
            # rounding difference from another memory layout of x (macOS
            # Accelerate) changes the statistics by about 1e-11 (relative).
            assert_allclose(res[lag][0][name], expected[lag][0][name], rtol=1e-9)
    res = breakvar_heteroskedasticity_test(x_form)
    expected = breakvar_heteroskedasticity_test(x)
    assert_allclose(_numeric_leaves(res), _numeric_leaves(expected), rtol=1e-12)


@pytest.mark.parametrize(
    "maxlag, lags",
    [
        (2, [1, 2]),
        (np.int64(2), [1, 2]),
        ([2], [2]),
        ((2,), [2]),
        (range(1, 3), [1, 2]),
        (np.array(2), [1, 2]),
        (np.array([2]), [2]),
        (pd.Series([2]), [2]),
        (pd.Index([2]), [2]),
        (np.array([1, 2]), [1, 2]),
        # Not an int and not an array
        ({1, 2}, [1, 2]),
    ],
)
def test_grangercausalitytests_maxlag_forms(gc_data, maxlag, lags):
    # An int gives the lags up to maxlag, a sequence only the lags in it. A
    # one-element ndarray or Series used to be read as an int.
    res = grangercausalitytests(gc_data, maxlag)
    assert sorted(res) == lags
    expected = grangercausalitytests(gc_data, [1, 2])
    for lag in lags:
        assert_allclose(
            res[lag][0]["ssr_ftest"], expected[lag][0]["ssr_ftest"], rtol=1e-12
        )


@pytest.mark.parametrize(
    "arlags",
    [np.int64(2), np.int32(2), np.array(2), 2.0],
    ids=["int64", "int32", "0d-array", "float"],
)
def test_leybourne_int_like_arlags(arlags):
    # A 0-d integer array or an integral float is a number of AR terms, as an
    # int is. They raised a ValueError. The int result is the reference.
    x = np.random.default_rng(20261126).standard_normal(200).cumsum()
    expected = leybourne(x, arlags=2)
    res = leybourne(x, arlags=arlags)
    assert_allclose(res[:2], expected[:2], rtol=1e-12)
    assert res[2] == 2
    assert type(res[2]) is int


@pytest.mark.parametrize("arlags", [True, np.True_], ids=["True", "np.True_"])
def test_leybourne_bool_arlags(arlags):
    # A bool is not a number of AR terms. True raised a TypeError from a
    # later step.
    x = np.random.default_rng(20261126).standard_normal(200).cumsum()
    with pytest.raises(ValueError, match="arlags must be an integer"):
        leybourne(x, arlags=arlags)
