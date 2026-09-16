"""
Test functions for sm.rlm
"""
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
from scipy import stats

import statsmodels.api as sm
from statsmodels.iolib.summary import Summary
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale, mad

DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1


def load_stackloss():
    from statsmodels.datasets.stackloss import load
    data = load()
    data.endog = np.asarray(data.endog)
    data.exog = np.asarray(data.exog)
    return data


class CheckRlmResultsMixin:
    """
    res2 contains  results from Rmodelwrap or were obtained from a statistical
    packages such as R, Stata, or SAS and written to results.results_rlm

    Covariance matrices were obtained from SAS and are imported from
    results.results_rlm
    """
    def test_params(self):
        assert_almost_equal(self.res1.params, self.res2.params, DECIMAL_4)

    decimal_standarderrors = DECIMAL_4

    def test_standarderrors(self):
        assert_almost_equal(self.res1.bse, self.res2.bse,
                            self.decimal_standarderrors)

    # TODO: get other results from SAS, though if it works for one...
    def test_confidenceintervals(self):
        if not hasattr(self.res2, "conf_int"):
            pytest.skip("Results from R")

        assert_almost_equal(self.res1.conf_int(), self.res2.conf_int(),
                            DECIMAL_4)

    decimal_scale = DECIMAL_4

    def test_scale(self):
        assert_almost_equal(self.res1.scale, self.res2.scale,
                            self.decimal_scale)

    def test_weights(self):
        assert_almost_equal(self.res1.weights, self.res2.weights, DECIMAL_4)

    def test_residuals(self):
        assert_almost_equal(self.res1.resid, self.res2.resid, DECIMAL_4)

    def test_degrees(self):
        assert_almost_equal(self.res1.model.df_model, self.res2.df_model,
                            DECIMAL_4)
        assert_almost_equal(self.res1.model.df_resid, self.res2.df_resid,
                            DECIMAL_4)

    def test_bcov_unscaled(self):
        if not hasattr(self.res2, "bcov_unscaled"):
            pytest.skip("No unscaled cov matrix from SAS")

        assert_almost_equal(self.res1.bcov_unscaled,
                            self.res2.bcov_unscaled, DECIMAL_4)

    decimal_bcov_scaled = DECIMAL_4

    def test_bcov_scaled(self):
        assert_almost_equal(self.res1.bcov_scaled, self.res2.h1,
                            self.decimal_bcov_scaled)
        assert_almost_equal(self.res1.h2, self.res2.h2,
                            self.decimal_bcov_scaled)
        assert_almost_equal(self.res1.h3, self.res2.h3,
                            self.decimal_bcov_scaled)

    def test_tvalues(self):
        if not hasattr(self.res2, "tvalues"):
            pytest.skip("No tvalues in benchmark")

        assert_allclose(self.res1.tvalues, self.res2.tvalues, rtol=0.003)

    def test_tpvalues(self):
        # test comparing tvalues and pvalues with normal implementation
        # make sure they use normal distribution (inherited in results class)
        params = self.res1.params
        tvalues = params / self.res1.bse
        pvalues = stats.norm.sf(np.abs(tvalues)) * 2
        half_width = stats.norm.isf(0.025) * self.res1.bse
        conf_int = np.column_stack((params - half_width, params + half_width))

        assert_almost_equal(self.res1.tvalues, tvalues)
        assert_almost_equal(self.res1.pvalues, pvalues)
        assert_almost_equal(self.res1.conf_int(), conf_int)


class TestRlm(CheckRlmResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.data = load_stackloss()  # class attributes for subclasses
        cls.data.exog = sm.add_constant(cls.data.exog, prepend=False)
        # Test precisions
        cls.decimal_standarderrors = DECIMAL_1
        cls.decimal_scale = DECIMAL_3

        model = RLM(cls.data.endog, cls.data.exog, M=norms.HuberT())
        cls.model = model
        results = model.fit()
        h2 = model.fit(cov="H2").bcov_scaled
        h3 = model.fit(cov="H3").bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import Huber
        self.res2 = Huber()

    @pytest.mark.smoke
    def test_summary(self):
        self.res1.summary()

    @pytest.mark.smoke
    def test_summary2(self):
        self.res1.summary2()

    @pytest.mark.smoke
    def test_chisq(self):
        assert isinstance(self.res1.chisq, np.ndarray)

    @pytest.mark.smoke
    def test_predict(self):
        assert isinstance(self.model.predict(self.res1.params), np.ndarray)


class TestHampel(TestRlm):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        # Test precisions
        cls.decimal_standarderrors = DECIMAL_2
        cls.decimal_scale = DECIMAL_3
        cls.decimal_bcov_scaled = DECIMAL_3

        model = RLM(cls.data.endog, cls.data.exog, M=norms.Hampel())
        results = model.fit()
        h2 = model.fit(cov="H2").bcov_scaled
        h3 = model.fit(cov="H3").bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import Hampel
        self.res2 = Hampel()


class TestRlmBisquare(TestRlm):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        # Test precisions
        cls.decimal_standarderrors = DECIMAL_1

        model = RLM(cls.data.endog, cls.data.exog, M=norms.TukeyBiweight())
        results = model.fit()
        h2 = model.fit(cov="H2").bcov_scaled
        h3 = model.fit(cov="H3").bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import BiSquare
        self.res2 = BiSquare()


class TestRlmAndrews(TestRlm):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        model = RLM(cls.data.endog, cls.data.exog, M=norms.AndrewWave())
        results = model.fit()
        h2 = model.fit(cov="H2").bcov_scaled
        h3 = model.fit(cov="H3").bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import Andrews
        self.res2 = Andrews()


# --------------------------------------------------------------------
# tests with Huber scaling

class TestRlmHuber(CheckRlmResultsMixin):
    @classmethod
    def setup_class(cls):
        cls.data = load_stackloss()
        cls.data.exog = sm.add_constant(cls.data.exog, prepend=False)

        model = RLM(cls.data.endog, cls.data.exog, M=norms.HuberT())
        results = model.fit(scale_est=HuberScale())
        h2 = model.fit(cov="H2", scale_est=HuberScale()).bcov_scaled
        h3 = model.fit(cov="H3", scale_est=HuberScale()).bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import HuberHuber
        self.res2 = HuberHuber()


class TestHampelHuber(TestRlm):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        model = RLM(cls.data.endog, cls.data.exog, M=norms.Hampel())
        results = model.fit(scale_est=HuberScale())
        h2 = model.fit(cov="H2", scale_est=HuberScale()).bcov_scaled
        h3 = model.fit(cov="H3", scale_est=HuberScale()).bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import HampelHuber
        self.res2 = HampelHuber()


class TestRlmBisquareHuber(TestRlm):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        model = RLM(cls.data.endog, cls.data.exog, M=norms.TukeyBiweight())
        results = model.fit(scale_est=HuberScale())
        h2 = model.fit(cov="H2", scale_est=HuberScale()).bcov_scaled
        h3 = model.fit(cov="H3", scale_est=HuberScale()).bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import BisquareHuber
        self.res2 = BisquareHuber()


class TestRlmAndrewsHuber(TestRlm):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        model = RLM(cls.data.endog, cls.data.exog, M=norms.AndrewWave())
        results = model.fit(scale_est=HuberScale())
        h2 = model.fit(cov="H2", scale_est=HuberScale()).bcov_scaled
        h3 = model.fit(cov="H3", scale_est=HuberScale()).bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import AndrewsHuber
        self.res2 = AndrewsHuber()


class TestRlmSresid(CheckRlmResultsMixin):
    # Check GH#187
    @classmethod
    def setup_class(cls):
        cls.data = load_stackloss()  # class attributes for subclasses
        cls.data.exog = sm.add_constant(cls.data.exog, prepend=False)
        # Test precisions
        cls.decimal_standarderrors = DECIMAL_1
        cls.decimal_scale = DECIMAL_3

        model = RLM(cls.data.endog, cls.data.exog, M=norms.HuberT())
        results = model.fit(conv="sresid")
        h2 = model.fit(cov="H2").bcov_scaled
        h3 = model.fit(cov="H3").bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import Huber
        self.res2 = Huber()


@pytest.mark.smoke
def test_missing():
    # see GH#2083
    import statsmodels.formula.api as smf

    d = pd.DataFrame({"Foo": [1, 2, 10, 149], "Bar": [1, 2, 3, np.nan]})
    smf.rlm("Foo ~ Bar", data=d)


def test_rlm_start_values():
    data = sm.datasets.stackloss.load_pandas()
    exog = sm.add_constant(data.exog, prepend=False)
    model = RLM(data.endog, exog, M=norms.HuberT())
    results = model.fit()
    start_params = [0.7156402, 1.29528612, -0.15212252, -39.91967442]
    result_sv = model.fit(start_params=start_params)
    assert_allclose(results.params, result_sv.params)


def test_rlm_start_values_errors():
    data = sm.datasets.stackloss.load_pandas()
    exog = sm.add_constant(data.exog, prepend=False)
    model = RLM(data.endog, exog, M=norms.HuberT())
    start_params = [0.7156402, 1.29528612, -0.15212252]
    with pytest.raises(ValueError):
        model.fit(start_params=start_params)

    start_params = np.array([start_params, start_params]).T
    with pytest.raises(ValueError):
        model.fit(start_params=start_params)


def test_rlm_scale_est_callback_receives_model():
    data = sm.datasets.stackloss.load_pandas()
    exog = sm.add_constant(data.exog, prepend=False)
    model = RLM(data.endog, exog, M=norms.HuberT())

    class ScaleEstimator:
        def __init__(self):
            self.calls = []

        def __call__(self, rlm_model, resid):
            self.calls.append((rlm_model, resid.copy()))
            return mad(resid, center=0)

    scale_est = ScaleEstimator()
    result = model.fit(scale_est=scale_est)

    assert scale_est.calls
    assert all(call[0] is model for call in scale_est.calls)
    assert_allclose(result.scale, mad(result.resid, center=0))


def test_rlm_scale_est_one_and_two_inputs():
    data = sm.datasets.stackloss.load_pandas()
    exog = sm.add_constant(data.exog, prepend=False)
    model = RLM(data.endog, exog, M=norms.HuberT())

    def one_input(resid):
        median = np.median(resid)
        c = np.sqrt(np.pi / 2)
        return c * np.mean(np.abs(resid - median))

    def two_inputs(model, resid):
        median = np.median(resid)
        c = np.sqrt(np.pi / 2)
        return c * np.mean(np.abs(resid - median)) * np.sqrt(model.nobs / model.df_resid)

    result_1 = model.fit(scale_est=one_input)
    result_2 = model.fit(scale_est=two_inputs)
    assert_allclose(result_1.scale, result_2.scale)


def test_rlm_scale_est_resid_callable_df_correction():
    data = sm.datasets.stackloss.load_pandas()
    exog = sm.add_constant(data.exog, prepend=False)
    model = RLM(data.endog, exog, M=norms.HuberT())

    result = model.fit(scale_est=lambda resid: mad(resid, center=0))

    expected = mad(result.resid, center=0) * np.sqrt(model.nobs / model.df_resid)
    assert_allclose(result.scale, expected)


@pytest.fixture(scope="module",
                params=[norms.AndrewWave, norms.LeastSquares, norms.HuberT,
                        norms.TrimmedMean, norms.TukeyBiweight, norms.Hampel,
                        norms.RamsayE])
def norm(request):
    return request.param()


@pytest.fixture(scope="module")
def perfect_fit_data(request):
    from statsmodels.tools.tools import Bunch
    rs = np.random.RandomState(1249328932)
    exog = rs.standard_normal((1000, 1))
    endog = exog + exog ** 2
    exog = sm.add_constant(np.c_[exog, exog ** 2])
    return Bunch(endog=endog, exog=exog, const=(3.2 * np.ones_like(endog)))


def test_perfect_fit(perfect_fit_data, norm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = RLM(perfect_fit_data.endog, perfect_fit_data.exog, M=norm).fit()
    assert_allclose(res.params, np.array([0, 1, 1]), atol=1e-8)


def test_perfect_const(perfect_fit_data, norm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = RLM(perfect_fit_data.const, perfect_fit_data.exog, M=norm).fit()
    assert_allclose(res.params, np.array([3.2, 0, 0]), atol=1e-8)


@pytest.mark.parametrize("conv", ["weights", "coefs", "sresid"])
def test_alt_criterion(conv):
    data = load_stackloss()
    data.exog = sm.add_constant(data.exog, prepend=False)
    base = RLM(data.endog, data.exog, M=norms.HuberT()).fit()
    alt = RLM(data.endog, data.exog, M=norms.HuberT()).fit(conv=conv)
    assert_allclose(base.params, alt.params)


def test_bad_criterion():
    data = load_stackloss()
    data.exog = np.asarray(data.exog)
    data.endog = np.asarray(data.endog)
    data.exog = sm.add_constant(data.exog, prepend=False)
    mod = RLM(data.endog, data.exog, M=norms.HuberT())
    with pytest.raises(ValueError, match="conv"):
        mod.fit(conv="unknown")


def test_fit_history_scale():
    # GH#9219 fit_history["scale"] recorded the inner WLS fit's scale rather
    # than the robust scale estimate, so the recorded series did not match the
    # robust scale recomputed from each iteration's parameters.
    data = load_stackloss()
    data.exog = sm.add_constant(np.asarray(data.exog), prepend=False)
    res = RLM(np.asarray(data.endog), data.exog, M=norms.HuberT()).fit()
    # Cross-checked against R MASS::rlm(stack.loss ~ ., data=stackloss,
    # psi=psi.huber, k=1.345, scale.est="MAD"): robust scale s = 2.4407
    # (statsmodels: 2.44054, agreeing to 4 significant figures).
    assert_allclose(res.scale, 2.4405, atol=1e-3)
    # Verify the full history, not just the last entry. fit_history["params"]
    # is seeded with a convergence sentinel, so its tail aligns with
    # fit_history["scale"]. At every iteration the recorded scale must equal the
    # robust MAD scale recomputed from that iteration's parameters; the pre-fix
    # code stored the inner WLS scale (~6.80) instead, so the whole series was
    # wrong, not only the final entry.
    endog, exog = res.model.endog, res.model.exog
    hist_scale = res.fit_history["scale"]
    hist_params = res.fit_history["params"][1:]
    assert len(hist_scale) == len(hist_params)
    for recorded, params in zip(hist_scale, hist_params, strict=True):
        assert_allclose(recorded, mad(endog - exog @ params, center=0))
    assert_allclose(hist_scale[-1], res.scale)


def test_summary_after_remove_data():
    # summary() must still work after remove_data() has been called
    data = load_stackloss()
    data.exog = sm.add_constant(data.exog, prepend=False)
    res = RLM(data.endog, data.exog, M=norms.HuberT()).fit()

    assert isinstance(res.summary(), Summary)
    res.remove_data()
    assert isinstance(res.summary(), Summary)


def test_summary_title():
    # GH: summary()'s `if title is not None:` always overwrote any
    # explicitly-provided title with the default, since the sentinel
    # default is 0 (not None), so only an explicit title=None ever
    # survived -- the exact opposite of the documented behavior.
    data = load_stackloss()
    data.exog = sm.add_constant(data.exog, prepend=False)
    res = RLM(data.endog, data.exog, M=norms.HuberT()).fit()

    default_title = "Robust Linear Model Regression Results"
    assert default_title in str(res.summary())
    assert default_title in str(res.summary(title=None))

    custom_title = "My Custom Title"
    smry = res.summary(title=custom_title)
    assert custom_title in str(smry)
    assert default_title not in str(smry)


def test_fit_invalid_options_raise():
    data = load_stackloss()
    data.exog = sm.add_constant(data.exog, prepend=False)
    mod = RLM(data.endog, data.exog, M=norms.HuberT())

    with pytest.raises(ValueError, match="cov"):
        mod.fit(cov="not-a-cov")
    with pytest.raises(ValueError, match="conv"):
        mod.fit(conv="not-a-conv")
    with pytest.raises(ValueError, match="scale_est"):
        mod.fit(scale_est="not-a-scale-est")

    # cov is upper-cased regardless of input case, unlike most other
    # string options in this codebase, which are lower-cased
    res_lower = mod.fit(cov="h2")
    res_upper = mod.fit(cov="H2")
    assert res_lower.cov == res_upper.cov == "H2"

    # scale_est="mad" (string form) matches the HuberScale-free default
    res_mad = mod.fit(scale_est="mad")
    res_default = mod.fit()
    assert_allclose(res_mad.params, res_default.params)


def test_rlm_results_direct_construction_validates_cov():
    # RLMResults.cov is a public constructor argument, independently
    # reachable without going through RLM.fit's validation
    from statsmodels.robust.robust_linear_model import RLMResults

    data = load_stackloss()
    data.exog = sm.add_constant(data.exog, prepend=False)
    mod = RLM(data.endog, data.exog, M=norms.HuberT())
    res = mod.fit()

    # lower-case input is accepted and stored upper-cased, same as fit()
    direct = RLMResults(
        mod, res.params, res.normalized_cov_params, res.scale, cov="h2"
    )
    assert direct.cov == "H2"
    assert_allclose(direct.bcov_scaled, mod.fit(cov="H2").bcov_scaled)

    with pytest.raises(ValueError, match="cov"):
        RLMResults(mod, res.params, res.normalized_cov_params, res.scale, cov="H4")
