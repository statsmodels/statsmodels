"""
Created on Thu Aug  3 21:08:49 2017

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

# load data into module namespace
from statsmodels.datasets.cpunish import load
from statsmodels.discrete import discrete_margins
from statsmodels.discrete.discrete_model import (
    Logit,
    MNLogit,
    NegativeBinomial,
    NegativeBinomialP,
    Poisson,
    Probit,
)
import statsmodels.discrete.tests.results.results_count_margins as res_stata
from statsmodels.tools import numdiff
from statsmodels.tools.tools import add_constant

cpunish_data = load()
cpunish_data.exog = np.asarray(cpunish_data.exog)
cpunish_data.endog = np.asarray(cpunish_data.endog)
cpunish_data.exog[:, 3] = np.log(cpunish_data.exog[:, 3])
exog = add_constant(cpunish_data.exog, prepend=False)
endog = cpunish_data.endog - 1  # avoid zero-truncation
exog /= np.round(exog.max(0), 3)


class CheckMarginMixin:
    rtol_fac = 1

    def test_margins_table(self):
        res1 = self.res1
        sl = self.res1_slice
        rf = self.rtol_fac
        assert_allclose(self.margeff.margeff, self.res1.params[sl], rtol=1e-5 * rf)
        assert_allclose(self.margeff.margeff_se, self.res1.bse[sl], rtol=1e-6 * rf)
        assert_allclose(self.margeff.pvalues, self.res1.pvalues[sl], rtol=5e-6 * rf)
        assert_allclose(self.margeff.conf_int(), res1.margins_table[sl, 4:6],
                        rtol=1e-6 * rf)


class TestPoissonMargin(CheckMarginMixin):

    @classmethod
    def setup_class(cls):
        # here we do not need to check convergence from default start_params
        start_params = [14.1709, 0.7085, -3.4548, -0.539, 3.2368,  -7.9299,
                        -5.0529]
        mod_poi = Poisson(endog, exog)
        res_poi = mod_poi.fit(start_params=start_params)
        # res_poi = mod_poi.fit(maxiter=100)
        marge_poi = res_poi.get_margeff()
        cls.res = res_poi
        cls.margeff = marge_poi

        cls.rtol_fac = 1
        cls.res1_slice = slice(None, None, None)
        cls.res1 = res_stata.results_poisson_margins_cont


class TestPoissonMarginDummy(CheckMarginMixin):

    @classmethod
    def setup_class(cls):
        # here we do not need to check convergence from default start_params
        start_params = [14.1709, 0.7085, -3.4548, -0.539, 3.2368,  -7.9299,
                        -5.0529]
        mod_poi = Poisson(endog, exog)
        res_poi = mod_poi.fit(start_params=start_params)
        marge_poi = res_poi.get_margeff(dummy=True)
        cls.res = res_poi
        cls.margeff = marge_poi

        cls.res1_slice = [0, 1, 2, 3, 5, 6]
        cls.res1 = res_stata.results_poisson_margins_dummy


class TestNegBinMargin(CheckMarginMixin):

    @classmethod
    def setup_class(cls):
        # here we do not need to check convergence from default start_params
        start_params = [13.1996, 0.8582, -2.8005, -1.5031, 2.3849, -8.5552,
                        -2.88, 1.14]
        mod = NegativeBinomial(endog, exog)
        res = mod.fit(start_params=start_params, method="nm", maxiter=2000)
        marge = res.get_margeff()
        cls.res = res
        cls.margeff = marge

        cls.res1_slice = slice(None, None, None)
        cls.res1 = res_stata.results_negbin_margins_cont
        cls.rtol_fac = 5e1
        # negbin has lower agreement with Stata in this case


class TestNegBinMarginDummy(CheckMarginMixin):

    @classmethod
    def setup_class(cls):
        # here we do not need to check convergence from default start_params
        start_params = [13.1996, 0.8582, -2.8005, -1.5031, 2.3849, -8.5552,
                        -2.88, 1.14]
        mod = NegativeBinomial(endog, exog)
        res = mod.fit(start_params=start_params, method="nm", maxiter=2000)
        marge = res.get_margeff(dummy=True)
        cls.res = res
        cls.margeff = marge

        cls.res1_slice = cls.res1_slice = [0, 1, 2, 3, 5, 6]
        cls.res1 = res_stata.results_negbin_margins_dummy
        cls.rtol_fac = 5e1


class TestNegBinPMargin(CheckMarginMixin):
    # this is the same as the nb2 version above for NB-P, p=2

    @classmethod
    def setup_class(cls):
        # here we do not need to check convergence from default start_params
        start_params = [13.1996, 0.8582, -2.8005, -1.5031, 2.3849, -8.5552,
                        -2.88, 1.14]
        mod = NegativeBinomialP(endog, exog)   # checks also that default p=2
        res = mod.fit(start_params=start_params, method="nm", maxiter=2000)
        marge = res.get_margeff()
        cls.res = res
        cls.margeff = marge

        cls.res1_slice = slice(None, None, None)
        cls.res1 = res_stata.results_negbin_margins_cont
        cls.rtol_fac = 5e1
        # negbin has lower agreement with Stata in this case


def test_discrete_margins_summary_single_equation():
    rs = np.random.RandomState(20260822)
    n = 200
    x = rs.standard_normal(n)
    exog = add_constant(np.column_stack([x, rs.standard_normal(n)]))
    endog = rs.poisson(np.exp(exog @ [0.2, 0.3, -0.1]))

    res = Poisson(endog, exog).fit(disp=0)
    marge = res.get_margeff()
    text = str(marge.summary())

    assert "Poisson Marginal Effects" in text
    assert res.model.endog_names in text
    for val, se in zip(marge.margeff, marge.margeff_se, strict=True):
        assert f"{val:.4f}" in text
        assert f"{se:.3f}" in text


def test_discrete_margins_summary_multi_equation():
    rs = np.random.RandomState(20260823)
    n = 300
    x = rs.standard_normal((n, 2))
    exog = add_constant(x)
    lin0 = exog @ [0.0, 0.0, 0.0]
    lin1 = exog @ [0.5, 0.8, -0.4]
    lin2 = exog @ [-0.3, -0.2, 0.6]
    p = np.exp(np.column_stack([lin0, lin1, lin2]))
    p /= p.sum(1, keepdims=True)
    endog = np.array([rs.choice(3, p=row) for row in p])

    res = MNLogit(endog, exog).fit(disp=0)
    marge = res.get_margeff()
    text = str(marge.summary())

    assert "MNLogit Marginal Effects" in text
    # one sub-table per outcome category
    assert marge.margeff.shape[1] == 3
    for eq in range(marge.margeff.shape[1]):
        for val in marge.margeff[:, eq]:
            assert f"{val:.4f}" in text


@pytest.fixture(scope="module")
def overall_margins_results():
    # regressors: constant, two continuous, a dummy and a count variable
    rs = np.random.RandomState(4158)
    n = 300
    exog = np.column_stack(
        [
            rs.uniform(0.5, 2, size=(n, 2)),
            rs.randint(0, 2, size=n),
            rs.randint(0, 5, size=n),
        ]
    )
    exog = add_constant(exog.astype(float))
    linpred = exog @ [-0.3, 0.4, -0.3, 0.5, 0.1]
    y_bin = (rs.uniform(size=n) < 1 / (1 + np.exp(-linpred))).astype(float)
    y_count = rs.poisson(np.exp(linpred))
    y_overdisp = rs.poisson(np.exp(linpred) * rs.gamma(2, 0.5, size=n))
    y_multi = rs.randint(0, 3, size=n)
    return {
        "logit": Logit(y_bin, exog).fit(disp=0),
        "probit": Probit(y_bin, exog).fit(disp=0),
        "poisson": Poisson(y_count, exog).fit(disp=0),
        "negbin": NegativeBinomial(y_overdisp, exog).fit(disp=0),
        "mnlogit": MNLogit(y_multi, exog).fit(disp=0),
    }


def _margeff_cov_overall_per_obs(results, method, dummy, count):
    # delta method covariance of the average marginal effects with the
    # Jacobian at each observation averaged afterwards, as before GH#4158
    model = results.model
    params = np.asarray(results.params).ravel("F")
    exog = model.exog
    J = getattr(model, "J", 1)
    jac = numdiff.approx_fprime_cs(params, model._derivative_exog, args=(exog, method))
    jac = jac.mean(1)
    const_idx = discrete_margins._get_const_index(exog)[1]
    if dummy:
        dummy_idx = discrete_margins._get_dummy_index(exog, const_idx)[0]
        jac = discrete_margins._margeff_cov_params_dummy(
            model, jac, params, exog, dummy_idx, method, J
        )
    if count:
        count_idx = discrete_margins._get_count_index(exog, const_idx)[0]
        jac = discrete_margins._margeff_cov_params_count(
            model, jac, params, exog, count_idx, method, J
        )
    return jac @ np.asarray(results.cov_params()) @ jac.T


_overall_margins_options = [
    ("dydx", False, False),
    ("eyex", False, False),
    ("dyex", False, False),
    ("eydx", False, False),
    ("dydx", True, False),
    ("dydx", False, True),
    ("eydx", True, True),
]


@pytest.mark.parametrize(
    "model, method, dummy, count",
    [
        (model, *options)
        for model in ["logit", "probit", "poisson", "negbin", "mnlogit"]
        for options in _overall_margins_options
        # eyex and dyex currently raise in MNLogit._derivative_exog
        if not (model == "mnlogit" and "ex" in options[0])
    ],
)
def test_margeff_overall_cov_per_obs(
    overall_margins_results, model, method, dummy, count
):
    # GH#4158: the Jacobian of the average marginal effect is computed
    # directly, the result has to agree with averaging the Jacobian of the
    # marginal effect at each observation
    res = overall_margins_results[model]
    marge = res.get_margeff(method=method, dummy=dummy, count=count)
    cov = _margeff_cov_overall_per_obs(res, method, dummy, count)
    # drop the constant as in get_margeff
    effects_idx = discrete_margins._get_const_index(res.model.exog)[0]
    effects_idx = np.tile(effects_idx, getattr(res.model, "J", 1))
    cov = cov[effects_idx][:, effects_idx]
    assert_allclose(marge.margeff_cov, cov, rtol=1e-12, atol=1e-16)
    se = np.sqrt(np.diag(cov))
    assert_allclose(np.ravel(marge.margeff_se, order="F"), se, rtol=1e-12)


@pytest.mark.thread_unsafe(reason="Monkeypatches statsmodels.tools.numdiff")
@pytest.mark.parametrize("model", ["logit", "mnlogit"])
def test_margeff_overall_jacobian_shape(overall_margins_results, model, monkeypatch):
    # GH#4158: the Jacobian for at="overall" must not have a nobs dimension
    res = overall_margins_results[model]
    shapes = []
    orig_approx_fprime_cs = numdiff.approx_fprime_cs

    def approx_fprime_cs(*args, **kwargs):
        jac = orig_approx_fprime_cs(*args, **kwargs)
        shapes.append(jac.shape)
        return jac

    monkeypatch.setattr(numdiff, "approx_fprime_cs", approx_fprime_cs)
    res.get_margeff(at="overall")
    k_effects = res.model.exog.shape[1] * getattr(res.model, "J", 1)
    assert shapes == [(k_effects, res.params.size)]
