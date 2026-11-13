"""NB-P regression cases from the cpunish discussion in gh-3863.

Reference values use direct log-gamma likelihoods differentiated at 70 digits,
independently of the production power-sum expansion. The original reporter
and the EXECUTIONS - 1 correction are Josef Perktold's.
"""
import json
from pathlib import Path
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest

from statsmodels.datasets import cpunish
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from statsmodels.discrete.discrete_model import NegativeBinomialP
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.tools import add_constant

_REFERENCE = json.loads(
    (Path(__file__).parent / "results/results_nbp_boundary.json").read_text()
)


def _input():
    data = cpunish.load_pandas()
    x = np.asarray(add_constant(data.exog / data.exog.max()))
    return np.asarray(data.endog), x, np.asarray(_REFERENCE["beta"])


@pytest.mark.parametrize("case", _REFERENCE["references"])
def test_cpunish_nbp_high_precision(case):
    y, x, beta = _input()
    params = np.r_[beta, case["alpha"]]
    if case["model"] == "NB-P":
        model = NegativeBinomialP(y, x, p=case["p"])
    else:
        model = ZeroInflatedNegativeBinomialP(
            y - 1, x, exog_infl=x[:, :2], p=case["p"]
        )
        params = np.r_[case["gamma"], params]
    assert_allclose(model.loglikeobs(params), case["ll"], rtol=2e-12, atol=2e-11)
    assert_allclose(model.score_obs(params), case["score_obs"],
                    rtol=2e-10, atol=2e-10)
    assert_allclose(model.hessian(params), case["hessian"],
                    rtol=2e-8 if case["alpha"] == .001 else 2e-9, atol=2e-9)


@pytest.mark.parametrize("p", [1.5, 2.5])
def test_nbp_transformed_score_and_response_derivative(p):
    y, x, beta = _input()
    model = NegativeBinomialP(y, x, p=p)
    mu = np.exp(x @ beta)
    # Independently differentiate the Poisson-limit score in the response.
    expected = np.column_stack((x, (y - .5 - mu) / mu**(2-p)))
    assert_allclose(model._deriv_score_obs_dendog(np.r_[beta, 0.]), expected,
                    rtol=2e-13, atol=2e-12)
    model._transparams = True
    params = np.r_[beta, np.log(1e-8)]
    assert_allclose(model.score_obs(params),
                    approx_fprime_cs(params, model.loglikeobs),
                    rtol=2e-11, atol=1e-10)
    assert_allclose(model.score(params), model.score_obs(params).sum(0),
                    rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("p", [1, 2])
def test_nbp_log_alpha_fit_preserves_public_start(p):
    y, x, beta = _input()
    start = np.r_[beta, .1]
    saved = start.copy()
    with warnings.catch_warnings():
        # The NB2 optimum is on the boundary: successful optimization does
        # not imply that the observed information can supply a covariance.
        warnings.simplefilter("ignore")
        result = NegativeBinomialP(y, x, p=p).fit(
            start_params=start, use_transparams=True,
            method="bfgs", maxiter=300, disp=0,
        )
    assert result.mle_retvals["converged"]
    assert np.isfinite(result.params).all()
    assert result.params[-1] > 0
    assert_allclose(start, saved, rtol=0, atol=0)
    assert result.llf >= -32.12561
