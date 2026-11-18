import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from causal_compiler import (
    emit_intervention_predict_mlir,
    emit_predict_sequence_mlir,
    evaluate_intervention_predict,
    extract_intervention_inputs,
)


@pytest.fixture
def glm_df():
    rng = np.random.default_rng(123)
    n = 80
    t = rng.uniform(0.0, 2.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    eta = -0.3 + 0.7 * t + 0.4 * x1 - 0.5 * x2 + 0.2 * x3 + 0.3 * t * x1 - 0.4 * t * x2
    prob = 1.0 / (1.0 + np.exp(-eta))
    y_bin = rng.binomial(1, prob)
    y_cnt = rng.poisson(np.exp(np.clip(eta, -2.0, 2.0)))
    y_cont = eta + rng.normal(scale=0.1, size=n)
    return pd.DataFrame(
        {
            "y_bin": y_bin,
            "y_cnt": y_cnt,
            "y_cont": y_cont,
            "T": t,
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )


@pytest.mark.parametrize(
    "formula,fit_fn",
    [
        ("y_cont ~ T * (x1 + x2) + x3", lambda f, d: smf.ols(f, data=d).fit()),
        ("y_bin ~ T * (x1 + x2) + x3", lambda f, d: smf.logit(f, data=d).fit(disp=False)),
        ("y_cnt ~ T * (x1 + x2) + x3", lambda f, d: smf.poisson(f, data=d).fit(disp=False)),
        (
            "y_bin ~ T * (x1 + x2) + x3",
            lambda f, d: smf.glm(f, data=d, family=sm.families.Binomial()).fit(),
        ),
    ],
    ids=["ols", "logit", "poisson", "glm_binomial"],
)
def test_evaluate_intervention_predict_matches_statsmodels(glm_df, formula, fit_fn):
    results = fit_fn(formula, glm_df)
    doses = [0.0, 0.5, 1.5, 3.0]
    scoring = glm_df.iloc[:15].copy()

    actual = evaluate_intervention_predict(results, "T", doses, scoring)
    expected = np.column_stack(
        [results.predict(scoring.assign(T=d)).to_numpy() for d in doses]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_emit_predict_sequence_and_fused_mlir(glm_df):
    results = smf.logit("y_bin ~ T * (x1 + x2) + x3", data=glm_df).fit(disp=False)
    inputs = extract_intervention_inputs(results, "T", [0.0, 1.0, 2.0], glm_df.iloc[:10])

    unfused = emit_predict_sequence_mlir(inputs)
    fused = emit_intervention_predict_mlir(inputs)

    assert unfused.count('"ci.predict"') == 3
    assert 'link = "logit"' in unfused
    assert '"ci.intervention_predict"' in fused
    assert "-> tensor<10x3xf64>" in fused
    assert "doses = array<f64:" in fused
