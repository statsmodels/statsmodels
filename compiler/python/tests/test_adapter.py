import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from causal_compiler import evaluate_ols_contrast, extract_ols_contrast


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 40
    t = rng.uniform(0.0, 2.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 1.2 + 0.8 * t + 0.5 * x1 - 0.3 * x2 + 1.1 * x3 + 0.4 * t * x1 - 0.7 * t * x2
    y = y + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "T": t, "x1": x1, "x2": x2, "x3": x3})


def test_evaluate_ols_contrast_matches_statsmodels_predict(sample_df):
    results = smf.ols("y ~ T * (x1 + x2) + x3", data=sample_df).fit()
    scoring = sample_df.iloc[:10].copy()

    actual = evaluate_ols_contrast(results, "T", (0.0, 1.5), scoring)

    treated = results.predict(scoring.assign(T=1.5))
    control = results.predict(scoring.assign(T=0.0))
    expected = treated - control

    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-12)
    assert actual.index.equals(scoring.index)
    assert actual.name == "treatment_effect"


def test_extract_handles_reversed_interaction_order(sample_df):
    results = smf.ols("y ~ x1 * T + x2", data=sample_df).fit()
    inputs = extract_ols_contrast(results, "T", (0.0, 1.0), sample_df)

    assert inputs.interaction_columns == ("x1",)
    assert inputs.treatment_delta == 1.0
    actual = evaluate_ols_contrast(results, "T", (0.0, 1.0), sample_df)
    expected = results.predict(sample_df.assign(T=1.0)) - results.predict(
        sample_df.assign(T=0.0)
    )
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-12)


def test_extract_rejects_non_ols_model(sample_df):
    binary = sample_df.assign(yb=(sample_df["y"] > sample_df["y"].median()).astype(int))
    results = smf.logit("yb ~ T * x1", data=binary).fit(disp=False)

    with pytest.raises(TypeError, match="OLS"):
        extract_ols_contrast(results, "T", (0.0, 1.0), binary)


def test_extract_rejects_transformed_treatment_or_higher_order(sample_df):
    transformed = smf.ols("y ~ T * x1 + I(T ** 2)", data=sample_df).fit()
    with pytest.raises(ValueError, match="transformed treatment"):
        extract_ols_contrast(transformed, "T", (0.0, 1.0), sample_df)

    three_way = smf.ols("y ~ T * x1 * x2", data=sample_df).fit()
    with pytest.raises(ValueError, match="two-factor"):
        extract_ols_contrast(three_way, "T", (0.0, 1.0), sample_df)


def test_extract_rejects_equal_or_bool_treatment_values(sample_df):
    results = smf.ols("y ~ T * x1", data=sample_df).fit()
    with pytest.raises(ValueError, match="distinct finite"):
        extract_ols_contrast(results, "T", (1.0, 1.0), sample_df)
    with pytest.raises(ValueError, match="distinct finite"):
        extract_ols_contrast(results, "T", (False, True), sample_df)
