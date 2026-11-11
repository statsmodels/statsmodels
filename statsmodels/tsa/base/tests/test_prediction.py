from statsmodels.compat.pandas import MONTH_END

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statsmodels.tsa.base.prediction import PredictionResults


@pytest.fixture(params=[True, False])
def data(request):
    mean = np.arange(10.0)
    variance = np.arange(1, 11.0)
    if not request.param:
        return mean, variance
    idx = pd.date_range("2000-1-1", periods=10, freq=MONTH_END)
    return pd.Series(mean, index=idx), pd.Series(variance, index=idx)


def test_basic(data):
    is_pandas = isinstance(data[0], pd.Series)
    pred = PredictionResults(data[0], data[1])
    np.testing.assert_allclose(data[0], pred.predicted_mean)
    np.testing.assert_allclose(data[1], pred.var_pred_mean)
    if is_pandas:
        assert isinstance(pred.predicted_mean, pd.Series)
        assert isinstance(pred.var_pred_mean, pd.Series)
        assert isinstance(pred.se_mean, pd.Series)
    frame = pred.summary_frame()
    assert isinstance(frame, pd.DataFrame)
    assert list(
        frame.columns == ["mean", "mean_se", "mean_ci_lower", "mean_ci_upper"]
    )


@pytest.mark.parametrize("dist", [None, "norm", "t", stats.norm()])
def test_dist(data, dist):
    df = 10 if dist == "t" else None
    pred = PredictionResults(data[0], data[1], dist=dist, df=df)
    basic = PredictionResults(data[0], data[1])
    ci = pred.conf_int()
    basic_ci = basic.conf_int()
    if dist == "t":
        assert np.all(np.asarray(ci != basic_ci))
    else:
        assert np.all(np.asarray(ci == basic_ci))


def test_tvalues(data):
    # tvalues is exported as part of PredictionResults but had no direct
    # test: it is the ratio of the predicted mean to its standard error.
    pred = PredictionResults(data[0], data[1])
    expected = np.asarray(data[0]) / pred.se_mean
    np.testing.assert_allclose(pred.tvalues, expected)
    if isinstance(data[0], pd.Series):
        assert isinstance(pred.tvalues, pd.Series)
        assert pred.tvalues.name == "tvalues"


@pytest.mark.parametrize("alternative", ["two-sided", "larger", "smaller"])
def test_t_test(data, alternative):
    # t_test had no direct test coverage. Check the returned statistic
    # against the manual z-score, and the p-value against an independent
    # computation via scipy for each alternative.
    pred = PredictionResults(data[0], data[1])
    value = 1.0
    stat, pvalue = pred.t_test(value=value, alternative=alternative)

    mean = np.asarray(data[0])
    se = np.asarray(pred.se_mean)
    expected_stat = (mean - value) / se
    np.testing.assert_allclose(stat, expected_stat)

    if alternative == "two-sided":
        expected_pvalue = stats.norm.sf(np.abs(expected_stat)) * 2
    elif alternative == "larger":
        expected_pvalue = stats.norm.sf(expected_stat)
    else:
        expected_pvalue = stats.norm.cdf(expected_stat)
    np.testing.assert_allclose(pvalue, expected_pvalue)


def test_t_test_invalid_alternative(data):
    pred = PredictionResults(data[0], data[1])
    with pytest.raises(ValueError, match="alternative must be one of"):
        pred.t_test(alternative="not-a-real-alternative")


@pytest.mark.parametrize(
    "alias,canonical",
    [("2-sided", "two-sided"), ("2s", "two-sided"), ("l", "larger"), ("s", "smaller")],
)
def test_t_test_alternative_deprecated_alias(data, alias, canonical):
    # undocumented short forms still work but warn, and are equivalent to
    # spelling out the documented alternative
    pred = PredictionResults(data[0], data[1])
    with pytest.warns(FutureWarning, match="is a deprecated alias"):
        alias_result = pred.t_test(value=1.0, alternative=alias)
    canonical_result = pred.t_test(value=1.0, alternative=canonical)
    np.testing.assert_allclose(alias_result, canonical_result)
