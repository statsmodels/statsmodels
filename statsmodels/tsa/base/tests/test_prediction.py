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


@pytest.mark.parametrize("alpha", [0.05, 0.2])
@pytest.mark.parametrize(
    "dist, df, reference",
    [
        (None, None, stats.norm()),
        ("norm", None, stats.norm()),
        ("t", 2.5, stats.t(2.5)),
        (stats.t(5), None, stats.t(5)),
    ],
    ids=["default", "norm", "t", "frozen"],
)
def test_conf_int_summary_frame(data, dist, df, reference, alpha):
    # Intervals are the mean -/+ the 1 - alpha / 2 quantile times the
    # standard error
    pred = PredictionResults(data[0], data[1], dist=dist, df=df)
    mean = np.asarray(data[0])
    se = np.sqrt(np.asarray(data[1]))
    q = reference.ppf(1 - alpha / 2)
    lower, upper = mean - q * se, mean + q * se

    ci = pred.conf_int(alpha=alpha)
    np.testing.assert_allclose(np.asarray(ci), np.column_stack([lower, upper]))
    frame = pred.summary_frame(alpha=alpha)
    assert list(frame.columns) == ["mean", "mean_se", "mean_ci_lower", "mean_ci_upper"]
    np.testing.assert_allclose(
        frame.to_numpy(), np.column_stack([mean, se, lower, upper])
    )
    if isinstance(data[0], pd.Series):
        assert list(ci.columns) == ["lower", "upper"]
        assert ci.index.equals(data[0].index)
        assert frame.index.equals(data[0].index)
    else:
        assert isinstance(ci, np.ndarray)
        assert frame.index.equals(pd.RangeIndex(10))


@pytest.mark.parametrize("dist", [None, "norm", stats.norm()])
def test_df_only_with_t(dist):
    with pytest.raises(ValueError, match='df must be None when dist is not "t"'):
        PredictionResults(np.zeros(3), np.ones(3), dist=dist, df=5)


def test_invalid_dist():
    with pytest.raises(ValueError, match="dist must be a None"):
        PredictionResults(np.zeros(3), np.ones(3), dist="cauchy")


def test_row_labels():
    labels = ["a", "b", "c"]
    mean = np.array([1.0, 2.0, 3.0])
    var = np.array([1.0, 4.0, 9.0])
    pred = PredictionResults(mean, var, row_labels=labels)
    assert pred.row_labels == labels
    assert pred.predicted_mean.index.equals(pd.Index(labels))
    assert pred.se_mean.index.equals(pd.Index(labels))
    assert pred.summary_frame().index.equals(pd.Index(labels))
    np.testing.assert_allclose(pred.se_mean, [1.0, 2.0, 3.0])

    # Given labels replace the index of a Series
    pred = PredictionResults(pd.Series(mean, index=[7, 8, 9]), var, row_labels=labels)
    assert pred.predicted_mean.index.equals(pd.Index(labels))

    # Without labels or a pandas index, the results are arrays
    pred = PredictionResults(mean, var)
    assert pred.row_labels is None
    assert isinstance(pred.predicted_mean, np.ndarray)
    assert isinstance(pred.se_mean, np.ndarray)


@pytest.mark.parametrize(
    "row_labels", [None, pd.date_range("2000-01-01", periods=4, freq="MS")]
)
def test_covariance_var_pred_mean(row_labels):
    # var_pred_mean can hold the covariance matrix of each prediction
    rng = np.random.default_rng(20261121)
    nobs, k = 4, 2
    mean = rng.standard_normal((nobs, k))
    a = rng.standard_normal((nobs, k, k))
    var = a @ a.transpose(0, 2, 1) + np.eye(k)
    pred = PredictionResults(mean, var, row_labels=row_labels)

    assert pred.var_pred_mean.shape == (nobs, k, k)
    np.testing.assert_allclose(pred.var_pred_mean, var)
    expected_se = np.sqrt([[var[t, i, i] for i in range(k)] for t in range(nobs)])
    np.testing.assert_allclose(np.asarray(pred.se_mean), expected_se)
    if row_labels is None:
        # The lower limits of all series come first, then the upper ones
        q = stats.norm.ppf(0.975)
        expected = np.hstack([mean - q * expected_se, mean + q * expected_se])
        np.testing.assert_allclose(pred.conf_int(), expected)
    else:
        assert isinstance(pred.se_mean, pd.DataFrame)
        assert pred.se_mean.index.equals(row_labels)
