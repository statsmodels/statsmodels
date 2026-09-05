"""Numerical references use small samples with hand-calculated moments."""

from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from statsmodels.regression.linear_model import OLS
from statsmodels.treatment.treatment_effects import TreatmentEffect


@pytest.fixture
def teff():
    # Control x=(0, 2), treated x=(2, 4): both sample variances are 2.
    x = np.array([0.0, 2.0, 2.0, 4.0])
    exog = np.column_stack([np.ones(4), x])
    selection = SimpleNamespace(
        predict=lambda: np.array([0.2, 0.4, 0.5, 0.8]),
        model=SimpleNamespace(exog=exog, exog_names=["const", "x"]),
    )
    return TreatmentEffect(
        OLS([0.0, 1.0, 3.0, 2.0], np.ones((4, 1))),
        np.array([0, 0, 1, 1]),
        results_select=selection,
    )


def test_overlap_unclipped(teff):
    teff.results_select.predict = lambda: np.array([0.0, 0.2, 0.8, 1.0])
    teff.ps_bounds = np.array([0.2, 0.8])
    result = teff.overlap_summary()
    assert list(result.index) == ["control", "treated"]
    assert_allclose(
        result[["min", "q25", "median", "q75", "max"]],
        [[0, 0.05, 0.1, 0.15, 0.2], [0.8, 0.85, 0.9, 0.95, 1]],
    )
    assert_allclose(result[["nobs", "n_below", "n_above"]], [[2, 1, 0], [2, 0, 1]])
    assert result.nobs.dtype.kind == "i"


@pytest.mark.parametrize(
    "target,means",
    [
        ("all", [8 / 7, 36 / 13]),
        (1, [16 / 11, 3]),
        (0, [1, 12 / 5]),
        ("treated", [16 / 11, 3]),
        ("control", [1, 12 / 5]),
        ("untreated", [1, 12 / 5]),
    ],
)
def test_balance_hand_calculated(teff, target, means):
    # ATE control weights=(5/4,5/3), treated=(2,5/4).
    # ATET control weights=(1/4,2/3), treated=(1,1).
    # ATC control weights=(1,1), treated=(1,1/4).
    table = teff.balance_table(effect_group=target)
    row = table.loc["x"]
    assert_allclose(row[["mean_control", "mean_treated"]], [1, 3])
    assert_allclose(row[["mean_control_weighted", "mean_treated_weighted"]], means)
    assert_allclose(row.smd, np.sqrt(2))
    assert_allclose(row.smd_weighted, (means[1] - means[0]) / np.sqrt(2))
    assert table.loc["const", ["smd", "smd_weighted"]].isna().all()


def test_balance_uses_clipped_scores(teff):
    teff.results_select.predict = lambda: np.array([0.0, 0.2, 0.8, 1.0])
    teff.prob_select = np.clip(teff.results_select.predict(), 0.2, 0.8)
    before = teff.prob_select.copy()
    result = teff.balance_table()
    assert_allclose(result.loc["x", ["smd", "smd_weighted"]], np.sqrt(2))
    assert_allclose(teff.prob_select, before)


def test_custom_exog_and_binary(teff):
    x = pd.DataFrame({"binary": [0, 1, 1, 1], "separated": [0, 0, 1, 1]})
    result = teff.balance_table(x)
    # Sample variances are 1/2 and 0: pooled SD=1/2.
    assert_allclose(result.loc["binary", "smd"], 1)
    assert_allclose(result.loc["binary", "smd_weighted"], 6 / 7)
    assert result.loc["separated", ["smd", "smd_weighted"]].isna().all()
    assert list(teff.balance_table(x.to_numpy()).index) == ["x0", "x1"]


@pytest.mark.parametrize(
    "exog",
    [
        np.ones(4),
        np.ones((3, 2)),
        np.empty((4, 0)),
        np.full((4, 1), np.nan),
        np.full((4, 1), np.inf),
    ],
)
def test_invalid_exog(teff, exog):
    with pytest.raises(ValueError, match="exog must"):
        teff.balance_table(exog)


@pytest.mark.parametrize(
    "prob",
    [[0.2, 0.5], [0.2, np.nan, 0.5, 0.8], [-0.1, 0.2, 0.5, 0.8], [0.1, 0.2, 0.5, 1.1]],
)
def test_invalid_predictions(teff, prob):
    teff.results_select.predict = lambda: prob
    with pytest.raises(ValueError, match="selection predictions"):
        teff.overlap_summary()


@pytest.mark.parametrize("method", ["overlap_summary", "balance_table"])
def test_missing_selection(teff, method):
    del teff.results_select
    with pytest.raises(ValueError, match="require results_select"):
        getattr(teff, method)()


@pytest.mark.parametrize("treatment", [[0, 0, 0, 0], [0, 0, 1, 2], [[0, 0, 1, 1]]])
def test_invalid_treatment(teff, treatment):
    teff.treatment = np.array(treatment)
    with pytest.raises(ValueError, match="treatment"):
        teff.overlap_summary()


def test_small_arm(teff):
    teff.treatment = np.array([0, 0, 0, 1])
    assert teff.overlap_summary().loc["treated", "nobs"] == 1
    with pytest.raises(ValueError, match="at least two"):
        teff.balance_table()


def test_invalid_target(teff):
    with pytest.raises(ValueError, match="effect_group"):
        teff.balance_table(effect_group="invalid")
