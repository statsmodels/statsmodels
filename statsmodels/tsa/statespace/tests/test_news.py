r"""
Tests for news results

Author: Chad Fulton
License: BSD-3
"""
from statsmodels.compat.pandas import (
    FUTURE_STACK,
    assert_frame_equal,
    assert_series_equal,
)

import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest

from statsmodels import datasets
from statsmodels.tsa.statespace import (
    dynamic_factor,
    sarimax,
    structural,
    varmax,
)

dta = datasets.macrodata.load_pandas().data
dta.index = pd.period_range(start="1959Q1", end="2009Q3", freq="Q")


def check_impact_indices(news, impact_dates, impacted_variables):
    # Note: the index for impacts is only a time index, because we compute
    # impacts for all variables during these time periods.
    for attr in ["total_impacts", "update_impacts", "revision_impacts",
                 "post_impacted_forecasts", "prev_impacted_forecasts"]:
        val = getattr(news, attr)
        assert_(val.index.equals(impact_dates))
        assert_equal(val.columns.tolist(), impacted_variables)


def check_revision_indices(news, revisions_index):
    # Note: revision indices must be a MultiIndex, because for each time
    # period, not all variables necessarily revised.

    # If there are no revisions, we just check the length is zero
    if news.news_results.revision_impacts is None:
        assert_equal(len(news.revisions_iloc["revision date"]), 0)
        assert_equal(len(news.revisions_iloc["revised variable"]), 0)
        assert_equal(len(news.revisions_ix["revision date"]), 0)
        assert_equal(len(news.revisions_ix["revised variable"]), 0)
    # Otherwise, check that the revision indexes are correct
    else:
        # Get the integer locations of the given indexes
        dates = news.previous.model._index
        endog_names = news.previous.model.endog_names
        if isinstance(endog_names, str):
            endog_names = [endog_names]
        desired_ix = revisions_index.to_frame().reset_index(drop=True)
        desired_iloc = desired_ix.copy()
        desired_iloc["revision date"] = [
            dates.get_loc(date) for date in desired_ix["revision date"]]
        desired_iloc["revised variable"] = [
            endog_names.index(name)
            for name in desired_ix["revised variable"]]

        assert_(news.revisions_iloc.equals(
            desired_iloc.astype(news.revisions_iloc.dtypes)))
        assert_(news.revisions_ix.equals(desired_ix))


def check_update_indices(news, updates_index):
    # Note: update indices are also a MultiIndex, for the same reason as the
    # revision indices.
    # If there are no updates, we just check the length is zero
    if news.news_results.update_impacts is None:
        assert_equal(len(news.updates_iloc["update date"]), 0)
        assert_equal(len(news.updates_iloc["updated variable"]), 0)
        assert_equal(len(news.updates_ix["update date"]), 0)
        assert_equal(len(news.updates_ix["updated variable"]), 0)
    # Otherwise, check that the revision indexes are correct
    else:
        # Get the integer locations of the given indexes
        dates = news.updated.model._index
        endog_names = news.updated.model.endog_names
        if isinstance(endog_names, str):
            endog_names = [endog_names]
        desired_ix = updates_index.to_frame().reset_index(drop=True)
        desired_iloc = desired_ix.copy()
        desired_iloc["update date"] = [
            dates.get_loc(date) for date in desired_ix["update date"]]
        desired_iloc["updated variable"] = [
            endog_names.index(name)
            for name in desired_ix["updated variable"]]

        assert_(news.updates_iloc.equals(
            desired_iloc.astype(news.updates_iloc.dtypes)))
        assert_(news.updates_ix.equals(desired_ix))


def check_news_indices(news, updates_index, impact_dates):
    # News are computed only from updates, so the news indices are the same as
    # the update indices
    if len(updates_index):
        news_index = updates_index
    else:
        news_index = pd.MultiIndex.from_product(
            [[], []], names=["update date", "updated variable"])
    endog_names = news.previous.model.endog_names
    if isinstance(endog_names, str):
        endog_names = [endog_names]

    assert_(news.news.index.equals(news_index))
    assert_(news.update_forecasts.index.equals(news_index))
    assert_(news.update_realized.index.equals(news_index))
    assert_(news.weights.index.equals(news_index))

    weights_columns = pd.MultiIndex.from_product([impact_dates, endog_names])
    assert_(news.weights.columns.equals(weights_columns))


def check_news(news, revisions, updates, impact_dates, impacted_variables,
               revisions_index, updates_index,
               revision_impacts, update_impacts,
               prev_impacted_forecasts, post_impacted_forecasts,
               update_forecasts, update_realized, news_desired, weights):
    # Note: we use atol=1e-12 to handle cases where impacts, etc. are equal to
    # zero, but numerical precision of the Kalman filter procedures gives an
    # answer of e.g., 1e-16.

    # Note: Here we set the tolerance to be slightly negative, since some of
    # the tests have weights or impacts exactly equal to zero, while we still
    # want to include those in tests.
    news.tolerance = -1e-10

    # - Indexes --------------------------------------------------------------
    # Index of impacts
    check_impact_indices(news, impact_dates, impacted_variables)

    # Reivision indices
    check_revision_indices(news, revisions_index)

    # Update indices
    check_update_indices(news, updates_index)

    # News indices
    check_news_indices(news, updates_index, impact_dates)

    # - Impacts --------------------------------------------------------------
    if updates:
        assert_allclose(news.update_impacts, update_impacts, atol=1e-12)
    else:
        assert_(np.all(news.update_impacts.isna()))

    # Impacts from revisions
    if revisions:
        assert_allclose(news.revision_impacts, revision_impacts, atol=1e-12)
    else:
        assert_(news.news_results.revision_impacts is None)
        assert_(np.all(news.revision_impacts.isna()))

    # Total impacts
    total_impacts = (news.revision_impacts.astype(float).fillna(0) +
                     news.update_impacts.astype(float).fillna(0))
    assert_allclose(news.total_impacts, total_impacts, atol=1e-12)

    # - Impacted variable forecasts ------------------------------------------
    assert_allclose(news.prev_impacted_forecasts, prev_impacted_forecasts,
                    atol=1e-12)
    assert_allclose(news.post_impacted_forecasts, post_impacted_forecasts,
                    atol=1e-12)

    # - News -----------------------------------------------------------------
    assert_allclose(news.update_forecasts, update_forecasts, atol=1e-12)
    assert_allclose(news.update_realized, update_realized, atol=1e-12)
    # The "news" is simply the forecast error
    assert_allclose(news.news, news_desired, atol=1e-12)
    # The weight is zero on previously known data, and is geometrically
    # declining (according to the AR parameter) in the forecast period
    assert_allclose(news.weights, weights, atol=1e-12)

    # - Table: data revisions ------------------------------------------------
    assert_equal(news.data_revisions.columns.tolist(),
                 ["revised", "observed (prev)", "detailed impacts computed"])
    assert_equal(news.data_revisions.index.names,
                 ["revision date", "revised variable"])
    assert_(news.data_revisions.index.equals(revisions_index))

    # - Table: data updates --------------------------------------------------
    assert_equal(news.data_updates.columns.tolist(),
                 ["observed", "forecast (prev)"])
    assert_equal(news.data_updates.index.names,
                 ["update date", "updated variable"])
    assert_(news.data_updates.index.equals(news.news.index))
    assert_allclose(news.data_updates["forecast (prev)"],
                    news.update_forecasts, atol=1e-12)
    assert_allclose(news.data_updates["observed"], news.update_realized,
                    atol=1e-12)

    # - Table: details_by_impact ---------------------------------------------
    details_by_impact = news.details_by_impact
    desired = ["observed", "forecast (prev)", "news", "weight", "impact"]
    assert_equal(details_by_impact.columns.tolist(), desired)
    desired = ["impact date", "impacted variable",
               "update date", "updated variable"]
    assert_equal(details_by_impact.index.names, desired)

    if updates:
        actual = (news.details_by_impact["forecast (prev)"]
                      .drop_duplicates()
                      .reset_index([0, 1])["forecast (prev)"])
        assert_allclose(actual, news.update_forecasts, atol=1e-12)
        actual = (news.details_by_impact["observed"]
                      .drop_duplicates().reset_index([0, 1])["observed"])
        assert_allclose(actual, news.update_realized, atol=1e-12)
        actual = (news.details_by_impact["news"]
                      .drop_duplicates().reset_index([0, 1])["news"])
        assert_allclose(actual, news.news, atol=1e-12)

        # Weights
        assert_allclose(details_by_impact["weight"].unstack([0, 1]),
                        news.weights, atol=1e-12)

        # Impact of news
        actual = (news.details_by_impact["impact"]
                      .unstack([2, 3]).sum(axis=1).unstack(1))
        assert_allclose(actual, news.update_impacts, atol=1e-12)

    # - Table: details_by_update ---------------------------------------------
    details_by_update = news.details_by_update
    desired = ["news", "weight", "impact"]
    assert_equal(details_by_update.columns.tolist(), desired)
    desired = ["update date", "updated variable", "observed",
               "forecast (prev)", "impact date", "impacted variable"]
    assert_equal(details_by_update.index.names, desired)

    if updates:
        # News
        # Special case for Pandas = 0.23, see above
        actual = (news.details_by_update["news"]
                      .drop_duplicates().reset_index([2, 3, 4, 5])["news"])
        assert_allclose(actual, news.news, atol=1e-12)

        # Weights
        assert_allclose(news.details_by_update["weight"].unstack([4, 5]),
                        news.weights, atol=1e-12)

        # Impact of news
        actual = (news.details_by_update["impact"]
                      .unstack([4, 5]).sum(axis=0).unstack(1))
        assert_allclose(actual, news.update_impacts, atol=1e-12)

    # - Table: impacts -------------------------------------------------------
    impacts = news.impacts
    desired = ["estimate (prev)", "impact of revisions", "impact of news",
               "total impact", "estimate (new)"]
    assert_equal(impacts.columns.tolist(), desired)
    desired = ["impact date", "impacted variable"]
    assert_equal(impacts.index.names, desired)

    assert_allclose(
        impacts.loc[:, "estimate (prev)"],
        news.prev_impacted_forecasts.stack(**FUTURE_STACK),
        atol=1e-12,
    )
    assert_allclose(
        impacts.loc[:, "impact of revisions"],
        news.revision_impacts.astype(float).fillna(0).stack(**FUTURE_STACK),
        atol=1e-12,
    )
    assert_allclose(
        impacts.loc[:, "impact of news"],
        news.update_impacts.astype(float).fillna(0).stack(**FUTURE_STACK),
        atol=1e-12,
    )
    assert_allclose(
        impacts.loc[:, "total impact"],
        news.total_impacts.stack(**FUTURE_STACK),
        atol=1e-12,
    )
    assert_allclose(
        impacts.loc[:, "estimate (new)"],
        news.post_impacted_forecasts.stack(**FUTURE_STACK),
        atol=1e-12,
    )


@pytest.mark.parametrize("revisions", [True, False])
@pytest.mark.parametrize("updates", [True, False])
@pytest.mark.parametrize("revisions_details_start", [True, False, -2])
def test_sarimax_time_invariant(revisions, updates, revisions_details_start):
    # Construct previous and updated datasets
    endog = dta["infl"].copy()
    comparison_type = None
    if updates:
        endog1 = endog.loc[:"2009Q2"].copy()
        endog2 = endog.loc[:"2009Q3"].copy()
    else:
        endog1 = endog.loc[:"2009Q3"].copy()
        endog2 = endog.loc[:"2009Q3"].copy()
        # Without updates and without NaN values, we need to specify that
        # the type of the comparison object that we're passing is "updated"
        comparison_type = "updated"
    if revisions:
        endog1.iloc[-1] = 0.

    # Get the previous results object and compute the news
    mod = sarimax.SARIMAX(endog1)
    res = mod.smooth([0.5, 1.0])
    news = res.news(endog2, start="2009Q2", end="2010Q1",
                    comparison_type=comparison_type,
                    revisions_details_start=revisions_details_start)

    # Compute the true values for each combination of (revsions, updates)
    impact_dates = pd.period_range(start="2009Q2", end="2010Q1", freq="Q")
    impacted_variables = ["infl"]

    # Revisions
    if revisions and updates:
        revisions_index = pd.MultiIndex.from_arrays(
            [endog1.index[-1:], ["infl"]],
            names=["revision date", "revised variable"])
        # If we have updates, the revision is to 2009Q2
        revision_impacts = endog2.iloc[-2] * 0.5**np.arange(4).reshape(4, 1)
    elif revisions:
        revisions_index = pd.MultiIndex.from_arrays(
            [endog1.index[-1:], ["infl"]],
            names=["revision date", "revised variable"])
        # With no updates, the revision is to 2009Q3
        revision_impacts = np.r_[
            0, endog2.iloc[-1] * 0.5**np.arange(3)].reshape(4, 1)
    else:
        revisions_index = pd.MultiIndex.from_arrays(
            [[], []], names=["revision date", "revised variable"])
        revision_impacts = None

    # Updates
    if updates:
        updates_index = pd.MultiIndex.from_arrays(
            [pd.period_range(start="2009Q3", periods=1, freq="Q"), ["infl"]],
            names=["update date", "updated variable"])
        update_impacts = np.array([[
            0, endog.loc["2009Q3"] - 0.5 * endog.loc["2009Q2"],
            0.5 * endog.loc["2009Q3"] - 0.5**2 * endog.loc["2009Q2"],
            0.5**2 * endog.loc["2009Q3"] - 0.5**3 * endog.loc["2009Q2"]]]).T
    else:
        updates_index = pd.MultiIndex.from_arrays(
            [[], []], names=["update date", "updated variable"])
        update_impacts = None
    print(update_impacts)

    # Impact forecasts
    if updates:
        prev_impacted_forecasts = np.r_[
            endog1.iloc[-1] * 0.5**np.arange(4)].reshape(4, 1)
    else:
        prev_impacted_forecasts = np.r_[
            endog1.iloc[-2], endog1.iloc[-1] * 0.5**np.arange(3)].reshape(4, 1)
    post_impacted_forecasts = np.r_[
        endog2.iloc[-2], 0.5 ** np.arange(3) * endog2.iloc[-1]].reshape(4, 1)

    # News
    if updates:
        # Note: update_forecasts is created using the endog2 dataset even if
        # there were revisions, because it should be computed after revisions
        # have already been taken into account
        update_forecasts = [0.5 * endog2.loc["2009Q2"]]
        update_realized = [endog2.loc["2009Q3"]]
        news_desired = [update_realized[i] - update_forecasts[i]
                        for i in range(len(update_forecasts))]
        weights = pd.DataFrame(np.r_[0, 0.5**np.arange(3)]).T
    else:
        update_forecasts = pd.Series([], dtype=np.float64)
        update_realized = pd.Series([], dtype=np.float64)
        news_desired = pd.Series([], dtype=np.float64)
        weights = pd.DataFrame(np.zeros((0, 4)))

    # Run unit tests
    check_news(news, revisions, updates, impact_dates, impacted_variables,
               revisions_index, updates_index,
               revision_impacts, update_impacts,
               prev_impacted_forecasts, post_impacted_forecasts,
               update_forecasts, update_realized, news_desired, weights)


@pytest.mark.parametrize("revisions", [True, False])
@pytest.mark.parametrize("updates", [True, False])
@pytest.mark.parametrize("which", ["exog", "trend"])
def test_sarimax_time_varying(revisions, updates, which):
    # This is primarily a test that the `news` method works with a time-varying
    # setup (i.e., time-varying state space matrices). It tests a time-varying
    # SARIMAX model where the time-varying component has been set to zeros
    # against a time-invariant version of the model.

    # Construct previous and updated datasets
    endog = dta["infl"].copy()
    comparison_type = None
    if updates:
        endog1 = endog.loc[:"2009Q2"].copy()
        endog2 = endog.loc[:"2009Q3"].copy()
    else:
        endog1 = endog.loc[:"2009Q3"].copy()
        endog2 = endog.loc[:"2009Q3"].copy()
        # Without updates and without NaN values, we need to specify that
        # the type of the comparison object that we're passing is "updated"
        comparison_type = "updated"
    if revisions:
        endog1.iloc[-1] = 0.

    exog1 = None
    exog2 = None
    trend = "n"
    if which == "exog":
        exog1 = np.ones_like(endog1)
        exog2 = np.ones_like(endog2)
    elif which == "trend":
        trend = "t"

    # Compute the news from a model with a trend/exog term (so the model is
    # time-varying), but with the coefficient set to zero (so that it will be
    # equivalent to the time-invariant model)
    mod1 = sarimax.SARIMAX(endog1, exog=exog1, trend=trend)
    res1 = mod1.smooth([0., 0.5, 1.0])
    news1 = res1.news(endog2, exog=exog2, start="2008Q1", end="2009Q3",
                      comparison_type=comparison_type)

    # Compute the news from a model without a trend term
    mod2 = sarimax.SARIMAX(endog1)
    res2 = mod2.smooth([0.5, 1.0])
    news2 = res2.news(endog2, start="2008Q1", end="2009Q3",
                      comparison_type=comparison_type)

    attrs = ["total_impacts", "update_impacts", "revision_impacts", "news",
             "weights", "update_forecasts", "update_realized",
             "prev_impacted_forecasts", "post_impacted_forecasts",
             "revisions_iloc", "revisions_ix", "updates_iloc", "updates_ix"]

    for attr in attrs:
        w = getattr(news1, attr)
        x = getattr(news2, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)


@pytest.mark.parametrize("revisions", [True, False])
@pytest.mark.parametrize("updates", [True, False])
def test_unobserved_components_time_varying(revisions, updates):
    # This is primarily a test that the `news` method works with a time-varying
    # setup (i.e., time-varying state space matrices). It tests a time-varying
    # UnobservedComponents model where the time-varying component has been set
    # to zeros against a time-invariant version of the model.

    # Construct previous and updated datasets
    endog = dta["infl"].copy()
    comparison_type = None
    if updates:
        endog1 = endog.loc[:"2009Q2"].copy()
        endog2 = endog.loc[:"2009Q3"].copy()
    else:
        endog1 = endog.loc[:"2009Q3"].copy()
        endog2 = endog.loc[:"2009Q3"].copy()
        # Without updates and without NaN values, we need to specify that
        # the type of the comparison object that we're passing is "updated"
        comparison_type = "updated"
    if revisions:
        endog1.iloc[-1] = 0.

    exog1 = np.ones_like(endog1)
    exog2 = np.ones_like(endog2)

    # Compute the news from a model with a trend/exog term (so the model is
    # time-varying), but with the coefficient set to zero (so that it will be
    # equivalent to the time-invariant model)
    mod1 = structural.UnobservedComponents(endog1, "llevel", exog=exog1)
    res1 = mod1.smooth([0.5, 0.2, 0.0])
    news1 = res1.news(endog2, exog=exog2, start="2008Q1", end="2009Q3",
                      comparison_type=comparison_type)

    # Compute the news from a model without a trend term
    mod2 = structural.UnobservedComponents(endog1, "llevel")
    res2 = mod2.smooth([0.5, 0.2])
    news2 = res2.news(endog2, start="2008Q1", end="2009Q3",
                      comparison_type=comparison_type)

    attrs = ["total_impacts", "update_impacts", "revision_impacts", "news",
             "weights", "update_forecasts", "update_realized",
             "prev_impacted_forecasts", "post_impacted_forecasts",
             "revisions_iloc", "revisions_ix", "updates_iloc", "updates_ix"]

    for attr in attrs:
        w = getattr(news1, attr)
        x = getattr(news2, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)


@pytest.mark.parametrize("revisions", [True, False])
@pytest.mark.parametrize("updates", [True, False])
def test_varmax_time_invariant(revisions, updates):
    # Construct previous and updated datasets
    endog = dta[["realgdp", "unemp"]].copy()
    endog["realgdp"] = np.log(endog["realgdp"]).diff() * 400
    endog = endog.iloc[1:]
    comparison_type = None
    if updates:
        endog1 = endog.loc[:"2009Q2"].copy()
        endog2 = endog.loc[:"2009Q3"].copy()
    else:
        endog1 = endog.loc[:"2009Q3"].copy()
        endog2 = endog.loc[:"2009Q3"].copy()
        # Without updates and without NaN values, we need to specify that
        # the type of the comparison object that we're passing is "updated"
        comparison_type = "updated"
    if revisions:
        # TODO: add test for only one of the variables revising?
        endog1.iloc[-1] = 0.

    # Get the previous results object and compute the news
    mod = varmax.VARMAX(endog1, trend="n")
    params = np.r_[0.5, 0.1, 0.2, 0.9, 1., 0.1, 1.1]
    res = mod.smooth(params)
    news = res.news(endog2, start="2009Q2", end="2010Q1",
                    comparison_type=comparison_type)

    # Compute the true values for each combination of (revsions, updates)
    impact_dates = pd.period_range(start="2009Q2", end="2010Q1", freq="Q")
    impacted_variables = ["realgdp", "unemp"]

    # Matrix powers of transition matrix
    Z = np.zeros((2, 2))
    T0 = np.eye(2)
    T1 = mod["transition"]
    T2 = T1 @ T1
    T3 = T1 @ T2

    # Revisions
    if revisions and updates:
        revisions_index = pd.MultiIndex.from_product(
            [endog1.index[-1:], ["realgdp", "unemp"]],
            names=["revision date", "revised variable"])
        # If we have updates, the revision is to 2009Q2
        # Note: this ".values" and all of those below are only required for
        # Pandas = 0.23, and can be removed once that is no longer a supported
        # dependency
        tmp = endog2.iloc[-2].values
        revision_impacts = np.c_[T0 @ tmp, T1 @ tmp, T2 @ tmp, T3 @ tmp].T
    elif revisions:
        revisions_index = pd.MultiIndex.from_product(
            [endog1.index[-1:], ["realgdp", "unemp"]],
            names=["revision date", "revised variable"])
        # With no updates, the revision is to 2009Q3
        tmp = endog2.iloc[-1].values
        revision_impacts = np.c_[Z @ tmp, T0 @ tmp, T1 @ tmp, T2 @ tmp].T
    else:
        revisions_index = pd.MultiIndex.from_product(
            [[], []],
            names=["revision date", "revised variable"])
        revision_impacts = None

    # Impact forecasts
    if updates:
        tmp = endog1.iloc[-1].values
        prev_impacted_forecasts = np.c_[T0 @ tmp, T1 @ tmp,
                                        T2 @ tmp, T3 @ tmp].T
        tmp = endog2.iloc[-2].values
        rev_impacted_forecasts = np.c_[T0 @ tmp, T1 @ tmp,
                                       T2 @ tmp, T3 @ tmp].T
    else:
        tmp = endog1.iloc[-1].values
        prev_impacted_forecasts = np.c_[
            T0 @ endog1.iloc[-2], T0 @ tmp, T1 @ tmp, T2 @ tmp].T
        tmp = endog2.iloc[-1].values
        rev_impacted_forecasts = np.c_[
            T0 @ endog2.iloc[-2], T0 @ tmp, T1 @ tmp, T2 @ tmp].T
    tmp = endog2.iloc[-1].values
    post_impacted_forecasts = np.c_[
        T0 @ endog2.iloc[-2], T0 @ tmp, T1 @ tmp, T2 @ tmp].T

    # Updates
    if updates:
        updates_index = pd.MultiIndex.from_product(
            [pd.period_range(start="2009Q3", periods=1, freq="Q"),
             ["realgdp", "unemp"]], names=["update date", "updated variable"])
        update_impacts = post_impacted_forecasts - rev_impacted_forecasts
    else:
        updates_index = pd.MultiIndex.from_product(
            [[], []], names=["update date", "updated variable"])
        update_impacts = None

    # News
    if updates:
        # Note: update_forecasts is created using the endog2 dataset even if
        # there were revisions, because it should be computed after revisions
        # have already been taken into account
        update_forecasts = T1 @ endog2.loc["2009Q2"].values
        update_realized = endog2.loc["2009Q3"].values
        news_desired = [update_realized[i] - update_forecasts[i]
                        for i in range(len(update_forecasts))]
        columns = pd.MultiIndex.from_product(
            [impact_dates, impacted_variables],
            names=["impact dates", "impacted variables"])
        weights = pd.DataFrame(np.zeros((2, 8)), index=updates_index,
                               columns=columns)
        weights.loc[:, "2009Q2"] = Z
        weights.loc[:, "2009Q3"] = T0
        weights.loc[:, "2009Q4"] = T1.T
        weights.loc[:, "2010Q1"] = T2.T
    else:
        update_forecasts = pd.Series([], dtype=np.float64)
        update_realized = pd.Series([], dtype=np.float64)
        news_desired = pd.Series([], dtype=np.float64)
        weights = pd.DataFrame(np.zeros((0, 8)))

    # Run unit tests
    check_news(news, revisions, updates, impact_dates, impacted_variables,
               revisions_index, updates_index,
               revision_impacts, update_impacts,
               prev_impacted_forecasts, post_impacted_forecasts,
               update_forecasts, update_realized, news_desired, weights)


@pytest.mark.parametrize("revisions", [True, False])
@pytest.mark.parametrize("updates", [True, False])
@pytest.mark.parametrize("which", ["exog", "trend"])
def test_varmax_time_varying(revisions, updates, which):
    # This is primarily a test that the `news` method works with a time-varying
    # setup (i.e., time-varying state space matrices). It tests a time-varying
    # VARMAX model where the time-varying component has been set to zeros
    # against a time-invariant version of the model.

    # Construct previous and updated datasets
    endog = dta[["realgdp", "unemp"]].copy()
    endog["realgdp"] = np.log(endog["realgdp"]).diff() * 400
    endog = endog.iloc[1:]
    comparison_type = None
    if updates:
        endog1 = endog.loc[:"2009Q2"].copy()
        endog2 = endog.loc[:"2009Q3"].copy()
    else:
        endog1 = endog.loc[:"2009Q3"].copy()
        endog2 = endog.loc[:"2009Q3"].copy()
        # Without updates and without NaN values, we need to specify that
        # the type of the comparison object that we're passing is "updated"
        comparison_type = "updated"
    if revisions:
        # TODO: add test for only one of the variables revising?
        endog1.iloc[-1] = 0.

    exog1 = None
    exog2 = None
    trend = "n"
    if which == "exog":
        exog1 = np.ones_like(endog1["realgdp"])
        exog2 = np.ones_like(endog2["realgdp"])
        params1 = np.r_[0.5, 0.1, 0.2, 0.9, 0., 0., 1., 0.1, 1.1]
        params2 = np.r_[0.5, 0.1, 0.2, 0.9, 1., 0.1, 1.1]
    elif which == "trend":
        trend = "t"
        params1 = np.r_[0., 0., 0.5, 0.1, 0.2, 0.9, 1., 0.1, 1.1]
        params2 = np.r_[0.5, 0.1, 0.2, 0.9, 1., 0.1, 1.1]

    # Compute the news from a model with a trend/exog term (so the model is
    # time-varying), but with the coefficient set to zero (so that it will be
    # equivalent to the time-invariant model)
    mod1 = varmax.VARMAX(endog1, exog=exog1, trend=trend)
    res1 = mod1.smooth(params1)
    news1 = res1.news(endog2, exog=exog2, start="2008Q1", end="2009Q3",
                      comparison_type=comparison_type)

    # Compute the news from a model without a trend term
    mod2 = varmax.VARMAX(endog1, trend="n")
    res2 = mod2.smooth(params2)
    news2 = res2.news(endog2, start="2008Q1", end="2009Q3",
                      comparison_type=comparison_type)

    attrs = ["total_impacts", "update_impacts", "revision_impacts", "news",
             "weights", "update_forecasts", "update_realized",
             "prev_impacted_forecasts", "post_impacted_forecasts",
             "revisions_iloc", "revisions_ix", "updates_iloc", "updates_ix"]

    for attr in attrs:
        w = getattr(news1, attr)
        x = getattr(news2, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)


@pytest.mark.parametrize("revisions", [True, False])
@pytest.mark.parametrize("updates", [True, False])
def test_dynamic_factor_time_varying(revisions, updates):
    # This is primarily a test that the `news` method works with a time-varying
    # setup (i.e., time-varying state space matrices). It tests a time-varying
    # DynamicFactor model where the time-varying component has been set to
    # zeros against a time-invariant version of the model.

    # Construct previous and updated datasets
    endog = dta[["realgdp", "unemp"]].copy()
    endog["realgdp"] = np.log(endog["realgdp"]).diff() * 400
    endog = endog.iloc[1:]
    comparison_type = None
    if updates:
        endog1 = endog.loc[:"2009Q2"].copy()
        endog2 = endog.loc[:"2009Q3"].copy()
    else:
        endog1 = endog.loc[:"2009Q3"].copy()
        endog2 = endog.loc[:"2009Q3"].copy()
        # Without updates and without NaN values, we need to specify that
        # the type of the comparison object that we're passing is "updated"
        comparison_type = "updated"
    if revisions:
        # TODO: add test for only one of the variables revising?
        endog1.iloc[-1] = 0.

    exog1 = np.ones_like(endog1["realgdp"])
    exog2 = np.ones_like(endog2["realgdp"])
    params1 = np.r_[0.9, 0.2, 0.0, 0.0, 1.2, 1.1, 0.5, 0.2]
    params2 = np.r_[0.9, 0.2, 1.2, 1.1, 0.5, 0.2]

    # Compute the news from a model with an exog term (so the model is
    # time-varying), but with the coefficient set to zero (so that it will be
    # equivalent to the time-invariant model)
    mod1 = dynamic_factor.DynamicFactor(endog1, exog=exog1,
                                        k_factors=1, factor_order=2)
    res1 = mod1.smooth(params1)
    news1 = res1.news(endog2, exog=exog2, start="2008Q1", end="2009Q3",
                      comparison_type=comparison_type)

    # Compute the news from a model without a trend term
    mod2 = dynamic_factor.DynamicFactor(endog1, k_factors=1, factor_order=2)
    res2 = mod2.smooth(params2)
    news2 = res2.news(endog2, start="2008Q1", end="2009Q3",
                      comparison_type=comparison_type)

    attrs = ["total_impacts", "update_impacts", "revision_impacts", "news",
             "weights", "update_forecasts", "update_realized",
             "prev_impacted_forecasts", "post_impacted_forecasts",
             "revisions_iloc", "revisions_ix", "updates_iloc", "updates_ix"]

    for attr in attrs:
        w = getattr(news1, attr)
        x = getattr(news2, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
        else:
            assert_frame_equal(w, x)


@pytest.mark.parametrize("revisions", [True, False])
@pytest.mark.parametrize("updates", [True, False])
def test_defaults(revisions, updates):
    # Construct previous and updated datasets
    endog = dta["infl"].copy()
    if updates:
        endog1 = endog.loc[:"2009Q2"].copy()
        endog2 = endog.loc[:"2009Q3"].copy()
    else:
        endog1 = endog.loc[:"2009Q3"].copy()
        endog2 = endog.loc[:"2009Q3"].copy()
    if revisions:
        # TODO: add test for only one of the variables revising?
        endog1.iloc[-1] = 0.

    # Get the previous results object and compute the news
    mod1 = sarimax.SARIMAX(endog1)
    res1 = mod1.smooth([0.5, 1.0])

    mod2 = sarimax.SARIMAX(endog2)
    res2 = mod2.smooth([0.5, 1.0])

    news_updated_data = res1.news(endog2, comparison_type="updated")
    news_previous_data = res2.news(endog1, comparison_type="previous")
    news_updated_results = res1.news(res2, comparison_type="updated")
    news_previous_results = res2.news(res1, comparison_type="previous")

    attrs = ["total_impacts", "update_impacts", "revision_impacts", "news",
             "weights", "update_forecasts", "update_realized",
             "prev_impacted_forecasts", "post_impacted_forecasts",
             "revisions_iloc", "revisions_ix", "updates_iloc", "updates_ix"]

    for attr in attrs:
        w = getattr(news_updated_data, attr)
        x = getattr(news_previous_data, attr)
        y = getattr(news_updated_results, attr)
        z = getattr(news_previous_results, attr)
        if isinstance(x, pd.Series):
            assert_series_equal(w, x)
            assert_series_equal(w, y)
            assert_series_equal(w, z)
        else:
            assert_frame_equal(w, x)
            assert_frame_equal(w, y)
            assert_frame_equal(w, z)


def test_default_impact_date_is_previous_last_observation():
    # The default is the last observation of the previous data, whichever of
    # the previous and the updated results calls `news`
    endog = dta["infl"].copy()
    prev = sarimax.SARIMAX(endog.iloc[:-1]).smooth([0.5, 1.0])
    upd = sarimax.SARIMAX(endog).smooth([0.5, 1.0])
    desired = pd.period_range(start="2009Q2", periods=1, freq="Q")

    for news in (prev.news(upd), upd.news(prev), prev.news(endog)):
        assert_(news.total_impacts.index.equals(desired))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": 10},
        {"end": 10},
        {"periods": 2},
        {"start": 10, "end": 12, "periods": 3},
    ],
)
def test_start_end_periods_exactly_two(kwargs):
    # Of start, end and periods, exactly two must be given if any is
    endog = dta["infl"].copy()
    res = sarimax.SARIMAX(endog.iloc[:-1]).smooth([0.5, 1.0])
    with pytest.raises(ValueError, match="exactly two must be specified"):
        res.news(endog, **kwargs)


@pytest.mark.parametrize("kwargs", [{"start": 10}, {"end": 10}, {"periods": 1}])
def test_impact_date_with_start_end_periods(kwargs):
    endog = dta["infl"].copy()
    res = sarimax.SARIMAX(endog.iloc[:-1]).smooth([0.5, 1.0])
    with pytest.raises(ValueError, match="Cannot use the `impact_date`"):
        res.news(endog, impact_date=10, **kwargs)


def test_comparison_types():
    endog = dta["infl"].copy()
    endog.iloc[-1] = np.nan
    msg = "Could not automatically determine the type of comparison"

    mod = sarimax.SARIMAX(endog)
    res = mod.smooth([0.5, 1.0])

    # Test that the comparison type cannot be determined if the given dataset
    # or results dataset is identical to the previous dataset
    with pytest.raises(ValueError, match=msg):
        res.news(endog)
    with pytest.raises(ValueError, match=msg):
        res.news(res)

    # Test that if the comparison type is specified, the news can be
    # computed from data that has the same shape and number of NaNs (this can
    # happen if there are only revisions but no updates)
    news = res.news(endog, comparison_type="previous")
    assert_allclose(news.total_impacts, 0)
    news = res.news(endog, comparison_type="updated")
    assert_allclose(news.total_impacts, 0)
    news = res.news(res, comparison_type="updated")
    assert_allclose(news.total_impacts, 0)
    news = res.news(res, comparison_type="updated")
    assert_allclose(news.total_impacts, 0)


def test_invalid_comparison_type():
    endog = dta["infl"].copy()
    mod = sarimax.SARIMAX(endog)
    res = mod.smooth([0.5, 1.0])
    with pytest.raises(ValueError, match="comparison_type"):
        res.news(endog, comparison_type="not-a-comparison-type")


@pytest.mark.parametrize("use_periods", [True, False])
def test_start_end_dates(use_periods):
    endog = dta["infl"].copy()
    if use_periods:
        index_range = pd.period_range
    else:
        def index_range(*args, **kwargs):
            return pd.period_range(*args, **kwargs).to_timestamp(freq="Q")
        endog = endog.to_timestamp(freq="Q")
    mod = sarimax.SARIMAX(endog.iloc[:-1])
    res = mod.smooth([0.5, 1.0])

    # Default is the last observation of the previous data
    news = res.news(endog)
    desired = index_range(start="2009Q2", periods=1, freq="Q")
    assert_(news.total_impacts.index.equals(desired))

    # Start (dates), periods
    news = res.news(endog, start="2009Q1", periods=1)
    desired = index_range(start="2009Q1", periods=1, freq="Q")
    assert_(news.total_impacts.index.equals(desired))
    news = res.news(endog, start="2009Q1", periods=2)
    desired = index_range(start="2009Q1", periods=2, freq="Q")
    assert_(news.total_impacts.index.equals(desired))

    # Start (int), periods
    news = res.news(endog, start=mod.nobs - 1, periods=1)
    desired = index_range(start="2009Q2", periods=1, freq="Q")
    assert_(news.total_impacts.index.equals(desired))
    news = res.news(endog, start=mod.nobs - 2, periods=2)
    desired = index_range(start="2009Q1", periods=2, freq="Q")
    assert_(news.total_impacts.index.equals(desired))

    # End (dates), periods
    news = res.news(endog, end="2009Q1", periods=1)
    desired = index_range(end="2009Q1", periods=1, freq="Q")
    assert_(news.total_impacts.index.equals(desired))
    news = res.news(endog, end="2009Q1", periods=2)
    desired = index_range(end="2009Q1", periods=2, freq="Q")
    assert_(news.total_impacts.index.equals(desired))

    # End (int), periods
    news = res.news(endog, end=mod.nobs - 1, periods=1)
    desired = index_range(end="2009Q2", periods=1, freq="Q")
    assert_(news.total_impacts.index.equals(desired))
    news = res.news(endog, end=mod.nobs - 2, periods=2)
    desired = index_range(end="2009Q1", periods=2, freq="Q")
    assert_(news.total_impacts.index.equals(desired))

    # Start (dates), end (dates)
    news = res.news(endog, start="2009Q1", end="2009Q1")
    desired = index_range(start="2009Q1", end="2009Q1", freq="Q")
    assert_(news.total_impacts.index.equals(desired))
    news = res.news(endog, start="2009Q1", end="2009Q2")
    desired = index_range(start="2009Q1", end="2009Q2", freq="Q")
    assert_(news.total_impacts.index.equals(desired))

    # Start (dates), end (int)
    news = res.news(endog, start="2009Q1", end=mod.nobs - 2)
    desired = index_range(start="2009Q1", end="2009Q1", freq="Q")
    assert_(news.total_impacts.index.equals(desired))
    predicted = res.predict(start="2009Q1", end=mod.nobs - 2)
    assert_(news.total_impacts.index.equals(predicted.index))
    news = res.news(endog, start="2009Q1", end=mod.nobs - 1)
    desired = index_range(start="2009Q1", end="2009Q2", freq="Q")
    assert_(news.total_impacts.index.equals(desired))
    predicted = res.predict(start="2009Q1", end=mod.nobs - 1)
    assert_(news.total_impacts.index.equals(predicted.index))

    # Start (int), end (dates)
    news = res.news(endog, start=mod.nobs - 2, end="2009Q1")
    desired = index_range(start="2009Q1", end="2009Q1", freq="Q")
    assert_(news.total_impacts.index.equals(desired))
    predicted = res.predict(start=mod.nobs - 2, end="2009Q1")
    assert_(news.total_impacts.index.equals(predicted.index))
    news = res.news(endog, start=mod.nobs - 2, end="2009Q2")
    desired = index_range(start="2009Q1", end="2009Q2", freq="Q")
    assert_(news.total_impacts.index.equals(desired))
    predicted = res.predict(start=mod.nobs - 2, end="2009Q2")
    assert_(news.total_impacts.index.equals(predicted.index))

    # Negative indexes
    # Note that negative indexes are always computed relative to the updated
    # sample, which in this case is 1 observation more than is in `mod.nobs`
    total_nobs = len(endog)
    assert_equal(total_nobs, mod.nobs + 1)

    # Start (dates), end (int)
    desired = index_range(start="2009Q1", end="2009Q1", freq="Q")
    for end in [mod.nobs - 2, total_nobs - 3, -3]:
        news = res.news(endog, start="2009Q1", end=end)
        assert_(news.total_impacts.index.equals(desired))
        # Note: predict does not allow negative indexing
        if end > 0:
            predicted = res.predict(start="2009Q1", end=end)
            assert_(news.total_impacts.index.equals(predicted.index))

    # Start (int), end (dates)
    desired = index_range(start="2009Q1", end="2009Q1", freq="Q")
    for start in [mod.nobs - 2, total_nobs - 3, -3]:
        news = res.news(endog, start=start, end="2009Q1")
        assert_(news.total_impacts.index.equals(desired))
        # Note: predict does not allow negative indexing
        if end > 0:
            predicted = res.predict(start=start, end="2009Q1")
            assert_(news.total_impacts.index.equals(predicted.index))


@pytest.mark.parametrize("which", ["range", "range2", "int64",
                                   "numpy", "list"])
def test_start_end_int(which):
    endog = dta["infl"].copy()
    nobs = len(endog)
    if which == "range":
        endog.index = pd.RangeIndex(nobs)
        endog_init = endog.iloc[:-1]
        index_plus2 = pd.RangeIndex(nobs + 2)
    if which == "range2":
        endog.index = pd.RangeIndex(stop=nobs * 2, step=2)
        endog_init = endog.iloc[:-1]
        index_plus2 = pd.RangeIndex((nobs + 2) * 2, step=2)
    elif which == "int64":
        endog.index = pd.Index(np.arange(nobs))
        endog_init = endog.iloc[:-1]
        index_plus2 = pd.Index(np.arange(nobs + 2))
    elif which == "numpy":
        endog = endog.values
        endog_init = endog[:-1]
        index_plus2 = pd.RangeIndex(nobs + 2)
    elif which == "list":
        endog = endog.tolist()
        endog_init = endog[:-1]
        index_plus2 = pd.RangeIndex(nobs + 2)

    mod = sarimax.SARIMAX(endog_init)
    res = mod.smooth([0.5, 1.0])

    # Default is the last in-sample period
    news = res.news(endog)
    desired = index_plus2[-4:-3]
    assert_(news.total_impacts.index.equals(desired))

    # Start, periods
    news = res.news(endog, start=mod.nobs - 1, periods=1)
    desired = index_plus2[-4:-3]
    assert_(news.total_impacts.index.equals(desired))
    news = res.news(endog, start=mod.nobs - 2, periods=2)
    desired = index_plus2[-5:-3]
    assert_(news.total_impacts.index.equals(desired))

    # End, periods
    news = res.news(endog, end=mod.nobs - 1, periods=1)
    desired = index_plus2[-4:-3]
    assert_(news.total_impacts.index.equals(desired))
    news = res.news(endog, end=mod.nobs - 2, periods=2)
    desired = index_plus2[-6:-4]
    assert_(news.total_impacts.index.equals(desired))

    # Start, end
    # Note: end is inclusive, like `get_prediction`.
    news = res.news(endog, start=mod.nobs - 2, end=mod.nobs - 1)
    desired = index_plus2[-5:-3]
    assert_(news.total_impacts.index.equals(desired))
    if which not in ["numpy", "list"]:
        predicted = res.predict(start=mod.nobs - 2, end=mod.nobs - 1)
        assert_(news.total_impacts.index.equals(predicted.index))

    news = res.news(endog, start=mod.nobs, end=mod.nobs)
    desired = index_plus2[-3:-2]
    assert_(news.total_impacts.index.equals(desired))
    if which not in ["numpy", "list"]:
        predicted = res.predict(start=mod.nobs, end=mod.nobs)
        assert_(news.total_impacts.index.equals(predicted.index))

    news = res.news(endog, start=mod.nobs, end=mod.nobs + 1)
    desired = index_plus2[-3:-1]
    assert_(news.total_impacts.index.equals(desired))
    if which not in ["numpy", "list"]:
        predicted = res.predict(start=mod.nobs, end=mod.nobs + 1)
        assert_(news.total_impacts.index.equals(predicted.index))


def test_invalid():
    endog = dta["infl"].copy()
    mod = sarimax.SARIMAX(endog.iloc[:-1])
    res = mod.smooth([0.5, 1.0])

    msg = "The index associated with the updated results is not a superset"
    with pytest.raises(ValueError, match=msg):
        res.news(endog.values)


@pytest.mark.parametrize("revisions_details_start", [True, -10, 200])
def test_detailed_revisions(revisions_details_start):
    # Construct original and revised datasets
    y = np.log(dta[["realgdp", "realcons",
                    "realinv", "cpi"]]).diff().iloc[1:] * 100
    y.iloc[-1, 0] = np.nan

    y_revised = y.copy()
    revisions = {
        ("2009Q2", "realgdp"): 1.1,
        ("2009Q3", "realcons"): 0.5,
        ("2009Q2", "realinv"): -0.3,
        ("2009Q2", "cpi"): 0.2,
        ("2009Q3", "cpi"): 0.2,
    }
    for key, diff in revisions.items():
        y_revised.loc[key] += diff

    # Create model and results
    mod = varmax.VARMAX(y, trend="n")
    ar_coeff = {
        "realgdp": 0.9,
        "realcons": 0.8,
        "realinv": 0.7,
        "cpi": 0.6
    }
    params = np.r_[np.diag(list(ar_coeff.values())).flatten(),
                   [1, 0, 1, 0, 0, 1, 0, 0, 0, 1]]
    res = mod.smooth(params)
    res_revised = res.apply(y_revised)
    news = res_revised.news(res, comparison_type="previous", tolerance=-1,
                            revisions_details_start=revisions_details_start)

    # Tests
    data_revisions = news.data_revisions
    revision_details = news.revision_details_by_update.reset_index([2, 3])

    for key, diff in revisions.items():
        assert_allclose(data_revisions.loc[key, "revised"], y_revised.loc[key])
        assert_allclose(data_revisions.loc[key, "observed (prev)"], y.loc[key])
        assert_equal(
            # Need to manually cast to numpy for compatibility with
            # pandas==1.2.5
            np.array(data_revisions.loc[key, "detailed impacts computed"]),
            True)
        assert_allclose(revision_details.loc[key, "revised"],
                        y_revised.loc[key])
        assert_allclose(revision_details.loc[key, "observed (prev)"],
                        y.loc[key])
        assert_allclose(revision_details.loc[key, "revision"], diff)

    # For revisions to the impact period, the own-weight is equal to 1.
    key = ("2009Q3", "realcons", "2009Q3", "realcons")
    assert_allclose(revision_details.loc[key, "weight"], 1)
    assert_allclose(revision_details.loc[key, "impact"],
                    revisions[("2009Q3", "realcons")])
    key = ("2009Q3", "cpi", "2009Q3", "cpi")
    assert_allclose(revision_details.loc[key, "weight"], 1)
    assert_allclose(revision_details.loc[key, "impact"],
                    revisions[("2009Q3", "cpi")])

    # For revisions just before the impact period, all weights are equal to
    # zero unless there is a missing value in the impact period, in which case
    # the weight is equal to the AR coefficient
    key = ("2009Q2", "realgdp", "2009Q3", "realgdp")
    assert_allclose(revision_details.loc[key, "weight"], ar_coeff["realgdp"])
    assert_allclose(revision_details.loc[key, "impact"],
                    ar_coeff["realgdp"] * revisions[("2009Q2", "realgdp")])
    key = ("2009Q2", "realinv", "2009Q3", "realinv")
    assert_allclose(revision_details.loc[key, "weight"], 0)
    assert_allclose(revision_details.loc[key, "impact"], 0)
    key = ("2009Q2", "cpi", "2009Q3", "cpi")
    assert_allclose(revision_details.loc[key, "weight"], 0)
    assert_allclose(revision_details.loc[key, "impact"], 0)

    # Check the impacts table

    # Since we only have revisions, all impacts are due to revisions
    assert_allclose(news.impacts["impact of news"], 0)
    assert_allclose(news.impacts["total impact"],
                    news.impacts["impact of revisions"])

    # Check the values for estimates
    for name in ["cpi", "realcons", "realinv"]:
        assert_allclose(
            news.impacts.loc[("2009Q3", name), "estimate (new)"],
            y_revised.loc["2009Q3", name])
        assert_allclose(
            news.impacts.loc[("2009Q3", name), "estimate (prev)"],
            y.loc["2009Q3", name])
    # Have to handle real GDP separately since the 2009Q3 value is missing
    name = "realgdp"
    assert_allclose(
        news.impacts.loc[("2009Q3", name), "estimate (new)"],
        y_revised.loc["2009Q2", name] * ar_coeff[name])
    assert_allclose(
        news.impacts.loc[("2009Q3", name), "estimate (prev)"],
        y.loc["2009Q2", name] * ar_coeff[name])

    # Check that the values of revision impacts sum up correctly
    assert_allclose(
        news.impacts["impact of revisions"],
        revision_details.groupby(level=[2, 3]).sum()["impact"]
    )


@pytest.mark.parametrize("revisions_details_start", [False, 202])
def test_grouped_revisions(revisions_details_start):
    # Tests for computing revision impacts when all revisions are grouped
    # together (i.e., no detailed impacts are computed )
    # Construct original and revised datasets
    y = np.log(dta[["realgdp", "realcons",
                    "realinv", "cpi"]]).diff().iloc[1:] * 100
    y.iloc[-1, 0] = np.nan

    y_revised = y.copy()
    revisions = {
        ("2009Q2", "realgdp"): 1.1,
        ("2009Q3", "realcons"): 0.5,
        ("2009Q2", "realinv"): -0.3,
        ("2009Q2", "cpi"): 0.2,
        ("2009Q3", "cpi"): 0.2,
    }
    for key, diff in revisions.items():
        y_revised.loc[key] += diff

    # Create model and results
    mod = varmax.VARMAX(y, trend="n")
    ar_coeff = {
        "realgdp": 0.9,
        "realcons": 0.8,
        "realinv": 0.7,
        "cpi": 0.6
    }
    params = np.r_[np.diag(list(ar_coeff.values())).flatten(),
                   [1, 0, 1, 0, 0, 1, 0, 0, 0, 1]]
    res = mod.smooth(params)
    res_revised = res.apply(y_revised)
    news = res_revised.news(res, comparison_type="previous", tolerance=-1,
                            revisions_details_start=revisions_details_start)

    # Tests
    data_revisions = news.data_revisions
    revision_details = news.revision_details_by_update.reset_index([2, 3])

    for key in revisions.keys():
        assert_allclose(data_revisions.loc[key, "revised"], y_revised.loc[key])
        assert_allclose(data_revisions.loc[key, "observed (prev)"], y.loc[key])
        assert_equal(
            # Need to manually cast to numpy for compatibility with
            # pandas==1.2.5
            np.array(data_revisions.loc[key, "detailed impacts computed"]),
            False)

    # For grouped data, should not have any of revised, observed (prev),
    # revision, weight
    key = ("2009Q3", "all prior revisions", "2009Q3")
    cols = ["revised", "observed (prev)", "revision", "weight"]
    assert np.all(revision_details.loc[key, cols].isna())

    # Expected grouped impacts are the sum of the detailed impacts from
    # `test_detailed_revisions`
    assert_allclose(revision_details.loc[key + ("realgdp",), "impact"],
                    ar_coeff["realgdp"] * revisions[("2009Q2", "realgdp")])
    assert_allclose(revision_details.loc[key + ("realcons",), "impact"],
                    revisions[("2009Q3", "realcons")])
    assert_allclose(revision_details.loc[key + ("realinv",), "impact"], 0)
    assert_allclose(revision_details.loc[key + ("cpi",), "impact"],
                    revisions[("2009Q3", "cpi")])

    # Check the values for estimates
    for name in ["cpi", "realcons", "realinv"]:
        assert_allclose(
            news.impacts.loc[("2009Q3", name), "estimate (new)"],
            y_revised.loc["2009Q3", name])
        assert_allclose(
            news.impacts.loc[("2009Q3", name), "estimate (prev)"],
            y.loc["2009Q3", name])
    # Have to handle real GDP separately since the 2009Q3 value is missing
    name = "realgdp"
    assert_allclose(
        news.impacts.loc[("2009Q3", name), "estimate (new)"],
        y_revised.loc["2009Q2", name] * ar_coeff[name])
    assert_allclose(
        news.impacts.loc[("2009Q3", name), "estimate (prev)"],
        y.loc["2009Q2", name] * ar_coeff[name])

    # Check that the values of revision impacts sum up correctly
    assert_allclose(
        news.impacts["impact of revisions"],
        revision_details.groupby(level=[2, 3]).sum()["impact"]
    )


@pytest.mark.parametrize("revisions_details_start", [-1, 201])
def test_mixed_revisions(revisions_details_start):
    # Construct original and revised datasets
    y = np.log(dta[["realgdp", "realcons",
                    "realinv", "cpi"]]).diff().iloc[1:] * 100
    y.iloc[-1, 0] = np.nan

    y_revised = y.copy()
    revisions = {
        ("2009Q2", "realgdp"): 1.1,
        ("2009Q3", "realcons"): 0.5,
        ("2009Q2", "realinv"): -0.3,
        ("2009Q2", "cpi"): 0.2,
        ("2009Q3", "cpi"): 0.2,
    }
    for key, diff in revisions.items():
        y_revised.loc[key] += diff

    # Create model and results
    mod = varmax.VARMAX(y, trend="n")
    ar_coeff = {
        "realgdp": 0.9,
        "realcons": 0.8,
        "realinv": 0.7,
        "cpi": 0.6
    }
    params = np.r_[np.diag(list(ar_coeff.values())).flatten(),
                   [1, 0, 1, 0, 0, 1, 0, 0, 0, 1]]
    res = mod.smooth(params)
    res_revised = res.apply(y_revised)
    news = res_revised.news(res, comparison_type="previous", tolerance=-1,
                            revisions_details_start=revisions_details_start)

    # Tests
    data_revisions = news.data_revisions
    revision_details = news.revision_details_by_update.reset_index([2, 3])

    for key in revisions.keys():
        assert_allclose(data_revisions.loc[key, "revised"], y_revised.loc[key])
        assert_allclose(data_revisions.loc[key, "observed (prev)"], y.loc[key])
        # Revisions to 2009Q2 are grouped (i.e., no details are computed),
        # while revisions to 2009Q3 have detailed impacts computed
        expected_details_computed = key[0] == "2009Q3"
        assert_equal(
            # Need to manually cast to numpy for compatibility with
            # pandas==1.2.5
            np.array(data_revisions.loc[key, "detailed impacts computed"]),
            expected_details_computed)

    # For grouped data, should not have any of revised, observed (prev),
    # revision, weight
    key = ("2009Q2", "all prior revisions", "2009Q3")
    cols = ["revised", "observed (prev)", "revision", "weight"]
    assert np.all(revision_details.loc[key, cols].isna())

    # Expected grouped impacts are the sum of the detailed impacts from
    # `test_detailed_revisions` for revisions to 2009Q2
    assert_allclose(revision_details.loc[key + ("realgdp",), "impact"],
                    ar_coeff["realgdp"] * revisions[("2009Q2", "realgdp")])
    assert_allclose(revision_details.loc[key + ("realinv",), "impact"], 0)

    # Expected detailed impacts are for revisions to 2009Q3
    # (for revisions to the impact period, the own-weight is equal to 1)
    key = ("2009Q3", "realcons", "2009Q3", "realcons")
    assert_allclose(revision_details.loc[key, "weight"], 1)
    assert_allclose(revision_details.loc[key, "impact"],
                    revisions[("2009Q3", "realcons")])
    key = ("2009Q3", "cpi", "2009Q3", "cpi")
    assert_allclose(revision_details.loc[key, "weight"], 1)
    assert_allclose(revision_details.loc[key, "impact"],
                    revisions[("2009Q3", "cpi")])

    # Check the values for estimates
    for name in ["cpi", "realcons", "realinv"]:
        assert_allclose(
            news.impacts.loc[("2009Q3", name), "estimate (new)"],
            y_revised.loc["2009Q3", name])
        assert_allclose(
            news.impacts.loc[("2009Q3", name), "estimate (prev)"],
            y.loc["2009Q3", name])
    # Have to handle real GDP separately since the 2009Q3 value is missing
    name = "realgdp"
    assert_allclose(
        news.impacts.loc[("2009Q3", name), "estimate (new)"],
        y_revised.loc["2009Q2", name] * ar_coeff[name])
    assert_allclose(
        news.impacts.loc[("2009Q3", name), "estimate (prev)"],
        y.loc["2009Q2", name] * ar_coeff[name])

    # Check that the values of revision impacts sum up correctly
    assert_allclose(
        news.impacts["impact of revisions"],
        revision_details.groupby(level=[2, 3]).sum()["impact"]
    )


def test_news_summary_methods():
    # Regression test: summary_revisions() used a block `.iloc[:, 2:-1] =`
    # assignment to write formatted strings back into a float-dtype slice,
    # which pandas >= 2.2 rejects as a lossy set. It had no test coverage at
    # all (unlike summary_news, which already used the safe per-column
    # assignment pattern this was fixed to match), so the break went
    # unnoticed. This test exercises every summary/detail method on
    # NewsResults and checks their content against independently computable
    # values, not just that they run without raising.
    y = np.log(dta[["realgdp", "realcons",
                    "realinv", "cpi"]]).diff().iloc[1:] * 100
    y.iloc[-1, 0] = np.nan

    y_revised = y.copy()
    revisions = {
        ("2009Q2", "realgdp"): 1.1,
        ("2009Q3", "realcons"): 0.5,
        ("2009Q2", "realinv"): -0.3,
        ("2009Q2", "cpi"): 0.2,
        ("2009Q3", "cpi"): 0.2,
    }
    for key, diff in revisions.items():
        y_revised.loc[key] += diff

    mod = varmax.VARMAX(y, trend="n")
    ar_coeff = {"realgdp": 0.9, "realcons": 0.8, "realinv": 0.7, "cpi": 0.6}
    params = np.r_[np.diag(list(ar_coeff.values())).flatten(),
                   [1, 0, 1, 0, 0, 1, 0, 0, 0, 1]]
    res = mod.smooth(params)
    res_revised = res.apply(y_revised)
    news = res_revised.news(res, comparison_type="previous", tolerance=-1)

    # get_impacts (no grouping) must sum to the same total as total_impacts,
    # which is independently verified elsewhere in this module.
    assert_allclose(
        news.get_impacts().to_numpy().sum(),
        news.total_impacts.to_numpy().sum(),
    )

    # get_details combines news + revision details; row count is exactly
    # the sum of the two source tables it concatenates.
    details = news.get_details()
    assert_equal(
        len(details),
        len(news.data_updates) + len(news.revision_details_by_impact),
    )

    # summary_revisions: check the rendered table reproduces the known
    # revision amounts and previous values, not just that it doesn't raise.
    revisions_table = news.summary_revisions()
    rendered = str(revisions_table)
    for (date, variable), diff in revisions.items():
        assert_(f"{diff:.2f}" in rendered)
        prev_value = y.loc[date, variable]
        assert_(f"{prev_value:.2f}" in rendered)

    # summary_news: unchanged from before (already covered), no updates in
    # this fixture so nothing further to check content-wise here
    assert_(str(news.summary_news()).strip() != "")
    assert_equal(len(news.data_updates), 0)

    # summary_impacts: for every impacted variable, "estimate (new)" must
    # equal "estimate (prev)" + "total impact" -- and "total impact" must
    # match total_impacts, already independently verified above via
    # get_impacts()/total_impacts.
    impacts_text = str(news.summary_impacts())
    for variable in news.total_impacts.columns:
        total_impact = news.total_impacts.loc["2009Q3", variable]
        prev = news.prev_impacted_forecasts.loc["2009Q3", variable]
        new = news.post_impacted_forecasts.loc["2009Q3", variable]
        assert_allclose(new - prev, total_impact, atol=1e-10)
        assert_(f"{total_impact:.2f}" in impacts_text)
        assert_(f"{prev:.2f}" in impacts_text)
        assert_(f"{new:.2f}" in impacts_text)

    # impacted_variable= must narrow the table to just that variable's row
    cpi_text = str(news.summary_impacts(impacted_variable="cpi"))
    assert_(f'{news.total_impacts.loc["2009Q3", "cpi"]:.2f}' in cpi_text)
    for variable in ("realgdp", "realcons", "realinv"):
        assert_(variable not in cpi_text)

    # summary_details(source="revisions"): grouping by impacted variable
    # must reproduce the same total-impact values as summary_impacts,
    # since this dataset has revisions only (no updates)
    details_text = str(news.summary_details(source="revisions"))
    for variable in news.total_impacts.columns:
        total_impact = news.total_impacts.loc["2009Q3", variable]
        assert_(f"{total_impact:.2f}" in details_text)

    # summary(): combines the header metadata with the same impacts table
    full_text = str(news.summary())
    assert_("# of revisions" in full_text)
    assert_("# of new datapoints" in full_text)
    # 5 raw revisions, 0 new datapoints, matching this fixture's setup
    header = full_text.split("Impacts", maxsplit=1)[0]
    assert_(str(len(revisions)) in header)
    assert_(str(len(news.data_updates)) in header)
    for variable in news.total_impacts.columns:
        total_impact = news.total_impacts.loc["2009Q3", variable]
        assert_(f"{total_impact:.2f}" in full_text)

    # summary_details' source parameter: "news" is the default (already
    # exercised above); "revisions" is a separate code path.
    assert_(str(news.summary_details(source="revisions")).strip() != "")
    with pytest.raises(ValueError, match="source"):
        news.summary_details(source="invalid")

    # revision_details_by_impact aggregates by (impact date, impacted
    # variable): the two cpi revisions (2009Q2 and 2009Q3) both land on the
    # same impact date/variable and are combined into one "all prior
    # revisions" row, so there are 4 rows (one per impacted variable) rather
    # than 5 (one per raw revision). Since there are no updates in this
    # comparison, summing by impacted variable must reproduce total_impacts
    # exactly.
    rev_by_impact = news.revision_details_by_impact
    assert_equal(len(rev_by_impact), 4)
    by_var = rev_by_impact.reset_index().groupby("impacted variable")["impact"].sum()
    assert_allclose(
        by_var.sort_index(), news.total_impacts.loc["2009Q3"].sort_index()
    )


# - Tables, summaries and options of NewsResults ------------------------------


def univariate_news(revision=True, name="infl", **kwargs):
    # AR(1) with phi = 0.5 and sigma2 = 1. The previous dataset ends in 2009Q2,
    # where its value is 0 if `revision`; the updated dataset adds 2009Q3.
    endog = dta["infl"].rename(name)
    previous = endog.loc[:"2009Q2"].copy()
    if revision:
        previous.iloc[-1] = 0.0
    res = sarimax.SARIMAX(previous).smooth([0.5, 1.0])
    kwargs.setdefault("revisions_details_start", True)
    return res.news(endog, start="2009Q2", end="2009Q4", **kwargs)


def bivariate_news(update=None, **kwargs):
    # The VAR(1) of test_varmax_time_invariant. Both 2009Q2 values are revised
    # from 0 and both 2009Q3 values are new (equal to `update` if given).
    endog = dta[["realgdp", "unemp"]].copy()
    endog["realgdp"] = np.log(endog["realgdp"]).diff() * 400
    endog = endog.iloc[1:]
    if update is not None:
        endog.loc["2009Q3"] = update
    previous = endog.loc[:"2009Q2"].copy()
    previous.iloc[-1] = 0.0
    res = varmax.VARMAX(previous, trend="n").smooth(
        np.r_[0.5, 0.1, 0.2, 0.9, 1.0, 0.1, 1.1])
    kwargs.setdefault("revisions_details_start", True)
    if "impact_date" not in kwargs:
        kwargs.update(start="2009Q2", end="2010Q1")
    return res.news(endog, **kwargs)


def table_values(table):
    # Header and rows of a SimpleTable as lists of str
    return [[str(cell) for cell in row] for row in table.data]


def frame_rows(frame):
    # (label, row) pairs of a DataFrame, as DataFrame.iterrows gives them;
    # pandas 3.2 deprecates iterrows
    return ((frame.index[i], frame.iloc[i]) for i in range(len(frame)))


def sparsified(values):
    # Blank each value that repeats the value of the previous row
    return [str(value) if i == 0 or value != values[i - 1] else ""
            for i, value in enumerate(values)]


def test_univariate_details_closed_form():
    # AR(1) with phi = 0.5: the revision r of 2009Q2 has weights 1, phi and
    # phi**2 on 2009Q2-2009Q4; the news n of 2009Q3 (its forecast error given
    # the revised data) has weights 0, 1 and phi.
    news = univariate_news()
    y = dta["infl"]
    r = y.loc["2009Q2"]
    n = y.loc["2009Q3"] - 0.5 * y.loc["2009Q2"]
    dates = pd.period_range(start="2009Q2", end="2009Q4", freq="Q")

    details = news.revision_details_by_impact
    assert_equal(details.index.names, ["impact date", "impacted variable",
                                       "revision date", "revised variable"])
    assert_(details.index.get_level_values(0).equals(dates))
    assert_allclose(details["weight"], [1, 0.5, 0.25], rtol=1e-10)
    assert_allclose(details["revision"], r, rtol=1e-12)
    assert_allclose(details["revised"], r, rtol=1e-12)
    assert_allclose(details["observed (prev)"], 0)
    assert_allclose(details["impact"], r * np.array([1, 0.5, 0.25]),
                    rtol=1e-10)

    # The news has no impact on 2009Q2, which is dropped by the tolerance
    details = news.details_by_impact
    assert_(details.index.get_level_values(0).equals(dates[1:]))
    assert_allclose(details["news"], n, rtol=1e-10)
    assert_allclose(details["forecast (prev)"], 0.5 * r, rtol=1e-10)
    assert_allclose(details["weight"], [1, 0.5], rtol=1e-10)
    assert_allclose(details["impact"], n * np.array([1, 0.5]), rtol=1e-10)

    # The estimates: the previous dataset gives 0 for every date; the updated
    # dataset gives y for 2009Q2-2009Q3 and the forecast phi * y for 2009Q4.
    # (atol here and below for values that are zero up to rounding)
    impacts = news.impacts
    assert_allclose(impacts["estimate (prev)"], 0, atol=1e-12)
    assert_allclose(impacts["impact of revisions"],
                    r * np.array([1, 0.5, 0.25]), rtol=1e-10)
    assert_allclose(impacts["impact of news"], n * np.array([0, 1, 0.5]),
                    rtol=1e-10, atol=1e-12)
    desired = [y.loc["2009Q2"], y.loc["2009Q3"], 0.5 * y.loc["2009Q3"]]
    assert_allclose(impacts["estimate (new)"], desired, rtol=1e-10)
    assert_allclose(impacts["total impact"],
                    impacts["estimate (new)"] - impacts["estimate (prev)"],
                    rtol=1e-10)


@pytest.mark.parametrize("revisions_details_start", [True, False])
def test_details_by_impact_sum_to_impacts(revisions_details_start):
    # Each details table summed over the revisions or updates gives the impacts
    news = bivariate_news(revisions_details_start=revisions_details_start)
    news.tolerance = -1
    by_impact = news.revision_details_by_impact["impact"]
    by_update = news.revision_details_by_update["impact"]
    if revisions_details_start:
        assert_equal(news.n_revisions_grouped, 0)
        # weight times revision
        details = news.revision_details_by_impact
        assert_allclose(details["impact"],
                        details["weight"] * details["revision"], rtol=1e-12)
    expected = news.revision_impacts.stack(**FUTURE_STACK)
    assert_allclose(by_impact.groupby(level=[0, 1]).sum(), expected,
                    rtol=1e-10)
    assert_allclose(by_update.groupby(level=[4, 5]).sum(), expected,
                    rtol=1e-10)
    by_impact = news.details_by_impact["impact"]
    by_update = news.details_by_update["impact"]
    expected = news.update_impacts.stack(**FUTURE_STACK)
    # atol for the impacts on 2009Q2, which are zero up to rounding
    assert_allclose(by_impact.groupby(level=[0, 1]).sum(), expected,
                    rtol=1e-10, atol=1e-12)
    assert_allclose(by_update.groupby(level=[4, 5]).sum(), expected,
                    rtol=1e-10, atol=1e-12)
    # The impacts of revisions and news add up to the change in the estimates
    assert_allclose(
        news.revision_impacts + news.update_impacts,
        news.post_impacted_forecasts - news.prev_impacted_forecasts,
        rtol=1e-10)


@pytest.mark.parametrize(
    "value", ["unemp", ["unemp"], np.array(["unemp"]), slice("unemp", None)])
@pytest.mark.parametrize("setter", [True, False])
def test_impacted_variable(value, setter):
    full = bivariate_news()
    if setter:
        news = bivariate_news()
        assert_(news.impacted_variable is None)
        news.impacted_variable = value
    else:
        news = bivariate_news(impacted_variable=value)
    assert_(news.impacted_variable is value)
    for attr, level in [("details_by_impact", 1),
                        ("revision_details_by_impact", 1),
                        ("details_by_update", 5),
                        ("revision_details_by_update", 5),
                        ("impacts", 1)]:
        desired = getattr(full, attr)
        desired = desired[desired.index.get_level_values(level) == "unemp"]
        assert_(len(desired) > 0)
        actual = getattr(news, attr)
        assert_frame_equal(actual, desired)


def test_tolerance():
    news = bivariate_news(tolerance=-1)
    full = {attr: getattr(news, attr) for attr in [
        "details_by_impact", "revision_details_by_impact",
        "details_by_update", "revision_details_by_update", "impacts"]}
    news.tolerance = 1.0
    assert_allclose(news.tolerance, 1.0)
    for attr, desired in full.items():
        if attr == "impacts":
            cols = ["impact of revisions", "impact of news"]
            mask = (np.abs(desired[cols]) > 1.0).any(axis=1)
        else:
            mask = np.abs(desired["impact"]) > 1.0
        # The tolerance drops some but not all rows
        assert_(0 < mask.sum() < len(mask))
        assert_frame_equal(getattr(news, attr), desired[mask])


IMPACTS_COLUMNS = ["estimate (prev)", "impact of revisions", "impact of news",
                   "total impact", "estimate (new)"]


@pytest.mark.parametrize("revision", [True, False])
@pytest.mark.parametrize("show_revisions_columns", [None, True, False])
def test_summary_impacts_univariate(revision, show_revisions_columns):
    news = univariate_news(revision=revision)
    table = news.summary_impacts(
        show_revisions_columns=show_revisions_columns, float_format="%.4f")
    # The impacted variable is a single one, shown in the title
    assert_("[impacted variable = infl]" in table.title)
    columns = IMPACTS_COLUMNS
    if not (show_revisions_columns or
            (show_revisions_columns is None and revision)):
        columns = ["estimate (prev)", "impact of news", "estimate (new)"]
    values = table_values(table)
    assert_equal(values[0], ["impact date"] + columns)
    impacts = news.impacts.droplevel(1)
    assert_equal([row[0] for row in values[1:]],
                 [str(date) for date in impacts.index])
    for row, (_, desired) in zip(values[1:], frame_rows(impacts), strict=True):
        assert_equal(row[1:], [f"{value:.4f}" for value in desired[columns]])


@pytest.mark.parametrize("sparsify", [True, False])
def test_summary_impacts_by_impacted_variable(sparsify):
    news = bivariate_news()
    table = news.summary_impacts(groupby="impacted variable",
                                 sparsify=sparsify, float_format="%.4f")
    assert_equal(table.title, "Impacts")
    values = table_values(table)
    assert_equal(values[0], ["impacted variable", "impact date"]
                 + IMPACTS_COLUMNS)
    impacts = news.impacts.swaplevel().sort_index()
    variables = impacts.index.get_level_values(0).tolist()
    assert_equal([row[0] for row in values[1:]],
                 sparsified(variables) if sparsify else variables)
    assert_equal([row[1] for row in values[1:]],
                 [str(date) for date in impacts.index.get_level_values(1)])
    for row, (_, desired) in zip(values[1:], frame_rows(impacts), strict=True):
        assert_equal(row[2:], [f"{value:.4f}" for value in desired])


def test_summary_impacts_selection():
    news = bivariate_news()
    impacts = news.impacts
    # A single impact date, grouped by impacted variable: the date is in the
    # title and its column is dropped
    table = news.summary_impacts(impact_date="2009Q3",
                                 groupby="impacted variable")
    assert_("[impact date = 2009Q3]" in table.title)
    values = table_values(table)
    assert_equal(values[0], ["impacted variable"] + IMPACTS_COLUMNS)
    desired = impacts.loc["2009Q3"]
    assert_equal([row[0] for row in values[1:]], desired.index.tolist())
    for row, (_, value) in zip(values[1:], frame_rows(desired), strict=True):
        assert_equal(row[1:], [f"{x:.2f}" for x in value])
    # A range of impact dates and a single impacted variable, without
    # sparsifying: every key is shown
    table = news.summary_impacts(impact_date=slice("2009Q4", "2010Q1"),
                                 impacted_variable=["unemp"], sparsify=False)
    values = table_values(table)
    assert_equal(values[0], ["impact date", "impacted variable"]
                 + IMPACTS_COLUMNS)
    desired = impacts.loc[(slice("2009Q4", "2010Q1"), "unemp"), :]
    assert_equal(len(values) - 1, len(desired))
    for row, (key, value) in zip(values[1:], frame_rows(desired), strict=True):
        assert_equal(row[:2], [str(key[0]), key[1]])
        assert_equal(row[2:], [f"{x:.2f}" for x in value])

    with pytest.raises(ValueError, match="Invalid groupby for impacts table"):
        news.summary_impacts(groupby="update date")


def test_summary_details_univariate_closed_form():
    # The weights of test_univariate_details_closed_form: the weight 1 of a
    # revision or update on its own date is shown as "1.0".
    news = univariate_news()
    y = dta["infl"]
    r = y.loc["2009Q2"]
    n = y.loc["2009Q3"] - 0.5 * y.loc["2009Q2"]

    def f(x):
        return f"{x:.4f}"

    table = news.summary_details(float_format="%.4f")
    assert_equal(table.title, "Details of news for [updated variable = "
                              "infl, impacted variable = infl]")
    assert_equal(table_values(table), [
        ["update date", "observed", "forecast (prev)", "impact date", "news",
         "weight", "impact"],
        ["2009Q3", f(y.loc["2009Q3"]), f(0.5 * r), "2009Q3", f(n), "1.0",
         f(n)],
        ["", "", "", "2009Q4", f(n), f(0.5), f(0.5 * n)],
    ])

    table = news.summary_details(source="revisions", float_format="%.4f")
    assert_equal(table.title, "Details of revisions for [revised variable = "
                              "infl, impacted variable = infl]")
    assert_equal(table_values(table), [
        ["revision date", "revised", "observed (prev)", "impact date",
         "revision", "weight", "impact"],
        ["2009Q2", f(r), f(0), "2009Q2", f(r), "1.0", f(r)],
        ["", "", "", "2009Q3", f(r), f(0.5), f(0.5 * r)],
        ["", "", "", "2009Q4", f(r), f(0.25), f(0.25 * r)],
    ])

    # In multiple tables, the variables of a univariate model are in the
    # title, not in columns
    tables = news.summary_details(multiple_tables=True)
    assert_equal(len(tables), 1)
    assert_equal(tables[0].title, "Details of news for [update date = 2009Q3, "
                                  "updated variable = infl, impacted variable = "
                                  "infl]")
    assert_equal(table_values(tables[0])[0], [
        "observed", "forecast (prev)", "impact date", "news", "weight",
        "impact"])

    # Selecting the single variable explicitly gives the same table, and a
    # single impact date leaves a single row
    desired = table_values(news.summary_details())
    for kwargs in [{"impacted_variable": "infl"},
                   {"updated_variable": "infl"}]:
        assert_equal(table_values(news.summary_details(**kwargs)), desired)
    table = news.summary_details(impact_date="2009Q4", float_format="%.4f")
    assert_("impact date = 2009Q4" in table.title)
    assert_equal(table_values(table)[1:],
                 [["2009Q3", f(y.loc["2009Q3"]), f(0.5 * r), f(n), f(0.5),
                   f(0.5 * n)]])


def details_rows(news, source, groupby):
    # The rows of the details table without sparsifying, built from
    # details_by_impact or revision_details_by_impact
    if source == "news":
        details = news.details_by_impact
        date, variable, current, prev, change = (
            "update date", "updated variable", "observed", "forecast (prev)",
            "news")
    else:
        details = news.revision_details_by_impact
        date, variable, current, prev, change = (
            "revision date", "revised variable", "revised", "observed (prev)",
            "revision")
    keys = {
        "impact date": ["impact date", "impacted variable", date, variable],
        "impacted variable": ["impacted variable", "impact date", variable,
                              date],
        "update date": [date, variable, "impact date", "impacted variable"],
        "updated variable": [variable, date, "impacted variable",
                             "impact date"],
    }[groupby]
    if groupby in ["update date", "updated variable"]:
        columns = keys[:2] + [current, prev] + keys[2:]
    else:
        columns = keys + [current, prev]
    columns += [change, "weight", "impact"]
    details = details.reset_index().sort_values(keys)
    return columns, keys, details[columns]


@pytest.mark.parametrize("source", ["news", "revisions"])
@pytest.mark.parametrize("groupby", ["impact date", "impacted variable",
                                     "update date", "updated variable"])
def test_summary_details_groupby(source, groupby):
    news = bivariate_news()
    table = news.summary_details(source=source, groupby=groupby,
                                 sparsify=False, float_format="%.4f")
    assert_equal(table.title, f"Details of {source}")
    values = table_values(table)
    columns, keys, desired = details_rows(news, source, groupby)
    assert_equal(values[0], columns)
    assert_equal(len(values) - 1, len(desired))
    is_key = np.isin(columns, keys)
    for row, (_, value) in zip(values[1:], frame_rows(desired), strict=True):
        row = np.array(row)
        assert_equal(row[is_key], [str(x) for x in value[is_key]])
        # The weights are rounded to 1.0 when they are within the tolerance
        # of 1, otherwise all numbers are shown with 4 digits (atol for the
        # rounding to 4 digits)
        assert_allclose(row[~is_key].astype(float),
                        value[~is_key].astype(float), atol=5e-5)
    # The underscore spelling is the same option
    alias = news.summary_details(source=source,
                                 groupby=groupby.replace(" ", "_"),
                                 sparsify=False, float_format="%.4f")
    assert_equal(table_values(alias), values)


def test_summary_details_sparsify_impact_date():
    news = bivariate_news()
    table = news.summary_details(groupby="impact date")
    # The single update date is moved to the title
    assert_equal(table.title, "Details of news for [update date = 2009Q3]")
    values = table_values(table)
    columns, keys, desired = details_rows(news, "news", "impact date")
    columns.remove("update date")
    assert_equal(values[0], columns)
    for i, key in enumerate(columns[:3]):
        actual = [row[i] for row in values[1:]]
        assert_equal(actual, sparsified(desired[key].tolist()))
    # The data columns are not sparsified with this grouping
    assert_equal([row[3] for row in values[1:]],
                 [f"{x:.2f}" for x in desired["observed"]])

    with pytest.raises(ValueError, match="Invalid groupby for details table"):
        news.summary_details(groupby="revision date")


def test_summary_details_sparsify_equal_values():
    # Equal observed values of different updated variables at the same date
    # are not blanked
    news = bivariate_news(update=1.0)
    table = news.summary_details(float_format="%.4f", sparsify=True)
    values = table_values(table)
    columns, keys, desired = details_rows(news, "news", "update date")
    assert_equal(values[0], columns)
    variables = desired["updated variable"].tolist()
    expected = ["1.0000" if i == 0 or variable != variables[i - 1] else ""
                for i, variable in enumerate(variables)]
    assert_equal([row[2] for row in values[1:]], expected)
    assert_equal(expected.count("1.0000"), 2)


def test_summary_details_update_date():
    # AR(1) with phi = 0.5 and two updates, I2 and I3 (the forecast errors of
    # 2009Q2 and 2009Q3 given the data through 2009Q1). I2 only has an impact
    # on 2009Q2 and I3 has weights 1 and phi on 2009Q3 and 2009Q4.
    endog = dta["infl"]
    res = sarimax.SARIMAX(endog.loc[:"2009Q1"]).smooth([0.5, 1.0])
    news = res.news(endog, start="2009Q2", end="2009Q4")
    y1, y2, y3 = endog.loc["2009Q1":"2009Q3"]
    i2 = y2 - 0.5 * y1
    i3 = y3 - 0.25 * y1

    def f(x):
        return f"{x:.4f}"

    header = ["update date", "observed", "forecast (prev)", "impact date",
              "news", "weight", "impact"]
    rows2 = [["2009Q2", f(y2), f(0.5 * y1), "2009Q2", f(i2), "1.0", f(i2)]]
    rows3 = [["2009Q3", f(y3), f(0.25 * y1), "2009Q3", f(i3), "1.0", f(i3)],
             ["", "", "", "2009Q4", f(i3), f(0.5), f(0.5 * i3)]]
    table = news.summary_details(float_format="%.4f")
    assert_equal(table_values(table), [header] + rows2 + rows3)
    table = news.summary_details(update_date="2009Q3", float_format="%.4f")
    assert_equal(table_values(table), [header] + rows3)
    # With a single row left, its impact date is moved to the title
    table = news.summary_details(update_date="2009Q2", float_format="%.4f")
    assert_("impact date = 2009Q2" in table.title)
    del header[3], rows2[0][3]
    assert_equal(table_values(table), [header] + rows2)


def test_summary_revision_selection():
    # revision_date and revised_variable select the details of revisions. In
    # the AR(1) model, the revisions of 2009Q1 and 2009Q2 have impacts on
    # 2009Q1 and on 2009Q2-2009Q4.
    endog = dta["infl"]
    previous = endog.loc[:"2009Q2"].copy()
    previous.loc["2009Q1":"2009Q2"] = 0.0
    res = sarimax.SARIMAX(previous).smooth([0.5, 1.0])
    news = res.news(endog, start="2009Q1", end="2009Q4",
                    revisions_details_start=True)
    summary = news.summary(include_revisions_tables=True,
                           revision_date="2009Q2")
    desired = news.summary_details(source="revisions", update_date="2009Q2")
    assert_equal(str(summary.tables[-1]), str(desired))
    values = table_values(desired)
    assert_equal([row[0] for row in values[1:]],
                 ["2009Q2"] + [""] * (len(values) - 2))
    assert_(str(desired) != str(news.summary_details(source="revisions")))

    news = bivariate_news()
    summary = news.summary(include_details_tables=True,
                           include_revisions_tables=True,
                           revised_variable="unemp")
    desired = news.summary_details(source="revisions",
                                   updated_variable="unemp",
                                   multiple_tables=True)
    assert_equal(len(desired), 1)
    assert_("revised variable = unemp" in desired[0].title)
    assert_equal(str(summary.tables[-1]), str(desired[0]))
    unselected = news.summary_details(source="revisions",
                                      multiple_tables=True)
    assert_(str(desired[0]) != str(unselected[0]))


def test_summary_details_selection():
    news = bivariate_news()
    table = news.summary_details(
        impact_date="2010Q1", impacted_variable="unemp", update_date="2009Q3",
        updated_variable="realgdp", float_format="%.4f")
    # Every key has a single value, so they are moved to the title, except
    # the first key (of the default grouping by update date)
    assert_equal(table.title, (
        "Details of news for [updated variable = realgdp, impact date = "
        "2010Q1, impacted variable = unemp]"))
    desired = news.details_by_impact.loc[
        ("2010Q1", "unemp", "2009Q3", "realgdp")]
    assert_equal(table_values(table), [
        ["update date", "observed", "forecast (prev)", "news", "weight",
         "impact"],
        ["2009Q3"] + [f"{x:.4f}" for x in desired]])


@pytest.mark.parametrize("groupby", ["update date", "updated variable"])
def test_summary_details_multiple_tables(groupby):
    news = bivariate_news()
    tables = news.summary_details(groupby=groupby, multiple_tables=True)
    details = news.details_by_impact
    keys = details.index.get_level_values(groupby).unique().sort_values()
    assert_equal(len(tables), len(keys))
    for table, key in zip(tables, keys, strict=True):
        assert_(f"[{groupby} = {key}" in table.title)
        values = table_values(table)
        assert_(groupby not in values[0])
        subset = details[details.index.get_level_values(groupby) == key]
        assert_equal(len(values) - 1, len(subset))
        # The impacts of each table are those of its key (atol for the
        # rounding to 2 digits)
        assert_allclose(sorted(float(row[-1]) for row in values[1:]),
                        np.sort(subset["impact"].to_numpy()), atol=5e-3)


def test_summary_details_single_impact_date():
    # With a single impact date, its column is kept in multiple tables unless
    # it is selected
    news = bivariate_news(impact_date="2009Q4")
    tables = news.summary_details(multiple_tables=True)
    assert_equal(len(tables), 1)
    values = table_values(tables[0])
    assert_("impact date" in values[0])
    col = values[0].index("impact date")
    assert_equal({row[col] for row in values[1:]}, {"2009Q4", ""})
    tables = news.summary_details(impact_date="2009Q4", multiple_tables=True)
    assert_("impact date" not in table_values(tables[0])[0])
    assert_("impact date = 2009Q4" in tables[0].title)
    # Without multiple tables, the column is dropped in any case
    table = news.summary_details()
    assert_("impact date" not in table_values(table)[0])
    assert_("impact date = 2009Q4" in table.title)


def test_summary_details_weights_below_tolerance():
    # Weights smaller than the tolerance are shown as 0, while the impacts
    # (larger than the tolerance) are shown
    news = bivariate_news()
    news.tolerance = 0.3
    table = news.summary_details(sparsify=False)
    values = table_values(table)
    details = news.details_by_impact
    weight_col = values[0].index("weight")
    impact_col = values[0].index("impact")
    small = details[np.abs(details["weight"]) < 0.3]
    assert_(len(small) > 0)
    n_small = 0
    for row in values[1:]:
        key = (row[4], row[5], row[0], row[1])
        if key in small.index:
            n_small += 1
            impact = small.loc[key, "impact"]
            assert_equal(row[weight_col], "0")
            assert_equal(row[impact_col], f"{impact:.2f}")
    assert_equal(n_small, len(small))


@pytest.mark.parametrize("sparsify", [True, False])
def test_summary_news_and_revisions(sparsify):
    news = bivariate_news()
    table = news.summary_news(sparsify=sparsify)
    values = table_values(table)
    assert_equal(values[0], ["update date", "updated variable", "observed",
                             "forecast (prev)", "news"])
    updates = news.data_updates
    dates = [str(x) for x in updates.index.get_level_values(0)]
    assert_equal([row[0] for row in values[1:]],
                 sparsified(dates) if sparsify else dates)
    for row, (key, value) in zip(values[1:], frame_rows(updates), strict=True):
        desired = [value["observed"], value["forecast (prev)"],
                   news.news.loc[key]]
        assert_equal(row[1:], [key[1]] + [f"{x:.2f}" for x in desired])

    table = news.summary_revisions(sparsify=sparsify)
    values = table_values(table)
    assert_equal(values[0], ["revision date", "revised variable",
                             "observed (prev)", "revision",
                             "detailed impacts computed"])
    revisions = news.data_revisions
    dates = [str(x) for x in revisions.index.get_level_values(0)]
    assert_equal([row[0] for row in values[1:]],
                 sparsified(dates) if sparsify else dates)
    for row, (key, value) in zip(values[1:], frame_rows(revisions),
                                 strict=True):
        desired = [value["observed (prev)"],
                   value["revised"] - value["observed (prev)"]]
        assert_equal(row[1:],
                     [key[1]] + [f"{x:.2f}" for x in desired] + ["True"])


def summary_header(summary):
    # The rows of the top table of a summary as a dict
    out = {}
    for row in table_values(summary.tables[0]):
        out[row[0].strip()] = row[1].strip()
        out[row[2].strip()] = row[3].strip()
    return out


@pytest.mark.parametrize("include_details_tables", [None, True, False])
@pytest.mark.parametrize("include_revisions_tables", [True, False])
@pytest.mark.parametrize("name", ["infl", "y"])
def test_summary_univariate(include_details_tables, include_revisions_tables,
                            name):
    news = univariate_news(name=name)
    summary = news.summary(include_details_tables=include_details_tables,
                           include_revisions_tables=include_revisions_tables)
    header = summary_header(summary)
    assert_equal(header["Model:"], "SARIMAX")
    assert_equal(header["Original sample:"], "1959Q1")
    assert_equal(header["Update through:"], "2009Q3")
    assert_equal(header["# of revisions:"], "1")
    assert_equal(header["# of new datapoints:"], "1")
    assert_("# of included states:" not in header)

    # The details tables are included by default in univariate models
    details = include_details_tables in [None, True]
    desired = [news.summary_impacts(), news.summary_news()]
    if details:
        desired.append(news.summary_details())
    if include_revisions_tables:
        desired.append(news.summary_revisions())
        if details:
            desired.append(news.summary_details(source="revisions"))
    assert_equal([str(table) for table in summary.tables[1:]],
                 [str(table) for table in desired])


def test_summary_arguments():
    # Each selection argument of summary goes to the matching table
    news = univariate_news()
    summary = news.summary(
        impact_date=slice("2009Q3", "2009Q4"), update_date="2009Q3",
        revision_date="2009Q2", impacts_groupby="impact date",
        details_groupby="impact date", show_revisions_columns=False,
        sparsify=False, include_revisions_tables=True, float_format="%.3f")
    desired = [
        news.summary_impacts(impact_date=slice("2009Q3", "2009Q4"),
                             show_revisions_columns=False, sparsify=False,
                             float_format="%.3f"),
        news.summary_news(sparsify=False),
        news.summary_details(impact_date=slice("2009Q3", "2009Q4"),
                             update_date="2009Q3", groupby="impact date",
                             sparsify=False, float_format="%.3f"),
        news.summary_revisions(sparsify=False),
        news.summary_details(source="revisions",
                             impact_date=slice("2009Q3", "2009Q4"),
                             update_date="2009Q2", groupby="impact date",
                             sparsify=False, float_format="%.3f"),
    ]
    assert_equal([str(table) for table in summary.tables[1:]],
                 [str(table) for table in desired])
    # The selections are applied: two impact dates in the revision details
    values = table_values(summary.tables[-1])
    assert_equal(len(values) - 1, 2)


@pytest.mark.parametrize("include_details_tables", [None, True])
def test_summary_multivariate(include_details_tables):
    news = bivariate_news()
    summary = news.summary(include_details_tables=include_details_tables,
                           include_revisions_tables=True)
    header = summary_header(summary)
    assert_equal(header["Model:"], "VARMAX")
    assert_equal(header["# of revisions:"], "2")
    assert_equal(header["# of new datapoints:"], "2")
    # The details tables are not included by default in multivariate models;
    # they are one table for each update (revision) date
    desired = [news.summary_impacts(), news.summary_news()]
    if include_details_tables:
        desired += news.summary_details(multiple_tables=True)
    desired.append(news.summary_revisions())
    if include_details_tables:
        desired += news.summary_details(source="revisions",
                                        multiple_tables=True)
    assert_equal([str(table) for table in summary.tables[1:]],
                 [str(table) for table in desired])
    if include_details_tables:
        assert_equal(len(summary.tables), 6)


@pytest.mark.parametrize("state_index", [[0], [1, 0]])
def test_state_index(state_index):
    # In this VAR(1) the design matrix is the identity, so the impacts of the
    # news and the detailed impacts of the revisions through the first state
    # are the full impacts on realgdp and none on unemp
    full = bivariate_news()
    news = bivariate_news(state_index=state_index)
    assert_equal(news.state_index, np.sort(state_index))
    for attr in ["revision_detailed_impacts", "update_impacts"]:
        actual = getattr(news, attr)
        desired = getattr(full, attr)
        assert_allclose(actual["realgdp"], desired["realgdp"], rtol=1e-10)
        if len(state_index) == 1:
            # The design matrix has an exact zero, so there is no impact
            assert_allclose(actual["unemp"], 0, atol=1e-15)
        else:
            assert_allclose(actual["unemp"], desired["unemp"], rtol=1e-10)
    header = summary_header(news.summary())
    if len(state_index) == 1:
        assert_equal(header["# of included states:"], "1")
    else:
        assert_("# of included states:" not in header)


@pytest.mark.parametrize("include_revisions", [True, False])
@pytest.mark.parametrize("include_updates", [True, False])
def test_get_details_and_impacts(include_revisions, include_updates):
    news = bivariate_news()
    details = news.get_details(include_revisions=include_revisions,
                               include_updates=include_updates)
    assert_equal(details.columns.tolist(),
                 ["observed", "previous", "news", "weight", "impact"])
    assert_equal(details.index.names, ["impact date", "impacted variable",
                                       "update date", "updated variable"])
    desired = 0 * news.total_impacts
    if include_revisions:
        desired += news.revision_impacts
    if include_updates:
        desired += news.update_impacts
    desired = desired.stack(**FUTURE_STACK)
    # The details summed over the updates and revisions give the impacts
    # (atol for the impacts of the news on 2009Q2, zero up to rounding)
    actual = details["impact"].groupby(level=[0, 1]).sum()
    actual = actual.reindex(desired.index).fillna(0)
    assert_allclose(actual, desired, rtol=1e-10, atol=1e-12)

    impacts = news.get_impacts(include_revisions=include_revisions,
                               include_updates=include_updates)
    if include_revisions or include_updates:
        actual = impacts.sum().reindex(desired.index).fillna(0)
        assert_allclose(actual, desired, rtol=1e-10, atol=1e-12)
    else:
        assert_equal(impacts.shape, (0, 0))


# pandas < 3 warns in `get_impacts` with a `groupby` (stack without
# future_stack)
@pytest.mark.filterwarnings(
    "ignore:The previous implementation of stack:FutureWarning")
@pytest.mark.filterwarnings("ignore:The behavior of Index.insert:FutureWarning")
def test_get_impacts_groupby():
    news = bivariate_news()
    impacts = news.get_impacts()
    assert_equal(impacts.index.names, ["update date", "updated variable"])
    # The groups are those of the updated variables, for each update date
    grouped = news.get_impacts(groupby=lambda name: "all")
    for date in ["2009Q2", "2009Q3"]:
        desired = impacts.loc[date].sum(min_count=1)
        assert_allclose(grouped.loc[(date, "all")],
                        desired[grouped.columns], rtol=1e-12)
    # The sum over the update dates is the total impact (atol as in
    # test_get_details_and_impacts)
    desired = news.total_impacts.stack(**FUTURE_STACK)
    assert_allclose(grouped.sum()[desired.index], desired,
                    rtol=1e-10, atol=1e-12)
    # Each variable in its own group gives the impacts
    grouped = news.get_impacts(groupby={"realgdp": "realgdp",
                                        "unemp": "unemp"})
    assert_allclose(grouped, impacts.loc[grouped.index, grouped.columns],
                    rtol=1e-12)


# - Index types and selections of the NewsResults tables ----------------------


def indexed_news(index_type, multivariate):
    # The models of univariate_news (the AR(1) of infl) and bivariate_news
    # with the dates given as a PeriodIndex, a DatetimeIndex or integers (the
    # data given as ndarrays). The last previous value is revised and the
    # last updated one is new; the impact dates are the last two dates and
    # the next one (two for the bivariate model).
    if multivariate:
        endog = dta[["realgdp", "unemp"]].copy()
        endog["realgdp"] = np.log(endog["realgdp"]).diff() * 400
        endog = endog.iloc[1:]
        mod = varmax.VARMAX
        params = np.r_[0.5, 0.1, 0.2, 0.9, 1.0, 0.1, 1.1]
        kwargs = {"trend": "n"}
    else:
        endog = dta["infl"]
        mod = sarimax.SARIMAX
        params = [0.5, 1.0]
        kwargs = {}
    previous = endog.iloc[:-1].copy()
    previous.iloc[-1] = 0.0
    if index_type == "datetime":
        index = pd.date_range(endog.index[0].to_timestamp(),
                              periods=len(endog), freq="QS")
        endog = endog.set_axis(index)
        previous = previous.set_axis(index[:-1])
    elif index_type == "int":
        endog = endog.to_numpy()
        previous = previous.to_numpy()
    res = mod(previous, **kwargs).smooth(params)
    nobs = len(endog)
    end = nobs + 1 if multivariate else nobs
    return res.news(endog, start=nobs - 2, end=end,
                    revisions_details_start=True)


@pytest.mark.parametrize("index_type", ["period", "datetime", "int"])
@pytest.mark.parametrize("multivariate", [True, False])
@pytest.mark.parametrize("sparsify", [True, False])
def test_summary_impacts_index_types(index_type, multivariate, sparsify):
    # Every cell of the impacts table is that of the impacts DataFrame, with
    # the impact dates blanked where they repeat if sparsified (they were NaT
    # for dates, and integer dates raised a TypeError on pandas 3)
    news = indexed_news(index_type, multivariate)
    impacts = news.impacts
    table = news.summary_impacts(sparsify=sparsify, float_format="%.4f")
    values = table_values(table)
    # The single impacted variable of a univariate model is moved to the
    # title if sparsified
    keys = ["impact date", "impacted variable"]
    if sparsify and not multivariate:
        keys = ["impact date"]
    assert_equal(values[0], keys + IMPACTS_COLUMNS)
    dates = [str(date) for date in impacts.index.get_level_values(0)]
    if multivariate:
        # Each date has a row for each variable
        assert_equal(len(set(dates)), len(dates) // 2)
    assert_equal([row[0] for row in values[1:]],
                 sparsified(dates) if sparsify else dates)
    if len(keys) == 2:
        assert_equal([row[1] for row in values[1:]],
                     impacts.index.get_level_values(1).tolist())
    # atol for the rounding to 4 digits
    numbers = np.array([row[len(keys):] for row in values[1:]], dtype=float)
    assert_allclose(numbers, impacts.to_numpy(), rtol=0, atol=5e-5)
    # summary shows the same table
    summary = news.summary(sparsify=sparsify, float_format="%.4f")
    assert_equal(str(summary.tables[1]), str(table))


@pytest.mark.parametrize("multivariate, select_variable",
                         [(True, True), (False, True), (False, False)])
@pytest.mark.parametrize("impact_date, year, quarters", [
    ("2009Q3", 2009, [3]),
    (pd.Period("2009Q3", freq="Q"), 2009, [3]),
    # A partial date string selects several dates
    ("2009", 2009, [1, 2, 3, 4]),
])
def test_summary_impacts_single_date_and_variable(
        multivariate, select_variable, impact_date, year, quarters):
    # A single impact date and a single impacted variable select the rows of
    # the impacts DataFrame (they raised an AttributeError). Univariate
    # models select their variable by default.
    if multivariate:
        news = bivariate_news()
        variable = "unemp"
    else:
        news = univariate_news()
        variable = news.endog_names
    impacts = news.impacts
    dates = impacts.index.get_level_values(0)
    mask = ((impacts.index.get_level_values(1) == variable)
            & (dates.year == year) & np.isin(dates.quarter, quarters))
    desired = impacts[mask]
    assert_(len(desired) > 0)
    kwargs = {"impact_date": impact_date, "float_format": "%.4f"}
    if select_variable:
        kwargs["impacted_variable"] = variable
    table = news.summary_impacts(**kwargs)
    assert_(f"[impacted variable = {variable}]" in table.title)
    values = table_values(table)
    assert_equal(values[0], ["impact date"] + IMPACTS_COLUMNS)
    assert_equal([row[0] for row in values[1:]],
                 [str(date) for date in desired.index.get_level_values(0)])
    # atol for the rounding to 4 digits
    numbers = np.array([row[1:] for row in values[1:]], dtype=float)
    assert_allclose(numbers, desired.to_numpy(), rtol=0, atol=5e-5)
    # The impacts table of summary
    summary = news.summary(**kwargs)
    assert_equal(str(summary.tables[1]), str(table))
    # A date without impacts raises
    with pytest.raises(KeyError):
        news.summary_impacts(impact_date="2010Q4",
                             impacted_variable=variable)


@pytest.mark.parametrize("groupby", ["impact date", "impact_date",
                                     "Impact_Date", "impacted variable",
                                     "impacted_variable"])
def test_summary_impacts_groupby_spellings(groupby):
    # Every spelling of groupby sparsifies the first column (the underscore
    # spellings did not), and a single value of the other key is moved to
    # the title, "on" an impacted variable and "for" an impact date
    news = bivariate_news()
    key = groupby.lower().replace("_", " ")
    other = "impacted variable" if key == "impact date" else "impact date"
    impacts = news.impacts
    if key == "impacted variable":
        impacts = impacts.swaplevel().sort_index()
    table = news.summary_impacts(groupby=groupby, float_format="%.4f")
    assert_equal(table.title, "Impacts")
    values = table_values(table)
    assert_equal(values[0], [key, other] + IMPACTS_COLUMNS)
    first = [str(x) for x in impacts.index.get_level_values(0)]
    assert_equal([row[0] for row in values[1:]], sparsified(first))
    assert_equal([row[1] for row in values[1:]],
                 [str(x) for x in impacts.index.get_level_values(1)])
    # atol for the rounding to 4 digits
    numbers = np.array([row[2:] for row in values[1:]], dtype=float)
    assert_allclose(numbers, impacts.to_numpy(), rtol=0, atol=5e-5)

    if key == "impact date":
        table = news.summary_impacts(groupby=groupby,
                                     impacted_variable="unemp")
        assert_equal(table.title, "Impacts on [impacted variable = unemp]")
    else:
        table = news.summary_impacts(groupby=groupby, impact_date="2009Q3")
        assert_equal(table.title, "Impacts for [impact date = 2009Q3]")
    assert_equal(table_values(table)[0], [key] + IMPACTS_COLUMNS)


@pytest.mark.parametrize("source", ["news", "revisions"])
@pytest.mark.parametrize("groupby", ["impact date", "impacted variable"])
def test_summary_details_multiple_tables_by_impact(source, groupby):
    # One table for each impact date or impacted variable (they raised a
    # KeyError), with the rows of the details DataFrame for that key
    news = bivariate_news()
    tables = news.summary_details(source=source, groupby=groupby,
                                  multiple_tables=True, sparsify=False,
                                  float_format="%.4f")
    columns, keys, desired = details_rows(news, source, groupby)
    groups = desired[groupby].unique()
    assert_(len(groups) > 1)
    assert_equal(len(tables), len(groups))
    columns.remove(groupby)
    is_key = np.isin(columns, keys)
    for table, group in zip(tables, groups, strict=True):
        assert_equal(table.title, f"Details of {source} for [{groupby} = "
                                  f"{group}]")
        values = table_values(table)
        assert_equal(values[0], columns)
        subset = desired.loc[desired[groupby] == group, columns]
        assert_equal(len(values) - 1, len(subset))
        for row, (_, value) in zip(values[1:], frame_rows(subset),
                                   strict=True):
            row = np.array(row)
            assert_equal(row[is_key], [str(x) for x in value[is_key]])
            # The weights are rounded to 1.0 when they are within the
            # tolerance of 1 (atol for the rounding to 4 digits)
            assert_allclose(row[~is_key].astype(float),
                            value[~is_key].astype(float), rtol=0, atol=5e-5)

    # summary shows these tables (sparsified)
    summary = news.summary(include_details_tables=True,
                           include_revisions_tables=True,
                           details_groupby=groupby)
    desired = [news.summary_impacts(), news.summary_news()]
    desired += news.summary_details(groupby=groupby, multiple_tables=True)
    desired.append(news.summary_revisions())
    desired += news.summary_details(source="revisions", groupby=groupby,
                                    multiple_tables=True)
    assert_equal([str(table) for table in summary.tables[1:]],
                 [str(table) for table in desired])


@pytest.mark.parametrize("name", ["infl", "y"])
def test_univariate_defaults(name):
    # The univariate defaults apply whatever the name of the series (k_endog
    # was the length of the name, so they applied only to one-letter names)
    news = univariate_news(name=name)
    assert_equal(news.k_endog, 1)
    # summary shows a single details table of the news and of the
    # revisions by default, for each grouping
    for groupby in ["update date", "impact date"]:
        summary = news.summary(include_revisions_tables=True,
                               details_groupby=groupby)
        desired = [news.summary_impacts(), news.summary_news(),
                   news.summary_details(groupby=groupby),
                   news.summary_revisions(),
                   news.summary_details(source="revisions", groupby=groupby)]
        assert_equal([str(table) for table in summary.tables[1:]],
                     [str(table) for table in desired])
    # Details tables with a single row are shown
    summary = news.summary(include_revisions_tables=True,
                           impact_date="2009Q3")
    desired = [news.summary_impacts(impact_date="2009Q3"),
               news.summary_news(),
               news.summary_details(impact_date="2009Q3"),
               news.summary_revisions(),
               news.summary_details(source="revisions", impact_date="2009Q3")]
    assert_equal(len(table_values(desired[2])), 2)
    assert_equal(len(table_values(desired[4])), 2)
    assert_equal([str(table) for table in summary.tables[1:]],
                 [str(table) for table in desired])
    # In multiple tables, the variable is in the title, not in columns
    for source, variable in [("news", "updated variable"),
                             ("revisions", "revised variable")]:
        tables = news.summary_details(source=source, multiple_tables=True)
        assert_equal(len(tables), 1)
        assert_(tables[0].title.endswith(
            f"{variable} = {name}, impacted variable = {name}]"))
        header = table_values(tables[0])[0]
        assert_(variable not in header)
        assert_("impacted variable" not in header)


def test_summary_details_multiple_tables_single_variable():
    # A multivariate model keeps a single updated variable in a column of
    # multiple tables unless it is selected; only univariate models move it
    # to the title by default
    news = bivariate_news(update=[1.0, np.nan])
    tables = news.summary_details(multiple_tables=True)
    assert_equal(len(tables), 1)
    assert_("updated variable" in table_values(tables[0])[0])
    assert_equal(tables[0].title, "Details of news for [update date = 2009Q3]")
    tables = news.summary_details(multiple_tables=True,
                                  updated_variable="realgdp")
    assert_("updated variable" not in table_values(tables[0])[0])
    assert_equal(tables[0].title, "Details of news for [update date = "
                                  "2009Q3, updated variable = realgdp]")


@pytest.mark.parametrize("name", ["infl", "y"])
@pytest.mark.parametrize("case", ["revisions only", "mixed revisions",
                                  "below tolerance"])
def test_univariate_defaults_without_variable_rows(name, case):
    # The univariate defaults do not select the rows of the details tables:
    # selecting the variable raised a KeyError for an empty table (no news,
    # or no impact above the tolerance) and for revisions that are all
    # grouped, and dropped the grouped revisions ("all prior revisions")
    # otherwise. A single details table without rows is left out.
    endog = dta["infl"].rename(name)
    kwargs = {"start": "2009Q1", "end": "2009Q4"}
    if case == "revisions only":
        # All revisions grouped (the default)
        previous = endog.copy()
        kwargs["comparison_type"] = "updated"
    else:
        previous = endog.loc[:"2009Q2"].copy()
    if case == "mixed revisions":
        kwargs["revisions_details_start"] = "2009Q2"
    elif case == "below tolerance":
        kwargs["tolerance"] = 1e6
    previous.loc["2009Q1":"2009Q2"] = 0.0
    res = sarimax.SARIMAX(previous).smooth([0.5, 1.0])
    news = res.news(endog, **kwargs)

    summary = news.summary(include_revisions_tables=True)
    desired = [news.summary_impacts()]
    if case == "below tolerance":
        assert_equal(len(news.impacts), 0)
        assert_equal(len(news.details_by_impact), 0)
        assert_equal(len(news.revision_details_by_impact), 0)
        desired += [news.summary_news(), news.summary_revisions()]
    else:
        if case == "mixed revisions":
            desired += [news.summary_news(), news.summary_details()]
        else:
            assert_equal(len(news.details_by_impact), 0)
        desired += [news.summary_revisions(),
                    news.summary_details(source="revisions")]
    assert_equal([str(table) for table in summary.tables[1:]],
                 [str(table) for table in desired])
    if case == "below tolerance":
        return

    # The details of the revisions sum to the impacts of the revisions
    table = news.summary_details(source="revisions", sparsify=False,
                                 float_format="%.6f")
    values = table_values(table)
    header = values[0]
    variables = [row[header.index("revised variable")] for row in values[1:]]
    assert_("all prior revisions" in variables)
    dates = [row[header.index("impact date")] for row in values[1:]]
    impacts = [float(row[header.index("impact")]) for row in values[1:]]
    actual = pd.Series(impacts).groupby(dates).sum()
    revision_impacts = pd.Series(
        news.revision_impacts[name].to_numpy(),
        index=news.revision_impacts.index.astype(str))
    # The dates without rows have no impact (the observed 2009Q3, and its
    # forecast 2009Q4, if there are no news)
    actual = actual.reindex(revision_impacts.index, fill_value=0.0)
    # atol for the rounding to 6 digits
    assert_allclose(actual, revision_impacts, rtol=0, atol=1e-5)


# - Selections without impacts above the tolerance ----------------------------


@pytest.mark.parametrize("model, tolerance, kwargs", [
    # The observed 2009Q2 has no impact of the news (no revision)
    ("univariate", None, {"impact_date": "2009Q2"}),
    ("univariate", None, {"impact_date": "2009Q2", "impacted_variable": "infl"}),
    # A tolerance above every impact
    ("univariate", 1e6, {"impacted_variable": "infl"}),
    ("bivariate", 1e6, {"impact_date": "2009Q3"}),
    ("bivariate", 1e6, {"impacted_variable": "unemp"}),
    ("bivariate", 1e6, {"impact_date": "2009Q3", "impacted_variable": "unemp"}),
], ids=["uni-date", "uni-date-variable", "uni-variable", "bi-date",
        "bi-variable", "bi-date-variable"])
def test_summary_impacts_selection_within_tolerance(model, tolerance, kwargs):
    # An impact date or impacted variable whose impacts are all within the
    # tolerance selects no rows, so the table has only the header (it raised
    # a KeyError: the label was looked up in impacts, which leaves out these
    # rows). A label that is not an impact date or variable still raises.
    news_kwargs = {} if tolerance is None else {"tolerance": tolerance}
    if model == "univariate":
        news = univariate_news(revision=False, **news_kwargs)
        columns = ["estimate (prev)", "impact of news", "estimate (new)"]
    else:
        news = bivariate_news(**news_kwargs)
        columns = IMPACTS_COLUMNS
    # The labels are those of an impact date and an impacted variable, but
    # impacts has no row for them
    dates = news.post_impacted_forecasts.index.astype(str)
    assert_("impact_date" not in kwargs or kwargs["impact_date"] in dates)
    assert_("impacted_variable" not in kwargs
            or kwargs["impacted_variable"] in news.post_impacted_forecasts)
    impacts = news.impacts
    mask = np.ones(len(impacts), dtype=bool)
    if "impact_date" in kwargs:
        mask &= (impacts.index.get_level_values(0).astype(str)
                 == kwargs["impact_date"])
    if "impacted_variable" in kwargs:
        mask &= (impacts.index.get_level_values(1)
                 == kwargs["impacted_variable"])
    assert_equal(mask.sum(), 0)

    table = news.summary_impacts(**kwargs)
    assert_equal(table_values(table),
                 [["impact date", "impacted variable"] + columns])

    with pytest.raises(KeyError):
        news.summary_impacts(**{**kwargs, "impact_date": "2010Q4"})
    with pytest.raises(KeyError):
        news.summary_impacts(**{**kwargs, "impacted_variable": "cpi"})


def test_summary_impacts_list_within_tolerance():
    # A list with an impact date without impacts above the tolerance (the
    # observed 2009Q2) selects the rows of the other dates
    news = univariate_news(revision=False)
    impacts = news.impacts
    assert_equal(impacts.index.get_level_values(0).astype(str).tolist(),
                 ["2009Q3", "2009Q4"])
    table = news.summary_impacts(impact_date=["2009Q2", "2009Q3"],
                                 float_format="%.4f")
    values = table_values(table)
    columns = ["estimate (prev)", "impact of news", "estimate (new)"]
    assert_equal(values[0], ["impact date"] + columns)
    assert_equal([row[0] for row in values[1:]], ["2009Q3"])
    # atol for the rounding to 4 digits
    assert_allclose(np.array(values[1][1:], dtype=float),
                    impacts.loc["2009Q3", columns].to_numpy().ravel(),
                    rtol=0, atol=5e-5)


@pytest.mark.parametrize("source", ["news", "revisions"])
@pytest.mark.parametrize("key", ["impact_date", "impacted_variable",
                                 "update_date", "updated_variable"])
def test_summary_details_selection_within_tolerance(source, key):
    # A date or variable whose impacts are all within the tolerance (here
    # above every impact) selects no rows, so the details table has only
    # the header (it raised a KeyError: the label was looked up in the
    # details table, which leaves out these rows). A label that is not in
    # the details still raises.
    news = bivariate_news(tolerance=1e6)
    if source == "news":
        updates = news.updates_ix
        date_name, variable_name = "update date", "updated variable"
    else:
        updates = news.revisions_ix
        date_name, variable_name = "revision date", "revised variable"
    # Update (revision) date and variable of the data, impact date of the
    # news
    labels = {
        "impact_date": "2009Q3",
        "impacted_variable": "unemp",
        "update_date": str(updates[date_name].iloc[0]),
        "updated_variable": updates[variable_name].iloc[0],
    }
    assert_equal(len(news.impacts), 0)
    assert_equal(len(news.details_by_impact), 0)
    assert_equal(len(news.revision_details_by_impact), 0)
    columns, _, desired = details_rows(news, source, "update date")
    assert_equal(len(desired), 0)

    table = news.summary_details(source=source, **{key: labels[key]})
    assert_equal(table.title, f"Details of {source}")
    assert_equal(table_values(table), [columns])

    missing = "cpi" if key.endswith("variable") else "2010Q4"
    with pytest.raises(KeyError):
        news.summary_details(source=source, **{key: missing})


def test_summary_details_date_within_tolerance():
    # The observed 2009Q2 has no impact of the news (no revision): the
    # details table of this impact date has only the header, and a list
    # with it gives the rows of the other dates
    news = univariate_news(revision=False)
    columns, _, desired = details_rows(news, "news", "update date")
    impact_dates = desired["impact date"].astype(str)
    assert_equal(impact_dates.tolist(), ["2009Q3", "2009Q4"])
    table = news.summary_details(impact_date="2009Q2", sparsify=False)
    assert_equal(table_values(table), [columns])

    table = news.summary_details(impact_date=["2009Q2", "2009Q3"],
                                 float_format="%.4f")
    assert_("impact date = 2009Q3" in table.title)
    desired = desired[impact_dates == "2009Q3"]
    values = table_values(table)
    assert_equal(len(values) - 1, 1)
    row = dict(zip(values[0], values[1], strict=True))
    assert_equal(row["update date"], "2009Q3")
    # The weight 1 of an update on its own date is shown as "1.0" (atol for
    # the rounding to 4 digits)
    for name in ["observed", "forecast (prev)", "news", "weight", "impact"]:
        assert_allclose(float(row[name]), desired[name].iloc[0], rtol=0,
                        atol=5e-5)


@pytest.mark.parametrize("multivariate", [True, False])
def test_summary_date_within_tolerance(multivariate):
    # summary of an impact date whose impacts are all within the tolerance
    # (it raised a KeyError from summary_impacts and summary_details): the
    # impacts table has only the header, and the details tables without rows
    # are left out
    if multivariate:
        news = bivariate_news(tolerance=1e6)
        date = "2009Q3"
    else:
        news = univariate_news(revision=False)
        date = "2009Q2"
    summary = news.summary(impact_date=date, include_details_tables=True,
                           include_revisions_tables=True)
    impacts = news.summary_impacts(impact_date=date)
    assert_equal(len(table_values(impacts)), 1)
    desired = [impacts, news.summary_news()]
    if multivariate:
        desired.append(news.summary_revisions())
    assert_equal([str(table) for table in summary.tables[1:]],
                 [str(table) for table in desired])


@pytest.mark.parametrize("method", ["summary_impacts", "summary_details"])
def test_summary_invalid_groupby_message(method):
    # The words of the error message are separated by spaces (they missed a
    # space before "Got", and after the comma before "or")
    news = bivariate_news()
    if method == "summary_impacts":
        desired = ('Invalid groupby for impacts table. Valid options are '
                   '"impact date" or "impacted variable". Got "revision date".')
    else:
        desired = ('Invalid groupby for details table. Valid options are '
                   '"update date", "updated variable", "impact date", or '
                   '"impacted variable". Got "revision date".')
    with pytest.raises(ValueError) as excinfo:
        getattr(news, method)(groupby="Revision_Date")
    assert_equal(str(excinfo.value), desired)


@pytest.mark.parametrize(
    "start",
    [-2, np.int64(-2), np.int32(-2), np.array(-2), 36, np.int64(36), np.array(36)],
    ids=["int", "int64", "int32", "0d", "pos-int", "pos-int64", "pos-0d"],
)
def test_revisions_details_start_int_like(start):
    # A NumPy integer or a 0-d integer array is a period, as an int is, so
    # that a negative value is an offset from the end of the previous
    # dataset. A NumPy integer was looked up in the index of the updated
    # results, which has two more periods, and a 0-d array raised.
    rng = np.random.default_rng(20261126)
    index = pd.period_range("2000Q1", periods=40, freq="Q")
    y = pd.Series(rng.standard_normal(40), index=index, name="y")
    y_prev = y.iloc[:38].copy()
    y_upd = y.copy()
    y_upd.iloc[35] += 1.0
    y_upd.iloc[37] += 0.5
    res_prev = sarimax.SARIMAX(y_prev, order=(1, 0, 0)).smooth([0.5, 1.0])
    res_upd = res_prev.apply(y_upd)
    news = res_upd.news(
        res_prev, comparison_type="previous", revisions_details_start=start
    )
    # Revisions from period 36 = 38 - 2 on have detailed impacts
    revisions = news.data_revisions
    revision = revisions["revised"] - revisions["observed (prev)"]
    assert_allclose(revision, [1.0, 0.5], rtol=1e-12)
    assert_equal(np.asarray(revisions["detailed impacts computed"]), [False, True])


@pytest.mark.parametrize(
    "start, detailed", [(True, [True, True]), (False, [False, False])]
)
def test_revisions_details_start_bool(start, detailed):
    # A bool is a flag: all revisions are detailed or all are grouped
    rng = np.random.default_rng(20261126)
    index = pd.period_range("2000Q1", periods=40, freq="Q")
    y = pd.Series(rng.standard_normal(40), index=index, name="y")
    y_prev = y.iloc[:38].copy()
    y_upd = y.copy()
    y_upd.iloc[35] += 1.0
    y_upd.iloc[37] += 0.5
    res_prev = sarimax.SARIMAX(y_prev, order=(1, 0, 0)).smooth([0.5, 1.0])
    res_upd = res_prev.apply(y_upd)
    news = res_upd.news(
        res_prev, comparison_type="previous", revisions_details_start=start
    )
    revisions = news.data_revisions
    assert_equal(np.asarray(revisions["detailed impacts computed"]), detailed)


def test_revisions_details_start_float():
    # A float is not a period (strict), so it is taken as a date, which
    # pandas cannot convert to a period
    rng = np.random.default_rng(20261126)
    index = pd.period_range("2000Q1", periods=40, freq="Q")
    y = pd.Series(rng.standard_normal(40), index=index, name="y")
    res_prev = sarimax.SARIMAX(y.iloc[:38], order=(1, 0, 0)).smooth([0.5, 1.0])
    res_upd = res_prev.apply(y)
    with pytest.raises(ValueError):
        res_upd.news(res_prev, comparison_type="previous", revisions_details_start=36.0)


def var_news_state_index(state_index=None, revisions_details_start=True):
    # VAR(1) with an identity design matrix: the realgdp and unemp values of
    # 2009Q1 and 2009Q2 are revised from 0 and 2009Q3 is new; the impacts
    # are computed through 2010Q2 (three periods after the sample)
    endog = dta[["realgdp", "unemp"]].copy()
    endog["realgdp"] = np.log(endog["realgdp"]).diff() * 400
    endog = endog.iloc[1:]
    previous = endog.loc[:"2009Q2"].copy()
    previous.loc["2009Q1":"2009Q2"] = 0.0
    res = varmax.VARMAX(previous, trend="n").smooth(
        np.r_[0.5, 0.1, 0.2, 0.9, 1.0, 0.1, 1.1])
    return res.news(endog, start="2008Q4", end="2010Q2",
                    revisions_details_start=revisions_details_start,
                    state_index=state_index)


@pytest.mark.parametrize("revisions_details_start", [True, False, "2009Q2"])
def test_state_index_revision_impacts(revisions_details_start):
    # With state_index, every impact is the impact through those states: in
    # this VAR(1) with an identity design matrix, the impacts through the
    # state of each variable are the full impacts on that variable and none
    # on the other. That includes the impacts of the revisions, also those
    # that are grouped (all of them, or those of 2009Q1), which were the
    # impacts through all states.
    full = var_news_state_index(
        revisions_details_start=revisions_details_start)
    attrs = ["revision_impacts", "revision_detailed_impacts",
             "revision_grouped_impacts", "update_impacts"]

    def get(news, attr):
        # The detailed impacts are NaN if no revision has details
        return getattr(news, attr).fillna(0)

    # Each variable has impacts of revisions and news, so that the test
    # discriminates
    for attr in ["revision_impacts", "update_impacts"]:
        assert_(np.all(np.abs(get(full, attr)).max() > 0.1))
    columns = ["impact of revisions", "impact of news", "total impact"]
    variables = ["realgdp", "unemp"]
    for i, variable in enumerate(variables):
        other = variables[1 - i]
        news = var_news_state_index(
            state_index=[i], revisions_details_start=revisions_details_start)
        # atol for the impacts that are zero up to rounding errors (those on
        # the dates observed in both datasets, and the detailed impacts of
        # grouped revisions)
        for attr in attrs:
            assert_allclose(get(news, attr)[variable],
                            get(full, attr)[variable], rtol=1e-10,
                            atol=1e-13)
            # The design matrix has an exact zero, so there is no impact
            assert_allclose(get(news, attr)[other], 0, atol=1e-15)
        # The impacts table leaves out the rows without impacts, those of the
        # other variable; the estimates are those of all states
        impacts = news.impacts
        assert_equal(impacts.index.get_level_values(1).unique().tolist(),
                     [variable])
        desired = full.impacts.xs(variable, level=1, drop_level=False)
        assert_allclose(impacts[columns], desired[columns], rtol=1e-10,
                        atol=1e-13)
        for column in ["estimate (prev)", "estimate (new)"]:
            assert_allclose(impacts[column], desired[column], rtol=1e-12)

    # Both states give the impacts without state_index
    news = var_news_state_index(
        state_index=[1, 0], revisions_details_start=revisions_details_start)
    assert_allclose(news.impacts, full.impacts, rtol=1e-10, atol=1e-13)


def test_state_index_revision_impacts_time_varying_design():
    # With the regression coefficient as a state, the design matrix [1, x_t]
    # varies over time. The revisions of two values change both the
    # smoothed AR state and the smoothed coefficient, and the impacts
    # through each state add up to the impacts without state_index. The
    # last value is new, so that the impacts include a period after the
    # previous sample.
    rng = np.random.default_rng(5237)
    nobs = 40
    exog = rng.standard_normal(nobs)
    endog = 0.8 * exog + rng.standard_normal(nobs)
    previous = endog[:-1].copy()
    previous[30:32] += [1.5, -2.0]
    params = [0.5, 1.0]

    def smooth(y):
        mod = sarimax.SARIMAX(y, exog=exog[:len(y)], order=(1, 0, 0),
                              mle_regression=False)
        return mod.smooth(params)

    res_prev = smooth(previous)
    res = smooth(endog)

    def news(state_index, revisions_details_start=False):
        return res.news(res_prev, comparison_type="previous", start=28,
                        end=nobs - 1, state_index=state_index,
                        revisions_details_start=revisions_details_start)

    desired = news(None).revision_impacts.iloc[:, 0].to_numpy()
    impacts = [news([i]).revision_impacts.iloc[:, 0].to_numpy()
               for i in range(2)]
    # Both states have impacts, so that the test discriminates
    assert_(np.all(np.abs(impacts[0]) > 1e-3))
    assert_(np.all(np.abs(impacts[1]) > 1e-3))
    # At the observed dates other than the revised ones, the total impact is
    # zero, and the impacts through the two states cancel (atol for the
    # rounding errors of the impacts, which are up to 2 in absolute value)
    assert_allclose(impacts[0] + impacts[1], desired, rtol=1e-8,
                    atol=1e-12)
    # The detailed impacts (computed from the smoothed state weights) agree
    for i in range(2):
        detailed = news([i], revisions_details_start=True)
        assert_allclose(detailed.revision_detailed_impacts.iloc[:, 0],
                        impacts[i], rtol=1e-8)
        assert_allclose(detailed.revision_impacts.iloc[:, 0], impacts[i],
                        rtol=1e-8)
