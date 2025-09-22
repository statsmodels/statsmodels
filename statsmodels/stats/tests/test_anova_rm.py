from statsmodels.compat.pandas import assert_frame_equal

import os

from numpy.testing import (
    assert_array_almost_equal,
    assert_equal,
)
import pandas as pd
import pytest

from statsmodels.stats.anova import AnovaRM

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ANOVA_RM_DATA = os.path.join(CURRENT_PATH, "results", "anova-rm-test-data.csv")

data = pd.read_csv(ANOVA_RM_DATA)
data["DV"] = data["DV"].astype("int")


def test_single_factor_repeated_measures_anova():
    """
    Testing single factor repeated measures anova
    Results reproduces R `ezANOVA` function from library ez
    """
    df = AnovaRM(data.iloc[:16, :], "DV", "id", within=["B"]).fit()
    a = [[1, 7, 22.4, 0.002125452]]
    assert_array_almost_equal(df.anova_table.iloc[:, [1, 2, 0, 3]].values, a, decimal=5)


def test_two_factors_repeated_measures_anova():
    """
    Testing two factors repeated measures anova
    Results reproduces R `ezANOVA` function from library ez
    """
    df = AnovaRM(data.iloc[:48, :], "DV", "id", within=["A", "B"]).fit()
    a = [
        [1, 7, 40.14159, 3.905263e-04],
        [2, 14, 29.21739, 1.007549e-05],
        [2, 14, 17.10545, 1.741322e-04],
    ]
    assert_array_almost_equal(df.anova_table.iloc[:, [1, 2, 0, 3]].values, a, decimal=5)


def test_three_factors_repeated_measures_anova():
    """
    Testing three factors repeated measures anova
    Results reproduces R `ezANOVA` function from library ez
    """
    df = AnovaRM(data, "DV", "id", within=["A", "B", "D"]).fit()
    a = [
        [1, 7, 8.7650709, 0.021087505],
        [2, 14, 8.4985785, 0.003833921],
        [1, 7, 20.5076546, 0.002704428],
        [2, 14, 0.8457797, 0.450021759],
        [1, 7, 21.7593382, 0.002301792],
        [2, 14, 6.2416695, 0.011536846],
        [2, 14, 5.4253359, 0.018010647],
    ]
    assert_array_almost_equal(df.anova_table.iloc[:, [1, 2, 0, 3]].values, a, decimal=5)


def test_repeated_measures_invalid_factor_name():
    """
    Test with a factor name of 'C', which conflicts with patsy.
    """
    with pytest.raises(ValueError):
        AnovaRM(data.iloc[:16, :], "DV", "id", within=["C"])


def test_repeated_measures_collinearity():
    data1 = data.iloc[:48, :].copy()
    data1["E"] = data1["A"]
    with pytest.raises(ValueError):
        AnovaRM(data1, "DV", "id", within=["A", "E"])


def test_repeated_measures_unbalanced_data():
    with pytest.raises(ValueError):
        AnovaRM(data.iloc[1:48, :], "DV", "id", within=["A", "B"])


def test_repeated_measures_aggregation():
    df1 = AnovaRM(data, "DV", "id", within=["A", "B", "D"]).fit()
    double_data = pd.concat([data, data], axis=0)
    df2 = AnovaRM(
        double_data, "DV", "id", within=["A", "B", "D"], aggregate_func=pd.Series.mean
    ).fit()

    assert_frame_equal(df1.anova_table, df2.anova_table)


def test_repeated_measures_aggregation_one_subject_duplicated():
    df1 = AnovaRM(data, "DV", "id", within=["A", "B", "D"]).fit()
    data2 = pd.concat([data, data.loc[data["id"] == "1", :]], axis=0)
    data2 = data2.reset_index()
    df2 = AnovaRM(
        data2, "DV", "id", within=["A", "B", "D"], aggregate_func=pd.Series.mean
    ).fit()

    assert_frame_equal(df1.anova_table, df2.anova_table)


def test_repeated_measures_aggregate_func():
    double_data = pd.concat([data, data], axis=0)
    with pytest.raises(ValueError):
        AnovaRM(double_data, "DV", "id", within=["A", "B", "D"])

    m1 = AnovaRM(
        double_data, "DV", "id", within=["A", "B", "D"], aggregate_func=pd.Series.mean
    )
    m2 = AnovaRM(
        double_data, "DV", "id", within=["A", "B", "D"], aggregate_func=pd.Series.median
    )

    with pytest.raises(AssertionError):
        assert_equal(m1.aggregate_func, m2.aggregate_func)
    assert_frame_equal(m1.fit().anova_table, m2.fit().anova_table)


def test_repeated_measures_aggregate_func_mean():
    double_data = pd.concat([data, data], axis=0)
    m1 = AnovaRM(
        double_data, "DV", "id", within=["A", "B", "D"], aggregate_func=pd.Series.mean
    )

    m2 = AnovaRM(double_data, "DV", "id", within=["A", "B", "D"], aggregate_func="mean")

    assert_equal(m1.aggregate_func, m2.aggregate_func)


def test_repeated_measures_aggregate_compare_with_ezANOVA():
    # Results should reproduces those from R's `ezANOVA` (library ez).
    ez = pd.DataFrame(
        {
            "F Value": [
                8.7650709,
                8.4985785,
                20.5076546,
                0.8457797,
                21.7593382,
                6.2416695,
                5.4253359,
            ],
            "Num DF": [1, 2, 1, 2, 1, 2, 2],
            "Den DF": [7, 14, 7, 14, 7, 14, 14],
            "Pr > F": [
                0.021087505,
                0.003833921,
                0.002704428,
                0.450021759,
                0.002301792,
                0.011536846,
                0.018010647,
            ],
        },
        index=pd.Index(["A", "B", "D", "A:B", "A:D", "B:D", "A:B:D"]),
    )
    ez = ez[["F Value", "Num DF", "Den DF", "Pr > F"]]

    double_data = pd.concat([data, data], axis=0)
    df = (
        AnovaRM(
            double_data,
            "DV",
            "id",
            within=["A", "B", "D"],
            aggregate_func=pd.Series.mean,
        )
        .fit()
        .anova_table
    )

    assert_frame_equal(ez, df, check_dtype=False)
