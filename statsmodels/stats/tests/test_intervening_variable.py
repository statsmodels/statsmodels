import numpy as np
import statsmodels.api as sm
import os
from statsmodels.stats.intervening_variable import InterveningVariable
import pandas as pd
from numpy.testing import assert_allclose


# Compare Sobel indirect effect and confidence intervals to R package
# powerMediation
powerMediation_sobel_interval = pd.Series(
    {"lower ci bound": 0.007733533, "upper ci bound": 0.2976837}
)

# Compare bootstrap indirect effect and confidence intervals to R packages
# MBESS and lavaan
MBESS_bootstrap_interval = pd.Series(
    {"lower ci bound": 0.0571844, "upper ci bound": 0.28977}
)

MBESS_indirect_proportion = pd.Series(
    {"value": 0.42378, "lower ci bound": 0.1118, "upper ci bound": 1.3645}
)


def test_sobel():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(cur_dir, "results", "mackinnon2008.csv"))
    outcome_model = sm.OLS.from_formula("consume ~ room_temp + thirst", data)
    mediation_model = sm.OLS.from_formula("thirst ~ room_temp", data)
    results = (
        InterveningVariable(outcome_model, mediation_model, "room_temp")
        .fit()
        .summary()
    )

    sobel_interval = results.loc[
        "sobel indirect effect", ["lower ci bound", "upper ci bound"]
    ]
    # adjusting tolerance because of sqrt computation
    assert_allclose(sobel_interval, powerMediation_sobel_interval, rtol=0.03)


def test_bootstrap_interval():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(cur_dir, "results", "mackinnon2008.csv"))
    outcome_model = sm.OLS.from_formula("consume ~ room_temp + thirst", data)
    mediation_model = sm.OLS.from_formula("thirst ~ room_temp", data)
    np.random.seed(42)
    # increase number of n_reps so that bootstrap models converge to each other
    results = (
        InterveningVariable(outcome_model, mediation_model, "room_temp")
        .fit(n_reps=10000)
        .summary()
    )

    bootstrap_values = results.loc[
        "bootstrap indirect effect", ["lower ci bound", "upper ci bound"]
    ]
    # bootstrap values vary somewhat widely, even among the different R
    # implementations
    assert_allclose(bootstrap_values, MBESS_bootstrap_interval, rtol=0.13)


def test_bootstrap_proportion():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(cur_dir, "results", "mackinnon2008.csv"))
    outcome_model = sm.OLS.from_formula("consume ~ room_temp + thirst", data)
    mediation_model = sm.OLS.from_formula("thirst ~ room_temp", data)
    np.random.seed(42)
    # increase number of n_reps so that bootstrap models converge to each other
    results = (
        InterveningVariable(outcome_model, mediation_model, "room_temp")
        .fit(n_reps=10000)
        .summary()
    )

    bootstrap_proportion_values = results.loc[
        "bootstrap indirect/total",
        ["value", "lower ci bound", "upper ci bound"],
    ]
    # bootstrap values vary somewhat widely, even among the different R
    # implementations. atol = 0.3 means a difference of 3% in proportion
    assert_allclose(
        bootstrap_proportion_values, MBESS_indirect_proportion, atol=0.03
    )
