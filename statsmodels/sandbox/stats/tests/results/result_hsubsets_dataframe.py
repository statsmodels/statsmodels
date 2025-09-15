import numpy as np
import pandas as pd


expected_df_alpha001 = pd.DataFrame(
    {
        "S1": [
            -0.1895218561539811,
            0.44285644726317464,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.1985,
        ],
        "S2": [
            np.nan,
            0.44285644726317464,
            0.8662790048687181,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.7019,
        ],
        "S3": [
            np.nan,
            np.nan,
            0.8662790048687181,
            1.5964711342473101,
            1.632750270531383,
            1.7108333884711162,
            np.nan,
            np.nan,
            0.0212,
        ],
        "S4": [
            np.nan,
            np.nan,
            np.nan,
            1.5964711342473101,
            1.632750270531383,
            1.7108333884711162,
            1.8993667306691449,
            np.nan,
            0.9313,
        ],
        "S5": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            2.922411879049647,
            1.0,
        ],
    }
)
expected_df_alpha001.index = [
    "Group 2",
    "Group 1",
    "Group 3",
    "Group 5",
    "Group 6",
    "Group 4",
    "Group 7",
    "Group 8",
    "min p-value",
]
expected_df_alpha001.index.name = "Group"

expected_df_alpha005 = pd.DataFrame(
    {
        "S1": [
            -0.1895218561539811,
            0.44285644726317464,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.1985,
        ],
        "S2": [
            np.nan,
            0.44285644726317464,
            0.8662790048687181,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.7019,
        ],
        "S3": [
            np.nan,
            np.nan,
            0.8662790048687181,
            1.5964711342473101,
            1.632750270531383,
            np.nan,
            np.nan,
            np.nan,
            0.0533,
        ],
        "S4": [
            np.nan,
            np.nan,
            np.nan,
            1.5964711342473101,
            1.632750270531383,
            1.7108333884711162,
            1.8993667306691449,
            np.nan,
            0.9313,
        ],
        "S5": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            2.922411879049647,
            1.0,
        ],
    }
)
expected_df_alpha005.index = [
    "Group 2",
    "Group 1",
    "Group 3",
    "Group 5",
    "Group 6",
    "Group 4",
    "Group 7",
    "Group 8",
    "min p-value",
]
expected_df_alpha005.index.name = "Group"

expected_df_alpha01 = pd.DataFrame(
    {
        "S1": [
            -0.1895218561539811,
            0.44285644726317464,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.1985,
        ],
        "S2": [
            np.nan,
            0.44285644726317464,
            0.8662790048687181,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.7019,
        ],
        "S3": [
            np.nan,
            np.nan,
            np.nan,
            1.5964711342473101,
            1.632750270531383,
            1.7108333884711162,
            1.8993667306691449,
            np.nan,
            0.9313,
        ],
        "S4": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            2.922411879049647,
            1.0,
        ],
    }
)
expected_df_alpha01.index = [
    "Group 2",
    "Group 1",
    "Group 3",
    "Group 5",
    "Group 6",
    "Group 4",
    "Group 7",
    "Group 8",
    "min p-value",
]
expected_df_alpha01.index.name = "Group"

expected_df_unbalanced_alpha005 = pd.DataFrame(
    {
        "S1": [
            0.668436023659397,
            0.738023170728835,
            0.885418477024479,
            1.113164460110977,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.924598821413583,
        ],
        "S2": [
            np.nan,
            np.nan,
            0.885418477024479,
            1.113164460110977,
            1.921209214216464,
            np.nan,
            np.nan,
            np.nan,
            0.090673607136369,
        ],
        "S3": [
            np.nan,
            np.nan,
            np.nan,
            1.113164460110977,
            1.921209214216464,
            2.124227257491577,
            2.164976646569281,
            np.nan,
            0.08097476143252,
        ],
        "S4": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            1.921209214216464,
            2.124227257491577,
            2.164976646569281,
            2.511205239821201,
            0.737364397497922,
        ],
    }
)
expected_df_unbalanced_alpha005.index = [
    "Group 2",
    "Group 1",
    "Group 4",
    "Group 3",
    "Group 5",
    "Group 7",
    "Group 6",
    "Group 8",
    "min p-value",
]
expected_df_unbalanced_alpha005.index.name = "Group"