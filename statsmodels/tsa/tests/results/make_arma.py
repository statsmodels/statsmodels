import numpy as np

from statsmodels.iolib import savetxt
from statsmodels.tsa.arima_process import arma_generate_sample

rs = np.random.RandomState(12345)

# no constant
y_arma11 = arma_generate_sample(
    [1.0, -0.75], [1.0, 0.35], nsample=250, distrvs=rs.standard_normal
)
y_arma14 = arma_generate_sample(
    [1.0, -0.75], [1.0, 0.35, -0.75, 0.1, 0.35], nsample=250, distrvs=rs.standard_normal
)
y_arma41 = arma_generate_sample(
    [1.0, -0.75, 0.25, 0.25, -0.75],
    [1.0, 0.35],
    nsample=250,
    distrvs=rs.standard_normal,
)
y_arma22 = arma_generate_sample(
    [1.0, -0.75, 0.45], [1.0, 0.35, -0.9], nsample=250, distrvs=rs.standard_normal
)

y_arma50 = arma_generate_sample(
    [1.0, -0.75, 0.35, -0.3, -0.2, 0.1], [1.0], nsample=250, distrvs=rs.standard_normal
)

y_arma02 = arma_generate_sample(
    [1.0], [1.0, 0.35, -0.75], nsample=250, distrvs=rs.standard_normal
)


# constant
constant = 4.5
y_arma11c = (
    arma_generate_sample(
        [1.0, -0.75], [1.0, 0.35], nsample=250, distrvs=rs.standard_normal
    )
    + constant
)
y_arma14c = (
    arma_generate_sample(
        [1.0, -0.75],
        [1.0, 0.35, -0.75, 0.1, 0.35],
        nsample=250,
        distrvs=rs.standard_normal,
    )
    + constant
)
y_arma41c = (
    arma_generate_sample(
        [1.0, -0.75, 0.25, 0.25, -0.75],
        [1.0, 0.35],
        nsample=250,
        distrvs=rs.standard_normal,
    )
    + constant
)
y_arma22c = (
    arma_generate_sample(
        [1.0, -0.75, 0.45], [1.0, 0.35, -0.9], nsample=250, distrvs=rs.standard_normal
    )
    + constant
)

y_arma50c = (
    arma_generate_sample(
        [1.0, -0.75, 0.35, -0.3, -0.2, 0.1],
        [1.0],
        nsample=250,
        distrvs=rs.standard_normal,
    )
    + constant
)

y_arma02c = (
    arma_generate_sample(
        [1.0], [1.0, 0.35, -0.75], nsample=250, distrvs=rs.standard_normal
    )
    + constant
)

savetxt(
    "y_arma_data.csv",
    np.column_stack(
        (
            y_arma11,
            y_arma14,
            y_arma41,
            y_arma22,
            y_arma50,
            y_arma02,
            y_arma11c,
            y_arma14c,
            y_arma41c,
            y_arma22c,
            y_arma50c,
            y_arma02c,
        )
    ),
    names=[
        "y_arma11",
        "y_arma14",
        "y_arma41",
        "y_arma22",
        "y_arma50",
        "y_arma02",
        "y_arma11c",
        "y_arma14c",
        "y_arma41c",
        "y_arma22c",
        "y_arma50c",
        "y_arma02c",
    ],
    delimiter=",",
)
