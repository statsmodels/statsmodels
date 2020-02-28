# testing environment for holtwinters simulation

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# matplotlib.use("GTK3Agg")

# austourists dataset from fpp2 package
# https://cran.r-project.org/web/packages/fpp2/index.html
data = [
    30.05251,
    19.14850,
    25.31769,
    27.59144,
    32.07646,
    23.48796,
    28.47594,
    35.12375,
    36.83848,
    25.00702,
    30.72223,
    28.69376,
    36.64099,
    23.82461,
    29.31168,
    31.77031,
    35.17788,
    19.77524,
    29.60175,
    34.53884,
    41.27360,
    26.65586,
    28.27986,
    35.19115,
    42.20566,
    24.64917,
    32.66734,
    37.25735,
    45.24246,
    29.35048,
    36.34421,
    41.78208,
    49.27660,
    31.27540,
    37.85063,
    38.83704,
    51.23690,
    31.83855,
    41.32342,
    42.79900,
    55.70836,
    33.40714,
    42.31664,
    45.15712,
    59.57608,
    34.83733,
    44.84168,
    46.97125,
    60.01903,
    38.37118,
    46.97586,
    50.73380,
    61.64687,
    39.29957,
    52.67121,
    54.33232,
    66.83436,
    40.87119,
    51.82854,
    57.49191,
    65.25147,
    43.06121,
    54.76076,
    59.83447,
    73.25703,
    47.69662,
    61.09777,
    66.05576,
]
index = pd.date_range("1999-03-01", "2015-12-01", freq="3MS")
austourists = pd.Series(data, index)


# ETS(M, A, M)
es = ExponentialSmoothing(
    austourists, seasonal_periods=4, trend="add",seasonal="mul"
)

fit = es.fit(smoothing_level=0.3156, smoothing_slope=1e-4, smoothing_seasonal=1e-4)
print(fit.summary())

forecast = fit.predict(start=0)
austourists.plot()
forecast.plot(color='r')
plt.show()
