%matplotlib inline
import matplotlib.pyplot as plt
import pandas
import statsmodels.api as sm

arsenic = [
    3.2, 2.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 1.7, 1.5, 1.0, 1.0, 1.0, 1.0,
    0.9, 0.9, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5
]

censored = [
    False, False, True, True, True, True, True,
    True, True, True, False, False, True, True,
    True, True, False, True, False, False, False,
    False, False, False
]

data = pandas.DataFrame(dict(res=arsenic, cen=censored))

# with a dataframe:
ros = sm.stats.RobustROSEstimator(data=data, result='res', censorship='cen')
ros.estimate()

fig, ax = plt.subplots()
ros.plot(ax=ax, show_raw=True, ylog=False)

