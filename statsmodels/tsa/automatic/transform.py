from scipy.stats.mstats import gmean
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels import datasets


def create_exog(nobs, d=0, seasonal_periods=1):
    t = np.arange(1, nobs+1)
    exog = []
    for i in range(d+1):
        exog.append(t**i)
    if seasonal_periods > 1:
        season = np.zeros((seasonal_periods-1, nobs))
        for s in range(0, seasonal_periods):
            for i in range(1, seasonal_periods):
                for j in range(nobs):
                    if j % seasonal_periods == i-1:
                        season[i-1][j] = 1
        for series in season:
            exog.append(series)
    exog = np.array(exog)
    exog = exog.transpose()
    return exog


def predict_lambda(endog, lambda_range, d, seasonal_periods, offset=True, gridsize=11):
    nobs = len(endog)
    if np.any(endog < 0):
        raise ValueError('Negative values hould not be present in endog')

    exog = create_exog(nobs, d=d, seasonal_periods=seasonal_periods)
    if offset:
        endog = endog + np.min(endog)/2

    y_gmean = gmean(endog)

    min_ssr = np.inf
    for lambda_val in np.linspace(lambda_range[0], lambda_range[1], num=gridsize):
        if lambda_val == 0:
            y_transform = y_gmean * np.log(endog)
        else:
            y_transform = ((endog ** lambda_val) - 1) / (lambda_val * (y_gmean ** (lambda_val - 1)))
        model = sm.OLS(y_transform[:nobs], exog).fit()
    #     print(model.ssr)
        if model.ssr < min_ssr:
            min_ssr = model.ssr
            lam = lambda_val
    return lam
