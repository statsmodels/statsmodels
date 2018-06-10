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
    return exog


def predict_lambda(endog, lambda_range, nobs, d, seasonal_periods, offset=True):

    for val in endog:
        if val < 0:
            raise ValueError('Negative values hould not be present in endog')

    exog = create_exog(nobs, d=d, seasonal_periods=seasonal_periods)
    exog = np.array(exog)
    exog = exog.transpose()
    if offset:
        endog = endog + np.min(endog)/2

    y_dot = gmean(endog)

    min_ssr = np.inf
    for lambda_val in range(lambda_range[0], lambda_range[1]+1):
        if lambda_val == 0:
            v = y_dot * np.log(endog)
        else:
            v = ((endog**lambda_val) - 1)/(lambda_val * (y_dot**lambda_val-1))
        model = sm.OLS(v[:nobs], exog).fit()
        predictions = model.predict(exog)
    #     print(model.ssr)
        if model.ssr < min_ssr:
            min_ssr = model.ssr
            lam = lambda_val
    return lam
