import pandas as pd
import statsmodels.api as sm
import numpy as np


def backward_elimination(x, y, sl):
    x = pd.DataFrame(np.append(np.ones((x.shape[0], 1)).astype(int), x, axis=1))
    rSqData = pd.DataFrame(r_sq_elimination(x, y))
    finalModel = p_elimination(rSqData, y, sl)
    return finalModel


def r_sq_elimination(x, y):
    x = pd.DataFrame(x)
    ols_regeressor = sm.OLS(exog=x, endog=y).fit()
    bestValue = float("{0:.4f}".format(ols_regeressor.rsquared_adj))
    noOfColumn = x.shape[1]
    bestModel = pd.DataFrame(x)
    foundNew = False
    for col in x.columns:
        temp = x.drop(col, axis=1)
        ols_regeressor = sm.OLS(endog=y, exog=temp).fit()
        rValue = float("{0:.4f}".format(ols_regeressor.rsquared_adj))
        if bestValue < rValue:
            bestValue = rValue
            bestModel = temp
            foundNew = True

    if foundNew == True:
        bestModel = r_sq_elimination(bestModel, y)
        return bestModel
    return bestModel


def p_elimination(x, y, sl):
    x = pd.DataFrame(x)
    noCols = x.shape[1]
    for i in range(0, noCols):
        ols_regeressor = sm.OLS(exog=x, endog=y).fit()
        pValues = ols_regeressor.pvalues
        if max(pValues) > sl:
            x = x.drop(np.argmax(pValues), axis=1)
        else:
            break
    return x
