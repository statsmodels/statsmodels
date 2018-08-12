
if __name__ == '__main__':
    import statsmodels.api as sm
    from pandas import DataFrame
    data = sm.datasets.longley.load()
    df = DataFrame(data.exog, columns=data.exog_name)
    y = data.endog
    # data.exog = sm.add_constant(data.exog)
    df['intercept'] = 1.
    olsresult = sm.OLS(y, df).fit()
    rlmresult = sm.RLM(y, df).fit()

    # olswrap = RegressionResultsWrapper(olsresult)
    # rlmwrap = RLMResultsWrapper(rlmresult)

    data = sm.datasets.wfs.load()
    # get offset
    offset = np.log(data.exog[:, -1])
    exog = data.exog[:, :-1]

    # convert dur to dummy
    exog = sm.tools.categorical(exog, col=0, drop=True)
    # drop reference category
    # convert res to dummy
    exog = sm.tools.categorical(exog, col=0, drop=True)
    # convert edu to dummy
    exog = sm.tools.categorical(exog, col=0, drop=True)
    # drop reference categories and add intercept
    exog = sm.add_constant(exog[:, [1, 2, 3, 4, 5, 7, 8, 10, 11, 12]], prepend=False)

    endog = np.round(data.endog)
    mod = sm.GLM(endog, exog, family=sm.families.Poisson()).fit()
    # glmwrap = GLMResultsWrapper(mod)
