import patsy
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import numpy as np


def model2dataframe(colname, model, dataframe, **kwargs):
    # string patching the model
    temp_model = colname + ' ~ ' + model
    # create the linear model and perform the fit
    model_result = ols(temp_model, data=dataframe, **kwargs).fit()
    # calculate the variance of each parameter
    # do not use the covariance as it wil be too complicate to move around
    cov_m = model_result.cov_params()
    stds = {c: np.sqrt(cov_m[c][c]) for c in cov_m}
    # keeps track of some global statistics
    statistics = {'r2': model_result.rsquared,
                  'adj_r2': model_result.rsquared_adj}
    # put them togher with the result for each term
    result_df = pd.DataFrame({'params': model_result.params,
                              'pvals': model_result.pvalues,
                              'std': stds,
                              'statistics': statistics})
    # add the complexive results for f-value and the total p-value
    fisher_df = pd.DataFrame({'params': {'_f_test': model_result.fvalue},
                              'pvals': {'_f_test': model_result.f_pvalue}})
    #merge them and unstack to obtain a hierarchically indexed series
    res_series = pd.concat([result_df, fisher_df]).unstack()
    res_series.name = colname
    return res_series.dropna()


def multimodel(model, dataframe, column_list=None, **kwargs):
    # data normalization
    # if None take all the numerical columns that aren't present in the model
    # it's not waterproof but is a good enough criterion for everyday use
    if column_list is None:
        column_list = [name for name in dataframe.columns
                      if dataframe[name].dtype != object and name not in model]
    # if it's a single string transform it in a single element list
    if isinstance(column_list, basestring):
        column_list = [column_list]
    # perform each model and retrieve the statistics
    col_results = {}
    for col_name in column_list:
        res = model2dataframe(col_name, model, dataframe, **kwargs)
        col_results[col_name] = res
    #mangle them togheter and sort by complexive p-value
    total = pd.DataFrame(col_results)
    total = total.T.sort([('pvals', '_f_test')])
    total.index.name = 'endogenous vars'
    return total

if __name__ == '__main__':
    data = sm.datasets.longley.load_pandas()
    df = data.exog
    df['TOTEMP'] = data.endog

    print multimodel('GNP + 0', df, ['GNPDEFL', 'TOTEMP', 'POP'])
    print
    print multimodel('GNP + 1', df)
    print
    print multimodel('GNP + 0', df, ['I(GNPDEFL**2)', 'center(TOTEMP)'])