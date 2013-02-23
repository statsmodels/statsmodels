import patsy
import pandas as pd
from statsmodels.formula.api import ols
import numpy as np


def _model2dataframe(colname, model, dataframe, model_type=ols, **kwargs):
    """return a series containing the summary of a linear model

    The linear model is defined on a pandas dataframe via the formula
    syntax, and take separately the columns name and the model description

    All the exceding parameters will be redirected to the linear model
    """
    # string patching the model
    temp_model = colname + ' ~ ' + model
    # create the linear model and perform the fit
    model_result = model_type(temp_model, data=dataframe, **kwargs).fit()
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


def multiOLS(model, dataframe, column_list=None, model_type=ols, **kwargs):
    """apply a linear model to several endogenous variables on a dataframe

    Take a linear model definition via formula and a dataframe that will be
    the environment of the model, and apply the linear model to a subset
    (or all) of the columns of the dataframe. It will return a dataframe
    with part of the information from the linear model summary.

    Parameters
    ----------
    model : string
        formula description of the model
    dataframe : pandas.dataframe
        dataframe where the model will be evaluated
    column_list : list of strings
        Names of the columns to analyze with the model.
        If None (Default) it will perform the function on all the
        eligible columns (numerical type and not in the model definition)
    model_type : model class
        The type of model to be used. The default is the linear model.
        Should be one of the model defined in the formula api.

    all the other parameters will be directed to the model creation.

    Returns
    -------
    summary : pandas.DataFrame
        a dataframe containing an extract from the summary of the model
        obtained for each columns. It will give the model complexive f test
        result and p-value, and the regression value and standard deviarion
        for each of the regressors

    Notes
    -----
    The main application of this function is on system biology to perform
    a linear model testing of a lot of different parameters, like the
    different genetic expression of several genes.

    There is no automatic correction of the p-values.

    See Also
    --------
    statsmodels.stats.multitest
        contains several functions to perform the multiple p-value correction

    Examples
    --------
    Using the longley data as dataframe example

    >>> import statsmodels.api as sm
    >>> data = sm.datasets.longley.load_pandas()
    >>> df = data.exog
    >>> df['TOTEMP'] = data.endog

    This will perform the specified linear model on all the
    other columns of the dataframe
    >>> multiOLS('GNP + 1', df)

    This select only a certain subset of the columns
    >>> multiOLS('GNP + 0', df, ['GNPDEFL', 'TOTEMP', 'POP'])

    It is possible to specify a trasformation also on the target column,
    conforming to the patsy formula specification
    >>> multiOLS('GNP + 0', df, ['I(GNPDEFL**2)', 'center(TOTEMP)'])

    As the keywords are reported to the linear model, is possible to specify
    for example the subset of the dataframe on which perform the analysis
    >> multiOLS('GNP + 1', df, subset=df.GNPDEFL > 90)

    """
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
        res = _model2dataframe(col_name, model, dataframe, **kwargs)
        col_results[col_name] = res
    #mangle them togheter and sort by complexive p-value
    summary = pd.DataFrame(col_results)
    summary = total.T.sort([('pvals', '_f_test')])
    summary.index.name = 'endogenous vars'
    return summary

if __name__ == '__main__':
    import statsmodels.api as sm

    data = sm.datasets.longley.load_pandas()
    df = data.exog
    df['TOTEMP'] = data.endog

    print multiOLS('GNP + 0', df, ['GNPDEFL', 'TOTEMP', 'POP'])
    print
    print multiOLS('GNP + 1', df)
    print
    print multiOLS('GNP + 0', df, ['I(GNPDEFL**2)', 'center(TOTEMP)'])