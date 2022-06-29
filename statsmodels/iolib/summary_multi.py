
import copy

import numpy as np
import pandas as pd

from statsmodels.iolib import summary


def pretty_conf_str(x, coef, conf_low, conf_up, fmt):
    """Pretty style of estimated value and confidence interval."""
    est = f"{x[coef]:{fmt}}"
    low = f"{x[conf_low]:{fmt}}"
    up = f"{x[conf_up]:{fmt}}"
    return f"{est} ({low}, {up})"


def add_params_summary(result, func=None, alpha=0.05, fmt=".3f"):
    """
    Add "params_summary" attribute to any type of the result.

    Parameters
    ----------
    result : Any type of result instance
        Instance to be added with "params_summary" attribute.
    func : function
        Function for transforming results.
        Ex. np.exp for odds ratios or risk ratios.
    alpha : float
        Significance level for a confidence interval.
    fmt : string
        Format for string results. Default is ".3f".
        Example result is "0.880 (0.839, 0.922)" for "str" column.

    Returns
    -------
    result
        The result instance with "params_summary" attribute.
    """
    params_summary = summary.summary_params_frame(result, alpha=alpha)
    coef = "coef"
    conf_low = f"[{alpha/2}"
    conf_up = f"{1-alpha/2}]"
    if func is not None:
        coef = "f(coef)"
        col = func(params_summary["coef"])
        params_summary.insert(4, "f(coef)", col)
        conf_names = [conf_low, conf_up]
        params_summary[conf_names] = func(params_summary[conf_names])

    params_summary["str"] = (
        params_summary
        .apply(
            lambda x: pretty_conf_str(x, coef, conf_low, conf_up, fmt),
            axis=1
            )
    )
    result.params_summary = params_summary
    return result


def simple_summary_col(
        results, accessor=None, index=None, fill_value=None, rename_index=None,
        ):
    """
    Summarize results horizontally, imputing index with fill_value, and
    rename index.

    Parameters
    ----------
    results : array_like
        An array-like object of fitting results.
    accessor : function
        Function to access each model result, which is
        summarized and displayed.
    index : array_like
        Index names to be displayed. Also, if an element of index
        is not in the result, insert a value of "fill_value" for that row.
    fill_value : array_like
        If index name in "index" is not existed, this value is used.
    rename_index : dictionary
        Dictionary for replacing index names.

    Returns
    -------
    DataFrame
        Summarized results from multiple models.
    """
    if accessor is None:
        def accessor(x):
            return x.params

    if fill_value is None:
        fill_value = np.nan

    result_table = []
    for res in results:
        if res is None:
            result_table.append(None)
            continue
        res = accessor(res)
        if not isinstance(res, pd.Series):
            res = pd.Series(res)
        result_table.append(res)
    result_df = pd.concat(result_table, axis=1)

    if index is not None:
        for ind in index:
            if ind not in result_df.index:
                result_df.loc[ind] = fill_value
        result_df = result_df.loc[index]
    if rename_index is not None:
        result_df.rename(index=rename_index, inplace=True)
    return result_df


def multi_model_summary(results, accessor=None,
                        columns=None, index=None,
                        fill_value=None, rename_index=None):
    """
    Summarize multiple model results horizontally.

    Summarize multiple model results horizontally by accessing
    common structure. If you want to customly layout results,
    see "mosaic_model_summary".

    Parameters
    ----------
    results : array_like
        An array-like object of fitting results.
    accessor : function
        Function to access each model result, which is
        summarized and displayed.
    columns : array_like
        Column names for each model.
    index : array_like
        Index names to be displayed. Also, if an element of index
        is not in the result, insert a value of "fill_value" for that row.
    fill_value : array_like
        If index name in "index" is not existed, this value is used.
    rename_index : dictionary
        Dictionary for replacing index names.

    Returns
    -------
    DataFrame
        Summarized results from multiple models.

    Example
    -------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> import statsmodels.formula.api as smf
    >>> df = sm.datasets.anes96.load_pandas().data
    >>>
    >>> inc = "income"
    >>> df[inc] = (df[inc]
    >>>           .mask(df[inc] <= 24, "17-24")
    >>>           .mask(df[inc] <= 16, "9-16")
    >>>           .mask(df[inc] <= 8, "0-8")
    >>>          )
    >>>
    >>> res1 = smf.glm(formula="vote ~ logpopul",
    >>>                data=df, family = sm.families.Binomial()).fit()
    >>> res2 = smf.glm(formula="vote ~ logpopul + income",
    >>>                data=df, family=sm.families.Binomial()).fit()
    >>>
    >>> results = [res1, res2]
    >>> results = [sm.add_params_summary(res, func=np.exp, alpha=0.05)
    >>>            for res in results]
    >>>
    >>> sm.multi_model_summary(
    >>>     results,
    >>>     accessor=lambda x : x.params_summary["str"],
    >>>     columns=None,
    >>>     index = ["logpopul", 'income[T.0-8]',
    >>>              'income[T.9-16]', 'income[T.17-24]'],
    >>>     fill_value="Ref.",
    >>>     )
                                Model1                Model2
    logpopul         0.895 (0.859, 0.933)  0.898 (0.861, 0.937)
    income[T.0-8]                    Ref.                  Ref.
    income[T.9-16]                    NaN  1.489 (0.930, 2.386)
    income[T.17-24]                   NaN  2.609 (1.685, 4.039)

    See Also
    --------
    statsmodels.iolib.summary_multi.mosaic_model_summary
    """
    result_df = simple_summary_col(
        results, accessor=accessor, index=index,
        fill_value=fill_value, rename_index=rename_index
        )

    if columns is None:
        columns = [f"Model{i+1}" for i in range(result_df.shape[1])]
    result_df.columns = columns
    return result_df


def mosaic_model_summary(
        results, mosaic=None, accessor=None,
        columns=None, index=None, fill_value=None, rename_index=None,
        rows=None
        ):
    """
    Custom layout of multiple model summaries.

    results : array_like
        An array-like object of fitting results.
    accessor : function
        Function to access each model result, which is
        summarized and displayed.
    columns : array_like
        Column names for each model.
    index : array_like
        Index names to be displayed. Also, if an element of index
        is not in the result, insert a value of "fill_value" for that row.
    fill_value : array_like
        If index name in "index" is not existed, this value is used.
    rename_index : dictionary
        Dictionary for replacing index names.
    rows : array_like
        Row names for multi index. See Example for usage.

    Returns
    -------
    DataFrame
        Summarized results from multiple models.

    Example
    -------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> import statsmodels.formula.api as smf
    >>> df = sm.datasets.anes96.load_pandas().data
    >>>
    >>> inc = "income"
    >>> df[inc] = (df[inc]
    >>>           .mask(df[inc] <= 24, "17-24")
    >>>           .mask(df[inc] <= 16, "9-16")
    >>>           .mask(df[inc] <= 8, "0-8")
    >>>          )
    >>>
    >>> res1 = smf.glm(formula="vote ~ logpopul",
    >>>                data=df, family=sm.families.Binomial()).fit()
    >>> res2 = smf.glm(formula="vote ~ logpopul + income",
    >>>                data=df, family=sm.families.Binomial()).fit()
    >>>
    >>> df_60 = df.loc[df["age"].le(60)]
    >>> res3 = smf.glm(formula="vote ~ logpopul",
    >>>                data=df_60, family=sm.families.Binomial()).fit()
    >>> res4 = smf.glm(formula="vote ~ logpopul + income",
    >>>                data=df_60, family=sm.families.Binomial()).fit()
    >>>
    >>> results = [res1, res2, res3, res4]
    >>> results = [sm.add_params_summary(res, func=np.exp, alpha=0.05)
    >>>            for res in results]
    >>>
    >>> mosaic = [[0,1],
    >>>           [2,3]]
    >>> sm.mosaic_model_summary(results, mosaic=mosaic,
    >>>                         accessor=lambda x: x.params_summary["str"],
    >>>                         columns=["model1", "model2"],
    >>>                         rows=["All","Age<=60"])
                                        model1                model2
    Row     index
    All     Intercept        0.930 (0.790, 1.094)  0.469 (0.310, 0.710)
            logpopul         0.895 (0.859, 0.933)  0.898 (0.861, 0.937)
            income[T.17-24]                   NaN  2.609 (1.685, 4.039)
            income[T.9-16]                    NaN  1.489 (0.930, 2.386)
    Age<=60 Intercept        0.931 (0.775, 1.118)  0.497 (0.299, 0.826)
            logpopul         0.874 (0.834, 0.916)  0.880 (0.839, 0.922)
            income[T.17-24]                   NaN  2.376 (1.402, 4.028)
            income[T.9-16]                    NaN  1.239 (0.690, 2.225)

    See Also
    --------
    statsmodels.iolib.summary_multi.multi_model_summary
    """
    result_df = simple_summary_col(
        results, accessor=accessor, index=index,
        fill_value=fill_value, rename_index=rename_index,
        )

    n_col = len(mosaic[0])
    n_row = len(mosaic)
    if columns is None:
        columns = [f"Model{i+1}" for i in range(n_col)]
    if rows is None:
        rows = [i for i in range(n_row)]

    mosaic_res = copy.deepcopy(mosaic)
    mosaic_df = [None for i in range(n_row)]
    for i, row in enumerate(mosaic):
        for j, col in enumerate(row):
            if col is None:
                mosaic_res[i][j] = pd.Series(
                    data=np.nan, index=result_df.index
                    )
            else:
                mosaic_res[i][j] = result_df.iloc[:, col]
        mosaic_df[i] = pd.concat(mosaic_res[i], axis=1)
        mosaic_df[i].columns = columns
        mosaic_df[i]["Row"] = rows[i]
    mosaic_df = (
        pd.concat(mosaic_df, axis=0)
        .reset_index()
        .set_index(["Row", "index"])
    )

    return mosaic_df
