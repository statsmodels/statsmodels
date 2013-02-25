from patsy import dmatrix
import pandas as pd
from statsmodels.api import OLS
from statsmodels.api import stats
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency


def _model2dataframe(model_endog, model_exog, model_type=OLS, **kwargs):
    """return a series containing the summary of a linear model

    All the exceding parameters will be redirected to the linear model
    """
    # create the linear model and perform the fit
    model_result = model_type(model_endog, model_exog, **kwargs).fit()
    # keeps track of some global statistics
    statistics = pd.Series({'r2': model_result.rsquared,
                  'adj_r2': model_result.rsquared_adj})
    # put them togher with the result for each term
    result_df = pd.DataFrame({'params': model_result.params,
                              'pvals': model_result.pvalues,
                              'std': model_result.bse,
                              'statistics': statistics})
    # add the complexive results for f-value and the total p-value
    fisher_df = pd.DataFrame({'params': {'_f_test': model_result.fvalue},
                              'pvals': {'_f_test': model_result.f_pvalue}})
    # merge them and unstack to obtain a hierarchically indexed series
    res_series = pd.concat([result_df, fisher_df]).unstack()
    return res_series.dropna()


def multiOLS(model, dataframe, column_list=None, model_type=OLS,
    method='fdr_bh', alpha=0.05, **kwargs):
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
    column_list : list of strings, optional
        Names of the columns to analyze with the model.
        If None (Default) it will perform the function on all the
        eligible columns (numerical type and not in the model definition)
    model_type : model class, optional
        The type of model to be used. The default is the linear model.
        Can be any linear model (OLS, WLS, GLS, etc..)
    method: string, optional
        the method used to perform the pvalue correction for multiple testing.
        default is the Benjamini/Hochberg, other available methods are:

            `bonferroni` : one-step correction
            `sidak` : on-step correction
            `holm-sidak` :
            `holm` :
            `simes-hochberg` :
            `hommel` :
            `fdr_bh` : Benjamini/Hochberg
            `fdr_by` : Benjamini/Yekutieli

    alpha: float, optional
        the significance level used for the pvalue correction (default 0.05)

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

    Even a single column name can be given without enclosing it in a list
    >>> multiOLS('GNP + 0', df, 'GNPDEFL')
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
    # as the model will use always the same endogenous variables
    # we can create them once and reuse
    model_exog = dmatrix(model, data=dataframe, return_type="dataframe")
    for col_name in column_list:
        # it will try to interpret the column name as a valid dataframe
        # index as it can be several times faster. If it fails it
        # interpret it as a patsy formula (for example for centering)
        try:
            model_endog = dataframe[col_name]
        except KeyError:
            model_endog = dmatrix(col_name + ' + 0', data=dataframe)
        # retrieve the result and store them
        res = _model2dataframe(model_endog, model_exog, model_type, **kwargs)
        col_results[col_name] = res
    # mangle them togheter and sort by complexive p-value
    summary = pd.DataFrame(col_results)
    # order by the p-value: the most useful model first!
    summary = summary.T.sort([('pvals', '_f_test')])
    summary.index.name = 'endogenous vars'
    # implementing the pvalue correction method
    smt = stats.multipletests
    for (key1, key2) in summary:
        if key1 != 'pvals':
            continue
        p_values = summary[key1, key2]
        corrected = smt(p_values, method=method, alpha=alpha)[1]
        # extend the dataframe of results with the column
        # of the corrected p_values
        summary['adj_' + key1, key2] = corrected
    return summary


def _test_group(pvalues, group, alpha=0.05):
    """test if the objects in the group are different from the general set.

    The test is performed on the pvalues set (ad a pandas series) over
    the group specified via a fisher exact test.
    """
    totals = len(pvalues)
    total_significant = np.sum(pvalues < alpha)
    cross_index = [c for c in group if c in pvalues.index]
    # how many are significant and not in the group
    group_total = len(cross_index)
    group_sign = len([c for c in cross_index if pvalues[c] < alpha])
    group_nonsign = group_total - group_sign
    # how many are significant and not outside the group
    extern_sign = total_significant - group_sign
    extern_nonsign = totals - total_significant - group_nonsign
    # make the fisher test
    test = fisher_exact
    table = [[extern_nonsign, extern_sign], [group_nonsign, group_sign]]
    pvalue = test(np.array(table))[1]
    # is the group more represented or less?
    increase = (group_sign / group_total) > (total_significant / totals)
    return pvalue, increase


def multigroup(pvals, groups, alpha=0.05):
    """Test if the groups given are differently significant than the rest.

    For each group test with an exact fisher test if the fraction of
    significatively functional models if different from the one expected
    from the general fraction.

    Parameters
    ----------
    pvals: pandas series
        the pvalus of the variables under analysis
    groups: dict of list
        the name of each category of variables under exam.
        each one is a list of the variables included
    alpha: float
        the significance level for the analysis

    Returns
    -------
    result_df: pandas dataframe
        for each group returns:

            pvals - the fisher p value of the test
            adj_pvals - the adjusted pvals
            increase - if the group if described better than expected or worse

    Notes
    -----
    This test allow to see if a category of variables is generally better
    suited to be described for the model. For example to see if a predictor
    gives more information on demographic or economical parameters,
    by creating two groups containing the endogenous variables of each
    category.

    This function is conceived for medical dataset with a lot of variables
    that can be easily grouped into functional groups. This is because
    The significativity of a group require a rather large number of
    composing elements.

    Examples
    --------
    A toy example on a real dataset, the Guerry dataset from R
    >>> url = "http://vincentarelbundock.github.com/"
    >>> url = url + "Rdatasets/csv/HistData/Guerry.csv"
    >>> df = pd.read_csv(url, index_col='dept')

    evaluate the relationship between the variuos paramenters whith the Wealth
    >>> pvals = multiOLS('Wealth', df)['adj_pvals', '_f_test']

    define the groups
    >>> groups['crime'] = ['Crime_prop', 'Infanticide',
    ...     'Crime_parents', 'Desertion', 'Crime_pers']
    >>> groups['religion'] = ['Donation_clergy', 'Clergy', 'Donations']
    >>> groups['wealth'] = ['Commerce', 'Lottery', 'Instruction', 'Literacy']

    do the analysis of the significativity
    >>> multigroup(pvals, groups)
    """
    results = {'pvals': {}, 'increase': {}}
    for group_name, group_list in groups.iteritems():
        res = _test_group(pvals, group_list, alpha=alpha)
        results['pvals'][group_name] = res[0]
        results['increase'][group_name] = res[1]
    result_df = pd.DataFrame(results).sort('pvals')
    smt = stats.multipletests
    corrected = smt(result_df['pvals'], method='fdr_bh')[1]
    result_df['adj_pvals'] = corrected
    return result_df
