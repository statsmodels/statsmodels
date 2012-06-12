import numpy as np
from scipy import stats
from pandas import DataFrame, Index
from statsmodels.formula.formulatools import (_remove_intercept_charlton,
                                              _assign)

#TODO: remove when possible
def has_intercept(column_info):
    from charlton.desc import INTERCEPT
    return INTERCEPT in column_info.term_to_columns

#NOTE: these need to take into account weights !

def anova_single(model, **kwargs):
    """
    ANOVA table for one fitted linear model.

    Parameters
    ----------
    model : fitted linear model results instance
        A fitted linear model
    typ : int or str {1,2,3} or {"I","II","III"}
        Type of sum of squares to use.

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".

    Notes
    -----
    Use of this function is discouraged. Use anova_lm instead.
    """
    test = kwargs.get("test", "F")
    scale = kwargs.get("scale", None)
    typ = kwargs.get("typ", 1)

    endog = model.model.endog
    exog = model.model.exog
    nobs = exog.shape[0]

    response_name = model.model.endog_names
    column_info = model.model._data._orig_exog.column_info
    exog_names = model.model.exog_names
    n_rows = (len(column_info.term_to_columns) - has_intercept(column_info)
              + 1) #+1 for resids

    pr_test = "PR(>%s)" % test
    names = ['df', 'sum_sq', 'mean_sq', test, pr_test]

    table = DataFrame(np.empty((n_rows, 5)), columns = names)

    if typ in [1,"I"]:
        return anova1_lm_single(model, endog, exog, nobs, column_info, table,
                                n_rows, test, pr_test)
    elif typ in [2, "II"]:
        return anova2_lm_single(model, column_info, n_rows, test, pr_test)
    elif typ in [3, "III"]:
        return anova3_lm_single
    elif typ in [4, "IV"]:
        raise NotImplemented("Type IV not yet implemented")
    else:
        raise ValueError("Type %s not understood" % str(typ))

def anova1_lm_single(model, endog, exog, nobs, column_info, table, n_rows, test,
                     pr_test):
    """
    ANOVA table for one fitted linear model.

    Parameters
    ----------
    model : fitted linear model results instance
        A fitted linear model

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".

    Notes
    -----
    Use of this function is discouraged. Use anova_lm instead.
    """
    #maybe we should rethink using pinv > qr in OLS/linear models?
    q,r = np.linalg.qr(exog)
    effects = np.dot(q.T, endog)

    assign, term_names = _assign(column_info, intercept=True) #keep intercept

    arr = np.zeros((len(term_names), len(assign)))
    arr[assign, range(len(assign))] = 1
    sum_sq = np.dot(arr, effects**2)
    #NOTE: assumes intercept is first column
    intercept = has_intercept(column_info)
    sum_sq = sum_sq[intercept:]
    term_names = term_names[intercept:]

    table.index = Index(term_names + ['Residual'])
    # fill in residual
    table.ix['Residual'][['sum_sq','df', test, pr_test]] = (model.ssr,
                                                            model.df_resid,
                                                            np.nan, np.nan)
    # sort in order of the way the terms appear in Formula with resid last
    table['mean_sq'] = table['sum_sq'] / table['df']
    if test == 'F':
        table[:n_rows][test] = ((table['sum_sq']/table['df'])/
                                (model.ssr/model.df_resid))
        table[:n_rows][pr_test] = stats.f.sf(table["F"], table["df"],
                                model.df_resid)
    return table

#NOTE: the below is not agnostic about formula...
def anova2_lm_single(model, column_info, n_rows, test, pr_test):
    """
    ANOVA type II table for one fitted linear model.

    Parameters
    ----------
    model : fitted linear model results instance
        A fitted linear model

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".

    Notes
    -----
    Use of this function is discouraged. Use anova_lm instead.

    Type II
    Sum of Squares compares marginal contribution of terms. Thus, it is
    not particularly useful for models with significant interaction terms.
    """
    terms_info = column_info.term_to_columns.copy()
    terms_info = _remove_intercept_charlton(terms_info)

    names = ['sum_sq', 'df', test, pr_test]

    table = DataFrame(np.empty((n_rows, 4)), columns = names)
    cov = np.asarray(model.cov_params())
    col_order = []
    index = []
    for i, (term, cols) in enumerate(terms_info.iteritems()):
        # grab all varaibles except interaction effects that contain term
        # need two hypotheses matrices L1 is most restrictive, ie., term==0
        # L2 is everything except term==0
        L1 = range(cols[0], cols[1])
        L2 = []
        term_set = set(term.factors)
        for t,col in terms_info.iteritems(): # for the term you have
            other_set = set(t.factors)
            if term_set.issubset(other_set) and not term_set == other_set:
                # on a higher order term containing current `term`
                L1.extend(range(col[0], col[1]))
                L2.extend(range(col[0], col[1]))

        L1 = np.eye(model.model.exog.shape[1])[L1]
        L2 = np.eye(model.model.exog.shape[1])[L2]

        if L2.size:
            LVL = np.dot(np.dot(L1,cov),L2.T)
            from scipy import linalg
            orth_compl,_ = linalg.qr(LVL)
            r = L1.shape[0] - L2.shape[0]
            # L1|2
            # use the non-unique orthogonal completion since L12 is rank r
            L12 = np.dot(orth_compl[:,-r:].T, L1)
        else:
            L12 = L1
            r = L1.shape[0]
        if test == 'F':
            f = model.f_test(L12)
            table.ix[i][test] = test_value = f.fvalue
            table.ix[i][pr_test] = f.pvalue

        # need to back out SSR from f_test, not quite sure how
        table.ix[i]['df'] = r
        col_order.append(cols[0])
        index.append(term.name())

    table.index = Index(index + ['Residual'])
    table = table.ix[np.argsort(col_order + [model.model.exog.shape[1]+1])]
    # back out sum of squares from f_test
    ssr = table[test] * table['df'] * model.ssr/model.df_resid
    table['sum_sq'] = ssr
    # fill in residual
    table.ix['Residual'][['sum_sq','df', test, pr_test]] = (model.ssr,
                                                            model.df_resid,
                                                            np.nan, np.nan)

    return table

def anova_lm(*args, **kwargs):
    """
    ANOVA table for one or more fitted linear models.

    Parmeters
    ---------
    args : fitted linear model results instance
        One or more fitted linear models

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
    model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".
    typ : str or int {"I","II","III"} or {1,2,3}
        The type of ANOVA test to perform. See notes.
    robust : {None, "hc1", "hc2", "hc3", "hc4"}
        Use heteroscedasticity-corrected coefficient covariance matrix.
        If robust covariance is desired, it is recommended to use `hc3`.

    Returns
    -------
    anova : DataFrame
    A DataFrame containing.

    Notes
    -----
    Model statistics are given in the order of args. Models must have
    a formula_str attribute.

    See Also
    --------
    model_results.compare_f_test, model_results.compare_lm_test
    """
    typ = kwargs.get('typ', 1)

    ### Farm Out Single model ANOVA Type I, II, III, and IV ###

    if len(args) == 1:
        model = args[0]
        return anova_single(model, **kwargs)

    try:
        assert typ in [1,"II"]
    except:
        raise ValueError("Multiple models only supported for type I. "
                         "Got type %s" % str(typ))

    ### COMPUTE ANOVA TYPE I ###

    # if given a single model
    if len(args) == 1:
        return anova_lm_single(*args, **kwargs)

    # received multiple fitted models

    test = kwargs.get("test", "F")
    scale = kwargs.get("scale", None)
    n_models = len(args)

    model_formula = []
    pr_test = "PR(>%s)" % test
    names = ['df_resid', 'ssr', 'df_diff', 'ss_diff', test, pr_test]
    table = DataFrame(np.empty((n_models, 6)), columns = names)

    if not scale: # assume biggest model is last
        scale = args[-1].scale

    table["ssr"] = map(getattr, args, ["ssr"]*n_models)
    table["df_resid"] = map(getattr, args, ["df_resid"]*n_models)
    table.ix[1:]["df_diff"] = np.diff(map(getattr, args, ["df_model"]*n_models))
    table["ss_diff"] = -table["ssr"].diff()
    if test == "F":
        table["F"] = table["ss_diff"] / table["df_diff"] / scale
        table[pr_test] = stats.f.sf(table["F"], table["df_diff"],
                             table["df_resid"])

    return table


if __name__ == "__main__":
    import pandas
    from statsmodels.formula.api import ols
    # in R
    #library(car)
    #write.csv(Moore, "moore.csv", row.names=FALSE)
    moore = pandas.read_table('moore.csv', delimiter=",", skiprows=1,
                                names=['partner_status','conformity',
                                    'fcategory','fscore'])
    moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',
                    df=moore).fit()

    mooreB = ols('conformity ~ C(partner_status, Sum)', df=moore).fit()

    # for each term you just want to test vs the model without its
    # higher-order terms

    # using Monette-Fox slides and Marden class notes for linear algebra /
    # orthogonal complement
    # https://netfiles.uiuc.edu/jimarden/www/Classes/STAT324/

    table = anova_lm(moore_lm, typ=2)
