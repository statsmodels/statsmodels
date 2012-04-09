import numpy as np
from scipy import stats
from pandas import DataFrame, Index
from charlton.desc import INTERCEPT # these are not formula independent..
                                    # what could a null term evaluate to?

#NOTE: these need to take into account weights !

def _orthogonal_complement(X, subset):
    """
    Returns the orthgonal complement to the projection matrix M0 that spans
    a `subset` of the columns in X where `subset` is a column index for X.

    The idea is that given a projection matrix Ma that spans the columns of X.
    For a null hypothesis M0 that spans a subset of X, Xa, and that is itself
    a subset of Ma. The sum of squared residuals can be obtained from Ma0, the
    set of vectors in Ma that are orthgonal to M0. Ie., it's the projectsion
    of y onto the *marginal* contribution of Ma0 = Ma - M0 is what is in Ma
    and not M0 that helps
    explain y.

    See Lecture Notes Chapter 5 here [1] for details.

    [1] https://netfiles.uiuc.edu/jimarden/www/Classes/STAT324/
    """
    return np.linalg.qr

def anova_single(model, typ, **kwargs):
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

    endog = model.model.endog
    exog = model.model.exog
    nobs = exog.shape[0]

    response_name = model.model.endog_names
    model_formula = []
    terms_info = model.model._data._orig_exog.column_info.term_to_columns
    exog_names = model.model.exog_names
    n_rows = len(terms_info) - (INTERCEPT in terms_info) + 1 # for resids

    pr_test = "PR(>%s)" % test
    names = ['df', 'sum_sq', 'mean_sq', test, pr_test]

    table = DataFrame(np.empty((n_rows, 5)), columns = names)

    if typ in [1,"I"]:
        return anova1_lm_single(model, endog, exog, nobs, terms_info, table,
                                n_rows, test, pr_test)
    elif typ in [2, "II"]:
        return anova2_lm_single(model, terms_info, table, n_rows, test,
                               pr_test)
    elif typ in [3, "III"]:
        return anova3_lm_single
    elif typ in [4, "IV"]:
        raise NotImplemented("Type IV not yet implemented")
    else:
        raise ValueError("Type %s not understood" % str(typ))

def anova1_lm_single(model, endog, exog, nobs, terms_info, table, n_rows, test,
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
    #NOTE: to get the full q,r the same as R use scipy.linalg.qr with
    # pivoting
    q,r = np.linalg.qr(exog)
    effects = np.dot(q.T,endog)

    if INTERCEPT in terms_info:
        terms_info.pop(INTERCEPT)

    index = []
    col_order = []
    for i, (term, cols) in enumerate(terms_info.iteritems()):
        table.ix[i]['sum_sq'] = np.sum(effects[cols[0]:cols[1]]**2)
        table.ix[i]['df'] = cols[1]-cols[0]
        col_order.append(cols[0])
        index.append(term.name())
    table.index = Index(index + ['Residual'])
    # fill in residual
    table.ix['Residual'][['sum_sq','df', test, pr_test]] = (model.ssr,
                                                            model.df_resid,
                                                            np.nan, np.nan)
    # sort in order of the way the terms appear in Formula with resid last
    table = table.ix[np.argsort(col_order + [exog.shape[1]+1])]
    table['mean_sq'] = table['sum_sq'] / table['df']
    if test == 'F':
        table[:n_rows][test] = ((table['sum_sq']/table['df'])/
                                (model.ssr/model.df_resid))
        table[:n_rows][pr_test] = stats.f.sf(table["F"], table["df"],
                                model.df_resid)
    return table

#NOTE: the below is not agnostic about formula...
def anova2_lm_single(model, terms_info, table, n_rows, test, pr_test):
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
    if INTERCEPT in terms_info:
        terms_info.pop(INTERCEPT)
        intercept = 1
    else:
        intercept = 0
    assign = {}
    for i, (term, cols) in enumerate(terms_info.iteritems()):
        # grab all effects except interaction effects that contain term
        subset_effect = 0 # anything without term
        term_set = set(term.factors)
        for t,col in terms_info.iteritems():
            other_set = set(t.factors)
            if term_set.issubset(other_set) and not term_set == other_set:
                continue
            elif term_set == other_set:
                main_effect = np.sum(effects[col[0]:col[1]]**2)
            subset_effect += np.sum(effects[col[0]:col[1]]**2)

        ssr_woutterm = [(col[0], col[1]) for t,col in terms_info.iteritems()
                        if not term.factor in t.factors]
        ssr_wterm = ssrwoutterm + np.sum(effects[cols[0]:cols[1]]**2)

        # need to do a full pass over the terms to determine what we need
        #assign.extend(

        #table.ix[i]['sum_sq'] = np.sum(effects[cols[0]:cols[1]]**2)
        #table.ix[i]['df'] = cols[1]-cols[0]
        #col_order.append(cols[0])
        #index.append(term.name())
        #assign.extend([])
    table.index = Index(index + ['Residual'])
    # fill in residual
    table.ix['Residual'][['sum_sq','df', test, pr_test]] = (model.ssr,
                                                            model.df_resid,
                                                            np.nan, np.nan)

    table = table.ix[np.argsort(col_order + [exog.shape[1]+1])]
    table['mean_sq'] = table['sum_sq'] / table['df']
    if test == 'F':
        table[:n_rows][test] = ((table['sum_sq']/table['df'])/
                                (model.ssr/model.df_resid))
        table[:n_rows][pr_test] = stats.f.sf(table["F"], table["df"],
                                model.df_resid)
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
        return anova_single(model, typ, **kwargs)

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
    moore = pandas.read_table('moore.csv', delimiter=" ", skiprows=1,
                                names=['partner_status','conformity',
                                    'fcategory','fscore'])
    moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',
                    df=moore).fit()

    mooreB = ols('conformity ~ C(partner_status, Sum)', df=moore).fit()

    # try to estimate the residuals using projections



    from statsmodels.stats.contrast import Contrast
    exog = moore_lm.model.exog
    endog = moore_lm.model.endog
    c = np.atleast_2d(Contrast(exog[:,[3]], exog).contrast_matrix)
    c = np.r_[c,np.zeros((5,6))]
    C0 = np.eye(6) - np.dot(c,np.linalg.pinv(c))
    X0 = np.dot(exog, C0)
    R0 = np.eye(45) - np.dot(X0, np.linalg.pinv(X0))
    R = np.eye(45) - np.dot(exog, moore_lm.model.pinv_wexog)
    M = R0 - R
    ssr = np.dot(np.dot(M,endog),np.dot(M,endog))

