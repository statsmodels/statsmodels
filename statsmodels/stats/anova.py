from statsmodels.compat.python import lrange, lmap
import numpy as np
from scipy import stats
from pandas import DataFrame, Index
from statsmodels.formula.formulatools import (_remove_intercept_patsy,
                                    _has_intercept, _intercept_idx)

def _get_covariance(model, robust):
    if robust is None:
        return model.cov_params()
    elif robust == "hc0":
        se = model.HC0_se
        return model.cov_HC0
    elif robust == "hc1":
        se = model.HC1_se
        return model.cov_HC1
    elif robust == "hc2":
        se = model.HC2_se
        return model.cov_HC2
    elif robust == "hc3":
        se = model.HC3_se
        return model.cov_HC3
    else: # pragma: no cover
        raise ValueError("robust options %s not understood" % robust)

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
    robust = kwargs.get("robust", None)
    if robust:
        robust = robust.lower()

    endog = model.model.endog
    exog = model.model.exog
    nobs = exog.shape[0]

    response_name = model.model.endog_names
    design_info = model.model.data.design_info
    exog_names = model.model.exog_names
    # +1 for resids
    n_rows = (len(design_info.terms) - _has_intercept(design_info) + 1)

    pr_test = "PR(>%s)" % test
    names = ['df', 'sum_sq', 'mean_sq', test, pr_test]

    table = DataFrame(np.zeros((n_rows, 5)), columns = names)

    if typ in [1,"I"]:
        return anova1_lm_single(model, endog, exog, nobs, design_info, table,
                                n_rows, test, pr_test, robust)
    elif typ in [2, "II"]:
        return anova2_lm_single(model, design_info, n_rows, test, pr_test,
                robust)
    elif typ in [3, "III"]:
        return anova3_lm_single(model, design_info, n_rows, test, pr_test,
                robust)
    elif typ in [4, "IV"]:
        raise NotImplemented("Type IV not yet implemented")
    else: # pragma: no cover
        raise ValueError("Type %s not understood" % str(typ))

def anova1_lm_single(model, endog, exog, nobs, design_info, table, n_rows, test,
                     pr_test, robust):
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
    effects = getattr(model, 'effects', None)
    if effects is None:
        q,r = np.linalg.qr(exog)
        effects = np.dot(q.T, endog)

    arr = np.zeros((len(design_info.terms), len(design_info.column_names)))
    slices = [design_info.slice(name) for name in design_info.term_names]
    for i,slice_ in enumerate(slices):
        arr[i, slice_] = 1

    sum_sq = np.dot(arr, effects**2)
    #NOTE: assumes intercept is first column
    idx = _intercept_idx(design_info)
    sum_sq = sum_sq[~idx]
    term_names = np.array(design_info.term_names) # want boolean indexing
    term_names = term_names[~idx]

    index = term_names.tolist()
    table.index = Index(index + ['Residual'])
    table.ix[index, ['df', 'sum_sq']] = np.c_[arr[~idx].sum(1), sum_sq]
    if test == 'F':
        table.ix[:n_rows, test] = ((table['sum_sq']/table['df'])/
                                (model.ssr/model.df_resid))
        table.ix[:n_rows, pr_test] = stats.f.sf(table["F"], table["df"],
                                model.df_resid)

    # fill in residual
    table.ix['Residual', ['sum_sq','df', test, pr_test]] = (model.ssr,
                                                            model.df_resid,
                                                            np.nan, np.nan)
    table['mean_sq'] = table['sum_sq'] / table['df']
    return table

#NOTE: the below is not agnostic about formula...
def anova2_lm_single(model, design_info, n_rows, test, pr_test, robust):
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
    terms_info = design_info.terms[:] # copy
    terms_info = _remove_intercept_patsy(terms_info)

    names = ['sum_sq', 'df', test, pr_test]

    table = DataFrame(np.zeros((n_rows, 4)), columns = names)
    cov = _get_covariance(model, None)
    robust_cov = _get_covariance(model, robust)
    col_order = []
    index = []
    for i, term in enumerate(terms_info):
        # grab all varaibles except interaction effects that contain term
        # need two hypotheses matrices L1 is most restrictive, ie., term==0
        # L2 is everything except term==0
        cols = design_info.slice(term)
        L1 = lrange(cols.start, cols.stop)
        L2 = []
        term_set = set(term.factors)
        for t in terms_info: # for the term you have
            other_set = set(t.factors)
            if term_set.issubset(other_set) and not term_set == other_set:
                col = design_info.slice(t)
                # on a higher order term containing current `term`
                L1.extend(lrange(col.start, col.stop))
                L2.extend(lrange(col.start, col.stop))

        L1 = np.eye(model.model.exog.shape[1])[L1]
        L2 = np.eye(model.model.exog.shape[1])[L2]

        if L2.size:
            LVL = np.dot(np.dot(L1,robust_cov),L2.T)
            from scipy import linalg
            orth_compl,_ = linalg.qr(LVL)
            r = L1.shape[0] - L2.shape[0]
            # L1|2
            # use the non-unique orthogonal completion since L12 is rank r
            L12 = np.dot(orth_compl[:,-r:].T, L1)
        else:
            L12 = L1
            r = L1.shape[0]
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        if test == 'F':
            f = model.f_test(L12, cov_p=robust_cov)
            table.ix[i, test] = test_value = f.fvalue
            table.ix[i, pr_test] = f.pvalue

        # need to back out SSR from f_test
        table.ix[i, 'df'] = r
        col_order.append(cols.start)
        index.append(term.name())

    table.index = Index(index + ['Residual'])
    table = table.ix[np.argsort(col_order + [model.model.exog.shape[1]+1])]
    # back out sum of squares from f_test
    ssr = table[test] * table['df'] * model.ssr/model.df_resid
    table['sum_sq'] = ssr
    # fill in residual
    table.ix['Residual', ['sum_sq','df', test, pr_test]] = (model.ssr,
                                                            model.df_resid,
                                                            np.nan, np.nan)

    return table

def anova3_lm_single(model, design_info, n_rows, test, pr_test, robust):
    n_rows += _has_intercept(design_info)
    terms_info = design_info.terms

    names = ['sum_sq', 'df', test, pr_test]

    table = DataFrame(np.zeros((n_rows, 4)), columns = names)
    cov = _get_covariance(model, robust)
    col_order = []
    index = []
    for i, term in enumerate(terms_info):
        # grab term, hypothesis is that term == 0
        cols = design_info.slice(term)
        L1 = np.eye(model.model.exog.shape[1])[cols]
        L12 = L1
        r = L1.shape[0]

        if test == 'F':
            f = model.f_test(L12, cov_p=cov)
            table.ix[i, test] = test_value = f.fvalue
            table.ix[i, pr_test] = f.pvalue

        # need to back out SSR from f_test
        table.ix[i, 'df'] = r
        #col_order.append(cols.start)
        index.append(term.name())

    table.index = Index(index + ['Residual'])
    #NOTE: Don't need to sort because terms are an ordered dict now
    #table = table.ix[np.argsort(col_order + [model.model.exog.shape[1]+1])]
    # back out sum of squares from f_test
    ssr = table[test] * table['df'] * model.ssr/model.df_resid
    table['sum_sq'] = ssr
    # fill in residual
    table.ix['Residual', ['sum_sq','df', test, pr_test]] = (model.ssr,
                                                            model.df_resid,
                                                            np.nan, np.nan)
    return table

def anova_lm(*args, **kwargs):
    """
    ANOVA table for one or more fitted linear models.

    Parameters
    ----------
    args : fitted linear model results instance
        One or more fitted linear models
    scale : float
        Estimate of variance, If None, will be estimated from the largest
        model. Default is None.
    test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide. Default is "F".
    typ : str or int {"I","II","III"} or {1,2,3}
        The type of ANOVA test to perform. See notes.
    robust : {None, "hc0", "hc1", "hc2", "hc3"}
        Use heteroscedasticity-corrected coefficient covariance matrix.
        If robust covariance is desired, it is recommended to use `hc3`.

    Returns
    -------
    anova : DataFrame
    A DataFrame containing.

    Notes
    -----
    Model statistics are given in the order of args. Models must have
    been fit using the formula api.

    See Also
    --------
    model_results.compare_f_test, model_results.compare_lm_test

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.formula.api import ols
    >>> moore = sm.datasets.get_rdataset("Moore", "car", cache=True) # load
    >>> data = moore.data
    >>> data = data.rename(columns={"partner.status" :
    ...                             "partner_status"}) # make name pythonic
    >>> moore_lm = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',
    ...                 data=data).fit()
    >>> table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 ANOVA DataFrame
    >>> print(table)
    """
    typ = kwargs.get('typ', 1)

    ### Farm Out Single model ANOVA Type I, II, III, and IV ###

    if len(args) == 1:
        model = args[0]
        return anova_single(model, **kwargs)

    try:
        assert typ in [1,"I"]
    except:
        raise ValueError("Multiple models only supported for type I. "
                         "Got type %s" % str(typ))

    ### COMPUTE ANOVA TYPE I ###

    # if given a single model
    if len(args) == 1:
        return anova_single(*args, **kwargs)

    # received multiple fitted models

    test = kwargs.get("test", "F")
    scale = kwargs.get("scale", None)
    n_models = len(args)

    model_formula = []
    pr_test = "Pr(>%s)" % test
    names = ['df_resid', 'ssr', 'df_diff', 'ss_diff', test, pr_test]
    table = DataFrame(np.zeros((n_models, 6)), columns = names)

    if not scale: # assume biggest model is last
        scale = args[-1].scale

    table["ssr"] = lmap(getattr, args, ["ssr"]*n_models)
    table["df_resid"] = lmap(getattr, args, ["df_resid"]*n_models)
    table.ix[1:, "df_diff"] = -np.diff(table["df_resid"].values)
    table["ss_diff"] = -table["ssr"].diff()
    if test == "F":
        table["F"] = table["ss_diff"] / table["df_diff"] / scale
        table[pr_test] = stats.f.sf(table["F"], table["df_diff"],
                             table["df_resid"])
        # for earlier scipy - stats.f.sf(np.nan, 10, 2) -> 0 not nan
        table[pr_test][table['F'].isnull()] = np.nan

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
                    data=moore).fit()

    mooreB = ols('conformity ~ C(partner_status, Sum)', data=moore).fit()

    # for each term you just want to test vs the model without its
    # higher-order terms

    # using Monette-Fox slides and Marden class notes for linear algebra /
    # orthogonal complement
    # https://netfiles.uiuc.edu/jimarden/www/Classes/STAT324/

    table = anova_lm(moore_lm, typ=2)
