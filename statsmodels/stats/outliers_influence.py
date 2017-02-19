# -*- coding: utf-8 -*-
"""Influence and Outlier Measures

Created on Sun Jan 29 11:16:09 2012

Author: Josef Perktold
License: BSD-3
"""
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.tools import maybe_unwrap_results

# outliers test convenience wrapper

def outlier_test(model_results, method='bonf', alpha=.05, labels=None,
                 order=False):
    """
    Outlier Tests for RegressionResults instances.

    Parameters
    ----------
    model_results : RegressionResults instance
        Linear model results
    method : str
        - `bonferroni` : one-step correction
        - `sidak` : one-step correction
        - `holm-sidak` :
        - `holm` :
        - `simes-hochberg` :
        - `hommel` :
        - `fdr_bh` : Benjamini/Hochberg
        - `fdr_by` : Benjamini/Yekutieli
        See `statsmodels.stats.multitest.multipletests` for details.
    alpha : float
        familywise error rate
    order : bool
        Whether or not to order the results by the absolute value of the
        studentized residuals. If labels are provided they will also be sorted.

    Returns
    -------
    table : ndarray or DataFrame
        Returns either an ndarray or a DataFrame if labels is not None.
        Will attempt to get labels from model_results if available. The
        columns are the Studentized residuals, the unadjusted p-value,
        and the corrected p-value according to method.

    Notes
    -----
    The unadjusted p-value is stats.t.sf(abs(resid), df) where
    df = df_resid - 1.
    """
    from scipy import stats # lazy import
    infl = getattr(model_results, 'get_influence', None)
    if infl is None:
        results = maybe_unwrap_results(model_results)
        raise AttributeError("model_results object %s does not have a "
                "get_influence method." % results.__class__.__name__)
    resid = infl().resid_studentized_external
    if order:
        idx = np.abs(resid).argsort()[::-1]
        resid = resid[idx]
        if labels is not None:
            labels = np.array(labels)[idx].tolist()
    df = model_results.df_resid - 1
    unadj_p = stats.t.sf(np.abs(resid), df) * 2
    adj_p = multipletests(unadj_p, alpha=alpha, method=method)

    data = np.c_[resid, unadj_p, adj_p[1]]
    if labels is None:
        labels = getattr(model_results.model.data, 'row_labels', None)
    if labels is not None:
        from pandas import DataFrame
        return DataFrame(data,
                         columns=['student_resid', 'unadj_p', method+"(p)"],
                         index=labels)
    return data

#influence measures

def reset_ramsey(res, degree=5):
    '''Ramsey's RESET specification test for linear models

    This is a general specification test, for additional non-linear effects
    in a model.


    Notes
    -----
    The test fits an auxiliary OLS regression where the design matrix, exog,
    is augmented by powers 2 to degree of the fitted values. Then it performs
    an F-test whether these additional terms are significant.

    If the p-value of the f-test is below a threshold, e.g. 0.1, then this
    indicates that there might be additional non-linear effects in the model
    and that the linear model is mis-specified.


    References
    ----------
    http://en.wikipedia.org/wiki/Ramsey_RESET_test

    '''
    order = degree + 1
    k_vars = res.model.exog.shape[1]
    #vander without constant and x:
    y_fitted_vander = np.vander(res.fittedvalues, order)[:, :-2] #drop constant
    exog = np.column_stack((res.model.exog, y_fitted_vander))
    res_aux = OLS(res.model.endog, exog).fit()
    #r_matrix = np.eye(degree, exog.shape[1], k_vars)
    r_matrix = np.eye(degree-1, exog.shape[1], k_vars)
    #df1 = degree - 1
    #df2 = exog.shape[0] - degree - res.df_model  (without constant)
    return res_aux.f_test(r_matrix) #, r_matrix, res_aux



def variance_inflation_factor(exog, exog_idx):
    '''variance inflation factor, VIF, for one exogenous variable

    The variance inflation factor is a measure for the increase of the
    variance of the parameter estimates if an additional variable, given by
    exog_idx is added to the linear regression. It is a measure for
    multicollinearity of the design matrix, exog.

    One recommendation is that if VIF is greater than 5, then the explanatory
    variable given by exog_idx is highly collinear with the other explanatory
    variables, and the parameter estimates will have large standard errors
    because of this.

    Parameters
    ----------
    exog : ndarray, (nobs, k_vars)
        design matrix with all explanatory variables, as for example used in
        regression
    exog_idx : int
        index of the exogenous variable in the columns of exog

    Returns
    -------
    vif : float
        variance inflation factor

    Notes
    -----
    This function does not save the auxiliary regression.

    See Also
    --------
    xxx : class for regression diagnostics  TODO: doesn't exist yet

    References
    ----------
    http://en.wikipedia.org/wiki/Variance_inflation_factor

    '''
    k_vars = exog.shape[1]
    x_i = exog[:, exog_idx]
    mask = np.arange(k_vars) != exog_idx
    x_noti = exog[:, mask]
    r_squared_i = OLS(x_i, x_noti).fit().rsquared
    vif = 1. / (1. - r_squared_i)
    return vif


class OLSInfluence(object):
    '''class to calculate outlier and influence measures for OLS result

    Parameters
    ----------
    results : Regression Results instance
        currently assumes the results are from an OLS regression

    Notes
    -----
    One part of the results can be calculated without any auxiliary regression
    (some of which have the `_internal` postfix in the name. Other statistics
    require leave-one-observation-out (LOOO) auxiliary regression, and will be
    slower (mainly results with `_external` postfix in the name).
    The auxiliary LOOO regression only the required results are stored.

    Using the LOO measures is currently only recommended if the data set
    is not too large. One possible approach for LOOO measures would be to
    identify possible problem observations with the _internal measures, and
    then run the leave-one-observation-out only with observations that are
    possible outliers. (However, this is not yet available in an automized way.)

    This should be extended to general least squares.

    The leave-one-variable-out (LOVO) auxiliary regression are currently not
    used.

    '''

    def __init__(self, results):
        #check which model is allowed
        self.results = maybe_unwrap_results(results)
        self.nobs, self.k_vars = results.model.exog.shape
        self.endog = results.model.endog
        self.exog = results.model.exog
        self.model_class = results.model.__class__

        self.sigma_est = np.sqrt(results.mse_resid)

        self.aux_regression_exog = {}
        self.aux_regression_endog = {}

    @cache_readonly
    def hat_matrix_diag(self):
        '''(cached attribute) diagonal of the hat_matrix for OLS

        Notes
        -----
        temporarily calculated here, this should go to model class
        '''
        return (self.exog * self.results.model.pinv_wexog.T).sum(1)

    @cache_readonly
    def resid_press(self):
        '''(cached attribute) PRESS residuals
        '''
        hii = self.hat_matrix_diag
        return self.results.resid / (1 - hii)

    @cache_readonly
    def influence(self):
        '''(cached attribute) influence measure

        matches the influence measure that gretl reports
        u * h / (1 - h)
        where u are the residuals and h is the diagonal of the hat_matrix
        '''
        hii = self.hat_matrix_diag
        return self.results.resid * hii / (1 - hii)

    @cache_readonly
    def hat_diag_factor(self):
        '''(cached attribute) factor of diagonal of hat_matrix used in influence

        this might be useful for internal reuse
        h / (1 - h)
        '''
        hii = self.hat_matrix_diag
        return hii / (1 - hii)

    @cache_readonly
    def ess_press(self):
        '''(cached attribute) error sum of squares of PRESS residuals
        '''
        return np.dot(self.resid_press, self.resid_press)

    @cache_readonly
    def resid_studentized_internal(self):
        '''(cached attribute) studentized residuals using variance from OLS

        this uses sigma from original estimate
        does not require leave one out loop
        '''
        return self.get_resid_studentized_external(sigma=None)
        #return self.results.resid / self.sigma_est

    @cache_readonly
    def resid_studentized_external(self):
        '''(cached attribute) studentized residuals using LOOO variance

        this uses sigma from leave-one-out estimates

        requires leave one out loop for observations
        '''
        sigma_looo = np.sqrt(self.sigma2_not_obsi)
        return self.get_resid_studentized_external(sigma=sigma_looo)

    def get_resid_studentized_external(self, sigma=None):
        '''calculate studentized residuals

        Parameters
        ----------
        sigma : None or float
            estimate of the standard deviation of the residuals. If None, then
            the estimate from the regression results is used.

        Returns
        -------
        stzd_resid : ndarray
            studentized residuals

        Notes
        -----
        studentized residuals are defined as ::

           resid / sigma / np.sqrt(1 - hii)

        where resid are the residuals from the regression, sigma is an
        estimate of the standard deviation of the residuals, and hii is the
        diagonal of the hat_matrix.

        '''
        hii = self.hat_matrix_diag
        if sigma is None:
            sigma2_est = self.results.mse_resid
            #can be replace by different estimators of sigma
            sigma = np.sqrt(sigma2_est)

        return  self.results.resid / sigma / np.sqrt(1 - hii)

    @cache_readonly
    def dffits_internal(self):
        '''(cached attribute) dffits measure for influence of an observation

        based on resid_studentized_internal
        uses original results, no nobs loop

        '''
        #TODO: do I want to use different sigma estimate in
        #      resid_studentized_external
        # -> move definition of sigma_error to the __init__
        hii = self.hat_matrix_diag
        dffits_ = self.resid_studentized_internal * np.sqrt(hii / (1 - hii))
        dffits_threshold = 2 * np.sqrt(self.k_vars * 1. / self.nobs)
        return dffits_, dffits_threshold

    @cache_readonly
    def dffits(self):
        '''(cached attribute) dffits measure for influence of an observation

        based on resid_studentized_external,
        uses results from leave-one-observation-out loop

        It is recommended that observations with dffits large than a
        threshold of 2 sqrt{k / n} where k is the number of parameters, should
        be investigated.

        Returns
        -------
        dffits: float
        dffits_threshold : float

        References
        ----------
        `Wikipedia <http://en.wikipedia.org/wiki/DFFITS>`_

        '''
        #TODO: do I want to use different sigma estimate in
        #      resid_studentized_external
        # -> move definition of sigma_error to the __init__
        hii = self.hat_matrix_diag
        dffits_ = self.resid_studentized_external * np.sqrt(hii / (1 - hii))
        dffits_threshold = 2 * np.sqrt(self.k_vars * 1. / self.nobs)
        return dffits_, dffits_threshold

    @cache_readonly
    def dfbetas(self):
        '''(cached attribute) dfbetas

        uses results from leave-one-observation-out loop
        '''
        dfbetas = self.results.params - self.params_not_obsi#[None,:]
        dfbetas /= np.sqrt(self.sigma2_not_obsi[:,None])
        dfbetas /=  np.sqrt(np.diag(self.results.normalized_cov_params))
        return dfbetas

    @cache_readonly
    def sigma2_not_obsi(self):
        '''(cached attribute) error variance for all LOOO regressions

        This is 'mse_resid' from each auxiliary regression.

        uses results from leave-one-observation-out loop
        '''
        return np.asarray(self._res_looo['mse_resid'])

    @cache_readonly
    def params_not_obsi(self):
        '''(cached attribute) parameter estimates for all LOOO regressions

        uses results from leave-one-observation-out loop
        '''
        return np.asarray(self._res_looo['params'])

    @cache_readonly
    def det_cov_params_not_obsi(self):
        '''(cached attribute) determinant of cov_params of all LOOO regressions

        uses results from leave-one-observation-out loop
        '''
        return np.asarray(self._res_looo['det_cov_params'])

    @cache_readonly
    def cooks_distance(self):
        '''(cached attribute) Cooks distance

        uses original results, no nobs loop

        '''
        hii = self.hat_matrix_diag
        #Eubank p.93, 94
        cooks_d2 = self.resid_studentized_internal**2 / self.k_vars
        cooks_d2 *= hii / (1 - hii)

        from scipy import stats
        #alpha = 0.1
        #print stats.f.isf(1-alpha, n_params, res.df_modelwc)
        pvals = stats.f.sf(cooks_d2, self.k_vars, self.results.df_resid)

        return cooks_d2, pvals

    @cache_readonly
    def cov_ratio(self):
        '''(cached attribute) covariance ratio between LOOO and original

        This uses determinant of the estimate of the parameter covariance
        from leave-one-out estimates.
        requires leave one out loop for observations

        '''
        #don't use inplace division / because then we change original
        cov_ratio = (self.det_cov_params_not_obsi
                            / np.linalg.det(self.results.cov_params()))
        return cov_ratio

    @cache_readonly
    def resid_var(self):
        '''(cached attribute) estimate of variance of the residuals

        ::

           sigma2 = sigma2_OLS * (1 - hii)

        where hii is the diagonal of the hat matrix

        '''
        #TODO:check if correct outside of ols
        return self.results.mse_resid * (1 - self.hat_matrix_diag)

    @cache_readonly
    def resid_std(self):
        '''(cached attribute) estimate of standard deviation of the residuals

        See Also
        --------
        resid_var

        '''
        return np.sqrt(self.resid_var)


    def _ols_xnoti(self, drop_idx, endog_idx='endog', store=True):
        '''regression results from LOVO auxiliary regression with cache


        The result instances are stored, which could use a large amount of
        memory if the datasets are large. There are too many combinations to
        store them all, except for small problems.

        Parameters
        ----------
        drop_idx : int
            index of exog that is dropped from the regression
        endog_idx : 'endog' or int
            If 'endog', then the endogenous variable of the result instance
            is regressed on the exogenous variables, excluding the one at
            drop_idx. If endog_idx is an integer, then the exog with that
            index is regressed with OLS on all other exogenous variables.
            (The latter is the auxiliary regression for the variance inflation
            factor.)

        this needs more thought, memory versus speed
        not yet used in any other parts, not sufficiently tested
        '''
        #reverse the structure, access store, if fail calculate ?
        #this creates keys in store even if store = false ! bug
        if endog_idx == 'endog':
            stored = self.aux_regression_endog
            if hasattr(stored, drop_idx):
                return stored[drop_idx]
            x_i = self.results.model.endog

        else:
            #nested dictionary
            try:
                self.aux_regression_exog[endog_idx][drop_idx]
            except KeyError:
                pass

            stored = self.aux_regression_exog[endog_idx]
            stored = {}

            x_i = self.exog[:, endog_idx]

        k_vars = self.exog.shape[1]
        mask = np.arange(k_vars) != drop_idx
        x_noti = self.exog[:, mask]
        res = OLS(x_i, x_noti).fit()
        if store:
            stored[drop_idx] = res

        return res

    def _get_drop_vari(self, attributes):
        '''regress endog on exog without one of the variables

        This uses a k_vars loop, only attributes of the OLS instance are stored.

        Parameters
        ----------
        attributes : list of strings
           These are the names of the attributes of the auxiliary OLS results
           instance that are stored and returned.

        not yet used
        '''
        from statsmodels.sandbox.tools.cross_val import LeaveOneOut

        endog = self.results.model.endog
        exog = self.exog

        cv_iter = LeaveOneOut(self.k_vars)
        res_loo = defaultdict(list)
        for inidx, outidx in cv_iter:
            for att in attributes:
                res_i = self.model_class(endog, exog[:,inidx]).fit()
                res_loo[att].append(getattr(res_i, att))

        return res_loo

    @cache_readonly
    def _res_looo(self):
        '''collect required results from the LOOO loop

        all results will be attached.
        currently only 'params', 'mse_resid', 'det_cov_params' are stored

        regresses endog on exog dropping one observation at a time

        this uses a nobs loop, only attributes of the OLS instance are stored.
        '''
        from statsmodels.sandbox.tools.cross_val import LeaveOneOut
        get_det_cov_params = lambda res: np.linalg.det(res.cov_params())

        endog = self.endog
        exog = self.exog

        params = np.zeros(exog.shape, dtype=np.float)
        mse_resid = np.zeros(endog.shape, dtype=np.float)
        det_cov_params = np.zeros(endog.shape, dtype=np.float)

        cv_iter = LeaveOneOut(self.nobs)
        for inidx, outidx in cv_iter:
            res_i = self.model_class(endog[inidx], exog[inidx]).fit()
            params[outidx] = res_i.params
            mse_resid[outidx] = res_i.mse_resid
            det_cov_params[outidx] = get_det_cov_params(res_i)

        return dict(params=params, mse_resid=mse_resid,
                       det_cov_params=det_cov_params)

    def summary_frame(self):
        """
        Creates a DataFrame with all available influence results.

        Returns
        -------
        frame : DataFrame
            A DataFrame with all results.

        Notes
        -----
        The resultant DataFrame contains six variables in addition to the
        DFBETAS. These are:

        * cooks_d : Cook's Distance defined in `Influence.cooks_distance`
        * standard_resid : Standardized residuals defined in
          `Influence.resid_studentized_internal`
        * hat_diag : The diagonal of the projection, or hat, matrix defined in
          `Influence.hat_matrix_diag`
        * dffits_internal : DFFITS statistics using internally Studentized
          residuals defined in `Influence.dffits_internal`
        * dffits : DFFITS statistics using externally Studentized residuals
          defined in `Influence.dffits`
        * student_resid : Externally Studentized residuals defined in
          `Influence.resid_studentized_external`
        """
        from pandas import DataFrame

        # row and column labels
        data = self.results.model.data
        row_labels = data.row_labels
        beta_labels = ['dfb_' + i for i in data.xnames]

        # grab the results
        summary_data = DataFrame(dict(
                            cooks_d = self.cooks_distance[0],
                            standard_resid = self.resid_studentized_internal,
                            hat_diag = self.hat_matrix_diag,
                            dffits_internal = self.dffits_internal[0],
                            student_resid = self.resid_studentized_external,
                            dffits = self.dffits[0],
                                        ),
                            index = row_labels)
        #NOTE: if we don't give columns, order of above will be arbitrary
        dfbeta = DataFrame(self.dfbetas, columns=beta_labels,
                            index=row_labels)

        return dfbeta.join(summary_data)

    def summary_table(self, float_fmt="%6.3f"):
        '''create a summary table with all influence and outlier measures

        This does currently not distinguish between statistics that can be
        calculated from the original regression results and for which a
        leave-one-observation-out loop is needed

        Returns
        -------
        res : SimpleTable instance
           SimpleTable instance with the results, can be printed

        Notes
        -----
        This also attaches table_data to the instance.



        '''
        #print self.dfbetas

#        table_raw = [ np.arange(self.nobs),
#                      self.endog,
#                      self.fittedvalues,
#                      self.cooks_distance(),
#                      self.resid_studentized_internal,
#                      self.hat_matrix_diag,
#                      self.dffits_internal,
#                      self.resid_studentized_external,
#                      self.dffits,
#                      self.dfbetas
#                      ]
        table_raw = [ ('obs', np.arange(self.nobs)),
                      ('endog', self.endog),
                      ('fitted\nvalue', self.results.fittedvalues),
                      ("Cook's\nd", self.cooks_distance[0]),
                      ("student.\nresidual", self.resid_studentized_internal),
                      ('hat diag', self.hat_matrix_diag),
                      ('dffits \ninternal', self.dffits_internal[0]),
                      ("ext.stud.\nresidual", self.resid_studentized_external),
                      ('dffits', self.dffits[0])
                      ]
        colnames, data = lzip(*table_raw) #unzip
        data = np.column_stack(data)
        self.table_data = data
        from statsmodels.iolib.table import SimpleTable, default_html_fmt
        from statsmodels.iolib.tableformatting import fmt_base
        from copy import deepcopy
        fmt = deepcopy(fmt_base)
        fmt_html = deepcopy(default_html_fmt)
        fmt['data_fmts'] = ["%4d"] + [float_fmt] * (data.shape[1] - 1)
        #fmt_html['data_fmts'] = fmt['data_fmts']
        return SimpleTable(data, headers=colnames, txt_fmt=fmt,
                           html_fmt=fmt_html)


def summary_table(res, alpha=0.05):
    """
    Generate summary table of outlier and influence similar to SAS

    Parameters
    ----------
    alpha : float
       significance level for confidence interval

    Returns
    -------
    st : SimpleTable instance
       table with results that can be printed
    data : ndarray
       calculated measures and statistics for the table
    ss2 : list of strings
       column_names for table (Note: rows of table are observations)
    """

    from scipy import stats
    from statsmodels.sandbox.regression.predstd import wls_prediction_std

    infl = OLSInfluence(res)

    #standard error for predicted mean
    #Note: using hat_matrix only works for fitted values
    predict_mean_se = np.sqrt(infl.hat_matrix_diag*res.mse_resid)

    tppf = stats.t.isf(alpha/2., res.df_resid)
    predict_mean_ci = np.column_stack([
                        res.fittedvalues - tppf * predict_mean_se,
                        res.fittedvalues + tppf * predict_mean_se])


    #standard error for predicted observation
    tmp = wls_prediction_std(res, alpha=alpha)
    predict_se, predict_ci_low, predict_ci_upp = tmp

    predict_ci = np.column_stack((predict_ci_low, predict_ci_upp))

    #standard deviation of residual
    resid_se = np.sqrt(res.mse_resid * (1 - infl.hat_matrix_diag))

    table_sm = np.column_stack([
                                  np.arange(res.nobs) + 1,
                                  res.model.endog,
                                  res.fittedvalues,
                                  predict_mean_se,
                                  predict_mean_ci[:,0],
                                  predict_mean_ci[:,1],
                                  predict_ci[:,0],
                                  predict_ci[:,1],
                                  res.resid,
                                  resid_se,
                                  infl.resid_studentized_internal,
                                  infl.cooks_distance[0]
                                  ])


    #colnames, data = lzip(*table_raw) #unzip
    data = table_sm
    ss2 = ['Obs', 'Dep Var\nPopulation', 'Predicted\nValue', 'Std Error\nMean Predict', 'Mean ci\n95% low', 'Mean ci\n95% upp', 'Predict ci\n95% low', 'Predict ci\n95% upp', 'Residual', 'Std Error\nResidual', 'Student\nResidual', "Cook's\nD"]
    colnames = ss2
    #self.table_data = data
    #data = np.column_stack(data)
    from statsmodels.iolib.table import SimpleTable, default_html_fmt
    from statsmodels.iolib.tableformatting import fmt_base
    from copy import deepcopy
    fmt = deepcopy(fmt_base)
    fmt_html = deepcopy(default_html_fmt)
    fmt['data_fmts'] = ["%4d"] + ["%6.3f"] * (data.shape[1] - 1)
    #fmt_html['data_fmts'] = fmt['data_fmts']
    st = SimpleTable(data, headers=colnames, txt_fmt=fmt,
                       html_fmt=fmt_html)

    return st, data, ss2

