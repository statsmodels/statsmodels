# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:38:12 2019
Based on script sequential_ols_qr.py Created on Sun Jun 17 17:41:48 2012

Author: Josef Perktold
License: BSD-3

"""

import numpy as np

from statsmodels.tools.decorators import cache_readonly
import statsmodels.tools.eval_measures as evm


def sweep(k, rs):
    k_vars = rs.shape[1] - 1
    # TODO: can we do inplace sweep ?
    rkk = rs[k, k].copy()
    rs_next = rs.copy()
    not_k = np.ones(k_vars+1, bool)
    not_k[k] = False
    # sweep in 4 steps
    rs_next[:, k] /= rkk
    rs_next[k, :] /= -rkk
    # next is same as outer but dot is more explicit
    rs_next[not_k[:, None] * not_k] -= np.dot(rs[not_k, k:k+1],
                                              rs[k:k+1, not_k]).ravel() / rkk
    rs_next[k, k] = 1. / rkk

    return rs_next


def get_sweep_matrix_data(xy, sweep_idx):
    """use pinv to create a sweep matrix
    """
    k_vars = xy.shape[1]
    idx_not = np.asarray([i for i in range(k_vars) if i not in sweep_idx])
    idx_sweep = np.asarray(sweep_idx)
    x = xy[:, idx_sweep]
    if x.ndim == 1:
        x = x[:, None]
    y = xy[:, idx_not]
    if y.ndim == 1:
        y = y[:, None]
    xi = np.linalg.pinv(x)
    xiy = xi.dot(y)
    resid = y - x.dot(xi.dot(y))
    sm = np.block([[xi.dot(xi.T), -xiy], [xiy.T, resid.T.dot(resid)]])
    return sm


def sweeps2isexog(sweeps, k=None, isexog0=None):
    """
    convert sequence of sweeps to is_exog
    """
    if k is None:
        k = max(sweeps) + 1

    if isexog0 is None:
        is_exog = np.zeros(k, np.bool)
    else:
        is_exog = np.asarray(isexog0, dtype=np.bool)

    is_exog_all = []
    for idx in sweeps:
        is_exog[idx] = ~is_exog[idx]
        is_exog_all.append(is_exog.copy())  # need copy because of inplace
    return is_exog_all


def sweeps2isexog_iter(sweeps, k=None, isexog0=None):
    """iterator to convert sequence of sweeps to is_exog
    """
    if k is None:
        k = max(sweeps) + 1

    if isexog0 is None:
        is_exog = np.zeros(k, np.bool)
    else:
        is_exog = np.asarray(isexog0, dtype=np.bool)

    for idx in sweeps:
        is_exog[idx] = ~is_exog[idx]
        yield is_exog.copy()


def isexog2sweeps(isexog):
    sweeps = []
    for i in range(len(isexog) - 1):
        diff = isexog[i + 1] != isexog[i]
        tosweep = np.nonzero(diff)[0]
        sweeps.extend(tosweep.tolist())
    return sweeps


def fillsweeps(sweeps, k):
    """doesn't make sense, we don't know what's missing"""

    ex_iter = sweeps2isexog_iter(sweeps, k)
    sweeps_new = []
    previous = ex_iter.next()
    for _ in range(len(sweeps) - 1):
        current = ex_iter.next()
        diff = current != previous  # elementwise
        tosweep = np.nonzero(diff)[0]
        sweeps_new.extend(tosweep.tolist())
    return sweeps_new


def binary_sets(k, dtype=np.bool):
    all_subsets = []
    a = np.arange(2**k)
    for i in range(1, k+1):
        d, a = np.divmod(a, 2**(k-i))
        all_subsets.append(d)

    return np.array(all_subsets, dtype=dtype).T.tolist()


def dist(x, y):
    """Manhattan (sum abs) distance between two arrays
    """
    return np.abs(np.asarray(x) - np.asarray(y)).sum()


def swap_li(li, idx0, idx1):
    """inplace swap of two elements in list
    """
    z = li[idx0].copy()
    li[idx0] = li[idx1]
    li[idx1] = z
    return None


def swap_sort(li_all, maxiter=2, reverse=True, max_dist=1, max_len=2**3):
    """rearrange swaps to shorten distance

    This should provide some improvements for a sequence of models to reduce
    the number of sweeps. Trying to get close to optimal sequence is
    time consuming and can become more expensive than performing extra
    sweeps. (travelling salesman problem)

    Warning: this might make the sweep sequence worse than just filling the
    gaps with extra sweeps

    """
    # make list of ndarrays with copies
    # list is better for swapping, need ndarray for dist
    li = [np.asarray(i).copy() for i in li_all]
    start_index = 0
    count_swaps_total = 0
    for _ in range(maxiter):
        count_swaps = 0
        for i in range(start_index, len(li) - 1):
            x = li[i]
            d = dist(x, li[i + 1])
            if d > 1:
                if count_swaps == 0:
                    start_index = i
                candidates = range(i+1, min(i + 1 + max_len, len(li)))
                if reverse:
                    candidates = candidates[::-1]
                for j in candidates:
                    dc = dist(x, li[j])
                    if d > dc and dc < (max_dist + 0.5):
                        swap_li(li, i+1, j)
                        count_swaps += 1
                        break
        if count_swaps == 0:
            break
        count_swaps_total += count_swaps

    return li, (count_swaps_total, count_swaps)


def loglike_ssr(ssr, nobs):
    '''
    Get likelihood function value from ssr for OLS model.

    Parameters
    ----------
    params : array-like
        The coefficients with which to estimate the loglikelihood.

    Returns
    -------
    The concentrated likelihood function evaluated at params.
    '''
    nobs2 = nobs/2.
    return -nobs2 * (np.log(2*np.pi) + np.log(ssr / nobs) + 1)


class SequentialOLSQR(object):

    def __init__(self, endog, exog, start_idx=0):
        self.endog = endog
        self.exog = exog
        self.start_idx = start_idx
        self.nobs, self.k_vars = exog.shape

        self.xy = np.column_stack((exog, endog))
        self.q, self.r = np.linalg.qr(self.xy)
        self.ssr = self.r[-1, -1]**2
        self.ss_contrib = ss_contrib = self.r[:-1, -1]**2
        self.ess_uncentered = ss_contrib.sum()
        self.uncentered_tss = np.dot(endog.T, endog)
        self.ess_all = ss_contrib.cumsum()
        assert self.uncentered_tss, self.ess_all[-1] + self.ssr
        self.ssr_all = self.uncentered_tss - ss_contrib.cumsum()
        self.dfmodelwc = np.arange(1, self.k_vars+1)

    def r_inv(self):
        return np.linalg.inv(self.r[:-1, :-1])

    def normalized_covparams(self):
        return np.dot(self.r_inv.T, self.r_inv)

    def llf_all(self):
        return loglike_ssr(self.ssr_all, self.nobs)

    def aic_all(self):
        ssrs = self.ssrs
        nobs, k_vars = self.nobs, self.k_vars
        aic = [evm.aic_sigma(ssrs[i], nobs, self.dfmodelwc[i])
               for i in range(k_vars)]
        return np.array(aic)

    def bic_all(self):
        ssrs = self.ssrs
        nobs, k_vars = self.nobs, self.k_vars
        bic = [evm.aic_sigma(ssrs[i], nobs, self.dfmodelwc[i])
               for i in range(k_vars)]
        return np.array(bic)

    def ic_sigma_all(self, ic='bic'):
        ic_func = getattr(evm, ic + '_sigma')
        # ic_func is vectorized with arrays
        ic = ic_func(self.ssr_all, self.nobs, self.dfmodelwc)
        return np.asarray(ic)

    def ic_all(self, ic='bic'):
        ic_func = getattr(evm, ic)
        # ic_func is vectorized with arrays
        ic = ic_func(self.llf_all(), self.nobs, self.dfmodelwc)
        return np.asarray(ic)

    def min_ic_idx(self, ic='bic'):
        '''index of minimum ic model, contains idx + 1 variables/columns

        '''
        ic = self.ic_all(ic=ic)
        argmin_ic = self.start_idx + np.argmin(ic[self.start_idx:])
        return argmin_ic


class StepwiseOLSSweep(object):

    def __init__(self, endog, exog, store_r=False, standardized=False,
                 ddof_std=0, ddof_model=0):
        '''

        standardized requires no constant in exog

        possible extension:
            start from a pinv solution for initially included variables
        '''
        self.endog = endog
        self.exog = exog
        self.ddof_std = ddof_std
        self.ddof_model = ddof_model

        xy = np.column_stack((exog, endog))
        self.k_vars_x = exog.shape[1]
        if endog.ndim == 2:
            self.k_vars_y = endog.shape[1]
        else:
            self.k_vars_y = 1
        self.k_vars_all = self.k_vars_y + self.k_vars_x
        self.nobs = xy.shape[0]
        self.mean_data = xy.mean(0)
        self.std_data = xy.std(ddof_std)
        if standardized:  # zscore
            # xy[:,1:] /= xy[:,1:].std(0)  #if we had a constant in column 1
            xy /= np.sqrt((xy * xy).mean(0))
            xy = (xy - self.mean_data) / self.std_data
            self.ddof_model += 1  # add constant to df

        rs0 = np.dot(xy.T, xy)
        self.rs0 = rs0
        self.is_exog = np.zeros(self.k_vars_all, bool)
        self.history = []
        self.rs_current = rs0.copy()
        self.xy = xy

    def update_history(self):
        # add variables from the current model to the history
        # separate method so we can add things here

        self.history.append((self.is_exog,
                             self.rss,
                             self.params,
                             ))

    @property
    def rss(self):
        '''current residual sum of squares of endog variables

        '''
        ret = self.rs_current[-self.k_vars_y:, -self.k_vars_y:]
        if self.k_vars_y == 1:
            ret = np.squeeze(ret)[()]
        return ret

    ssr = rss  # alias consistent with OLS

    @property
    def params(self):
        '''current parameter estimates of endog on included exog

        view
        '''
        # rows for all endog, cols for included exog
        ret = self.rs_current[-self.k_vars_y:, self.is_exog]
        return ret

    @property
    def df_resid(self):
        k_incl = self.is_exog.sum()

        return self.nobs - k_incl - self.ddof_model

    @property
    def df_modelwc(self):
        return self.nobs - self.df_resid

    @property
    def scale2(self):
        return self.rs_current[-1, -1] / self.df_resid

    @property
    def bse(self):
        '''current parameter estimates of endog on included exog

        view
        '''
        ret = self.normalized_cov_params
        ret *= self.scale2
        ret = np.sqrt(ret)
        return ret

    def sweep(self, k, update=True):
        '''sweep variable k

        Parameters
        ----------
        k : int
            index of the variable to be swept
        update : bool
            If True (default), then the attributes are updated. If False,
            then the instance stays unchanged and the new matrix is returned.

        Returns
        -------
        None or rs_next
            see update parameter

        Notes
        -----
        updates is mostly for checking results, we should get most of the
        results without full updating. See the rss_diff, params_new methods.

        '''
        # sweep function creates a copy
        rs_next = sweep(k, self.rs_current)
        if update:
            self.rs_current = rs_next
            self.is_exog[k] = ~self.is_exog[k]  # flip switch
            self.update_history()
        else:
            return rs_next

    @property
    def tvalues(self):
        """
        Return the t-statistic for a given parameter estimate.
        """
        return self.params / self.bse

    @property
    def pvalues(self):
        """The two-tailed p values for the t-stats of the params."""
        from scipy import stats  # lazy for now
        return stats.t.sf(np.abs(self.tvalues), self.df_resid) * 2

    @property
    def normalized_cov_params(self):
        return self.rs_current[self.is_exog, self.is_exog]

    @property
    def llf(self):
        llf = loglike_ssr(self.ssr, self.nobs)
        return llf

    @property
    def aic(self):
        r"""
        Akaike's information criteria.

        For a model with a constant :math:`-2llf + 2(df\_model + 1)`. For a
        model without a constant :math:`-2llf + 2(df\_model)`.
        """
        return -2 * self.llf + 2 * self.df_modelwc

    @property
    def bic(self):
        r"""
        Bayes' information criteria.

        For a model with a constant :math:`-2llf + \log(n)(df\_model+1)`.
        For a model without a constant :math:`-2llf + \log(n)(df\_model)`.
        """
        return -2 * self.llf + np.log(self.nobs) * self.df_modelwc

    def rss_diff(self, endog_idx=-1):
        '''change in rss when one of a variable is swept

        for now:
            look only at the effect on a single endog
            only case endog_idx=-1
        '''
        rr = self.rs_current
        return (rr[-1, :-1]**2 / np.diag(rr)[:-1])

    def ftest_sweep(self):
        '''get f_test for sweeping a variable

        Warning: currently for adding only
        incomplete vectorization

        '''
        from scipy import stats  # lazy for now

        # need to make the following conditional on is_endog, add or drop
        ssr_diff = self.rss_diff()
        ssr_full = self.rss - ssr_diff
        # this only works for adding variables, not for sweep to drop
        df_full = self.df_resid - 1  # add one variable
        df_diff = 1

        f_value = ssr_diff / df_diff / ssr_full * df_full
        p_value = stats.f.sf(f_value, df_diff, df_full)
        return f_value, p_value

    def params_new(self, endog_idx=-1):
        '''new parameters when one variable is swept

        for now:
            look only at the effect on a single endog
            only case endog_idx=-1
        '''
        rr = self.rs_current
        return (rr[-1, :-1] / np.diag(rr)[:-1, None])

    def params_diff(self, endog_idx=-1):
        '''change in rss when one variable is swept

        for now:
            look only at the effect on a single endog
            only case endog_idx=-1
        '''
        rr = self.rs_current
        return rr[-1, :-1] - self.params_new(endog_idx=endog_idx)

    def get_results(self):
        '''run OLS regression on current model and return results

        '''
        from statsmodels.regression.linear_model import OLS
        res = OLS(self.endog, self.exog[:, self.is_exog[:self.k_vars_x]]).fit()
        return res


# all subset regression


def get_all_subset_sweeps(k_vars):
    """get sweeps for all subsets regression

    sweeps by Darlington
    """
    k = k_vars
    all_sweeps = np.ones(2**k, np.int)
    for i in range(1, k):
        all_sweeps[::2**i] += 1
    all_sweeps = np.roll(all_sweeps, -1) - 1
    return all_sweeps


def get_max_subset_sweeps(k, k_max, force=False):
    if k > 20 and not force:
        raise ValueError("large number of cases 2^k = %d, "
                         "use force=True to continue" % 2**k)

    sweeps_all = get_all_subset_sweeps(k)
    isexog_all = sweeps2isexog(sweeps_all, k)
    isexog_all_arr = np.asarray(isexog_all)
    count_exog = isexog_all_arr.sum(1)
    isexog_le5 = isexog_all_arr[count_exog <= k_max]
    sweeps_filled = isexog2sweeps(isexog_le5)
    return np.asarray(sweeps_filled)


class SelectionResults(object):
    """class to hold and analyse variable selection results
    """
    def __init__(self, res_all, attributes=None, **kwds):
        # Note: we don't want to have access to the instance that created
        # the results because that will change inplace.
        self.isexog = np.array([i[1] for i in res_all])
        self.aic = np.array([res_i[3] for res_i in res_all])
        self.exog_idx = [np.nonzero(ii)[0] for ii in self.isexog]
        self.res_all = res_all
        self.__dict__.update(kwds)

    @cache_readonly
    def idx_sort_aic(self):
        # do we need option for sort criterion
        return np.argsort(self.aic)

    def sorted_frame(self, by='aic'):
        if by != 'aic':
            raise NotImplementedError('`by` is not used yet')
        sort_idx = self.idx_sort_aic
        import pandas as pd
        res_df = pd.DataFrame()
        res_df['exog_idx'] = [self.exog_idx[ii] for ii in sort_idx]
        res_df['aic'] = self.aic[sort_idx]
        return res_df


def all_subset(endog, exog, keep_exog=0, keep_attr=None, k_max=None):
    """all subset regression

    Parameters
    ----------
    endog : ndarray
        endogenous or dependent variable
    exog : ndarray
        exogenous variables, regressors
    keep_exog : int, default is 0
        How many exogeous variables to keep in all subset regression.
        This assumes that the exogenous variables that are of interest are
        in the first columns of `exog`
    keep_attr : None
        not used yet.
        Which additional attributes to keep for each subset model.

    Returns
    -------
    res : instance of SelectionResults
        The instance contains the stored results for each regression and
        provides additional summary and sorting methods.
        Note: Currently, the set of models is not unique if k_max is used
        to limit the number of explanatory variables in the model.

    See Also
    --------
    SelectionResults : class for sequence of subset regression
    """

    if keep_attr is not None:
        raise NotImplementedError('keep_attr is not used yet')

    if k_max is None:
        all_sweeps = (get_all_subset_sweeps(exog.shape[1] - keep_exog) +
                      keep_exog)
    else:
        all_sweeps = (get_max_subset_sweeps(exog.shape[1] - keep_exog, k_max) +
                      keep_exog)

    stols_all = StepwiseOLSSweep(endog, exog)
    # sweep exog that we want to keep
    for ii in range(keep_exog):
        stols_all.sweep(ii)

    res_all = [[[], stols_all.is_exog[:-1].copy(), stols_all.rss,
                stols_all.aic]]
    params_keep_all = []
    bse_keep_all = []
    df_resid_all = []
    df_resid_all.append(stols_all.df_resid)
    if keep_exog > 0:
        # add results for initial "empty" model
        # TODO: check params is 2-dim, bse is 1-dim
        params_keep_all.append(stols_all.params[0, :keep_exog])
        bse_keep_all.append(stols_all.bse[:keep_exog])

    for ii in all_sweeps[:-1]:   # last sweep goes back to empty model
        stols_all.sweep(ii)
        res_all.append([ii, stols_all.is_exog[:-1].copy(), stols_all.rss,
                        stols_all.aic])
        df_resid_all.append(stols_all.df_resid)
        if keep_exog > 0:
            # TODO: check params is 2-dim, bse is 1-dim
            params_keep_all.append(stols_all.params[0, :keep_exog])
            bse_keep_all.append(stols_all.bse[:keep_exog])

    if k_max is not None:
        # filter out models with more than k_max exog
        res_all = [res for res in res_all if res[1].sum() <= k_max]

    res = SelectionResults(res_all,
                           params_keep_all=params_keep_all,
                           bse_keep_all=bse_keep_all,
                           df_resid_all=df_resid_all)

    return res
