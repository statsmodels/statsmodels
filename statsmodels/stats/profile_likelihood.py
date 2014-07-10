# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 19:27:45 2014

Author: Josef Perktold
License: BSD-3

"""

import numpy as np
from scipy import stats
from scipy import optimize


def conf_int_profile(results, index=None, alpha=0.05):
    """profile confidence interval for models with offset

    This function uses offset to fix the parameter. It only applies if the
    model allows for offset and uses a linear combination of explanatory
    variables with parameters, i.e. the model uses `f(exog @ params)` where
    `f` can be an arbitrary function.
    This currently includes GLM, Poisson and NegativeBinomial.

    Parameters
    ----------
    results : results instance
        A results instance where the underlying model satisfies the above
        criteria.
    index : None or iterable of integers
        The index of the parameter for which the profile confidence interval
        is calculated. If None, then the profile confidence intervals are
        calculated for all parameters.
    alpha : float in (0, 1)
        significance level for the likelihood ratio.

    Returns
    -------
    conf_int : ndarray, 1d or 2d
        confidence interval with parameters in rows.
    (res_low, res_upp) : tuple of results instances
        results instances of the parameters at the lower and upper bound of
        the confidence intervals.

    Notes
    -----
    This is not yet optimized for performance (uses default start_params and
    creates new model instances).

    """
    res = results  # alias for shortcut
    endog = res.model.endog
    exog = res.model.exog
    k_params = len(res.params)

    if index is None:
        index = range(k_params)
    else:
        if not hasattr(index, '__iter__'):
            index = [index]

    # loglikelihood threshold
    chi2 = stats.chi2.isf(alpha, 1)
    llf_threshold = res.llf - 0.5 * chi2

    mod_kwds = res.model._get_init_kwds()
    if 'offset' not in mod_kwds:
        raise ValueError('this only works for models that allow for offset')
    offset_model = mod_kwds.pop('offset')
    if offset_model is None:
        offset_model = 0


    def llf_p(p1, idx, return_res=False):
        """function for root of loglike equality
        """
        offset = exog[:, idx] * p1 + offset_model
        keep = list(range(k_params))
        keep.pop(idx)
        #exog = res2.exog[:, 1:]
        modp = res.model.__class__(endog, exog[:, keep], offset=offset,
                                   **mod_kwds)
        # SVD didn't converge exception, when using start_params,
        # try to get constrainte parameter as start_params
        resp = modp.fit() #start_params=res.params[1:])
        if not return_res:
            return resp.llf - llf_threshold
        else:
            return resp

    ci_profile = []
    for idx in index:
        upp = res.params[idx] + 1* np.diff(res.conf_int()[idx])  # twice the confidence
        low = res.params[idx] - 1* np.diff(res.conf_int()[idx])  # twice the confidence
        ci_upp_profile = optimize.brentq(llf_p, res.params[idx], upp, args=(idx,))
        ci_low_profile = optimize.brentq(llf_p, low, res.params[idx], args=(idx,))
        resp_upp = llf_p(ci_upp_profile, idx, return_res=True)
        resp_low = llf_p(ci_low_profile, idx, return_res=True)
        ci_profile.append([ci_low_profile, ci_upp_profile])

    cip = np.array(ci_profile).squeeze()
    return cip, (resp_low, resp_upp)
