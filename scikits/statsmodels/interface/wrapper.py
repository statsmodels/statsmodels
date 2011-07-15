import functools
import types

import numpy as np

from scikits.statsmodels.regression.linear_model import RegressionResults, OLS
from scikits.statsmodels.robust.robust_linear_model import RLMResults
from scikits.statsmodels.genmod.generalized_linear_model import GLMResults
from pandas import DataFrame

class ResultsWrapper(object):
    _wrap_attrs = {}
    _wrap_methods = {}

    def __init__(self, results):
        self._results = results

    def __dir__(self):
        return [x for x in dir(self._results)]

    def __getattribute__(self, attr):
        get = lambda name: object.__getattribute__(self, name)
        results = get('_results')

        if attr == '__class__':
            return type(results)

        try:
            return get(attr)
        except AttributeError:
            pass

        obj = getattr(results, attr)
        data = results.model._data
        how = self._wrap_attrs.get(attr)
        if how:
            obj = data.wrap_output(obj, how=how)

        return obj

class RegressionResultsWrapper(ResultsWrapper):

    _wrap_attrs = {
        'params' : 'columns',
        'chisq' : 'columns',
        'bse' : 'columns',
        'pvalues' : 'columns',
        'tvalues' : 'columns',
        'resid' : 'rows',
        'sresid' : 'rows',
        'weights' : 'rows',
        'fittedvalues' : 'rows',
        'wresid' : 'rows',
        'normalized_cov_params' : 'cov',
        'bcov_unscaled' : 'cov',
        'bcov_scaled' : 'cov',
        'HC0_se' : 'columns',
        'HC1_se' : 'columns',
        'HC2_se' : 'columns',
        'HC3_se' : 'columns'
    }

    _wrap_methods = {
        'norm_resid' : 'rows',
        'cov_params' : 'cov'
    }

class RLMResultsWrapper(RegressionResultsWrapper):
    _wrap_methods = {
        'cov_params' : 'cov'
    }

class GLMResultsWrapper(RegressionResultsWrapper):
    _wrap_attrs = RegressionResultsWrapper._wrap_attrs.copy()

    _wrap_attrs.update({
            'resid_anscombe' : 'rows',
            'resid_deviance' : 'rows',
            'resid_pearson' : 'rows',
            'resid_response' : 'rows',
            'resid_working' : 'rows'
     })

    _wrap_methods = {
        'cov_params' : 'cov'
    }

def make_wrapper(func, how):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        results = object.__getattribute__(self, '_results')
        data = results.model._data
        return data.wrap_output(func(results, *args, **kwargs), how)

    return wrapper

def populate_wrapper(klass, wrapping):
    for meth, how in klass._wrap_methods.iteritems():
        func = getattr(wrapping, meth)
        wrapper = make_wrapper(func, how)
        setattr(klass, meth, wrapper)

populate_wrapper(RegressionResultsWrapper, RegressionResults)
populate_wrapper(RLMResultsWrapper, RLMResults)
populate_wrapper(GLMResultsWrapper, GLMResults)

if __name__ == '__main__':
    import scikits.statsmodels.api as sm
    data = sm.datasets.longley.load()

    df = DataFrame(data.exog, columns=data.exog_name)
    y = data.endog

    # data.exog = sm.add_constant(data.exog)
    df['intercept'] = 1.
    olsresult = sm.OLS(y, df).fit()
    rlmresult = sm.RLM(y, df).fit()
    olswrap = RegressionResultsWrapper(olsresult)
    rlmwrap = RLMResultsWrapper(rlmresult)

    data = sm.datasets.wfs.load()
    # get offset
    offset = np.log(data.exog[:,-1])
    exog = data.exog[:,:-1]

    # convert dur to dummy
    exog = sm.tools.categorical(exog, col=0, drop=True)
    # drop reference category
    # convert res to dummy
    exog = sm.tools.categorical(exog, col=0, drop=True)
    # convert edu to dummy
    exog = sm.tools.categorical(exog, col=0, drop=True)
    # drop reference categories and add intercept
    exog = sm.add_constant(exog[:,[1,2,3,4,5,7,8,10,11,12]])

    endog = np.round(data.endog)
    mod = sm.GLM(endog, exog, family=sm.families.Poisson()).fit()

    glmwrap = GLMResultsWrapper(mod)
