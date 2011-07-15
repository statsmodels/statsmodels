import functools
import types

from scikits.statsmodels.regression.linear_model import RegressionResults, OLS
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
        'bse' : 'columns',
        'pvalues' : 'columns',
        'tvalues' : 'columns',
        'resid' : 'rows',
        'fittedvalues' : 'rows',
        'wresid' : 'rows',
        'normalized_cov_params' : 'cov',
        'HC0_se' : 'columns',
        'HC1_se' : 'columns',
        'HC2_se' : 'columns',
        'HC3_se' : 'columns'
    }

    _wrap_methods = {
        'norm_resid' : 'rows',
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

if __name__ == '__main__':
    import scikits.statsmodels.api as sm
    data = sm.datasets.longley.load()

    df = DataFrame(data.exog, columns=data.exog_name)
    y = data.endog

    # data.exog = sm.add_constant(data.exog)
    df['intercept'] = 1.
    result = OLS(y, df).fit()
    wrapper = RegressionResultsWrapper(result)
