import inspect
import functools
import types

import numpy as np

class ResultsWrapper(object):
    """
    Class which wraps a statsmodels estimation Results class and steps in to
    reattach metadata to results (if available)
    """
    _wrap_attrs = {}
    _wrap_methods = {}

    def __init__(self, results):
        self._results = results
        self.__doc__ = results.__doc__

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

def union_dicts(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result

def make_wrapper(func, how):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        results = object.__getattribute__(self, '_results')
        data = results.model._data
        return data.wrap_output(func(results, *args, **kwargs), how)

    argspec = inspect.getargspec(func)
    formatted = inspect.formatargspec(argspec.args, varargs=argspec.varargs,
                                      defaults=argspec.defaults)

##    wrapper.__doc__ = "%s%s\n%s" % (func.im_func.func_name, formatted,
##                                    wrapper.__doc__)

    return wrapper

def populate_wrapper(klass, wrapping):
    for meth, how in klass._wrap_methods.iteritems():
        if not hasattr(wrapping, meth):
            continue

        func = getattr(wrapping, meth)
        wrapper = make_wrapper(func, how)
        setattr(klass, meth, wrapper)

if __name__ == '__main__':
    import scikits.statsmodels.api as sm
    from pandas import DataFrame
    data = sm.datasets.longley.load()
    df = DataFrame(data.exog, columns=data.exog_name)
    y = data.endog
    # data.exog = sm.add_constant(data.exog)
    df['intercept'] = 1.
    olsresult = sm.OLS(y, df).fit()
    rlmresult = sm.RLM(y, df).fit()

    # olswrap = RegressionResultsWrapper(olsresult)
    # rlmwrap = RLMResultsWrapper(rlmresult)

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
    # glmwrap = GLMResultsWrapper(mod)
