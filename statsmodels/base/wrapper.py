import inspect
import functools

import numpy as np
from statsmodels.compat.python import get_function_name, iteritems, getargspec

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

        try:
            results = get('_results')
        except AttributeError:
            pass

        try:
            return get(attr)
        except AttributeError:
            pass

        obj = getattr(results, attr)
        data = results.model.data
        how = self._wrap_attrs.get(attr)
        if how and isinstance(how, tuple):
            obj = data.wrap_output(obj, how[0], *how[1:])
        elif how:
            obj = data.wrap_output(obj, how=how)

        return obj

    def __getstate__(self):
        #print 'pickling wrapper', self.__dict__
        return self.__dict__

    def __setstate__(self, dict_):
        #print 'unpickling wrapper', dict_
        self.__dict__.update(dict_)

    def save(self, fname, remove_data=False):
        '''save a pickle of this instance

        Parameters
        ----------
        fname : string or filehandle
            fname can be a string to a file path or filename, or a filehandle.
        remove_data : bool
            If False (default), then the instance is pickled without changes.
            If True, then all arrays with length nobs are set to None before
            pickling. See the remove_data method.
            In some cases not all arrays will be set to None.

        '''
        from statsmodels.iolib.smpickle import save_pickle

        if remove_data:
            self.remove_data()

        save_pickle(self, fname)

    @classmethod
    def load(cls, fname):
        from statsmodels.iolib.smpickle import load_pickle
        return load_pickle(fname)


def union_dicts(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result


def make_wrapper(func, how):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        results = object.__getattribute__(self, '_results')
        data = results.model.data
        if how and isinstance(how, tuple):
            obj = data.wrap_output(func(results, *args, **kwargs), how[0], how[1:])
        elif how:
            obj = data.wrap_output(func(results, *args, **kwargs), how)
        return obj

    argspec = getargspec(func)
    formatted = inspect.formatargspec(argspec[0], varargs=argspec[1],
                                      defaults=argspec[3])

    func_name = get_function_name(func)

    wrapper.__doc__ = "%s%s\n%s" % (func_name, formatted, wrapper.__doc__)

    return wrapper


def populate_wrapper(klass, wrapping):
    for meth, how in iteritems(klass._wrap_methods):
        if not hasattr(wrapping, meth):
            continue

        func = getattr(wrapping, meth)
        wrapper = make_wrapper(func, how)
        setattr(klass, meth, wrapper)

if __name__ == '__main__':
    import statsmodels.api as sm
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
    offset = np.log(data.exog[:, -1])
    exog = data.exog[:, :-1]

    # convert dur to dummy
    exog = sm.tools.categorical(exog, col=0, drop=True)
    # drop reference category
    # convert res to dummy
    exog = sm.tools.categorical(exog, col=0, drop=True)
    # convert edu to dummy
    exog = sm.tools.categorical(exog, col=0, drop=True)
    # drop reference categories and add intercept
    exog = sm.add_constant(exog[:, [1, 2, 3, 4, 5, 7, 8, 10, 11, 12]], prepend=False)

    endog = np.round(data.endog)
    mod = sm.GLM(endog, exog, family=sm.families.Poisson()).fit()
    # glmwrap = GLMResultsWrapper(mod)
