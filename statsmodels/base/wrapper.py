import inspect
import functools

from statsmodels.compat.python import iteritems, getargspec


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

    try:  # Python 3.3+
        sig = inspect.signature(func)
        formatted = str(sig)
    except AttributeError:
        # TODO: Remove when Python 2.7 is dropped
        argspec = getargspec(func)
        formatted = inspect.formatargspec(argspec[0],
                                          varargs=argspec[1],
                                          defaults=argspec[3])

    wrapper.__doc__ = "%s%s\n%s" % (func.__name__, formatted, wrapper.__doc__)

    return wrapper


def populate_wrapper(klass, wrapping):
    for meth, how in iteritems(klass._wrap_methods):
        if not hasattr(wrapping, meth):
            continue

        func = getattr(wrapping, meth)
        wrapper = make_wrapper(func, how)
        setattr(klass, meth, wrapper)
