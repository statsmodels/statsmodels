from .py3k import reduce, lzip, lmap, range

def iteritems(obj, **kwargs):
    """replacement for six's iteritems for Python2/3 compat
       uses 'iteritems' if available and otherwise uses 'items'.

       Passes kwargs to method.
    """
    func = getattr(obj, "iteritems", None)
    if not func:
        func = obj.items
    return func(**kwargs)


def iterkeys(obj, **kwargs):
    func = getattr(obj, "iterkeys", None)
    if not func:
        func = obj.keys
    return func(**kwargs)


def itervalues(obj, **kwargs):
    func = getattr(obj, "itervalues", None)
    if not func:
        func = obj.values
    return func(**kwargs)


def get_function_name(func):
    try:
        return func.im_func.func_name
    except AttributeError:
        #Python 3
        return func.__name__

def get_class(func):
    try:
        return func.im_class
    except AttributeError:
        #Python 3
        return func.__self__.__class__
