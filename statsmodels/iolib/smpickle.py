'''Helper files for pickling'''
from statsmodels.compat.python import cPickle

class EmptyContextManager(object):
    """
    This class is needed to allow file-like object to be used as
    context manager, but without getting closed.
    """
    def __init__(self, obj):
        self._obj = obj

    def __enter__(self):
        return self._obj

    def __exit__(self, *args):
        return False

def _get_file_obj(fname, mode):
    """
    Light wrapper to handle strings and let files (anything else) pass through
    """
    try:
        fh = open(fname, mode)
    except (IOError, TypeError):
        fh = EmptyContextManager(fname)
    return fh

def save_pickle(obj, fname):
    """
    Save the object to file via pickling.

    Parameters
    ----------
    fname : str
        Filename to pickle to
    """
    with _get_file_obj(fname, 'wb') as fout:
        cPickle.dump(obj, fout, protocol=-1)


def load_pickle(fname):
    """
    Load a previously saved object from file

    Parameters
    ----------
    fname : str
        Filename to unpickle

    Notes
    -----
    This method can be used to load *both* models and results.
    """
    with _get_file_obj(fname, 'rb') as fin:
        return cPickle.load(fin)

