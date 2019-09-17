"""
Handle file opening for read/write
"""
from numpy.lib._iotools import _is_string_like


class EmptyContextManager(object):
    """
    This class is needed to allow file-like object to be used as
    context manager, but without getting closed.
    """
    def __init__(self, obj):
        self._obj = obj

    def __enter__(self):
        '''When entering, return the embedded object'''
        return self._obj

    def __exit__(self, *args):
        '''Do not hide anything'''
        return False

    def __getattr__(self, name):
        return getattr(self._obj, name)


def _open(fname, mode, encoding):
    if fname.endswith('.gz'):
        import gzip
        return gzip.open(fname, mode, encoding=encoding)
    else:
        return open(fname, mode, encoding=encoding)


def get_file_obj(fname, mode='r', encoding=None):
    """
    Light wrapper to handle strings and let files (anything else) pass through.

    It also handle '.gz' files.

    Parameters
    ----------
    fname : str or file-like object
        File to open / forward
    mode : str
        Argument passed to the 'open' or 'gzip.open' function
    encoding : str
        For Python 3 only, specify the encoding of the file

    Returns
    -------
    A file-like object that is always a context-manager. If the `fname` was
    already a file-like object, the returned context manager *will not
    close the file*.
    """
    if _is_string_like(fname):
        return _open(fname, mode, encoding)
    try:
        # Make sure the object has the write methods
        if 'r' in mode:
            fname.read
        if 'w' in mode or 'a' in mode:
            fname.write
    except AttributeError:
        raise ValueError('fname must be a string or a file-like object')
    return EmptyContextManager(fname)
