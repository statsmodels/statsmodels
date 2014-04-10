'''Helper files for pickling'''
from statsmodels.compat.python import cPickle

def _get_file_obj(fname, mode):
    """
    Light wrapper to handle strings and let files (anything else) pass through
    """
    try:
        fh = open(fname, mode)
    except (IOError, TypeError):
        fh = fname
    return fh

def save_pickle(obj, fname):
    """
    Save the object to file via pickling.

    Parameters
    ----------
    fname : str
        Filename to pickle to
    """
    fout = _get_file_obj(fname, 'wb')
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
    fin = _get_file_obj(fname, 'rb')
    return cPickle.load(fin)

