"""Helper files for pickling"""
from statsmodels.compat.python import cPickle
from statsmodels.iolib.openfile import get_file_obj


def save_pickle(obj, fname):
    """
    Save the object to file via pickling.

    Parameters
    ----------
    fname : str
        Filename to pickle to
    """
    with get_file_obj(fname, 'wb') as fout:
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
    with get_file_obj(fname, 'rb') as fin:
        return cPickle.load(fin)
