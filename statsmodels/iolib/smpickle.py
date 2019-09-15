"""Helper files for pickling"""
from statsmodels.iolib.openfile import get_file_obj


def save_pickle(obj, fname):
    """
    Save the object to file via pickling.

    Parameters
    ----------
    fname : str
        Filename to pickle to
    """
    import pickle

    with get_file_obj(fname, 'wb') as fout:
        pickle.dump(obj, fout, protocol=-1)


def load_pickle(fname):
    """
    Load a previously saved object; **use only on trusted files**,
    as unpickling can run arbitrary code.  (i.e. calling this on a
    malicious file can wipe or take over your system.)

    Parameters
    ----------
    fname : str
        Filename to unpickle

    Notes
    -----
    This method can be used to load *both* models and results.
    """
    import pickle

    with get_file_obj(fname, 'rb') as fin:
        return pickle.load(fin)
