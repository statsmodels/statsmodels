"""Helper files for pickling"""

import pickle

from statsmodels.iolib.openfile import get_file_obj


def save_pickle(obj, fname):
    """
    Save the object to file via pickling.

    Parameters
    ----------
    fname : {str, pathlib.Path}
        Filename to pickle to
    """
    with get_file_obj(fname, "wb") as fout:
        pickle.dump(obj, fout, protocol=-1)


class _CompatUnpickler(pickle.Unpickler):
    """
    Unpickler that remaps module paths for backward compatibility.

    statsmodels.tools.decorators was renamed to
    statsmodels.tools._decorators. Pickle files created with older
    versions of statsmodels reference the old module path by name,
    so we remap it here during loading.
    """

    _MODULE_REMAP = {
        "statsmodels.tools.decorators": "statsmodels.tools._decorators",
    }

    def find_class(self, module, name):
        module = self._MODULE_REMAP.get(module, module)
        return super().find_class(module, name)


def load_pickle(fname):
    """
    Load a previously saved object

    .. warning::

       Loading pickled models is not secure against erroneous or maliciously
       constructed data. Never unpickle data received from an untrusted or
       unauthenticated source.

    Parameters
    ----------
    fname : {str, pathlib.Path}
        Filename to unpickle

    Notes
    -----
    This method can be used to load *both* models and results.
    """
    with get_file_obj(fname, "rb") as fin:
        return _CompatUnpickler(fin).load()
