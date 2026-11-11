from statsmodels.compat.python import lrange

from io import BytesIO
import os
import pathlib
from pathlib import Path
import tempfile

from numpy.testing import assert_equal

from statsmodels.iolib.smpickle import load_pickle, save_pickle


def test_pickle():
    tmpdir = tempfile.mkdtemp(prefix="pickle")
    a = lrange(10)

    # test with str
    path_str = tmpdir + "/res.pkl"
    save_pickle(a, path_str)
    b = load_pickle(path_str)
    assert_equal(a, b)

    # test with pathlib
    path_pathlib = pathlib.Path(tmpdir) / "res2.pkl"
    save_pickle(a, path_pathlib)
    c = load_pickle(path_pathlib)
    assert_equal(a, c)

    # cleanup, tested on Windows
    try:
        Path(path_str).unlink()
        Path(path_pathlib).unlink()
        Path(tmpdir).rmdir()
    except OSError:
        pass
    assert not Path(tmpdir).exists()

    # test with file handle
    fh = BytesIO()
    save_pickle(a, fh)
    fh.seek(0, 0)
    d = load_pickle(fh)
    fh.close()
    assert_equal(a, d)


def test_pickle_supports_open():
    tmpdir = tempfile.mkdtemp(prefix="pickle")
    a = lrange(10)

    class SubPath:
        def __init__(self, path):
            self._path = pathlib.Path(path)

        def open(
            self,
            mode="r",
            buffering=-1,
            encoding=None,
            errors=None,
            newline=None,
        ):
            return self._path.open(
                mode=mode,
                buffering=buffering,
                encoding=encoding,
                errors=errors,
                newline=newline,
            )

    # test with pathlib
    path_pathlib = SubPath(tmpdir + os.pathsep + "res2.pkl")
    save_pickle(a, path_pathlib)
    c = load_pickle(path_pathlib)
    assert_equal(a, c)
