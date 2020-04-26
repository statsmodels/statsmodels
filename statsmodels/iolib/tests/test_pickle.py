from io import BytesIO
import pathlib
import tempfile

from numpy.testing import assert_equal

from statsmodels.compat.python import lrange
from statsmodels.iolib.smpickle import save_pickle, load_pickle


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
        import os

        os.remove(path_str)
        os.remove(path_pathlib)
        os.rmdir(tmpdir)
    except (OSError, IOError):
        pass
    assert not os.path.exists(tmpdir)

    # test with file handle
    fh = BytesIO()
    save_pickle(a, fh)
    fh.seek(0, 0)
    d = load_pickle(fh)
    fh.close()
    assert_equal(a, d)
