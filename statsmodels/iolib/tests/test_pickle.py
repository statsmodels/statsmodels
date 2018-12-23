import tempfile

from numpy.testing import assert_equal

from statsmodels.compat.python import lrange, BytesIO
from statsmodels.iolib.smpickle import save_pickle, load_pickle


def test_pickle():
    tmpdir = tempfile.mkdtemp(prefix='pickle')
    a = lrange(10)
    save_pickle(a, tmpdir+'/res.pkl')
    b = load_pickle(tmpdir+'/res.pkl')
    assert_equal(a, b)

    # cleanup, tested on Windows
    try:
        import os
        os.remove(tmpdir+'/res.pkl')
        os.rmdir(tmpdir)
    except (OSError, IOError):
        pass
    assert not os.path.exists(tmpdir)

    # test with file handle
    fh = BytesIO()
    save_pickle(a, fh)
    fh.seek(0, 0)
    c = load_pickle(fh)
    fh.close()
    assert_equal(a, c)
