'''Helper files for pickling'''

def _get_file_obj(fname, mode):
    """
    Light wrapper to handle strings and let files (anything else) pass through
    """
    try:
        fh = open(fname, mode)
    except (IOError, TypeError):
        fh = fname
    return fh

def save_pickle(self, fname):
    """
    Save the object to file via pickling.

    Parameters
    ---------
    fname : str
        Filename to pickle to
    """
    import cPickle as pickle
    fout = _get_file_obj(fname, 'wb')
    pickle.dump(self, fout, protocol=-1)


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
    import cPickle as pickle
    fin = _get_file_obj(fname, 'rb')
    return pickle.load(fin)

def test_pickle():
    import tempfile
    from numpy.testing import assert_equal
    tmpdir = tempfile.mkdtemp(prefix='pickle')
    a = range(10)
    save_pickle(a, tmpdir+'/res.pkl')
    b = load_pickle(tmpdir+'/res.pkl')
    assert_equal(a, b)

    #cleanup tested on Windows
    import os
    os.remove(tmpdir+'/res.pkl')
    os.rmdir(tmpdir)
    assert not os.path.exists(tmpdir)

    import StringIO
    fh = StringIO.StringIO()
    save_pickle(a, fh)
    fh.seek(0,0)
    c = load_pickle(fh)
    fh.close()
    assert_equal(a,b)


if __name__ == '__main__':
    test_pickle()
