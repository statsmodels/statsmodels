import os
try:
    from os.path import relpath
except: # python 2.5

    def relpath(path, start=os.curdir):
        """Return a relative version of a path"""
        if not path:
            raise ValueError("no path specified")
        start_list = os.path.abspath(start).split(os.path.sep)
        path_list = os.path.abspath(path).split(os.path.sep)
        # Work out how much of the filepath is shared by start and path.
        i = len(os.path.commonprefix([start_list, path_list]))
        rel_list = [os.path.pardir] * (len(start_list)-i) + path_list[i:]
        if not rel_list:
            return os.curdir
        return os.path.join(*rel_list)

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('statsmodels', parent_package, top_path)

    # these are subpackages because they have Cython code
    config.add_subpackage('nonparametric')
    config.add_subpackage('tsa')

    #TODO: delegate the non-test stuff to subpackages
    config.add_data_files('sandbox/panel/test_data.txt')
    config.add_data_files('stats/libqsturng/tests/bootleg.dat')
    config.add_data_files('stats/libqsturng/CH.r')
    config.add_data_files('stats/libqsturng/LICENSE.txt')

    curdir = os.path.abspath(os.path.dirname(__file__))

    extradatafiles = [relpath(os.path.join(r,d),start=curdir)
                      for r,ds,f in os.walk(os.path.join(curdir, 'datasets'))
                      for d in f if not os.path.splitext(d)[1] in
                          ['.py', '.pyc']]
    for f in extradatafiles:
        config.add_data_files(f)

    # add all the test and results directories for non *.py files
    for root, dirnames, filenames in os.walk(curdir):
        for dir_name in dirnames:
            if dir_name in ['tests', 'results'] and root != 'sandbox':
                config.add_data_dir(relpath(
                                    os.path.join(root, dir_name),
                                    start = curdir)
                                    )

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
