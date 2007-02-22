from os.path import join
import sys
sys.path.insert(0,"lib")

from numpy.distutils.misc_util import get_numpy_include_dirs

from neuroimaging import packages, __version__, __doc__, ENTHOUGHT_TRAITS_DEF


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True)
#                       quiet=True)

    config.add_subpackage('neuroimaging', 'lib/neuroimaging')
    return config


def main(packages):
    from numpy.distutils.core import setup

    #packages = ['']+list(packages)
    #package_dir = {'': 'lib'}

    setup( name = 'neuroimaging',
           version = __version__,
           description = 'This is a neuroimaging python package',
           author = 'Various',
           author_email = 'nipy-devel@neuroimaging.scipy.org',
           #ext_package = 'neuroimaging',
           #packages=packages,
           #package_dir = package_dir,
           url = 'http://neuroimaging.scipy.org',
           long_description = __doc__,
           configuration = configuration)


if __name__ == "__main__":
    main(packages)
