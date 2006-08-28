import os, glob, string, shutil
import sys
sys.path.insert(0,"lib")
from distutils.core import setup, Extension
from neuroimaging import packages, __version__, __doc__, ENTHOUGHT_TRAITS_DEF
from numpy.distutils.misc_util import get_numpy_include_dirs


def main(packages):

    packages = ['']+list(packages)
    ext_modules = [Extension('image.formats.minc._mincutils',
                                  [apply(os.path.join, 'lib/neuroimaging/core/image/formats/minc/_mincutils.c'.split('/'))],
                             extra_link_args=["-lminc"],
                            include_dirs=get_numpy_include_dirs())]

    package_dir = {'': 'lib'}

    if not ENTHOUGHT_TRAITS_DEF:
        ext_modules += [Extension('utils.enthought.traits.ctraits',
                                  [apply(os.path.join, 'lib/neuroimaging/utils/enthought/traits/ctraits.c'.split('/'))])]
        package_dir['neuroimaging.utils.enthought'] = apply(os.path.join, 'lib/neuroimaging/utils/enthought/'.split('/'))

    setup( name = 'neuroimaging',
           version = __version__,
           description = 'This is a neuroimaging python package',
           author = 'Various',
           author_email = 'nipy-devel@neuroimaging.scipy.org',
           ext_package = 'neuroimaging',
           packages=packages,
           ext_modules=ext_modules,
           package_dir = package_dir,
           url = 'http://neuroimaging.scipy.org',
           long_description = __doc__)


if __name__ == "__main__": main(packages)
