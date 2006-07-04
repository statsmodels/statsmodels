import os, glob, string, shutil
import sys
sys.path.insert(0,"lib")
from distutils.core import setup, Extension
from neuroimaging import packages, __version__, __doc__
from neuroimaging.defines import enthought_traits_def


def main(packages):

    packages = ['']+list(packages)
    ext_modules = []
    package_dir = {'': 'lib'}

    ENTHOUGHT_TRAITS_DEF, _ = enthought_traits_def()
    if not ENTHOUGHT_TRAITS_DEF:
        ext_modules += [Extension('enthought.traits.ctraits',
                                  [apply(os.path.join, 'lib/enthought/traits/ctraits.c'.split('/'))])]

    setup( name = 'neuroimaging',
           version = __version__,
           description = 'This is a neuroimaging python package',
           author = 'Various',
           author_email = 'nipy-devel@neuroimaging.scipy.org',
           packages=packages,
           ext_modules=ext_modules,
           package_dir=package_dir,
           url = 'http://neuroimaging.scipy.org',
           long_description = __doc__)

    print ENTHOUGHT_TRAITS_DEF, 'enthought'

if __name__ == "__main__": main(packages)
