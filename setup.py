import os, glob, string, shutil
import sys
sys.path.insert(0,"lib")
from distutils.core import setup, Extension
from neuroimaging import packages, __version__, __doc__


def main(packages):

    packages = ['']+list(packages)
    ext_modules = []
    package_dir = {'': 'lib'}

    if 'neuroimaging.extra.enthought' in packages:
        ext_modules += [Extension('extra.enthought.traits.ctraits',
                                  ['lib/neuroimaging/extra/enthought/lib/traits/ctraits.c'])]
        package_dir['neuroimaging.extra.enthought'] = 'lib/neuroimaging/extra/enthought/lib'

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
