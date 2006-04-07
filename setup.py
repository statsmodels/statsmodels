import os, glob, string, shutil
import sys
sys.path.insert(0,"lib")
from distutils.core import setup
from neuroimaging import packages, __version__

def main():

    setup( name = 'neuroimaging',
           version = __version__,
           description = 'This is a neuroimaging python package',
           author = 'Various, one of whom is Jonathan Taylor',
           author_email = 'jonathan.taylor@stanford.edu',
           ext_package = 'neuroimaging',
           packages=packages,
           package_dir = {'': 'lib'},
           url = 'http://neuroimaging.scipy.org',
           long_description =
           '''
           ''')


if __name__ == "__main__": main()
