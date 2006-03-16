import os, glob, string, shutil
from distutils.core import setup

# Packages

packages = ['neuroimaging', 'neuroimaging.statistics', 'neuroimaging.image', 'neuroimaging.reference', 'neuroimaging.data', 'neuroimaging.image.formats', 'neuroimaging.image.formats.analyze', 'neuroimaging.fmri', 'neuroimaging.fmri.fmristat', 'neuroimaging.visualization', 'neuroimaging.visualization.cmap']


def main():

    setup (name = 'neuroimaging',
           version = '0.01a',
           description = 'This is a neuroimaging python package',
           author = 'Various, one of whom is Jonathan Taylor',
           author_email = 'jonathan.taylor@stanford.edu',
           ext_package = 'neuroimaging',
           packages=packages,
           package_dir = {'neuroimaging': 'lib'},
           url = 'http://neuroimaging.scipy.org',
           long_description =
           '''
           ''')



if __name__ == "__main__":
    main()



