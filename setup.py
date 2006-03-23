import os, glob, string, shutil
from distutils.core import setup

# Packages

packages = ['neuroimaging',
            'neuroimaging.statistics',
            'neuroimaging.statistics.tests',
            'neuroimaging.image',
            'neuroimaging.image.tests',
            'neuroimaging.reference',
            'neuroimaging.reference.tests',
            'neuroimaging.data',
            'neuroimaging.data.tests',
            'neuroimaging.image.formats',
            'neuroimaging.image.formats.tests',
            'neuroimaging.image.formats.analyze',
            'neuroimaging.fmri',
            'neuroimaging.fmri.tests',
            'neuroimaging.fmri.fmristat',
            'neuroimaging.fmri.fmristat.tests',
            'neuroimaging.visualization',
            'neuroimaging.visualization.cmap']


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



