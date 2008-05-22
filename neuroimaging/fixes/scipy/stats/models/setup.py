from os.path import join,split


def configuration(parent_package='',top_path=None, package_name='models'):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(package_name,parent_package,top_path)

    config.add_subpackage('*')

    config.add_data_dir('tests')

    config.add_extension('_bspline',
                         sources = ["_bspline.c"],
                         )

    return config

if __name__ == '__main__':

    from numpy.distutils.core import setup

    #package_name = 'neuroimaging.fixes.scipy.stats.models'
    package_name = 'scipy.stats.models'

    setup(**configuration(top_path='',
                          package_name=package_name).todict())
