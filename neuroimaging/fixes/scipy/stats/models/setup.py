# from os.path import join,split

#def configuration(parent_package='',top_path=None, package_name='models'):
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    #config = Configuration(package_name,parent_package,top_path)
    config = Configuration('models',parent_package,top_path)

    #config.add_subpackage('*')
    config.add_subpackage('family')
    config.add_subpackage('robust')
    config.add_data_dir('tests')
    config.add_extension('_hbspline',
                         sources=['src/bspline_ext.c',
                                  'src/bspline_impl.c'],
    )

    return config

if __name__ == '__main__':

    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

    #package_name = 'neuroimaging.fixes.scipy.stats.models'
    #package_name = 'scipy.stats.models'

    #setup(**configuration(top_path='',
    #                      package_name=package_name).todict())
