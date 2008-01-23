def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('statistics', parent_package, top_path)

    config.add_data_dir('tests')

    from intrinsic_volumes import extension
    name, source, d = extension

    config.add_extension(name, source, **d)
    config.add_data_files(source[0])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
