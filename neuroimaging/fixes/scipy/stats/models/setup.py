from os.path import join,split


try:
    # This will work when building as part of nipy, with SciPy already
    # installed
    from scipy import weave
    weave_dir = split(weave.__file__)[0]
except ImportError:
    # As part of the SciPy build, we can't rely on scipy being available yet,
    # so we out the weave directory locally, from our current build path.
    weave_dir = split(__file__)[0]

def configuration(parent_package='',top_path=None, package_name='models'):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(package_name,parent_package,top_path)

    config.add_subpackage('*')

    config.add_data_dir('tests')

    # Set up B-Spline code.  The C++ sources are weave auto-generated code, but
    # we actually ship the included sources so that end-users don't need to run
    # weave.  Only developers who touch the underlying sources should ever need
    # to touch the generation code to update the C++.
    bspline_src = ["_bspline.cpp"]

    # XXX - Hack to fetch the weave paths.  So far I can't find if weave's API
    # supports directly the querying of the weave paths.
    weave_dir = split(weave.__file__)[0]

    config.add_extension('_bspline',
                         sources = bspline_src,
                         include_dirs = [weave_dir,join(weave_dir,'scxx')],
                         )

    ## try:
    ##     from scipy.stats.models.bspline_module import mod
    ##     n, s, d = weave_ext(mod)
    ##     config.add_extension(n, s, **d)
    ## except ImportError: pass

    return config

## def weave_ext(mod):
##     d = mod.setup_extension().__dict__
##     n = d['name']; del(d['name'])
##     s = d['sources']; del(d['sources'])
##     return n, s, d

if __name__ == '__main__':

    from numpy.distutils.core import setup

    #package_name = 'neuroimaging.fixes.scipy.stats.models'
    package_name = 'scipy.stats.models'

    setup(**configuration(top_path='',
                          package_name=package_name).todict())
