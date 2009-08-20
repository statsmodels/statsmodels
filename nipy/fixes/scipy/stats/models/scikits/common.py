descr = """
Statmodels (?) is a python package that provides an interface to SciPy for
statistical computations including descriptive statistics and
fitting statistical models.

Brief history of the major codebase...

LICENSE: TBD
"""

DISTNAME = 'scikits.statmodels'
DESCRIPTION = 'Statistical computations and models for use with SciPy'
LONG_DESCRIPTION = desc
MAINTAINER = ''
MAINAINER_EMAIL =''
URL = ''
LICENSE = ''
DOWNLOAD_URL = ''

MAJ = 0
MIN = 1
REV = 1
DEV = True

def build_ver_str():
    return '%d.%d.%d' % (MAJ,MIN,REV)

def fbuild_fver_str():
    if DEV:
        return build_ver_str() +'dev'
    else:
        return build_ver_str()

VERSION = build_ver_str()

def write_version():
    f = open(fname, "w")
    f.writelines("version = '%s'\n" % build_ver_str())
    f.writelines("dev = %s\n" % DEV)
    f.writelines("full_version = '%s'\n" % build_fver_str())
    f.close()

